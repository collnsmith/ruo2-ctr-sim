"""Global fits across datasets (potentials, film thicknesses).

What the potential does to each rod: one dataset per condition (a film at one potential, or at
OCV), each with its potential in the metadata (potential_V). Each parameter is then

    shared        one value for every dataset (substrate, B factors, ...)
    per film      one value per film, shared by all potentials of that film (thickness, roughness,
                  strain: a potential does not change the film); datasets of one film are those with
                  the same 'film' metadata (else the same thickness_nm)
    per dataset   one scoped copy "<scope>.<name>" per dataset: what the potential changes (the CUS
                  coverage and species theta, x_OH, x_O, their heights z_OH, ..., near-surface O)

Each dataset also has its own scale. series() gives a per-dataset parameter against potential.

    datasets = [d_04V, d_08V, d_10V]                  # each with meta potential_V
    fit = global_fit(model, datasets, per_dataset=["x_OH", "z_OH"], per_film=["thickness_mean_tl"])
    fit.refine()
    res = fit.report()
    pot, x, dx = series(res, "x_OH")                  # x_OH vs potential with errors

Links work across datasets: model.params.link("ds2.z_OH", "ds1.z_OH") ties two datasets together.
"""
import re

import numpy as np

from .fit import Fit


def scope_name(ds, i):
    """A scope usable in link expressions: the dataset's own scope, else ds1, ds2, ..."""
    s = ds.scope or f"ds{i + 1}"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s):
        raise ValueError(f"dataset scope {s!r} must be a plain identifier (letters, digits, _)")
    return s


def make_per_dataset(model, datasets, names):
    """Give every dataset its own copy of the parameters in `names` (same value, bounds, fit flag).
    The global parameter stays as the default but is no longer fitted. A copy that a per-film fit
    tied to another dataset's copy is set free again."""
    for i, ds in enumerate(datasets):
        ds.scope = scope_name(ds, i)
    scopes = {ds.scope for ds in datasets}
    new = set()
    for n in names:
        g = model.params[n]
        if g.linked:
            raise ValueError(f"{n} is linked ({g.expr}); unlink it before making it per dataset")
        for ds in datasets:
            sn = f"{ds.scope}.{n}"
            if sn not in model.params:
                model.params.add(name=sn, value=g.value, min=g.min, max=g.max, fit=g.fit, unit=g.unit,
                                 group=g.group, description=f"{g.description} ({ds.name})")
                new.add(sn)
            else:
                p = model.params[sn]
                if p.linked and re.fullmatch(rf"(\w+)\.{re.escape(n)}", p.expr.strip()) and \
                        p.expr.strip().split(".")[0] in scopes:
                    model.params.unlink(sn)
                    p.fit = g.fit or p.fit
        g.fit = False
    seed_thickness(model, datasets, new)        # a new copy of the thickness starts at the dataset's
    return model


def shadowed(params, datasets):
    """{global parameter: [its copies]} for the global parameters that no dataset uses, because every
    dataset has its own scoped copy (e.g. thickness_mean_tl once each film has ds1.thickness_mean_tl,
    ...). Changing or fitting such a global parameter has no effect on the fit."""
    datasets = list(datasets)
    scopes = [ds.scope for ds in datasets]
    if not datasets or not all(scopes):
        return {}
    have = [{p.base for p in params if p.scope == s} for s in scopes]
    common = set.intersection(*have)
    return {n: [f"{s}.{n}" for s in scopes] for n in sorted(common) if n in params}


def film_of(ds):
    """The film of a dataset: its 'film' and thickness_nm metadata (scans of one film name but another
    thickness are another film), else None (one film)."""
    f, t = ds.meta.get("film"), ds.meta.get("thickness_nm")
    if not f and t is None:
        return None
    return f"{f or ''}|{'' if t is None else f'{t:g}'}"


def seed_thickness(model, datasets, names):
    """Start the dataset copies in `names` of thickness_mean_tl at the dataset's thickness_nm (the
    film's nominal thickness, per scan in the scan library), converted to trilayers with the film's
    spacing (ruo2_a of the settings and the current eps_perp). Bounds that the copy took from the
    global parameter (set for another film, perhaps) are moved, same width, to centre on it when the
    nominal value lies outside them. Returns {copy: value} set."""
    from ..core.lattice import film_trilayers
    out = {}
    for ds in datasets:
        sn = f"{ds.scope}.thickness_mean_tl"
        nm = ds.meta.get("thickness_nm")
        if sn not in names or nm is None or sn not in model.params or model.params[sn].linked:
            continue
        eps = model.params.view(ds.scope)["eps_perp"] if "eps_perp" in model.params else 0.0
        p = model.params[sn]
        v = film_trilayers(nm, model.settings["ruo2_a"], eps)
        if not p.min <= v <= p.max:
            half = 0.5 * (p.max - p.min)
            p.min, p.max = max(1.0, v - half), v + half
        p.value = float(v)
        out[sn] = p.value
    return out


def films(datasets):
    """{film: [datasets]} in order of first appearance."""
    out = {}
    for ds in datasets:
        out.setdefault(film_of(ds), []).append(ds)
    return out


def make_per_film(model, datasets, names):
    """One value per film of the parameters in `names`: every dataset gets its copy, and the copies of
    a film's later datasets follow its first one (links), so all potentials of a film share it."""
    make_per_dataset(model, datasets, names)
    for group in films(datasets).values():
        lead = group[0]
        for n in names:
            model.params[f"{lead.scope}.{n}"].fit = model.params[f"{lead.scope}.{n}"].fit or model.params[n].fit
            for ds in group[1:]:
                model.params.link(f"{ds.scope}.{n}", f"{lead.scope}.{n}")
    return model


def global_fit(model, datasets, per_dataset=(), per_film=(), **fit_kw):
    """Fit object for several datasets: `per_dataset` parameters get one copy per dataset,
    `per_film` parameters one value per film (shared by the film's potentials), all other free
    parameters are shared. Each dataset also gets its own scale."""
    datasets = list(datasets)
    if len(datasets) < 2:
        raise ValueError("a global fit needs at least two datasets")
    both = set(per_dataset) & set(per_film)
    if both:
        raise ValueError(f"{', '.join(sorted(both))}: per dataset or per film, not both")
    make_per_dataset(model, datasets, per_dataset)
    if per_film and len(films(datasets)) > 1:       # one film: per film is the same as shared
        make_per_film(model, datasets, list(per_film))
    return Fit(model, datasets, **fit_kw)


def series(result, name, key="potential_V"):
    """(x, values, errors) of a per-dataset parameter across the datasets of a result, sorted by
    the metadata `key` (potential by default; datasets without it are numbered 0, 1, ...)."""
    rows = []
    for i, meta in enumerate(result.dataset_meta):
        full = f"{meta['scope']}.{name}" if meta.get("scope") else name
        if full not in result.values:
            full = name
        x = meta.get(key, i)
        rows.append((float(x), result.values[full], result.errors.get(full, np.nan)))
    rows.sort()
    x, v, e = (np.array(c, float) for c in zip(*rows))
    return x, v, e

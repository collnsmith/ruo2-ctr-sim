"""Global fits across datasets (potentials, film thicknesses).

Each parameter is either shared (one global value, e.g. the film structure and B factors) or per
dataset (one scoped copy "<scope>.<name>" per dataset, e.g. the CUS species and their heights).

    datasets = [d_04V, d_08V, d_10V]                  # each with meta potential_V
    fit = global_fit(model, datasets, per_dataset=["x_OH", "z_OH"])
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
    The global parameter stays as the default but is no longer fitted."""
    for i, ds in enumerate(datasets):
        ds.scope = scope_name(ds, i)
    for n in names:
        g = model.params[n]
        if g.linked:
            raise ValueError(f"{n} is linked ({g.expr}); unlink it before making it per dataset")
        for ds in datasets:
            sn = f"{ds.scope}.{n}"
            if sn not in model.params:
                model.params.add(name=sn, value=g.value, min=g.min, max=g.max, fit=g.fit, unit=g.unit,
                                 group=g.group, description=f"{g.description} ({ds.name})")
        g.fit = False
    return model


def global_fit(model, datasets, per_dataset=(), **fit_kw):
    """Fit object for several datasets: `per_dataset` parameters get one copy per dataset, all
    other free parameters are shared. Each dataset also gets its own scale."""
    datasets = list(datasets)
    if len(datasets) < 2:
        raise ValueError("a global fit needs at least two datasets")
    make_per_dataset(model, datasets, per_dataset)
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

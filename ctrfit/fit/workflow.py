"""The guided fitting workflow of the scan plan, as plain functions (the GUI calls these).

Step 1  film on the even rods (H + K even): thickness, roughness, eps_perp and the scale, with a
        global search, a thickness profile (fringe minima about one trilayer apart) around the value
        the search found (not the starting value) and refinement.
Step 2  surface on the odd rods (H + K odd): CUS coverage, OH share, OH height and the scale.
Step 3  everything free at once (all rods, all datasets), refined from steps 1 and 2.

Each step frees its parameters (plus whatever the caller lists in `extra`), runs, and leaves the
model at the result. The free set before the step is restored afterwards, except in step 3.
Steps 1 and 2 solve the scales exactly in the global search and refine from its `starts` best
distinct points, and every step also refines from the film thickness `hops` trilayers up and down
(the neighbouring fringe minima), keeping the lowest chi2 (Fit.refine). So they need no close
starting guess, only bounds that bracket the answer.
"""
import numpy as np

from ..core.ctrmodel import substrate_bragg_L
from .fit import Fit

FILM_PARAMS = ["thickness_mean_tl", "roughness_tl", "eps_perp", "scale"]
SURFACE_PARAMS = ["theta", "x_OH", "z_OH", "scale"]


def rod_parity_subset(ds, parity):
    """Rods with (H + K) % 2 == parity (0: even, 1: odd); None if there are none."""
    labels = [lab for lab in ds.rod_labels if sum(ds.rod_hk(lab)) % 2 == parity]
    return ds.select_rods(labels, name=f"{ds.name} {'odd' if parity else 'even'}") if labels else None


def _subsets(datasets, parity):
    out = []
    for ds in datasets:
        sub = rod_parity_subset(ds, parity)
        if sub is not None:
            sub.scope = ds.scope
            out.append(sub)
    if not out:
        raise ValueError(f"no {'odd' if parity else 'even'} (H + K) rods in the data")
    return out


def _free(model, names, datasets):
    """Parameter names to free: scoped per-dataset copies ("ds2.x_OH") where they exist for these
    datasets, else the global parameter."""
    scopes = [ds.scope for ds in datasets if ds.scope] if len(datasets) > 1 else []
    out = []
    for n in dict.fromkeys(names):
        scoped = [f"{s}.{n}" for s in scopes if f"{s}.{n}" in model.params]
        out += scoped if scoped else ([n] if n in model.params else [])
    return out


def _run(model, datasets, names, de, profile=None, progress=None, fit_hook=None, starts=3, hops=1, cores=1):
    before = model.params.free_names
    model.params.fit_only(_free(model, names, datasets))
    fit = Fit(model, datasets, cores=cores)
    fit.progress = progress
    if fit_hook:
        fit_hook(fit)
    try:
        if de:
            fit.run("de", **de)
        if profile and profile[0] in fit.names and not fit.cancelled:
            grid = profile[1] if profile[1] is not None else thickness_window(model, profile[0])
            fit.profile(profile[0], grid)
        if not fit.cancelled:
            fit.refine(starts=starts if de else 1, hops=hops)
        return fit, fit.report()
    finally:
        model.params.fit_only([n for n in before if n in model.params])


def step_film(model, datasets, de=None, thickness_grid=None, extra=(), **kw):
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    de = dict(maxiter=30, popsize=10, seed=1) if de is None else de
    _prepare_scales(model, datasets)
    return _run(model, _subsets(datasets, 0), FILM_PARAMS + list(extra), de,
                ("thickness_mean_tl", thickness_grid), **kw)


def thickness_window(model, name="thickness_mean_tl", half=3.0, step=0.5):
    """Profile grid +-half (TL) around the parameter's current value (after the global search), within
    its bounds."""
    p = model.params[name]
    lo, hi = max(p.min, p.value - half), min(p.max, p.value + half)
    return np.arange(lo, hi + 1e-9, step)


def step_surface(model, datasets, de=None, extra=(), **kw):
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    de = dict(maxiter=25, popsize=10, seed=2) if de is None else de
    _prepare_scales(model, datasets)
    return _run(model, _subsets(datasets, 1), SURFACE_PARAMS + list(extra), de, **kw)


def step_all(model, datasets, names=None, progress=None, fit_hook=None, hops=1, cores=1, **_):
    """Refine everything listed (default: film + surface parameters) on all rods of all datasets, also
    from the film thickness `hops` trilayers up and down (Fit.refine)."""
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    _prepare_scales(model, datasets)
    model.params.fit_only(_free(model, names or FILM_PARAMS + SURFACE_PARAMS, datasets))
    fit = Fit(model, datasets, cores=cores)
    fit.progress = progress
    if fit_hook:
        fit_hook(fit)
    fit.refine(hops=hops)
    return fit, fit.report()


def prepare(model, datasets, per_dataset=(), per_film=()):
    """Before a run with several datasets: scopes and scales, per-dataset copies of `per_dataset`
    and one value per film of `per_film` (call this on the thread that owns the parameter table)."""
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    _prepare_scales(model, datasets)
    if len(datasets) > 1 and per_dataset:
        from .global_fit import make_per_dataset
        make_per_dataset(model, datasets, [n for n in per_dataset if n in model.params])
    if len(datasets) > 1 and per_film:
        from .global_fit import films, make_per_film
        names = [n for n in per_film if n in model.params and n not in set(per_dataset)]
        if names and len(films(datasets)) > 1:      # one film: per film is the same as shared
            make_per_film(model, datasets, names)


def _prepare_scales(model, datasets):
    """Several datasets: give each a scope and its own scale before the subsets are made. The copies are
    fitted when the shared scale was (which then stands for them and is no longer fitted itself)."""
    if len(datasets) > 1:
        g = model.params["scale"]
        for i, ds in enumerate(datasets):
            ds.scope = ds.scope or f"ds{i + 1}"
            n = f"{ds.scope}.scale"
            if n not in model.params:
                model.params.add(name=n, value=g.value, min=g.min, max=g.max, fit=g.fit, unit=g.unit, group="scale",
                                 description=f"Scale of dataset {ds.name}")
            elif g.fit and not model.params[n].linked:
                model.params[n].fit = True
        g.fit = False


# ---------------------------------------------------------------------- steps that only select
# The fitting window's guided buttons: each selects what is fitted (fit flags), on which rods and how,
# and leaves the run to the Run fit button (so the selection can be adjusted first).
GUIDED = [
    dict(key="scales", title="0. Scales only", rods="all", method="refine only", scan=False,
         help="Only the dataset scales, on all rods: the right intensity level before anything else."),
    dict(key="film", title="1. Film: thickness, strain", rods="even", method="DE + refine", scan=True,
         help="Film thickness, out-of-plane strain and the scales on the even rods (H + K even: mostly the film), "
              "with a global search and a thickness scan around its result."),
    dict(key="film_shape", title="2. Film: + roughness, spread", rods="even", method="refine only", scan=False,
         help="Step 1 plus the roughness and the thickness spread, refined on the even rods."),
    dict(key="surface", title="3. Surface: coverage, species", rods="odd", method="DE + refine", scan=False,
         help="CUS coverage and species on the odd rods (H + K odd: mostly O): theta, the scales, and the species "
              "the composition holds (x_OH and z_OH when both OH and H2O are there; z_H2O alone at x_OH = 0, "
              "z_OH alone at x_OH = 1)."),
    dict(key="surface_more", title="4. Surface: + bridging O", rods="odd", method="refine only", scan=False,
         help="Step 3 plus the bridging O height (dz_Obr: its protonation), refined on the odd rods."),
    dict(key="cell", title="5. Cell: electrolyte film", rods="all", method="refine only", scan=False,
         help="The electrolyte film thickness t_liquid and the scales on all rods (set t_window to your window "
              "first; it lowers low L most)."),
    dict(key="all", title="6. Everything together", rods="all", method="refine only", scan=False,
         help="Everything of steps 2 and 4 (and t_liquid if it is in use), refined on all rods."),
]
GUIDED_BY_KEY = {g["key"]: g for g in GUIDED}


def _values(P, base, datasets):
    """The values of a parameter as the datasets see it (their copies, else the shared one)."""
    out = [P[f"{ds.scope}.{base}"].value for ds in datasets if ds.scope and f"{ds.scope}.{base}" in P]
    return out or ([P[base].value] if base in P else [])


def _species(P, datasets):
    x = _values(P, "x_OH", datasets)
    if x and all(v <= 0 for v in x):
        return ["z_H2O"]
    if x and all(v >= 1 for v in x):
        return ["z_OH"]
    return ["x_OH", "z_OH"]


def step_names(model, key, datasets=()):
    """The parameters (base names) a guided step fits."""
    P = model.params
    film = ["thickness_mean_tl", "eps_perp", "scale"]
    shape = film + ["roughness_tl", "thickness_spread_tl"]
    surface = ["theta", "scale"] + _species(P, datasets)
    more = surface + ["dz_Obr"]
    in_use = any(v > 0 for v in _values(P, "t_liquid", datasets))
    names = {"scales": ["scale"], "film": film, "film_shape": shape, "surface": surface, "surface_more": more,
             "cell": ["t_liquid", "scale"], "all": shape + more + (["t_liquid"] if in_use else [])}[key]
    return [n for n in dict.fromkeys(names) if n in P]


def active_names(model, datasets, names, also=()):
    """Of these names, those a run gives copies: fitted, about to be fitted (also), or with copies already."""
    P = model.params
    scopes = {ds.scope for ds in datasets if ds.scope}
    return [n for n in names if n in P and (P[n].fit or n in also or any(f"{s}.{n}" in P for s in scopes))]


def select_step(model, datasets, key, per_dataset=(), per_film=()):
    """Set the fit flags of a guided step (making the per-dataset / per-film copies it needs first); returns
    the step with 'names', the parameters now free."""
    step = dict(GUIDED_BY_KEY[key])
    datasets = list(datasets)
    names = step_names(model, key, datasets)
    if len(datasets) > 1:
        prepare(model, datasets, active_names(model, datasets, per_dataset, names),
                active_names(model, datasets, [n for n in per_film if n not in set(per_dataset)], names))
    model.params.fit_only(_free(model, names, datasets))
    step["names"] = model.params.free_names
    return step


def rod_subsets(datasets, rods):
    """The datasets for a run on 'all', 'even' (H + K even) or 'odd' rods (datasets without such rods left
    out; each subset keeps its dataset's scope)."""
    datasets = list(datasets)
    if rods == "all":
        return datasets
    return _subsets(datasets, {"even": 0, "odd": 1}[rods])


def near_bragg(ds, half_width, model=None, values=None):
    """True for the points of dataset `ds` within ±half_width (r.l.u., edge included) of a TiO2 Bragg peak
    of their rod (core.ctrmodel.substrate_bragg_L: only where the substrate scatters, e.g. L = 3 and 5 on
    (0 1), not L = 1). A rod's L offset in this dataset (from `values`, the dataset's parameter values, else
    the model's) moves its peaks: the model is taken at L + dL, so the data's peak sits at L_Bragg - dL."""
    near = np.zeros(len(ds), bool)
    if not half_width or half_width <= 0 or not len(ds):
        return near
    if values is None:
        values = model.params.view(ds.scope) if model is not None else {}
    for lab, idx in ds.rods().items():
        H, K = ds.rod_hk(lab)
        L = ds.L[idx] + float(values.get(f"dL_{H}_{K}", 0.0))
        peaks = substrate_bragg_L(H, K, L.min() - half_width, L.max() + half_width)
        if peaks.size:
            near[idx] = np.abs(L[:, None] - peaks[None, :]).min(axis=1) <= half_width + 1e-9   # edge included
    return near


def skip_bragg(datasets, half_width, model=None):
    """(the datasets without their points near TiO2 Bragg peaks, notes). Each subset keeps its dataset's
    scope; a dataset with nothing left sits the run out (noted). 0: everything kept."""
    out, notes = [], []
    for ds in datasets:
        near = near_bragg(ds, half_width, model)
        if not near.any():
            out.append(ds)
        elif near.all():
            notes.append(f"{ds.name}: every point is within ±{half_width:g} of a Bragg peak; it sits this run out")
        else:
            out.append(ds.subset(~near))
            notes.append(f"{ds.name}: {int(near.sum())} of {len(ds)} points within ±{half_width:g} of a TiO2 Bragg "
                         "peak skipped")
    if datasets and not out:
        raise ValueError(f"every point is within ±{half_width:g} of a Bragg peak")
    return out, notes


def unused_free(model, datasets):
    """Free parameters that cannot act on these datasets: the copies of datasets not among them (unless a
    parameter they use follows them by a link). A run holds them fixed."""
    P = model.params
    scopes = {ds.scope for ds in datasets}
    used = {p.name for p in P if p.scope in scopes or not p.scope}
    todo = [p for p in P if p.name in used and p.linked]
    seen = set()
    while todo:                                   # what the used ones follow, and what that follows
        q = todo.pop()
        if q.name in seen:
            continue
        seen.add(q.name)
        for ref in q._expr.names:
            try:
                r = P.resolve(ref, q.scope)
            except Exception:
                continue
            used.add(r)
            if P[r].linked:
                todo.append(P[r])
    return [n for n in P.free_names if n not in used]

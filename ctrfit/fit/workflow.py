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

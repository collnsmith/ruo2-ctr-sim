"""The guided fitting workflow of the scan plan, as plain functions (the GUI calls these).

Step 1  film on the even rods (H + K even): thickness, roughness, eps_perp and the scale, with a
        global search, a thickness profile (fringe minima about one trilayer apart) and refinement.
Step 2  surface on the odd rods (H + K odd): CUS coverage, OH share, OH height and the scale.
Step 3  everything free at once (all rods, all datasets), refined from steps 1 and 2.

Each step frees its parameters (plus whatever the caller lists in `extra`), runs, and leaves the
model at the result. The free set before the step is restored afterwards, except in step 3.
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


def _run(model, datasets, names, de, profile=None, progress=None, fit_hook=None):
    before = model.params.free_names
    model.params.fit_only(_free(model, names, datasets))
    fit = Fit(model, datasets)
    fit.progress = progress
    if fit_hook:
        fit_hook(fit)
    try:
        if de:
            fit.run("de", **de)
        if profile and profile[0] in fit.names and not fit.cancelled:
            fit.profile(profile[0], profile[1])
        if not fit.cancelled:
            fit.refine()
        return fit, fit.report()
    finally:
        model.params.fit_only([n for n in before if n in model.params])


def step_film(model, datasets, de=None, thickness_grid=None, extra=(), **kw):
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    de = dict(maxiter=30, popsize=10, seed=1) if de is None else de
    if thickness_grid is None:
        p = model.params["thickness_mean_tl"]
        lo, hi = max(p.min, p.value - 3), min(p.max, p.value + 3)
        thickness_grid = np.arange(lo, hi + 1e-9, 0.5)
    _prepare_scales(model, datasets)
    return _run(model, _subsets(datasets, 0), FILM_PARAMS + list(extra), de,
                ("thickness_mean_tl", thickness_grid), **kw)


def step_surface(model, datasets, de=None, extra=(), **kw):
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    de = dict(maxiter=25, popsize=10, seed=2) if de is None else de
    _prepare_scales(model, datasets)
    return _run(model, _subsets(datasets, 1), SURFACE_PARAMS + list(extra), de, **kw)


def step_all(model, datasets, names=None, progress=None, fit_hook=None):
    """Refine everything listed (default: film + surface parameters) on all rods of all datasets."""
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    _prepare_scales(model, datasets)
    model.params.fit_only(_free(model, names or FILM_PARAMS + SURFACE_PARAMS, datasets))
    fit = Fit(model, datasets)
    fit.progress = progress
    if fit_hook:
        fit_hook(fit)
    fit.refine()
    return fit, fit.report()


def prepare(model, datasets, per_dataset=()):
    """Before a run with several datasets: scopes and scales, and per-dataset copies of
    `per_dataset` (call this on the thread that owns the parameter table)."""
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    _prepare_scales(model, datasets)
    if len(datasets) > 1 and per_dataset:
        from .global_fit import make_per_dataset
        make_per_dataset(model, datasets, [n for n in per_dataset if n in model.params])


def _prepare_scales(model, datasets):
    """Several datasets: give each a scope and a scale before the subsets are made."""
    if len(datasets) > 1:
        for i, ds in enumerate(datasets):
            ds.scope = ds.scope or f"ds{i + 1}"
            n = f"{ds.scope}.scale"
            if n not in model.params:
                g = model.params["scale"]
                model.params.add(name=n, value=g.value, min=g.min, max=g.max, fit=False, unit=g.unit, group="scale",
                                 description=f"Scale of dataset {ds.name}")

"""Parameter presets: start values, bounds and fit flags (and the Per dataset / Per film lists) applied
to the current model, which keeps its settings (energy, lattice, ...). A value can depend on the
dataset's potential: its copy in each dataset takes the rule that matches the dataset's potential_V.

A preset is a JSON file:

    {"name": "...", "notes": ["..."],
     "per_dataset": "theta, z_H2O, dz_Obr, t_liquid",          # optional: the window's lists
     "per_film": "thickness_mean_tl, roughness_tl, eps_perp",
     "parameters": {
         "z_OH":   {"value": 2.07, "fit": false},
         "theta":  {"value": 0.95, "min": 0.7, "max": 1.0, "fit": true},
         "dz_Obr": {"value": 0.35, "min": -0.1, "max": 0.5, "fit": true,
                    "by_potential": [[0.6, 0.9, 0.2]]}}}       # [min V, max V, value]; null = open end

    notes = apply_preset(model, datasets, read_preset(path))

With several datasets the per-dataset and per-film copies of the preset's fitted parameters are made
first (workflow.prepare), so the potential rules land on each dataset's own copy. A dataset without a
potential (OCV) takes the plain value.
"""
import json
import math

from . import workflow

FIELDS = ("value", "min", "max", "fit", "by_potential")


def read_preset(path):
    with open(path, encoding="utf-8") as fh:
        preset = json.load(fh)
    check_preset(preset)
    return preset


def check_preset(preset):
    pars = preset.get("parameters")
    if not isinstance(pars, dict) or not pars:
        raise ValueError("a preset needs a 'parameters' object: {name: {value, min, max, fit, by_potential}}")
    for name, spec in pars.items():
        unknown = set(spec) - set(FIELDS)
        if unknown:
            raise ValueError(f"preset {name}: unknown fields {sorted(unknown)} (use {', '.join(FIELDS)})")
        for rule in spec.get("by_potential", ()):
            if len(rule) != 3:
                raise ValueError(f"preset {name}: by_potential rules are [min V, max V, value]")
    return preset


def _names(names):
    """A list of names from 'a, b; c' or from a list."""
    if isinstance(names, (list, tuple)):
        return [str(n).strip() for n in names if str(n).strip()]
    return [n.strip() for n in str(names or "").replace(";", ",").split(",") if n.strip()]


def value_for(spec, potential):
    """The preset value for a dataset at this potential (None: no potential, the plain value)."""
    if potential is not None:
        for lo, hi, v in spec.get("by_potential", ()):
            if (lo is None or potential >= lo) and (hi is None or potential <= hi):
                return float(v)
    return spec.get("value")


def _set(p, spec, value, notes):
    if p.linked:
        return
    if "min" in spec:
        p.min = float(spec["min"])
    if "max" in spec:
        p.max = float(spec["max"])
    if p.min > p.max:
        raise ValueError(f"preset {p.name}: min {p.min:g} > max {p.max:g}")
    if value is not None:
        v = float(value)
        if not p.min <= v <= p.max:
            v2 = min(max(v, p.min), p.max)
            notes.append(f"{p.name}: {v:g} is outside [{p.min:g}, {p.max:g}], set to {v2:g}")
            v = v2
        p.value = v
    if "fit" in spec:
        p.fit = bool(spec["fit"])


def apply_preset(model, datasets=(), preset=None, per_dataset=None, per_film=None):
    """Apply a preset to the model (and to the copies of its datasets). per_dataset / per_film: the lists
    to make copies for (default: the preset's, else none). Returns notes (unknown names, clipped values)."""
    check_preset(preset)
    P, notes = model.params, []
    datasets = list(datasets)
    pars = preset["parameters"]
    for name, spec in pars.items():
        if name not in P:
            notes.append(f"{name}: not a parameter of this model (skipped)")
            continue
        _set(P[name], spec, spec.get("value"), notes)
    if len(datasets) > 1:                           # copies for the fitted ones, then the per-potential values
        per = _names(per_dataset if per_dataset is not None else preset.get("per_dataset"))
        film = [n for n in _names(per_film if per_film is not None else preset.get("per_film")) if n not in per]
        active = lambda names: [n for n in names if n in P and (P[n].fit or _has_copies(P, n, datasets))]
        workflow.prepare(model, datasets, active(per), active(film))
        for ds in datasets:
            pot = ds.meta.get("potential_V")
            pot = None if pot is None or (isinstance(pot, float) and math.isnan(pot)) else float(pot)
            for name, spec in pars.items():
                cn = f"{ds.scope}.{name}"
                if ds.scope and cn in P:
                    _set(P[cn], spec, value_for(spec, pot), notes)
    return notes


def _has_copies(P, name, datasets):
    return any(ds.scope and f"{ds.scope}.{name}" in P for ds in datasets)

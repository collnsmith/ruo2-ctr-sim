"""Parameter presets: start values, bounds and fit flags (and the Per dataset / Per film lists) applied
to the current model, which keeps its settings (energy, lattice, ...). A value can depend on the
dataset's potential (by_potential, matched to potential_V) or on its film thickness (by_thickness, matched
to thickness_nm): each dataset's copy takes the rule that matches it.

A preset is a JSON file:

    {"name": "...", "notes": ["..."],
     "per_dataset": "theta, z_H2O, dz_Obr, t_liquid",          # optional: the window's lists
     "per_film": "thickness_mean_tl, roughness_tl, eps_perp",
     "parameters": {
         "z_OH":   {"value": 2.07, "fit": false},
         "theta":  {"value": 0.95, "min": 0.7, "max": 1.0, "fit": true},
         "dz_Obr": {"value": 0.35, "min": -0.1, "max": 0.5, "fit": true,
                    "by_potential": [[0.6, 0.9, 0.2]]},          # [min V, max V, value]; null = open end
         "relax_001": {"value": 0, "fit": false,
                       "by_thickness": [[5.5, null, {"value": 0.3, "max": 1, "fit": true}]]}}}
                                                                # [min nm, max nm, value or {value, min, max, fit}]

    notes = apply_preset(model, datasets, read_preset(path))

A rule's last item is a value or an object of value / min / max / fit. The plain fields apply first, then a
matching by_thickness rule, then a matching by_potential rule (the first match of each). With several
datasets the per-dataset and per-film copies are made first (workflow.prepare) for the preset's fitted
parameters and for those whose rules differ between the datasets, so the rules land on each dataset's own
copy (a per-film copy takes its film's thickness). With one dataset the rules apply to the parameters
themselves. A dataset without a potential (OCV) or thickness takes the plain fields.
"""
import json
import math

from . import workflow

FIELDS = ("value", "min", "max", "fit", "by_potential", "by_thickness")
RULE_FIELDS = ("value", "min", "max", "fit")
RULES = (("by_thickness", "min nm, max nm, value"), ("by_potential", "min V, max V, value"))


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
        for key, form in RULES:
            for rule in spec.get(key, ()):
                if len(rule) != 3:
                    raise ValueError(f"preset {name}: {key} rules are [{form}]")
                if isinstance(rule[2], dict) and set(rule[2]) - set(RULE_FIELDS):
                    raise ValueError(f"preset {name}: a {key} rule sets only {', '.join(RULE_FIELDS)}")
    return preset


def _names(names):
    """A list of names from 'a, b; c' or from a list."""
    if isinstance(names, (list, tuple)):
        return [str(n).strip() for n in names if str(n).strip()]
    return [n.strip() for n in str(names or "").replace(";", ",").split(",") if n.strip()]


def _match(rules, x):
    """The fields of the first rule [lo, hi, value or {fields}] that contains x (None: no match)."""
    if x is None:
        return None
    for lo, hi, v in rules:
        if (lo is None or x >= lo) and (hi is None or x <= hi):
            return dict(v) if isinstance(v, dict) else dict(value=float(v))
    return None


def fields_for(spec, potential=None, thickness=None):
    """value / min / max / fit for a dataset at this potential (V) and film thickness (nm): the plain fields,
    then the matching by_thickness rule, then the matching by_potential rule."""
    out = {k: spec[k] for k in RULE_FIELDS if k in spec}
    for key, x in (("by_thickness", thickness), ("by_potential", potential)):
        hit = _match(spec.get(key, ()), x)
        if hit:
            out.update(hit)
    return out


def value_for(spec, potential, thickness=None):
    """The preset value for a dataset at this potential and thickness (None: the plain value)."""
    return fields_for(spec, potential, thickness).get("value")


def _number(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def dataset_keys(ds):
    """(potential V, thickness nm) of a dataset for the preset rules; None where unknown."""
    return _number(ds.meta.get("potential_V")), _number(ds.meta.get("thickness_nm"))


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
    to make copies for (default: the preset's, else none). Returns notes (unknown names, clipped values,
    rules that cannot apply because the parameter is shared)."""
    check_preset(preset)
    P, notes = model.params, []
    datasets = list(datasets)
    pars = {n: spec for n, spec in preset["parameters"].items() if n in P}
    for name in preset["parameters"]:
        if name not in P:
            notes.append(f"{name}: not a parameter of this model (skipped)")
    keys = [dataset_keys(ds) for ds in datasets]
    for name, spec in pars.items():                 # one dataset: its rules apply to the parameter itself
        _set(P[name], fields_for(spec, *keys[0]) if len(datasets) == 1 else spec,
             value_for(spec, *keys[0]) if len(datasets) == 1 else spec.get("value"), notes)
    if len(datasets) > 1:                           # copies for the fitted / varying ones, then each one's rules
        per = _names(per_dataset if per_dataset is not None else preset.get("per_dataset"))
        film = [n for n in _names(per_film if per_film is not None else preset.get("per_film")) if n not in per]
        varies = {n for n, spec in pars.items() if len({repr(sorted(fields_for(spec, *k).items())) for k in keys}) > 1}
        active = lambda names: [n for n in names if n in P and (P[n].fit or n in varies or _has_copies(P, n, datasets))]
        workflow.prepare(model, datasets, active(per), active(film))
        for name in sorted(varies):
            if not _has_copies(P, name, datasets):
                notes.append(f"{name}: its values differ by potential or thickness, but it is shared (not in the "
                             "Per dataset / Per film lists): the plain value is used")
        for ds, (pot, thick) in zip(datasets, keys):
            for name, spec in pars.items():
                cn = f"{ds.scope}.{name}"
                if ds.scope and cn in P:
                    f = fields_for(spec, pot, thick)
                    _set(P[cn], f, f.get("value"), notes)
    return notes


def _has_copies(P, name, datasets):
    return any(ds.scope and f"{ds.scope}.{name}" in P for ds in datasets)

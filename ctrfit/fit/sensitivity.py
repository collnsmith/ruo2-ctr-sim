"""Where along a rod a parameter changes the model: the relative change of |F| when one parameter
moves by +-step (its fit error, else a small step), used to colour the model curve.

    s, step, note = curve_effect(model.params, "x_OH", lambda v: ev(v), scope="ds2")
    # s[i] = |ln F(p + step) - ln F(p - step)| / 2 at each L of the evaluator (about the fractional change)

A link that follows the parameter moves with it (e.g. z_H2O = z_OH + 0.4). The parameter values are
restored afterwards; call it when no fit is running.
"""
import math
import re

import numpy as np

from ..model.parameters import scoped_view


def effect_step(p, error=None):
    """The step for a parameter: its fit error when known, else 2 % of its value but at most 1 % of
    its bound range (1 % of the range for a value of 0; 0.01 without bounds)."""
    if error is not None and math.isfinite(error) and error > 0:
        return float(error), "1σ"
    span = p.max - p.min if math.isfinite(p.max - p.min) and p.max > p.min else None
    cands = [c for c in (0.02 * abs(p.value) if p.value else None, 0.01 * span if span else None) if c]
    return (float(min(cands)) if cands else 0.01), "step"


def target(params, name, scope=""):
    """The parameter that sets `name` for a dataset of this scope: a shared parameter the dataset
    replaces with its own copy -> that copy; a copy that follows another by a plain link (one value per
    film) -> the one it follows. Anything else: name itself."""
    if name in params and not params[name].scope and scope and f"{scope}.{name}" in params:
        name = f"{scope}.{name}"
    seen = set()
    while name in params and params[name].linked and name not in seen:
        seen.add(name)
        ref = params[name].expr.strip()
        if not re.fullmatch(r"[A-Za-z_][\w.]*", ref):
            break
        try:
            name = params.resolve(ref, params[name].scope)
        except Exception:
            break
    return name


def affects(params, name, scope=""):
    """None if `name` can be moved to see what it does to a dataset of this scope, else the reason."""
    p = params[name]
    if p.linked:
        return f"{name} is linked ({p.expr}): select the parameter it follows"
    if not p.scope and scope and f"{scope}.{p.base}" in params:
        return f"this dataset uses its own copy {scope}.{p.base}"
    return None


def curve_effect(params, name, evaluate, scope="", step=None, error=None):
    """(s, step, note): s the relative change of |F| at every point of `evaluate` (a function of the values
    one dataset sees, returning |F|) for name +- step; None and the reason when the parameter does not
    act on this dataset. step defaults to effect_step(error)."""
    why = affects(params, name, scope)
    if why:
        return None, None, why
    p = params[name]
    if step is None:
        step, kind = effect_step(p, error)
    else:
        kind = "step"
    v0 = p.value
    lo, hi = max(p.min, v0 - step), min(p.max, v0 + step)
    if hi <= lo:
        return None, step, f"{name} cannot move within its bounds"
    try:
        p.value = hi
        Fp = np.asarray(evaluate(scoped_view(params.values(), scope)), float)
        p.value = lo
        Fm = np.asarray(evaluate(scoped_view(params.values(), scope)), float)
    finally:
        p.value = v0
        params.values()                        # links back to the restored value
    tiny = 1e-300
    s = np.abs(np.log(np.maximum(Fp, tiny)) - np.log(np.maximum(Fm, tiny))) / 2 * (2 * step / (hi - lo))
    if not np.nanmax(s, initial=0.0) > 1e-12:            # e.g. another dataset's copy that nothing here follows
        whose = f" (it belongs to {p.scope})" if p.scope and p.scope != scope else ""
        return None, step, f"{name} does not change this dataset{whose}"
    unit = f" {p.unit}" if p.unit else ""
    return s, step, f"{name} ± {step:.3g}{unit} ({kind})"

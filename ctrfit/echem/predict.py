"""CTR-model intensity at one reflection as the CUS composition changes with potential.

The structure factor is linear in the CUS fractions (each species is its own partly occupied set
of atoms, the empty sites take the rest), so the intensity is a quadratic function of the three
fractions (H2O, OH, O). PointModel evaluates the full CTRModel at 16 compositions, fits that
quadratic exactly and checks it, so a whole I(V) curve or a fit costs microseconds.

Composition path vs potential: states in order (e.g. H2O -> OH -> O), one sigmoidal transition
between neighbours (E0, width). The share past transition k is the product of the sigmoids up to
k, so fractions are never negative even if transitions overlap. `theta` is the CUS coverage of the
adsorbed states ("empty" is a state with no adsorbate). An optional cathodic shift moves every
transition by -shift on the cathodic sweep (hysteresis).
"""
import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ..core.ctrmodel import CTRModel
from ..core.settings import ADSORBATE_SPECIES, DEFAULTS
from .analysis import sigmoid

STATES = ADSORBATE_SPECIES + ("empty",)


def parse_states(text):
    """'H2O -> OH -> O' (also commas or spaces) -> ['H2O', 'OH', 'O']."""
    parts = [p.strip() for p in text.replace("->", ",").replace(">", ",").replace(" ", ",").split(",") if p.strip()]
    norm = {s.lower(): s for s in STATES}
    out = []
    for p in parts:
        if p.lower() not in norm:
            raise ValueError(f"unknown state '{p}': use {', '.join(STATES)}")
        out.append(norm[p.lower()])
    if len(out) < 2:
        raise ValueError("a composition path needs at least two states, e.g. H2O -> OH")
    return out


def parse_transitions(text, n):
    """'0.85, 0.03; 1.15, 0.05' -> E0 [V] and widths [V] for n transitions."""
    rows = [r for r in text.replace("\n", ";").split(";") if r.strip()]
    if len(rows) != n:
        raise ValueError(f"{n} transition(s) needed (one 'E0, width' per state change), got {len(rows)}")
    E, w = [], []
    for r in rows:
        vals = [float(x) for x in r.replace(",", " ").split()]
        if len(vals) != 2 or vals[1] <= 0:
            raise ValueError(f"transition '{r.strip()}': give 'E0 (V), width (V > 0)'")
        E.append(vals[0])
        w.append(vals[1])
    return np.array(E), np.array(w)


def state_vector(state, theta=1.0):
    v = np.zeros(len(ADSORBATE_SPECIES))
    if state != "empty":
        v[ADSORBATE_SPECIES.index(state)] = theta
    return v


def comp_vector(comp):
    return np.array([float(comp.get(s, 0.0)) for s in ADSORBATE_SPECIES])


class PointModel:
    """|F|^2 of the full CTR model at (H K L) for any CUS composition, via an exact quadratic."""

    def __init__(self, params=None, hkl=(0, 1, 1.5), mode=None):
        self.params = dict(DEFAULTS, **(params or {}))
        self.hkl = tuple(float(x) for x in hkl)
        self.model = CTRModel(self.params)
        self.mode = mode
        H, K, L = self.hkl
        if abs(H - round(H)) > 1e-9 or abs(K - round(K)) > 1e-9:
            raise ValueError("H and K must be integers (rods); L can be any value")
        self.H, self.K, self.L = int(round(H)), int(round(K)), L
        self._fit_quadratic()

    def direct(self, c):
        comp = {s: float(x) for s, x in zip(ADSORBATE_SPECIES, c) if x > 0}
        return float(self.model.intensity(self.H, self.K, [self.L], comp, self.mode)[0])

    @staticmethod
    def _features(C):
        C = np.atleast_2d(C)
        cols = [np.ones(len(C))] + [C[:, i] for i in range(3)]
        cols += [C[:, i] * C[:, j] for i, j in itertools.combinations_with_replacement(range(3), 2)]
        return np.column_stack(cols)

    def _fit_quadratic(self):
        design = [np.zeros(3)]
        for i in range(3):
            for x in (0.5, 1.0):
                v = np.zeros(3)
                v[i] = x
                design.append(v)
        for i, j in itertools.combinations(range(3), 2):
            for a, b in ((0.5, 0.5), (0.25, 0.5)):
                v = np.zeros(3)
                v[i], v[j] = a, b
                design.append(v)
        design.append(np.array([1 / 3, 1 / 3, 1 / 3]))
        D = np.array(design)
        y = np.array([self.direct(c) for c in D])
        self.coef = np.linalg.lstsq(self._features(D), y, rcond=None)[0]
        check = np.array([[0.2, 0.3, 0.1], [0.6, 0.1, 0.25], [0.05, 0.7, 0.2]])
        yc = np.array([self.direct(c) for c in check])
        err = np.max(np.abs(self.intensity(check) - yc) / np.maximum(np.abs(yc), 1e-12))
        self.exact = bool(err < 1e-7)
        self.check_error = float(err)

    def intensity(self, C):
        C = np.atleast_2d(np.asarray(C, float))
        if getattr(self, "exact", True):           # (also while the quadratic is being checked)
            return self._features(C) @ self.coef
        return np.array([self.direct(c) for c in C])

    def of_comp(self, comp):
        return float(self.intensity(comp_vector(comp))[0])


# ----------------------------------------------------------------------------------------------
# composition path vs potential
# ----------------------------------------------------------------------------------------------
def path_fractions(V, states, E0, widths, theta=1.0, branch=None, cathodic_shift=0.0):
    """(N, 3) fractions of H2O, OH, O along the path (see the module docstring)."""
    V = np.asarray(V, float)
    E0, widths = np.asarray(E0, float), np.asarray(widths, float)
    if len(E0) != len(states) - 1 or len(widths) != len(states) - 1:
        raise ValueError(f"{len(states)} states need {len(states) - 1} transitions")
    shift = np.zeros_like(V) if branch is None else np.where(np.asarray(branch) < 0, cathodic_shift, 0.0)
    share = np.ones_like(V)                         # fraction that has passed the transitions so far
    occ = []
    for k in range(len(states)):
        if k < len(states) - 1:
            s = sigmoid((V - (E0[k] - shift)) / widths[k])
            occ.append(share * (1 - s))
            share = share * s
        else:
            occ.append(share)
    C = np.zeros((V.size, 3))
    for st, f in zip(states, occ):
        C += f[:, None] * state_vector(st, theta)[None, :]
    return C


def simulate_path(pm, V, states, E0, widths, theta=1.0, branch=None, cathodic_shift=0.0):
    return pm.intensity(path_fractions(V, states, E0, widths, theta, branch, cathodic_shift))


@dataclass
class PathFit:
    states: list
    E0: np.ndarray
    E0_err: np.ndarray
    widths: np.ndarray
    widths_err: np.ndarray
    scale: float
    scale_err: float
    theta: float
    cathodic_shift: float
    shift_err: float
    red_chi2: float
    notes: list = field(default_factory=list)

    def curve(self, pm, V, branch=None):
        return self.scale * simulate_path(pm, V, self.states, self.E0, self.widths, self.theta, branch,
                                          self.cathodic_shift)


def fit_path(pm, V, I, sigma, states, E0, widths, theta=1.0, branch=None, fit_shift=False, fit_theta=False,
             scale=None):
    """Fit transitions (E0, widths), the intensity scale and optionally the cathodic shift and the
    coverage, so that scale * I_model(V) matches the data."""
    V, I = np.asarray(V, float), np.asarray(I, float)
    ok = np.isfinite(V) & np.isfinite(I)
    br = None if branch is None else np.asarray(branch)[ok]
    V, I = V[ok], I[ok]
    s = np.asarray(sigma, float)[ok] if sigma is not None else np.full(V.size, max(np.std(I), 1e-12) * 0.05)
    s = np.where(s > 0, s, 1.0)
    n = len(states) - 1
    vmin, vmax = V.min(), V.max()
    span = max(vmax - vmin, 1e-3)
    if scale is None:
        m0 = simulate_path(pm, V, states, E0, widths, theta, br)
        scale = float(np.sum(I * m0 / s ** 2) / np.sum(m0 ** 2 / s ** 2)) if np.any(m0) else 1.0
    p0 = [scale, *E0, *widths] + ([0.0] if fit_shift else []) + ([theta] if fit_theta else [])
    lb = [0.0] + [vmin - 0.2 * span] * n + [1e-3] * n + ([-0.5 * span] if fit_shift else []) + ([0.0] if fit_theta else [])
    ub = [np.inf] + [vmax + 0.2 * span] * n + [span] * n + ([0.5 * span] if fit_shift else []) + ([1.0] if fit_theta else [])
    p0 = np.clip(p0, np.array(lb) + 1e-12, np.array(ub) - 1e-12)

    def unpack(p):
        i = 1 + 2 * n
        sh = p[i] if fit_shift else 0.0
        th = p[i + int(fit_shift)] if fit_theta else theta
        return p[0], p[1:1 + n], p[1 + n:1 + 2 * n], sh, th

    def res(p):
        sc, e, w, sh, th = unpack(p)
        return (sc * simulate_path(pm, V, states, e, w, th, br, sh) - I) / s

    r = least_squares(res, p0, bounds=(lb, ub), x_scale="jac", max_nfev=5000)
    dof = max(V.size - len(p0), 1)
    red = float(np.sum(r.fun ** 2) / dof)
    err = np.sqrt(np.clip(np.diag(np.linalg.pinv(r.jac.T @ r.jac)) * max(red, 1.0), 0, None))
    sc, e, w, sh, th = unpack(r.x)
    i = 1 + 2 * n
    out = PathFit(list(states), np.array(e), err[1:1 + n], np.array(w), err[1 + n:1 + 2 * n], float(sc),
                  float(err[0]), float(th), float(sh), float(err[i]) if fit_shift else 0.0, red)
    if fit_theta and (th < 0.02 or th > 0.98):
        out.notes.append(f"coverage at its limit ({th:.2f})")
    if not pm.exact:
        out.notes.append(f"the quadratic shortcut was not exact (error {pm.check_error:.1e}); direct evaluation used")
    return out


# ----------------------------------------------------------------------------------------------
# coverage from intensity
# ----------------------------------------------------------------------------------------------
@dataclass
class CoverageResult:
    p: np.ndarray                    # path coordinate: 0 = first state, 1 = second, ...
    fractions: np.ndarray            # (N, 3) H2O, OH, O
    ambiguous: np.ndarray            # more than one composition on the path gives this intensity
    out_of_range: np.ndarray         # no composition on the path gives it (closest used)
    grid_p: np.ndarray
    grid_I: np.ndarray
    notes: list = field(default_factory=list)


def path_grid(pm, states, theta=1.0, n=400):
    """Intensity along the path with linear mixing between neighbouring states."""
    p = np.linspace(0, len(states) - 1, n * (len(states) - 1) + 1)
    k = np.minimum(np.floor(p).astype(int), len(states) - 2)
    f = p - k
    C = np.array([(1 - fi) * state_vector(states[ki], theta) + fi * state_vector(states[ki + 1], theta)
                  for ki, fi in zip(k, f)])
    return p, pm.intensity(C), C


def coverage_from_intensity(pm, I_rel, states, theta=1.0, start_p=0.0, sigma=None):
    """Where on the path each (scaled) intensity sits. I_rel = I_measured / scale (model units).
    Points are followed in order: among several solutions the one closest to the previous point is
    taken (the first point starts near start_p). With sigma, a point counts as out of range only
    when it misses the path's intensity range by more than 2 sigma (noise is no error)."""
    gp, gI, gC = path_grid(pm, states, theta)
    I_rel = np.asarray(I_rel, float)
    p_out = np.full(I_rel.size, np.nan)
    amb = np.zeros(I_rel.size, bool)
    oor = np.zeros(I_rel.size, bool)
    prev = start_p
    diff = gI[None, :] - I_rel[:, None]
    for i, r in enumerate(I_rel):
        if not np.isfinite(r):
            continue
        d = diff[i]
        cross = np.flatnonzero(np.sign(d[:-1]) * np.sign(d[1:]) <= 0)
        if cross.size == 0:
            j = int(np.argmin(np.abs(d)))
            p_out[i] = gp[j]
            oor[i] = sigma is None or abs(d[j]) > 2 * float(np.asarray(sigma)[i])
        else:
            roots = gp[cross] + (gp[cross + 1] - gp[cross]) * d[cross] / np.where(d[cross] - d[cross + 1] == 0, 1,
                                                                                  d[cross] - d[cross + 1])
            roots = np.unique(np.round(roots, 6))
            amb[i] = roots.size > 1
            p_out[i] = roots[np.argmin(np.abs(roots - prev))]
        prev = p_out[i]
    k = np.clip(np.floor(np.nan_to_num(p_out)).astype(int), 0, len(states) - 2)
    f = np.nan_to_num(p_out) - k
    fr = np.array([(1 - fi) * state_vector(states[ki], theta) + fi * state_vector(states[ki + 1], theta)
                   for ki, fi in zip(k, f)])
    fr[~np.isfinite(p_out)] = np.nan
    res = CoverageResult(p_out, fr, amb, oor, gp, gI)
    if amb.any():
        res.notes.append(f"{int(amb.sum())} point(s) match more than one composition on the path (intensity not "
                         "monotonic along it): the one closest to the previous point was used")
    if oor.any():
        res.notes.append(f"{int(oor.sum())} point(s) are outside the intensity range of the path: check the scale, "
                         "background, path or coverage")
    return res


def scale_from_reference(pm, I_ref, comp_ref):
    """Scale (counts per model intensity) that puts the relaxation-period intensity on a known state."""
    m = pm.of_comp(comp_ref)
    if m <= 0:
        raise ValueError("the model intensity of the reference state is zero at this reflection")
    return float(I_ref) / m


__all__ = ["STATES", "parse_states", "parse_transitions", "PointModel", "path_fractions", "simulate_path",
           "PathFit", "fit_path", "CoverageResult", "coverage_from_intensity", "path_grid", "scale_from_reference"]

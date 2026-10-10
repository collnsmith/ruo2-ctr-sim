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

Near-surface oxygen (NearSurface): the lattice O of the top ns_tl trilayers (simulator setting)
get an extra B factor dB and a u change du, switched on by their own sigmoid in potential (by
default the first transition's E0 and width), e.g. for proton insertion. It changes the
oxygen-only Bragg peaks (H + K odd rods) on both flanks.
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
    """|F|^2 of the full CTR model at (H K L) for any CUS composition, via an exact quadratic.

    With a near-surface oxygen change (extra lattice-O B factor dB and u change du in the top
    ns_tl trilayers, on top of the settings' ns_B_O / ns_du), the quadratic is fitted on a grid of
    (dB, du) and its coefficients are interpolated (cubic) between grid nodes."""

    B_STEP, U_NODES = 0.5, 7

    def __init__(self, params=None, hkl=(0, 1, 1.5), mode=None):
        from ..core.fast import FastCTR
        self.params = dict(DEFAULTS, **(params or {}))
        self.hkl = tuple(float(x) for x in hkl)
        self.model = CTRModel(self.params)
        self.mode = mode
        H, K, L = self.hkl
        if abs(H - round(H)) > 1e-9 or abs(K - round(K)) > 1e-9:
            raise ValueError("H and K must be integers (rods); L can be any value")
        self.H, self.K, self.L = int(round(H)), int(round(K)), L
        self._fast = FastCTR(self.H, self.K, [self.L])
        self._base_ns = (self.model.ns_B_O, self.model.ns_du)
        self._coef_cache = {}
        self._grid = None
        self.coef = self._coefficients(0.0, 0.0)
        self._check()

    # ------------------------------------------------------------------ direct evaluation
    def direct(self, c, dB=0.0, du=0.0):
        comp = {s: float(x) for s, x in zip(ADSORBATE_SPECIES, c) if x > 0}
        m = self.model
        m.ns_B_O, m.ns_du = self._base_ns[0] + float(dB), self._base_ns[1] + float(du)
        try:
            return float(self._fast.intensity(m, comp, self.mode)[0])
        finally:
            m.ns_B_O, m.ns_du = self._base_ns

    @staticmethod
    def _features(C):
        C = np.atleast_2d(C)
        cols = [np.ones(len(C))] + [C[:, i] for i in range(3)]
        cols += [C[:, i] * C[:, j] for i, j in itertools.combinations_with_replacement(range(3), 2)]
        return np.column_stack(cols)

    _DESIGN = None

    @classmethod
    def _design(cls):
        if cls._DESIGN is None:
            d = [np.zeros(3)]
            for i in range(3):
                for x in (0.5, 1.0):
                    v = np.zeros(3)
                    v[i] = x
                    d.append(v)
            for i, j in itertools.combinations(range(3), 2):
                for a, b in ((0.5, 0.5), (0.25, 0.5)):
                    v = np.zeros(3)
                    v[i], v[j] = a, b
                    d.append(v)
            d.append(np.array([1 / 3, 1 / 3, 1 / 3]))
            cls._DESIGN = np.array(d)
        return cls._DESIGN

    def _coefficients(self, dB, du):
        key = (round(float(dB), 9), round(float(du), 9))
        if key not in self._coef_cache:
            D = self._design()
            y = np.array([self.direct(c, dB, du) for c in D])
            self._coef_cache[key] = np.linalg.lstsq(self._features(D), y, rcond=None)[0]
        return self._coef_cache[key]

    def _check(self):
        check = np.array([[0.2, 0.3, 0.1], [0.6, 0.1, 0.25], [0.05, 0.7, 0.2]])
        yc = np.array([self.direct(c) for c in check])
        err = np.max(np.abs(self._features(check) @ self.coef - yc) / np.maximum(np.abs(yc), 1e-12))
        self.exact = bool(err < 1e-7)
        self.check_error = float(err)

    # ------------------------------------------------------------------ near-surface grid
    @property
    def ns_available(self):
        return self.model.ns_tl > 0

    def _ensure_grid(self, dB, du):
        from scipy.interpolate import CubicSpline, RegularGridInterpolator
        if not self.ns_available:
            raise ValueError("a near-surface oxygen change needs 'Affected top trilayers' (ns_tl) > 0 in the "
                             "simulator settings")
        b_lo, b_hi = min(0.0, float(np.min(dB))), max(0.0, float(np.max(dB)))
        u_lo, u_hi = min(0.0, float(np.min(du))), max(0.0, float(np.max(du)))
        g = self._grid
        if g is not None and g["b"][0] <= b_lo and g["b"][-1] >= b_hi and g["u"][0] <= u_lo and g["u"][-1] >= u_hi:
            return g
        b_lo = np.floor(b_lo / self.B_STEP) * self.B_STEP
        b_hi = np.ceil(b_hi / self.B_STEP) * self.B_STEP
        nb = max(4, int(round((b_hi - b_lo) / self.B_STEP)) + 1)
        if b_hi - b_lo < 3 * self.B_STEP:
            b_hi = b_lo + 3 * self.B_STEP
        bn = np.linspace(b_lo, b_hi, nb)
        un = np.array([0.0]) if u_lo == u_hi == 0.0 else np.linspace(u_lo, u_hi if u_hi > u_lo else u_lo + 1e-4,
                                                                        self.U_NODES)
        coef = np.array([[self._coefficients(b, u) for u in un] for b in bn])        # (nb, nu, 10)
        if un.size == 1:
            spl = CubicSpline(bn, coef[:, 0, :], axis=0)
            fn = lambda b, u: spl(b)                                                 # noqa: E731
        else:
            rgi = RegularGridInterpolator((bn, un), coef, method="cubic")
            fn = lambda b, u: rgi(np.column_stack([b, u]))                            # noqa: E731
        self._grid = dict(b=bn, u=un, fn=fn)
        return self._grid

    def coefficients_at(self, dB, du=None):
        """(N, 10) quadratic coefficients for per-point near-surface changes."""
        dB = np.atleast_1d(np.asarray(dB, float))
        du = np.zeros_like(dB) if du is None else np.broadcast_to(np.asarray(du, float), dB.shape)
        if not np.any(dB) and not np.any(du):
            return np.broadcast_to(self.coef, (dB.size, self.coef.size))
        g = self._ensure_grid(dB, du)
        return np.atleast_2d(g["fn"](dB, du))

    # ------------------------------------------------------------------ evaluation
    def intensity(self, C, dB=None, du=None):
        """Intensity for compositions C (N, 3), optionally with near-surface changes dB, du (scalars or
        per row)."""
        C = np.atleast_2d(np.asarray(C, float))
        F = self._features(C)
        if dB is None and du is None:
            if self.exact:
                return F @ self.coef
            return np.array([self.direct(c) for c in C])
        dB = np.broadcast_to(np.asarray(0.0 if dB is None else dB, float), (len(C),))
        du = np.broadcast_to(np.asarray(0.0 if du is None else du, float), (len(C),))
        if not self.exact:
            return np.array([self.direct(c, b, u) for c, b, u in zip(C, dB, du)])
        return np.einsum("ij,ij->i", F, self.coefficients_at(dB, du))

    def of_comp(self, comp, dB=0.0, du=0.0):
        if not dB and not du:
            return float(self.intensity(comp_vector(comp))[0])
        return float(self.intensity(comp_vector(comp), dB, du)[0])


@dataclass
class NearSurface:
    """Potential-dependent change of the near-surface lattice oxygen: extra B factor dB (A^2) and u
    change du at full extent, switched on by a sigmoid (E0, width); E0 / width None follow the
    first transition of the composition path."""
    dB: float = 0.0
    du: float = 0.0
    E0: float = None
    width: float = None

    @property
    def active(self):
        return bool(self.dB) or bool(self.du)

    def extent(self, V, E0_path, w_path, branch=None, cathodic_shift=0.0):
        V = np.asarray(V, float)
        E = E0_path[0] if self.E0 is None else self.E0
        w = w_path[0] if self.width is None else self.width
        shift = np.zeros_like(V) if branch is None else np.where(np.asarray(branch) < 0, cathodic_shift, 0.0)
        return sigmoid((V - (E - shift)) / w)


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


def simulate_path(pm, V, states, E0, widths, theta=1.0, branch=None, cathodic_shift=0.0, ns=None):
    """Intensity along the composition path; ns (NearSurface) adds the near-surface oxygen change."""
    C = path_fractions(V, states, E0, widths, theta, branch, cathodic_shift)
    if ns is None or not ns.active:
        return pm.intensity(C)
    x = ns.extent(V, E0, widths, branch, cathodic_shift)
    return pm.intensity(C, ns.dB * x, ns.du * x)


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
    ns: object = None                 # NearSurface after the fit (None: not used)
    ns_err: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def curve(self, pm, V, branch=None):
        return self.scale * simulate_path(pm, V, self.states, self.E0, self.widths, self.theta, branch,
                                          self.cathodic_shift, self.ns)


def fit_path(pm, V, I, sigma, states, E0, widths, theta=1.0, branch=None, fit_shift=False, fit_theta=False,
             scale=None, ns=None, fit_ns_B=False, fit_ns_u=False):
    """Fit transitions (E0, widths), the intensity scale and optionally the cathodic shift, the
    coverage and the near-surface oxygen change (ns: NearSurface; its dB and du with fit_ns_B /
    fit_ns_u), so that scale * I_model(V) matches the data."""
    ns = ns or NearSurface()
    if (fit_ns_B or fit_ns_u) and not pm.ns_available:
        raise ValueError("fitting the near-surface oxygen needs 'Affected top trilayers' (ns_tl) > 0")
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
        m0 = simulate_path(pm, V, states, E0, widths, theta, br, ns=ns)
        scale = float(np.sum(I * m0 / s ** 2) / np.sum(m0 ** 2 / s ** 2)) if np.any(m0) else 1.0
    B_MAX, U_MAX = 15.0, 0.02
    p0 = [scale, *E0, *widths] + ([0.0] if fit_shift else []) + ([theta] if fit_theta else []) \
        + ([ns.dB] if fit_ns_B else []) + ([ns.du] if fit_ns_u else [])
    lb = [0.0] + [vmin - 0.2 * span] * n + [1e-3] * n + ([-0.5 * span] if fit_shift else []) \
        + ([0.0] if fit_theta else []) + ([0.0] if fit_ns_B else []) + ([-U_MAX] if fit_ns_u else [])
    ub = [np.inf] + [vmax + 0.2 * span] * n + [span] * n + ([0.5 * span] if fit_shift else []) \
        + ([1.0] if fit_theta else []) + ([B_MAX] if fit_ns_B else []) + ([U_MAX] if fit_ns_u else [])
    p0 = np.clip(p0, np.array(lb) + 1e-12, np.array(ub) - 1e-12)
    if fit_ns_B or fit_ns_u:                        # one grid for the whole fit range
        pm.coefficients_at(np.array([0.0, B_MAX if fit_ns_B else ns.dB]),
                           np.array([-U_MAX if fit_ns_u else ns.du, U_MAX if fit_ns_u else ns.du]))

    def unpack(p):
        i = 1 + 2 * n
        sh = p[i] if fit_shift else 0.0
        i += int(fit_shift)
        th = p[i] if fit_theta else theta
        i += int(fit_theta)
        dB = p[i] if fit_ns_B else ns.dB
        i += int(fit_ns_B)
        du = p[i] if fit_ns_u else ns.du
        return p[0], p[1:1 + n], p[1 + n:1 + 2 * n], sh, th, NearSurface(dB, du, ns.E0, ns.width)

    def res(p):
        sc, e, w, sh, th, nsp = unpack(p)
        return (sc * simulate_path(pm, V, states, e, w, th, br, sh, nsp) - I) / s

    r = least_squares(res, p0, bounds=(lb, ub), x_scale="jac", max_nfev=5000)
    dof = max(V.size - len(p0), 1)
    red = float(np.sum(r.fun ** 2) / dof)
    err = np.sqrt(np.clip(np.diag(np.linalg.pinv(r.jac.T @ r.jac)) * max(red, 1.0), 0, None))
    sc, e, w, sh, th, nsp = unpack(r.x)
    i = 1 + 2 * n
    out = PathFit(list(states), np.array(e), err[1:1 + n], np.array(w), err[1 + n:1 + 2 * n], float(sc),
                  float(err[0]), float(th), float(sh), float(err[i]) if fit_shift else 0.0, red,
                  nsp if nsp.active or fit_ns_B or fit_ns_u else None)
    j = i + int(fit_shift) + int(fit_theta)
    if fit_ns_B:
        out.ns_err["dB"] = float(err[j])
        j += 1
        if nsp.dB > 0.98 * B_MAX:
            out.notes.append(f"near-surface O disorder at its limit ({nsp.dB:.1f} A^2)")
    if fit_ns_u:
        out.ns_err["du"] = float(err[j])
        if abs(nsp.du) > 0.98 * U_MAX:
            out.notes.append(f"near-surface O u change at its limit ({nsp.du:+.4f})")
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


def coverage_from_intensity(pm, I_rel, states, theta=1.0, start_p=0.0, sigma=None, dB=None, du=None):
    """Where on the path each (scaled) intensity sits. I_rel = I_measured / scale (model units).
    Points are followed in order: among several solutions the one closest to the previous point is
    taken (the first point starts near start_p). With sigma, a point counts as out of range only
    when it misses the path's intensity range by more than 2 sigma (noise is no error).
    dB, du (per point): a known near-surface oxygen change at each point (e.g. from a path fit), so
    only the CUS composition is inverted."""
    gp, gI, gC = path_grid(pm, states, theta)
    I_rel = np.asarray(I_rel, float)
    p_out = np.full(I_rel.size, np.nan)
    amb = np.zeros(I_rel.size, bool)
    oor = np.zeros(I_rel.size, bool)
    prev = start_p
    if dB is not None or du is not None:
        dBa = np.broadcast_to(np.nan_to_num(np.asarray(0.0 if dB is None else dB, float)), I_rel.shape)
        dua = np.broadcast_to(np.nan_to_num(np.asarray(0.0 if du is None else du, float)), I_rel.shape)
        coef = pm.coefficients_at(dBa, dua)                                  # (N, 10)
        diff = coef @ pm._features(gC).T - I_rel[:, None]                    # (N, grid)
    else:
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


__all__ = ["STATES", "parse_states", "parse_transitions", "PointModel", "NearSurface", "path_fractions", "simulate_path",
           "PathFit", "fit_path", "CoverageResult", "coverage_from_intensity", "path_grid", "scale_from_reference"]

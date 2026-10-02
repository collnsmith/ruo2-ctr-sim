"""Bragg peak indexing from psic angles: best-guess HKL for peaks found while aligning.

Given the lattice, the energy and the angles of two or more Bragg peaks (orientation unknown):

1. Q of every peak in the phi frame from its angles; |Q| gives candidate reflections.
2. For pairs of peaks, candidate pairs whose angle between Q vectors matches give an orientation U
   (Busing and Levy).
3. Every orientation indexes all peaks; the one that puts most peaks on allowed integer HKL with
   the smallest misfit wins, and U is refined by least squares on all indexed peaks.

Lattice symmetry makes some indexings equivalent: sign changes of H, K, L, and, for the TiO2(110)
surface cell (a2 = a3), swapping K and L. Bragg peaks alone cannot tell them apart. Pass the
surface normal (phi frame, e.g. (0, 0, 1) if it is along the phi axis) to choose the one whose L
axis is along the normal; the result lists the equivalent alternatives either way.
With one peak, only the |Q| candidates are listed.
"""
import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .psic import Psic, as_angles, rotation_from_vector, u_from_two_vectors


# ----------------------------------------------------------------------------------------------
# allowed reflections
# ----------------------------------------------------------------------------------------------
def allowed_all(hkl):
    return True


def allowed_tio2_surface(hkl):
    """Bragg reflections of the TiO2(110) surface cell, from the rutile structure factor:
    K + L even (centring) and the rutile glide/screw extinctions (computed, not tabulated)."""
    return _tio2_rule()(hkl)


_RULE = {}


def _tio2_rule():
    if "tio2" not in _RULE:
        from ..core.ctrmodel import CTRModel
        from ..core.settings import DEFAULTS
        m = CTRModel(dict(DEFAULTS, anom_mode="off"))
        cache = {}

        def rule(hkl):
            h, k, l = (int(round(x)) for x in hkl)
            key = (h, k, l)
            if key not in cache:
                F = abs(m.sf_atoms(m.SUB_CELL, h, k, [float(l)])[0])
                cache[key] = F > 1e-3
            return cache[key]
        _RULE["tio2"] = rule
    return _RULE["tio2"]


ALLOWED = {"all": allowed_all, "tio2_surface": allowed_tio2_surface}


def candidate_reflections(lattice, q_max, allowed=allowed_all, h_max=12):
    """(N, 3) integer HKL with 0 < |Q| <= q_max that the rule allows, and their |Q|."""
    lim = [min(h_max, int(math.ceil(q_max / np.linalg.norm(lattice.B[:, i])) + 1)) for i in range(3)]
    rng = [range(-n, n + 1) for n in lim]
    hkl = np.array([h for h in itertools.product(*rng) if any(h)], float)
    q = lattice.q(hkl)
    keep = q <= q_max
    hkl, q = hkl[keep], q[keep]
    ok = np.array([allowed(h) for h in hkl], bool)
    return hkl[ok].astype(int), q[ok]


# ----------------------------------------------------------------------------------------------
# result
# ----------------------------------------------------------------------------------------------
@dataclass
class IndexedPeak:
    angles: dict
    q_obs: float
    hkl_float: np.ndarray
    hkl: tuple
    dq: float                 # |Q_calc - Q_obs| (1/Å) with the refined UB
    indexed: bool
    forbidden: bool = False   # on integer HKL that the structure forbids (wrong energy/lattice, or not Bragg)


@dataclass
class IndexResult:
    UB: np.ndarray
    U: np.ndarray
    peaks: list
    rms_dq: float
    n_indexed: int
    alternatives: list = field(default_factory=list)   # equivalent indexings: list of lists of HKL
    notes: list = field(default_factory=list)
    candidates: list = field(default_factory=list)     # one-peak mode: [(hkl, |Q|), ...]

    @property
    def summary(self):
        lines = []
        if self.UB is None:
            lines.append("One peak: candidates by |Q| only (no orientation).")
            for h, q in self.candidates:
                lines.append(f"  ({h[0]:>3d} {h[1]:>3d} {h[2]:>3d})  |Q| = {q:.4f} 1/Å")
            return "\n".join(lines + self.notes)
        lines.append(f"{self.n_indexed} of {len(self.peaks)} peaks indexed, rms |dQ| = {self.rms_dq:.2e} 1/Å")
        lines.append(f"{'#':>2} {'del':>8} {'eta':>8} {'chi':>8} {'phi':>8} {'nu':>8} {'mu':>8}   "
                     f"{'H':>7} {'K':>7} {'L':>7}   guess        |dQ|")
        for i, p in enumerate(self.peaks):
            a = p.angles
            h = p.hkl_float
            guess = (f"({p.hkl[0]} {p.hkl[1]} {p.hkl[2]})" if p.indexed else
                     f"forbidden ({p.hkl[0]} {p.hkl[1]} {p.hkl[2]})" if p.forbidden else "not indexed")
            lines.append(f"{i + 1:>2} " + " ".join(f"{a[m]:>8.3f}" for m in ("del", "eta", "chi", "phi", "nu", "mu"))
                         + f"   {h[0]:>7.3f} {h[1]:>7.3f} {h[2]:>7.3f}   {guess:<12} {p.dq:.1e}")
        lines.append("UB (1/Å, 2 pi convention):")
        lines += ["  " + " ".join(f"{x:>10.5f}" for x in row) for row in self.UB]
        if len(self.alternatives) > 1:
            lines.append(f"{len(self.alternatives)} equivalent indexings fit equally well (lattice symmetry):")
            for alt in self.alternatives[:6]:
                lines.append("  " + ", ".join(f"({h[0]} {h[1]} {h[2]})" for h in alt))
        return "\n".join(lines + self.notes)


# ----------------------------------------------------------------------------------------------
# indexing
# ----------------------------------------------------------------------------------------------
def _score(U, B, qs, allowed, tol_hkl):
    UB = U @ B
    hf = np.linalg.solve(UB, qs.T).T
    hi = np.round(hf)
    ok = np.all(np.abs(hf - hi) < tol_hkl, axis=1) & np.any(hi != 0, axis=1)
    ok &= np.array([allowed(h) for h in hi], bool)
    dq = np.linalg.norm((UB @ hi.T).T - qs, axis=1)
    return int(ok.sum()), float(np.sqrt(np.mean(dq[ok] ** 2))) if ok.any() else np.inf, hi.astype(int), ok


def _refine_u(U, B, qs, hkl):
    def resid(r):
        return ((rotation_from_vector(r) @ U @ B @ hkl.T).T - qs).ravel()
    res = least_squares(resid, np.zeros(3), method="lm")
    return rotation_from_vector(res.x) @ U


def index_peaks(peaks, wavelength, lattice, allowed=allowed_all, tol_q=0.01, tol_angle=1.0, tol_hkl=0.15,
                normal_phi=None, geo=None, max_pairs=6):
    """Best-guess HKL for Bragg peaks given as angle sets (dicts or psic-ordered sequences).

    tol_q: relative |Q| tolerance for candidates; tol_angle: degrees between Q vectors;
    tol_hkl: how far from integers an indexed peak may be. normal_phi: surface normal in the phi
    frame, used to choose among equivalent indexings (L along the normal)."""
    geo = geo or Psic()
    angs = [as_angles(p) for p in peaks]
    qs = np.array([geo.q_phi(a, wavelength) for a in angs])
    qn = np.linalg.norm(qs, axis=1)
    if np.any(qn < 1e-9):
        raise ValueError("a peak has Q = 0 (detector at zero)")
    hkl_all, q_all = candidate_reflections(lattice, qn.max() * (1 + tol_q) + 1e-9, allowed)
    cands = [hkl_all[np.abs(q_all - q) <= tol_q * q] for q in qn]
    notes = [f"{i + 1}: no allowed reflection with |Q| = {q:.4f} 1/Å (check the energy, lattice and angles)"
             for i, (q, c) in enumerate(zip(qn, cands)) if len(c) == 0]
    if len(angs) == 1:
        c = cands[0]
        order = np.argsort(np.abs(lattice.q(c) - qn[0])) if len(c) else []
        return IndexResult(None, None, [], np.nan, 0, notes=notes,
                           candidates=[(tuple(int(x) for x in c[i]), float(lattice.q(c[i])[0])) for i in order[:20]])
    B = lattice.B
    # most informative pairs: Q vectors closest to perpendicular, both with candidates
    pairs = sorted(((abs(np.dot(qs[i], qs[j]) / (qn[i] * qn[j])), i, j)
                    for i, j in itertools.combinations(range(len(angs)), 2) if len(cands[i]) and len(cands[j])))
    pairs = [(i, j) for c, i, j in pairs if c < 0.999][:max_pairs]
    if not pairs:
        raise ValueError("need two peaks with candidate reflections and non-parallel Q vectors")
    cos_tol = math.radians(tol_angle)
    solutions = []
    for i, j in pairs:
        ang_obs = math.acos(np.clip(np.dot(qs[i], qs[j]) / (qn[i] * qn[j]), -1, 1))
        for hi in cands[i]:
            vi = B @ hi
            for hj in cands[j]:
                vj = B @ hj
                c = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
                if c > 0.9999 or abs(math.acos(np.clip(c, -1, 1)) - ang_obs) > cos_tol:
                    continue
                U = u_from_two_vectors(vi, vj, qs[i], qs[j])
                n, rms, h_int, ok = _score(U, B, qs, allowed, tol_hkl)
                if n >= 2:
                    solutions.append((n, rms, U, h_int, ok))
    if not solutions:
        raise ValueError("no orientation indexes two peaks; loosen tol_q / tol_angle or check the lattice")
    best_n = max(s[0] for s in solutions)
    top = [s for s in solutions if s[0] == best_n]
    refined = []
    for n, rms, U, h_int, ok in top:
        U2 = _refine_u(U, B, qs[ok], h_int[ok].astype(float))
        n2, rms2, h2, ok2 = _score(U2, B, qs, allowed, tol_hkl)
        refined.append((n2, rms2, U2, h2, ok2))
    best_rms = min(r[1] for r in refined)
    equiv = [r for r in refined if r[0] == best_n and r[1] <= max(3 * best_rms, best_rms + 1e-4)]
    alternatives, seen = [], set()
    for r in equiv:
        key = tuple(map(tuple, r[3]))
        if key not in seen:
            seen.add(key)
            alternatives.append([tuple(int(x) for x in h) for h in r[3]])
    if normal_phi is not None:
        n_hat = np.asarray(normal_phi, float) / np.linalg.norm(normal_phi)

        def along_normal(r):
            l_axis = r[2] @ B @ np.array([0.0, 0.0, 1.0])
            return -float(np.dot(l_axis / np.linalg.norm(l_axis), n_hat))
        equiv.sort(key=lambda r: (along_normal(r), r[1]))
        notes.append("Chosen among the equivalent indexings: L along the given surface normal.")
    else:
        equiv.sort(key=lambda r: r[1])
        if len(alternatives) > 1:
            notes.append("Give the surface normal (phi frame) to choose among the equivalent indexings.")
    n, rms, U, h_int, ok = equiv[0]
    UB = U @ B
    out = []
    for a, q, h, good, qv in zip(angs, qn, h_int, ok, qs):
        hf = np.linalg.solve(UB, qv)
        dq = float(np.linalg.norm(UB @ h - qv)) if good else float("nan")
        forbidden = bool(not good and np.all(np.abs(hf - h) < tol_hkl) and np.any(h != 0))
        out.append(IndexedPeak(a, float(q), hf, tuple(int(x) for x in h), dq, bool(good), forbidden))
    if any(p.forbidden for p in out):
        notes.append("forbidden: the peak sits on integer HKL that the structure does not allow; check the energy "
                     "and lattice, or it is not a substrate Bragg peak (film, multiple scattering, powder).")
    return IndexResult(UB, U, out, rms, n, alternatives=alternatives, notes=notes)

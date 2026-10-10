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

Tags fix the choice: a peak given with want=(H, K, L) (a '[H K L]' tag after the angles) is called
that, by applying the lattice symmetry operation (proper rotation of the metric that keeps every
peak on an allowed reflection) that satisfies the most tags, earlier tags first. Two tagged peaks
with non-parallel Q fix the indexing completely, so it no longer changes when peaks are added. A tag
that cannot be met (not a symmetry equivalent of the peak, or in conflict with earlier tags) is
reported with the closest equivalent that works. Tags win over the surface-normal hint.
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
    want: tuple = None        # the [H K L] tag, if any


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
    tags: list = field(default_factory=list)           # one line per [H K L] tag: met, or why not

    @property
    def summary(self):
        lines = []
        if self.UB is None:
            lines.append("One peak: candidates by |Q| only (no orientation).")
            for h, q in self.candidates:
                lines.append(f"  ({h[0]:>3d} {h[1]:>3d} {h[2]:>3d})  |Q| = {q:.4f} 1/Å")
            return "\n".join(lines + self.tags + self.notes)
        lines.append(f"{self.n_indexed} of {len(self.peaks)} peaks indexed, rms |dQ| = {self.rms_dq:.2e} 1/Å")
        lines.append(f"{'#':>2} {'del':>8} {'eta':>8} {'chi':>8} {'phi':>8} {'nu':>8} {'mu':>8}   "
                     f"{'H':>7} {'K':>7} {'L':>7}   guess        |dQ|")
        for i, p in enumerate(self.peaks):
            a = p.angles
            h = p.hkl_float
            mark = "" if p.want is None else " *" if tuple(p.hkl) == tuple(p.want) and p.indexed else " !"
            guess = (f"({p.hkl[0]} {p.hkl[1]} {p.hkl[2]}){mark}" if p.indexed else
                     f"forbidden ({p.hkl[0]} {p.hkl[1]} {p.hkl[2]})" if p.forbidden else "not indexed")
            lines.append(f"{i + 1:>2} " + " ".join(f"{a[m]:>8.3f}" for m in ("del", "eta", "chi", "phi", "nu", "mu"))
                         + f"   {h[0]:>7.3f} {h[1]:>7.3f} {h[2]:>7.3f}   {guess:<12} {p.dq:.1e}")
        if self.tags:
            lines.append("Tags (* met, ! not met):")
            lines += ["  " + t for t in self.tags]
        lines.append("UB (1/Å, 2 pi convention):")
        lines += ["  " + " ".join(f"{x:>10.5f}" for x in row) for row in self.UB]
        if len(self.alternatives) > 1:
            lines.append(f"{len(self.alternatives)} equivalent indexings fit equally well (lattice symmetry):")
            for alt in self.alternatives[:6]:
                lines.append("  " + ", ".join(f"({h[0]} {h[1]} {h[2]})" for h in alt))
        return "\n".join(lines + self.notes)


# ----------------------------------------------------------------------------------------------
# lattice symmetry and tags
# ----------------------------------------------------------------------------------------------
_OPS = {}


def symmetry_ops(lattice, tol=1e-6):
    """Integer matrices M (entries -1, 0, 1, det +1) that keep the reciprocal metric: if h indexes a
    peak with U, M h indexes it equally well with the rotation U B M^-1 B^-1."""
    key = tuple(round(getattr(lattice, n), 9) for n in ("a", "b", "c", "alpha", "beta", "gamma"))
    if key not in _OPS:
        G = lattice.B.T @ lattice.B
        all_m = np.array(list(itertools.product((-1, 0, 1), repeat=9)), float).reshape(-1, 3, 3)
        all_m = all_m[np.abs(np.linalg.det(all_m) - 1) < 1e-9]
        dev = np.abs(np.einsum("nji,jk,nkl->nil", all_m, G, all_m) - G).max(axis=(1, 2))
        _OPS[key] = [m.astype(int) for m in all_m[dev < tol * np.abs(G).max()]]
    return _OPS[key]


def _fmt(h):
    return "(" + " ".join(str(int(x)) for x in h) + ")"


def _apply_tags(U, B, lattice, h_int, ok, wants, allowed, n_hat):
    """Re-index with the symmetry operation that meets the most tags (earlier tags first, then the
    surface normal). Returns U, h_int and the tag report lines."""
    tagged = [i for i, w in enumerate(wants) if w is not None]
    report = {}
    for i in tagged:
        if not ok[i]:
            report[i] = (f"peak {i + 1}: tag {_fmt(wants[i])} ignored, the peak is not on an allowed reflection")
    use = [i for i in tagged if ok[i]]
    if not use:
        return U, h_int, [report[i] for i in tagged]
    valid = []
    for M in symmetry_ops(lattice):
        h2 = h_int @ M.T
        if all(allowed(h) for h in h2[ok]):
            valid.append((M, h2))
    Binv = np.linalg.inv(B)

    def rot(M):
        return U @ B @ np.linalg.inv(M) @ Binv

    def key(item):
        M, h2 = item
        miss = tuple(tuple(h2[i]) != tuple(wants[i]) for i in use)
        dist = sum(float(np.linalg.norm(h2[i] - np.array(wants[i]))) for i, m in zip(use, miss) if m)
        normal = 0.0
        if n_hat is not None:
            l_axis = rot(M) @ B @ np.array([0.0, 0.0, 1.0])
            normal = -float(np.dot(l_axis / np.linalg.norm(l_axis), n_hat))
        return (sum(miss), miss, round(dist, 9), round(normal, 9), tuple(np.abs(M - np.eye(3)).ravel()))
    valid.sort(key=key)
    M, h_best = valid[0]
    best = key(valid[0])
    for i in use:
        w = tuple(wants[i])
        got = tuple(int(x) for x in h_best[i])
        if got == w:
            report[i] = f"peak {i + 1}: {_fmt(w)} as tagged"
            continue
        equivalents = {tuple(int(x) for x in h2[i]) for _, h2 in valid}
        lat_q = lattice.q(np.array([w, got], float))
        if w in equivalents:
            why = "conflicts with the tags on earlier peaks"
        elif abs(lat_q[0] - lat_q[1]) > 1e-6 * lat_q[1]:
            why = f"|Q| of {_fmt(w)} is {lat_q[0]:.4f} 1/Å, the peak is at {lat_q[1]:.4f} 1/Å"
        elif not allowed(w):
            why = f"{_fmt(w)} is a forbidden reflection"
        else:
            why = f"{_fmt(w)} has the same |Q| but is not a symmetry equivalent of this peak's reflection"
        closest = min(equivalents, key=lambda h: (float(np.linalg.norm(np.subtract(h, w))), h))
        line = f"peak {i + 1}: {_fmt(w)} is not possible ({why}); indexed as {_fmt(got)}, the equivalent that fits"
        line += " the other tags" if len(use) > 1 else ""
        if closest != got:
            line += f"; nearest equivalent overall: {_fmt(closest)} (conflicts with the other tags)"
        report[i] = line
    same = {tuple(map(tuple, h2)) for item in valid if key(item)[:4] == best[:4] for h2 in [item[1]]}
    lines = [report[i] for i in tagged]
    if len(same) > 1:
        lines.append(f"{len(same)} equivalent indexings still agree with the tags: tag one more peak whose Q is "
                     "not parallel to the tagged ones to fix the indexing")
    return rot(M), h_best.astype(int), lines


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
    frame, used to choose among equivalent indexings (L along the normal). A peak dict with
    want=(H, K, L) is a tag (see the module docstring)."""
    geo = geo or Psic()
    angs = [as_angles(p) for p in peaks]
    wants = [tuple(int(x) for x in p["want"]) if isinstance(p, dict) and p.get("want") is not None else None
             for p in peaks]
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
        tags = []
        if wants[0] is not None:
            listed = [tuple(int(x) for x in h) for h in c]
            tags.append(f"peak 1: {_fmt(wants[0])} " + ("is a candidate" if wants[0] in listed else
                                                       "is not among the candidates (|Q| does not match)"))
        return IndexResult(None, None, [], np.nan, 0, notes=notes, tags=tags,
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
    tags = []
    if any(w is not None for w in wants):
        n_hat_ = None if normal_phi is None else np.asarray(normal_phi, float) / np.linalg.norm(normal_phi)
        U, h_int, tags = _apply_tags(U, B, lattice, h_int, ok, wants, allowed, n_hat_)
        notes = [x for x in notes if not x.startswith(("Give the surface normal", "Chosen among"))]
        notes.append("Chosen among the equivalent indexings: the [H K L] tags"
                     + (", then L along the surface normal." if normal_phi is not None else "."))
    UB = U @ B
    out = []
    for a, q, h, good, qv, w in zip(angs, qn, h_int, ok, qs, wants):
        hf = np.linalg.solve(UB, qv)
        dq = float(np.linalg.norm(UB @ h - qv)) if good else float("nan")
        forbidden = bool(not good and np.all(np.abs(hf - h) < tol_hkl) and np.any(h != 0))
        out.append(IndexedPeak(a, float(q), hf, tuple(int(x) for x in h), dq, bool(good), forbidden, w))
    if any(p.forbidden for p in out):
        notes.append("forbidden: the peak sits on integer HKL that the structure does not allow; check the energy "
                     "and lattice, or it is not a substrate Bragg peak (film, multiple scattering, powder).")
    return IndexResult(UB, U, out, rms, n, alternatives=alternatives, notes=notes, tags=tags)

"""Orientation (UB) refinement from a growing list of reflections, in the spirit of SPEC's reflex file.

Keep adding Bragg peaks as they are found (HKL plus the six psic angles, each at its own energy) and
refine UB by least squares on all of them after every addition:

    minimise  sum_i | Q_phi(angles_i + offsets, lambda_i) - U B(lattice) h_i |^2

What is refined is chosen with `free`:

    "orientation"   U only (3 rotation angles); the lattice stays as given. Needs 2 reflections.
    "scale"         U and one common scale of a, b, c (ratios and angles kept).
    "abc"           U and a, b, c (angles kept): strained films, thermal expansion.
    "all"           U and all six lattice parameters.
    "ub"            all nine UB elements (no lattice assumed); the lattice is derived from UB.
                    Needs 3 reflections whose HKL are not coplanar.

A tuple of lattice parameter names, e.g. ("a", "c"), frees just those. `offsets` lists motors whose
zero offset is refined too (true angle = reading + offset); they need more reflections than they
free parameters and are usually only determined with a spread of 2theta.

The result gives UB, U, the lattice with standard errors (from the Jacobian, scaled by the
residual), per-reflection residuals (angle between measured and calculated Q, relative |Q| error,
HKL computed from the measured angles), outliers and warnings about undetermined parameters.
"""
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .psic import (MOTORS, Lattice, Psic, as_angles, energy_to_wavelength, rotation_from_vector,
                   u_from_two_vectors)

LATTICE_NAMES = ("a", "b", "c", "alpha", "beta", "gamma")
PRESETS = {"orientation": (), "scale": "scale", "abc": ("a", "b", "c"), "all": LATTICE_NAMES, "ub": "ub"}
REFLEX_HELP = ("One reflection per line:  H K L  del eta chi phi nu mu  [energy keV]  # label\n"
               "HKL may be left out (six or seven numbers): it is then guessed from the current UB.")


# ----------------------------------------------------------------------------------------------
# reflection list
# ----------------------------------------------------------------------------------------------
@dataclass
class Reflection:
    hkl: tuple
    angles: dict
    energy_kev: float
    label: str = ""
    use: bool = True

    @property
    def wavelength(self):
        return energy_to_wavelength(self.energy_kev)

    def to_dict(self):
        return dict(hkl=[float(x) for x in self.hkl], angles={m: float(self.angles[m]) for m in MOTORS},
                    energy_kev=float(self.energy_kev), label=self.label, use=bool(self.use))

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(float(x) for x in d["hkl"]), as_angles(d["angles"]), float(d["energy_kev"]),
                   d.get("label", ""), bool(d.get("use", True)))

    def line(self):
        h = " ".join(f"{x:g}" for x in self.hkl)
        a = " ".join(f"{self.angles[m]:.4f}" for m in MOTORS)
        return f"{h}  {a}  {self.energy_kev:.4f}" + (f"  # {self.label}" if self.label else "")


def guess_hkl(angles, energy_kev, UB, geo=None):
    """HKL (rounded to integers) of a peak from its angles and the current UB, and the unrounded HKL."""
    geo = geo or Psic()
    h = geo.hkl(as_angles(angles), np.asarray(UB, float), energy_to_wavelength(energy_kev))
    return tuple(float(round(x)) for x in h), h


class ReflectionList:
    """Reflections found so far (the reflex file). JSON round trip and a plain text format."""

    def __init__(self, reflections=None):
        self.items = list(reflections or [])

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, i):
        return self.items[i]

    @property
    def used(self):
        return [r for r in self.items if r.use]

    def add(self, hkl, angles, energy_kev, label="", use=True, UB=None, geo=None):
        """Append a reflection. hkl=None guesses it from UB (rounded to integers)."""
        if hkl is None:
            if UB is None:
                raise ValueError("give the HKL of this reflection (no UB yet to guess it from)")
            hkl = guess_hkl(angles, energy_kev, UB, geo)[0]
            if not any(hkl):
                raise ValueError("these angles are at HKL (0 0 0) with the current UB: give the HKL")
        hkl = tuple(float(x) for x in hkl)
        if len(hkl) != 3:
            raise ValueError("HKL needs three numbers")
        r = Reflection(hkl, as_angles(angles), float(energy_kev), label, use)
        self.items.append(r)
        return r

    def remove(self, index):
        return self.items.pop(index)

    def clear(self):
        self.items.clear()

    def to_dict(self):
        return dict(reflections=[r.to_dict() for r in self.items])

    @classmethod
    def from_dict(cls, d):
        return cls([Reflection.from_dict(x) for x in d.get("reflections", [])])

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        text = Path(path).read_text(encoding="utf-8")
        if text.lstrip().startswith("{"):
            return cls.from_dict(json.loads(text))
        out = cls()
        for row in parse_reflection_lines(text, None):
            out.add(row["hkl"], row["angles"], row["energy_kev"], row["label"])
        return out

    def text(self):
        return "\n".join(["# H K L  del eta chi phi nu mu  energy_keV  # label"] +
                         [("" if r.use else "#off ") + r.line() for r in self.items])


def parse_reflection_lines(text, energy_kev):
    """Rows {hkl (or None), angles, energy_kev, label} from text: 'H K L del eta chi phi nu mu [E] # label',
    or without HKL: 'del eta chi phi nu mu [E] # label'. Lines starting with # are skipped."""
    rows = []
    for raw in text.splitlines():
        line, _, label = raw.partition("#")
        if not line.strip():
            continue
        nums = [float(x) for x in re.split(r"[\s,;]+", line.strip()) if x]
        if len(nums) in (9, 10):
            hkl, ang, rest = nums[:3], nums[3:9], nums[9:]
        elif len(nums) in (6, 7):
            hkl, ang, rest = None, nums[:6], nums[6:]
        else:
            raise ValueError(f"expected 'H K L' + six angles (+ energy), or six angles (+ energy): {raw.strip()}")
        e = rest[0] if rest else energy_kev
        if e is None:
            raise ValueError(f"no energy for: {raw.strip()}")
        rows.append(dict(hkl=hkl, angles=as_angles(ang), energy_kev=float(e), label=label.strip()))
    return rows


# ----------------------------------------------------------------------------------------------
# lattice from UB
# ----------------------------------------------------------------------------------------------
def lattice_from_ub(UB):
    """Direct lattice (Å, degrees) from UB: (UB)^T UB = B^T B is the reciprocal metric (2 pi convention)."""
    Gs = np.asarray(UB, float).T @ np.asarray(UB, float)
    G = (2 * math.pi) ** 2 * np.linalg.inv(Gs)
    a, b, c = np.sqrt(np.diag(G))

    def ang(g, n):
        return math.degrees(math.acos(np.clip(g / n, -1, 1)))
    return Lattice(a, b, c, ang(G[1, 2], b * c), ang(G[0, 2], a * c), ang(G[0, 1], a * b), name="from UB")


def nearest_rotation(M):
    u, _, vt = np.linalg.svd(M)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


# ----------------------------------------------------------------------------------------------
# refinement
# ----------------------------------------------------------------------------------------------
@dataclass
class ReflectionFit:
    index: int                 # position in the list
    hkl: tuple
    hkl_obs: np.ndarray        # HKL from the measured angles with the refined UB
    dq: float                  # |Q_obs - Q_calc| (1/Å)
    dangle: float              # angle between Q_obs and Q_calc (degrees)
    dq_rel: float              # (|Q_obs| - |Q_calc|) / |Q_calc|
    outlier: bool = False


@dataclass
class RefineResult:
    UB: np.ndarray
    U: np.ndarray
    lattice: Lattice
    lattice_err: dict
    offsets: dict
    offsets_err: dict
    fits: list
    rms_dq: float
    rms_angle: float
    free: object
    n_used: int
    n_params: int
    warnings: list = field(default_factory=list)

    @property
    def summary(self):
        lat = self.lattice
        free = self.free if isinstance(self.free, str) else ("lattice " + " ".join(self.free) if self.free
                                                              else "orientation")
        lines = [f"Refined {free}" + (f" + offsets of {' '.join(self.offsets)}" if self.offsets else "")
                 + f" on {self.n_used} reflections ({3 * self.n_used} equations, {self.n_params} parameters)",
                 f"rms |dQ| = {self.rms_dq:.2e} 1/Å, rms angle between measured and calculated Q = "
                 f"{self.rms_angle:.4f}°", ""]
        parts = []
        for n in LATTICE_NAMES:
            v, e = getattr(lat, n), self.lattice_err.get(n)
            unit = " Å" if n in "abc" else "°"
            parts.append(f"{n} = {v:.5f}" + (f" ± {e:.5f}" if e is not None else "") + unit)
        lines += ["Lattice: " + ", ".join(parts[:3]), "         " + ", ".join(parts[3:])]
        for m, v in self.offsets.items():
            e = self.offsets_err.get(m)
            lines.append(f"Offset {m}: {v:+.4f}" + (f" ± {e:.4f}" if e is not None else "") +
                         "°  (true angle = reading + offset)")
        lines += ["", f"{'#':>3} {'H':>6} {'K':>6} {'L':>6}   {'H obs':>8} {'K obs':>8} {'L obs':>8}   "
                      f"{'dangle °':>9} {'d|Q|/|Q|':>10}"]
        for f in self.fits:
            lines.append(f"{f.index + 1:>3} " + " ".join(f"{x:>6g}" for x in f.hkl) + "   "
                         + " ".join(f"{x:>8.4f}" for x in f.hkl_obs)
                         + f"   {f.dangle:>9.4f} {f.dq_rel:>10.2e}" + ("   <- outlier?" if f.outlier else ""))
        lines += ["", "UB (1/Å, 2 pi convention):"] + ["  " + " ".join(f"{x:>10.6f}" for x in row) for row in self.UB]
        lines.append(f"\nSPEC:  setlat {lat.a:.5f} {lat.b:.5f} {lat.c:.5f} {lat.alpha:.4f} {lat.beta:.4f} "
                     f"{lat.gamma:.4f}")
        if self.warnings:
            lines += [""] + [f"Warning: {w}" for w in self.warnings]
        return "\n".join(lines)

    def to_dict(self):
        return dict(UB=self.UB.tolist(), lattice=self.lattice.to_dict(), lattice_err=self.lattice_err,
                    offsets=self.offsets, offsets_err=self.offsets_err, rms_dq=self.rms_dq,
                    rms_angle=self.rms_angle, n_used=self.n_used, warnings=self.warnings)


def _free_names(free):
    free = PRESETS.get(free, free) if isinstance(free, str) else tuple(free)
    if isinstance(free, str):
        return free
    bad = set(free) - set(LATTICE_NAMES)
    if bad:
        raise ValueError(f"unknown lattice parameters {sorted(bad)}; use {', '.join(LATTICE_NAMES)}")
    return tuple(n for n in LATTICE_NAMES if n in free)


def _initial_u(lattice, refl, geo, UB):
    if UB is not None:
        return nearest_rotation(np.asarray(UB, float) @ np.linalg.inv(lattice.B))
    qs = [geo.q_phi(r.angles, r.wavelength) for r in refl]
    cs = [lattice.B @ np.asarray(r.hkl, float) for r in refl]
    for i in range(len(refl)):
        for j in range(i + 1, len(refl)):
            if (np.linalg.norm(np.cross(cs[i], cs[j])) > 1e-3 * np.linalg.norm(cs[i]) * np.linalg.norm(cs[j]) and
                    np.linalg.norm(np.cross(qs[i], qs[j])) > 1e-3 * np.linalg.norm(qs[i]) * np.linalg.norm(qs[j])):
                return u_from_two_vectors(cs[i], cs[j], qs[i], qs[j])
    raise ValueError("need two reflections with non-parallel HKL to start the orientation")


def refine_ub(reflections, lattice, free="orientation", offsets=(), geo=None, UB=None, outlier_factor=3.0):
    """Least-squares UB from all used reflections (see the module docstring). `reflections` is a
    ReflectionList or a list of Reflection; UB (optional) is the starting orientation."""
    geo = geo or Psic()
    items = list(reflections)
    idx = [i for i, r in enumerate(items) if r.use]
    refl = [items[i] for i in idx]
    free = _free_names(free)
    if set(offsets) - set(MOTORS):
        raise ValueError(f"offsets: unknown motors {sorted(set(offsets) - set(MOTORS))}; use {' '.join(MOTORS)}")
    offsets = tuple(m for m in MOTORS if m in set(offsets))
    H = np.array([r.hkl for r in refl], float).reshape(-1, 3)
    if free == "ub":
        if len(refl) < 3 or np.linalg.matrix_rank(H, tol=1e-6) < 3:
            raise ValueError("free UB needs at least three reflections whose HKL are not coplanar")
    elif len(refl) < 2:
        raise ValueError("need at least two reflections")
    n_lat = {"ub": 9, "scale": 1}.get(free, len(free) if not isinstance(free, str) else 0)
    n_par = (0 if free == "ub" else 3) + n_lat + len(offsets)
    if 3 * len(refl) < n_par:
        raise ValueError(f"{n_par} parameters need at least {math.ceil(n_par / 3)} reflections")
    lam = np.array([r.wavelength for r in refl])
    k = 2 * math.pi / lam

    def q_obs(off):
        out = []
        for r, wl in zip(refl, lam):
            a = dict(r.angles)
            for m, v in zip(offsets, off):
                a[m] += v
            out.append(geo.q_phi(a, wl))
        return np.array(out)

    def lattice_of(p):
        if free == "scale":
            s = p[0]
            return Lattice(lattice.a * s, lattice.b * s, lattice.c * s, lattice.alpha, lattice.beta, lattice.gamma,
                           lattice.name)
        vals = {n: getattr(lattice, n) for n in LATTICE_NAMES}
        vals.update(zip(free, p))
        return Lattice(**vals, name=lattice.name)

    if free == "ub":
        Q0 = q_obs(np.zeros(len(offsets)))
        ub0 = np.linalg.lstsq(H, Q0, rcond=None)[0].T
        p0 = np.concatenate([ub0.ravel(), np.zeros(len(offsets))])
        U0 = None

        def unpack(p):
            UBm = p[:9].reshape(3, 3)
            return UBm, p[9:]
    else:
        U0 = _initial_u(lattice, refl, geo, UB)
        lat0 = [1.0] if free == "scale" else [getattr(lattice, n) for n in free]
        p0 = np.concatenate([np.zeros(3), lat0, np.zeros(len(offsets))])

        def unpack(p):
            U = rotation_from_vector(p[:3]) @ U0
            return U @ lattice_of(p[3:3 + n_lat]).B, p[3 + n_lat:]

    def resid(p):
        UBm, off = unpack(p)
        return ((q_obs(off) - H @ UBm.T) / k[:, None]).ravel()      # in units of k: comparable across energies

    # rotation in radians, lengths in Å, angles and offsets in degrees: let the Jacobian set the scales
    res = least_squares(resid, p0, method="lm" if len(refl) * 3 >= len(p0) else "trf", x_scale="jac",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=5000)
    p = res.x
    UBm, off = unpack(p)
    if free == "ub":
        lat = lattice_from_ub(UBm)
        U = UBm @ np.linalg.inv(lat.B)
    else:
        lat = lattice_of(p[3:3 + n_lat]) if n_lat else lattice
        U = rotation_from_vector(p[:3]) @ U0

    # errors from the Jacobian, scaled by the residual
    r = res.fun
    dof = len(r) - len(p)
    warnings = []
    lat_err, off_err = {}, {}
    J = res.jac
    sv = np.linalg.svd(J, compute_uv=False) if J.size else np.array([1.0])
    if sv.size and sv[-1] < 1e-9 * sv[0]:
        warnings.append("some parameters are not determined by these reflections (add reflections with "
                        "different HKL directions or 2theta, or free fewer parameters)")
    if dof > 0:
        s2 = float(r @ r) / dof
        cov = s2 * np.linalg.pinv(J.T @ J)
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
        if free == "scale":
            for n in "abc":
                lat_err[n] = float(err[3] * getattr(lattice, n))
        elif free == "ub":
            lat_err = _lattice_errors_from_ub(UBm, cov[:9, :9])
        else:
            lat_err = {n: float(e) for n, e in zip(free, err[3:3 + n_lat])}
        off_err = {m: float(e) for m, e in zip(offsets, err[len(p) - len(offsets):])}
        sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        corr = cov / np.outer(sd, sd)
        names = (["rx", "ry", "rz"] if free != "ub" else [f"UB{i}{j}" for i in range(3) for j in range(3)])
        names += (["scale"] if free == "scale" else [] if free == "ub" else list(free)) + list(offsets)
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if abs(corr[i, j]) > 0.98 and (names[i] in offsets or names[j] in offsets
                                                or names[i] in LATTICE_NAMES or names[j] in LATTICE_NAMES):
                    warnings.append(f"{names[i]} and {names[j]} are strongly correlated ({corr[i, j]:+.2f})")
    elif dof == 0 and len(p) > 3:
        warnings.append("as many parameters as equations: no error estimate, every reflection fits exactly")
    if free == "ub":
        dev = float(np.max(np.abs(U.T @ U - np.eye(3))))
        if dev > 1e-3:
            warnings.append(f"U from the free UB is not a pure rotation (|U^T U - 1| = {dev:.1e}): the HKL or "
                            "the lattice may be inconsistent")

    # per-reflection residuals
    q = q_obs(off)
    qc = H @ UBm.T
    fits = []
    for j, i in enumerate(idx):
        dq = float(np.linalg.norm(q[j] - qc[j]))
        cosang = q[j] @ qc[j] / (np.linalg.norm(q[j]) * np.linalg.norm(qc[j]))
        fits.append(ReflectionFit(i, items[i].hkl, np.linalg.solve(UBm, q[j]), dq,
                                  math.degrees(math.acos(np.clip(cosang, -1, 1))),
                                  float(np.linalg.norm(q[j]) / np.linalg.norm(qc[j]) - 1)))
    dqs = np.array([f.dq for f in fits])
    rms_dq = float(np.sqrt(np.mean(dqs ** 2)))
    rms_ang = float(np.sqrt(np.mean([f.dangle ** 2 for f in fits])))
    if len(fits) > 3 and dof > 0:
        for j, f in enumerate(fits):
            others = np.delete(dqs, j)
            ref = max(float(np.sqrt(np.mean(others ** 2))), 1e-6)
            if f.dq > outlier_factor * ref and f.dangle > 0.02:
                f.outlier = True
        n_out = sum(f.outlier for f in fits)
        if n_out:
            warnings.append(f"{n_out} reflection(s) fit much worse than the rest: check their HKL, or switch them "
                            "off and refine again")
    return RefineResult(UBm, U, lat, lat_err, {m: float(v) for m, v in zip(offsets, off)}, off_err, fits, rms_dq,
                        rms_ang, free, len(refl), len(p), warnings)


def _lattice_errors_from_ub(UB, cov_ub, h=1e-7):
    """Lattice parameter errors from the UB covariance by numerical propagation."""
    base = lattice_from_ub(UB)
    vals0 = np.array([getattr(base, n) for n in LATTICE_NAMES])
    D = np.zeros((6, 9))
    flat = UB.ravel()
    for i in range(9):
        d = flat.copy()
        d[i] += h
        lat = lattice_from_ub(d.reshape(3, 3))
        D[:, i] = (np.array([getattr(lat, n) for n in LATTICE_NAMES]) - vals0) / h
    err = np.sqrt(np.clip(np.diag(D @ cov_ub @ D.T), 0, None))
    return {n: float(e) for n, e in zip(LATTICE_NAMES, err)}

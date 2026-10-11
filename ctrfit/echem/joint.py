"""Joint fit of several stationary CV scans (different HKL, same sample and potential program).

Every scan shares the physics, and each has its own intensity scale:

    I_scan(V) = scale_scan * I_model(HKL_scan; CUS fractions(V), near-surface change(V); structure)

Shared parameters: the composition path (E0 and width per transition, cathodic shift, coverage
theta), the near-surface lattice-O change (extra B at full extent, its own E0 / width or the first
transition's), and any numeric simulator setting as a structure parameter (e.g. oh_z, h2o_z, oh_dz,
h2o_dz, oh_B, relax_Obr, ...). ns_tl (whole trilayers) is profiled: one fit per value.

Speed: for one structure the intensity at an HKL is exactly a quadratic polynomial in
(c_H2O, c_OH, c_O, eps) with eps = exp(-dB q^2 / 16 pi^2) - 1 (the near-surface O Debye-Waller
change); 15 coefficients from ~30 evaluations of the full model per HKL, checked against direct
evaluation. A relaxed film top with H != 0 has its own q, and gets its own eps.
"""
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ..core.ctrmodel import CTRModel
from ..core.fast import FastCTR
from ..core.settings import ADSORBATE_SPECIES, DEFAULTS, SPECS, STUDY_KEYS
from . import analysis as an
from .predict import NearSurface, path_fractions

STRUCTURE_KEYS = [k for k in STUDY_KEYS if k not in ("energy_kev", "thickness_nm", "theta", "ns_B_O")]
DEFAULT_STRUCTURE = ("oh_z", "oh_dz", "h2o_z", "h2o_dz")


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
@dataclass
class JointScan:
    label: str
    hkl: tuple
    V: np.ndarray
    I: np.ndarray
    sigma: np.ndarray
    branch: np.ndarray
    scale0: float = None              # start value of the scale (counts per model intensity)
    use: bool = True

    @classmethod
    def from_session(cls, session, label=None):
        """The analysed points of a Session (after run()): chosen cycles, with a potential, sweeping."""
        r, o = session.res, session.opt
        if r.I_corr is None:
            raise ValueError("run the analysis of this scan first")
        cycles = an.parse_cycles(o.cycles, r.cycle)
        use = np.isin(r.cycle, cycles) & np.isfinite(r.V) & (r.branch != 0)
        if use.sum() < 5:
            raise ValueError("fewer than 5 usable points in this scan")
        if r.scale is None:
            try:
                session.simulate()
            except ValueError:
                pass
        h = session.hkl()
        return cls(label or f"#{session.opt.scan} ({h[0]:g} {h[1]:g} {h[2]:g})", tuple(h), r.V[use].copy(),
                   r.I_corr[use].copy(), r.sigma_corr[use].copy(), r.branch[use].astype(int).copy(), r.scale)

    def to_dict(self):
        return dict(label=self.label, hkl=list(self.hkl), V=self.V.tolist(), I=self.I.tolist(),
                    sigma=self.sigma.tolist(), branch=self.branch.tolist(), scale0=self.scale0, use=self.use)

    @classmethod
    def from_dict(cls, d):
        return cls(d["label"], tuple(d["hkl"]), np.array(d["V"]), np.array(d["I"]), np.array(d["sigma"]),
                   np.array(d["branch"], int), d.get("scale0"), d.get("use", True))


# ----------------------------------------------------------------------------------------------
# exact polynomial at one HKL
# ----------------------------------------------------------------------------------------------
class StructurePoint:
    """|F|^2 at one HKL for one structure as an exact quadratic in (CUS fractions, eps per region)."""

    def __init__(self, model, hkl, mode=None, B_probe=None):
        H, K, L = hkl
        self.H, self.K, self.L = int(round(H)), int(round(K)), float(L)
        self.model, self.mode = model, mode
        self.fast = FastCTR(self.H, self.K, [self.L])
        m = model
        qq = lambda a1: (2 * math.pi) ** 2 * ((self.H / a1) ** 2 + (self.K / m.A2) ** 2 + (self.L / m.C3) ** 2)  # noqa: E731
        self.Q = [qq(m.A1) / (16 * math.pi ** 2)]
        if m.relaxed and self.H != 0 and abs(m.a1_top - m.A1) > 1e-12:
            self.Q.append(qq(m.a1_top) / (16 * math.pi ** 2))
        self.nvar = 3 + len(self.Q)
        if B_probe is None:     # one region: 3 exponentials in dB; two regions: 6 (needs >= 6 values)
            B_probe = (0.0, 2.0, 6.0) if len(self.Q) == 1 else (0.0, 0.7, 1.6, 2.8, 4.5, 6.5, 9.0)
        self._base_B = m.ns_B_O
        rng = np.random.default_rng(12345)
        X, y = [], []
        comps = [np.zeros(3), *np.eye(3), *[0.5 * np.eye(3)[i] for i in range(3)],
                 *[0.5 * (np.eye(3)[i] + np.eye(3)[j]) for i, j in itertools.combinations(range(3), 2)]]
        for dB in B_probe:
            for c in comps:
                X.append(self._vars(c, dB)[0])
                y.append(self.direct(c, dB))
        A = self._features(np.array(X))
        self.coef = np.linalg.lstsq(A, np.array(y), rcond=None)[0]
        chk = [(rng.dirichlet(np.ones(4))[:3], float(rng.uniform(0, 8))) for _ in range(3)]
        yc = np.array([self.direct(c, b) for c, b in chk])
        pc = self._features(np.array([self._vars(c, b)[0] for c, b in chk])) @ self.coef
        self.check_error = float(np.max(np.abs(pc - yc) / np.maximum(np.abs(yc), 1e-12)))
        if self.check_error > 1e-6:
            raise RuntimeError(f"the quadratic form is not exact at ({H} {K} {L}) (error {self.check_error:.1e})")

    def _vars(self, c, dB):
        c = np.atleast_2d(c)
        dB = np.broadcast_to(np.asarray(dB, float), (len(c),))
        eps = [np.exp(-dB * q) - 1 for q in self.Q]
        return np.column_stack([c] + eps)

    def _features(self, X):
        X = np.atleast_2d(X)
        cols = [np.ones(len(X))] + [X[:, i] for i in range(self.nvar)]
        cols += [X[:, i] * X[:, j] for i, j in itertools.combinations_with_replacement(range(self.nvar), 2)]
        return np.column_stack(cols)

    def direct(self, c, dB=0.0):
        m = self.model
        m.ns_B_O = self._base_B + float(dB)
        try:
            comp = {s: float(x) for s, x in zip(ADSORBATE_SPECIES, c) if x > 0}
            return float(self.fast.intensity(m, comp, self.mode)[0])
        finally:
            m.ns_B_O = self._base_B

    def intensity(self, C, dB=0.0):
        return self._features(self._vars(C, dB)) @ self.coef


# ----------------------------------------------------------------------------------------------
# joint fit
# ----------------------------------------------------------------------------------------------
@dataclass
class JointParam:
    name: str
    value: float
    free: bool = False
    min: float = -np.inf
    max: float = np.inf
    unit: str = ""
    description: str = ""


@dataclass
class JointResult:
    values: dict
    errors: dict
    red_chi2: float
    chi2_per_scan: dict
    n_points: int
    n_free: int
    ns_tl: int
    notes: list = field(default_factory=list)
    tied: bool = False

    def summary(self):
        lines = [f"Joint fit of {len(self.chi2_per_scan)} scans, {self.n_points} points, {self.n_free} free parameters, "
                 f"near-surface region {self.ns_tl} TL: reduced chi2 {self.red_chi2:.2f}"]
        for k, v in self.values.items():
            if self.tied and k in ("ns_E0", "ns_width"):
                lines.append(f"  {k:24s} {'tied to transition 1':>12s}")
                continue
            e = self.errors.get(k)
            lines.append(f"  {k:24s} {v:12.5g}" + (f" ± {e:.3g}" if e is not None else "   (fixed)"))
        lines.append("Reduced chi2 per scan: " + ", ".join(f"{k} {v:.2f}" for k, v in self.chi2_per_scan.items()))
        return "\n".join(lines + ["Note: " + n for n in self.notes])


class JointFit:
    """See the module docstring. `scans`: list of JointScan; `sim_params`: simulator settings."""

    def __init__(self, scans, sim_params=None, states=("H2O", "OH"), E0=(0.35,), widths=(0.05,), theta=1.0,
                 cathodic_shift=0.0, ns_dB=0.0, ns_E0=None, ns_width=None, structure=DEFAULT_STRUCTURE, mode=None):
        self.scans = [s for s in scans]
        self.sim = dict(DEFAULTS, **(sim_params or {}))
        self.states = list(states)
        self.mode = mode
        self._cache = {}
        n = len(self.states) - 1
        if len(E0) != n or len(widths) != n:
            raise ValueError(f"{len(self.states)} states need {n} transitions")
        P = {}

        def add(name, value, free=False, lo=-np.inf, hi=np.inf, unit="", desc=""):
            P[name] = JointParam(name, float(value), free, lo, hi, unit, desc)
        for k in range(n):
            add(f"E0_{k + 1}", E0[k], True, -2.0, 3.0, "V", f"{self.states[k]} -> {self.states[k + 1]} potential")
            add(f"width_{k + 1}", widths[k], True, 1e-3, 1.0, "V", f"width of transition {k + 1}")
        add("cathodic_shift", cathodic_shift, False, -0.5, 0.5, "V", "transitions move by -shift on cathodic sweeps")
        add("theta", theta, False, 0.0, 1.0, "", "CUS coverage of the adsorbed states")
        add("ns_dB", ns_dB, False, 0.0, 20.0, "Å²", "near-surface lattice-O extra B factor at full extent")
        add("ns_E0", E0[0] if ns_E0 is None else ns_E0, False, -2.0, 3.0, "V",
            "where the near-surface change switches on")
        add("ns_width", widths[0] if ns_width is None else ns_width, False, 1e-3, 1.0, "V",
            "width of the near-surface change")
        self.ns_tied = ns_E0 is None and ns_width is None
        for key in structure:
            self.add_structure(key, P)
        for s in self.scans:
            add(f"scale:{s.label}", s.scale0 if s.scale0 else 1.0, True, 0.0, np.inf, "", f"intensity scale of {s.label}")
        self.params = P

    # ------------------------------------------------------------------ parameters
    def add_structure(self, key, P=None):
        P = self.params if P is None else P
        if key not in STRUCTURE_KEYS:
            raise KeyError(f"'{key}' is not a numeric structure setting")
        s = SPECS[key]
        P[key] = JointParam(key, float(self.sim[key]), False, s.get("min", -np.inf), s.get("max", np.inf),
                            s.get("unit", ""), s["label"])

    def set(self, name, value=None, free=None, lo=None, hi=None):
        p = self.params[name]
        if value is not None:
            p.value = float(value)
        if free is not None:
            p.free = bool(free)
        if lo is not None:
            p.min = float(lo)
        if hi is not None:
            p.max = float(hi)

    @property
    def structure_keys(self):
        return [k for k in self.params if k in STRUCTURE_KEYS]

    # ------------------------------------------------------------------ model
    def _points(self, values, ns_tl):
        key = (ns_tl,) + tuple(round(values[k], 12) for k in self.structure_keys)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            p = dict(self.sim, ns_tl=ns_tl, **{k: values[k] for k in self.structure_keys})
            model = CTRModel(p)
            self._cache[key] = {s.hkl: StructurePoint(model, s.hkl, self.mode) for s in self.scans if s.use}
        return self._cache[key]

    def curve(self, scan, values, V=None, branch=None, ns_tl=None):
        """Model intensity (scaled) for one scan at its points or at V / branch."""
        ns_tl = self.sim["ns_tl"] if ns_tl is None else ns_tl
        V = scan.V if V is None else np.asarray(V, float)
        branch = scan.branch if branch is None else branch
        n = len(self.states) - 1
        E0 = [values[f"E0_{k + 1}"] for k in range(n)]
        w = [values[f"width_{k + 1}"] for k in range(n)]
        sh, th = values["cathodic_shift"], values["theta"]
        C = path_fractions(V, self.states, E0, w, th, branch, sh)
        ns = NearSurface(values["ns_dB"], 0.0, None if self.ns_tied else values["ns_E0"],
                         None if self.ns_tied else values["ns_width"])
        dB = ns.dB * ns.extent(V, E0, w, branch, sh) if ns.dB else 0.0
        sp = self._points(values, ns_tl)[scan.hkl]
        return values[f"scale:{scan.label}"] * sp.intensity(C, dB)

    # ------------------------------------------------------------------ fitting
    def fit(self, ns_tl=None, max_nfev=400):
        ns_tl = int(self.sim["ns_tl"] if ns_tl is None else ns_tl)
        scans = [s for s in self.scans if s.use]
        if not scans:
            raise ValueError("no scans to fit")
        if (self.params["ns_dB"].free or self.params["ns_dB"].value) and ns_tl <= 0:
            raise ValueError("a near-surface change needs ns_tl > 0")
        if self.ns_tied:
            for k in ("ns_E0", "ns_width"):
                self.params[k].free = False
        names = [k for k, p in self.params.items() if p.free]
        if not names:
            raise ValueError("no free parameters")
        lo = np.array([self.params[k].min for k in names])
        hi = np.array([self.params[k].max for k in names])
        x0 = np.clip([self.params[k].value for k in names], lo + 1e-12, hi - 1e-12)
        base = {k: p.value for k, p in self.params.items()}

        def values_of(x):
            v = dict(base)
            v.update(zip(names, x))
            return v

        def resid(x):
            v = values_of(x)
            return np.concatenate([(self.curve(s, v, ns_tl=ns_tl) - s.I) / np.where(s.sigma > 0, s.sigma, 1.0)
                                   for s in scans])
        r = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", diff_step=1e-6, max_nfev=max_nfev)
        v = values_of(r.x)
        npts = sum(s.V.size for s in scans)
        dof = max(npts - len(names), 1)
        red = float(np.sum(r.fun ** 2) / dof)
        try:
            cov = np.linalg.pinv(r.jac.T @ r.jac) * max(red, 1.0)
            err = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            err = np.full(len(names), np.nan)
        per, i = {}, 0
        for s in scans:
            k = s.V.size
            per[s.label] = float(np.sum(r.fun[i:i + k] ** 2) / max(k, 1))
            i += k
        res = JointResult(v, dict(zip(names, map(float, err))), red, per, npts, len(names), ns_tl, tied=self.ns_tied)
        for k in names:
            p = self.params[k]
            if np.isfinite(p.min) and np.isfinite(p.max) and min(v[k] - p.min, p.max - v[k]) < 1e-3 * (p.max - p.min):
                res.notes.append(f"{k} is at its bound")
        if not r.success:
            res.notes.append(f"the optimiser stopped: {r.message}")
        if len(names) >= npts / 3:
            res.notes.append("many free parameters for the number of points: treat the values as a consistency check")
        self.result = res
        return res

    def apply(self, result=None):
        """Copy fitted values into the parameters (start values of the next fit)."""
        result = result or self.result
        for k, val in result.values.items():
            self.params[k].value = float(val)

    def profile_ns_tl(self, values=(1, 2, 3, 4, 5, 6)):
        """One fit per near-surface depth; returns [(ns_tl, JointResult)], the start values unchanged."""
        start = {k: p.value for k, p in self.params.items()}
        out = []
        for tl in values:
            for k, v in start.items():
                self.params[k].value = v
            out.append((int(tl), self.fit(ns_tl=int(tl))))
        for k, v in start.items():
            self.params[k].value = v
        return out

    # ------------------------------------------------------------------ io
    def to_dict(self):
        return dict(states=self.states, sim=self.sim, mode=self.mode, ns_tied=self.ns_tied,
                    params=[p.__dict__ for p in self.params.values()], scans=[s.to_dict() for s in self.scans])

    @classmethod
    def from_dict(cls, d):
        scans = [JointScan.from_dict(x) for x in d["scans"]]
        n = len(d["states"]) - 1
        j = cls(scans, d["sim"], d["states"], [0.0] * n, [0.05] * n, structure=(), mode=d.get("mode"))
        j.params = {}
        for p in d["params"]:
            j.params[p["name"]] = JointParam(**{k: (float(v) if k in ("value", "min", "max") else v)
                                                for k, v in p.items()})
        j.ns_tied = d.get("ns_tied", True)
        return j

    def save(self, path):
        def enc(o):
            return None if isinstance(o, float) and not np.isfinite(o) else o
        d = self.to_dict()
        for p in d["params"]:
            p["min"], p["max"] = enc(p["min"]), enc(p["max"])
        Path(path).write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        for p in d["params"]:
            p["min"] = -np.inf if p["min"] is None else p["min"]
            p["max"] = np.inf if p["max"] is None else p["max"]
        return cls.from_dict(d)


__all__ = ["JointScan", "StructurePoint", "JointParam", "JointResult", "JointFit", "STRUCTURE_KEYS",
           "DEFAULT_STRUCTURE"]

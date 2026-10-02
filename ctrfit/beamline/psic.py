"""SPEC psic (4S+2D) diffractometer geometry, after H. You, J. Appl. Cryst. 32, 614 (1999).

Laboratory frame: y along the incident beam, x vertical (up), z = x cross y (horizontal).
Sample circles: mu (about x), eta (about -z), chi (about y), phi (about -z), stacked
mu -> eta -> chi -> phi. Detector circles: nu (about x) carrying delta (about -z).

    Q_lab = (NU DELTA - I) k_in,     k_in = (0, k, 0),   k = 2 pi / lambda
    Q_lab = MU ETA CHI PHI U B h      (h = (H, K, L), B with 2 pi)

Angles are in degrees and ordered as in SPEC psic: del, eta, chi, phi, nu, mu. A beamline whose
motors turn the other way can give per-motor signs (Psic(signs={"eta": -1})).
"""
import math
from dataclasses import dataclass, field

import numpy as np

HC = 12.398419843       # keV * Angstrom
MOTORS = ("del", "eta", "chi", "phi", "nu", "mu")


def energy_to_wavelength(e_kev):
    return HC / e_kev


def wavelength_to_energy(lam):
    return HC / lam


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ----------------------------------------------------------------------------------------------
# lattice
# ----------------------------------------------------------------------------------------------
@dataclass
class Lattice:
    """Direct lattice (Å, degrees). B follows Busing and Levy with the 2 pi convention."""
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    name: str = "custom"

    def __post_init__(self):
        al, be, ga = (math.radians(x) for x in (self.alpha, self.beta, self.gamma))
        a, b, c = self.a, self.b, self.c
        vol = a * b * c * math.sqrt(1 - math.cos(al) ** 2 - math.cos(be) ** 2 - math.cos(ga) ** 2
                                    + 2 * math.cos(al) * math.cos(be) * math.cos(ga))
        self.volume = vol
        a_s = 2 * math.pi * b * c * math.sin(al) / vol
        b_s = 2 * math.pi * a * c * math.sin(be) / vol
        c_s = 2 * math.pi * a * b * math.sin(ga) / vol
        al_s = math.acos((math.cos(be) * math.cos(ga) - math.cos(al)) / (math.sin(be) * math.sin(ga)))
        be_s = math.acos((math.cos(al) * math.cos(ga) - math.cos(be)) / (math.sin(al) * math.sin(ga)))
        ga_s = math.acos((math.cos(al) * math.cos(be) - math.cos(ga)) / (math.sin(al) * math.sin(be)))
        self.B = np.array([[a_s, b_s * math.cos(ga_s), c_s * math.cos(be_s)],
                           [0, b_s * math.sin(ga_s), -c_s * math.sin(be_s) * math.cos(al)],
                           [0, 0, 2 * math.pi / c]])

    def q(self, hkl):
        """|Q| (1/Å) of one reflection or an (N, 3) array."""
        return np.linalg.norm(np.atleast_2d(hkl) @ self.B.T, axis=1)

    def d(self, hkl):
        return 2 * math.pi / self.q(hkl)

    def to_dict(self):
        return dict(a=self.a, b=self.b, c=self.c, alpha=self.alpha, beta=self.beta, gamma=self.gamma, name=self.name)


def tio2_surface_lattice(a=4.5937, c=2.9587):
    """TiO2(110) surface cell of this project: a1 = c (H, [001]), a2 = a3 = sqrt(2) a (K, [1-10]; L, [110])."""
    return Lattice(c, math.sqrt(2) * a, math.sqrt(2) * a, name="TiO2(110) surface cell")


def tio2_conventional_lattice(a=4.5937, c=2.9587):
    return Lattice(a, a, c, name="TiO2 rutile (conventional)")


LATTICES = {
    "tio2_surface": tio2_surface_lattice,
    "tio2": tio2_conventional_lattice,
    "ruo2_surface": lambda: Lattice(3.1066, math.sqrt(2) * 4.4919, math.sqrt(2) * 4.4919,
                                    name="RuO2(110) surface cell (bulk)"),
}


# ----------------------------------------------------------------------------------------------
# diffractometer
# ----------------------------------------------------------------------------------------------
@dataclass
class Psic:
    """psic geometry with optional per-motor signs."""
    signs: dict = field(default_factory=dict)

    def _a(self, angles, name):
        return math.radians(self.signs.get(name, 1) * float(angles[name]))

    def sample_matrix(self, angles):
        """Z = MU ETA CHI PHI (phi frame -> lab)."""
        return (rot_x(self._a(angles, "mu")) @ rot_z(-self._a(angles, "eta")) @ rot_y(self._a(angles, "chi"))
                @ rot_z(-self._a(angles, "phi")))

    def detector_matrix(self, angles):
        """NU DELTA (rotates the incident direction onto the scattered direction)."""
        return rot_x(self._a(angles, "nu")) @ rot_z(-self._a(angles, "del"))

    def k_in(self, wavelength):
        return np.array([0.0, 2 * math.pi / wavelength, 0.0])

    def k_out(self, angles, wavelength):
        return self.detector_matrix(angles) @ self.k_in(wavelength)

    def q_lab(self, angles, wavelength):
        return self.k_out(angles, wavelength) - self.k_in(wavelength)

    def q_phi(self, angles, wavelength):
        """Q in the phi-axis frame (where Q = U B h)."""
        return self.sample_matrix(angles).T @ self.q_lab(angles, wavelength)

    def two_theta(self, angles):
        """Scattering angle: cos(2 theta) = cos(delta) cos(nu)."""
        return math.degrees(math.acos(np.clip(math.cos(self._a(angles, "del")) * math.cos(self._a(angles, "nu")),
                                              -1, 1)))

    def hkl(self, angles, UB, wavelength):
        return np.linalg.solve(UB, self.q_phi(angles, wavelength))

    def surface_angles(self, angles, normal_phi, wavelength):
        """(alpha_in, beta_out) in degrees: incidence and exit angles relative to the surface whose
        normal (phi frame) is normal_phi."""
        n = self.sample_matrix(angles) @ (np.asarray(normal_phi, float) / np.linalg.norm(normal_phi))
        ki = self.k_in(wavelength) / (2 * math.pi / wavelength)
        ko = self.k_out(angles, wavelength) / (2 * math.pi / wavelength)
        return math.degrees(math.asin(np.clip(-ki @ n, -1, 1))), math.degrees(math.asin(np.clip(ko @ n, -1, 1)))


def as_angles(values):
    """dict from a dict or a sequence in SPEC psic order (del, eta, chi, phi, nu, mu)."""
    if isinstance(values, dict):
        return {m: float(values.get(m, 0.0)) for m in MOTORS}
    vals = list(values)
    if len(vals) != 6:
        raise ValueError("give six angles: del eta chi phi nu mu")
    return dict(zip(MOTORS, map(float, vals)))


# ----------------------------------------------------------------------------------------------
# orientation
# ----------------------------------------------------------------------------------------------
def _triad(v1, v2):
    t1 = v1 / np.linalg.norm(v1)
    t3 = np.cross(v1, v2)
    t3 /= np.linalg.norm(t3)
    return np.column_stack([t1, np.cross(t3, t1), t3])


def u_from_two_vectors(c1, c2, o1, o2):
    """Rotation U with U c1 || o1 and U c2 in the (o1, o2) plane (Busing and Levy)."""
    return _triad(o1, o2) @ _triad(c1, c2).T


def ub_from_two_reflections(lattice, h1, angles1, h2, angles2, wavelength, geo=None):
    """UB from two reflections with known HKL (like SPEC or0 / or1)."""
    geo = geo or Psic()
    U = u_from_two_vectors(lattice.B @ np.asarray(h1, float), lattice.B @ np.asarray(h2, float),
                           geo.q_phi(as_angles(angles1), wavelength), geo.q_phi(as_angles(angles2), wavelength))
    return U @ lattice.B


def rotation_from_vector(r):
    """Rotation matrix from a rotation vector (axis * angle in radians)."""
    th = float(np.linalg.norm(r))
    if th < 1e-15:
        return np.eye(3)
    k = np.asarray(r) / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


# ----------------------------------------------------------------------------------------------
# HKL -> angles
# ----------------------------------------------------------------------------------------------
@dataclass
class Mode:
    """Which motors move to reach a reflection; the others stay at `fixed` values.

    constraint: None, ("alpha", deg) fixed incidence, ("beta", deg) fixed exit, or ("alpha=beta",)
    symmetric. len(free) must be 3 plus one per constraint. Needs a surface normal for constraints.
    """
    free: tuple
    constraint: tuple = None
    name: str = ""

    @property
    def n_equations(self):
        return 3 + (1 if self.constraint else 0)


MODES = {
    # name: (mode, motors that must be set in `fixed`); surface modes need the surface normal
    "horizontal surface, fixed alpha": (Mode(("del", "nu", "phi", "eta"), ("alpha", 0.5), "eta sets alpha"),
                                        dict(chi=90.0, mu=0.0)),
    "vertical surface, fixed alpha": (Mode(("del", "nu", "phi", "mu"), ("alpha", 0.5), "mu sets alpha"),
                                      dict(eta=0.0, chi=0.0)),
    "horizontal surface, alpha = beta": (Mode(("del", "nu", "phi", "eta"), ("alpha=beta",)), dict(chi=90.0, mu=0.0)),
    "vertical surface, alpha = beta": (Mode(("del", "nu", "phi", "mu"), ("alpha=beta",)), dict(eta=0.0, chi=0.0)),
    "four-circle vertical (del, eta, chi)": (Mode(("del", "eta", "chi"), None), dict(nu=0.0, mu=0.0)),
}
DEFAULT_LIMITS = {"del": (-5.0, 120.0), "nu": (-5.0, 120.0)}


def mode_with_alpha(name, alpha):
    """A copy of a MODES entry with the fixed incidence (or exit) angle set."""
    mode, fixed = MODES[name]
    c = mode.constraint
    if c and c[0] in ("alpha", "beta"):
        mode = Mode(mode.free, (c[0], float(alpha)), mode.name)
    return mode, dict(fixed)


def angles_for_hkl(hkl, UB, wavelength, mode, fixed=None, normal_phi=None, geo=None, limits=None,
                   current=None, n_starts=3, max_solutions=8):
    """All solutions (list of dicts of the six angles, plus alpha, beta, two_theta) for one
    reflection in `mode`, sorted by distance to `current` (or to `fixed`). Motors not free keep
    the values in `fixed` (default 0). limits: {motor: (lo, hi)} in degrees."""
    from scipy.optimize import least_squares
    geo = geo or Psic()
    fixed = as_angles(fixed or {})
    if len(mode.free) != mode.n_equations:
        raise ValueError(f"mode needs {mode.n_equations} free motors, got {len(mode.free)}")
    if mode.constraint and normal_phi is None:
        raise ValueError("this mode constrains alpha / beta: give the surface normal (phi frame)")
    q_target = UB @ np.asarray(hkl, float)
    k = 2 * math.pi / wavelength
    if np.linalg.norm(q_target) > 2 * k:
        return []
    limits = limits or {}

    def full(x):
        a = dict(fixed)
        a.update(zip(mode.free, x))
        return a

    def resid(x):
        a = full(x)
        r = list((geo.sample_matrix(a) @ q_target - geo.q_lab(a, wavelength)) / k)
        if mode.constraint:
            al, be = geo.surface_angles(a, normal_phi, wavelength)
            c = mode.constraint
            if c[0] == "alpha":
                r.append(math.radians(al - c[1]))
            elif c[0] == "beta":
                r.append(math.radians(be - c[1]))
            else:
                r.append(math.radians(al - be))
        return np.array(r)

    grid = np.linspace(-120, 120, n_starts)
    starts = np.array(np.meshgrid(*[grid] * len(mode.free))).reshape(len(mode.free), -1).T
    if current is not None:
        starts = np.vstack([[as_angles(current)[m] for m in mode.free], starts])
    sols = []
    for x0 in starts:
        res = least_squares(resid, x0, method="lm", xtol=1e-13, ftol=1e-13, max_nfev=400)
        if not res.success or np.max(np.abs(res.fun)) > 1e-8:
            continue
        x = (res.x + 180) % 360 - 180
        a = full(x)
        if any(not (lo <= a[m] <= hi) for m, (lo, hi) in limits.items()):
            continue
        if any(np.allclose([a[m] for m in MOTORS], [s[m] for m in MOTORS], atol=1e-4) for s in sols):
            continue
        out = dict(a)
        out["two_theta"] = geo.two_theta(a)
        if normal_phi is not None:
            out["alpha"], out["beta"] = geo.surface_angles(a, normal_phi, wavelength)
        sols.append(out)
        if len(sols) >= max_solutions:
            break
    ref = as_angles(current) if current is not None else fixed
    sols.sort(key=lambda s: sum(min(abs(s[m] - ref[m]) % 360, 360 - abs(s[m] - ref[m]) % 360) for m in MOTORS))
    return sols


def parse_angle_lines(text):
    """Angle sets from text: one peak per line, either six numbers in psic order
    (del eta chi phi nu mu) or name=value pairs (e.g. 'del=20.1 eta=10.05 chi=90 phi=12 nu=0 mu=0.5').
    Blank lines and lines starting with # are skipped; a trailing '# comment' is kept as the label."""
    import re
    out = []
    for raw in text.splitlines():
        line, _, label = raw.partition("#")
        if not line.strip():
            continue
        pairs = re.findall(r"([A-Za-z]+)\s*[=:]\s*(-?[\d.eE+-]+)", line)
        if pairs:
            names = {n.lower(): float(v) for n, v in pairs}
            alias = {"delta": "del", "theta": "eta", "th": "eta", "gamma": "nu", "tth": "del"}
            names = {alias.get(n, n): v for n, v in names.items()}
            unknown = set(names) - set(MOTORS)
            if unknown:
                raise ValueError(f"unknown motor names {sorted(unknown)} in: {raw.strip()}")
            a = as_angles(names)
        else:
            nums = [float(x) for x in re.split(r"[\s,;]+", line.strip()) if x]
            a = as_angles(nums)
        a["label"] = label.strip()
        out.append(a)
    return out

"""Fast |F| for all points of a dataset in one call.

    ev = model.evaluator(H, K, L)       # compile once for these points
    F = ev()                            # current parameter values
    F = ev(values)                      # or a dict from model.params.view(scope)

Compiling resolves the anomalous terms once and sets up the point arrays. Each call rebuilds the
(cheap) engine object from the parameter values; the expensive parts (substrate rod, buried film
sums, phase matrices, adsorbate structure factors) are cached by content in ctrfit.core.fast and
recomputed only when the values they depend on change. With only surface parameters free, the
substrate and the buried film are therefore computed once.

The absorption in the electrolyte and window (t_liquid, t_window) multiplies |F| per point through the
exit angle, precomputed for the points. Per-rod L offsets (dL_H_K, when present and not 0) move the
points, and an L resolution (res_L > 0) averages |F|^2 over Gauss-Hermite nodes around each point: the
moved / spread points get their own cached engine (recomputed whenever an offset or res_L changes; with
res_L on, each evaluation costs RES_NODES times as much).
"""
import numpy as np

from ..core.absorption import path_factor, transmission_F
from ..core.fast import FastCTR
from ..core.film import FilmCTRModel
from ..core.sf import resolve_anomalous
from .model import RES_NODES, resolution_nodes


class Evaluator:
    def __init__(self, model, H, K, L, free=None, scope=""):
        self.model, self.scope = model, scope
        self.fast = FastCTR(H, K, L)
        self.pts = self.fast.pts
        self.free = list(model.params.free_names if free is None else free)
        notes = []
        self.anomalous = resolve_anomalous(model.settings, notes)
        self.notes = notes
        rods = sorted(set(zip(self.pts.H.tolist(), self.pts.K.tolist())))
        self.rods = rods
        self.rod_index = np.array([rods.index(hk) for hk in zip(self.pts.H.tolist(), self.pts.K.tolist())])
        self.n_calls = 0
        s = model.settings
        from .model import lattice
        self.energy = s["energy_kev"]
        self.path = path_factor(self.pts.H, self.pts.K, self.pts.L, self.energy, lattice(s), s["alpha_in_deg"])
        self._shifted = None                 # (offsets, res_L, FastCTR at the shifted / spread points)

    @property
    def n(self):
        return self.pts.n

    def values(self, values=None):
        return self.model.params.view(self.scope) if values is None else values

    def engine(self, values=None):
        v = self.values(values)
        s, geometry, comp = self.model.template.engine_inputs(self.model.settings, v)
        return FilmCTRModel(s, geometry, anomalous=self.anomalous), comp

    def _fast_for(self, v):
        """The engine at the points, moved by the rods' L offsets and spread over the resolution nodes when
        res_L > 0 (the plain one when all offsets are 0 and the resolution is off)."""
        off = np.array([float(v.get(f"dL_{H}_{K}", 0.0)) for H, K in self.rods])
        res = max(float(v.get("res_L", 0.0)), 0.0)
        if not off.any() and not res:
            return self.fast
        if self._shifted is None or not np.array_equal(self._shifted[0], off) or self._shifted[1] != res:
            L = self.pts.L + off[self.rod_index]
            H, K = self.pts.H, self.pts.K
            if res:
                x, _ = resolution_nodes()
                L = (L[None, :] + res * x[:, None]).ravel()
                H, K = np.tile(H, RES_NODES), np.tile(K, RES_NODES)
            self._shifted = (off, res, FastCTR(H, K, L))
        return self._shifted[2]

    def intensity(self, values=None):
        """|F|^2 without scale factors (at L + the rod's offset, averaged over the L resolution)."""
        v = self.values(values)
        m, comp = self.engine(v)
        self.n_calls += 1
        I = self._fast_for(v).intensity(m, comp)
        if I.size != self.pts.n:                     # the resolution nodes: their weighted mean per point
            _, w = resolution_nodes()
            I = w @ I.reshape(RES_NODES, self.pts.n)
        return I

    def scales(self, values=None):
        """scale x rod scale x absorption in the electrolyte and window, per point."""
        v = self.values(values)
        rod = np.array([v.get(f"rodscale_{H}_{K}", 1.0) for H, K in self.rods])
        return v["scale"] * rod[self.rod_index] * transmission_F(self.path, self.energy, v.get("t_liquid", 0.0),
                                                                   v.get("t_window", 0.0))

    def __call__(self, values=None):
        """Model |F| at every point, including the dataset and rod scales."""
        v = self.values(values)
        return self.scales(v) * np.sqrt(self.intensity(v))

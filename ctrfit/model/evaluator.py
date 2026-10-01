"""Fast |F| for all points of a dataset in one call.

    ev = model.evaluator(H, K, L)       # compile once for these points
    F = ev()                            # current parameter values
    F = ev(values)                      # or a dict from model.params.view(scope)

Compiling resolves the anomalous terms once and sets up the point arrays. Each call rebuilds the
(cheap) engine object from the parameter values; the expensive parts (substrate rod, buried film
sums, phase matrices, adsorbate structure factors) are cached by content in ctrfit.core.fast and
recomputed only when the values they depend on change. With only surface parameters free, the
substrate and the buried film are therefore computed once.
"""
import numpy as np

from ..core.fast import FastCTR
from ..core.film import FilmCTRModel
from ..core.sf import resolve_anomalous


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

    @property
    def n(self):
        return self.pts.n

    def values(self, values=None):
        return self.model.params.view(self.scope) if values is None else values

    def engine(self, values=None):
        v = self.values(values)
        s, geometry, comp = self.model.template.engine_inputs(self.model.settings, v)
        return FilmCTRModel(s, geometry, anomalous=self.anomalous), comp

    def intensity(self, values=None):
        """|F|^2 without scale factors."""
        m, comp = self.engine(values)
        self.n_calls += 1
        return self.fast.intensity(m, comp)

    def scales(self, values=None):
        v = self.values(values)
        rod = np.array([v.get(f"rodscale_{H}_{K}", 1.0) for H, K in self.rods])
        return v["scale"] * rod[self.rod_index]

    def __call__(self, values=None):
        """Model |F| at every point, including the dataset and rod scales."""
        v = self.values(values)
        return self.scales(v) * np.sqrt(self.intensity(v))

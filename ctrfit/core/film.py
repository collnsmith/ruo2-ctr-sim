"""Film thickness distribution with a continuous mean, and the engine driven by fit parameters.

thickness_weights() is the only physics change of the fitting model against ctr_engine: the centre
of the Gaussian thickness distribution is continuous instead of a whole trilayer count.
"""
import math

import numpy as np

from .ctrmodel import CTRModel

MIN_SPREAD_TL = 0.3


def _taper(d, a, b):
    """1 for d <= a, 0 for d >= b, smooth (cosine) in between."""
    t = np.clip((d - a) / (b - a), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * t))


def thickness_weights(mean_tl, spread_tl, min_spread=MIN_SPREAD_TL):
    """[(n, weight)] over whole trilayer counts n >= 1 for a continuous mean thickness.

    Gaussian weights exp(-(n - mean)^2 / (2 s^2)) with s = max(spread, min_spread), windowed by a
    cosine taper between |n - mean| = 2.5 s + 1 and 2.5 s + 1.5, so the weights change smoothly with
    the mean and the spread. For a whole-number mean, the trilayer counts kept by the legacy window
    (floor(mean - 2.5 s) to ceil(mean + 2.5 s)) get taper 1 and every other count taper 0 whenever
    ceil(2.5 s) - 2.5 s >= 0.5; then the weights equal CTRModel.film_n_weights() exactly. This holds
    for the default spread (0.32 nm, about 0.99 trilayers).
    """
    s = max(float(spread_tl), min_spread)
    a, b = 2.5 * s + 1.0, 2.5 * s + 1.5
    lo = max(1, int(math.floor(mean_tl - b)))
    hi = max(lo, int(math.ceil(mean_tl + b)))
    ns = np.arange(lo, hi + 1)
    d = np.abs(ns - mean_tl)
    w = np.exp(-0.5 * (d / s) ** 2) * _taper(d, a, b)
    keep = w > 0
    if not keep.any():                     # mean far below 1: all weight on one trilayer
        return [(1, 1.0)]
    ns, w = ns[keep], w[keep]
    return list(zip(ns, w / w.sum()))


class FilmCTRModel(CTRModel):
    """CTRModel whose film geometry comes from fit parameters (in trilayers) instead of nm settings.

    geometry keys (all optional): mean_tl, spread_tl, rough_tl, coherent_tl, extra_layers (list).
    """

    def __init__(self, params, geometry=None, min_spread=MIN_SPREAD_TL):
        super().__init__(params)
        g = dict(geometry or {})
        self.min_spread = min_spread
        self.n_mean = float(g.get("mean_tl", self.n_film))
        self.n_film = max(1, int(round(self.n_mean)))
        if "spread_tl" in g:
            self.film_n_spread = float(g["spread_tl"])
        if "rough_tl" in g:
            self.sigma_rough = float(g["rough_tl"])
        if "coherent_tl" in g:
            self.n_coherent = max(0, int(round(g["coherent_tl"])))
        if "extra_layers" in g:
            self.extra_layers = list(g["extra_layers"])
        self.relaxed = self.relax_001 > 0 and self.n_film > self.n_coherent

    def film_n_weights(self):
        return thickness_weights(self.n_mean, self.film_n_spread, self.min_spread)

"""OCV vs 0 V (and 0.7 V vs 0 V) per rod and film: shift in L of the fringe pattern (cross-correlation of detrended ln F),
the fringe period, and the Bragg-peak position, to tell a film change (fringes move, Bragg fixed) from realignment (all move)."""
import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
sys.modules["xraydb"] = None
import numpy as np
from ctrfit.project.project import Project
from ctrfit.fit.global_fit import film_of
p = Project.load(sys.argv[1])
rods = {}
for ds in p.datasets:
    v = ds.meta.get("potential_V")
    rods[(film_of(ds).split("|")[0], ds.rod_labels[0], "OCV" if v is None else f"{v:g} V")] = ds
def detr(ds, Lg):
    o = np.argsort(ds.L); y = np.log(np.interp(Lg, ds.L[o], ds.F[o]))
    return y - np.polyval(np.polyfit(Lg, y, 4), Lg)
for (film, rod, k), ds in sorted(rods.items()):
    if k not in ("OCV", "0.7 V") or (film, rod, "0 V") not in rods: continue
    ref = rods[(film, rod, "0 V")]
    lo, hi = max(ds.L.min(), ref.L.min()) + 0.05, min(ds.L.max(), ref.L.max()) - 0.05
    if hi - lo < 0.4: continue
    Lg = np.linspace(lo, hi, 400); dL = Lg[1] - Lg[0]
    # leave out the substrate Bragg peaks (+-0.12) from the pattern
    from ctrfit.core.ctrmodel import substrate_bragg_L
    H, K = int(ds.H[0]), int(ds.K[0])
    keep = np.ones(Lg.size, bool)
    for Lb in substrate_bragg_L(H, K, 0, hi + 1):
        keep &= np.abs(Lg - Lb) > 0.12
    a, b = detr(ds, Lg) * keep, detr(ref, Lg) * keep
    lags = np.arange(-60, 61)
    cc = [np.sum(a[max(0, -l):Lg.size - max(0, l)] * b[max(0, l):Lg.size - max(0, -l)]) /
          np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) for l in lags]
    i = int(np.argmax(cc))
    print(f"{film:6s} ({rod:5s}) {k:5s} vs 0 V: L {lo:.2f}-{hi:.2f}, fringe shift {lags[i] * dL:+.3f} L (corr {cc[i]:.2f}, at 0 {cc[60]:.2f}), "
          f"rms of ln-ratio {np.std((a - b)[keep]):.2f}")

"""Whole-rod comparison: each condition's OH rod against the film's 0 V rod (interpolated), one free scale, points within
0.08 of the TiO2 Bragg peak left out. chi2/pt with sigma_eff, and the same after a free linear tilt in L (absorption /
footprint), plus the rms of ln(ratio). Also with each rod's sigma inflated to the scatter of its own smooth fit."""
import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
sys.modules["xraydb"] = None
import numpy as np
from ctrfit.project.project import Project
from ctrfit.fit.global_fit import film_of
from ctrfit.core.ctrmodel import substrate_bragg_L
p = Project.load(sys.argv[1])
rods = {}
for ds in p.datasets:
    v = ds.meta.get("potential_V")
    rods[(film_of(ds).split("|")[0], ds.rod_labels[0], "OCV" if v is None else f"{v:g} V")] = ds
def scatter(ds):                      # point scatter about a local quadratic (11 points): the real repeatability
    o = np.argsort(ds.L); L, y = ds.L[o], np.log(ds.F[o]); r = []
    for i in range(5, L.size - 5):
        c = np.polyfit(L[i - 5:i + 6] - L[i], y[i - 5:i + 6], 2); r.append(y[i] - c[-1])
    return np.std(r) * np.sqrt(11 / 8)
for film, rod in (("4.1 nm", "0 -1"), ("7 nm", "0 1"), ("4.1 nm", "-1 0"), ("4.1 nm", "0 2"), ("7 nm", "0 2"), ("7 nm", "2 0")):
    ref = rods.get((film, rod, "0 V"))
    if ref is None: continue
    o = np.argsort(ref.L)
    H, K = int(ref.H[0]), int(ref.K[0])
    print(f"{film} ({rod}): scatter of ln F about a smooth curve, 0 V rod {scatter(ref):.3f}")
    for k in ("OCV", "-0.238 V", "-0.1 V", "0.15 V", "0.35 V", "0.7 V"):
        d = rods.get((film, rod, k))
        if d is None: continue
        L = d.L
        keep = (L > ref.L.min() + 0.01) & (L < ref.L.max() - 0.01)
        for Lb in substrate_bragg_L(H, K, 0, L.max() + 1):
            keep &= np.abs(L - Lb) > 0.08
        L = L[keep]
        y = np.log(d.F[keep]) - np.log(np.interp(L, ref.L[o], ref.F[o]))
        s = np.hypot(d.sigma_eff[keep] / d.F[keep], np.interp(L, ref.L[o], (ref.sigma_eff / ref.F)[o]))
        sr = np.hypot(scatter(d), scatter(ref))
        r0 = y - np.average(y, weights=s ** -2)
        A = np.column_stack([np.ones(L.size), L]); c, *_ = np.linalg.lstsq(A / s[:, None], y / s, rcond=None); r1 = y - A @ c
        print(f"   {k:8s} vs 0 V: {L.size:3d} pts, rms ln-ratio {np.std(r0):.3f} (tilt removed {np.std(r1):.3f}); "
              f"chi2/pt with sigma_eff {np.mean((r0 / s) ** 2):6.2f} (tilt removed {np.mean((r1 / s) ** 2):6.2f}); "
              f"with the scatter as sigma {np.mean((r1 / sr) ** 2):5.2f}")

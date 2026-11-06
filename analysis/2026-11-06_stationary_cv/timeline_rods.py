"""OH rods split by scan (logbook order), compared pairwise: same-potential repeats (true repeatability on the OH rod)
and pairs measured back to back at different potentials. rms of ln(I_a / I_b) over the overlap, one free scale, points
within 0.08 of the TiO2 Bragg peak left out; the other rod interpolated onto each point's L."""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))
sys.modules["xraydb"] = None
import numpy as np
from ctrfit.project.project import Project
from ctrfit.fit.global_fit import film_of
from ctrfit.core.ctrmodel import substrate_bragg_L
p = Project.load(os.path.join(here, "..", "2026-11-05_oh_fits", "oh_rods_fit.ctrproj.json"))
parts = {}
for ds in p.datasets:
    f = film_of(ds).split("|")[0]
    if ds.rod_labels[0] not in ("0 1", "0 -1"): continue
    v = ds.meta.get("potential_V"); k = "OCV" if v is None else f"{v:g} V"
    sc = np.asarray(ds.scan)
    for s in np.unique(sc):
        m = sc == s
        parts[(f, k, str(s))] = (ds.L[m], ds.F[m], ds.sigma_eff[m], int(ds.H[0]), int(ds.K[0]))
for key in sorted(parts): print("  ", key, f"L {parts[key][0].min():.2f}-{parts[key][0].max():.2f}, {parts[key][0].size} pts")
def rms(a, b):
    La, Fa, sa, H, K = parts[a]; Lb, Fb, sb, _, _ = parts[b]
    keep = (La > Lb.min() + 0.005) & (La < Lb.max() - 0.005)
    for Lbr in substrate_bragg_L(H, K, 0, 5):
        keep &= np.abs(La - Lbr) > 0.08
    o = np.argsort(Lb)
    y = np.log(Fa[keep]) - np.log(np.interp(La[keep], Lb[o], Fb[o]))
    return float(np.std(y - y.mean())), int(keep.sum())
import json
json.dump({"|".join(k): dict(L=[float(parts[k][0].min()), float(parts[k][0].max())], n=int(parts[k][0].size)) for k in parts},
          open(os.path.join(here, "timeline_rods_parts.json"), "w"), indent=1)

# split the 7 nm -0.1 V and 0.7 V files into their two measurements by L grid (0.01 steps: scans 325-326 / 333-334;
# 0.0188 steps: the later repeats 337-340)
for k in [k for k in list(parts) if k[0] == "7 nm" and k[1] in ("-0.1 V", "0.7 V")]:
    L, F, s, H, K = parts.pop(k)
    on = np.abs(L * 100 - np.round(L * 100)) < 1e-4
    parts[(k[0], k[1], "a")] = (L[on], F[on], s[on], H, K)
    parts[(k[0], k[1], "b")] = (L[~on], F[~on], s[~on], H, K)
g = lambda f, k, t=None: next(x for x in parts if x[0] == f and x[1] == k and (t is None or x[2] == t))
print("\n7 nm, in logbook order: OCV (317-320), 0.7 (325-326), 0.35 (327-328), 0.15 (329-330), 0 (331-332), -0.1 (333-334), -0.1 (337-338), 0.7 (339-340)")
seq7 = [("OCV", None), ("0.7 V", "a"), ("0.35 V", None), ("0.15 V", None), ("0 V", None), ("-0.1 V", "a"), ("-0.1 V", "b"), ("0.7 V", "b")]
out = {}
for (k1, t1), (k2, t2) in zip(seq7[:-1], seq7[1:]):
    r, n = rms(g("7 nm", k1, t1), g("7 nm", k2, t2)); out[f"7 nm {k1}{t1 or ''} -> {k2}{t2 or ''}"] = r
    print(f"   back to back {k1:7s}{t1 or ' '} -> {k2:7s}{t2 or ' '}: rms {r:.3f} ({n} pts)")
for k in ("0.7 V", "-0.1 V"):
    r, n = rms(g("7 nm", k, "b"), g("7 nm", k, "a")); out[f"7 nm repeat {k}"] = r
    print(f"   repeat at {k:7s} (hours apart)      : rms {r:.3f} ({n} pts)")
r, n = rms(g("7 nm", "0.7 V", "b"), g("7 nm", "-0.1 V", "b")); print(f"   0.7 (339-340) vs -0.1 (337-338), same L grid: rms {r:.3f} ({n} pts)")
print("\n4.1 nm, in logbook order: 0 (483-484), -0.1 (485-486), 0.15 (487-488), 0.7 (489-490), -0.238 (491-492), 0.35 (493-494); fresh spot; CVs; OCV (504-505)")
seq4 = ["0 V", "-0.1 V", "0.15 V", "0.7 V", "-0.238 V", "0.35 V", "OCV"]
for k1, k2 in zip(seq4[:-1], seq4[1:]):
    r, n = rms(g("4.1 nm", k1), g("4.1 nm", k2)); out[f"4.1 nm {k1} -> {k2}"] = r
    print(f"   back to back {k1:8s} -> {k2:8s}: rms {r:.3f} ({n} pts)" + ("   <- different spot (fresh CV spot)" if k2 == "OCV" else ""))
r, n = rms(g("4.1 nm", "0.7 V"), g("4.1 nm", "0 V")); print(f"   0.7 vs 0 V (1.5 h apart?): rms {r:.3f}")
json.dump(out, open(os.path.join(here, "timeline_rods.json"), "w"), indent=1)

print("\n7 nm, same L grid (0.01 steps), not back to back:")
for (k1, t1), (k2, t2) in ((("0.7 V", "a"), ("0 V", None)), (("0.7 V", "a"), ("-0.1 V", "a")), (("0.35 V", None), ("-0.1 V", "a")),
                           (("OCV", None), ("0 V", None)), (("OCV", None), ("0.35 V", None))):
    r, n = rms(g("7 nm", k1, t1), g("7 nm", k2, t2)); out[f"7 nm {k1}{t1 or ''} vs {k2}{t2 or ''}"] = r
    print(f"   {k1:7s}{t1 or ' '} vs {k2:7s}{t2 or ' '}: rms {r:.3f} ({n} pts)")
json.dump(out, open(os.path.join(here, "timeline_rods.json"), "w"), indent=1)

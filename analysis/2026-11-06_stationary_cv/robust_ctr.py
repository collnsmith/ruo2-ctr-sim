"""CTR-side robustness: duplicate files, double ratios over L pairs / windows / L offsets with an empirical error,
and the model's sensitivity at the CV HKLs over the plausible film structures."""
import itertools, sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
sys.modules["xraydb"] = None
import numpy as np
from ctrfit.project.project import Project
from ctrfit.fit.global_fit import film_of
p = Project.load(sys.argv[1]); m, P = p.model, p.model.params
rods = {}
for ds in p.datasets:
    v = ds.meta.get("potential_V")
    rods[(film_of(ds).split("|")[0], ds.rod_labels[0], "OCV" if v is None else f"{v:g} V")] = ds
print("== duplicates: identical or near-identical files at different conditions")
for (k1, d1), (k2, d2) in itertools.combinations(rods.items(), 2):
    if k1[:2] != k2[:2]: continue
    same_L = d1.L.size == d2.L.size and np.allclose(d1.L, d2.L, atol=1e-6)
    if same_L:
        r = d1.F / d2.F
        print(f"  {k1[0]} ({k1[1]}) {k1[2]} vs {k2[2]}: same L grid; F ratio spread {np.std(r / r.mean()):.4f}"
              + ("  <-- IDENTICAL" if np.std(r / r.mean()) < 1e-6 else ""))
print("== double ratios D = [I(La)/I(Lb)](E) / [..](0 V); local mean +-w; L shift dLs applied to the E rod")
def loc(ds, L, w, sh=0.0):
    k = np.abs(ds.L + sh - L) <= w
    return (ds.F[k] ** 2).mean() if k.sum() >= 2 else np.nan, int(k.sum())
cases = {"4.1 nm": ("0 -1", [2.62, 2.70, 2.75, 2.80, 2.84], [3.10, 3.16, 3.20]),
         "7 nm": ("0 1", [2.77, 2.80, 2.85], [3.10, 3.16, 3.20, 3.25])}
for film, (rod, Las, Lbs) in cases.items():
    ref = rods[(film, rod, "0 V")]
    print(f"  {film} ({rod}): points per window at w 0.03: " + ", ".join(f"{L}:{loc(ref, L, 0.03)[1]}" for L in Las + Lbs))
    for k in ("OCV", "-0.238 V", "-0.1 V", "0.15 V", "0.35 V", "0.7 V"):
        if (film, rod, k) not in rods: continue
        d = rods[(film, rod, k)]
        Ds = []
        for La, Lb, w, sh in itertools.product(Las, Lbs, (0.02, 0.03, 0.05), (-0.003, 0.0, 0.003)):
            a, _ = loc(d, Lb, w, sh); b, _ = loc(d, La, w, sh); a0, _ = loc(ref, Lb, w); b0, _ = loc(ref, La, w)
            Ds.append((a / b) / (a0 / b0))
        Ds = np.array(Ds); Ds = Ds[np.isfinite(Ds)]
        print(f"    {k:8s}: D median {np.median(Ds):.2f}, 10-90 % {np.percentile(Ds, 10):.2f}-{np.percentile(Ds, 90):.2f} ({Ds.size} choices)")
print("== model sensitivity at the CV points: change for x_OH 0.3 -> 0.7 (coherent), over thickness and dz_Obr")
lead = {}
for ds in p.datasets:
    lead.setdefault(film_of(ds).split("|")[0], ds.scope)
pts = [(0, -1, 2.75), (0, -1, 2.84), (0, -1, 3.16), (0, -1, 3.25)]
for film, tls in (("4.1 nm", (10.0, 11.0, 12.0, 12.8, 13.5, 14.5)), ("7 nm", (20.0, 22.0, 24.0))):
    v0 = dict(P.view(lead[film])); v0["theta"] = 1.0
    for tl in tls:
        line = []
        for H, K, L in pts:
            ch = []
            for dz in (0.0, 0.2, 0.35):
                v = dict(v0, thickness_mean_tl=tl, dz_Obr=dz)
                I0 = m.intensity(H, K, [L], dict(v, x_OH=0.3))[0]; I1 = m.intensity(H, K, [L], dict(v, x_OH=0.7))[0]
                ch.append(100 * (I1 / I0 - 1))
            line.append(f"L {L}: {min(ch):+5.1f}..{max(ch):+5.1f}")
        print(f"  {film} {tl:5.1f} TL | " + " | ".join(line))

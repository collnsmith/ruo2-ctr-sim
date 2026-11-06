"""Double ratio D = [I(La)/I(Lb)](E) / [I(La)/I(Lb)](0 V) from the CTR files (local means of |F|^2 within +-w in L, with
errors), against the same quantity from the stationary CVs (0 -> 0.7 V)."""
import json, sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
sys.modules["xraydb"] = None
import numpy as np
from ctrfit.project.project import Project
from ctrfit.fit.global_fit import film_of
p = Project.load(sys.argv[1]); cv = json.load(open(sys.argv[2]))
rods = {}
for ds in p.datasets:
    if ds.rod_labels[0] in ("0 1", "0 -1"):
        v = ds.meta.get("potential_V"); rods[(film_of(ds).split("|")[0], "OCV" if v is None else f"{v:g} V")] = ds
def local(ds, L, w=0.03):
    k = np.abs(ds.L - L) <= w
    if k.sum() < 2: return np.nan, np.nan
    I, s = ds.F[k] ** 2, 2 * ds.F[k] * ds.sigma_eff[k]
    return I.mean(), np.sqrt(np.sum(s ** 2)) / k.sum()
pairs = {"4.1 nm": [(3.16, 2.75), (3.25, 2.75), (3.16, 2.84)], "7 nm": [(3.16, 2.75), (3.25, 2.75)]}
cvq = {}
for s, r in cv.items():
    if r["film"] in ("4.1", "7") and r["H"] == 0:
        cvq.setdefault((r["film"] + " nm", r["L"]), []).append(1 + r["sum"]["change"])
for film, pp in pairs.items():
    for La, Lb in pp:
        a0, sa0 = local(rods[(film, "0 V")], La); b0, sb0 = local(rods[(film, "0 V")], Lb)
        line = []
        for k in ("OCV", "-0.238 V", "-0.1 V", "0.15 V", "0.35 V", "0.7 V"):
            if (film, k) not in rods: continue
            a, sa = local(rods[(film, k)], La); b, sb = local(rods[(film, k)], Lb)
            D = (a / b) / (a0 / b0)
            e = D * np.sqrt((sa / a) ** 2 + (sb / b) ** 2 + (sa0 / a0) ** 2 + (sb0 / b0) ** 2)
            line.append(f"{k} {D:.2f}±{e:.2f}")
        ca, cb = cvq.get((film, La)), cvq.get((film, Lb))
        cvD = f"{np.mean(ca) / np.mean(cb):.2f}" if ca and cb else "n/a"
        print(f"{film} I({La})/I({Lb}) vs 0 V | CTR: " + ", ".join(line) + f" | CV 0->0.7 V (sum): {cvD}")

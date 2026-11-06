"""At each CV point: model film(+surface) and substrate intensities, their ratio against the measured Ru/Ti counts,
and the relative change for x_OH 0 -> 1 in the coherent picture (one spot) and the split picture (film spot alone)."""
import json, sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
sys.modules["xraydb"] = None
import numpy as np
from ctrfit.fit.global_fit import film_of
from ctrfit.project.project import Project

p = Project.load(sys.argv[1]); cv = json.load(open(sys.argv[2]))
m, P = p.model, p.model.params
lead = {}
for ds in p.datasets:
    lead.setdefault(film_of(ds).split("|")[0], ds.scope)
src = {"4.1": "4.1 nm", "7": "7 nm", "3.3": "4.1 nm"}
rows = {}
for s, r in cv.items():
    v = dict(P.view(lead[src[r["film"]]])); v["theta"] = 1.0
    if r["film"] == "3.3": v["thickness_mean_tl"] = 10.2
    H, K, L = r["H"], r["K"], np.array([r["L"]])
    out = {}
    for x in (0.0, 0.5, 1.0):
        for dz in (0.0, 0.35):
            fm, comp = m.engine(dict(v, x_OH=x, dz_Obr=dz))
            out[(x, dz)] = (float(fm.film_intensity(H, K, L, comp)[0]), float(abs(fm.F_sub(H, K, L)[0]) ** 2),
                            float(fm.intensity(H, K, L, comp)[0]))
    f0, s0, c0 = out[(0.0, 0.35)]; f1, s1, c1 = out[(1.0, 0.35)]
    g0, _, d0 = out[(0.0, 0.0)]; g1, _, d1 = out[(1.0, 0.0)]
    rows[s] = dict(film_over_sub=f0 / s0, coh=c1 / c0 - 1, split=f1 / f0 - 1, coh_dz0=d1 / d0 - 1, split_dz0=g1 / g0 - 1,
                   meas_ratio=r["counts"]["Ru"] / r["counts"]["Ti"], ru=r["Ru"]["change"], ti=r["Ti"]["change"], sm=r["sum"]["change"])
    q = rows[s]
    print(f"{s} {r['film']:>3s} nm ({H:2d} {K:2d} {r['L']:4.2f}) film/sub model {q['film_over_sub']:6.2f} meas Ru/Ti {q['meas_ratio']:5.2f} | "
          f"x_OH 0->1 (dz .35 / 0): coherent {100*q['coh']:+6.1f} / {100*q['coh_dz0']:+6.1f} %, film spot {100*q['split']:+6.1f} / {100*q['split_dz0']:+6.1f} % "
          f"| measured Ru {100*q['ru']:+6.1f}, sum {100*q['sm']:+6.1f} %")
json.dump(rows, open(sys.argv[3], "w"), indent=1)

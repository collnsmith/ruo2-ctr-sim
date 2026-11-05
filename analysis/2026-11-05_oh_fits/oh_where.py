import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))   # the repository; sys.modules["xraydb"] = None
import numpy as np
from ctrfit.project.project import Project
from ctrfit.fit.global_fit import film_of
p = Project.load(sys.argv[1])
m, P, D = p.model, p.model.params, p.datasets
lead = {}
for d in D:
    lead.setdefault(film_of(d).split("|")[0], d.scope)
L = np.round(np.arange(0.3, 3.55, 0.05), 3)
print("relative change of |F| for x_OH 0.3 -> 0.7 (theta 1), along (0 1); Bragg at L = 3:")
rows = {}
for film, sc in lead.items():
    v = dict(P.view(sc)); v.update(theta=1.0)
    a = m.F(0, 1, L, dict(v, x_OH=0.3)); b = m.F(0, 1, L, dict(v, x_OH=0.7))
    rows[film] = np.abs(b / a - 1)
for i in range(0, L.size, 4):
    print(f"  L {L[i]:.2f}: " + "  ".join(f"{f} {100 * rows[f][i]:5.1f} %" for f in rows))
print("your measured (0 +-1) range L 2.5-3.4: median " + ", ".join(
    f"{f} {100 * np.median(rows[f][(L >= 2.5) & (L <= 3.4)]):.1f} %" for f in rows))
print("stationary CV points:")
for hkl in [(0, 1, 2.75), (0, 1, 3.16), (0, 1, 3.25), (0, -1, 1.25), (0, -1, 2.84), (-2, 0, 1.75), (-1, 0, 2.32), (-2, 0, 1.0)]:
    out = []
    for film, sc in lead.items():
        v = dict(P.view(sc)); v.update(theta=1.0)
        a = m.F(hkl[0], hkl[1], np.array([hkl[2]]), dict(v, x_OH=0.3))[0]
        b = m.F(hkl[0], hkl[1], np.array([hkl[2]]), dict(v, x_OH=0.7))[0]
        out.append(f"{film} {100 * (b / a - 1):+6.1f} %")
    print(f"  ({hkl[0]} {hkl[1]} {hkl[2]}): " + "  ".join(out))

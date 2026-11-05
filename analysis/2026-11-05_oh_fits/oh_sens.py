import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))   # the repository; sys.modules["xraydb"] = None
import numpy as np
from ctrfit.project.project import Project
from ctrfit.fit.global_fit import film_of

def main():
    p = Project.load(sys.argv[1])
    m, P = p.model, p.model.params
    sel = p.scans
    for r in sel.rows():
        sel.set_use([r.key], r.rods != ["0 0"])
    from ctrfit.fit.scans import regroup
    D, _ = regroup(m, sel, p.datasets)
    out = []
    for ds in D:
        v = dict(P.view(ds.scope))
        v["theta"] = 1.0
        ev = m.evaluator(ds.H, ds.K, ds.L, scope=ds.scope)
        def F(x, dz):
            w = dict(v, x_OH=x, dz_Obr=dz)
            return ev(w)
        s = ds.sigma_eff
        def dchi(a, b):                       # a vs b with b's best scale
            c = np.sum(a * b / s ** 2) / np.sum(b * b / s ** 2)
            return float(np.sum(((a - c * b) / s) ** 2))
        dz0 = v["dz_Obr"]
        A = F(0.3, dz0)
        d1 = dchi(A, F(0.7, dz0))
        d2 = min(dchi(A, F(0.7, dz)) for dz in np.linspace(-0.2, 0.4, 25))
        out.append((d1, d2, film_of(ds).split("|")[0], ds.meta.get("potential_V"), ds.rod_labels[0], len(ds),
                    float(ds.L.min()), float(ds.L.max())))
    print("x_OH 0.3 vs 0.7 told apart by each scan (delta chi2; scale refit | scale and dz_Obr refit), largest first:")
    for d1, d2, film, pot, rod, n, lo, hi in sorted(out, key=lambda t: -t[1]):
        print(f"  {film:6s} {str(pot):6s} ({rod:4s}) L {lo:.2f}-{hi:.2f} n {n:3d}: {d1:8.1f} | {d2:8.1f}")

if __name__ == "__main__":
    main()

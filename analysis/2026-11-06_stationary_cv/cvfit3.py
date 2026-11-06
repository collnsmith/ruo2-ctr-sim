"""Stationary CVs through the CTR-fitted structure: per film, x_OH(E) = x0 + (x1 - x0) s((E - E0)/w), optionally with
dz_Obr(E) = d0 + (d1 - d0) s(...) on the same step; the CV's change at each HKL is I(x(E), dz(E)) / I(x(0), dz(0)) - 1
(plus a small offset per scan), against the cycle-averaged curves (both sweeps)."""
import json
import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
sys.modules["xraydb"] = None
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

from ctrfit.fit.global_fit import film_of
from ctrfit.project.project import Project

X = np.linspace(0, 1, 21)
DZ = np.linspace(-0.15, 0.45, 13)


def sig(z):
    return 0.5 * (1 + np.tanh(z))


def main():
    global SPLIT
    proj, curves_path, out, roi = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    SPLIT = False                              # hybrid: the Ru ROI against the coherent model
    skip = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else set()
    p = Project.load(proj)
    m, P, D = p.model, p.model.params, p.datasets
    lead = {}
    for ds in D:
        lead.setdefault(film_of(ds).split("|")[0], ds.scope)
    cv = {k: dict(r, curves=r[roi]["curves"]) for k, r in json.load(open(curves_path)).items() if k not in skip}
    films = {"4.1": "4.1 nm", "7": "7 nm", "3.3": "4.1 nm"}          # 3.3 nm: the 4.1 nm structure, thinner
    tables = {}
    for s, r in cv.items():
        film = r["film"]
        v = dict(P.view(lead[films[film]]))
        v["theta"] = 1.0
        if film == "3.3":
            v["thickness_mean_tl"] = 10.2
        key = (film, r["H"], r["K"], r["L"])
        if key not in tables:
            grid = np.empty((X.size, DZ.size))
            for i, x in enumerate(X):
                for j, dz in enumerate(DZ):
                    fm, comp = m.engine(dict(v, x_OH=x, dz_Obr=dz))
                    Lq = np.array([r["L"]])
                    grid[i, j] = float((fm.film_intensity if SPLIT else fm.intensity)(r["H"], r["K"], Lq, comp)[0])
            tables[key] = RegularGridInterpolator((X, DZ), grid, method="cubic")
        r["table"] = key
    results = {}
    for film in ("7", "3.3", "4.1"):
        scans = [s for s, r in cv.items() if r["film"] == film and (r["H"], r["K"]) != (2, 0)]
        pts = []
        for s in scans:
            for br in ("anodic", "cathodic"):
                c = cv[s]["curves"][br]
                for E, dI, e in zip(c["E"], c["dI"], c["err"]):
                    pts.append((s, E, dI, np.hypot(e, 0.005)))
        S = sorted(set(scans))
        for variant in ("x_OH only", "x_OH + dz_Obr"):
            both = variant.endswith("dz_Obr")

            def model(q):
                x0, x1, E0, w = q[:4]
                d0, d1 = (q[4], q[5]) if both else (0.35, 0.35)
                off = dict(zip(S, q[6 if both else 4:]))
                res = []
                for s, E, dI, e in pts:
                    t = tables[cv[s]["table"]]
                    f = sig((E - E0) / w)
                    f0 = sig((0.0 - E0) / w)
                    a = t([[x0 + (x1 - x0) * f, d0 + (d1 - d0) * f]])[0]
                    b = t([[x0 + (x1 - x0) * f0, d0 + (d1 - d0) * f0]])[0]
                    res.append((a / b - 1 + off[s] - dI) / e)
                return np.array(res)
            best = None
            for x0, x1 in ((0.0, 1.0), (1.0, 0.0), (0.2, 0.6), (0.6, 0.2)):
                q0 = [x0, x1, 0.35, 0.08] + ([0.35, 0.0] if both else []) + [0.0] * len(S)
                lo = [0, 0, -0.1, 0.01] + ([-0.15, -0.15] if both else []) + [-0.1] * len(S)
                hi = [1, 1, 0.8, 0.5] + ([0.45, 0.45] if both else []) + [0.1] * len(S)
                f = least_squares(model, q0, bounds=(lo, hi))
                if best is None or f.cost < best.cost:
                    best = f
            q = best.x
            J = best.jac
            dof = max(len(pts) - len(q), 1)
            red = 2 * best.cost / dof
            try:
                cov = np.linalg.inv(J.T @ J) * red
                err = np.sqrt(np.diag(cov))
            except np.linalg.LinAlgError:
                err = np.full(q.size, np.nan)
            names = ["x0", "x1", "E0", "w"] + (["d0", "d1"] if both else [])
            results[f"{film} nm | {variant}"] = dict(zip(names, map(float, q)), errors=dict(zip(names, map(float, err))),
                                                    red_chi2=float(red), n=len(pts), scans=S)
            vals = ", ".join(f"{n} {q[i]:.2f}±{err[i]:.2f}" for i, n in enumerate(names))
            print(f"{film:>3s} nm, {variant:14s}: red chi2 {red:6.2f}  {vals}", flush=True)
    # 4.1 nm: the largest x_OH change the flat CVs allow (a step mid-window, x0 free)
    scans = [s for s, r in cv.items() if r["film"] == "4.1"]
    pts = [(s, E, dI, np.hypot(e, 0.005)) for s in scans for br in ("anodic", "cathodic")
           for E, dI, e in zip(cv[s]["curves"][br]["E"], cv[s]["curves"][br]["dI"], cv[s]["curves"][br]["err"])]
    chi = lambda x0, x1: sum(((tables[cv[s]["table"]]([[x0 + (x1 - x0) * sig((E - 0.35) / 0.08), 0.35]])[0]
                               / tables[cv[s]["table"]]([[x0 + (x1 - x0) * sig(-0.35 / 0.08), 0.35]])[0] - 1 - dI) / e) ** 2
                             for s, E, dI, e in pts)
    base = min(chi(x, x) for x in (0.0, 0.5, 1.0))
    for x0 in (0.0, 0.5, 0.9):
        allowed = [d for d in np.linspace(0, 1 - x0, 21) if chi(x0, x0 + d) <= base + 4 * max(1.0, base / len(pts))]
        print(f"4.1 nm from x_OH {x0:.1f}: largest increase the flat CVs allow {max(allowed):.2f}")
    json.dump(results, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()

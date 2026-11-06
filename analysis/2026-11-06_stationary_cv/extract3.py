"""CV curves per ROI (Ru, Ti, sum), timing found by a shape-free search (period and phase that make the signal the most
a function of E and sweep direction), and for the flat scans the largest response any plausible timing allows."""
import json, sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
import numpy as np
from ctrfit.echem.analysis import dominant_period
exec(open(__file__.replace("extract3.py", "extract.py")).read().split("rows = info()")[0])   # info, load, potential
FCA = CA / CYCLE
BINS = np.linspace(E1, E2, 15)

def r2(rel, n, a, b, P, d):
    """fraction of the variance explained by a free curve in (E, branch): 14 bins per sweep."""
    E, br = potential(n, a + FCA * P + d, P)
    k = np.arange(a, b); k = k[br[k] != 0]
    lab = (br[k] > 0) * 100 + np.digitize(E[k], BINS)
    tot = np.var(rel[k]) * k.size
    if tot <= 0: return 0.0
    within = sum(np.var(rel[k][lab == u]) * np.sum(lab == u) for u in np.unique(lab))
    return 1 - within / tot

def curves(rel, n, a, b, P, d):
    E, br = potential(n, a + FCA * P + d, P)
    k = np.arange(a, b); k = k[br[k] != 0]
    out = {}
    for s, lab in ((1, "anodic"), (-1, "cathodic")):
        m, e, c = [], [], []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            q = k[(br[k] == s) & (E[k] >= lo) & (E[k] < hi)]
            if q.size > 1:
                m.append(float(np.mean(rel[q] - 1))); e.append(float(np.std(rel[q]) / np.sqrt(q.size))); c.append(float(0.5 * (lo + hi)))
        out[lab] = dict(E=c, dI=m, err=e)
    A = np.column_stack([np.ones(k.size), E[k]])
    coef, *_ = np.linalg.lstsq(A, rel[k], rcond=None)
    res = rel[k] - A @ coef
    cov = np.linalg.inv(A.T @ A) * np.var(res)
    return out, float(coef[1] * (E2 - E1)), float(np.sqrt(cov[1, 1]) * (E2 - E1)), float(np.std(res))

OLD = json.load(open(sys.argv[4]))
out = {}
for r in info():
    ru, ti, sru, sti = load(r)
    y = ru + ti
    n = np.arange(y.size, dtype=float)
    a, b = r["start"], min(r["stop"], y.size - 1)
    base = slice(a, a + 6)                                   # the CA at 0.0 V (about 8 points at the slowest rate)
    sig = {"Ru": ru / ru[base].mean(), "Ti": ti / ti[base].mean(), "sum": y / y[base].mean()}
    old = OLD[str(r["scan"])]
    P, d = old["P"], old["shift"]
    periodic = old["how"].startswith("period")
    q = r2(sig["Ru"], n, a, b, P, d)
    res = dict(r, P=float(P), shift=float(d), r2_ru=float(q), periodic=bool(periodic),
               counts=dict(Ru=float(ru[base].mean()), Ti=float(ti[base].mean())))
    for name, s in sig.items():
        c, sl, sle, noise = curves(s, n, a, b, P, d)
        res[name] = dict(curves=c, change=sl, change_err=sle, noise=noise)
    if not periodic:                                         # largest linear response over any timing
        mx = 0.0
        for Pp in np.linspace(65, 125, 31):
            for dd in np.linspace(-0.15, 0.15, 13) * Pp:
                mx = max(mx, abs(curves(sig["sum"], n, a, b, Pp, dd)[1]))
        res["sum"]["max_any_timing"] = mx
    out[str(r["scan"])] = res
    f = lambda k: f"{100 * res[k]['change']:+6.1f}±{100 * res[k]['change_err']:.1f}"
    print(f"{r['scan']} {r['film']:>3s} nm ({r['H']:2d} {r['K']:2d} {r['L']:4.2f}) Ru/Ti {ru[base].mean() / ti[base].mean():5.2f} "
          f"P {P:6.1f} shift {d:+6.1f} R2(Ru) {q:.2f} {'periodic' if periodic else 'flat    '} "
          f"change Ru {f('Ru')} Ti {f('Ti')} sum {f('sum')} %"
          + (f"  max any timing {100 * res['sum']['max_any_timing']:.1f} %" if not periodic else ""))
json.dump(out, open(sys.argv[3], "w"), indent=1, default=float)

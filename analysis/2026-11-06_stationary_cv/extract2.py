import json, sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
import numpy as np
from ctrfit.echem.external import parse_roi_sums
from ctrfit.echem.analysis import dominant_period
exec(open(__file__.replace("extract2.py", "extract.py")).read().split("rows = info()")[0])   # info, load, potential

FCA = CA / CYCLE                                 # the 30 s hold in cycles
rows = info()
out = {}
for r in rows:
    ru, ti, sru, sti = load(r)
    y, sy = ru + ti, np.hypot(sru, sti)
    n = np.arange(y.size, dtype=float)
    a, b = r["start"], min(r["stop"], y.size - 1)        # a Stop past the last image: the last image
    span = b - a
    P0 = None
    for sig in (y, ru):                          # the integrated signal, else the Ru ROI alone
        P0 = dominant_period(n[a:b], sig[a:b], min_period=40)
        if P0:
            break
    if P0:
        N = max(0.5, round(2 * (span / P0 - FCA)) / 2)              # whole or half cycles
        how = "period from the data"
    else:
        N = max(1, round(span / (0.30 * CYCLE) - FCA))               # about 0.3 points per second
        how = "flat: cycles from 0.3 points/s"
    P = span / (N + FCA)
    base = slice(a, int(round(a + FCA * P)))                          # the 30 s CA at 0.0 V
    rel = y / np.mean(y[base])
    srel = sy / np.mean(y[base])
    best = (0.0, 0.0)
    for d in np.linspace(-0.15, 0.15, 61) * P:                       # timing: the phase that follows the potential best
        E, br = potential(n, a + FCA * P + d, P)
        k = slice(a, b)
        c = np.corrcoef(E[k], rel[k])[0, 1] if np.std(rel[k]) > 0 else 0
        if np.isfinite(c) and abs(c) > abs(best[1]):
            best = (d, c)
    d = best[0] if P0 else 0.0
    E, br = potential(n, a + FCA * P + d, P)
    k = np.arange(a, b)
    k = k[(br[k] != 0)]
    bins = np.linspace(E1, E2, 15)
    curves = {}
    for bsgn, lab in ((1, "anodic"), (-1, "cathodic")):
        m, s, c = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            q = k[(br[k] == bsgn) & (E[k] >= lo) & (E[k] < hi)]
            if q.size:
                m.append(float(np.mean(rel[q] - 1)))
                s.append(float(np.std(rel[q]) / np.sqrt(q.size)) if q.size > 1 else float(srel[q][0]))
                c.append(float(0.5 * (lo + hi)))
        curves[lab] = dict(E=c, dI=m, err=s)
    # amplitude of the response that follows the potential: rel = a0 + a1 * E (fit over the sweeps)
    A = np.column_stack([np.ones(k.size), E[k]])
    coef, *_ = np.linalg.lstsq(A, rel[k], rcond=None)
    resid = rel[k] - A @ coef
    cov = np.linalg.inv(A.T @ A) * np.var(resid)
    out[r["scan"]] = dict(r, N=N, P=P, how=how, shift=float(d), corr=float(best[1]), slope=float(coef[1] * (E2 - E1)),
                          slope_err=float(np.sqrt(cov[1, 1]) * (E2 - E1)), noise=float(np.std(resid)), curves=curves)
    print(f"{r['scan']} {r['film']:>3s} nm ({r['H']} {r['K']} {r['L']}): {N:g} cycles, P {P:.1f} pts ({how}), "
          f"timing shift {d:+.1f} pts, change 0->0.7 V {100 * coef[1] * (E2 - E1):+.1f} ± {100 * np.sqrt(cov[1, 1]) * (E2 - E1):.1f} %, "
          f"noise {100 * np.std(resid):.1f} %")
json.dump({str(k): v for k, v in out.items()}, open(sys.argv[3], "w"), indent=1, default=float)

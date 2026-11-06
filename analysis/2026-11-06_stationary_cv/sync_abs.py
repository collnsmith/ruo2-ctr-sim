"""Absolute sync of the stationary CV scans: every X-ray point gets the potential EC-Lab applied at that moment.

Times: SPEC point time = the file's #D time + the point's Epoch column; EC-Lab time = the sequence start (file timestamp,
shared by the CA and CV files of a sequence) + time/s. Both clocks are local beamline time.

Step 1, the clock offset. For the scans whose X-ray signal follows the potential (3.3 and 7 nm), the offset d that
maximises |corr(I, E(t - d))| (I with a quadratic drift removed). Cross-check: the user's Start marks against the EC-Lab
sequence start. If d agrees across scans and days, it is applied to day 4.

Step 2, the response with no free timing. With d fixed, each scan is fitted as
    I(t) / I0 = c0 + c1 t + c2 t^2 + A g(E(t))
with g a fixed switch shape (0 at 0 V, 1 at 0.7 V), so A is the change from 0 to 0.7 V. Two shapes: linear in E, and a
logistic step at 0.23 V (the 3.3 / 7 nm half-way potential, 10-90 % over about 0.25 V). The error of A is the larger of
(1) the spread of A refitted to the model plus phase-randomised copies of its residual (same noise spectrum) and
(2) a Newey-West (autocorrelation-robust) error. (A first version shifted E(t) in time instead; with about 2 cycles per
scan that gives too few independent draws and underestimated the 4.1 nm errors by up to 1.8x.)
Writes sync_abs.json and fig_sync_abs.png."""
import datetime as dt
import glob, json, os, sys, warnings
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))
sys.modules["xraydb"] = None
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from galvani import BioLogic
from ctrfit.echem.external import parse_roi_sums
from ctrfit.datatools.specfile import parse_spec_file

SPECF = os.path.join(here, "data", "08_9_3.3nm.spec")
SPEC = parse_spec_file(SPECF)
D0 = dt.datetime(2026, 10, 9, 13, 37, 24)                 # #D of the file, the time of #E (Epoch column = s since it)
D2, D3, D4 = "Collin_08102026", "Collin_08112026", "Collin_08122026"
# scan: (film, H K L, Ru ROI, Ti ROI, EC folder, EC file pattern, Start, Stop)
SCANS = {
    120: ("3.3", "0 -1 2.75", "ROI1", "ROI2", D2, "*scan119_0*", 73, 433),
    125: ("3.3", "0 -1 3.25", "ROI1", "ROI2", D2, "*scan125_0*", 50, 432),
    147: ("3.3", "0 -1 1.25", "ROI1", "ROI2", D2, "*scan147_0*", 50, 312),
    222: ("3.3", "0 -1 3.16", "ROI1", "ROI2", D2, "*scan222_0*", 50, 250),
    223: ("3.3", "0 -1 3.16", "ROI1", "ROI2", D2, "*scan223_0*", 51, 410),
    311: ("7", "0 1 2.75", "ROI2", "ROI1", D3, "*scan311_0*", 50, 287),
    314: ("7", "0 1 3.16", "ROI2", "ROI1", D3, "*scan314_0*", 50, 308),
    316: ("7", "0 1 3.25", "ROI2", "ROI1", D3, "*scan316_0*", 50, 211),
    341: ("7", "0 1 2.75", "ROI2", "ROI1", D3, "*scan341_0*", None, None),
    343: ("7", "0 1 2.75", "ROI2", "ROI1", D3, "*scan342_0*", None, None),
    344: ("7", "0 1 2.75", "ROI2", "ROI1", D3, "scan247_0*", 20, 350),
    496: ("4.1", "0 -1 2.75", "ROI1", "ROI2", D4, "*scan496_0*", 20, 209),
    497: ("4.1", "0 -1 2.75", "ROI1", "ROI2", D4, "*scan497_lower_0*", 20, 182),
    498: ("4.1", "0 -1 3.16", "ROI1", "ROI2", D4, "*scan498_lower_0*", 20, 216),
    501: ("4.1", "0 -1 3.25", "ROI1", "ROI2", D4, "*scan501_0*", 20, 210),
    503: ("4.1", "0 -1 2.84", "ROI1", "ROI2", D4, "*scan503_0*", 20, 182),
    519: ("4.1", "-2 0 1.75", "ROI1", "ROI2", D4, "*scan519_0*", 20, 242),
}
CV_RATE = 0.005

def spec_times(n):
    d = SPEC[n].data
    return np.array([(D0 + dt.timedelta(seconds=float(e))) for e in d["Epoch"]])

def ec_trace(folder, pat):
    fs = sorted(glob.glob(os.path.join(here, "data", "echem", folder, pat)))
    fs = [f for f in fs if f.endswith(".mpr")]
    t, E, start = [], [], None
    for f in fs:
        m = BioLogic.MPRfile(f)
        start = start or m.timestamp
        t.append(m.data["time/s"]); E.append(m.data["control/V"])
    t, E = np.concatenate(t), np.concatenate(E)
    o = np.argsort(t)
    return start, t[o], E[o]

def xray(n, ru, ti):
    tab = parse_roi_sums(open(os.path.join(here, "data", f"ROIs_s{n}_sums.txt")).read())[n]
    y = np.asarray(tab[ru], float) + np.asarray(tab[ti], float)
    return y

def shapes(E):
    lin = np.clip(E / 0.7, 0, 1)
    s = lambda x: 1 / (1 + np.exp(-(x - 0.23) / 0.057))
    logi = (s(E) - s(0.0)) / (s(0.7) - s(0.0))
    return {"linear": lin, "logistic": logi}

def fit(t, y, g):
    tt = (t - t.mean()) / max(np.ptp(t), 1)
    X = np.c_[np.ones_like(tt), tt, tt ** 2, g]
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c[3], y - X @ c

def hac(t, u, g, lag=None):
    """Newey-West standard error of the g coefficient (quadratic drift in the model), residual u."""
    tt = (t - t.mean()) / max(np.ptp(t), 1)
    X = np.c_[np.ones_like(tt), tt, tt ** 2, g]
    n = len(u); lag = lag or max(5, int(n / 8))
    S = (X * u[:, None]).T @ (X * u[:, None])
    for l in range(1, lag + 1):
        G = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None])
        S += (1 - l / (lag + 1)) * (G + G.T)
    B = np.linalg.inv(X.T @ X)
    return float(np.sqrt((B @ S @ B)[3, 3]))


def series(n, d):
    """X-ray points (t in s from the EC start, relative intensity) and E at t - d, within the scan's EC record."""
    film, hkl, ru, ti, folder, pat, a, b = SCANS[n]
    y = xray(n, ru, ti)
    ts = spec_times(n)[: y.size]
    y = y[: ts.size]
    start, te, Ee = ec_trace(folder, pat)
    t = np.array([(x - start).total_seconds() for x in ts])
    tmax = te.max()
    if n == 222:
        tmax = 730.0                                            # 5 mV/s part only (the 25 mV/s part is left out)
    ok = (t - d >= te.min() + 30.5) & (t - d <= tmax)           # after the 30 s hold, inside the EC record
    E = np.interp(t - d, te, Ee)
    base = (t - d >= te.min()) & (t - d < te.min() + 30)       # the 30 s hold at 0 V
    I0 = np.mean(y[base]) if base.sum() >= 3 else np.mean(y[ok][:10])
    return t[ok], y[ok] / I0, E[ok], start, ts, (a, b)

out = {"offset": {}, "start_marks": {}, "response": {}}
# step 1: clock offset per responsive scan; cross-check with the Start marks
for n, (film, hkl, ru, ti, folder, pat, a, b) in SCANS.items():
    start, te, Ee = ec_trace(folder, pat)
    ts = spec_times(n)
    if a is not None and a < ts.size:
        out["start_marks"][n] = (start - ts[a]).total_seconds()     # EC start minus the time of the marked point
    if film == "4.1":
        continue
    best = (0, 0.0)
    for d in np.arange(-60, 61, 1.0):                             # inside a quarter cycle: no half-period alias
        t, y, E, *_ = series(n, d)
        if t.size < 30:
            continue
        _, r = fit(t, y, np.zeros_like(t))                          # drift removed
        c = np.corrcoef(r, E)[0, 1]
        if abs(c) > abs(best[1]):
            best = (d, c)
    out["offset"][n] = dict(film=film, d=best[0], corr=best[1])
day = {"3.3": [], "7": []}
for n, v in out["offset"].items():
    if abs(v["corr"]) > 0.5:
        day[v["film"]].append(v["d"])
d_all = [x for v in day.values() for x in v]
D = float(np.median(d_all)); D_sd = float(np.std(d_all)); D_RANGE = 10.0   # +-10 s covers the scatter and any lag
out["offset_used"] = dict(d=D, sd=D_sd, per_day={k: v for k, v in day.items()})

# step 2: the response with the offset fixed
rng = np.random.default_rng(0)
for n, (film, hkl, *_rest) in SCANS.items():
    t, y, E, start, ts, ab = series(n, D)
    if t.size < 30:
        continue
    res = {}
    for name in ("linear", "logistic"):
        g = shapes(E)[name]
        A, u = fit(t, y, g)
        yhat = y - u
        F = np.fft.rfft(u)
        boot = []
        for _ in range(500):
            ph = np.exp(1j * rng.uniform(0, 2 * np.pi, F.size)); ph[0] = 1
            if u.size % 2 == 0:
                ph[-1] = 1
            boot.append(fit(t, yhat + np.fft.irfft(F * ph, n=u.size), g)[0])
        e_s, e_h = float(np.std(boot)), hac(t, u, g)
        res[name] = dict(A=float(A), err=max(e_s, e_h), err_surrogate=e_s, err_hac=e_h)
        # the offset uncertainty: refit at D +- sd
        for dd in (-D_RANGE, D_RANGE):
            t2, y2, E2, *_ = series(n, D + dd)
            res[name].setdefault("A_offset_range", []).append(float(fit(t2, y2, shapes(E2)[name])[0]))
    out["response"][n] = dict(film=film, hkl=hkl, n_points=int(t.size), cycles=float(np.ptp(t) / 280.0), **res)

# per reflection: weighted means (logistic shape) and the 4.1 nm response as a fraction of the 3.3 / 7 nm response
groups = {"2.75": {"3.3": [120], "7": [311, 341, 343], "4.1": [496, 497]}, "3.16": {"3.3": [222, 223], "7": [314], "4.1": [498]},
          "3.25": {"3.3": [125], "7": [316], "4.1": [501]}}
def wmean(ns, shape="logistic"):
    a = np.array([out["response"][n][shape]["A"] for n in ns]); e = np.array([out["response"][n][shape]["err"] for n in ns])
    w = 1 / e ** 2
    return float(np.sum(w * a) / w.sum()), float(1 / np.sqrt(w.sum()))
summ = {}
for L, g in groups.items():
    m = {f: wmean(ns) for f, ns in g.items()}
    r = {}
    for ref in ("3.3", "7"):
        a, ea = m["4.1"]; b, eb = m[ref]
        r[ref] = (a / b, abs(a / b) * np.hypot(ea / a if a else 0, eb / b)) if a else (0.0, ea / abs(b))
        r[ref] = (float(a / b), float(np.hypot(ea / b, a * eb / b ** 2)))
    summ[L] = dict(mean=m, ratio_41_to=r)
out["summary"] = summ
json.dump(out, open(os.path.join(here, "sync_abs.json"), "w"), indent=1, default=float)

# figure: the change from 0 to 0.7 V at each reflection, every scan, the measured potential and fixed timing
COL = {"3.3": "#2a78d6", "4.1": "#eb6834", "7": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8.5, 4.6))
xs = {"2.75": 0, "3.16": 1, "3.25": 2}
off = {"3.3": -0.22, "4.1": 0.0, "7": 0.22}
for L, g in groups.items():
    for f, ns in g.items():
        for j, n in enumerate(ns):
            v = out["response"][n]["logistic"]
            x = xs[L] + off[f] + (j - (len(ns) - 1) / 2) * 0.05
            ax.errorbar(x, 100 * v["A"], yerr=100 * v["err"], fmt="o", ms=6, color=COL[f], elinewidth=1.2, capsize=0,
                        label=f"{f} nm" if (L == "2.75" and j == 0) else None)
ax.axhline(0, color=GRID, lw=0.9, zorder=0)
ax.set_xticks(range(3)); ax.set_xticklabels(["(0 ±1 2.75)", "(0 ±1 3.16)", "(0 ±1 3.25)"])
ax.set_ylabel("X-ray change, 0 to 0.7 V (%)")
ax.set_title("Response to the applied potential, timing fixed by the EC-Lab and SPEC clocks", loc="left", color=INK, fontsize=10.5)
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="film, one point per scan")
fig.text(0.01, -0.02, "Each point: I/I0 = drift (quadratic) + A g(E), E the EC-Lab potential at the point's SPEC time (offset "
         f"{D:+.0f} s), g a step at 0.23 V. Error bars: the larger of a noise-surrogate and an autocorrelation-robust error.", fontsize=8, color=MUTED)
fig.tight_layout(); fig.savefig(os.path.join(here, "fig_sync_abs.png"), dpi=150, bbox_inches="tight")
print("start marks: EC start minus marked point time (s):", {k: round(v, 1) for k, v in out["start_marks"].items()})
print("offset per responsive scan (s, corr):", {k: (v["d"], round(v["corr"], 2)) for k, v in out["offset"].items()})
print(f"offset used: {D:+.1f} s (spread {D_sd:.1f}; +-{D_RANGE:g} s tested)", out["offset_used"]["per_day"])
for n, v in out["response"].items():
    a, b = v["linear"], v["logistic"]
    print(f"{n} {v['film']:>3} nm ({v['hkl']}) n={v['n_points']:4d} cycles {v['cycles']:.1f}  linear A = {100*a['A']:+6.2f} +- {100*a['err']:.2f} %   "
          f"logistic A = {100*b['A']:+6.2f} +- {100*b['err']:.2f} %  (offset +-: {[round(100*x, 2) for x in b['A_offset_range']]})")
for L, v in out["summary"].items():
    print(L, {f: f"{100*a:+.2f} +- {100*e:.2f} %" for f, (a, e) in v["mean"].items()},
          "4.1 / 3.3:", "%.3f +- %.3f" % tuple(v["ratio_41_to"]["3.3"]), " 4.1 / 7:", "%.3f +- %.3f" % tuple(v["ratio_41_to"]["7"]))

"""Is there a weak potential-locked response in the 4.1 nm CV scans? Shape-free tests at the applied potential.

For each scan and ROI: the signal divided by the SPEC monitor (relative to its mean), the potential EC-Lab applied at each
point's SPEC time (sync_abs.py), points after the 30 s hold up to the logged Stop. A quadratic drift in time is removed.
  - E-locked test: the residual is binned in E (7 bins of 0.1 V, both sweeps pooled). Q = sum_k n_k mean_k^2 / var.
  - Sweep-resolved test: 14 bins (E x sweep direction).
  - Linear amplitude: the change from 0 to 0.7 V of a straight line in E.
Errors and p-values: the residual after the drift (no E term) is phase-randomised (same power spectrum, so the same
noise correlations, no link to E), 2000 surrogates; the linear amplitude also gets a Newey-West (autocorrelation-robust)
error, and the larger of the two is used. (Shifting E(t) in time, used first, gives too few independent draws with only
about 2 cycles per scan and underestimates the error.) Responding scans (3.3, 7 nm) are run the same way as a check.
Writes weak41.json."""
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "sync_abs.py")).read()
__file__ = os.path.join(here, "sync_abs.py")
exec(src.split('out = {"offset": {}')[0])
import numpy as np

D = json.load(open(os.path.join(here, "sync_abs.json")))["offset_used"]["d"]
SCANS41 = [496, 497, 498, 501, 503, 519]
CHECK = [222, 314, 316, 125]
ROIS = ["ROI1", "ROI2", "ROI5", "ROI6", "sum"]
BINS = np.linspace(0, 0.7, 8)
rng = np.random.default_rng(1)


def data(n, roi):
    film, hkl, ru, ti, folder, pat, a, b = SCANS[n]
    tab = parse_roi_sums(open(os.path.join(here, "data", f"ROIs_s{n}_sums.txt")).read())[n]
    if roi == "sum":
        y = np.asarray(tab[ru], float) + np.asarray(tab[ti], float)
    else:
        if roi not in tab:
            return None
        y = np.asarray(tab[roi], float)
    sp = SPEC[n].data
    m = min(y.size, len(sp["Epoch"]))
    y = y[:m] / (sp["Monitor"][:m] / np.mean(sp["Monitor"][:m])) / sp["transm"][:m]
    ts = spec_times(n)[:m]
    start, te, Ee = ec_trace(folder, pat)
    t = np.array([(x - start).total_seconds() for x in ts])
    tmax = 730.0 if n == 222 else te.max()
    if b is not None:
        tmax = min(tmax, t[min(b, m - 1)])
    ok = (t - D >= te.min() + 30.5) & (t - D <= tmax)
    E = np.interp(t - D, te, Ee)
    return t[ok], y[ok] / np.mean(y[ok]), E[ok]


def detrend(t, y):
    tt = (t - t.mean()) / np.ptp(t)
    X = np.c_[np.ones_like(tt), tt, tt ** 2]
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ c


def stats(t, r, E):
    dE = np.gradient(E, t)
    k = np.clip(np.digitize(E, BINS) - 1, 0, BINS.size - 2)
    v = np.var(r)
    q1 = sum(np.sum(k == i) * np.mean(r[k == i]) ** 2 for i in range(BINS.size - 1) if np.any(k == i)) / v
    kd = k + (BINS.size - 1) * (dE < 0)
    q2 = sum(np.sum(kd == i) * np.mean(r[kd == i]) ** 2 for i in range(2 * (BINS.size - 1)) if np.any(kd == i)) / v
    tt = (t - t.mean()) / np.ptp(t)
    X = np.c_[np.ones_like(tt), tt, tt ** 2, E / 0.7]
    lin = np.linalg.lstsq(X, r, rcond=None)[0][3]
    return q1, q2, lin


def newey_west(t, r, E, lag=None):
    """Autocorrelation-robust standard error of the E / 0.7 coefficient (quadratic drift in the model)."""
    tt = (t - t.mean()) / np.ptp(t)
    X = np.c_[np.ones_like(tt), tt, tt ** 2, E / 0.7]
    c, *_ = np.linalg.lstsq(X, r, rcond=None)
    u = r - X @ c
    n = len(u); lag = lag or max(5, int(n / 8))
    S = (X * u[:, None]).T @ (X * u[:, None])
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        G = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None])
        S += w * (G + G.T)
    B = np.linalg.inv(X.T @ X)
    return float(np.sqrt((B @ S @ B)[3, 3]))


def curve(t, r, E):
    dE = np.gradient(E, t)
    k = np.clip(np.digitize(E, BINS) - 1, 0, BINS.size - 2)
    out = {}
    for sw, m in (("anodic", dE > 0), ("cathodic", dE < 0), ("both", np.ones_like(dE, bool))):
        out[sw] = [[float(np.mean(r[m & (k == i)])), float(np.std(r[m & (k == i)]) / np.sqrt(max(np.sum(m & (k == i)), 1)))]
                   if np.any(m & (k == i)) else [np.nan, np.nan] for i in range(BINS.size - 1)]
    return out


res = {}
for n in SCANS41 + CHECK:
    for roi in ROIS:
        dd = data(n, roi)
        if dd is None:
            continue
        t, y, E = dd
        r = detrend(t, y)
        q1, q2, lin = stats(t, r, E)
        F = np.fft.rfft(r)
        null = []
        for _ in range(2000):
            ph = np.exp(1j * rng.uniform(0, 2 * np.pi, F.size)); ph[0] = 1
            if r.size % 2 == 0:
                ph[-1] = 1
            rs = np.fft.irfft(F * ph, n=r.size)
            null.append(stats(t, rs, E))
        null = np.array(null)
        hac = newey_west(t, r, E)
        p1 = float((1 + np.sum(null[:, 0] >= q1)) / (1 + len(null)))
        p2 = float((1 + np.sum(null[:, 1] >= q2)) / (1 + len(null)))
        res[f"{n} {roi}"] = dict(scan=n, roi=roi, film=SCANS[n][0], hkl=SCANS[n][1], n_points=int(t.size),
                                 lin=float(lin), lin_err=float(max(np.std(null[:, 2]), hac)), lin_err_surr=float(np.std(null[:, 2])),
                                 lin_err_hac=float(hac), p_E=p1, p_E_dir=p2, curve=curve(t, r, E),
                                 noise=float(np.std(r)))
json.dump(res, open(os.path.join(here, "weak41.json"), "w"), indent=1)
print(f"{'scan roi':12s} {'film':>4s} {'hkl':12s} {'lin 0->0.7 V':>16s} {'(surr / HAC)':>14s} {'p(E)':>7s} {'p(E,dir)':>8s}  point noise")
for k, v in res.items():
    print(f"{k:12s} {v['film']:>4s} {v['hkl']:12s} {100*v['lin']:+7.2f} +- {100*v['lin_err']:4.2f}% ({100*v['lin_err_surr']:.2f} / {100*v['lin_err_hac']:.2f}) "
          f"{v['p_E']:7.4f} {v['p_E_dir']:8.4f}  {100*v['noise']:.2f}%")
# combined over the 4.1 nm scans, per ROI: chi2 of the linear amplitudes
from math import erf
for roi in ROIS:
    z = [res[f"{n} {roi}"]["lin"] / res[f"{n} {roi}"]["lin_err"] for n in SCANS41 if f"{n} {roi}" in res]
    chi2 = float(np.sum(np.square(z)))
    from scipy.stats import chi2 as C
    print(f"4.1 nm combined {roi}: chi2 {chi2:.1f} for {len(z)} scans, p = {C.sf(chi2, len(z)):.3f}; z per scan {np.round(z, 1)}")

# figure: the three scans the user flagged, binned by the applied potential (drift removed), both sweeps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
COL = {"anodic": "#eb6834", "cathodic": "#9a998f"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
mid = 0.5 * (BINS[1:] + BINS[:-1])
fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True)
for j, n in enumerate((498, 503, 519)):
    for i, roi in enumerate(("sum", "ROI1")):
        ax = axs[i, j]; v = res[f"{n} {roi}"]
        for sw, dx in (("anodic", -0.008), ("cathodic", 0.008)):
            c = np.array(v["curve"][sw]); ax.errorbar(mid + dx, 100 * c[:, 0], yerr=100 * c[:, 1], fmt="o-", ms=4, lw=1,
                                                      color=COL[sw], label=f"{sw} sweep")
        ax.axhline(0, color=GRID, lw=0.9)
        ax.set_title(f"{n} ({v['hkl']}), {'Ru + Ti' if roi == 'sum' else 'ROI1 (Ru spot)'}", loc="left", fontsize=10, color=INK)
        ax.text(0.02, 0.97, f"0 to 0.7 V: {100*v['lin']:+.2f} ± {100*v['lin_err']:.2f} %\nshape-free p = {v['p_E']:.2f}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=MUTED)
        ax.set_ylim(-3.5, 3.5); ax.grid(axis="y", color=GRID, lw=0.5)
        if j == 0: ax.set_ylabel("signal minus drift (%)")
        if i == 1: ax.set_xlabel("E applied (V vs Ag/AgCl)")
axs[0, 2].legend(frameon=False, fontsize=8.5, loc="lower right")
fig.suptitle("4.1 nm scans 498, 503, 519 binned by the applied potential (bars: standard error of each bin, white noise)",
             x=0.01, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(here, "fig_weak41.png"), dpi=150, bbox_inches="tight")

"""Deep search for any response of the 4.1 nm film (day 4) to the EC control, in every data stream at hand.

Part A, the stationary CV scans (X-ray). Every channel:
  ROI sums (ROI1 Ru spot, ROI2 Ti spot, ROI3 / ROI4 background boxes, ROI5, ROI6, Ru + Ti, Ru / Ti ratio) and every SPEC
  counter that varies (Eiger1-4, Eiger_Max, APD, IONCHAMBER2, BPM3 sum and position, Monitor).
  Intensities are divided by the monitor (relative to its mean) and the transmission; the beam channels (Monitor,
  IONCHAMBER2, BPM3) are left raw and serve as negative controls (they should not follow the potential).
  Timing: every point's SPEC time, the potential EC-Lab applied at that time (clock offset from sync_abs.json).
  Tests on the CV part (after the 30 s hold, up to the logged Stop; 222 only its 5 mV/s part), quadratic drift removed:
    step   A of a step at 0.23 V (the 3.3 / 7 nm switch shape), change from 0 to 0.7 V
    lin    A of a straight line in E, change from 0 to 0.7 V
    dir    B of the sweep direction (anodic +1, cathodic -1): a hysteresis or charging-like term
    shape  shape-free: 7 E bins, both sweeps pooled (p-value only)
    ocv    the jump at the start of the EC sequence: mean of the 30 s hold at 0 V minus the mean of the last points before
           EC-Lab started (the cell at rest, 0.37-0.40 V on day 4)
  Errors: the larger of a residual-surrogate error (phase-randomised residual of the fitted model, 400 draws) and a
  Newey-West error; shape p from 1000 phase-randomised surrogates of the drift-only residual; ocv error from differences of
  adjacent windows of the same lengths along the detrended CV part.
Part B, the CTR rods (oh_rods_fit project): for each 4.1 nm rod, every pair of potentials: rms of ln(F1/F2) with a free
  scale (TiO2 Bragg peak +-0.08 left out), and the amplitude alpha of the change the CTR model predicts if the 4.1 nm
  surface switched like 7 nm (the 7 nm states at the two potentials, 0 A bridging-O variant), fitted to ln(F1/F2) with a free
  offset. alpha = 1: a 7 nm-sized switch; 0: none. The same on the 7 nm (0 1) rods as a check.
Part C, the EC record of day 4: the charge at every potential step (first two points minus the level after), per volt,
  against the 3.3 and 7 nm steps; the CV current terms (ec_leak.json).
Writes deep41.json and prints the 4.1 nm findings ranked by significance, with Benjamini-Hochberg q-values."""
import json, os, sys, glob
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "sync_abs.py")).read()
__file__ = os.path.join(here, "sync_abs.py")
exec(src.split('out = {"offset": {}')[0])                      # SCANS, SPEC, spec_times, ec_trace, shapes, hac, ...
import numpy as np
from math import erf, sqrt

D = json.load(open(os.path.join(here, "sync_abs.json")))["offset_used"]["d"]
rng = np.random.default_rng(7)
S41 = [496, 497, 498, 501, 503, 519]
REF = [120, 125, 147, 222, 223, 311, 314, 316, 341, 343, 344]
SPECCH = ["Eiger1", "Eiger2", "Eiger3", "Eiger4", "Eiger_Max", "APD", "IONCHAMBER2", "BPM3", "BPM3x", "BPM3y", "Monitor"]
BEAM = {"IONCHAMBER2", "BPM3", "BPM3x", "BPM3y", "Monitor"}
BINS = np.linspace(0, 0.7, 8)


def pnorm(z):
    return 1 - erf(abs(z) / sqrt(2))


def channels(n):
    """{name: (t from the EC start minus the offset, y relative)} for every channel of scan n, and E(t), the EC window."""
    film, hkl, ru, ti, folder, pat, a, b = SCANS[n]
    tab = parse_roi_sums(open(os.path.join(here, "data", f"ROIs_s{n}_sums.txt")).read())[n]
    sp = SPEC[n].data
    m = min(len(next(iter(tab.values()))), len(sp["Epoch"]))
    mon = sp["Monitor"][:m] / np.mean(sp["Monitor"][:m]); tr = sp["transm"][:m]
    ts = spec_times(n)[:m]
    start, te, Ee = ec_trace(folder, pat)
    t = np.array([(x - start).total_seconds() for x in ts]) - D
    out = {}
    for c in tab:
        if c in ("scan", "image") or c.endswith("_err"):
            continue
        out[c] = np.asarray(tab[c], float)[:m] / mon / tr
    out["Ru+Ti"] = out[ru] + out[ti]
    out["Ru/Ti"] = out[ru] / out[ti]
    for c in SPECCH:
        if c in ("BPM3x", "BPM3y"):
            A, B, C, Dd = (np.asarray(sp[k], float)[:m] for k in ("BPM3A", "BPM3B", "BPM3C", "BPM3D"))
            out[c] = 1 + ((A - C) / (A + C) if c == "BPM3x" else (B - Dd) / (B + Dd))
            continue
        if c not in sp:
            continue
        v = np.asarray(sp[c], float)[:m]
        out[c] = v if c in BEAM else v / mon / tr
    out = {k: v / np.mean(v) for k, v in out.items() if np.ptp(v) > 0}
    tmax = 730.0 if n == 222 else te.max()
    if b is not None:
        tmax = min(tmax, t[min(b, m - 1)])
    E = np.interp(t, te, Ee, left=np.nan)
    return t, out, E, te.min(), tmax


def design(t, E, extra):
    tt = (t - t.mean()) / np.ptp(t)
    return np.c_[np.ones_like(tt), tt, tt ** 2, *extra]


def fit_err(t, y, cols, k):
    """Coefficient k of the model (drift + cols), its error: max(residual surrogate, Newey-West)."""
    X = design(t, None, cols)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ c
    F = np.fft.rfft(u)
    boot = []
    for _ in range(400):
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, F.size)); ph[0] = 1
        if u.size % 2 == 0:
            ph[-1] = 1
        yb = X @ c + np.fft.irfft(F * ph, n=u.size)
        boot.append(np.linalg.lstsq(X, yb, rcond=None)[0][3 + k])
    lag = max(5, int(len(u) / 8))
    S = (X * u[:, None]).T @ (X * u[:, None])
    for l in range(1, lag + 1):
        G = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None])
        S += (1 - l / (lag + 1)) * (G + G.T)
    B = np.linalg.inv(X.T @ X)
    e_hac = float(np.sqrt((B @ S @ B)[3 + k, 3 + k]))
    return float(c[3 + k]), max(float(np.std(boot)), e_hac)


def shape_p(t, y, E):
    X = design(t, None, [])
    u = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    k = np.clip(np.digitize(E, BINS) - 1, 0, BINS.size - 2)
    Q = lambda r: sum(np.sum(k == i) * np.mean(r[k == i]) ** 2 for i in range(BINS.size - 1) if np.any(k == i)) / np.var(r)
    q0 = Q(u); F = np.fft.rfft(u); cnt = 0
    for _ in range(1000):
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, F.size)); ph[0] = 1
        if u.size % 2 == 0:
            ph[-1] = 1
        cnt += Q(np.fft.irfft(F * ph, n=u.size)) >= q0
    return float((1 + cnt) / 1001)


def ocv_jump(t, y, win, hold_end=30.0):
    pre = (t < -2) & (t > -60)
    hold = (t > 3) & (t < hold_end)
    if pre.sum() < 4 or hold.sum() < 4:
        return None
    d = float(np.mean(y[hold]) - np.mean(y[pre]))
    # null: adjacent windows of the same sizes along the detrended CV part
    tw, yw = t[win], y[win]
    X = design(tw, None, []); r = yw - X @ np.linalg.lstsq(X, yw, rcond=None)[0]
    n1, n2 = int(pre.sum()), int(hold.sum())
    diffs = [np.mean(r[i + n1:i + n1 + n2]) - np.mean(r[i:i + n1]) for i in range(0, r.size - n1 - n2)]
    e = float(np.std(diffs)) if len(diffs) > 10 else np.nan
    return d, e, n1, n2


partA = {}
for n in S41 + REF:
    t, ch, E, t0, tmax = channels(n)
    win = (t >= t0 + 30.5) & (t <= tmax) & np.isfinite(E)
    tw, Ew = t[win], E[win]
    dE = np.gradient(Ew, tw); sgn = np.sign(dE)
    g = shapes(Ew)["logistic"]
    for name, y in ch.items():
        yw = y[win]
        if yw.size < 40:
            continue
        r = {}
        a, e = fit_err(tw, yw, [g], 0); r["step"] = (a, e)
        a, e = fit_err(tw, yw, [Ew / 0.7], 0); r["lin"] = (a, e)
        a, e = fit_err(tw, yw, [Ew / 0.7, sgn], 1); r["dir"] = (a, e)
        r["shape_p"] = shape_p(tw, yw, Ew)
        oj = ocv_jump(t, y, win)
        if oj:
            r["ocv"] = (oj[0], oj[1])
        partA[f"{n}|{name}"] = dict(scan=n, film=SCANS[n][0], hkl=SCANS[n][1], channel=name, n=int(yw.size), **r)
    print("A", n, "done", flush=True)

# ------------------------------------------------------------------ part B: CTR rods
from ctrfit.project.project import Project
from ctrfit.fit.global_fit import film_of
p = Project.load(os.path.join(here, "..", "2026-11-05_oh_fits", "oh_rods_fit.ctrproj.json"))
m, P = p.model, p.model.params
lead = {}
for ds in p.datasets:
    lead.setdefault(film_of(ds).split("|")[0], ds.scope)
summ = json.load(open(os.path.join(here, "..", "2026-11-05_oh_fits", "oh_rods_fit_summary.json")))["results"]["B fixed 0"]


def st7(e):
    k = f"7 nm {e:g} V"
    if k not in summ["profiles"]:
        e = min((-0.1, 0.0, 0.15, 0.35, 0.7), key=lambda x: abs(x - e)); k = f"7 nm {e:g} V"
    return dict(theta=1.0, x_OH=summ["profiles"][k]["best"], dz_Obr=summ["dz_Obr"][f"{e:g} V"], z_H2O=summ["z_H2O"])


rods = {}
for ds in p.datasets:
    name = ds.name
    if not (name.startswith("4.1 nm") or name.startswith("7 nm")):
        continue
    H, K, L = (np.asarray(x, float) for x in (ds.H, ds.K, ds.L))
    F, s = np.asarray(ds.F, float), np.asarray(ds.sigma, float)
    cond = name.split(" (")[0]
    for hk in sorted(set(zip(np.round(H).astype(int), np.round(K).astype(int)))):
        sel = (np.round(H) == hk[0]) & (np.round(K) == hk[1])
        rods.setdefault((cond.split()[0] + " nm", hk), {})[cond] = (L[sel], F[sel], s[sel])


def potential(cond):
    if "OCV" in cond:
        return 0.37
    return float(cond.split("CA ")[1].split(" V")[0]) if "CA " in cond else None


partB = []
for (film, hk), conds in rods.items():
    names = sorted(conds)
    v = dict(P.view(lead[film]))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c1, c2 = names[i], names[j]
            L1, F1, s1 = conds[c1]; L2, F2, s2 = conds[c2]
            k1 = {round(x, 3): q for q, x in enumerate(L1)}
            idx = [(k1[round(x, 3)], q) for q, x in enumerate(L2) if round(x, 3) in k1]
            if len(idx) < 10:
                continue
            a_, b_ = np.array(idx).T
            L = L1[a_]
            ok = np.abs(L - np.round(L)) > 0.08
            ok &= (F1[a_] > 0) & (F2[b_] > 0)
            if hk[0] % 2 == 0 and hk[1] % 2 == 0:
                ok &= np.abs(L / 2 - np.round(L / 2)) * 2 > 0.08
            L, f1, f2 = L[ok], F1[a_][ok], F2[b_][ok]
            r = np.log(f1 / f2); r -= r.mean()
            rms = float(np.sqrt(np.mean(r ** 2)))
            e1, e2 = potential(c1), potential(c2)
            alpha = None
            if e1 is not None and e2 is not None and abs(e1 - e2) > 0.1:
                I1 = np.asarray(m.intensity(hk[0], hk[1], L, dict(v, **st7(e1))), float)
                I2 = np.asarray(m.intensity(hk[0], hk[1], L, dict(v, **st7(e2))), float)
                tpl = 0.5 * np.log(I1 / I2); tpl -= tpl.mean()            # ln F ratio predicted for a 7 nm-like switch
                if np.sum(tpl ** 2) > 0:
                    a = float(np.sum(tpl * r) / np.sum(tpl ** 2))
                    res = r - a * tpl
                    # error from the residual scatter, with neighbouring points correlated: inflate by sqrt(3)
                    ea = float(np.std(res) / np.sqrt(np.sum(tpl ** 2)) * np.sqrt(3))
                    alpha = (a, ea, float(np.sqrt(np.mean(tpl ** 2))))
            partB.append(dict(film=film, rod=f"({hk[0]} {hk[1]})", c1=c1, c2=c2, n=int(r.size), rms=rms, alpha=alpha))
print("B done", flush=True)

# ------------------------------------------------------------------ part C: charge at potential steps
from galvani import BioLogic
def steps_in(f):
    d = BioLogic.MPRfile(f).data
    t, E, I = d["time/s"], d["control/V"], d["<I>/mA"] * 1e3
    dt = np.median(np.diff(t))
    out = []
    jumps = np.flatnonzero(np.abs(np.diff(E)) > 0.02) + 1
    for j in jumps:
        if j + 12 >= t.size:
            continue
        base = np.mean(I[j + 4:j + 12])                     # the level after the step (2-6 s later at 0.5 s per point)
        q = float(np.sum(I[j:j + 2] - base) * dt)           # uC in the first two points
        out.append(dict(dE=float(E[j] - E[j - 1]), q=q, noise=float(np.std(I[j + 4:j + 12]) * dt * np.sqrt(2))))
    return out

partC = {}
for film, folder in (("3.3", "Collin_08102026"), ("7", "Collin_08112026"), ("4.1", "Collin_08122026")):
    rows = []
    for f in sorted(glob.glob(os.path.join(here, "data", "echem", folder, "*CA*.mpr"))):
        if "_CV_" in f:
            continue
        rows += steps_in(f)
    dE = np.array([r["dE"] for r in rows]); q = np.array([r["q"] for r in rows]); nz = np.array([r["noise"] for r in rows])
    if dE.size:
        C = float(np.sum(q * dE) / np.sum(dE ** 2))           # uC per V, least squares through 0
        eC = float(np.sqrt(np.sum((nz * dE) ** 2)) / np.sum(dE ** 2))
        partC[film] = dict(n_steps=int(dE.size), C_uF=C, err=eC, dE=dE.tolist(), q=q.tolist())
print("C done", flush=True)
json.dump(dict(partA=partA, partB=partB, partC=partC), open(os.path.join(here, "deep41.json"), "w"), indent=1, default=float)

# ------------------------------------------------------------------ report
tests = []
for k, v in partA.items():
    if v["film"] != "4.1":
        continue
    for tname in ("step", "lin", "dir", "ocv"):
        if tname in v and v[tname][1] and np.isfinite(v[tname][1]) and v[tname][1] > 0:
            z = v[tname][0] / v[tname][1]
            tests.append(dict(where=f"CV {v['scan']} ({v['hkl']})", channel=v["channel"], test=tname, est=v[tname][0],
                              err=v[tname][1], z=z, p=pnorm(z), beam=v["channel"] in BEAM))
    tests.append(dict(where=f"CV {v['scan']} ({v['hkl']})", channel=v["channel"], test="shape", est=np.nan, err=np.nan,
                      z=np.nan, p=v["shape_p"], beam=v["channel"] in BEAM))
for r in partB:
    if r["film"] == "4.1 nm" and r["alpha"]:
        a, e, size = r["alpha"]
        tests.append(dict(where=f"CTR {r['rod']} {r['c1']} vs {r['c2']}", channel="F", test="alpha (7 nm-like switch)",
                          est=a, err=e, z=a / e, p=pnorm(a / e), beam=False))
if "4.1" in partC:
    c = partC["4.1"]
    tests.append(dict(where=f"EC steps day 4 ({c['n_steps']})", channel="current", test="charge per volt (uF)",
                      est=c["C_uF"], err=c["err"], z=c["C_uF"] / c["err"], p=pnorm(c["C_uF"] / c["err"]), beam=False))
xs = [t for t in tests if not t["beam"]]
ps = np.array([t["p"] for t in xs]); order = np.argsort(ps); mtot = len(ps)
q = np.empty(mtot); prev = 1.0
for rank, i in enumerate(order[::-1]):
    k = mtot - rank
    prev = min(prev, ps[i] * mtot / k); q[i] = prev
for t, qq in zip(xs, q):
    t["q"] = float(qq)
json.dump(dict(partA=partA, partB=partB, partC=partC, tests_41=tests), open(os.path.join(here, "deep41.json"), "w"), indent=1, default=float)
print(f"\n4.1 nm tests (beam controls left out): {mtot}; expected by chance at p < 0.01: {0.01*mtot:.1f}, p < 0.001: {0.001*mtot:.2f}")
print(f"observed p < 0.01: {int(np.sum(ps < 0.01))}, p < 0.001: {int(np.sum(ps < 0.001))}")
for i in order[:30]:
    t = xs[i]
    est = "" if not np.isfinite(t["est"]) else f"{t['est']:+.4f} +- {t['err']:.4f} (z {t['z']:+.1f})"
    print(f"  p {t['p']:.2g}  q {t['q']:.2g}  {t['where']:34s} {t['channel']:12s} {t['test']:24s} {est}")
bt = [t for t in tests if t["beam"]]
pb = np.array([t["p"] for t in bt])
print(f"\nbeam controls: {len(pb)} tests, p < 0.01: {int(np.sum(pb < 0.01))}, p < 0.05: {int(np.sum(pb < 0.05))} (expected {0.05*len(pb):.1f})")
for t in sorted(bt, key=lambda t: t["p"])[:5]:
    print(f"  p {t['p']:.2g}  {t['where']:34s} {t['channel']:12s} {t['test']}")
print("\nreference films, channels with a step response (|z| > 4):")
for k, v in partA.items():
    if v["film"] != "4.1" and "step" in v and abs(v["step"][0] / v["step"][1]) > 4:
        print(f"  {v['scan']} {v['film']} nm {v['channel']:12s} step {100*v['step'][0]:+.2f} +- {100*v['step'][1]:.2f} %")
print("\nCTR alpha, 7 nm check:")
for r in partB:
    if r["film"] == "7 nm" and r["alpha"]:
        print(f"  {r['rod']} {r['c1']} vs {r['c2']}: rms {r['rms']:.3f}, alpha {r['alpha'][0]:+.2f} +- {r['alpha'][1]:.2f}")
print("CTR 4.1 nm:")
for r in partB:
    if r["film"] == "4.1 nm":
        a = f"alpha {r['alpha'][0]:+.2f} +- {r['alpha'][1]:.2f} (template rms {r['alpha'][2]:.3f})" if r["alpha"] else ""
        print(f"  {r['rod']} {r['c1']} vs {r['c2']}: n {r['n']}, rms {r['rms']:.3f} {a}")
print("\nEC steps, charge per volt (uF):", {k: (v["n_steps"], round(v["C_uF"], 3), round(v["err"], 3)) for k, v in partC.items()})

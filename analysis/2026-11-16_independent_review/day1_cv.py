"""Day 1 (3.3 nm, before the cell was rebuilt): the two stationary CV loopscans 91 and 94 at (0 1 2.73), SPEC counters only.

Data and timing helpers come from the earlier analysis (../2026-11-06_stationary_cv: its data folder, sync_abs.py and
sync_abs.json), which this script only reads. No ROI sums exist for these scans, so the detector ROIs SPEC kept during the
scan (Eiger1-4, Eiger_Max) are used, each divided by the monitor (relative to its mean); the Monitor itself is the negative
control. Timing as in sync_abs.py: SPEC point time = #D + Epoch, EC-Lab time = file timestamp + time/s, offset from
sync_abs.json. The day-1 EC files are single CV files (0 to 0.7 V at 5 mV/s, no 30 s hold).
Per channel:
  - the clock offset that best aligns the drift-removed signal with E (-60 to +60 s), a check on the timing;
  - A, the change from 0 to 0.7 V of the logistic switch shape (sync_abs.py), with the larger of a residual-surrogate and a
    Newey-West error;
  - p against 600 fake triangle potentials (as deep41b.py), with the beam signals as covariates;
  - for Eiger1, the half-way potential as in curves_sync.py (14 bins, sweeps averaged, first bin 0 %, plateau 100 %).
Writes day1_cv.json here."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.normpath(os.path.join(HERE, "..", "2026-11-06_stationary_cv"))
src = open(os.path.join(OLD, "sync_abs.py")).read()
__file__ = os.path.join(OLD, "sync_abs.py")
exec(src.split('out = {"offset": {}')[0])                      # SPEC, spec_times, ec_trace, shapes, fit, hac
import numpy as np

D = json.load(open(os.path.join(OLD, "sync_abs.json")))["offset_used"]["d"]
rng = np.random.default_rng(5)
DAY1 = {91: ("*timescan_91_C01*", 69), 94: ("*RuPeak_scan94_C01*", 47)}   # EC file, logged CV start point
CH = ["Eiger1", "Eiger2", "Eiger3", "Eiger4", "Eiger_Max", "Monitor"]
COV = ["IONCHAMBER2", "BPM3"]
BINS = np.linspace(0, 0.7, 15)
MID = 0.5 * (BINS[1:] + BINS[:-1])


def tri(t, P, ph):
    x = ((t / P) + ph) % 1.0
    return np.where(x < 0.5, 1.4 * x, 1.4 * (1 - x))


def fake_p(t, y, E, cov, n_fake=600):
    tt = (t - t.mean()) / np.ptp(t)
    base = [np.ones_like(tt), tt, tt ** 2] + [c - c.mean() for c in cov]
    a = np.linalg.lstsq(np.c_[tuple(base + [shapes(E)["logistic"]])], y, rcond=None)[0][-1]
    f = []
    while len(f) < n_fake:
        Ef = tri(t, rng.uniform(180, 420), rng.uniform(0, 1))
        if abs(np.corrcoef(Ef, E)[0, 1]) > 0.3:
            continue
        f.append(np.linalg.lstsq(np.c_[tuple(base + [shapes(Ef)["logistic"]])], y, rcond=None)[0][-1])
    f = np.array(f)
    return float(a), float((1 + np.sum(np.abs(f) >= abs(a))) / (1 + f.size))


def halfway(t, y, E):
    dE = np.gradient(E, t)
    cs = []
    for m in (dE > 0, dE < 0):
        k = np.digitize(E[m], BINS) - 1
        cs.append([np.mean(y[m][k == i]) if np.any(k == i) else np.nan for i in range(MID.size)])
    c = np.nanmean(cs, axis=0)
    f = (c - c[0]) / (np.nanmean(c[-3:]) - c[0])
    for i in range(1, f.size):
        if (f[i - 1] - 0.5) * (f[i] - 0.5) <= 0 and f[i] != f[i - 1]:
            return float(MID[i - 1] + (0.5 - f[i - 1]) / (f[i] - f[i - 1]) * (MID[i] - MID[i - 1]))
    return None


out = {"offset_s": D, "scans": {}}
for n, (pat, start_pt) in DAY1.items():
    sp = SPEC[n].data
    ts = spec_times(n)
    start, te, Ee = ec_trace("Collin_08092026", pat)
    t_raw = np.array([(x - start).total_seconds() for x in ts])
    mon = np.asarray(sp["Monitor"], float) / np.mean(sp["Monitor"])
    r = dict(hkl=" ".join(f"{float(v):.3f}" for v in SPEC[n].hkl), transm=float(np.mean(sp["transm"])),
             start_mark_s=float((start - ts[start_pt]).total_seconds()), channels={})
    t = t_raw - D
    E = np.interp(t, te, Ee, left=np.nan, right=np.nan)
    ok = np.isfinite(E)
    cov = [np.asarray(sp[c], float)[ok] / np.mean(sp[c]) for c in COV]
    for ch in CH:
        y_all = np.asarray(sp[ch], float) / (mon if ch != "Monitor" else 1.0)
        best = (0.0, 0.0)                                           # own clock offset
        for d in np.arange(-60, 61, 1.0):
            Ed = np.interp(t_raw - d, te, Ee, left=np.nan, right=np.nan)
            m = np.isfinite(Ed)
            _, u = fit(t_raw[m], y_all[m] / np.mean(y_all[m]), np.zeros(m.sum()))
            cc = np.corrcoef(u, Ed[m])[0, 1]
            if abs(cc) > abs(best[1]):
                best = (float(d), float(cc))
        tt, EE = t[ok], E[ok]
        yy = y_all[ok] / np.mean(y_all[ok][EE < 0.05])
        g = shapes(EE)["logistic"]
        A, u = fit(tt, yy, g)
        F = np.fft.rfft(u)
        boot = []
        for _ in range(400):
            ph = np.exp(1j * rng.uniform(0, 2 * np.pi, F.size)); ph[0] = 1
            if u.size % 2 == 0:
                ph[-1] = 1
            boot.append(fit(tt, (yy - u) + np.fft.irfft(F * ph, n=u.size), g)[0])
        err = max(float(np.std(boot)), hac(tt, u, g))
        a_cov, p = fake_p(tt, yy, EE, cov if ch != "Monitor" else [])
        r["channels"][ch] = dict(A=float(A), err=err, z=float(A / err), own_offset_s=best[0], own_corr=best[1],
                                 A_beam_cov=a_cov, p_fake=p)
        if ch == "Eiger1":
            r["halfway_V"] = halfway(tt, yy, EE)
            r["n_points"] = int(tt.size); r["cycles"] = float(np.ptp(tt) / 280.0)
    out["scans"][n] = r
json.dump(out, open(os.path.join(HERE, "day1_cv.json"), "w"), indent=1)
for n, r in out["scans"].items():
    print(f"scan {n} ({r['hkl']}): transm {r['transm']:g}, EC start minus logged CV start {r['start_mark_s']:+.1f} s, "
          f"{r['n_points']} points, {r['cycles']:.1f} cycles, half-way (Eiger1) {r['halfway_V']:.3f} V")
    for ch, v in r["channels"].items():
        print(f"   {ch:9s} A = {100 * v['A']:+6.2f} +- {100 * v['err']:.2f} % (z {v['z']:+.1f}); own offset {v['own_offset_s']:+.0f} s "
              f"(corr {v['own_corr']:+.2f}); fake-potential p {v['p_fake']:.3f}")

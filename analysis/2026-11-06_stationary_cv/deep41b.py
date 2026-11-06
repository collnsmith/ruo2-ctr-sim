"""Stage 2 of the deep search (deep41.py): calibrated significance.

Stage 1 showed that the beam channels (Monitor, BPM3, BPM3x/y), which cannot follow the cell potential, pass the
potential-locking tests far more often than chance (p < 0.01 in 14 of 150 tests, 1.5 expected): with about two cycles
per scan, beam glitches and drifts line up with the potential cycle by chance, and a Gaussian error model misses it.
So here:
  1. Beam covariates: every channel is fitted with the drift plus the beam signals (Monitor, BPM3 sum, BPM3x, BPM3y,
     IONCHAMBER2; a beam channel is not used to explain itself).
  2. Fake-potential null: the same fit with E(t) replaced by 600 fake triangle waves (0 to 0.7 V, period 180-420 s,
     random phase) that are unrelated to the real one (|corr| < 0.3 for E and for the sweep direction). The p-value is
     the fraction of fakes with an equal or larger effect. It carries the data's own glitches, drift and correlations.
  3. The same is run on the 3.3 / 7 nm scans (positive check), on the beam channels (negative check: their p-values
     should now be uniform), and on the day-4 EC current.
Terms: step (switch shape at 0.23 V), lin (straight line in E), dir (sweep direction).
Writes deep41b.json and prints the ranked 4.1 nm findings with Benjamini-Hochberg q-values."""
import json, os, sys, glob
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "deep41.py")).read()
__file__ = os.path.join(here, "deep41.py")
exec(src.split("partA = {}")[0])                               # channels(), shapes(), SCANS, D, ...
import numpy as np
from galvani import BioLogic

rng = np.random.default_rng(11)
COV = ["Monitor", "BPM3", "BPM3x", "BPM3y", "IONCHAMBER2"]
NFAKE = 600


def tri(t, P, ph):
    x = ((t / P) + ph) % 1.0
    return np.where(x < 0.5, 1.4 * x, 1.4 * (1 - x))           # 0 to 0.7 V


def regressors(E, t):
    dE = np.gradient(E, t)
    return {"step": shapes(E)["logistic"], "lin": E / 0.7, "dir": np.sign(dE)}


def coef(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def test_series(t, y, E, cov):
    """Real coefficients for step / lin / dir (each in its own model) and fake-null p-values."""
    tt = (t - t.mean()) / np.ptp(t)
    base = [np.ones_like(tt), tt, tt ** 2] + [c - c.mean() for c in cov]
    real = regressors(E, t)
    out = {}
    fakes = []
    while len(fakes) < NFAKE:
        P = rng.uniform(180, 420); ph = rng.uniform(0, 1)
        Ef = tri(t, P, ph)
        if abs(np.corrcoef(Ef, E)[0, 1]) > 0.3:
            continue
        rf = regressors(Ef, t)
        if abs(np.corrcoef(rf["dir"], real["dir"])[0, 1]) > 0.3:
            continue
        fakes.append(rf)
    for k in ("step", "lin", "dir"):
        X = np.c_[tuple(base + [real[k]])]
        a = coef(X, y)[-1]
        f = np.array([coef(np.c_[tuple(base + [rf[k]])], y)[-1] for rf in fakes])
        out[k] = dict(est=float(a), null_sd=float(np.std(f)), p=float((1 + np.sum(np.abs(f) >= abs(a))) / (1 + f.size)))
    return out


res = {}
for n in S41 + REF:
    t, ch, E, t0, tmax = channels(n)
    win = (t >= t0 + 30.5) & (t <= tmax) & np.isfinite(E)
    tw, Ew = t[win], E[win]
    for name, y in ch.items():
        yw = y[win]
        if yw.size < 40:
            continue
        cov = [ch[c][win] for c in COV if c in ch and c != name]
        for tag, cv in (("plain", []), ("beam-cov", cov)):
            r = test_series(tw, yw, Ew, cv)
            for k, v in r.items():
                res[f"{n}|{name}|{tag}|{k}"] = dict(scan=n, film=SCANS[n][0], hkl=SCANS[n][1], channel=name, model=tag, term=k,
                                                    beam=name in BEAM, **v)
    print("scan", n, "done", flush=True)

# day-4 EC current: the same terms against fake potentials, per CV file (6 files)
ec = {}
for f in sorted(glob.glob(os.path.join(here, "data", "echem", "Collin_08122026", "*_02_CV_C01.mpr"))):
    d = BioLogic.MPRfile(f).data
    t, E, I = np.asarray(d["time/s"], float), np.asarray(d["control/V"], float), np.asarray(d["<I>/mA"], float) * 1e6  # nA
    r = test_series(t, I, E, [])
    ec[os.path.basename(f)] = r
json.dump(dict(xray=res, ec_day4=ec), open(os.path.join(here, "deep41b.json"), "w"), indent=1)

# ------------------------------------------------------------------ report
def bh(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m_ = ps.size; q = np.empty(m_); prev = 1.0
    for rank, i in enumerate(o[::-1]):
        prev = min(prev, ps[i] * m_ / (m_ - rank)); q[i] = prev
    return q

for model in ("plain", "beam-cov"):
    beam = [v["p"] for v in res.values() if v["film"] == "4.1" and v["beam"] and v["model"] == model]
    xr = [v for v in res.values() if v["film"] == "4.1" and not v["beam"] and v["model"] == model]
    ps = np.array([v["p"] for v in xr]); q = bh(ps)
    beam = np.array(beam)
    print(f"\n=== 4.1 nm, model '{model}': {len(xr)} X-ray tests; p<0.05 {np.sum(ps<0.05)} (exp {0.05*len(ps):.1f}), "
          f"p<0.01 {np.sum(ps<0.01)} (exp {0.01*len(ps):.1f}), min q {q.min():.2f}")
    print(f"    beam controls: {beam.size} tests; p<0.05 {np.sum(beam<0.05)} (exp {0.05*beam.size:.1f}), p<0.01 {np.sum(beam<0.01)}")
    for i in np.argsort(ps)[:15]:
        v = xr[i]
        print(f"    p {v['p']:.3f}  q {q[i]:.2f}  {v['scan']} ({v['hkl']}) {v['channel']:12s} {v['term']:5s} "
              f"{100*v['est']:+.3f} % (fake sd {100*v['null_sd']:.3f} %)")
print("\n=== positive check: reference films, model 'beam-cov', Ru+Ti / Ru ROI, step term")
for v in res.values():
    if v["film"] != "4.1" and v["model"] == "beam-cov" and v["term"] == "step" and v["channel"] in ("Ru+Ti",):
        print(f"    {v['scan']} {v['film']} nm ({v['hkl']}) {v['channel']:6s} {100*v['est']:+.2f} %  fake sd {100*v['null_sd']:.2f} %  p {v['p']:.4f}")
print("\n=== day-4 EC current (nA) against fake potentials")
for f, r in ec.items():
    print("   ", f[-28:], {k: f"{v['est']:+.2f} (fake sd {v['null_sd']:.2f}, p {v['p']:.3f})" for k, v in r.items()})

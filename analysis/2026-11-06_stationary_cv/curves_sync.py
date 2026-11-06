"""CV curves and half-way potentials with the clock-synced potential (sync_abs.py), against the earlier timing (cv3.json).

For each OH-sensitive CV scan: the X-ray signal relative to the 30 s hold at 0 V, binned in E (14 bins, 0-0.7 V) per
sweep, with the potential E(t) that EC-Lab applied at each point's SPEC time (offset from sync_abs.json). The sweeps are
averaged, each scan's curve is scaled from its first bin (0 %) to its plateau (mean of the last three bins, 100 %), and the
half-way potential is where it crosses 50 %. Done with the raw signal (as cv3 / slide 12b) and with a quadratic drift
removed. Writes curves_sync.json."""
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "sync_abs.py")).read()
__file__ = os.path.join(here, "sync_abs.py")
exec(src.split('out = {"offset": {}')[0])                      # helpers: series, SCANS, ...
import numpy as np

D = json.load(open(os.path.join(here, "sync_abs.json")))["offset_used"]["d"]
BINS = np.linspace(0, 0.7, 15)
MID = 0.5 * (BINS[1:] + BINS[:-1])
USE = {"3.3": [120, 222, 223], "7": [311, 314, 341, 343, 344], "4.1": [496, 497, 498]}   # L 2.75 and 3.16, as slide 12b
EXTRA = [125, 316, 501, 503, 147]

def curve(n, detrend):
    t, y, E, *_ = series(n, D)
    if detrend:
        tt = (t - t.mean()) / np.ptp(t)
        X = np.c_[np.ones_like(tt), tt, tt ** 2]
        g = shapes(E)["logistic"]
        c, *_ = np.linalg.lstsq(np.c_[X, g], y, rcond=None)
        y = y - X[:, 1:] @ c[1:3]                                # remove the drift, keep the response
    dE = np.gradient(E, t)
    out = {}
    for sw, m in (("anodic", dE > 0), ("cathodic", dE < 0)):
        k = np.digitize(E[m], BINS) - 1
        out[sw] = [float(np.mean(y[m][k == i]) - 1) if np.any(k == i) else np.nan for i in range(MID.size)]
    out["mean"] = list(np.nanmean([out["anodic"], out["cathodic"]], axis=0))
    return out

def halfway(c):
    c = np.asarray(c, float)
    f = (c - c[0]) / (np.nanmean(c[-3:]) - c[0])
    for i in range(1, f.size):
        if (f[i - 1] - 0.5) * (f[i] - 0.5) <= 0 and f[i] != f[i - 1]:
            return float(MID[i - 1] + (0.5 - f[i - 1]) / (f[i] - f[i - 1]) * (MID[i] - MID[i - 1])), f
    return None, f

old = json.load(open(os.path.join(here, "cv3.json")))
res = {"offset_s": D, "scans": {}, "films": {}}
for film, ns in list(USE.items()) + [("extra", EXTRA)]:
    for n in ns:
        r = {}
        for tag, det in (("raw", False), ("detrended", True)):
            c = curve(n, det)
            h, f = halfway(c["mean"])
            r[tag] = dict(curve=c, halfway=h, fraction=[float(x) for x in f])
        if str(n) in old:
            co = np.mean([old[str(n)]["sum"]["curves"][b]["dI"] for b in ("anodic", "cathodic")], axis=0)
            r["halfway_old_timing"] = halfway(co)[0]
        res["scans"][n] = r
for film, ns in USE.items():
    if film == "4.1":
        continue
    for tag in ("raw", "detrended"):
        fr = np.mean([res["scans"][n][tag]["fraction"] for n in ns], axis=0)
        h = None
        for i in range(1, fr.size):
            if (fr[i - 1] - 0.5) * (fr[i] - 0.5) <= 0:
                h = float(MID[i - 1] + (0.5 - fr[i - 1]) / (fr[i] - fr[i - 1]) * (MID[i] - MID[i - 1])); break
        res["films"].setdefault(film, {})[tag] = dict(halfway=h, scans=[res["scans"][n][tag]["halfway"] for n in ns],
                                                      fraction=[float(x) for x in fr])
json.dump(res, open(os.path.join(here, "curves_sync.json"), "w"), indent=1, default=float)
for n, r in res["scans"].items():
    print(n, "half-way: raw %s  detrended %s  old timing %s" % tuple(
        ("%.3f" % v) if isinstance(v, float) else "-" for v in (r["raw"]["halfway"], r["detrended"]["halfway"], r.get("halfway_old_timing"))))
for film, v in res["films"].items():
    print(film, {k: (round(x["halfway"], 3) if x["halfway"] else None, [round(s, 3) if s else None for s in x["scans"]]) for k, x in v.items()})

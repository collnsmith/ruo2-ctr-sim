"""Timing-free CV amplitudes: the best sinusoid over periods 60-130 points (phase free) and the 5-95 % range of the
5-point running median, for Ru+Ti and Ti alone, against the point-to-point noise. Also the same on the CA segment
before the CV (no potential change) and on a 30-point block shuffle of the data, as null references."""
import json, sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
import numpy as np
exec(open(sys.argv[4]).read().split("rows = info()")[0])      # info, load from extract.py
rng = np.random.default_rng(1)

def best_sine(y, periods=np.arange(60, 131, 1.0)):
    n = np.arange(y.size, dtype=float); best = 0.0
    for P in periods:
        A = np.column_stack([np.ones(y.size), np.sin(2 * np.pi * n / P), np.cos(2 * np.pi * n / P), n / y.size])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        best = max(best, 2 * np.hypot(c[1], c[2]))           # peak-to-peak
    return best

def rmed(y, k=5):
    return np.array([np.median(y[max(0, i - k // 2): i + k // 2 + 1]) for i in range(y.size)])

out = {}
for r in info():
    ru, ti, sru, sti = load(r)
    y = ru + ti
    a, b = r["start"], min(r["stop"], y.size - 1)
    rel = {"sum": y / y[a:a + 6].mean(), "Ti": ti / ti[a:a + 6].mean(), "Ru": ru / ru[a:a + 6].mean()}
    row = {}
    for k, s in rel.items():
        seg = s[a:b]
        noise = np.std(np.diff(seg)) / np.sqrt(2)
        m = rmed(seg)
        blocks = [seg[i:i + 30] for i in range(0, seg.size, 30)]
        shuf = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        row[k] = dict(sine_pp=best_sine(seg), range=float(np.percentile(m, 95) - np.percentile(m, 5)), noise=float(noise),
                      shuffled_sine_pp=float(np.median([best_sine(np.concatenate([blocks[i] for i in rng.permutation(len(blocks))]))
                                                         for _ in range(5)])))
    out[r["scan"]] = row
    s, t = row["sum"], row["Ti"]
    print(f"{r['scan']} {r['film']:>3s} nm ({r['H']:2d} {r['K']:2d} {r['L']:4.2f}) sum: sine p-p {100*s['sine_pp']:5.1f} %, "
          f"range {100*s['range']:5.1f} %, noise/pt {100*s['noise']:4.1f} %, block-shuffled sine {100*s['shuffled_sine_pp']:4.1f} % | "
          f"Ti sine p-p {100*t['sine_pp']:4.1f} % | Ru sine p-p {100*row['Ru']['sine_pp']:5.1f} %")
json.dump(out, open(sys.argv[3], "w"), indent=1, default=float)

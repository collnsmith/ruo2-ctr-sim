"""Second batch: controls 126 (-2 0 1.75) and 127 (-2 0 1, 25 mV/s), the CV repeat 223 (0 -1 3.16), and the potential
staircases 224 (0 -1 3.16) and 225 (0 -1 2.75): 15 levels 0, 0.05, ..., 0.7 V, 90 s each, from image 50 to the end."""
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from ctrfit.echem.external import parse_roi_sums
from ctrfit.echem.analysis import dominant_period
def load(s, cols):
    t = parse_roi_sums(open(os.path.join(here, "data", f"ROIs_s{s}_sums.txt")).read())[s]
    return sum(np.asarray(t[c], float) for c in cols)
def best_sine(y, periods):
    n = np.arange(y.size, dtype=float); best = (0.0, None)
    for P in periods:
        A = np.column_stack([np.ones(y.size), np.sin(2 * np.pi * n / P), np.cos(2 * np.pi * n / P), n / y.size])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        if 2 * np.hypot(c[1], c[2]) > best[0]: best = (2 * np.hypot(c[1], c[2]), P)
    return best
out = {}
# controls and the repeat: timing-free amplitude, same method as robust_cv.py
for s, cols, a, b, periods, lab in ((126, ("ROI1", "ROI2"), 50, 227, np.arange(60, 131.0), "3.3 nm (-2 0 1.75), 5 mV/s"),
                                    (127, ("ROI1", "ROI2"), 31, 78, np.arange(8, 41.0), "3.3 nm (-2 0 1), 25 mV/s"),
                                    (223, ("ROI1",), 51, 410, np.arange(60, 131.0), "3.3 nm (0 -1 3.16), 5 mV/s, repeat of 222")):
    y = load(s, cols); seg = y[a:b] / y[a:a + 6].mean()
    pp, P = best_sine(seg, periods)
    noise = np.std(np.diff(seg)) / np.sqrt(2)
    out[s] = dict(label=lab, sine_pp=float(pp), period=float(P), noise=float(noise))
    print(f"{s} {lab:42s}: best sine p-p {100*pp:5.1f} % at period {P:.0f} images, noise/pt {100*noise:.1f} %")
# 223: the same timing and linear-change analysis as extract2.py (CA 30 s at 0 V, then cycles)
y = load(223, ("ROI1",)); a, b = 51, 410; n = np.arange(y.size, dtype=float)
P0 = dominant_period(n[a:b], y[a:b], min_period=40); FCA = 30 / 280
N = round(2 * ((b - a) / P0 - FCA)) / 2; P = (b - a) / (N + FCA)
best = (0, 0)
for d in np.linspace(-0.15, 0.15, 61) * P:
    ph = ((n - (a + FCA * P + d)) / P) % 1; E = np.where(ph < 0.5, 1.4 * ph, 0.7 - 1.4 * (ph - 0.5)); E[n < a + FCA * P + d] = 0
    c = np.corrcoef(E[a:b], y[a:b])[0, 1]
    if abs(c) > abs(best[1]): best = (d, c)
d = best[0]; ph = ((n - (a + FCA * P + d)) / P) % 1; E = np.where(ph < 0.5, 1.4 * ph, 0.7 - 1.4 * (ph - 0.5))
k = np.arange(int(a + FCA * P + d), b); rel = y / y[a:a + 6].mean()
A = np.column_stack([np.ones(k.size), E[k]]); c, *_ = np.linalg.lstsq(A, rel[k], rcond=None)
res = rel[k] - A @ c; cov = np.linalg.inv(A.T @ A) * np.var(res)
out[223].update(cycles=float(N), P=float(P), shift=float(d), change=float(c[1] * 0.7), change_err=float(np.sqrt(cov[1, 1]) * 0.7))
print(f"223 timing: {N:g} cycles, P {P:.1f} images, shift {d:+.1f}; change 0 -> 0.7 V {100*c[1]*0.7:+.1f} +- {100*np.sqrt(cov[1,1])*0.7:.1f} %  (222: +12.7 %)")
# staircases: mean per level, levels assumed equal in length from image 50 to the last good image
for s, last, lab in ((224, 448, "(0 -1 3.16)"), (225, 427, "(0 -1 2.75)")):
    y = load(s, ("ROI1",)); base = y[20:50].mean()
    edges = np.linspace(50, last + 1, 16).round().astype(int)
    lev = [float(y[e0 + 3:e1].mean() / base - 1) for e0, e1 in zip(edges[:-1], edges[1:])]   # skip 3 images after each step
    err = [float(y[e0 + 3:e1].std() / np.sqrt(e1 - e0 - 3) / base) for e0, e1 in zip(edges[:-1], edges[1:])]
    pre = float(y[:50].mean() / base - 1)
    out[s] = dict(label=lab, levels_E=[round(0.05 * i, 2) for i in range(15)], change=lev, err=err, images_per_level=float((last + 1 - 50) / 15))
    print(f"{s} {lab} staircase ({(last + 1 - 50) / 15:.1f} images per level): " + " ".join(f"{100*v:+.0f}" for v in lev) + " %")
json.dump(out, open(os.path.join(here, "process_new.json"), "w"), indent=1)

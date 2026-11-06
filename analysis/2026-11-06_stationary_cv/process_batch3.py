"""Third batch: 227 (3.3 nm, -1 0 2.32), 341 and 343 (7 nm, 0 1 2.75), 519 (4.1 nm, -2 0 1.75). Integrated signal = the
non-overlapping signal ROIs (sub-ROIs inside another ROI left out). Timing-free amplitude as robust_cv.py; for the
responding scans also the linear change 0 -> 0.7 V with the timing found as in extract2.py."""
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ctrfit.echem.external import parse_roi_sums
from ctrfit.echem.analysis import dominant_period
def load(s):
    return parse_roi_sums(open(os.path.join(here, "data", f"ROIs_s{s}_sums.txt")).read())[s]
def best_sine(y, periods):
    n = np.arange(y.size, dtype=float); best = (0.0, None)
    for P in periods:
        A = np.column_stack([np.ones(y.size), np.sin(2 * np.pi * n / P), np.cos(2 * np.pi * n / P), n / y.size])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        if 2 * np.hypot(c[1], c[2]) > best[0]: best = (2 * np.hypot(c[1], c[2]), P)
    return best
def linear_change(y, a, b):
    n = np.arange(y.size, dtype=float); FCA = 30 / 280
    P0 = dominant_period(n[a:b], y[a:b], min_period=40)
    if not P0: return None
    N = max(0.5, round(2 * ((b - a) / P0 - FCA)) / 2); P = (b - a) / (N + FCA)
    def E_of(d):
        ph = ((n - (a + FCA * P + d)) / P) % 1
        return np.where(ph < 0.5, 1.4 * ph, 0.7 - 1.4 * (ph - 0.5))
    d = max(np.linspace(-0.15, 0.15, 61) * P, key=lambda d: abs(np.corrcoef(E_of(d)[a:b], y[a:b])[0, 1]))
    E = E_of(d); k = np.arange(int(a + FCA * P + d), b); rel = y / y[a:a + 6].mean()
    A = np.column_stack([np.ones(k.size), E[k]]); c, *_ = np.linalg.lstsq(A, rel[k], rcond=None)
    cov = np.linalg.inv(A.T @ A) * np.var(rel[k] - A @ c)
    return dict(cycles=float(N), P=float(P), shift=float(d), change=float(0.7 * c[1]), change_err=float(0.7 * np.sqrt(cov[1, 1])))
CASES = [  # scan, label, signal ROIs, (Ru, Ti) or None, start, last good image
    (227, "3.3 nm (-1 0 2.32)", ("ROI1",), None, 25, 341),
    (341, "7 nm (0 1 2.75)", ("ROI1", "ROI2"), ("ROI2", "ROI1"), 21, 169),
    (343, "7 nm (0 1 2.75)", ("ROI1", "ROI2"), ("ROI2", "ROI1"), 20, 143),   # signals collapse after image 143
    (519, "4.1 nm (-2 0 1.75)", ("ROI1",), None, 20, 242)]
out = {}
fig, axs = plt.subplots(len(CASES), 1, figsize=(11, 9))
for ax, (s, lab, sig, rt, a, b) in zip(axs, CASES):
    t = load(s); n_img = len(t["ROI1"])
    imgs = np.asarray(t["image"]); dup = int(np.sum(imgs[1:] == imgs[:-1]))
    y = sum(np.asarray(t[c], float) for c in sig)
    seg = y[a:b] / y[a:a + 6].mean()
    pp, P = best_sine(seg, np.arange(60, 131.0))
    noise = np.std(np.diff(seg)) / np.sqrt(2)
    r = dict(label=lab, images=n_img, repeated_image_index=dup, sine_pp=float(pp), period=float(P), noise=float(noise))
    line = f"{s} {lab:20s} ({n_img} images, {dup} repeated index): sum sine p-p {100*pp:5.1f} % (period {P:.0f}), noise/pt {100*noise:.1f} %"
    if rt:
        for name, c in zip(("Ru", "Ti"), rt):
            v = np.asarray(t[c], float)[a:b]; v = v / v[:6].mean()
            r[f"{name}_sine_pp"] = float(best_sine(v, np.arange(60, 131.0))[0])
        line += f" | Ru {100*r['Ru_sine_pp']:.1f} %, Ti {100*r['Ti_sine_pp']:.1f} %"
        lc = linear_change(y, a, b + 1)
        if lc: r.update(lc); line += f" | {lc['cycles']:g} cycles, P {lc['P']:.0f}, change 0->0.7 V {100*lc['change']:+.1f} +- {100*lc['change_err']:.1f} %"
    print(line); out[s] = r
    x = np.arange(len(y)); m = min(len(y), b + 10)
    ax.plot(x[:m], y[:m] / y[a:a + 6].mean(), "k-", lw=0.9, label="integrated")
    if rt:
        for name, c, col in (("Ru", rt[0], "#c0392b"), ("Ti", rt[1], "0.6")):
            v = np.asarray(t[c], float); ax.plot(x[:m], v[:m] / v[a:a + 6].mean(), color=col, lw=0.7, label=f"{name} ({c})")
    ax.axvline(a, color="b", lw=0.6); ax.axvline(b, color="b", lw=0.6)
    ax.set_title(f"{s}: {lab}", fontsize=9); ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(here, "look_batch3.png"), dpi=80)
json.dump(out, open(os.path.join(here, "process_batch3.json"), "w"), indent=1)

# 343 is too short for its own period (1.3 cycles before the signals collapse at image 144): use 341's period (same
# setup, the scan before), phase from the correlation with E as before
t = load(343); y = np.asarray(t["ROI1"], float) + np.asarray(t["ROI2"], float); a, b = 20, 144
n = np.arange(y.size, dtype=float); P = out[341]["P"]; FCA = 30 / 280
def E_of(d):
    ph = ((n - (a + FCA * P + d)) / P) % 1
    return np.where(ph < 0.5, 1.4 * ph, 0.7 - 1.4 * (ph - 0.5))
d = max(np.linspace(-0.15, 0.15, 61) * P, key=lambda d: abs(np.corrcoef(E_of(d)[a:b], y[a:b])[0, 1]))
E = E_of(d); k = np.arange(int(a + FCA * P + d), b); rel = y / y[a:a + 6].mean()
A = np.column_stack([np.ones(k.size), E[k]]); c, *_ = np.linalg.lstsq(A, rel[k], rcond=None)
cov = np.linalg.inv(A.T @ A) * np.var(rel[k] - A @ c)
out[343].update(P=float(P), shift=float(d), change=float(0.7 * c[1]), change_err=float(0.7 * np.sqrt(cov[1, 1])), period_from=341)
print(f"343 with 341's period ({P:.0f}): change 0->0.7 V {100*0.7*c[1]:+.1f} +- {100*0.7*np.sqrt(cov[1,1]):.1f} %")
json.dump(out, open(os.path.join(here, "process_batch3.json"), "w"), indent=1)

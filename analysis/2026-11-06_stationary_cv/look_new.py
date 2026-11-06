"""Raw traces of the second batch (126, 127, 223, 224, 225), relative to the first images after the EC start."""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ctrfit.echem.external import parse_roi_sums
def load(s):
    return parse_roi_sums(open(os.path.join(here, "data", f"ROIs_s{s}_sums.txt")).read())[s]
fig, axs = plt.subplots(5, 1, figsize=(12, 13))
for ax, (s, sig, start, stop, lab) in zip(axs, [(126, ("ROI1", "ROI2"), 50, 227, "3.3 nm (-2 0 1.75) CV"),
                                                (127, ("ROI1", "ROI2"), 31, 78, "3.3 nm (-2 0 1) CV 25 mV/s"),
                                                (223, ("ROI1",), 51, 410, "3.3 nm (0 -1 3.16) CV repeat"),
                                                (224, ("ROI1",), 50, 448, "3.3 nm (0 -1 3.16) steps 0-0.7 V"),
                                                (225, ("ROI1",), 50, 427, "3.3 nm (0 -1 2.75) steps 0-0.7 V")]):
    t = load(s)
    n = len(t["ROI1"]); print(s, "images", n, "columns", list(t.keys()))
    y = sum(np.asarray(t[c], float) for c in sig)
    m = min(n, stop + 5)
    x = np.arange(m)
    ax.plot(x, y[:m] / y[start:start + 6].mean(), "k-", lw=0.8)
    for c in sig if len(sig) > 1 else ():
        v = np.asarray(t[c], float); ax.plot(x, v[:m] / v[start:start + 6].mean(), lw=0.6, alpha=0.7, label=c)
    ax.axvline(start, color="b", lw=0.6); ax.axvline(stop, color="b", lw=0.6)
    ax.set_xticks(np.arange(0, m, 25), minor=True); ax.grid(which="both", alpha=0.25)
    ax.set_title(f"{s}: {lab}", fontsize=9); ax.legend(fontsize=7) if len(sig) > 1 else None
fig.tight_layout(); fig.savefig(os.path.join(here, "look_new.png"), dpi=80)

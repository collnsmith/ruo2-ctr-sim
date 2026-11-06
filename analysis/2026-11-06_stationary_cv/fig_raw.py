"""Raw ROI sums for one responding and one flat scan at the same surface-sensitive point (0 +-1 3.16): Ru+Ti and Ti,
each relative to the 30 s CA at 0 V, against images since the EC start (no timing model)."""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ctrfit.echem.external import parse_roi_sums
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
scans = [(314, "7 nm (0 1 3.16)", "#1baf7a", 50, 308, "ROI2", "ROI1"), (498, "4.1 nm (0 -1 3.16)", "#eb6834", 20, 216, "ROI1", "ROI2")]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, (s, lab, col, a, b, ru, ti) in zip(axs, scans):
    t = parse_roi_sums(open(os.path.join(here, "data", f"ROIs_s{s}_sums.txt")).read())[s]
    R, T = np.asarray(t[ru], float), np.asarray(t[ti], float)
    y = R + T; n = np.arange(y.size) - a; k = slice(max(a - 15, 0), b)
    ax.plot(n[k], (T / T[a:a + 6].mean())[k], color="#b9b8b0", lw=1, label="Ti ROI")
    ax.plot(n[k], (y / y[a:a + 6].mean())[k], color=col, lw=1.6, label="Ru+Ti (what the CTR measures)")
    ax.axvspan(0, 8, color=GRID, zorder=0)
    ax.set_title(f"{lab}, scan {s}", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("images since EC start (0.0 V hold shaded, then 0-0.7 V at 5 mV/s)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
axs[0].set_ylabel("intensity / intensity at 0.0 V hold"); axs[0].set_ylim(0.92, 1.28)
fig.tight_layout(); fig.savefig(os.path.join(here, "fig_raw.png"), dpi=150)

"""One panel per stationary scan: cycle-averaged change vs E for the Ru ROI, the Ti ROI and their sum (anodic solid,
cathodic dashed), with the coherent model's change for x_OH 0 -> 1 as a reference bar."""
import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
cv, parts = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
order = ["496", "497", "503", "498", "501", "120", "147", "222", "125", None, "311", "344", "314", "316", "356"]
fig, axs = plt.subplots(3, 5, figsize=(15, 8.2), sharex=True)
for ax, s in zip(axs.flat, order):
    if s is None:
        continue
    r = cv[s]
    for roi, col in (("Ru", "#c0392b"), ("Ti", "0.55"), ("sum", "k")):
        for br, ls in (("anodic", "-"), ("cathodic", "--")):
            c = r[roi]["curves"][br]
            ax.errorbar(c["E"], 100 * np.array(c["dI"]), 100 * np.array(c["err"]), color=col, ls=ls, lw=1.2 if roi == "sum" else 0.9,
                        marker="o" if br == "anodic" else None, ms=2.5, capsize=0, label=f"{roi} {br}" if s == order[0] else None)
    q = parts[s]["coh"] * 100
    ax.annotate("", xy=(0.75, q), xytext=(0.75, 0), arrowprops=dict(arrowstyle="->", color="#2471a3", lw=1.5))
    ax.axhline(0, color="0.8", lw=0.6)
    lo, hi = ax.get_ylim(); ax.set_ylim(min(lo, q - 3, -5), max(hi, q + 3, 5))
    flag = " (timing?)" if s == "147" else (" control" if s == "356" else "")
    ax.set_title(f"{s}: {r['film']} nm ({r['H']} {r['K']} {r['L']:g}){flag}", fontsize=9)
    ax.tick_params(labelsize=8); ax.set_xlim(-0.02, 0.79)
axs[1, 4].axis("off")
axs[1, 4].legend(*axs.flat[0].get_legend_handles_labels(), loc="center", fontsize=9, frameon=False,
                    title="blue arrow: coherent model,\nx_OH 0 -> 1 (dz_Obr 0.35)")
for ax in axs[-1]: ax.set_xlabel("E (V vs Ag/AgCl)")
for ax in axs[:, 0]: ax.set_ylabel("I / I(CA, 0 V) - 1 (%)")
fig.suptitle("Stationary CVs, 5 mV/s, 0 - 0.7 V: cycle-averaged change (4.1 nm top row, 3.3 nm middle, 7 nm bottom)", fontsize=11)
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=110)

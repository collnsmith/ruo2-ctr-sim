"""OH fraction from the (0 +-1) CTR fits per film and potential under three bridging-O assumptions (profiles: best and the
range within delta chi2 <= 4 red chi2), and the film difference at each potential."""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
OH = os.path.join(here, "..", "2026-11-05_oh_fits")
summ = json.load(open(os.path.join(OH, "oh_rods_fit_summary.json")))["results"]
VAR = [("Rao et al. 2017 values\n(RHE-corrected: 0.35 / 0.2 / 0 Å)", json.load(open(os.path.join(here, "ohfit_c2_summary.json")))),
       ("bridging O fixed at 0 Å", summ["B fixed 0"]), ("bridging O fitted\n(shared by the films per potential)", summ["A shared per potential"])]
COL = {"4.1 nm": "#eb6834", "7 nm": "#1baf7a"}; MK = {"4.1 nm": "o", "7 nm": "D"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
pots = [-0.238, -0.1, 0.0, 0.15, 0.35, 0.7]
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
for ax, (name, r) in zip(axs, VAR):
    ax.axvspan(-0.02, 0.72, color="#f3f2ee", zorder=0)
    for film, off in (("4.1 nm", -0.012), ("7 nm", 0.012)):
        E, b, lo, hi = [], [], [], []
        for e in pots:
            q = r["profiles"].get(f"{film} {e:g} V")
            if q: E.append(e + off); b.append(q["best"]); lo.append(q["best"] - q["lo"]); hi.append(q["hi"] - q["best"])
        ax.errorbar(E, b, yerr=[lo, hi], fmt=MK[film] + "-", color=COL[film], ms=6.5, mec="white", mew=1, lw=1.6,
                    elinewidth=1, capsize=0, label=film)
    ax.set_title(name, loc="left", fontsize=9.5, color=INK)
    ax.set_xlabel("E (V vs Ag/AgCl); shaded: CV window")
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_ylim(-0.05, 1.05)
axs[0].set_ylabel("OH fraction on the CUS sites (CTR fit)")
axs[0].legend(frameon=False, loc="lower left")
fig.suptitle("Same relative result under every bridging-O assumption: the films differ at or below 0 V, agree at 0.35 and 0.7 V",
             fontsize=10.5, color=INK, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(os.path.join(here, "fig_variants.png"), dpi=150)
for name, r in VAR:
    d = [f"{e:+g} V: {r['profiles'][f'4.1 nm {e:g} V']['best'] - r['profiles'][f'7 nm {e:g} V']['best']:+.2f}"
         for e in (-0.1, 0.0, 0.15, 0.35, 0.7)]
    print(name.split(chr(10))[0], "| 4.1 - 7 nm:", ", ".join(d), "| red chi2", round(r["red_chi2"], 2))

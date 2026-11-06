"""Figure for rod153.py: the 3.3 nm (0 -1 L) rod at 0.65 V from scans 153-221 against the film models, and the measured
3.3 nm CV changes at 0.7 V against the predictions from the CTR-fitted surface states (every structure and dz_Obr
assumption in rod153.json). Run rod153.py first."""
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
__file__ = os.path.join(here, "rod153.py")
exec(open(os.path.join(here, "rod153.py")).read().split("use = rod[")[0])
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

o = json.load(open(os.path.join(here, "rod153.json")))
rod = pd.DataFrame(o["rod"]); FREE = o["free"][:4]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
C33, C41, C7 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.1, 1.1, 0.9]))
Lm = np.linspace(0.3, 3.8, 1400)
fitpts = rod.scan.isin(o["fit_points"])
for a, lo, hi in ((ax[0], 0.3, 3.8), (ax[1], 2.4, 3.8)):
    a.semilogy(rod.L_nom[~fitpts], np.clip(rod["mean"][~fitpts], 1, None), "o", ms=4, mfc="white", mec=MUTED,
               label="not fitted (spot leaves the ROI, or within 0.08 of Bragg)")
    a.semilogy(rod.L_nom[fitpts], rod["mean"][fitpts], "o", ms=4.5, color=C33, label="fitted (spot in the ROI all scan)")
mods = [(dict(base, **state("B dz_Obr = 0", 0.7)), 0.0, 0.0, C41, "-", "4.1 nm CTR structure (10 TL, eps 1.44 %)"),
        (dict(P.view(lead["7 nm"]), **state("B dz_Obr = 0", 0.7)), 0.0, 0.0, C7, "-", "7 nm CTR structure (24 TL, eps 0.50 %)")]
for var, ls in (("B dz_Obr = 0", "--"), ("A dz_Obr fitted", ":")):
    x = o["fits"][var]["x"]
    mods.append((dict(base, **state(var, 0.7), **dict(zip(FREE, x[:4]))), x[4], x[5], INK, ls,
                 f"fit to this rod ({var}): {x[0]:.1f} TL, eps {x[2]:.2f} %, dL {x[4]:+.3f}"))
sel = fitpts.values
for v, dL, slope, c, ls, lab in mods:
    Im = I(v, Lm + dL) * np.exp(2 * slope * (Lm - 3))
    k = np.exp(np.median(np.log(rod["mean"][sel].values / np.interp(rod.L_nom[sel].values, Lm, Im))))
    for a in ax[:2]:
        a.semilogy(Lm, Im * k, color=c, ls=ls, lw=1.2, label=lab)
for Lc in (2.755, 3.168, 3.257):
    ax[1].axvline(Lc, color=MUTED, lw=0.7, ls=(0, (2, 3)))
    ax[1].text(Lc, 2.2e6, f"CV\n{Lc:.2f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
ax[0].set_ylim(80, 2e7); ax[1].set_ylim(80, 2e7); ax[1].set_xlim(2.4, 3.8)
ax[0].set_xlabel("L (SPEC)"); ax[1].set_xlabel("L (SPEC); dotted: the CV reflections (#Q)")
ax[0].set_ylabel("counts per image, ROI1 - background")
ax[0].set_title("3.3 nm (0 -1 L) at 0.65 V, scans 153-221", loc="left", color=INK, fontsize=10)
ax[1].set_title("near the (0 -1 3) Bragg peak", loc="left", color=INK, fontsize=10)
ax[0].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=1)
h, l = ax[1].get_legend_handles_labels()
ax[1].legend(h[2:], l[2:], fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.13))

# CV change at 0.7 V: measured 3.3 nm and 7 nm, and the range of 3.3 nm predictions
meas = o["measured"]; pr = o["predictions"]
t7 = {{"2.75": "(0 -1 2.755)", "3.16": "(0 -1 3.168)", "3.25": "(0 -1 3.257)"}.get(r["hkl"].split()[-1].rstrip(")"), "-"): r["measured"]
      for r in json.load(open(os.path.join(here, "ctr2cv.json")))["table"] if r["film"] == "7 nm" and r["E"] == 0.7}
a = ax[2]
for i, Lc in enumerate((2.755, 3.168, 3.257)):
    hkl = f"(0 -1 {Lc:g})"
    vals = [100 * q["0.7"] for var in pr[hkl].values() for tag, q in var.items() if tag != "7 nm structure"]
    a.add_patch(plt.Rectangle((i - 0.3, min(vals)), 0.6, max(vals) - min(vals), color=C33, alpha=0.18, lw=0))
    a.plot([i - 0.3, i + 0.3], [np.median(vals)] * 2, color=C33, lw=1, alpha=0.7)
    a.plot(i - 0.08, 100 * meas[hkl]["0.7"], "o", ms=8, color=C33, label="3.3 nm CV, measured" if i == 0 else None)
    if hkl in t7:
        a.plot(i + 0.08, 100 * t7[hkl], "D", ms=6.5, color=C7, label="7 nm CV, measured (0 1 2.75 / 3.16 / 3.25)" if i == 0 else None)
a.add_patch(plt.Rectangle((0, 0), 0, 0, color=C33, alpha=0.18, lw=0, label="3.3 nm predicted from CTR states\n(both rod fits, 3 dz_Obr assumptions)"))
a.axhline(0, color=GRID, lw=0.8)
a.set_xticks(range(3)); a.set_xticklabels(["L 2.755", "L 3.168", "L 3.257"]); a.set_xlim(-0.6, 2.6)
a.set_ylabel("CV change at 0.7 V from 0 V (%)")
a.set_title("CV change at 0.7 V: measured vs predicted", loc="left", color=INK, fontsize=10)
a.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.13))
a.grid(axis="y", color=GRID, lw=0.5); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(here, "fig_rod153.png"), dpi=150, bbox_inches="tight")

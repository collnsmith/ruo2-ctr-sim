"""Switch potential of the 3.3 and 7 nm films against the work function (WF) and the potential of maximum entropy (PME)
of RuO2 films vs thickness (table supplied by the user, 2026-11-16). PME is taken as V vs Ag/AgCl, as in the logbook
remark for the 4.1 nm film ("PME for this film is about 0 vs Ag/AgCl"); if the table is vs RHE, the switch points in
panel b move up by 0.238 V (panel c compares differences and is unaffected).
Switch potential: half-way of the X-ray switch with the clock-synced potential (curves_sync.json), mean and standard
error over scans (7 nm without the noisy 343). Writes slide_figures/slideB8_switch_vs_wf_pme.png and plot_data/13_wf_pme/."""
import csv, json, os
here = os.path.dirname(os.path.abspath(__file__))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

T = np.array([1.7, 2.8, 3.3, 5.0, 6.4, 6.5, 7.4, 13.9, 15.2])
WF = np.array([4.276, 4.362, 4.841, 4.78, 4.573, 4.551, 4.528, 4.915, 4.972])
PME = np.array([0.831, 0.737, 0.441, 0.131, 0.487, 0.481, 0.731, 0.487, 0.081])
PE = np.array([0.05, 0.05, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05])

c = json.load(open(os.path.join(here, "curves_sync.json")))
sw = {}
for film, ns in (("3.3", [120, 222, 223]), ("7", [311, 314, 341, 344])):
    v = np.array([c["scans"][str(n)]["raw"]["halfway"] for n in ns])
    sw[film] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)), v)
wf_sw = {"3.3": (4.841, 0.0, 0.0), "7": (float(np.interp(7.0, T, WF)), 4.551 - np.interp(7.0, T, WF), np.interp(7.0, T, WF) - 4.528)}
slope, icpt = np.polyfit(WF, PME, 1)
cov = np.polyfit(WF, PME, 1, cov=True)[1]
d_wf = wf_sw["3.3"][0] - wf_sw["7"][0]
pred_33 = sw["7"][0] + slope * d_wf
d_obs = sw["3.3"][0] - sw["7"][0]; d_err = float(np.hypot(sw["3.3"][1], sw["7"][1]))

COL = {"3.3": "#2a78d6", "4.1": "#eb6834", "7": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10.5,
                     "axes.titlelocation": "left"})
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
# a: WF vs thickness
a = ax[0]
for f, tt in (("3.3", 3.3), ("4.1", 4.1), ("7", 7.0)):
    a.axvline(tt, color=COL[f], lw=1.2, alpha=0.5, ls=(0, (3, 2)))
a.plot(T, WF, "o-", color=INK, ms=5, lw=1)
a.set_xscale("log"); a.set_xticks([2, 3, 5, 7, 10, 15]); a.set_xticklabels(["2", "3", "5", "7", "10", "15"])
a.xaxis.set_minor_formatter(NullFormatter())
for f, tt, ha in (("3.3", 3.3, "right"), ("4.1", 4.1, "left"), ("7", 7.0, "center")):
    a.text(tt, 1.0, f" {f} nm ", transform=a.get_xaxis_transform(), ha=ha, va="bottom", fontsize=8, color=COL[f])
a.set_xlabel("film thickness (nm)"); a.set_ylabel("work function (eV)")
a.set_title("a  Work function", pad=14)
a.grid(axis="y", color=GRID, lw=0.5)
# b: PME vs thickness, with the switch potentials
b = ax[1]
for f, tt in (("3.3", 3.3), ("4.1", 4.1), ("7", 7.0)):
    b.axvline(tt, color=COL[f], lw=1.2, alpha=0.5, ls=(0, (3, 2)))
b.errorbar(T, PME, yerr=PE, fmt="o-", color=INK, ms=5, lw=1, capsize=0, label="PME (table)")
b.errorbar([3.3], [sw["3.3"][0]], yerr=[sw["3.3"][1]], fmt="s", ms=9, color=COL["3.3"], mec="white", capsize=0, zorder=4)
b.errorbar([7.0], [sw["7"][0]], yerr=[sw["7"][1]], fmt="D", ms=8, color=COL["7"], mec="white", capsize=0, zorder=4)
b.plot([4.1], [0.0], "o", ms=8, mfc="white", mec=COL["4.1"], mew=1.5, zorder=4)
b.set_xscale("log"); b.set_xticks([2, 3, 5, 7, 10, 15]); b.set_xticklabels(["2", "3", "5", "7", "10", "15"])
b.xaxis.set_minor_formatter(NullFormatter())
for f, tt, ha in (("3.3", 3.3, "right"), ("4.1", 4.1, "left"), ("7", 7.0, "center")):
    b.text(tt, 1.0, f" {f} nm ", transform=b.get_xaxis_transform(), ha=ha, va="bottom", fontsize=8, color=COL[f])
b.set_xlabel("film thickness (nm)"); b.set_ylabel("potential (V vs Ag/AgCl)")
b.set_ylim(-0.15, 0.95)
b.set_title("b  PME and the switch potential", pad=14)
b.grid(axis="y", color=GRID, lw=0.5)
# c: vs WF
cax = ax[2]
wl = np.linspace(4.2, 5.03, 50)
cax.plot(wl, slope * wl + icpt, color=MUTED, lw=1, ls="--")
cax.errorbar(WF, PME, yerr=PE, fmt="o", color=INK, ms=5, capsize=0)
OFFS = {6.4: (6, 2), 6.5: (-20, -12)}
for tt, w, p in zip(T, WF, PME):
    cax.annotate(f"{tt:g}", (w, p), textcoords="offset points", xytext=OFFS.get(tt, (5, 3)), fontsize=7.5, color=MUTED)
w33, w7 = wf_sw["3.3"][0], wf_sw["7"][0]
cax.errorbar([w33], [sw["3.3"][0]], yerr=[sw["3.3"][1]], fmt="s", ms=9, color=COL["3.3"], mec="white", capsize=0, zorder=4)
cax.errorbar([w7], [sw["7"][0]], yerr=[sw["7"][1]], xerr=[[wf_sw["7"][1]], [wf_sw["7"][2]]], fmt="D", ms=8, color=COL["7"],
             mec="white", capsize=0, zorder=4)
cax.plot([w7, w33], [sw["7"][0], pred_33], color=COL["3.3"], lw=1.2, ls=":")
cax.plot([w33], [pred_33], "s", ms=9, mfc="white", mec=COL["3.3"], mew=1.5, zorder=4)
cax.set_xlabel("work function (eV)"); cax.set_ylabel("potential (V vs Ag/AgCl)")
cax.set_ylim(-0.15, 0.95)
cax.set_title("c  Against the work function", pad=14)
cax.grid(axis="y", color=GRID, lw=0.5)
h = [Line2D([], [], marker="o", ls="-", color=INK, ms=5, label="PME of films 1.7-15.2 nm (labels: nm)"),
     Line2D([], [], color=MUTED, ls="--", label=f"PME fit: slope {slope:.2f} ± {np.sqrt(cov[0, 0]):.2f} V/eV"),
     Line2D([], [], marker="s", ls="none", color=COL["3.3"], ms=8, label=f"switch, 3.3 nm: {sw['3.3'][0]:.2f} V"),
     Line2D([], [], marker="D", ls="none", color=COL["7"], ms=7, label=f"switch, 7 nm: {sw['7'][0]:.2f} V"),
     Line2D([], [], marker="s", ls=":", color=COL["3.3"], mfc="white", ms=8,
            label=f"3.3 nm if the switch moved like the PME: {pred_33:+.2f} V"),
     Line2D([], [], marker="o", ls="none", mfc="white", mec=COL["4.1"], ms=7, label="4.1 nm: no switch (logbook PME about 0 V)")]
fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
fig.savefig(os.path.join(here, "slide_figures", "slideB8_switch_vs_wf_pme.png"), dpi=150, bbox_inches="tight", facecolor="white")

od = os.path.join(here, "plot_data", "13_wf_pme"); os.makedirs(od, exist_ok=True)
with open(os.path.join(od, "wf_pme_table.csv"), "w", newline="") as f:
    f.write("# Work function and PME vs film thickness, supplied by the user (2026-11-16). PME taken as V vs Ag/AgCl.\n")
    w = csv.writer(f, lineterminator="\n"); w.writerow(["thickness_nm", "WF_eV", "WF_err", "PME_V", "PME_err"])
    for r in zip(T, WF, np.zeros_like(T), PME, PE):
        w.writerow([f"{x:g}" for x in r])
with open(os.path.join(od, "switch_points.csv"), "w", newline="") as f:
    f.write("# Half-way potential of the X-ray switch (V vs Ag/AgCl, clock-synced), mean and s.e. over scans; WF at that "
            "thickness from the table (7 nm interpolated between 6.5 and 7.4 nm).\n")
    w = csv.writer(f, lineterminator="\n"); w.writerow(["film_nm", "switch_V", "switch_se", "WF_eV", "scans_V"])
    for film in ("3.3", "7"):
        w.writerow([film, f"{sw[film][0]:.3f}", f"{sw[film][1]:.3f}", f"{wf_sw[film][0]:.3f}", " ".join(f"{x:.3f}" for x in sw[film][2])])
print(f"PME fit slope {slope:.3f} +- {np.sqrt(cov[0, 0]):.3f} V/eV; WF 3.3 - 7 nm = {d_wf:+.3f} eV")
print(f"switch: 3.3 nm {sw['3.3'][0]:.3f} +- {sw['3.3'][1]:.3f}, 7 nm {sw['7'][0]:.3f} +- {sw['7'][1]:.3f}; "
      f"difference {d_obs:+.3f} +- {d_err:.3f} V; PME-like expectation {slope * d_wf:+.3f} V; off by {(d_obs - slope * d_wf) / d_err:.1f} sigma")

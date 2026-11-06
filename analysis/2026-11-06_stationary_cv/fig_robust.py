"""Key evidence, model-free: (a) timing-free CV amplitude per scan at the shared HKLs, by film; (b) whole-rod change of the
OH rod against the same film's 0 V rod over the CTR series, with the repeatability band from rods that cannot see OH."""
import json, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
here = os.path.dirname(os.path.abspath(__file__))
amp = json.load(open(os.path.join(here, "robust_cv.json")))
info = json.load(open(os.path.join(here, "cv3.json")))
rodrms = json.load(open(os.path.join(here, "robust_rods.json")))
COL = {"3.3": "#2a78d6", "4.1": "#eb6834", "7": "#1baf7a"}
MK = {"3.3": "s", "4.1": "o", "7": "D"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
Ls = [1.25, 2.75, 2.84, 3.16, 3.25]
off = {"3.3": -0.22, "4.1": 0.0, "7": 0.18}
seen = set()
for s, r in info.items():
    if r["H"] != 0: continue
    f = r["film"]; x = Ls.index(r["L"]) + off[f]
    y = 100 * amp[s]["sum"]["sine_pp"]
    a.plot(x, y, MK[f], ms=8, color=COL[f], mec="white", mew=1.2, zorder=3, label=f"{f} nm" if f not in seen else None)
    seen.add(f)
    if s == "147":
        a.annotate("147: timing unclear,\nRu+Ti mostly Ti here", (x, y), (x + 0.25, y * 1.9), fontsize=7.5, color=MUTED,
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
pn = json.load(open(os.path.join(here, "process_new.json")))
b3 = json.load(open(os.path.join(here, "process_batch3.json")))
# controls: OH-blind reflections, as triangles in the film's colour
a.plot([], [], "v", ms=8, color=MUTED, mec="white", label="triangles: OH-blind controls (+-2 0 L), (-1 0 2.32)")
xc = len(Ls) + 0.1
for f, x, v in (("7", xc + 0.18, amp["356"]["sum"]["sine_pp"]), ("4.1", xc, b3["519"]["sine_pp"]),
                ("3.3", xc - 0.30, pn["126"]["sine_pp"]), ("3.3", xc - 0.20, pn["127"]["sine_pp"]), ("3.3", xc - 0.10, b3["227"]["sine_pp"])):
    a.plot(x, 100 * v, "v", ms=8, color=COL[f], mec="white", mew=1.2, zorder=3)
# repeats from the later batches
a.plot(Ls.index(3.16) - 0.10, 100 * pn["223"]["sine_pp"], MK["3.3"], ms=8, color=COL["3.3"], mec="white", mew=1.2, zorder=3)
for dx, s_ in ((0.10, "341"), (0.26, "343")):
    a.plot(Ls.index(2.75) + dx, 100 * b3[s_]["sine_pp"], MK["7"], ms=8, color=COL["7"], mec="white", mew=1.2, zorder=3)

a.set_yscale("log"); a.set_ylim(0.1, 80)
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
a.set_xticks(list(range(len(Ls))) + [len(Ls) + 0.1]); a.set_xticklabels([f"(0 ±1 {L:g})" for L in Ls] + ["controls"], fontsize=8)
a.set_ylabel("CV oscillation, Ru+Ti, peak-to-peak (%)")
a.set_title("a  Stationary CVs: period and phase left free", loc="left", color=INK, fontsize=10)
a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
h, l = a.get_legend_handles_labels(); o = [l.index(x) for x in ("3.3 nm", "4.1 nm", "7 nm")] + [3]
a.legend([h[i] for i in o], [l[i] for i in o], frameon=False, fontsize=7.5, loc="lower left")
tr = json.load(open(os.path.join(here, "timeline_rods.json")))
groups = [("7 nm\nsame potential,\nhours apart", [tr["7 nm repeat 0.7 V"], tr["7 nm repeat -0.1 V"]], "7", True),
          ("7 nm\n0.7 V vs -0.1 V\n(both directions)", [tr["7 nm 0.7 Va vs -0.1 Va"], tr["7 nm -0.1 Vb -> 0.7 Vb"]], "7", False),
          ("4.1 nm\nback to back,\nany two potentials", [tr[k] for k in tr if k.startswith("4.1 nm") and "->" in k and "OCV" not in k], "4.1", False)]
for i_, (lab, vals, f, hollow) in enumerate(groups):
    x = i_ + np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [i_]
    b.plot(x, vals, MK[f], ms=9, color=COL[f], mfc="white" if hollow else COL[f], mec=COL[f] if hollow else "white", mew=1.6, ls="none")
k = "4.1 nm 0.7 V -> -0.238 V"
b.annotate("0.7 V -> -0.238 V", (2 + np.linspace(-0.12, 0.12, len(groups[2][1]))[[kk for kk in tr if kk.startswith("4.1 nm") and "->" in kk and "OCV" not in kk].index(k)], tr[k]),
           (12, 14), textcoords="offset points", fontsize=7.5, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
b.set_xticks(range(3)); b.set_xticklabels([g_[0] for g_ in groups], fontsize=8)
b.set_xlim(-0.5, 2.5)
b.set_ylabel("rms of ln(I_a / I_b) over the OH rod")
b.set_title("b  CTR OH rods in measurement order (logbook)", loc="left", color=INK, fontsize=10)
b.grid(axis="y", color=GRID, lw=0.6); b.set_axisbelow(True); b.set_ylim(0, 0.1)
fig.tight_layout(); fig.savefig(os.path.join(here, "fig_robust.png"), dpi=150)

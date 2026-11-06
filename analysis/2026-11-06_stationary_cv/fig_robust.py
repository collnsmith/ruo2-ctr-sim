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
off = {"3.3": -0.18, "4.1": 0.0, "7": 0.18}
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
c = amp["356"]["sum"]["sine_pp"] * 100
a.plot(len(Ls) + 0.1, c, "v", ms=8, color=MUTED, mec="white", mew=1.2, zorder=3, label="7 nm (2 0 1.75), OH-blind control")
a.set_yscale("log"); a.set_ylim(0.1, 80)
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
a.set_xticks(list(range(len(Ls))) + [len(Ls) + 0.1]); a.set_xticklabels([f"(0 ±1 {L:g})" for L in Ls] + ["(2 0 1.75)"], fontsize=8)
a.set_ylabel("CV oscillation, Ru+Ti, peak-to-peak (%)")
a.set_title("a  Stationary CVs: period and phase left free", loc="left", color=INK, fontsize=10)
a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
h, l = a.get_legend_handles_labels(); o = [l.index(x) for x in ("3.3 nm", "4.1 nm", "7 nm")] + [3]
a.legend([h[i] for i in o], [l[i] for i in o], frameon=False, fontsize=8, loc="upper left")
pots = ["-0.238 V", "-0.1 V", "0.15 V", "0.35 V", "0.7 V", "OCV"]
lo, hi = rodrms["baseline"]
b.axhspan(lo, hi, color="#e6e5e0", zorder=0)
b.text(-0.2, (lo + hi) / 2, "repeatability: even rods, 7 nm OCV", fontsize=7.5, color=MUTED, va="center", ha="left")
for f, key in (("4.1", "4.1 nm"), ("7", "7 nm")):
    xs, ys = [], []
    for i, k in enumerate(pots):
        v = rodrms["oh"][key].get(k)
        if v is not None:
            xs.append(i); ys.append(v)
    b.plot(xs, ys, MK[f], ms=8, color=COL[f], mec="white", mew=1.2, zorder=3, ls="none")
    b.plot([x for x in xs if pots[x] != "OCV"], [y for x, y in zip(xs, ys) if pots[x] != "OCV"], color=COL[f], lw=2, zorder=2)
    b.annotate(f"{f} nm", (xs[-2], ys[-2]), (8, 0), textcoords="offset points", color=INK, fontsize=8.5, va="center")
b.set_xticks(range(len(pots))); b.set_xticklabels([p.replace(" V", "") for p in pots])
b.set_xlabel("potential (V vs Ag/AgCl), each against the same film's 0 V rod")
b.set_ylabel("rms of ln(I / I$_{0\\,V}$) over the OH rod")
b.set_title("b  CTR series: whole OH rod, one free scale", loc="left", color=INK, fontsize=10)
b.grid(axis="y", color=GRID, lw=0.6); b.set_axisbelow(True); b.set_ylim(0, 0.14)
fig.tight_layout(); fig.savefig(os.path.join(here, "fig_robust.png"), dpi=150)

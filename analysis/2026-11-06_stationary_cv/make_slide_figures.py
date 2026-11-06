"""Slide figures for slides_outline.txt, one PNG per slide (slideNN_*.png; backups slideB*_*.png), in slide_figures/.
Data come from plot_data/ (make_plot_data.py); slides 4 and 11 also evaluate the CTR model. Run: python make_slide_figures.py"""
import glob, json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))
sys.modules["xraydb"] = None
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

PD = os.path.join(here, "plot_data")
OUT = os.path.join(here, "slide_figures")
os.makedirs(OUT, exist_ok=True)
RHE = 0.238
COL = {"3.3": "#2a78d6", "4.1": "#eb6834", "7": "#1baf7a"}
MK = {"3.3": "s", "4.1": "o", "7": "D"}
INK, MUTED, GRID, TI = "#1f1f1e", "#6b6a63", "#e6e5e0", "#9a998f"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 12,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlelocation": "left", "figure.dpi": 100, "savefig.dpi": 150})
W, Hh = 13.33, 7.0                                  # 16:9 slide body


def rd(rel):
    return pd.read_csv(os.path.join(PD, rel), comment="#")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", name)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.7); ax.set_axisbelow(True)


def rhe_axis(ax):
    sec = ax.secondary_xaxis("top", functions=(lambda x: x + RHE, lambda x: x - RHE))
    sec.set_xlabel("E (V vs RHE)", color=MUTED, fontsize=12); sec.tick_params(colors=MUTED, labelsize=11)
    return sec


amp = rd("03_cv_summary/cv_amplitudes.csv")
CTRL_LAB = {(-2, 0, 1.75): "(-2 0 1.75)", (-2, 0, 1.0): "(-2 0 1)", (-1, 0, 2.32): "(-1 0 2.32)", (1, 0, 1.75): "(1 0 1.75)", (2, 0, 1.75): "(2 0 1.75)"}


def curve(film, s):
    f = glob.glob(os.path.join(PD, "02_cv_curves", f"{film}nm", f"s{s}_*.csv"))
    return pd.read_csv(f[0], comment="#") if f else None


# ===================================================================== slide 2: headline
def slide02():
    fig, ax = plt.subplots(figsize=(W, Hh))
    xs = {"3.3": 0, "4.1": 1, "7": 2}
    rng = np.random.default_rng(3)
    for _, r in amp.iterrows():
        f = f"{r.film_nm:g}"; x = xs[f] + rng.uniform(-0.18, 0.18)
        y = 100 * r.sum_pp
        if r.kind == "control":
            ax.plot(x + 0.32, y, "v", ms=11, color=COL[f], mfc="white", mew=2, zorder=3)
        elif r.scan == 147:
            ax.plot(x, y, MK[f], ms=11, color=COL[f], mfc="white", mew=2, zorder=3)
        else:
            ax.plot(x, y, MK[f], ms=11, color=COL[f], mec="white", mew=1.2, zorder=3)
    ax.set_ylim(-1.5, 42)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["3.3 nm", "4.1 nm", "7 nm"], fontsize=14)
    for t, f in zip(ax.get_xticklabels(), ("3.3", "4.1", "7")):
        t.set_color(COL[f])
    ax.set_xlim(-0.6, 2.8)
    ax.set_ylabel("oscillation with potential, peak-to-peak (%)")
    ax.set_title("Same protocol, same surface-sensitive reflections: the 4.1 nm film does not respond", fontsize=15)
    grid(ax)
    h = [Line2D([], [], marker="o", ls="none", color=INK, ms=10, label="surface-sensitive (0 ±1 L), one point per scan"),
         Line2D([], [], marker="v", ls="none", color=INK, mfc="white", mew=2, ms=10, label="OH-blind control reflection"),
         Line2D([], [], marker="s", ls="none", color=INK, mfc="white", mew=2, ms=10, label="3.3 nm scan 147 (hysteretic, signal mostly substrate)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=False)
    save(fig, "slide02_headline.png")


# ===================================================================== slide 3: measurements in order
EVENTS = {
    "3.3": [("cv", 120), ("cv", 125), ("ctrl", 126), ("ctrl", 127), ("ctrl", 135), ("cv", 147), ("ctr", 0.65, "153-221"),
            ("cv", 222), ("cv", 223), ("stair", 224), ("stair", 225), ("ctrl", 227)],
    "7": [("cv", 311), ("cv", 314), ("cv", 316), ("ocv", -0.02, "317-320"), ("ctr", 0.7, "325"), ("ctr", 0.35, "327"),
          ("ctr", 0.15, "329"), ("ctr", 0.0, "331"), ("ctr", -0.1, "333"), ("ctr", -0.1, "337"), ("ctr", 0.7, "339"),
          ("cv", 341), ("cv", 343), ("cv", 344), ("even", 0.0, "351"), ("even", 0.7, "353"), ("ctrl", 356), ("even", 0.0, "359"), ("even", 0.7, "361")],
    "4.1": [("ctr", 0.0, "483"), ("ctr", -0.1, "485"), ("ctr", 0.15, "487"), ("ctr", 0.7, "489"), ("ctr", -0.238, "491"),
            ("ctr", 0.35, "493"), ("spot", None), ("cv", 496), ("cv", 497), ("cv", 498), ("cv", 501), ("cv", 503),
            ("ocv", 0.378, "504"), ("evocv", 0.378, "506"), ("even", 0.0, "508"), ("even", 0.7, "510"), ("evocv", 0.378, "512"),
            ("evocv", 0.371, "513"), ("even", 0.0, "515"), ("even", 0.7, "517"), ("ctrl", 519), ("evocv", 0.4, "521"),
            ("even", 0.0, "523"), ("even", 0.7, "526")]}


def slide03():
    fig, axs = plt.subplots(3, 1, figsize=(W, 8.4), sharey=True)
    titles = {"3.3": "3.3 nm, day 2", "7": "7 nm, day 3", "4.1": "4.1 nm, day 4"}
    for ax, f in zip(axs, ("3.3", "7", "4.1")):
        ev = EVENTS[f]; x = 0; ctr_x, ctr_y, ticks, labs = [], [], [], []
        for e in ev:
            kind = e[0]
            if kind == "spot":
                ax.axvline(x - 0.5, color=INK, ls="--", lw=1.2)
                ax.text(x - 0.4, 0.86, "moved to a fresh spot", fontsize=11, color=INK, va="center")
                continue
            if kind in ("cv", "ctrl", "stair"):
                face = COL[f] if kind == "cv" else ("#c9c8c0" if kind == "ctrl" else "white")
                ax.add_patch(Rectangle((x - 0.32, 0.0), 0.64, 0.7, facecolor=face, edgecolor=COL[f] if kind != "ctrl" else MUTED,
                                       hatch="///" if kind == "stair" else None, lw=1.0, alpha=0.55 if kind == "cv" else 1, zorder=2))
                labs.append(str(e[1]))
            elif kind == "ctr":
                ctr_x.append(x); ctr_y.append(e[1]); labs.append(e[2])
            elif kind in ("ocv", "evocv"):
                ax.plot(x, e[1], "D" if kind == "ocv" else "d", ms=10 if kind == "ocv" else 8, color=COL[f] if kind == "ocv" else MUTED,
                        mfc="white", mew=2, zorder=4); labs.append(e[2])
            elif kind == "even":
                ax.plot(x, e[1], "s", ms=8, color=MUTED, zorder=4); labs.append(e[2])
            ticks.append(x); x += 1
        ax.plot(ctr_x, ctr_y, "-", color=COL[f], lw=1.5, zorder=3)
        ax.plot(ctr_x, ctr_y, "s", ms=11, color=COL[f], mec="white", mew=1.2, zorder=4)
        ax.set_xticks(ticks); ax.set_xticklabels(labs, rotation=90, fontsize=10)
        ax.set_xlim(-0.8, x - 0.2); ax.set_ylim(-0.35, 0.95)
        ax.set_title(titles[f], color=COL[f], fontsize=14)
        ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6]); grid(ax)
    axs[1].set_ylabel("potential (V vs Ag/AgCl)")
    axs[-1].set_xlabel("measurements in the order they were made (scan numbers)")
    h = [Line2D([], [], marker="s", ls="-", color=INK, ms=10, label="CTR, surface-sensitive rod, at its potential"),
         Patch(facecolor=INK, alpha=0.4, label="stationary CV, 0 to 0.7 V (film colour)"),
         Patch(facecolor="#c9c8c0", edgecolor=MUTED, label="CV on an OH-blind control reflection"),
         Patch(facecolor="white", edgecolor=INK, hatch="///", label="potential staircase 0 to 0.7 V"),
         Line2D([], [], marker="D", ls="none", color=INK, mfc="white", mew=2, ms=9, label="CTR at open circuit (rest potential)"),
         Line2D([], [], marker="s", ls="none", color=MUTED, ms=8, label="CTR, other rods")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "slide03_measurements.png")


# ===================================================================== model helpers (slides 4, 11)
_MODEL = {}


def model():
    if not _MODEL:
        from ctrfit.project.project import Project
        from ctrfit.fit.global_fit import film_of
        p = Project.load(os.path.join(here, "..", "2026-11-05_oh_fits", "oh_rods_fit.ctrproj.json"))
        lead = {}
        for ds in p.datasets:
            lead.setdefault(film_of(ds).split("|")[0], ds.scope)
        _MODEL.update(m=p.model, P=p.model.params, lead=lead)
    return _MODEL


def view(film):
    M = model(); v = dict(M["P"].view(M["lead"]["4.1 nm" if film in ("3.3", "4.1") else "7 nm"]), theta=1.0)
    if film == "3.3":
        v["thickness_mean_tl"] = 10.2
    return v


# ===================================================================== slide 4: why these reflections
def slide04():
    M = model(); m = M["m"]
    L = np.arange(0.4, 3.6, 0.005)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 6.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for f in ("4.1", "7"):
        v = dict(view(f), dz_Obr=0.35)
        ch = m.intensity(0, 1, L, dict(v, x_OH=0.7)) / m.intensity(0, 1, L, dict(v, x_OH=0.3)) - 1
        ch[np.abs(L - np.round(L)) < 0.03] = np.nan             # Bragg / extinction points: undefined ratio
        a.plot(L, 100 * ch, color=COL[f], lw=2.2, label=f"{f} nm structure")
    a.axvspan(2.92, 3.08, color=GRID, zorder=0)
    a.axhline(0, color=MUTED, lw=0.8)
    for Lc in (1.25, 2.75, 2.84, 3.16, 3.25):
        a.axvline(Lc, color=INK, lw=0.8, ls=":", zorder=1)
    a.set_ylim(-90, 60); a.set_xlim(0.4, 3.6)
    a.set_xticks([0.5, 1, 1.25, 1.5, 2, 2.5, 2.75, 3, 3.25, 3.5])
    a.set_xticklabels(["0.5", "1", "1.25", "1.5", "2", "2.5", "2.75", "3", "3.25", "3.5"])
    a.set_xlabel("L (dotted: the CV reflections; shaded: TiO2 Bragg peak)")
    a.set_ylabel("intensity change for OH fraction 0.3 → 0.7 (%)")
    a.set_title("a  Along the (0 1) rod (model)")
    a.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False); grid(a)
    s = rd("09_model_sensitivity/sensitivity.csv"); s = s[s.film_model.str.startswith("4.1")]
    order = [(0, -1, 1.25), (0, -1, 2.75), (0, -1, 2.84), (0, -1, 3.16), (0, -1, 3.25), (1, 0, 1.75), (-1, 0, 2.32), (-2, 0, 1.75), (2, 0, 1.75), (-2, 0, 1.0)]
    vals, labs, cols = [], [], []
    for H, K, Lv in order:
        r = s[(s.H == H) & (s.K == K) & (np.isclose(s.L, Lv))].iloc[0]
        vals.append(100 * abs(r["OH_0.3_to_0.7_dz0.35"])); labs.append(f"({H} {K} {Lv:g})" if H else f"(0 ±1 {Lv:g})")
        cols.append(INK if H == 0 else "#c9c8c0")
    y = np.arange(len(vals))[::-1]
    b.barh(y, vals, color=cols, height=0.65)
    for yy, vv in zip(y, vals):
        b.text(vv + 1, yy, f"{vv:.0f}%" if vv >= 1 else f"{vv:.1f}%", va="center", fontsize=11, color=INK)
    b.set_yticks(y); b.set_yticklabels(labs, fontsize=12); b.set_xlim(0, 60)
    b.set_xlabel("|intensity change| for OH 0.3 → 0.7 (%), 4.1 nm structure")
    b.set_title("b  Measured reflections: sensitive vs controls"); grid(b, "x")
    b.legend(handles=[Patch(color=INK, label="surface-sensitive"), Patch(color="#c9c8c0", label="control (OH-blind)")],
             loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, "slide04_reflections.png")


# ===================================================================== slide 5: raw traces
def trace(film, s):
    return pd.read_csv(glob.glob(os.path.join(PD, "01_cv_traces", f"{film}nm", f"s{s}_*.csv"))[0], comment="#")


def slide05():
    fig, axs = plt.subplots(1, 2, figsize=(W, 5.6), sharey=True)
    for ax, (f, s, lab) in zip(axs, (("7", 314, "7 nm (0 1 3.16), scan 314"), ("4.1", 498, "4.1 nm (0 -1 3.16), scan 498"))):
        t = trace(f, s); a0 = t.index[t.branch != "before"][0]
        x = t.image - a0
        ax.axvspan(0, (t.branch == "hold").sum(), color=GRID, zorder=0)
        ax.plot(x, t.Ti_rel, color=TI, lw=1.1, label="substrate (Ti) spot")
        ax.plot(x, t.sum_rel, color=COL[f], lw=2.0, label="film + substrate (what the CTR measures)")
        ax.set_title(lab, color=COL[f]); ax.set_xlabel("images since the potential program started")
        grid(ax)
    axs[0].set_ylabel("intensity / intensity at the 0 V hold"); axs[0].set_ylim(0.9, 1.25)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h + [Patch(color=GRID, label="30 s hold at 0 V")], l + ["30 s hold at 0 V"], loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "slide05_raw_traces.png")


# ===================================================================== slide 6: CV amplitudes
def slide06():
    fig, ax = plt.subplots(figsize=(W, 6.6))
    Ls = [1.25, 2.75, 2.84, 3.16, 3.25]; off = {"3.3": -0.24, "4.1": 0.0, "7": 0.24}
    seen = {}
    for _, r in amp.iterrows():
        f = f"{r.film_nm:g}"
        if r.kind == "control":
            continue
        xi = Ls.index(round(r.L, 2)) + off[f]
        n = seen.get((f, r.L), 0); seen[(f, r.L)] = n + 1
        xi += 0.07 * (n - 1) if n else 0
        ax.plot(xi, 100 * r.sum_pp, MK[f], ms=11, color=COL[f], mfc="white" if r.scan == 147 else COL[f],
                mec=COL[f] if r.scan == 147 else "white", mew=2 if r.scan == 147 else 1.2, zorder=3)
    ctrl = amp[amp.kind == "control"].reset_index(drop=True)
    xc = len(Ls) + 0.3
    for i, (_, r) in enumerate(ctrl.iterrows()):
        f = f"{r.film_nm:g}"
        ax.plot(xc + (i - (len(ctrl) - 1) / 2) * 0.13, 100 * r.sum_pp, "v", ms=11, color=COL[f], mec="white", mew=1.2, zorder=3)
    ax.set_yscale("log"); ax.set_ylim(0.1, 80)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(list(range(len(Ls))) + [xc]); ax.set_xticklabels([f"(0 ±1 {L:g})" for L in Ls] + ["controls"])
    ax.axvline(len(Ls) - 0.35, color=GRID, lw=1.2)
    ax.set_xlim(-0.6, xc + 0.6)
    ax.set_ylabel("oscillation, peak-to-peak (%)")
    ax.set_title("Best sine per scan, period and phase left free: no timing assumption")
    grid(ax)
    h = [Line2D([], [], marker=MK[f], ls="none", color=COL[f], ms=10, label=f"{f} nm") for f in ("3.3", "4.1", "7")]
    h += [Line2D([], [], marker="v", ls="none", color=INK, ms=10, label="OH-blind controls: (±2 0 L), (-1 0 2.32), (1 0 1.75)"),
          Line2D([], [], marker="s", ls="none", color=COL["3.3"], mfc="white", mew=2, ms=10, label="147: hysteretic; sum mostly substrate")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    save(fig, "slide06_cv_amplitudes.png")


# ===================================================================== slide 7: CV curves
PANELS7 = {("3.3", 1.25): [147], ("3.3", 2.75): [120], ("3.3", 3.16): [222, 223], ("3.3", 3.25): [125],
           ("4.1", 2.75): [496, 497], ("4.1", 2.84): [503], ("4.1", 3.16): [498], ("4.1", 3.25): [501],
           ("7", 2.75): [311, 341, 343, 344], ("7", 3.16): [314], ("7", 3.25): [316]}


def slide07():
    Ls = [1.25, 2.75, 2.84, 3.16, 3.25]; films = ["3.3", "4.1", "7"]
    ylim = {1.25: (-40, 8), 2.75: (-45, 10), 2.84: (-12, 12), 3.16: (-12, 25), 3.25: (-18, 22)}
    fig, axs = plt.subplots(3, 5, figsize=(W, 8.2), sharex=True)
    for i, f in enumerate(films):
        for j, Lv in enumerate(Ls):
            ax = axs[i, j]; scans = PANELS7.get((f, Lv))
            if not scans:
                ax.axis("off"); continue
            for k, s in enumerate(scans):
                c = curve(f, s); key = "Ru_change" if s == 147 else "sum_change"
                for br, ls in (("anodic", "-"), ("cathodic", "--")):
                    d = c[c.branch == br]
                    ax.plot(d.E_V_AgAgCl, 100 * d[key], ls, color=COL[f], lw=2.0 if k == 0 else 1.3, alpha=1 if k == 0 else 0.7)
            ax.axhline(0, color=MUTED, lw=0.7); ax.set_ylim(*ylim[Lv]); grid(ax)
            lab = ", ".join(map(str, scans)) + (" (Ru spot)" if scans == [147] else "")
            ax.set_title(lab, fontsize=11, color=MUTED)
    for i in range(3):
        vis = [j for j in range(5) if axs[i, j].axison]
        axs[i, vis[0]].set_ylabel(f"{films[i]} nm\nchange (%)", color=COL[films[i]])
    for j in range(5):
        below = [i for i in range(3) if axs[i, j].axison]
        axs[below[-1], j].tick_params(labelbottom=True); axs[below[-1], j].set_xlabel("E (V vs Ag/AgCl)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    for j, Lv in enumerate(Ls):
        top = [i for i in range(3) if axs[i, j].axison][0]
        pos = axs[top, j].get_position(); fig.text(pos.x0 + pos.width / 2, 0.905, f"(0 \u00b11 {Lv:g})", ha="center", fontsize=14, color=INK)
    h = [Line2D([], [], color=INK, lw=2, label="anodic sweep"), Line2D([], [], color=INK, lw=2, ls="--", label="cathodic sweep")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False)
    fig.suptitle("Cycle-averaged change from the 0 V hold (film + substrate signal, as in the CTRs)", x=0.01, ha="left", fontsize=15)
    save(fig, "slide07_cv_curves.png")


# ===================================================================== slide 8: CTR rods in measurement order
def rod(film, name):
    d = rd(f"05_ctr_rods/{film}nm/{name}.csv"); return d.L.values, d.F.values ** 2


def ratio(a, b):
    La, Ia = a; Lb, Ib = b
    keep = (La > Lb.min()) & (La < Lb.max()) & (np.abs(La - 3) > 0.08)
    r = np.log(Ia[keep] / np.interp(La[keep], Lb, Ib))
    L, v = La[keep], 100 * (np.exp(r - r.mean()) - 1)
    o = np.argsort(L); L, v = L[o], v[o]
    gap = np.where(np.diff(L) > 0.05)[0]                 # break the line across the excluded Bragg region
    return np.insert(L, gap + 1, np.nan), np.insert(v, gap + 1, np.nan)


def slide08():
    fig, axs = plt.subplots(1, 3, figsize=(W, 5.8), gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.9]))
    a, b, c = axs
    pairs7 = [("rod_0_1_0.7V_b", "rod_0_1_-0.1V_b", "0.7 V vs -0.1 V, back to back", INK, "-", 2.2),
              ("rod_0_1_0.7V_b", "rod_0_1_0.7V_a", "0.7 V vs 0.7 V, hours apart", TI, "-", 1.6),
              ("rod_0_1_-0.1V_b", "rod_0_1_-0.1V_a", "-0.1 V vs -0.1 V, hours apart", TI, "--", 1.6)]
    for na, nb, lab, col, ls, lw in pairs7:
        L, r = ratio(rod("7", na), rod("7", nb))
        a.plot(L, r, ls, color=col, lw=lw, marker="o" if col == INK else None, ms=4, label=lab)
    a.set_title("a  7 nm (0 1): ratio of two rods", color=COL["7"])
    b_pairs = [("rod_0_-1_0V", "rod_0_-1_-0.1V"), ("rod_0_-1_-0.1V", "rod_0_-1_0.15V"), ("rod_0_-1_0.15V", "rod_0_-1_0.7V"),
               ("rod_0_-1_-0.238V", "rod_0_-1_0.35V")]
    for na, nb in b_pairs:
        L, r = ratio(rod("4.1", nb), rod("4.1", na)); b.plot(L, r, color=TI, lw=1.3)
    L, r = ratio(rod("4.1", "rod_0_-1_-0.238V"), rod("4.1", "rod_0_-1_0.7V"))
    b.plot(L, r, color=INK, lw=2.2, marker="o", ms=4)
    b.set_title("b  4.1 nm (0 -1): rods measured back to back", color=COL["4.1"])
    b.legend(handles=[Line2D([], [], color=INK, lw=2.2, marker="o", ms=4, label="0.7 V → -0.238 V (largest step)"),
                      Line2D([], [], color=TI, lw=1.3, label="the other back-to-back pairs")], loc="upper center",
             bbox_to_anchor=(0.5, -0.17), frameon=False)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    for ax in (a, b):
        ax.axvspan(2.92, 3.08, color=GRID, zorder=0); ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_ylim(-45, 45); ax.set_xlabel("L (shaded: Bragg peak, left out)"); grid(ax)
    a.set_ylabel("ratio, scale removed (%)")
    tr = rd("06_ctr_rod_comparisons/rod_pairs.csv").set_index("comparison").rms_ln_ratio
    groups = [("7 nm\nsame\npotential", [tr["7 nm repeat 0.7 V"], tr["7 nm repeat -0.1 V"]], "7", True),
              ("7 nm\n0.7 vs\n-0.1 V", [tr["7 nm 0.7 Va vs -0.1 Va"], tr["7 nm -0.1 Vb -> 0.7 Vb"]], "7", False),
              ("4.1 nm\nback to\nback", [v for k, v in tr.items() if k.startswith("4.1 nm") and "->" in k and "OCV" not in k], "4.1", False)]
    for i, (lab, vals, f, hollow) in enumerate(groups):
        x = i + (np.linspace(-0.15, 0.15, len(vals)) if len(vals) > 1 else np.zeros(1))
        c.plot(x, vals, MK[f], ms=11, color=COL[f], mfc="white" if hollow else COL[f], mec=COL[f] if hollow else "white", mew=2, ls="none")
    c.set_xticks(range(3)); c.set_xticklabels([g[0] for g in groups], fontsize=11); c.set_xlim(-0.5, 2.5)
    c.set_ylim(0, 0.1); c.set_ylabel("rms of ln(ratio) over the rod"); c.set_title("c  Summary"); grid(c)
    fig.tight_layout()
    save(fig, "slide08_ctr_rods.png")


# ===================================================================== slide 9: OH fraction under two assumptions
def slide09():
    d = rd("07_ctr_oh_fraction/oh_fraction_by_assumption.csv"); d = d[d.condition != "OCV"]
    fig, axs = plt.subplots(1, 3, figsize=(W, 6.0))
    for ax, (asm, title) in zip(axs[:2], (("Rao 2017 (logbook RHE scale)", "a  Bridging O from Rao et al. 2017"),
                                          ("bridging O fixed at 0", "b  Bridging O fixed at 0 Å"))):
        for f, off in (("4.1", -0.012), ("7", 0.012)):
            q = d[(d.assumption == asm) & (d.film_nm == float(f))].sort_values("E_V_AgAgCl")
            ax.errorbar(q.E_V_AgAgCl + off, q.x_OH_best, yerr=[q.x_OH_best - q.x_OH_lo, q.x_OH_hi - q.x_OH_best], fmt=MK[f] + "-",
                        color=COL[f], ms=9, mec="white", mew=1, lw=1.8, elinewidth=1.2, capsize=0, label=f"{f} nm")
        ax.set_ylim(-0.05, 1.08); ax.set_title(title); ax.set_xlabel("E (V vs Ag/AgCl)"); grid(ax); rhe_axis(ax)
    axs[0].set_ylabel("OH fraction on the CUS sites (CTR fit)")
    axs[0].legend(loc="lower left", frameon=False)
    ax = axs[2]
    for asm, mk, mfc, lab, off in (("Rao 2017 (logbook RHE scale)", "o", INK, "Rao bridging O", -0.012), ("bridging O fixed at 0", "s", "white", "bridging O at 0 Å", 0.012)):
        q4 = d[(d.assumption == asm) & (d.film_nm == 4.1)].set_index("E_V_AgAgCl"); q7 = d[(d.assumption == asm) & (d.film_nm == 7.0)].set_index("E_V_AgAgCl")
        E = sorted(set(q4.index) & set(q7.index))
        diff = np.array([q4.x_OH_best[e] - q7.x_OH_best[e] for e in E])
        hr = lambda q, e: 0.5 * (q.x_OH_hi[e] - q.x_OH_lo[e])
        err = np.array([np.hypot(hr(q4, e), hr(q7, e)) for e in E])
        ax.errorbar(np.array(E) + off, diff, yerr=err, fmt=mk + "-", color=INK, mfc=mfc, mew=1.5, ms=9, lw=1.6, elinewidth=1.1, capsize=0, label=lab)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_ylim(-0.6, 1.0)
    ax.set_title("c  4.1 nm minus 7 nm"); ax.set_xlabel("E (V vs Ag/AgCl)"); ax.set_ylabel("difference in OH fraction (bars: both ranges)")
    ax.legend(loc="upper right", frameon=False); grid(ax); rhe_axis(ax)
    fig.tight_layout()
    save(fig, "slide09_oh_fraction.png")


# ===================================================================== slide 10: CTR -> CV
def slide10():
    p = rd("08_ctr_to_cv/predicted_from_ctr_states.csv")
    MKA = {"C' Rao (logbook RHE scale)": ("D", INK, "CTR state, Rao bridging O"), "B dz_Obr = 0": ("o", "white", "CTR state, bridging O at 0 Å")}
    panels = [("4.1", "(0 -1 2.75)", [496, 497]), ("4.1", "(0 -1 2.84)", [503]), ("4.1", "(0 -1 3.16)", [498]), ("4.1", "(0 -1 3.25)", [501]),
              ("7", "(0 +1 2.75)", [311, 344]), ("7", "(0 +1 3.16)", [314]), ("7", "(0 +1 3.25)", [316])]
    fig, axs = plt.subplots(2, 4, figsize=(W, 7.6), sharex=True)
    for ax, (f, refl, scans) in zip(axs.flat, panels):
        for k, s in enumerate(scans):
            c = curve(f, s)
            for br, ls in (("anodic", "-"), ("cathodic", "--")):
                q = c[c.branch == br]; ax.plot(q.E_V_AgAgCl, 100 * q.sum_change, ls, color=COL[f], lw=2 if k == 0 else 1.3, alpha=1 if k == 0 else 0.7)
        key = f"{f} nm {refl}"
        for j, (asm, (mk, mfc, lab)) in enumerate(MKA.items()):
            q = p[(p.film_reflection == key) & (p.assumption == asm)]
            ax.errorbar(q.E_V_AgAgCl + (j - 0.5) * 0.03, 100 * q.change_pred, yerr=[100 * (q.change_pred - q.change_lo).abs(), 100 * (q.change_hi - q.change_pred).abs()],
                        fmt=mk, color=INK, mfc=mfc, mew=1.4, ms=8, elinewidth=0.9, capsize=0, zorder=4)
        ax.axvspan(-0.3, 0, color="#f3f2ee", zorder=0); ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_title(f"{f} nm {refl.replace('+1', '1')}", color=COL[f], fontsize=13); grid(ax); ax.set_xlim(-0.3, 0.76)
    axs[0, 0].set_ylabel("change from 0 V (%)"); axs[1, 0].set_ylabel("change from 0 V (%)")
    for ax in axs[1]:
        ax.set_xlabel("E (V vs Ag/AgCl)")
    for ax in axs[0]:
        ax.tick_params(labelbottom=True)
    axs[1, 3].axis("off")
    h = [Line2D([], [], color=INK, lw=2, label="CV, anodic"), Line2D([], [], color=INK, lw=2, ls="--", label="CV, cathodic")]
    h += [Line2D([], [], marker=mk, ls="none", color=INK, mfc=mfc, mew=1.4, ms=9, label=lab) for mk, mfc, lab in MKA.values()]
    h += [Patch(color="#f3f2ee", label="outside the CV window")]
    axs[1, 3].legend(handles=h, loc="center", frameon=False, title="nothing adjusted to the CVs;\nbars: OH range at that potential", title_fontsize=12)
    fig.tight_layout()
    save(fig, "slide10_ctr_to_cv.png")


# ===================================================================== slide 11: CTR structure + Rao bridging O fitted to the CVs
def slide11():
    M = model(); m = M["m"]
    fit = json.load(open(os.path.join(here, "cvfit_sum.json")))["7 nm | x_OH only"]
    x0, x1, E0, w = fit["x0"], fit["x1"], fit["E0"], fit["w"]; eE0, ew = fit["errors"]["E0"], fit["errors"]["w"]
    sig = lambda z: 0.5 * (1 + np.tanh(z))
    Eg = np.linspace(0, 0.7, 71)
    fig, axs = plt.subplots(1, 3, figsize=(W, 5.8), gridspec_kw=dict(width_ratios=[1, 1, 1]))
    a, b, c = axs
    # a: the fitted step position (normalised), with the E0 uncertainty
    for dE in (-eE0, eE0):
        a.plot(Eg, sig((Eg - (E0 + dE)) / w), color=COL["7"], lw=1, alpha=0.5)
    a.fill_between(Eg, sig((Eg - (E0 - eE0)) / w), sig((Eg - (E0 + eE0)) / w), color=COL["7"], alpha=0.2, lw=0)
    a.plot(Eg, sig((Eg - E0) / w), color=COL["7"], lw=2.4)
    a.axhline(0, color=MUTED, lw=0.7)
    a.set_ylim(-0.08, 1.12); a.set_xlabel("E (V vs Ag/AgCl)"); a.set_ylabel("fraction of the OH change completed")
    a.set_title("a  Where the surface switches")
    a.legend(handles=[Line2D([], [], color=COL["7"], lw=2.4, label=f"7 nm: E0 = {E0:.2f} ± {eE0:.2f} V, width {w:.2f} V"),
                      Patch(color=COL["7"], alpha=0.25, label="E0 ± its error")],
             loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    grid(a); rhe_axis(a)
    # b: 7 nm data with the fitted model (offset per scan as in the fit)
    v7 = dict(view("7"), dz_Obr=0.35)
    for refl, (H, K, L), scans, mk in (("2.75", (0, 1, 2.75), [311, 344], "o"), ("3.16", (0, 1, 3.16), [314], "s"), ("3.25", (0, 1, 3.25), [316], "^")):
        I0 = m.intensity(H, K, [L], dict(v7, x_OH=x0 + (x1 - x0) * sig((0 - E0) / w)))[0]
        mod = np.array([m.intensity(H, K, [L], dict(v7, x_OH=x0 + (x1 - x0) * sig((e - E0) / w)))[0] / I0 - 1 for e in Eg])
        for s in scans:
            cc = curve("7", s); g = cc.groupby("E_V_AgAgCl").sum_change.mean()
            off = float(np.mean(g.values - np.interp(g.index.values, Eg, mod)))
            b.plot(g.index, 100 * g.values, mk, color=COL["7"], ms=6, alpha=0.8, mec="white", mew=0.6)
            b.plot(Eg, 100 * (mod + off), color=INK, lw=1.6)
    b.axhline(0, color=MUTED, lw=0.7); b.set_xlabel("E (V vs Ag/AgCl)"); b.set_ylabel("change from 0 V (%)")
    b.set_title("b  7 nm CVs and the fitted step", color=COL["7"]); grid(b)
    b.legend(handles=[Line2D([], [], marker="o", ls="none", color=COL["7"], ms=7, label="(0 1 2.75), scans 311, 344"),
                      Line2D([], [], marker="s", ls="none", color=COL["7"], ms=7, label="(0 1 3.16), scan 314"),
                      Line2D([], [], marker="^", ls="none", color=COL["7"], ms=7, label="(0 1 3.25), scan 316"),
                      Line2D([], [], color=INK, lw=1.6, label="model, one OH step")], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False, fontsize=11)
    # c: 4.1 nm data against what the same step would do to the 4.1 nm structure
    v4 = dict(view("4.1"), dz_Obr=0.35)
    for (H, K, L), scans, mk in (((0, -1, 2.75), [496, 497], "o"), ((0, -1, 3.16), [498], "s")):
        for s in scans:
            cc = curve("4.1", s); g = cc.groupby("E_V_AgAgCl").sum_change.mean()
            c.plot(g.index, 100 * g.values, mk, color=COL["4.1"], ms=6, alpha=0.85, mec="white", mew=0.6)
        for dx, ls in ((x1 - x0, "-"), (0.03, ":")):
            I0 = m.intensity(H, K, [L], dict(v4, x_OH=x0))[0]
            mod = np.array([m.intensity(H, K, [L], dict(v4, x_OH=x0 + dx * sig((e - E0) / w) - dx * sig(-E0 / w)))[0] / I0 - 1 for e in Eg])
            c.plot(Eg, 100 * mod, ls, color=INK, lw=1.6)
    c.axhline(0, color=MUTED, lw=0.7); c.set_xlabel("E (V vs Ag/AgCl)"); c.set_ylabel("change from 0 V (%)")
    c.set_title("c  4.1 nm CVs against the same step", color=COL["4.1"]); grid(c)
    c.legend(handles=[Line2D([], [], marker="o", ls="none", color=COL["4.1"], ms=7, label="(0 -1 2.75), scans 496, 497"),
                      Line2D([], [], marker="s", ls="none", color=COL["4.1"], ms=7, label="(0 -1 3.16), scan 498"),
                      Line2D([], [], color=INK, lw=1.6, label="7 nm-sized OH step"),
                      Line2D([], [], color=INK, lw=1.6, ls=":", label="0.03 OH step (the limit)")], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False, fontsize=11)
    fig.tight_layout()
    save(fig, "slide11_cv_fit.png")


# ===================================================================== slide 12: putting it together
def slide12():
    fig, ax = plt.subplots(figsize=(W, 5.2))
    LOW, HIGH = "#dddcd5", "#6b6a63"
    lanes = [("3.3", 0.0, 0.7, (0.22, 0.30), -0.015), ("7", -0.1, 0.7, (0.17, 0.26), -0.02), ("4.1", -0.238, 0.7, None, 0.378)]
    for i, (f, lo, hi, half, ocv) in enumerate(lanes):
        y = 2 - i
        if half:
            ax.add_patch(Rectangle((lo, y - 0.22), half[0] - lo, 0.44, color=LOW, lw=0))
            xs = np.linspace(half[0], half[1], 30)
            for x_, x2 in zip(xs[:-1], xs[1:]):
                t = (x_ - half[0]) / (half[1] - half[0]); col = np.array(matplotlib.colors.to_rgb(LOW)) * (1 - t) + np.array(matplotlib.colors.to_rgb(HIGH)) * t
                ax.add_patch(Rectangle((x_, y - 0.22), x2 - x_, 0.44, color=col, lw=0))
            ax.add_patch(Rectangle((half[1], y - 0.22), hi - half[1], 0.44, color=HIGH, lw=0))
        else:
            ax.add_patch(Rectangle((lo, y - 0.22), hi - lo, 0.44, color=HIGH, lw=0))
        ax.plot(ocv, y + 0.36, "v", ms=12, color=COL[f], zorder=4)
        ax.text(-0.43, y, f"{f} nm", color=COL[f], fontsize=15, va="center", ha="left", fontweight="bold")
    ax.set_xlim(-0.45, 0.78); ax.set_ylim(-0.6, 2.7); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("E (V vs Ag/AgCl)"); grid(ax, "x"); rhe_axis(ax)
    ax.set_title("Measured range of each film, coloured by the surface state the data show")
    h = [Patch(color=LOW, label="low-potential state"), Patch(color=HIGH, label="high-potential state (more OH-like in the CTR model)"),
         Patch(facecolor="#a5a49d", label="switch: half of the change is reached in this band"),
         Line2D([], [], marker="v", ls="none", color=INK, ms=11, label="rest potential (logbook)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    save(fig, "slide12_summary_diagram.png")


# ===================================================================== backups
def slideB2():
    scans = [("3.3", 120, "(0 -1 2.75)"), ("3.3", 222, "(0 -1 3.16)"), ("3.3", 125, "(0 -1 3.25)"), ("3.3", 147, "(0 -1 1.25)"),
             ("7", 311, "(0 1 2.75)"), ("7", 341, "(0 1 2.75)"), ("7", 344, "(0 1 2.75)"), ("7", 314, "(0 1 3.16)"), ("7", 316, "(0 1 3.25)"), ("7", 343, "(0 1 2.75)")]
    fig, axs = plt.subplots(2, 5, figsize=(W, 7.0), sharex=True)
    for ax, (f, s, refl) in zip(axs.flat, scans):
        c = curve(f, s); g = c.groupby("E_V_AgAgCl")[["Ru_change", "Ti_change"]].mean()
        ax.plot(g.index, 100 * g.Ti_change, color=TI, lw=2, label="substrate (Ti) spot")
        ax.plot(g.index, 100 * g.Ru_change, color=COL[f], lw=2.2, label="film (Ru) spot")
        ax.axhline(0, color=MUTED, lw=0.7); grid(ax)
        ax.set_title(f"{s}: {f} nm {refl}", fontsize=12, color=COL[f])
    for ax in axs[1]:
        ax.set_xlabel("E (V vs Ag/AgCl)")
    for ax in axs[:, 0]:
        ax.set_ylabel("change (%), sweeps averaged")
    fig.legend(handles=[Line2D([], [], color=INK, lw=2.2, label="film (Ru) spot, in the film colour"), Line2D([], [], color=TI, lw=2, label="substrate (Ti) spot")],
               loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False)
    fig.suptitle("The response sits in the film spot of the split peak", x=0.01, ha="left", fontsize=15)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, "slideB2_ru_ti_spots.png")


def slideB3():
    d = rd("07_ctr_oh_fraction/oh_fraction_by_assumption.csv"); d = d[(d.condition != "OCV") & (d.assumption == "bridging O fitted (unconstrained)")]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for f, off in (("4.1", -0.012), ("7", 0.012)):
        q = d[d.film_nm == float(f)].sort_values("E_V_AgAgCl")
        ax.errorbar(q.E_V_AgAgCl + off, q.x_OH_best, yerr=[q.x_OH_best - q.x_OH_lo, q.x_OH_hi - q.x_OH_best], fmt=MK[f] + "-", color=COL[f],
                    ms=9, mec="white", mew=1, lw=1.8, elinewidth=1.2, capsize=0, label=f"{f} nm")
    ax.set_ylim(-0.05, 1.08); ax.set_xlabel("E (V vs Ag/AgCl)"); ax.set_ylabel("OH fraction (CTR fit)")
    ax.set_title("Bridging O fitted together with OH: the two trade off, so the OH fraction is unconstrained")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False); grid(ax); rhe_axis(ax)
    save(fig, "slideB3_bridging_O_free.png")


def slideB5():
    k = rd("10_work_function/kpfm_jeong2026_read_from_plot.csv")
    fig, ax = plt.subplots(figsize=(W, 5.8))
    for f, t in (("3.3", 3.3), ("4.1", 4.1), ("7", 7.0)):
        ax.axvspan(t - 0.3, t + 0.3, color=COL[f], alpha=0.15, lw=0)
        ax.axvline(t, color=COL[f], lw=2)
    ax.plot(k.thickness_nm, k.work_function_eV, "-", color=INK, lw=1.4)
    ax.plot(k.thickness_nm, k.work_function_eV, "s", color=INK, ms=8)
    ax.set_xlabel("RuO2 thickness (nm)"); ax.set_ylabel("work function (eV)"); ax.set_xlim(0.5, 18); ax.set_ylim(4.1, 5.55)
    ax.set_title("Work function (Jeong et al. 2026, KPFM, different samples) with our films marked")
    grid(ax)
    h = [Line2D([], [], marker="s", color=INK, ms=8, label="KPFM, read from the published plot"),
         Patch(color=COL["3.3"], alpha=0.3, label="3.3 nm: switches"), Patch(color=COL["4.1"], alpha=0.3, label="4.1 nm: does not switch"),
         Patch(color=COL["7"], alpha=0.3, label="7 nm: switches"), Patch(color=MUTED, alpha=0.15, label="band: ±0.3 nm thickness-scale offset")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    save(fig, "slideB5_work_function.png")


def slideB7():
    fig, axs = plt.subplots(1, 2, figsize=(W, 5.6))
    for ax, (s, L, ref, refcol, reflab) in zip(axs, ((225, 2.75, ("3.3", 120), "Ru_change", "CV 120, film (Ru) spot"),
                                                    (224, 3.16, ("3.3", 223), "sum_change", "CV 223, same ROI type"))):
        lv = rd(f"04_staircases/s{s}_0_-1_{L:g}_levels.csv")
        ax.errorbar(lv.E_V_AgAgCl, 100 * lv.change, yerr=100 * lv.err, fmt="o-", color=INK, ms=8, lw=1.8, capsize=0, label=f"staircase {s}, level means")
        c = curve(*ref); g = c.groupby("E_V_AgAgCl")[refcol].mean()
        ax.plot(g.index, 100 * g.values, "--", color=COL["3.3"], lw=2, label=reflab + " (sweeps averaged)")
        ax.axhline(0, color=MUTED, lw=0.7); grid(ax)
        ax.set_title(f"3.3 nm (0 -1 {L:g})", color=COL["3.3"]); ax.set_xlabel("E (V vs Ag/AgCl)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    axs[0].set_ylabel("change from 0 V (%)")
    fig.suptitle("Potential staircases (after the window broke): they disagree with the CVs and with each other", x=0.01, ha="left", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "slideB7_staircases.png")


if __name__ == "__main__":
    for fn in (slide02, slide03, slide04, slide05, slide06, slide07, slide08, slide09, slide10, slide11, slide12, slideB2, slideB3, slideB5, slideB7):
        fn()


# ===================================================================== slide 12b: percent of the transition completed, all three films
def transition_data():
    """Model-free: each responding scan's cycle-averaged change (both sweeps averaged), scaled between its own first point
    (0-0.05 V) and its plateau (mean of the last three bins, 0.575-0.675 V), as a percent. A step is fitted per film. 4.1 nm: its change from 0 to 0.7 V as a fraction
    of the 7 nm change at the same reflection (linear fits, cv_amplitudes.csv)."""
    from scipy.optimize import curve_fit
    sig = lambda z: 0.5 * (1 + np.tanh(z))
    def norm_step(E, E0, w):
        a, b = sig((0 - E0) / w), sig((0.7 - E0) / w)
        return 100 * (sig((E - E0) / w) - a) / (b - a)
    use = {"7": [(311, 2.75), (341, 2.75), (343, 2.75), (344, 2.75), (314, 3.16)],     # like for like: L 2.75 and 3.16 only
           "3.3": [(120, 2.75), (222, 3.16), (223, 3.16)]}                              # (3.25: only the noisy 316 on 7 nm)
    out = {}
    for f, scans in use.items():
        Es, ps, pts = [], [], []
        for s, L in scans:
            c = curve(f, s); g = c.groupby("E_V_AgAgCl").sum_change.mean()
            low, plateau = g.iloc[0], g.iloc[-3:].mean()            # own ends: robust to a drifting baseline (223)
            p = 100 * (g.values - low) / (plateau - low)
            E_ = g.index.values; i = int(np.argmax(p >= 50))
            h50 = float(E_[i - 1] + (50 - p[i - 1]) * (E_[i] - E_[i - 1]) / (p[i] - p[i - 1])) if i > 0 else float(E_[0])
            pts.append((s, L, E_, p, h50)); Es += list(E_); ps += list(p)
        # film curve: the mean over its scans in each E bin (all scans share the bins); half-way where it crosses 50%
        E_ = pts[0][2]
        P_ = np.array([q[3] for q in pts]); mean = P_.mean(axis=0); se = P_.std(axis=0, ddof=1) / np.sqrt(len(pts))
        k = int(np.argmax(mean >= 50)); E_half = float(E_[k - 1] + (50 - mean[k - 1]) * (E_[k] - E_[k - 1]) / (mean[k] - mean[k - 1]))
        h = np.array([q[4] for q in pts])
        out[f] = dict(points=pts, E=E_, mean=mean, se=se, E_half=E_half, scan_half=(float(h.min()), float(h.max())),
                      E_half_err=float(h.std(ddof=1) / np.sqrt(len(h))))
    a = amp.set_index("scan")
    ref = {2.75: a.loc[[311, 341, 343, 344], "sum_change_0_to_0.7V"].mean(), 3.16: a.loc[314, "sum_change_0_to_0.7V"],
           3.25: a.loc[316, "sum_change_0_to_0.7V"]}
    r, e = [], []
    for s, L in ((496, 2.75), (497, 2.75), (498, 3.16), (501, 3.25)):
        r.append(a.loc[s, "sum_change_0_to_0.7V"] / ref[L]); e.append(abs(a.loc[s, "sum_change_err"] / ref[L]))
    r, e = np.array(r), np.array(e); wgt = 1 / e ** 2
    m41 = float(np.sum(wgt * r) / np.sum(wgt)); s41 = float(1 / np.sqrt(np.sum(wgt)))
    out["4.1"] = dict(ratio=m41, ratio_err=s41, per_scan=list(zip((496, 497, 498, 501), r, e)))
    return out


def slide12b():
    d = transition_data()
    fig, ax = plt.subplots(figsize=(W, 6.4))
    MKL = {2.75: "o", 3.16: "s", 3.25: "^"}
    for f in ("7", "3.3"):
        for s, L, E, p, h50 in d[f]["points"]:
            ax.plot(E, p, MKL[L], ms=5.5, color=COL[f], alpha=0.55, mec="none", zorder=2)
            ax.plot(h50, 122 if f == "7" else 116, MKL[L], ms=9, color=COL[f], mec="white", mew=1, zorder=4, clip_on=False)
        ax.fill_between(d[f]["E"], d[f]["mean"] - d[f]["se"], d[f]["mean"] + d[f]["se"], color=COL[f], alpha=0.15, lw=0, zorder=1)
        ax.plot(d[f]["E"], d[f]["mean"], color=COL[f], lw=3, zorder=3)
    q = d["4.1"]; band = 100 * (abs(q["ratio"]) + 2 * q["ratio_err"])
    ax.fill_between([0, 0.7], 100 - band, 100 + band, color=COL["4.1"], alpha=0.18, lw=0, zorder=1)
    ax.plot([0, 0.7], [100, 100], color=COL["4.1"], lw=3, zorder=3)
    ax.plot([-0.238, 0], [100, 100], color=COL["4.1"], lw=3, ls=(0, (2, 2)), zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8); ax.axhline(50, color=GRID, lw=1.2, ls="--", zorder=0)
    ax.set_xlim(-0.3, 0.75); ax.set_ylim(-25, 128)
    ax.set_xlabel("E (V vs Ag/AgCl)"); ax.set_ylabel("OH transition completed (%)\n0 = first CV point (0-0.05 V), 100 = plateau")
    ax.set_title("How far each surface has gone from its low-potential state to the high-potential state")
    grid(ax); rhe_axis(ax)
    h = [Line2D([], [], color=COL["7"], lw=3, label=f"7 nm (5 scans): half-way at {d['7']['E_half']:.2f} V; scans {d['7']['scan_half'][0]:.2f}-{d['7']['scan_half'][1]:.2f} V"),
         Line2D([], [], color=COL["3.3"], lw=3, label=f"3.3 nm (3 scans): half-way at {d['3.3']['E_half']:.2f} V; scans {d['3.3']['scan_half'][0]:.2f}-{d['3.3']['scan_half'][1]:.2f} V"),
         Line2D([], [], color=COL["4.1"], lw=3, label="4.1 nm: already there (CTR comparison); dashed: CTRs only, no CVs"),
         Patch(color=COL["4.1"], alpha=0.25, label=f"4.1 nm: change its CVs allow, ±{band:.0f}% of the 7 nm change"),
         Line2D([], [], marker="o", ls="none", color=MUTED, ms=7, label="CV points, L 2.75 (one colour per film)"),
         Line2D([], [], marker="s", ls="none", color=MUTED, ms=7, label="CV points, L 3.16"),
         Line2D([], [], marker="D", ls="none", color=MUTED, ms=0, label="markers above 110%: each scan's 50% point; bands: mean \u00b1 1 s.e.")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=11.5)
    save(fig, "slide12b_transition_completed.png")
    return d


# ===================================================================== slide 6b: the response at the applied potential (clock-synced)
def slide06b():
    r = rd("11_cv_synced/response.csv")
    fig, ax = plt.subplots(figsize=(W, 6.6))
    cols = [("0 ±1 2.75", 2.75), ("0 ±1 2.84", 2.84), ("0 ±1 3.16", 3.16), ("0 ±1 3.25", 3.25)]
    off = {"3.3": -0.24, "4.1": 0.0, "7": 0.24}
    for i, (lab, L) in enumerate(cols):
        for f in ("3.3", "4.1", "7"):
            q = r[(r.film_nm.astype(str).str.rstrip("0").str.rstrip(".") == f) & (r.L.round(2) == L) & (r.K != 0) & (r.scan != 344)]
            for j, (_, x) in enumerate(q.iterrows()):
                xi = i + off[f] + (j - (len(q) - 1) / 2) * 0.07
                ax.errorbar(xi, 100 * x.A_step, yerr=100 * x.err_step, fmt=MK[f], ms=10, color=COL[f], mec="white", mew=1.0,
                            elinewidth=1.6, capsize=0, zorder=3)
    c = r[r.scan == 519].iloc[0]
    xc = len(cols) + 0.1
    ax.errorbar(xc, 100 * c.A_step, yerr=100 * c.err_step, fmt="v", ms=11, color=COL["4.1"], mec="white", elinewidth=1.6, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
    ax.axvline(len(cols) - 0.45, color=GRID, lw=1.2)
    ax.set_xticks(list(range(len(cols))) + [xc]); ax.set_xticklabels([f"({c[0]})" for c in cols] + ["4.1 nm control\n(-2 0 1.75)"])
    ax.set_xlim(-0.6, xc + 0.6)
    ax.set_ylabel("X-ray change from 0 to 0.7 V (%)")
    ax.set_title("At the potential EC-Lab applied (timing from the instrument clocks, nothing fitted to the timing)")
    grid(ax)
    s = rd("11_cv_synced/summary.csv")
    def ratio(L, ref):
        q = s[(s.L.astype(str) == L) & (s.film_nm.astype(str).str.startswith("4.1"))].iloc[0]
        return q[f"ratio_to_{ref}"], q[f"ratio_err_{ref}"]
    b = {L: (100 * (ratio(L, "3p3")[0] + 2 * ratio(L, "3p3")[1]), 100 * (ratio(L, "7")[0] + 2 * ratio(L, "7")[1])) for L in ("2.75", "3.16")}
    h = [Line2D([], [], marker=MK[f], ls="none", color=COL[f], ms=10, label=f"{f} nm, one point per scan") for f in ("3.3", "4.1", "7")]
    h += [Line2D([], [], marker="v", ls="none", color=COL["4.1"], ms=10, label="OH-blind control, 4.1 nm (519)"),
          Line2D([], [], ls="none", label=f"4.1 nm response at most {b['2.75'][0]:.0f}% / {b['3.16'][0]:.0f}% of 3.3 nm and "
                 f"{b['2.75'][1]:.0f}% / {b['3.16'][1]:.0f}% of 7 nm at 2.75 / 3.16 (2 sigma)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=11.5)
    save(fig, "slide06b_cv_response_synced.png")


if __name__ == "__main__":
    slide12b()
    slide06b()

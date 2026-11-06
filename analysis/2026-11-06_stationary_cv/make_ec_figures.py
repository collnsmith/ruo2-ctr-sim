"""EC-Lab data of the beamtime (data/echem/, .mpr read with galvani): voltammograms per film, potential and current against
time, charging current and charge per CV, and the current at potential steps. Figures in ec_figures/, numbers in ec_summary.json.

Files are matched to the X-ray scans by their names. Two names do not carry the SPEC number: 3.3 nm 'scan119' is CV 120 (the
first (0 -1 2.75) CV of day 2), and on 7 nm 'scan342' and 'scan247' are taken as CVs 343 and 344 (order and duration). The day-4
files are named '7nm' but hold the 4.1 nm session (scans 483-524 in the logbook). Run: python make_ec_figures.py"""
import glob, json, os
here = os.path.dirname(os.path.abspath(__file__))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from galvani import BioLogic

ROOT = os.path.join(here, "data", "echem")
OUT = os.path.join(here, "ec_figures"); os.makedirs(OUT, exist_ok=True)
COL = {"3.3": "#2a78d6", "4.1": "#eb6834", "7": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 11.5, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
                     "savefig.dpi": 150})
D2, D3, D4 = "Collin_08102026", "Collin_08112026", "Collin_08122026"
CVS = {"3.3": [("120", D2, "*scan119_02_CV_C01.mpr"), ("125", D2, "*scan125_02_CV_C01.mpr"), ("147", D2, "*scan147_02_CV_C01.mpr"),
               ("222", D2, "*scan222_02_CV_C01.mpr"), ("223", D2, "*scan223_02_CV_C01.mpr"), ("126 c", D2, "*scan126_02_CV_C01.mpr"),
               ("135 c", D2, "*scan135_02_CV_C01.mpr"), ("227 c", D2, "*scan227_02_CV_C01.mpr")],
       "7": [("311", D3, "*scan311_02_CV_C01.mpr"), ("314", D3, "*scan314_02_CV_C01.mpr"), ("316", D3, "*scan316_02_CV_C01.mpr"),
             ("341", D3, "*scan341_02_CV_C01.mpr"), ("343", D3, "*scan342_02_CV_C01.mpr"), ("344", D3, "scan247_02_CV_C01.mpr"),
             ("356 c", D3, "CV_7nm_scan356_02_CV_C01.mpr")],
       "4.1": [("496", D4, "*scan496_02_CV_C01.mpr"), ("497", D4, "*scan497_lower_02_CV_C01.mpr"), ("498", D4, "*scan498_lower_02_CV_C01.mpr"),
               ("501", D4, "*scan501_02_CV_C01.mpr"), ("503", D4, "*scan503_02_CV_C01.mpr"), ("519 c", D4, "*scan519_02_CV_C01.mpr")]}
BINS = np.linspace(0, 0.7, 15); MID = 0.5 * (BINS[1:] + BINS[:-1])


def load(day, pat):
    fs = sorted(glob.glob(os.path.join(ROOT, day, pat)))
    T, C, E, I = [], [], [], []
    for f in fs:
        d = BioLogic.MPRfile(f).data
        T.append(d["time/s"]); C.append(d["control/V"]); E.append(d["Ewe/V"]); I.append(d["<I>/mA"] * 1e3)
    t = np.concatenate(T); o = np.argsort(t, kind="stable")
    return t[o], np.concatenate(C)[o], np.concatenate(E)[o], np.concatenate(I)[o]


def per_sweep(t, c, E, I):
    rising = np.gradient(c, t) > 0
    out = {}
    for br, m in (("anodic", rising), ("cathodic", ~rising)):
        out[br] = np.array([np.median(I[m & (E >= lo) & (E < hi)]) if np.sum(m & (E >= lo) & (E < hi)) > 5 else np.nan
                            for lo, hi in zip(BINS[:-1], BINS[1:])])
    return out


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white"); plt.close(fig); print("  ", name)


summary = {}
data = {}
for f, items in CVS.items():
    for lab, day, pat in items:
        t, c, E, I = load(day, pat)
        sw = per_sweep(t, c, E, I)
        dt = np.gradient(t); rising = np.gradient(c, t) > 0
        ncyc = (t[-1] - t[0]) / 280.0
        half = float(np.nanmedian((sw["anodic"] - sw["cathodic"])[3:11]) / 2)        # 0.15-0.55 V
        summary[f"{f} {lab}"] = dict(film=f, scan=lab, half_width_uA=half, C_uF=half / 5e-3,
                                     anodic_charge_per_cycle_uC=float(np.sum(I[rising] * dt[rising]) / ncyc),
                                     net_charge_uC=float(np.sum(I * dt)), noise_uA=float(np.std(I)),
                                     Ewe_minus_control_std_mV=float(np.std(E - c) * 1e3))
        data[(f, lab)] = (t, c, E, I, sw)

# ----------------------------------------------------------------- 1. voltammograms
fig, axs = plt.subplots(1, 4, figsize=(15.5, 5.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.9]))
for ax, f in zip(axs[:3], ("3.3", "7", "4.1")):
    for lab, day, pat in CVS[f]:
        t, c, E, I, sw = data[(f, lab)]
        ax.plot(MID, sw["anodic"], "-", color=COL[f] if not lab.endswith("c") else MUTED, lw=1.8, alpha=0.9)
        ax.plot(MID, sw["cathodic"], "--", color=COL[f] if not lab.endswith("c") else MUTED, lw=1.8, alpha=0.9)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_ylim(-6, 16); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.set_title(f"{f} nm", color=COL[f]); ax.set_xlabel("E (V vs Ag/AgCl)")
axs[0].set_ylabel("current (µA), median per 0.05 V bin")
ax = axs[3]
for lab, day, pat in CVS["4.1"]:
    t, c, E, I, sw = data[("4.1", lab)]
    ax.plot(E[::3], I[::3], ",", color="#c9c8c0", alpha=0.6, zorder=0)
    ax.plot(MID, sw["anodic"], "-", color=COL["4.1"], lw=1.8); ax.plot(MID, sw["cathodic"], "--", color=COL["4.1"], lw=1.8)
ax.axhline(0, color=MUTED, lw=0.8); ax.set_ylim(-1, 1); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
ax.set_title("4.1 nm, zoomed (grey: raw points)", color=COL["4.1"]); ax.set_xlabel("E (V vs Ag/AgCl)")
h = [Line2D([], [], color=INK, lw=1.8, label="anodic sweep"), Line2D([], [], color=INK, lw=1.8, ls="--", label="cathodic sweep"),
     Line2D([], [], color=MUTED, lw=1.8, label="CV on a control reflection (same electrode)")]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False)
fig.suptitle("Stationary CVs, 5 mV/s, 0 to 0.7 V: every scan, cycles combined", x=0.01, ha="left", fontsize=15)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
save(fig, "ec01_voltammograms.png")

# ----------------------------------------------------------------- 2. potential and current against time
fig, axs = plt.subplots(2, 3, figsize=(15.5, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.4]))
for j, (f, lab) in enumerate((("3.3", "120"), ("7", "341"), ("4.1", "496"))):
    t, c, E, I, sw = data[(f, lab)]
    axs[0, j].plot(t, E, color=COL[f], lw=1); axs[0, j].plot(t, c, color=INK, lw=0.8, ls=":")
    axs[0, j].set_title(f"{f} nm, CV {lab}", color=COL[f]); axs[0, j].set_ylim(-0.1, 0.8); axs[0, j].grid(axis="y", color=GRID)
    k = max(1, len(I) // 3000)
    axs[1, j].plot(t[::k], I[::k], color="#c9c8c0", lw=0.5)
    w = 25; sm = np.convolve(I, np.ones(w) / w, mode="same")
    axs[1, j].plot(t[::k], sm[::k], color=COL[f], lw=1.6)
    axs[1, j].axhline(0, color=MUTED, lw=0.8); axs[1, j].set_ylim(-6, 14); axs[1, j].grid(axis="y", color=GRID)
    axs[1, j].set_xlabel("time since the potential program started (s)")
axs[0, 0].set_ylabel("E (V vs Ag/AgCl)"); axs[1, 0].set_ylabel("current (µA)")
h = [Line2D([], [], color=INK, lw=1, label="measured potential (Ewe)"), Line2D([], [], color=INK, lw=0.8, ls=":", label="programmed potential"),
     Line2D([], [], color="#c9c8c0", lw=1, label="current, raw"), Line2D([], [], color=INK, lw=1.6, label="current, running mean")]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)
fig.suptitle("Same programme on every film; the potential follows it everywhere, the current does not", x=0.01, ha="left", fontsize=15)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
save(fig, "ec02_time_traces.png")

# ----------------------------------------------------------------- 3. per-scan summary
fig, axs = plt.subplots(1, 2, figsize=(14, 5.6))
xs = {"3.3": 0, "7": 1, "4.1": 2}; rng = np.random.default_rng(4)
for k, v in summary.items():
    f = v["film"]; x = xs[f] + rng.uniform(-0.2, 0.2); ctrl = v["scan"].endswith("c")
    for ax, key in zip(axs, ("half_width_uA", "anodic_charge_per_cycle_uC")):
        y = v[key]; top = 2.5 if key == "half_width_uA" else 1200
        clipped = y > top
        ax.plot(x, min(y, top * 0.98), "^" if clipped else ("v" if ctrl else "o"), ms=10, color=COL[f], mfc="white" if ctrl else COL[f], mew=2, zorder=3)
axs[0].set_ylim(-0.1, 2.5); axs[1].set_ylim(-50, 1200)
axs[0].set_ylabel("half the gap between sweeps, 0.15-0.55 V (µA)"); axs[1].set_ylabel("charge passed per anodic sweep (µC)")
axs[0].set_title("a  Charging current"); axs[1].set_title("b  Charge per cycle")
for ax in axs:
    ax.set_xticks(range(3)); ax.set_xticklabels(["3.3 nm", "7 nm", "4.1 nm"]); ax.axhline(0, color=MUTED, lw=0.8)
    for t_, f in zip(ax.get_xticklabels(), ("3.3", "7", "4.1")):
        t_.set_color(COL[f])
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
h = [Line2D([], [], marker="o", ls="none", color=INK, ms=9, label="CV, surface-sensitive reflection"),
     Line2D([], [], marker="v", ls="none", color=INK, mfc="white", mew=2, ms=9, label="CV, control reflection"),
     Line2D([], [], marker="^", ls="none", color=INK, ms=9, label="above the axis (spiky 7 nm scans 311, 314, 344)")]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "ec03_summary.png")

# ----------------------------------------------------------------- 4. current at potential steps
STEP = [("3.3", "3.3 nm staircase 225 (0.05 V steps)", D2, "*scan225_*_CA_C01.mpr"),
        ("7", "7 nm CTR holds 325-340", D3, "0p1M_H2SO4_5mVs_CA00_Lscan_7nm_scan323_C01.mpr"),
        ("4.1", "4.1 nm CTR holds 483-494", D4, "CA_7nm_scan483and484_C01.mpr")]
fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.4))
for ax, (f, title, day, pat) in zip(axs, STEP):
    t, c, E, I = load(day, pat)
    jumps = np.where(np.abs(np.diff(c)) > 0.02)[0]
    for j in jumps:
        w = (t > t[j] - 5) & (t < t[j] + 25)
        base = np.median(I[(t > t[j] - 30) & (t < t[j] - 1)]) if np.any((t > t[j] - 30) & (t < t[j] - 1)) else 0.0
        ax.plot(t[w] - t[j], I[w] - base, color=COL[f], lw=1, alpha=0.75)
    ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_title(title, color=COL[f], fontsize=13); ax.set_xlabel("time from the potential step (s)"); ax.grid(axis="y", color=GRID)
    lim = {"3.3": 8, "7": 60, "4.1": 1}[f]; ax.set_ylim(-lim, lim)
    summary[f"steps {f}"] = dict(n_steps=int(len(jumps)))
axs[0].set_ylabel("current minus the level before the step (µA)")
fig.suptitle("Current at each potential step (note the different scales: 8, 60 and 1 µA)", x=0.01, ha="left", fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "ec04_potential_steps.png")
json.dump(summary, open(os.path.join(here, "ec_summary.json"), "w"), indent=1)

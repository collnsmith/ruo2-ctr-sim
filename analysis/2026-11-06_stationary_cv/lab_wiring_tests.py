"""Lab wiring tests (2026-11-09, SP-300, grounded channel, dummy cell DC3 Randles circuit) against the day-4 beamline record.
Each test: CA 30 s at 0 V, then CV 0 to 0.7 V at 50 mV/s (no_WE: CV only, started vs Eoc). Files in data/lab_wiring_tests_20261109.
Configurations (user's labels): all = normal; gnd_to_red = ground lead also on the WE; no_WE = WE power and sense both off;
no_p1 = WE power off; no_p2 = CE power off; no_s3 = CE sense off; gnd_to_blue = ground lead on the CE.
Writes lab_wiring_tests.json and ec_figures/ec06_lab_wiring_tests.png."""
import glob, json, os, warnings
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from galvani import BioLogic
warnings.filterwarnings("ignore")
here = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(here, "data", "lab_wiring_tests_20261109")


def load(paths):
    t, c, E, I, R = [], [], [], [], []
    for p in paths:
        d = BioLogic.MPRfile(p).data
        if len(d) == 0:
            continue
        t.append(d["time/s"]); c.append(d["control/V"]); E.append(d["Ewe/V"]); I.append(d["<I>/mA"] * 1e3); R.append(d["I Range"])
    if not t:
        return None
    return [np.concatenate(x).astype(float) for x in (t, c, E, I, R)]


tests = {"all": "normal wiring", "gnd_to_red": "ground lead also on the WE", "no_WE": "WE power and sense disconnected",
         "no_p1": "WE power (P1) off, sense on", "no_p2": "CE power (P2) off", "no_s3": "CE sense (S3) off", "gnd_to_blue": "ground lead on the CE"}
out = {}
for k, lab in tests.items():
    r = load(sorted(glob.glob(os.path.join(LAB, f"{k}_*.mpr"))))
    if r is None:
        continue
    t, c, E, I, R = r
    out[k] = dict(label=lab, seconds=float(t.max() - t.min()), ewe_track_rms_mV=float(1e3 * np.std(E - c)),
                  ewe_track_max_mV=float(1e3 * np.max(np.abs(E - c))), I_mean_uA=float(I.mean()), I_sd_uA=float(I.std()),
                  I_max_abs_uA=float(np.abs(I).max()), I_ranges=sorted(set(R.astype(int).tolist())),
                  completed=bool(t.max() - t.min() > 20))
d4 = load([os.path.join(here, "data", "echem", "Collin_08122026", f"CA_7nm_scan498_lower_0{i}_{x}_C01.mpr") for i, x in ((1, "CA"), (2, "CV"))])
t, c, E, I, R = d4
out["beamline day 4 (CV 498)"] = dict(label="4.1 nm, beamline", seconds=float(t.max() - t.min()), ewe_track_rms_mV=float(1e3 * np.std(E - c)),
                                      ewe_track_max_mV=float(1e3 * np.max(np.abs(E - c))), I_mean_uA=float(I.mean()), I_sd_uA=float(I.std()),
                                      I_max_abs_uA=float(np.abs(I).max()), I_ranges=sorted(set(R.astype(int).tolist())), completed=True)
json.dump(out, open(os.path.join(here, "lab_wiring_tests.json"), "w"), indent=1)
for k, v in out.items():
    print(f"{k:24s} {v['label']:34s} {v['seconds']:6.1f} s  Ewe-ctrl rms {v['ewe_track_rms_mV']:6.2f} mV (max {v['ewe_track_max_mV']:6.1f})  "
          f"I mean {v['I_mean_uA']:+9.4f} sd {v['I_sd_uA']:8.4f} max|I| {v['I_max_abs_uA']:9.3f} uA  ranges {v['I_ranges']}")

# figure: one row per configuration; potential | current | what it shows
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
ROWS = [
    ("all", "A  Bench: normal wiring", "#1f1f1e", "uA",
     [("Runs", "yes"), ("Ewe tracks", "yes"), ("Current", "recorded (to 138 µA)"), ("Cell polarized", "yes"), ("OCV", "stable")]),
    ("gnd_to_red", "B  Bench: ground lead also on the WE", "#c0392b", "nA",
     [("Runs", "yes"), ("Ewe tracks", "yes"), ("Current", "none (flat −1.5 nA)"), ("Cell polarized", "yes (meter check)"),
      ("OCV", "stable")]),
    ("no_WE", "C  Bench: WE lead disconnected", "#7d5ba6", "nA",
     [("Runs", "yes, if started in range"), ("Ewe tracks", "yes"), ("Current", "none (0 ± 0.5 nA)"), ("Cell polarized", "no"),
      ("OCV", "drifts, never settles")]),
    ("d4", "D  Beamline day 4: 4.1 nm, CV 498", "#eb6834", "nA",
     [("Runs", "yes"), ("Ewe tracks", "yes (beamline noise)"), ("Current", "none above noise (±0.2 µA)"),
      ("Cell polarized", "not measured"), ("OCV", "stable 0.37-0.40 V, beam-sensitive")]),
]
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "axes.titlesize": 9.5})
fig, axs = plt.subplots(4, 3, figsize=(10.6, 6.6), gridspec_kw=dict(width_ratios=[1, 1, 0.78]), layout="constrained")
for i, (k, title, col, unit, verdict) in enumerate(ROWS):
    t, c, E, I, R = d4 if k == "d4" else load(sorted(glob.glob(os.path.join(LAB, f"{k}_*.mpr"))))
    ax = axs[i, 0]
    ax.plot(t, c, color="#c9c7bf", lw=3.2, solid_capstyle="butt")
    ax.plot(t, E, color=col, lw=0.8)
    ax.set_ylabel("E (V vs ref)")
    ax.set_title(title, loc="left", fontweight="bold", color=col if k != "all" else INK)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax = axs[i, 1]
    y = I if unit == "uA" else I * 1e3
    ax.plot(t, y, color=col, lw=0.7)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel(f"I ({'µA' if unit == 'uA' else 'nA'})")
    if k == "gnd_to_red":
        ax.set_ylim(-10, 10)
    if k == "no_WE":
        ax.set_ylim(-12, 12)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax = axs[i, 2]
    ax.axis("off")
    for j, (a, b) in enumerate(verdict):
        yy = 0.9 - j * 0.2
        ax.text(0.0, yy, a, transform=ax.transAxes, fontsize=8.3, color=MUTED, va="center")  # label column
        bad = b == "no" or b.startswith("none") or b.startswith("drifts")
        ax.text(0.40, yy, b, transform=ax.transAxes, fontsize=8.3, va="center", color="#b03a2e" if bad else INK,
                fontweight="bold" if k == "d4" else "normal")
for j in range(2):
    axs[-1, j].set_xlabel("time (s)")
axs[0, 0].text(0.0, 1.30, "Potential: grey = programme, colour = Ewe measured", transform=axs[0, 0].transAxes, fontsize=8.5,
               color=MUTED)
axs[0, 1].text(0.0, 1.30, "Recorded current (note the units)", transform=axs[0, 1].transAxes, fontsize=8.5, color=MUTED)
axs[0, 2].text(0.0, 1.30, "What the run shows", transform=axs[0, 2].transAxes, fontsize=8.5, color=MUTED)
fig.text(0.005, -0.012, "Bench: SP-300, grounded channel, dummy cell (Randles); CA 30 s at 0 V, then CV 0 to 0.7 V at 50 mV/s "
         "(C started vs the drifting OCV).\nBeamline: a simpler potentiostat (lowest range about 1 µA, no filtering); CV at 5 mV/s. "
         "Red: what fails. D matches B, not C: its OCV was stable.", fontsize=7.6, color=MUTED, ha="left", va="top")
fig.savefig(os.path.join(here, "ec_figures", "ec06_lab_wiring_tests.png"), dpi=150, bbox_inches="tight")

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

# figure: normal, ground on WE, WE disconnected, and day 4
rows = [("all", "Lab, normal wiring", "#1f1f1e"), ("gnd_to_red", "Lab, ground lead also on the WE", "#d62728"),
        ("no_WE", "Lab, WE disconnected (power and sense)", "#9467bd"), ("d4", "Beamline day 4, 4.1 nm (CV 498)", "#eb6834")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(4, 2, figsize=(12, 10.5), gridspec_kw=dict(width_ratios=[1, 1.2]))
for i, (k, lab, col) in enumerate(rows):
    r = d4 if k == "d4" else load(sorted(glob.glob(os.path.join(LAB, f"{k}_*.mpr"))))
    t, c, E, I, R = r
    ax = axs[i, 0]
    ax.plot(t, c, color="#9a998f", lw=2.5, label="programme (control)")
    ax.plot(t, E, color=col, lw=0.8, label="Ewe measured")
    ax.set_ylabel("E (V)"); ax.set_title(f"{lab}: potential", loc="left", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    ax = axs[i, 1]
    ax.plot(t, I * (1 if k in ("all",) else 1e3), color=col, lw=0.7)
    ax.set_ylabel("I (µA)" if k == "all" else "I (nA)")
    s = out["beamline day 4 (CV 498)"] if k == "d4" else out[k]
    ax.set_title(f"current: mean {s['I_mean_uA']*1e3:+.2f} nA, sd {s['I_sd_uA']*1e3:.2f} nA"
                 + (f", max {s['I_max_abs_uA']:.0f} µA" if k == "all" else ""), loc="left", fontsize=9.5)
    if k == "gnd_to_red":
        ax.set_ylim(-5, 5)
for ax in axs[-1]:
    ax.set_xlabel("time (s)")
fig.suptitle("Wiring tests on an SP-300 with a dummy cell: only 'ground lead also on the WE' keeps the cell in the circuit while the recorded "
             "current vanishes.\nThe WE-disconnected case also tracks the potential with no current, but its OCV drifts (user) and nothing "
             "is polarized. Day 4 shows ~150 nA of noise where the lab shows ~1 nA.", fontsize=10, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(here, "ec_figures", "ec06_lab_wiring_tests.png"), dpi=130)

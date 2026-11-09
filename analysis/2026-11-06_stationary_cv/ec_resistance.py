"""What EC-Lab plots as 'resistance' for the CA / CV files: Ewe / I (the files hold no resistance measurement).
One CV per film on a log axis, |Ewe / I| against time, points with negative current marked. Writes ec_figures/ec05_ewe_over_i.png."""
import os, warnings
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from galvani import BioLogic
warnings.filterwarnings("ignore")
here = os.path.dirname(os.path.abspath(__file__))
F = [("3.3 nm, CV 222", "Collin_08102026/0p1M_H2SO4_5and25mVs_CA_timescan_3p3nm_scan222_02_CV_C01.mpr", "#2a78d6"),
     ("7 nm, CV 341", "Collin_08112026/0p1M_H2SO4_5mVs_CVcheckScan_7nm_scan341_02_CV_C01.mpr", "#1baf7a"),
     ("4.1 nm, CV 498", "Collin_08122026/CA_7nm_scan498_lower_02_CV_C01.mpr", "#eb6834")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(2, 3, figsize=(13, 6.2), sharex="col", gridspec_kw=dict(height_ratios=[1, 1.4]))
for j, (lab, rel, c) in enumerate(F):
    d = BioLogic.MPRfile(os.path.join(here, "data", "echem", rel)).data
    t, E, I = d["time/s"], d["Ewe/V"], d["<I>/mA"] * 1e-3
    R = E / I
    ax = axs[0, j]
    ax.plot(t, R / 1e6, color=c, lw=0.6)
    ax.set_title(f"{lab}: Ewe / I, linear axis (as EC-Lab shows it)", fontsize=9.5, loc="left")
    ax.set_ylabel("Ewe / I (MOhm)")
    ax = axs[1, j]
    pos, neg = I > 0, I < 0
    ax.semilogy(t[pos], np.abs(R[pos]), ".", ms=1.5, color=c, label="current > 0")
    ax.semilogy(t[neg], np.abs(R[neg]), ".", ms=1.5, color="#6b6a63", label="current < 0 (Ewe / I negative)")
    ax.set_ylim(1e3, 1e11); ax.set_xlabel("time (s)"); ax.set_ylabel("|Ewe / I| (Ohm), log axis")
    ax.set_title(f"median {np.median(np.abs(R))/1e6:.2f} MOhm, {100*np.mean(neg):.0f}% negative", fontsize=9.5, loc="left")
    ax.legend(loc="upper right", fontsize=8, markerscale=6, frameon=False)
fig.suptitle("EC-Lab 'resistance' for CV files is Ewe divided by I. On 4.1 nm the current is noise around zero, so Ewe / I is 10-100x "
             "higher, flips sign half the time, and spikes when the noise crosses zero.", fontsize=10, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(os.path.join(here, "ec_figures"), exist_ok=True)
fig.savefig(os.path.join(here, "ec_figures", "ec05_ewe_over_i.png"), dpi=140)

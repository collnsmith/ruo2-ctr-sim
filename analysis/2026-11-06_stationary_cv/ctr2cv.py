"""CTR -> CV, no further fitting: the surface state at each CTR potential (x_OH from the OH-rod fits, dz_Obr as assumed,
z_H2O as fitted) under three bridging-O assumptions (B: dz_Obr = 0; A: dz_Obr fitted per potential, shared by the films;
C': Rao et al. 2017 values with the potentials converted to RHE), the film structure from the same CTR fits. Prediction at
each CV reflection: I(state at E) / I(state at 0 V) - 1, against the measured cycle-averaged CV change (Ru+Ti) at the same E."""
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))
sys.modules["xraydb"] = None
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ctrfit.fit.global_fit import film_of
from ctrfit.project.project import Project

OH = os.path.join(here, "..", "2026-11-05_oh_fits")
p = Project.load(os.path.join(OH, "oh_rods_fit.ctrproj.json"))
m, P = p.model, p.model.params
lead = {}
for ds in p.datasets:
    lead.setdefault(film_of(ds).split("|")[0], ds.scope)
summ = json.load(open(os.path.join(OH, "oh_rods_fit_summary.json")))["results"]
c2 = json.load(open(os.path.join(here, "ohfit_c2_summary.json")))
VAR = {"C' Rao (RHE-corrected)": c2, "B dz_Obr = 0": summ["B fixed 0"], "A dz_Obr fitted": summ["A shared per potential"]}
MK = {"C' Rao (RHE-corrected)": "D", "B dz_Obr = 0": "o", "A dz_Obr fitted": "s"}
cv = json.load(open(os.path.join(here, "cv3.json")))
POTS = {"4.1 nm": [-0.238, -0.1, 0.0, 0.15, 0.35, 0.7], "7 nm": [-0.1, 0.0, 0.15, 0.35, 0.7]}
key = lambda f, e: f"{f} {e:g} V"

def state(var, film, e):
    r = VAR[var]; q = r["profiles"][key(film, e)]
    return q["best"], q["lo"], q["hi"], r["dz_Obr"][f"{e:g} V"], r["z_H2O"]

def I(film, H, K, L, x, dz, zw):
    v = dict(P.view(lead[film]), theta=1.0, x_OH=x, dz_Obr=dz, z_H2O=zw)
    return float(m.intensity(H, K, [L], v)[0])

def measured(s, E):
    c = cv[s]["sum"]["curves"]
    vals = [np.interp(E, c[b]["E"], c[b]["dI"]) for b in ("anodic", "cathodic")]
    return float(np.mean(vals))

panels = []                                   # (film, H, K, L, scans)
for s, r in cv.items():
    if r["H"] != 0 or r["film"] not in ("4.1", "7"): continue
    film = r["film"] + " nm"
    for pnl in panels:
        if pnl[:4] == (film, r["H"], r["K"], r["L"]): pnl[4].append(s); break
    else:
        panels.append((film, r["H"], r["K"], r["L"], [s]))
panels.sort(key=lambda t: (t[0] != "4.1 nm", t[3]))
rows, out = [], {}
for film, H, K, L, scans in panels:
    for var in VAR:
        x0, _, _, dz0, zw = state(var, film, 0.0)
        I0 = I(film, H, K, L, x0, dz0, zw)
        for e in POTS[film]:
            x, lo, hi, dz, _ = state(var, film, e)
            pr = I(film, H, K, L, x, dz, zw) / I0 - 1
            rng = sorted(I(film, H, K, L, xx, dz, zw) / I0 - 1 for xx in (lo, hi))
            out.setdefault(f"{film} (0 {K:+d} {L:g})", {}).setdefault(var, []).append(dict(E=e, pred=pr, lo=rng[0], hi=rng[1]))
    for e in (0.15, 0.35, 0.7):
        meas = np.mean([measured(s, min(e, 0.675)) for s in scans])
        preds = {var: next(q["pred"] for q in out[f"{film} (0 {K:+d} {L:g})"][var] if q["E"] == e) for var in VAR}
        rows.append((film, f"(0 {K:+d} {L:g})", e, meas, preds))
print(f"{'film':7s}{'HKL':13s}{'E':>6s}{'CV meas':>9s}  " + "  ".join(f"{v[:12]:>12s}" for v in VAR))
for film, hkl, e, meas, preds in rows:
    print(f"{film:7s}{hkl:13s}{e:6.2f}{100*meas:+8.1f}%  " + "  ".join(f"{100*preds[v]:+11.1f}%" for v in VAR))
json.dump(dict(predictions=out, table=[dict(film=f, hkl=h, E=e, measured=m_, predicted=pr) for f, h, e, m_, pr in rows]),
          open(os.path.join(here, "ctr2cv.json"), "w"), indent=1)

# figure: one panel per film and reflection; measured CV (both sweeps) in the film colour, CTR predictions as markers
COL = {"4.1 nm": "#eb6834", "7 nm": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True)
for ax in axs.flat: ax.set_visible(False)
slots = {"4.1 nm": 0, "7 nm": 0}
for film, H, K, L, scans in panels:
    ax = axs[0 if film == "4.1 nm" else 1, slots[film]]; slots[film] += 1; ax.set_visible(True)
    ax.axvspan(-0.3, 0.0, color="#f3f2ee", zorder=0)
    for i, s in enumerate(scans):
        c = cv[s]["sum"]["curves"]
        for b, ls in (("anodic", "-"), ("cathodic", "--")):
            ax.plot(c[b]["E"], 100 * np.array(c[b]["dI"]), ls, color=COL[film], lw=1.8, alpha=1 if i == 0 else 0.6,
                    label=(f"CV {film} (scan {s})" if b == "anodic" else None))
    for j, var in enumerate(VAR):
        q = out[f"{film} (0 {K:+d} {L:g})"][var]
        E = np.array([t["E"] for t in q]) + (j - 1) * 0.018
        y = 100 * np.array([t["pred"] for t in q])
        yerr = 100 * np.array([[t["pred"] - t["lo"] for t in q], [t["hi"] - t["pred"] for t in q]])
        ax.errorbar(E, y, yerr=np.abs(yerr), fmt=MK[var], ms=5.5, color=INK, mfc="white" if j else INK, mew=1.1,
                    elinewidth=0.6, capsize=0, zorder=3, label=f"CTR states, {var}")
    ax.axhline(0, color=GRID, lw=0.8, zorder=1)
    ax.set_title(f"{film} (0 {K:+d} {L:g})", loc="left", color=INK, fontsize=9.5)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_xlim(-0.3, 0.75)
for ax in axs[1]: ax.set_xlabel("E (V vs Ag/AgCl); shaded: outside the CV window")
for ax in axs[:, 0]: ax.set_ylabel("change from 0 V (%)")
h, l = [], []
for ax in axs.flat:
    if ax.get_visible():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll.startswith("CTR") and ll not in l: h.append(hh); l.append(ll)
axs[1, 3].set_visible(True); axs[1, 3].axis("off")
axs[1, 3].legend(h + [plt.Line2D([], [], color=MUTED, lw=1.8), plt.Line2D([], [], color=MUTED, lw=1.8, ls="--")],
                 [x.replace("CTR states, ", "CTR fit, ") for x in l] + ["CV, anodic sweep", "CV, cathodic sweep"],
                 loc="center", frameon=False, fontsize=8.5, title="markers: CTR-fitted state at each potential\n(bars: x_OH range at that potential)")
fig.suptitle("Measured CV change against the CTR-fitted surface states (nothing adjusted to the CVs)", fontsize=11, color=INK, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(os.path.join(here, "fig_ctr2cv.png"), dpi=150)

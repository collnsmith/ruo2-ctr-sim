"""3.3 nm (0 -1 L) rod at 0.65 V from the phi rocking scans 153-221 (ROI sums only, no SPEC file), a 3.3 nm film
structure fitted to it, and the 3.3 nm CV changes predicted from the CTR-fitted surface states (as ctr2cv.py does for 7 nm).

Scan numbering: the logbook macro runs L 0.35 to 3.75 in steps of 0.05, which is exactly 69 scans (153-221).
Signal: ROI1 (raw sum, 18 x 26 px) minus the mean of ROI2 / ROI3 scaled to the ROI1 area.
Why the scans are used as stationary points: the ROI is fixed for every image; below L ~2.45 the spot walks out of it
during the 0.2 deg phi scan (step-shaped profiles, the floor equals the background), above it the spot stays in the ROI
for the whole scan and the scan is narrower than the rod's rocking width (flat profiles). So the fit uses L >= 2.45,
the mean over the images of each scan, a free scale, a free slope for the missing geometry corrections (no angles,
monitor or filters without the SPEC file), and a free L offset. Scans 205-208 (Bragg region, 206/207 attenuated
mid-scan) are left out."""
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))
sys.modules["xraydb"] = None
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from ctrfit.fit.global_fit import film_of
from ctrfit.project.project import Project

d = pd.read_csv(os.path.join(here, "data", "ROIs_s153_221_sums.txt"), sep="\t", comment="#", dtype={"image": str})
AREA = 18 * 26 / (64 * 102)
rows = []
for s, g in d.groupby("scan"):
    g = g.iloc[:61]
    sig = g.ROI1.values - 0.5 * (g.ROI2.values + g.ROI3.values) * AREA
    sm = np.convolve(sig, np.ones(5) / 5, "same")
    plateau = sig[sm >= 0.5 * sm[2:-2].max()]           # images with the spot in the ROI (used only below L 2.45)
    rows.append(dict(scan=int(s), L_nom=round(0.35 + 0.05 * (s - 153), 2), mean=sig.mean(), plateau=plateau.mean(),
                     err=np.sqrt(np.sum(g.ROI1_err.values ** 2)) / len(g)))
rod = pd.DataFrame(rows)

# model: the CTR fit project (4.1 and 7 nm); start from the 4.1 nm film (fully strained, about 10 TL)
p = Project.load(os.path.join(here, "..", "2026-11-05_oh_fits", "oh_rods_fit.ctrproj.json"))
m, P = p.model, p.model.params
lead = {}
for ds in p.datasets:
    lead.setdefault(film_of(ds).split("|")[0], ds.scope)
base = dict(P.view(lead["4.1 nm"]))
summ = json.load(open(os.path.join(here, "..", "2026-11-05_oh_fits", "oh_rods_fit_summary.json")))["results"]
VAR = {"C' Rao": json.load(open(os.path.join(here, "ohfit_c3_summary.json"))),
       "B dz_Obr = 0": summ["B fixed 0"], "A dz_Obr fitted": summ["A shared per potential"]}

def state(var, e, film="7 nm"):
    r = VAR[var]
    return dict(theta=1.0, x_OH=r["profiles"][f"{film} {e:g} V"]["best"], dz_Obr=r["dz_Obr"][f"{e:g} V"], z_H2O=r["z_H2O"])

def I(v, L):
    return np.asarray(m.intensity(0, -1, np.atleast_1d(L), v), float)

use = rod[(rod.L_nom >= 2.45) & ~rod.scan.between(205, 208)].reset_index(drop=True)
Ld, Fd = use.L_nom.values, np.sqrt(np.clip(use["mean"].values, 1, None))
sF = 0.5 * np.sqrt(use.err.values ** 2 + (0.05 * use["mean"].values) ** 2) / Fd   # 5 % floor on I

FREE = ["thickness_mean_tl", "thickness_spread_tl", "eps_perp", "roughness_tl"]
BOUNDS = ([5, 0.3, 0.0, 0.0], [16, 4.0, 2.5, 2.5])

def model_F(x, var):
    v = dict(base, **state(var, 0.7), **dict(zip(FREE, x[:4])))
    dL, slope = x[4], x[5]
    F = np.sqrt(I(v, Ld + dL)) * np.exp(slope * (Ld - 3))
    sc = np.sum(F * Fd / sF ** 2) / np.sum(F ** 2 / sF ** 2)          # scale solved exactly
    return sc * F, v

def resid(x, var):
    return (model_F(x, var)[0] - Fd) / sF

fits = {}
for var in VAR:
    best = None
    for t0 in (7, 8, 9, 10, 11, 12, 13):
        for dL0 in (-0.04, 0.0):
            x0 = [t0, 1.0, base["eps_perp"], 0.6, dL0, 0.0]
            r = least_squares(resid, x0, args=(var,), bounds=(BOUNDS[0] + [-0.1, -3], BOUNDS[1] + [0.1, 3]), x_scale="jac")
            if best is None or r.cost < best.cost:
                best = r
    J = best.jac; cov = np.linalg.pinv(J.T @ J) * (2 * best.cost / max(len(Fd) - len(best.x) - 1, 1))
    fits[var] = dict(x=best.x.tolist(), err=np.sqrt(np.diag(cov)).tolist(), chi2=2 * best.cost / (len(Fd) - len(best.x) - 1))

# predictions at the 3.3 nm CV reflections: I(state at E) / I(state at 0 V) - 1, true L = nominal + fitted offset
cv = json.load(open(os.path.join(here, "cv3.json")))
def measured(s, E):
    c = cv[s]["sum"]["curves"]
    return float(np.mean([np.interp(E, c[b]["E"], c[b]["dI"]) for b in ("anodic", "cathodic")]))
CVS = {2.75: ["120"], 3.16: ["222"], 3.25: ["125"], 1.25: ["147"]}
pred = {}
for var, f in fits.items():
    x = f["x"]; v33 = dict(base, **dict(zip(FREE, x[:4])))
    v7 = dict(P.view(lead["7 nm"]))
    for Ln, scans in CVS.items():
        for tag, v, dL in (("3.3 nm fitted, offset", v33, x[4]), ("3.3 nm fitted, no offset", v33, 0.0),
                           ("7 nm structure", v7, 0.0)):
            I0 = I(dict(v, **state(var, 0.0)), Ln + dL)[0]
            for e in (0.15, 0.35, 0.7):
                pr = I(dict(v, **state(var, e)), Ln + dL)[0] / I0 - 1
                pred.setdefault(f"(0 -1 {Ln:g})", {}).setdefault(var, {}).setdefault(tag, {})[f"{e:g}"] = pr
meas = {f"(0 -1 {Ln:g})": {f"{e:g}": float(np.mean([measured(s, min(e, 0.675)) for s in scans])) for e in (0.15, 0.35, 0.7)}
        for Ln, scans in CVS.items()}
out = dict(rod=rod.to_dict("records"), fit_points=use.scan.tolist(), free=FREE + ["dL", "slope"], fits=fits,
           predictions=pred, measured=meas)
json.dump(out, open(os.path.join(here, "rod153.json"), "w"), indent=1)

for var, f in fits.items():
    print(f"{var:16s} chi2r {f['chi2']:.2f}  " + "  ".join(f"{n}={a:.3f}+-{e:.3f}" for n, a, e in zip(FREE + ['dL', 'slope'], f['x'], f['err'])))
print()
for hkl in pred:
    print(hkl, "measured 3.3 nm CV:", {e: f"{100*v:+.1f}%" for e, v in meas[hkl].items()})
    for var in pred[hkl]:
        for tag, q in pred[hkl][var].items():
            print(f"   {var:16s} {tag:26s}", {e: f"{100*v:+.1f}%" for e, v in q.items()})

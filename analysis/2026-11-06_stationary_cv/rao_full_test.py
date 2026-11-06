"""How well do the OH rods fit with the CUS species AND the bridging O both set from Rao et al. 2017 (no OH fraction fitted),
compared with the two fits that fix only the bridging O? Per film: chi2 with every rod scale and the shared water height refit.
Rao, on the logbook RHE scale (-0.238 V Ag/AgCl = 0 V RHE): water on the CUS sites and protonated bridging O up to 0.35 V
(0.59 V RHE, below their 0.6 V boundary); half water / half OH and deprotonated bridging O at 0.7 V (0.94 V RHE, near their 1.0 V)."""
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))
sys.modules["xraydb"] = None
import numpy as np
from ctrfit.fit import workflow
from ctrfit.fit.fit import Fit
from ctrfit.fit.global_fit import film_of
from ctrfit.project.project import Project
OH = os.path.join(here, "..", "2026-11-05_oh_fits")
fname = lambda d: film_of(d).split("|")[0]
pkey = lambda d: "OCV" if d.meta.get("potential_V") is None else f"{d.meta['potential_V']:g} V"

def main():
    p = Project.load(os.path.join(OH, "oh_rods_fit.ctrproj.json")); m, P = p.model, p.model.params
    D, _ = workflow.skip_bragg(p.datasets, 0.03, m)
    ohr = [d for d in D if d.rod_labels[0] in ("0 1", "0 -1") and pkey(d) != "OCV"]
    lead = {}
    for d in ohr:
        sc = P[f"{d.scope}.x_OH"].expr.split(".")[0] if P[f"{d.scope}.x_OH"].linked else d.scope
        lead[(fname(d), pkey(d))] = sc
    dzl = {}
    for (f, k), sc in lead.items():
        dzl.setdefault(k, P[f"{sc}.dz_Obr"].expr.split(".")[0] if P[f"{sc}.dz_Obr"].linked else sc)
    base = {n: P[n].value for n in P.names}
    summ = json.load(open(os.path.join(OH, "oh_rods_fit_summary.json")))["results"]
    c3 = json.load(open(os.path.join(here, "ohfit_c3_summary.json")))
    scen = {
        "bridging O at 0 A, OH fraction fitted": lambda f, k: (summ["B fixed 0"]["profiles"][f"{f} {k}"]["best"], 0.0),
        "Rao bridging O, OH fraction fitted": lambda f, k: (c3["profiles"][f"{f} {k}"]["best"], c3["dz_Obr"][k]),
        "Rao CUS species AND bridging O (nothing fitted)": lambda f, k: ((0.5, 0.0) if k == "0.7 V" else (0.0, 0.35)),
        "same, water height fixed at Rao's 2.68 A": lambda f, k: ((0.5, 0.0) if k == "0.7 V" else (0.0, 0.35)),
    }
    out = {}
    for name, rule in scen.items():
        for n, v in base.items(): P[n].value = v
        for (f, k), sc in lead.items():
            x, dz = rule(f, k)
            P[f"{sc}.x_OH"].value = x; P[f"{sc}.theta"].value = 1.0
        for k, sc in dzl.items():
            P[f"{sc}.dz_Obr"].value = rule("4.1 nm" if ("4.1 nm", k) in lead else "7 nm", k)[1]
        if name.startswith("same"):
            P["z_H2O"].value = 2.68
        P.fit_only(([] if name.startswith("same") else ["z_H2O"]) + [f"{d.scope}.scale" for d in ohr])
        g = Fit(m, ohr, cores=1); g.refine(); r = g.report()
        res = {}
        for f in ("4.1 nm", "7 nm"):
            sub = [d for d in ohr if fname(d) == f]
            Fm = g.model_F() if hasattr(g, "model_F") else None
            chi = 0.0; n = 0
            for d, F in zip(ohr, Fm):
                if fname(d) != f: continue
                chi += float(np.sum(((d.F - F) / d.sigma_eff) ** 2)); n += len(d)
            res[f] = (chi, n)
        out[name] = dict(z_H2O=r.values["z_H2O"], **{f: dict(chi2=c, n=n, chi2_per_point=c / n) for f, (c, n) in res.items()})
        print(f"{name:50s}: n {res['4.1 nm'][1]}/{res['7 nm'][1]}, chi2 {res['4.1 nm'][0]:.0f}/{res['7 nm'][0]:.0f} | z_H2O {r.values['z_H2O']:.2f} | 4.1 nm chi2/pt {res['4.1 nm'][0]/res['4.1 nm'][1]:6.2f} | 7 nm chi2/pt {res['7 nm'][0]/res['7 nm'][1]:6.2f}")
    json.dump(out, open(os.path.join(here, "rao_full_test.json"), "w"), indent=1)

if __name__ == "__main__":
    main()

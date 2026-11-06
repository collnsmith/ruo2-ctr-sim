"""OH fit variant C': Rao's dz_Obr with the potentials converted from Ag/AgCl to RHE (+0.26 V in 0.1 M H2SO4).
Starts from the saved OH-fit project (film fixed at stage-1 values), refits x_OH, z_H2O and the scales on the (0 +-1) rods."""
import json, sys, time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
sys.modules["xraydb"] = None
import numpy as np
from ctrfit.fit import workflow
from ctrfit.fit.fit import Fit
from ctrfit.fit.global_fit import film_of
from ctrfit.project.project import Project

T0 = time.time()
SHIFT = 0.238                      # logbook: -0.238 V Ag/AgCl = 0 V RHE
RAO = lambda e_rhe: 0.35 if e_rhe < 0.6 else (0.2 if e_rhe <= 0.9 else 0.0)

def log(*a): print(f"[{time.time() - T0:6.0f} s]", *a, flush=True)
fname = lambda ds: film_of(ds).split("|")[0]
def pkey(ds):
    v = ds.meta.get("potential_V"); return "OCV" if v is None else f"{v:g} V"

def main():
    src, out = sys.argv[1], sys.argv[2]
    p = Project.load(src)
    m, P = p.model, p.model.params
    D, _ = workflow.skip_bragg(p.datasets, 0.03, m)
    ohr = [d for d in D if d.rod_labels[0] in {"0 1", "0 -1"}]
    lead = {}
    for d in D:
        sc = P[f"{d.scope}.x_OH"].expr.split(".")[0] if P[f"{d.scope}.x_OH"].linked else d.scope
        lead.setdefault((fname(d), pkey(d)), sc)
    dz_lead = {}
    for (f, k), sc in lead.items():
        if not P[f"{sc}.dz_Obr"].linked:
            dz_lead.setdefault(k, sc)
    for k, sc in dz_lead.items():
        e = 0.378 + SHIFT if k == "OCV" else float(k.split()[0]) + SHIFT   # OCV 0.378 V (logbook)
        P[f"{sc}.dz_Obr"].value = RAO(e)
    log("dz_Obr C':", {k: P[f"{sc}.dz_Obr"].value for k, sc in dz_lead.items()})
    x_free = [f"{sc}.x_OH" for sc in lead.values()]
    free = x_free + ["z_H2O"] + [f"{d.scope}.scale" for d in ohr]
    base = {n: P[n].value for n in P.names}
    best = None
    for x0 in (0.1, 0.5, 0.9):
        for n, v in base.items(): P[n].value = v
        for n in x_free: P[n].value = x0
        P.fit_only(free)
        g = Fit(m, ohr, cores=4); g.refine(starts=1); rr = g.report()
        log(f"start {x0}: red chi2 {rr.fom['red_chi2']:.2f}")
        if best is None or rr.fom["chi2"] < best[0]:
            best = (rr.fom["chi2"], {n: P[n].value for n in P.names}, rr)
    for n, v in best[1].items(): P[n].value = v
    rr = best[2]
    at = {n: P[n].value for n in P.names}
    prof = {}
    for (f_, k), sc in sorted(lead.items()):
        sub = [d for d in ohr if (fname(d), pkey(d)) == (f_, k)]
        if not sub: continue
        curve = []
        for x in np.linspace(0, 1, 21):
            for n, v in at.items(): P[n].value = v
            P[f"{sc}.x_OH"].value = x
            P.fit_only([f"{d.scope}.scale" for d in sub])
            h = Fit(m, sub, cores=1); h.refine()
            curve.append((float(x), float(h.report().fom["chi2"])))
        xs, cs = np.array(curve).T
        ok = xs[cs <= cs.min() + 4.0 * max(rr.fom["red_chi2"], 1.0)]
        prof[f"{f_} {k}"] = dict(best=float(xs[np.argmin(cs)]), lo=float(ok.min()), hi=float(ok.max()), curve=curve)
        log(f"   {f_} {k:8s} x_OH best {prof[f'{f_} {k}']['best']:.2f}, allowed {ok.min():.2f}-{ok.max():.2f}")
    json.dump(dict(red_chi2=rr.fom["red_chi2"], z_H2O=rr.values["z_H2O"],
                   dz_Obr={k: P[f"{sc}.dz_Obr"].value for k, sc in dz_lead.items()}, profiles=prof),
              open(out, "w"), indent=1, default=float)
    log(f"done: red chi2 {rr.fom['red_chi2']:.2f}, z_H2O {rr.values['z_H2O']:.2f}")

if __name__ == "__main__":
    main()

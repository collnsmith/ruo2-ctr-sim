import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))   # the repository; sys.modules["xraydb"] = None
import numpy as np
from ctrfit.project.project import Project
from ctrfit.fit.fit import Fit

def tilt(fit, D):
    ps = []
    for ds, F in zip(D, fit.model_F()):
        LB = 3.0 if ds.rod_labels[0] == "0 1" else 2.0
        k = np.abs(ds.L - LB) > 0.06
        ps.append(np.polyfit(np.log(ds.L[k]), np.log(ds.F[k] / F[k]), 1)[0])
    return np.mean(ps)

def main():
    p = Project.load(sys.argv[1])
    m, P, D = p.model, p.model.params, p.datasets
    surf = ["theta", "x_OH", "z_OH", "z_H2O", "dz_Obr", "dz_Ru6f", "dz_Oip", "dz_Rucus_OH", "t_liquid", "b_surf_Ru", "b_surf_O"]
    for n in surf:
        cps = [c for c in P.names if c.rpartition(".")[2] == n and "." in c]
        print(f"  {n}: shared {P[n].value:.3g} [{P[n].min:g},{P[n].max:g}]" + (f"; copies {[round(P[c].value, 3) for c in cps]}" if cps else ""))
    # physical bounds and starts; one surface relaxation set shared by all potentials, the coverage per dataset
    for n, (v, lo, hi) in dict(z_OH=(2.0, 1.8, 2.3), dz_Ru6f=(0.0, -0.2, 0.2), dz_Oip=(0.0, -0.2, 0.2),
                               dz_Rucus_OH=(0.0, -0.2, 0.2)).items():
        P[n].value, P[n].min, P[n].max = v, lo, hi
    for c in [c for c in P.names if c.rpartition(".")[2] == "dz_Obr"] + (["dz_Obr"]):
        P[c].min, P[c].max = -0.2, 0.4
        P[c].value = min(P[c].value, 0.3)
    theta = [c for c in P.names if c.rpartition(".")[2] == "theta" and "." in c] or ["theta"]
    dzobr = [c for c in P.names if c.rpartition(".")[2] == "dz_Obr" and "." in c] or ["dz_Obr"]
    scales = [f"{d.scope}.scale" for d in D]
    P.fit_only(["z_OH", "dz_Ru6f", "dz_Oip", "dz_Rucus_OH"] + theta + dzobr + scales)
    f0 = Fit(m, D, cores=4)
    print(f"before: red chi2 {f0.report().fom['red_chi2']:.1f}, tilt L^{tilt(f0, D):+.2f}")
    f = Fit(m, D, cores=4)
    f.run("de", maxiter=30, popsize=10, seed=1)
    f.refine(starts=2, hops=0)
    r = f.report()
    print(f"surface refit (film fixed): red chi2 {r.fom['red_chi2']:.1f}, tilt L^{tilt(f, D):+.2f}")
    print("  ", {n: round(r.values[n], 3) for n in ["z_OH", "dz_Ru6f", "dz_Oip", "dz_Rucus_OH"]})
    print("   theta", [round(r.values[c], 2) for c in theta], "dz_Obr", [round(r.values[c], 2) for c in dzobr])
    for ds, F in zip(D, f.model_F()):
        print(f"   {ds.scope} {ds.rod_labels[0]} {ds.meta.get('potential_V')}: chi2/pt {np.mean(((ds.F - F) / ds.sigma_eff) ** 2):.1f}")
    for w in r.warnings[:5]:
        print("   warning:", w)
    p.results.append(r.to_dict())
    p.save(sys.argv[2])

if __name__ == "__main__":
    main()

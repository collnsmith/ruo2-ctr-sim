import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))   # the repository; sys.modules["xraydb"] = None
import numpy as np
from ctrfit.project.project import Project
from ctrfit.fit import workflow
from ctrfit.fit.fit import Fit

FILM = ["thickness_mean_tl", "eps_perp", "roughness_tl", "thickness_spread_tl"]

def per_rod(fit, datasets):
    out = []
    for ds, F in zip(datasets, fit.model_F()):
        out.append(f"{ds.scope} {ds.rod_labels[0]} {ds.meta.get('potential_V')}: "
                   f"{np.mean(((ds.F - F) / ds.sigma_eff) ** 2):.1f}")
    return out

def main():
    p = Project.load(sys.argv[1])
    m, P, D = p.model, p.model.params, p.datasets
    workflow.prepare(m, D)                                   # ds5 joins the film's copies
    surf = sorted({n.rpartition(".")[2] for n in P.names} & {"theta", "x_OH", "x_O", "z_OH", "z_H2O", "dz_Obr",
                                                             "x_SO4", "t_liquid", "polar_A", "ns_du"})
    print("surface (fixed):", {n: round(P.view("ds2")[n], 3) for n in surf})
    set_ = dict(thickness_mean_tl=(21.8, 19.0, 24.0), eps_perp=(1.3, 0.8, 2.2), roughness_tl=(0.65, 0.2, 1.5),
                thickness_spread_tl=(0.8, 0.3, 3.0), interface_A=(3.24, 3.0, 3.5), relax_001=(0.0, 0.0, 1.0),
                coherent_tl=(30.0, 0.0, 300.0))
    for n, (v, lo, hi) in set_.items():
        q = P[f"ds2.{n}"]
        q.min, q.max, q.value = lo, hi, v
    scales = [f"{d.scope}.scale" for d in D]
    P.fit_only([f"ds2.{n}" for n in FILM] + scales)
    f = Fit(m, D, cores=4)
    f.run("de", maxiter=40, popsize=12, seed=1)
    f.profile("ds2.thickness_mean_tl", workflow.thickness_window(m, "ds2.thickness_mean_tl"))
    f.refine(starts=3, hops=1)
    r1 = f.report()
    print(f"\nA. uniform film, interface fixed 3.24: red chi2 {r1.fom['red_chi2']:.1f} (was 45.7)")
    print("  ", {n: f"{r1.values['ds2.' + n]:.3f} ± {r1.errors['ds2.' + n]:.3f}" for n in FILM})
    print("   per dataset:", "; ".join(per_rod(f, D)))
    print("   scales:", {s: round(r1.values[s], 2) for s in scales})
    for w in r1.warnings[:6]:
        print("   warning:", w)
    # B: interface free within 3.0-3.5
    P.fit_only([f"ds2.{n}" for n in FILM + ["interface_A"]] + scales)
    f2 = Fit(m, D, cores=4)
    f2.refine(starts=1, hops=1)
    r2 = f2.report()
    print(f"\nB. + interface_A free (3.0-3.5): red chi2 {r2.fom['red_chi2']:.1f}")
    print("  ", {n: f"{r2.values['ds2.' + n]:.3f} ± {r2.errors['ds2.' + n]:.3f}" for n in FILM + ["interface_A"]})
    print("   per dataset:", "; ".join(per_rod(f2, D)))
    for w in r2.warnings[:6]:
        print("   warning:", w)
    best = (r2, f2) if r2.fom["red_chi2"] < r1.fom["red_chi2"] - 0.5 else (r1, f)
    if best[0] is r1:                                        # keep A's values in the saved project
        for n, v in r1.values.items():
            P[n].value = v
        P["ds2.interface_A"].value, P["ds2.interface_A"].fit = 3.24, False
    p.results.append(best[0].to_dict())
    p.save(sys.argv[2])
    np.save(sys.argv[2] + ".curves.npy", np.array([np.concatenate([d.L, d.F, d.sigma_eff, F]) for d, F in
                                                   zip(D, best[1].model_F())], dtype=object), allow_pickle=True)
    print("\nsaved", sys.argv[2], "with", "B" if best[0] is r2 else "A")

if __name__ == "__main__":
    main()

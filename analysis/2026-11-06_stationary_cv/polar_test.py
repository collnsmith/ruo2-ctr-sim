"""Does a polar distortion through the whole film (Jeong et al.: polar mm2 phase below ~4 nm) fit the CTRs better than
one confined to the interface? Per film, all rods and potentials (near-Bragg points skipped): polar_A free with
polar_tl = 1.5 TL (interface) or 100 TL (film-wide), against polar_A = 0; film parameters, each surface group's x_OH
and dz_Obr, and every scale refit in each case."""
import sys, time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
sys.modules["xraydb"] = None
import numpy as np
from ctrfit.fit import workflow
from ctrfit.fit.fit import Fit
from ctrfit.fit.global_fit import film_of
from ctrfit.project.project import Project
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:5.0f} s]", *a, flush=True)
def main():
    p = Project.load(sys.argv[1]); m, P = p.model, p.model.params
    D, _ = workflow.skip_bragg(p.datasets, 0.03, m)
    P["polar_tl"].max = 200.0
    base = {n: P[n].value for n in P.names}
    for film in ("4.1 nm", "7 nm"):
        sub = [d for d in D if film_of(d).split("|")[0] == film]
        lead = sub[0].scope
        fl = P[f"{lead}.polar_A"].expr.split(".")[0] if P[f"{lead}.polar_A"].linked else lead
        groups = {}
        for d in sub:
            sc = P[f"{d.scope}.x_OH"].expr.split(".")[0] if P[f"{d.scope}.x_OH"].linked else d.scope
            groups[sc] = 1
        free_common = [f"{fl}.{n}" for n in ("thickness_mean_tl", "eps_perp", "roughness_tl", "thickness_spread_tl", "interface_A")] \
            + [f"{g}.{n}" for g in groups for n in ("x_OH", "dz_Obr")] + [f"{d.scope}.scale" for d in sub]
        log(f"{film}: {len(sub)} rod files, {sum(len(d) for d in sub)} points, {len(groups)} surface groups, polar_A at {fl}")
        for label, tl, starts in (("no polar shift", 1.5, (0.0,)), ("interface, 1.5 TL", 1.5, (0.085, -0.085)),
                                  ("film-wide, 100 TL", 100.0, (0.05, -0.05))):
            best = None
            for a0 in starts:
                for n, v in base.items(): P[n].value = v
                P["polar_tl"].value = tl
                P[f"{fl}.polar_A"].value = a0
                free = free_common + ([f"{fl}.polar_A"] if label != "no polar shift" else [])
                P.fit_only(free)
                f = Fit(m, sub, cores=4); f.refine(starts=1); r = f.report()
                if best is None or r.fom["chi2"] < best[0]:
                    best = (r.fom["chi2"], r)
            r = best[1]
            pa = r.values.get(f"{fl}.polar_A", 0.0); pe = r.errors.get(f"{fl}.polar_A", float("nan"))
            log(f"   {label:18s}: chi2 {r.fom['chi2']:9.1f}, red {r.fom['red_chi2']:6.2f}, polar_A {pa:+.3f} ± {pe:.3f} Å, "
                f"thickness {r.values[f'{fl}.thickness_mean_tl']:.1f} TL, eps_perp {r.values[f'{fl}.eps_perp']:.2f}")
if __name__ == "__main__":
    main()

"""OH from the rods that carry it: film pinned on the well-behaved rods, then x_OH on the (0 +-1) rods only,
under three assumptions for the bridging O height, with profiles."""
import copy
import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))   # the repository
sys.modules["xraydb"] = None
import numpy as np

from ctrfit.fit import workflow
from ctrfit.fit.fit import Fit
from ctrfit.fit.global_fit import film_of
from ctrfit.project.project import Project

OUT = str(__import__("pathlib").Path(__file__).resolve().parent / "ohfit")
T0 = time.time()
GOOD = {"0 1", "0 -1", "-1 0", "-2 0"}
OH_RODS = {"0 1", "0 -1"}
RAO = lambda pot: 0.35 if pot is None or pot < 0.6 else (0.2 if pot <= 0.9 else 0.0)   # dz_Obr by potential


def log(*a):
    print(f"[{time.time() - T0:6.0f} s]", *a, flush=True)


def fname(ds):
    return film_of(ds).split("|")[0]


def pkey(ds):
    v = ds.meta.get("potential_V")
    return "OCV" if v is None else f"{v:g} V"


def main():
    p = Project.load(sys.argv[1])                      # the autofit result (fresh model, both films)
    m, P, D_all = p.model, p.model.params, p.datasets
    D_all, notes = workflow.skip_bragg(D_all, 0.03, m)
    good = [d for d in D_all if d.rod_labels[0] in GOOD]
    ohr = [d for d in D_all if d.rod_labels[0] in OH_RODS]
    log(f"{len(good)} well-behaved scans, {len(ohr)} OH scans")
    lead = {}                                          # surface group lead per (film, potential), as in autofit
    for d in D_all:
        sc = P[f"{d.scope}.x_OH"].expr.split(".")[0] if P[f"{d.scope}.x_OH"].linked else d.scope
        lead.setdefault((fname(d), pkey(d)), sc)
    films = {}
    for d in D_all:
        films.setdefault(fname(d), P[f"{d.scope}.eps_perp"].expr.split(".")[0] if P[f"{d.scope}.eps_perp"].linked else d.scope)
    log("film leads", films)
    # theta = 1 everywhere; dz_Obr one per potential shared by the films (variant A): link the 7 nm leads to 4.1 nm
    pots = sorted({k[1] for k in lead})
    for (f, k), sc in lead.items():
        P[f"{sc}.theta"].value, P[f"{sc}.theta"].fit = 1.0, False
        P[f"{sc}.x_OH"].min, P[f"{sc}.x_OH"].max = 0.0, 1.0
    dz_lead = {}
    for k in pots:
        scs = [lead[(f, k)] for f in ("4.1 nm", "7 nm") if (f, k) in lead]
        dz_lead[k] = scs[0]
        for sc in scs[1:]:
            P.link(f"{sc}.dz_Obr", f"{scs[0]}.dz_Obr")
    P["b_surf_O"].value = 1.0
    fl = films
    P[f"{fl['4.1 nm']}.thickness_mean_tl"].min, P[f"{fl['4.1 nm']}.thickness_mean_tl"].max = 10.0, 15.0
    P[f"{fl['4.1 nm']}.thickness_mean_tl"].value = 12.7
    film_free = [f"{sc}.{n}" for sc in fl.values() for n in
                 ("thickness_mean_tl", "eps_perp", "roughness_tl", "thickness_spread_tl", "interface_A")]
    x_free = [f"{sc}.x_OH" for sc in lead.values()]
    dz_free = [f"{sc}.dz_Obr" for sc in dz_lead.values()]
    shared = ["z_H2O", "dz_Ru6f"]

    def scales(ds):
        return [f"{d.scope}.scale" for d in ds]

    # ---------------- 1. film on the well-behaved rods (surface free), three starting OH shares
    best = None
    for x0 in (0.1, 0.5, 0.9):
        for n in x_free:
            P[n].value = x0
        P.fit_only(film_free + x_free + dz_free + shared + scales(good))
        f = Fit(m, good, cores=4)
        f.refine(starts=1, hops=1)
        r = f.report()
        log(f"1 film + surface on {len(good)} scans, x_OH from {x0}: red chi2 {r.fom['red_chi2']:.2f}")
        if best is None or r.fom["chi2"] < best[0]:
            best = (r.fom["chi2"], {n: P[n].value for n in P.names}, r)
    for n, v in best[1].items():
        P[n].value = v
    r1 = best[2]
    for f_, sc in fl.items():
        log("  film", f_, {n: (round(r1.values[f"{sc}.{n}"], 3), round(r1.errors.get(f"{sc}.{n}", np.nan), 3)) for n in
                           ("thickness_mean_tl", "eps_perp", "roughness_tl", "thickness_spread_tl", "interface_A")})
    film_vals = {n: P[n].value for n in film_free}
    base = {n: P[n].value for n in P.names}
    # ---------------- 2. OH on the (0 +-1) rods only, film fixed, three dz_Obr assumptions
    results = {}
    for variant in ("A shared per potential", "B fixed 0", "C Rao by potential"):
        for n, v in base.items():
            P[n].value = v
        free = x_free + ["z_H2O"] + scales(ohr)
        if variant.startswith("A"):
            free += dz_free
        else:
            for k, sc in dz_lead.items():
                pot = None if k == "OCV" else float(k.split()[0])
                P[f"{sc}.dz_Obr"].value = 0.0 if variant.startswith("B") else RAO(pot)
        bestv = None
        for x0 in (0.1, 0.5, 0.9):
            for n in x_free:
                P[n].value = x0
            P.fit_only(free)
            g = Fit(m, ohr, cores=4)
            g.refine(starts=1)
            rr = g.report()
            if bestv is None or rr.fom["chi2"] < bestv[0]:
                bestv = (rr.fom["chi2"], {n: P[n].value for n in P.names}, rr)
        for n, v in bestv[1].items():
            P[n].value = v
        rr = bestv[2]
        log(f"2{variant[0]} OH ({variant}): red chi2 {rr.fom['red_chi2']:.2f}, z_H2O {rr.values['z_H2O']:.2f}")
        # profiles: x_OH of one group on a grid; refit its scale (and in A the shared dz_Obr with the other film)
        prof = {}
        at = {n: P[n].value for n in P.names}
        for (f_, k), sc in sorted(lead.items()):
            sub = [d for d in ohr if (fname(d), pkey(d)) == (f_, k)]
            if not sub:
                continue
            own = scales(sub)
            if variant.startswith("A"):
                other = [d for d in ohr if pkey(d) == k and fname(d) != f_]
                sub = sub + other
                own = scales(sub) + [f"{dz_lead[k]}.dz_Obr"] + [f"{lead[(fname(o), k)]}.x_OH" for o in other]
            curve = []
            for x in np.linspace(0, 1, 21):
                for n, v in at.items():
                    P[n].value = v
                P[f"{sc}.x_OH"].value = x
                P.fit_only(own)
                h = Fit(m, sub, cores=1)
                h.refine()
                curve.append((float(x), float(h.report().fom["chi2"])))
            xs, cs = np.array(curve).T
            thr = 4.0 * max(rr.fom["red_chi2"], 1.0)          # about 2 sigma, scaled by the misfit
            ok = xs[cs <= cs.min() + thr]
            prof[f"{f_} {k}"] = dict(best=float(xs[np.argmin(cs)]), lo=float(ok.min()), hi=float(ok.max()), curve=curve)
        for n, v in at.items():
            P[n].value = v
        results[variant] = dict(red_chi2=rr.fom["red_chi2"], z_H2O=rr.values["z_H2O"],
                                x_OH={f"{f_} {k}": (rr.values[f"{sc}.x_OH"], rr.errors.get(f"{sc}.x_OH")) for (f_, k), sc in lead.items()},
                                dz_Obr={k: P[f"{sc}.dz_Obr"].value for k, sc in dz_lead.items()},
                                profiles=prof)
        for key in sorted(prof, key=lambda s: (s.split(" ")[1:], s)):
            q = prof[key]
            log(f"   {key:16s} x_OH best {q['best']:.2f}, allowed {q['lo']:.2f}-{q['hi']:.2f}")
    json.dump(dict(film=film_vals, results=results), open(OUT + "_summary.json", "w"), indent=1, default=float)
    p.results.append(r1.to_dict())
    p.save(OUT + "_project.ctrproj.json")
    log("done")


if __name__ == "__main__":
    main()

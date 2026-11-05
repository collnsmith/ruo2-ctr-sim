"""Both films, start to finish, from the uploaded project's files and cuts (a fresh model).

Film parameters: one value per film. Surface (theta, x_OH, dz_Obr): one per film and potential (the rods of
one film at one potential share it by links). Every rod file keeps its own scale. Stages: scales, film (DE,
thickness scans, refine with hops), surface (two starting x_OH), everything, then conditional x_OH profiles.
"""
import copy
import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))   # the repository
sys.modules["xraydb"] = None
import numpy as np

from ctrfit.core.settings import DEFAULTS
from ctrfit.fit import workflow
from ctrfit.fit.fit import Fit
from ctrfit.fit.global_fit import film_of, make_per_dataset, make_per_film
from ctrfit.fit.scans import regroup
from ctrfit.model.model import Model
from ctrfit.project.project import Project

OUT = str(__import__("pathlib").Path(__file__).resolve().parent / "autofit")
FILM = ["thickness_mean_tl", "eps_perp", "roughness_tl", "thickness_spread_tl", "interface_A", "relax_001",
        "coherent_tl", "polar_A", "polar_sub_A"]
SURF = ["theta", "x_OH", "dz_Obr"]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f} s]", *a, flush=True)


def cond(ds):
    v = ds.meta.get("potential_V")
    return "OCV" if v is None else f"{v:g} V"


def chi_table(fit, D):
    rows = []
    for ds, F in zip(D, fit.model_F()):
        rows.append((ds.scope, film_of(ds), cond(ds), ds.rod_labels[0], float(np.mean(((ds.F - F) / ds.sigma_eff) ** 2))))
    return rows


def main():
    src = Project.load(sys.argv[1])
    sel = src.scans
    for r in sel.rows():                        # every rod scan of both films, not the specular (0 0)
        sel.set_use([r.key], r.rods != ["0 0"])
    settings = {k: v for k, v in src.model.settings.items() if k in DEFAULTS and DEFAULTS[k] != v}
    log("settings kept:", settings)
    model = Model.from_settings(dict(DEFAULTS, **settings))
    D, notes = regroup(model, sel, [])
    log(f"{len(D)} datasets, {sum(len(d) for d in D)} points; notes {notes}")
    P = model.params
    # ---------------- fixed physics and shared starts
    P.update(dict(z_OH=2.07, z_H2O=2.5, t_liquid=5.0, t_window=7.6, x_O=0.0, x_SO4=0.0, b_surf_O=1.0,
                  polar_A=0.085, polar_sub_A=0.063, polar_tl=1.5, polar_sub_tl=5.0, theta=0.9, x_OH=0.5,
                  dz_Obr=0.1, dz_Ru6f=0.0))
    for n, lo, hi in (("z_H2O", 2.15, 2.9), ("dz_Ru6f", -0.15, 0.15), ("b_surf_O", 0.3, 5.0), ("theta", 0.5, 1.0),
                      ("dz_Obr", -0.2, 0.4), ("interface_A", 3.0, 3.5), ("eps_perp", -0.5, 2.3),
                      ("roughness_tl", 0.2, 2.0), ("thickness_spread_tl", 0.3, 3.0)):
        P[n].min, P[n].max = lo, hi
    workflow.prepare(model, D)                 # a scale per dataset
    make_per_dataset(model, D, SURF)
    make_per_film(model, D, FILM)
    # surface: one per film and potential (links to the group's first dataset)
    groups = {}
    for ds in D:
        groups.setdefault((film_of(ds), cond(ds)), []).append(ds)
    for (film, c), members in groups.items():
        lead = members[0].scope
        for ds in members[1:]:
            for n in SURF:
                P.link(f"{ds.scope}.{n}", f"{lead}.{n}")
    leads = {g: m[0].scope for g, m in groups.items()}
    films = {}
    for ds in D:
        films.setdefault(film_of(ds), ds.scope)
    log("film leads:", films, "| surface groups:", {f"{g[0]} {g[1]}": f"{leads[g]} ({len(groups[g])} rods)" for g in groups})
    fl = films
    thick = {"4.1 nm": (12.7, 11.0, 14.5), "7 nm": (21.7, 19.5, 24.0)}
    for film, sc in fl.items():
        film = film.split("|")[0]
        t, lo, hi = thick[film]
        P[f"{sc}.thickness_mean_tl"].value, P[f"{sc}.thickness_mean_tl"].min, P[f"{sc}.thickness_mean_tl"].max = t, lo, hi
        P[f"{sc}.interface_A"].value = 3.24
        P[f"{sc}.roughness_tl"].value, P[f"{sc}.thickness_spread_tl"].value = 0.6, 0.8
        if film == "4.1 nm":                    # fully strained (Jeong et al.)
            P[f"{sc}.eps_perp"].value = 1.65
            P[f"{sc}.relax_001"].value, P[f"{sc}.coherent_tl"].value = 0.0, 40.0
        else:                                   # (2 0) shows no film: 1 TL in registry, the rest slightly relaxed
            P[f"{sc}.eps_perp"].value = 0.8
            P[f"{sc}.relax_001"].value, P[f"{sc}.coherent_tl"].value = 0.05, 1.0
    scales = [f"{d.scope}.scale" for d in D]
    film_free = [f"{sc}.{n}" for sc in fl.values() for n in
                 ("thickness_mean_tl", "eps_perp", "roughness_tl", "thickness_spread_tl", "interface_A")]
    surf_free = [f"{sc}.{n}" for sc in leads.values() for n in SURF] + ["z_H2O", "dz_Ru6f", "b_surf_O"]

    def run(free, label, de=False, starts=1, hops=0, profile=False):
        P.fit_only(free)
        f = Fit(model, D, cores=4)
        if de:
            f.run("de", maxiter=40, popsize=12, seed=1)
        if profile:
            for sc in fl.values():
                n = f"{sc}.thickness_mean_tl"
                f.profile(n, workflow.thickness_window(model, n))
        f.refine(starts=starts, hops=hops)
        r = f.report()
        log(f"{label}: red chi2 {r.fom['red_chi2']:.2f} ({len(free)} free)")
        return f, r

    run(scales, "1 scales")
    f, r = run(film_free + scales, "2 film", de=True, starts=3, hops=1, profile=True)
    for sc in fl.values():
        log("   film", sc, {n: round(r.values[f"{sc}.{n}"], 3) for n in
                         ("thickness_mean_tl", "eps_perp", "roughness_tl", "thickness_spread_tl", "interface_A")})
    best = None
    for x0 in (0.2, 0.8):                      # surface from two starting OH shares
        snap = copy.deepcopy(P.to_dict()) if hasattr(P, "to_dict") else None
        for sc in leads.values():
            P[f"{sc}.x_OH"].value = x0
        f, r = run(surf_free + scales, f"3 surface (x_OH from {x0})", starts=1)
        if best is None or r.fom["chi2"] < best[0]:
            best = (r.fom["chi2"], {n: P[n].value for n in P.names})
    for n, v in best[1].items():
        P[n].value = v
    f, r = run(film_free + surf_free + scales, "4 everything", starts=1, hops=1)
    proj = Project(model, D, [r.to_dict()], name="both films, autofit", scans=sel, options=src.options)
    proj.save(OUT + "_project.ctrproj.json")
    log("saved project")
    # ---------------- report
    names, cov = r.names, np.asarray(r.cov)
    idx = {n: i for i, n in enumerate(names)}
    table = []
    for (film, c), sc in sorted(leads.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        th, x = r.values[f"{sc}.theta"], r.values[f"{sc}.x_OH"]
        i, j = idx.get(f"{sc}.theta"), idx.get(f"{sc}.x_OH")
        var = (x ** 2 * cov[i, i] + th ** 2 * cov[j, j] + 2 * th * x * cov[i, j]) if i is not None and j is not None else np.nan
        zmean = x * r.values["z_OH"] + (1 - x) * r.values["z_H2O"]
        table.append(dict(film=film.split("|")[0], cond=c, rods=len(groups[(film, c)]), theta=th, theta_err=r.errors.get(f"{sc}.theta"),
                          x_OH=x, x_OH_err=r.errors.get(f"{sc}.x_OH"), OH=th * x, OH_err=float(np.sqrt(max(var, 0))),
                          dz_Obr=r.values[f"{sc}.dz_Obr"], z_mean=zmean))
    log("shared:", {n: round(r.values[n], 3) for n in ("z_H2O", "dz_Ru6f", "b_surf_O")})
    for sc in fl.values():
        log("film", sc, {n: (round(r.values[f"{sc}.{n}"], 3), round(r.errors.get(f"{sc}.{n}", np.nan), 3)) for n in
                         ("thickness_mean_tl", "eps_perp", "roughness_tl", "thickness_spread_tl", "interface_A")})
    for t in table:
        log(f"  {t['film']:6s} {t['cond']:8s} ({t['rods']} rods): theta {t['theta']:.2f}±{t['theta_err']:.2f}  "
            f"x_OH {t['x_OH']:.2f}±{t['x_OH_err']:.2f}  OH/CUS {t['OH']:.2f}±{t['OH_err']:.2f}  dz_Obr {t['dz_Obr']:+.2f}  "
            f"<z_O> {t['z_mean']:.2f}")
    for w in r.warnings:
        log("  warning:", w)
    rows = chi_table(f, D)
    for row in rows:
        log("  chi2/pt", *row[:4], f"{row[4]:.1f}")
    # ---------------- conditional x_OH profiles: each group's x_OH on a grid, its own theta, dz_Obr, scales refit
    prof = {}
    base = {n: P[n].value for n in P.names}
    for (film, c), sc in leads.items():
        own = [f"{sc}.theta", f"{sc}.dz_Obr"] + [f"{d.scope}.scale" for d in groups[(film, c)]]
        sub = groups[(film, c)]
        curve = []
        for x in np.linspace(0, 1, 11):
            for n, v in base.items():
                P[n].value = v
            P[f"{sc}.x_OH"].value = x
            P.fit_only(own)
            g = Fit(model, sub, cores=1)
            g.refine()
            curve.append((float(x), float(g.report().fom["chi2"])))
        film = film.split("|")[0]
        prof[f"{film} {c}"] = curve
        xs, cs = np.array(curve).T
        ok = xs[cs <= cs.min() + 4.0 * max(r.fom["red_chi2"], 1.0)]    # Δχ² ≤ 4·reduced χ² (scaled 2σ)
        log(f"  profile {film} {c}: best x_OH {xs[np.argmin(cs)]:.1f}, allowed {ok.min():.1f}-{ok.max():.1f}")
    for n, v in base.items():
        P[n].value = v
    json.dump(dict(table=table, profiles=prof, chi=rows, fom=r.fom), open(OUT + "_summary.json", "w"), indent=1,
              default=float)
    log("done")


if __name__ == "__main__":
    main()

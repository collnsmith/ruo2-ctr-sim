"""Compile the final data of this analysis into plain CSV files, one per plotted object, sorted in directories
(plot_data/). Everything is rebuilt from the raw ROI sums, the CTR projects and the saved results of the scripts in this
folder. Run from anywhere: python make_plot_data.py"""
import csv, json, os, shutil, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))
sys.modules["xraydb"] = None
import numpy as np
from ctrfit.echem.external import parse_roi_sums
from ctrfit.fit.global_fit import film_of
from ctrfit.project.project import Project

OUT = os.path.join(here, "plot_data")
OH = os.path.join(here, "..", "2026-11-05_oh_fits")
RHE = 0.238                                     # logbook: -0.238 V Ag/AgCl = 0 V RHE
E1, E2, CA = 0.0, 0.7, 30.0
BINS = np.linspace(E1, E2, 15)
J = lambda name: json.load(open(os.path.join(here, name)))


def write(rel, header, rows, note=None):
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        if note:
            for line in note.strip().splitlines():
                f.write(f"# {line}\n")
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.6g}" if isinstance(v, float) else v) for v in r])


def hkl(H, K, L):
    return f"{H}_{K}_{L:g}"


# ------------------------------------------------------------------ the stationary scans
cv3, timing = J("cv3.json"), J("cv_timing.json")
pn, b3, rb = J("process_new.json"), J("process_batch3.json"), J("robust_cv.json")
SC = {}
for s, r in cv3.items():                          # first batch: roles, timing and windows from the extraction
    SC[int(s)] = dict(film=r["film"], H=r["H"], K=r["K"], L=r["L"], rate=5, start=r["start"], stop=min(r["stop"], r["n"] - 1) if "n" in r else r["stop"],
                      sum=(r["ru"], r["ti"]), ru=r["ru"], ti=r["ti"], P=timing[s]["P"], shift=timing[s]["shift"],
                      timing="data" if timing[s]["how"].startswith("period") else "nominal 0.3 images/s",
                      kind="control" if (r["H"], r["K"]) == (2, 0) else "OH")
SC[147]["timing"] = "Ru peak spacing (71.5), hold at the logged Start, no shift"
SC[223] = dict(film="3.3", H=0, K=-1, L=3.16, rate=5, start=51, stop=410, sum=("ROI1",), ru=None, ti=None,
               P=pn["223"]["P"], shift=pn["223"]["shift"], timing="data", kind="OH")
SC[341] = dict(film="7", H=0, K=1, L=2.75, rate=5, start=21, stop=169, sum=("ROI1", "ROI2"), ru="ROI2", ti="ROI1",
               P=b3["341"]["P"], shift=b3["341"]["shift"], timing="data", kind="OH")
SC[343] = dict(film="7", H=0, K=1, L=2.75, rate=5, start=20, stop=143, sum=("ROI1", "ROI2"), ru="ROI2", ti="ROI1",
               P=b3["343"]["P"], shift=b3["343"]["shift"], timing="period of 341", kind="OH")
NOM = 0.30 * 2 * (E2 - E1) / 0.005               # nominal images per cycle at 0.3 images/s
for s, f, H, K, L, a, b, cols in ((126, "3.3", -2, 0, 1.75, 50, 227, ("ROI1", "ROI2")), (227, "3.3", -1, 0, 2.32, 25, 341, ("ROI1",)),
                                  (135, "3.3", 1, 0, 1.75, 50, 158, ("ROI1",)), (519, "4.1", -2, 0, 1.75, 20, 242, ("ROI1",))):
    SC[s] = dict(film=f, H=H, K=K, L=L, rate=5, start=a, stop=b, sum=cols, ru=None, ti=None, P=NOM, shift=0.0,
                 timing="nominal 0.3 images/s", kind="control")
SC[127] = dict(film="3.3", H=-2, K=0, L=1.0, rate=25, start=31, stop=78, sum=("ROI1", "ROI2"), ru=None, ti=None, P=None,
               shift=0.0, timing="none (25 mV/s, too few images per cycle)", kind="control")


def potential(n, n0, P):
    ph = ((n - n0) / P) % 1.0
    E = np.where(ph < 0.5, E1 + (E2 - E1) * 2 * ph, E2 - (E2 - E1) * (2 * ph - 1))
    return E, np.where(ph < 0.5, 1, -1)


def load(s):
    return parse_roi_sums(open(os.path.join(here, "data", f"ROIs_s{s}_sums.txt")).read())[s]


summary = []
for s in sorted(SC):
    c = SC[s]; t = load(s)
    col = lambda name: np.asarray(t[name], float)
    err = lambda name: col(name + "_err") if name + "_err" in t else np.sqrt(np.maximum(np.abs(col(name)), 1.0))
    y = sum(col(k) for k in c["sum"]); sy = np.sqrt(sum(err(k) ** 2 for k in c["sum"]))
    a, b = c["start"], min(c["stop"], y.size - 1)      # a Stop past the last image: the last image
    n = np.arange(b + 1, dtype=float)
    base = y[a:a + 6].mean()
    if c["P"]:
        fca = CA / (2 * (E2 - E1) / (c["rate"] / 1000.0))
        n0 = a + fca * c["P"] + c["shift"]
        E, br = potential(n, n0, c["P"])
        E = np.where(n < n0, 0.0, E); branch = np.where(n < a, "before", np.where(n < n0, "hold", np.where(br > 0, "anodic", "cathodic")))
        E = np.where(n < a, np.nan, E)
    else:
        E = np.full(n.size, np.nan); branch = np.where(n < a, "before", "cv")
    ru = col(c["ru"]) if c["ru"] else None; ti = col(c["ti"]) if c["ti"] else None
    rows = []
    for i in range(b + 1):
        rows.append([i, branch[i], E[i], E[i] + RHE if np.isfinite(E[i]) else np.nan, y[i], sy[i], y[i] / base, sy[i] / base,
                     ru[i] if ru is not None else np.nan, ti[i] if ti is not None else np.nan,
                     ru[i] / ru[a:a + 6].mean() if ru is not None else np.nan, ti[i] / ti[a:a + 6].mean() if ti is not None else np.nan])
    name = f"s{s}_{hkl(c['H'], c['K'], c['L'])}.csv"
    film_dir = f"{c['film']}nm"
    write(f"01_cv_traces/{film_dir}/{name}",
          ["image", "branch", "E_V_AgAgCl", "E_V_RHE", "sum_counts", "sum_err", "sum_rel", "sum_rel_err", "Ru_counts", "Ti_counts", "Ru_rel", "Ti_rel"],
          rows, note=f"scan {s}, {c['film']} nm ({c['H']} {c['K']} {c['L']:g}), {c['rate']} mV/s, {c['kind']}; EC start image {a}, last image {b}\n"
                     f"sum = {' + '.join(c['sum'])}; Ru = {c['ru']}; Ti = {c['ti']}; *_rel relative to the first 6 images of the 0 V hold\n"
                     f"E assigned from the timing: {c['timing']} (period {c['P'] if c['P'] else 'n/a'} images, shift {c['shift']:+.1f})")
    # cycle-averaged curves (14 E bins per sweep), the same method as extract3.py
    if c["P"]:
        k = np.arange(a, b); k = k[(branch[k] == "anodic") | (branch[k] == "cathodic")]   # as extract3.py (last image excluded)
        crow = []
        for brn in ("anodic", "cathodic"):
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                q = k[(branch[k] == brn) & (E[k] >= lo) & (E[k] < hi)]
                if q.size < 2: continue
                out = [brn, 0.5 * (lo + hi), 0.5 * (lo + hi) + RHE, int(q.size)]
                for sig in (y / base, ru / ru[a:a + 6].mean() if ru is not None else None, ti / ti[a:a + 6].mean() if ti is not None else None):
                    out += [float(np.mean(sig[q]) - 1), float(np.std(sig[q]) / np.sqrt(q.size))] if sig is not None else [np.nan, np.nan]
                crow.append(out)
        write(f"02_cv_curves/{film_dir}/{name}", ["branch", "E_V_AgAgCl", "E_V_RHE", "n_images", "sum_change", "sum_err", "Ru_change", "Ru_err", "Ti_change", "Ti_err"],
              crow, note=f"scan {s}: change relative to the 0 V hold, averaged over cycles in 0.05 V bins; timing {c['timing']}")
    # summary row
    if str(s) in rb:
        pp = (rb[str(s)]["sum"]["sine_pp"], rb[str(s)]["Ru"]["sine_pp"], rb[str(s)]["Ti"]["sine_pp"], rb[str(s)]["sum"]["noise"])
        lin = (cv3[str(s)]["sum"]["change"], cv3[str(s)]["sum"]["change_err"])
    elif str(s) in pn:
        r = pn[str(s)]; pp = (r["sine_pp"], np.nan, np.nan, r["noise"]); lin = (r.get("change", np.nan), r.get("change_err", np.nan))
    else:
        r = b3[str(s)]; pp = (r["sine_pp"], r.get("Ru_sine_pp", np.nan), r.get("Ti_sine_pp", np.nan), r["noise"]); lin = (r.get("change", np.nan), r.get("change_err", np.nan))
    if c["kind"] == "control": lin = (np.nan, np.nan)
    if s == 147: lin = (np.nan, np.nan)            # hysteretic: a linear change in E does not describe it
    ratio = float(col(c["ru"])[a:a + 6].mean() / col(c["ti"])[a:a + 6].mean()) if c["ru"] else np.nan
    summary.append([s, c["film"], c["H"], c["K"], c["L"], c["kind"], c["rate"], a, b, c["timing"], c["P"] if c["P"] else np.nan,
                    pp[0], pp[1], pp[2], pp[3], lin[0], lin[1], ratio])
write("03_cv_summary/cv_amplitudes.csv",
      ["scan", "film_nm", "H", "K", "L", "kind", "rate_mV_s", "start_image", "last_image", "timing", "period_images",
       "sum_pp", "Ru_pp", "Ti_pp", "noise_per_image", "sum_change_0_to_0.7V", "sum_change_err", "Ru_over_Ti_counts"],
      summary, note="Peak-to-peak of the best sine (period 60-130 images, phase free, linear drift term; 127: 8-40 images), as fractions.\n"
                    "sum_change_0_to_0.7V: linear fit of the change against E over the sweeps (OH-sensitive scans only; 147 omitted, hysteretic).\n"
                    "Excluded from fits: 147 (hysteretic). Not used: 224, 225 (staircases, 04_staircases).")

# ------------------------------------------------------------------ staircases
for s, last, L in ((224, 448, 3.16), (225, 427, 2.75)):
    t = load(s); y = np.asarray(t["ROI1"], float); base = y[20:50].mean()
    edges = np.linspace(50, last + 1, 16).round().astype(int)
    level = np.full(last + 1, np.nan)
    for i, (e0, e1) in enumerate(zip(edges[:-1], edges[1:])):
        level[e0:e1] = 0.05 * i
    write(f"04_staircases/s{s}_0_-1_{L:g}_trace.csv", ["image", "E_V_AgAgCl", "counts", "rel"],
          [[i, level[i], y[i], y[i] / base] for i in range(last + 1)],
          note=f"scan {s}, 3.3 nm (0 -1 {L:g}): 0 to 0.7 V in 0.05 V steps of 90 s from image 50 (levels assumed equal in length).\n"
               "Recorded after the window film broke; not used (README, outline B7).")
    r = pn[str(s)]
    write(f"04_staircases/s{s}_0_-1_{L:g}_levels.csv", ["E_V_AgAgCl", "E_V_RHE", "change", "err"],
          [[e, e + RHE, v, er] for e, v, er in zip(r["levels_E"], r["change"], r["err"])],
          note=f"scan {s}: mean of each level (first 3 images after a step skipped) relative to the 0 V hold before image 50")

# ------------------------------------------------------------------ CTR rods (data), 7 nm repeats split by L grid
p = Project.load(os.path.join(OH, "oh_rods_fit.ctrproj.json"))
SCANS = {("4.1 nm", "0 -1"): "483-494 (0, -0.1, 0.15, 0.7, -0.238, 0.35 V), OCV 504-505 (fresh spot)",
         ("7 nm", "0 1"): "OCV 317-320, 0.7 325-326 (a) and 339-340 (b), 0.35 327-328, 0.15 329-330, 0 331-332, -0.1 333-334 (a) and 337-338 (b)"}
for ds in p.datasets:
    f = film_of(ds).split("|")[0]; rod = ds.rod_labels[0]; v = ds.meta.get("potential_V")
    cond = "OCV" if v is None else f"{v:g}V"
    parts = [("", np.ones(len(ds), bool))]
    if f == "7 nm" and rod == "0 1" and v in (-0.1, 0.7):
        on = np.abs(ds.L * 100 - np.round(ds.L * 100)) < 1e-4
        parts = [("_a", on), ("_b", ~on)]
    for suf, m in parts:
        o = np.argsort(ds.L[m])
        write(f"05_ctr_rods/{f.replace(' ', '')}/rod_{rod.replace(' ', '_')}_{cond}{suf}.csv", ["L", "F", "sigma", "sigma_eff"],
              [[float(a_), float(b_), float(c_), float(d_)] for a_, b_, c_, d_ in zip(ds.L[m][o], ds.F[m][o], ds.sigma[m][o], ds.sigma_eff[m][o])],
              note=f"{f} ({rod}) at {cond}{' (' + suf[1:] + ': see README)' if suf else ''}; |F| as processed (arbitrary scale per file); "
                   f"sigma_eff includes the systematic floor\nscans: {SCANS.get((f, rod), 'see timeline.txt')}")
tr = J("timeline_rods.json")
write("06_ctr_rod_comparisons/rod_pairs.csv", ["comparison", "rms_ln_ratio"], [[k, v] for k, v in tr.items()],
      note="rms of ln(I_a / I_b) over the OH rod, one free scale, +-0.08 around the TiO2 Bragg peak left out (timeline_rods.py)\n"
           "7 nm a/b: the two measurements at -0.1 V and 0.7 V (README); '->' = measured back to back")

# ------------------------------------------------------------------ CTR OH fraction under the bridging-O assumptions
summ = json.load(open(os.path.join(OH, "oh_rods_fit_summary.json")))["results"]
c3 = J("ohfit_c3_summary.json")
rows = []
for lab, r in (("Rao 2017 (logbook RHE scale)", c3), ("bridging O fixed at 0", summ["B fixed 0"]), ("bridging O fitted (unconstrained)", summ["A shared per potential"])):
    for key, q in sorted(r["profiles"].items()):
        film, pot = key.split(" nm ")
        E = np.nan if pot == "OCV" else float(pot.split()[0])
        dz = r["dz_Obr"].get(pot, np.nan)
        rows.append([lab, film, pot, E, E + RHE if np.isfinite(E) else np.nan, q["best"], q["lo"], q["hi"], dz, r["red_chi2"]])
write("07_ctr_oh_fraction/oh_fraction_by_assumption.csv",
      ["assumption", "film_nm", "condition", "E_V_AgAgCl", "E_V_RHE", "x_OH_best", "x_OH_lo", "x_OH_hi", "dz_Obr_A", "red_chi2"], rows,
      note="OH fraction on the CUS sites from the (0 +-1) CTR fits, film fixed; lo/hi: delta chi2 <= 4 red chi2 (profiles).\n"
           "OCV rows are not comparable between films (different rest potentials and, for 4.1 nm, a different spot).")

# ------------------------------------------------------------------ CTR -> CV
c2 = J("ctr2cv.json")
rows = []
for key, byvar in c2["predictions"].items():
    for var, q in byvar.items():
        for t in q:
            rows.append([key, var, t["E"], t["E"] + RHE, t["pred"], t["lo"], t["hi"]])
write("08_ctr_to_cv/predicted_from_ctr_states.csv", ["film_reflection", "assumption", "E_V_AgAgCl", "E_V_RHE", "change_pred", "change_lo", "change_hi"], rows,
      note="Intensity change from the 0 V state predicted from the CTR-fitted surface state at each CTR potential (ctr2cv.py);\n"
           "lo/hi from the x_OH range at that potential. Measured CV curves: 02_cv_curves.")
write("08_ctr_to_cv/measured_vs_predicted.csv", ["film", "reflection", "E_V_AgAgCl", "measured_cv", "pred_rao", "pred_dz0", "pred_dz_fitted"],
      [[t["film"], t["hkl"], t["E"], t["measured"]] + [list(t["predicted"].values())[i] for i in range(3)] for t in c2["table"]],
      note="CV value: cycle average (both sweeps) of the sum at that E; predictions in the order Rao, dz_Obr = 0, dz_Obr fitted")

# ------------------------------------------------------------------ model sensitivity at the CV reflections
m, P = p.model, p.model.params
lead = {}
for ds in p.datasets:
    lead.setdefault(film_of(ds).split("|")[0], ds.scope)
pts = [(0, -1, 1.25), (0, -1, 2.75), (0, -1, 2.84), (0, -1, 3.16), (0, -1, 3.25), (1, 0, 1.75), (-1, 0, 2.32), (-2, 0, 1.75), (-2, 0, 1.0), (2, 0, 1.75)]
rows = []
for name, src, tl in (("3.3 (4.1 nm structure at 10.2 TL)", "4.1 nm", 10.2), ("4.1", "4.1 nm", None), ("7", "7 nm", None)):
    v = dict(P.view(lead[src]), theta=1.0)
    if tl: v["thickness_mean_tl"] = tl
    for H, K, L in pts:
        I0 = m.intensity(H, K, [L], v)[0]
        oh = [m.intensity(H, K, [L], dict(v, x_OH=0.7, dz_Obr=d))[0] / m.intensity(H, K, [L], dict(v, x_OH=0.3, dz_Obr=d))[0] - 1 for d in (0.0, 0.35)]
        lat = m.intensity(H, K, [L], dict(v, eps_perp=v["eps_perp"] + 0.1))[0] / I0 - 1
        rows.append([name, H, K, L, oh[0], oh[1], lat])
write("09_model_sensitivity/sensitivity.csv", ["film_model", "H", "K", "L", "OH_0.3_to_0.7_dz0", "OH_0.3_to_0.7_dz0.35", "film_spacing_plus_0.1pct"], rows,
      note="Coherent model, structures from the OH-fit project. The 7 nm model has its top relaxed in [001] (coherent_only), so its\n"
           "H != 0 sensitivities are zero by construction; use the 3.3/4.1 nm rows for the controls.")

# ------------------------------------------------------------------ work function (other samples, read from the group's plot)
write("10_work_function/kpfm_jeong2026_read_from_plot.csv", ["thickness_nm", "work_function_eV"],
      [[1.5, 4.25], [2.5, 4.38], [3.0, 4.35], [3.2, 4.73], [3.8, 5.40], [4.0, 4.86], [5.0, 4.78], [5.2, 4.84], [7.0, 4.44], [8.0, 4.66], [17.0, 5.05]],
      note="Jeong et al., Nat. Commun. 17, 2516 (2026): KPFM on different samples from ours; values read by eye from the plot (+-0.02 eV)")
print("written:", sum(len(fs) for _, _, fs in os.walk(OUT)), "files in", OUT)

"""Command line fitting (headless).

    python -m ctrfit fit model.json data.csv [more data files] [options]

model.json comes from Model.save(): template, fixed settings and parameters (the 'fit' flags pick
the free parameters). Data files: .csv with a header (H,K,L,F,sigma), whitespace .dat (H K L F
sigma) or .json from Dataset.save(). Writes result.json, summary.txt and PNG plots to --out.
"""
import argparse
import sys
from pathlib import Path


def build_parser():
    ap = argparse.ArgumentParser(prog="ctrfit", description="CTR fitting for RuO2(110)/TiO2(110) films")
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit", help="fit a model to one or more datasets")
    f.add_argument("model", help="model JSON (Model.save)")
    f.add_argument("data", nargs="+", help="data files (.csv, .dat, .json)")
    f.add_argument("--out", default="fit_results", help="output folder (default: fit_results)")
    f.add_argument("--method", default="de+lsq", choices=["de+lsq", "de", "lsq"],
                   help="global search then refinement (default), or one of them")
    f.add_argument("--fom", default="chi2", choices=["chi2", "log", "R1"], help="figure of merit for DE")
    f.add_argument("--maxiter", type=int, default=50, help="DE generations")
    f.add_argument("--popsize", type=int, default=12, help="DE population size per free parameter")
    f.add_argument("--seed", type=int, default=1, help="DE random seed")
    f.add_argument("--floor", type=float, default=None, help="systematic error floor (relative) for all data")
    f.add_argument("--intensities", action="store_true", help="data files hold I and sigma_I, not F")
    f.add_argument("--rods", default=None, help="only these rods, e.g. '0 1; 1 0'")
    f.add_argument("--per-dataset", default="", help="global fit of several data files: parameters fitted "
                   "separately per dataset, e.g. 'x_OH,z_OH' (all other free parameters are shared)")
    f.add_argument("--profile", default=None, help="after DE, scan a parameter on a grid and keep the best "
                   "minimum, e.g. 'thickness_mean_tl:15:25:0.5'")
    f.add_argument("--no-plots", action="store_true", help="skip the PNG plots")
    g = sub.add_parser("gui", help="open the fitting window")
    g.add_argument("files", nargs="*", help="data files, a model JSON, or a .ctrproj.json project")
    ak = sub.add_parser("ask", help="let Claude run the fit (needs anthropic and an API key)")
    ak.add_argument("model", help="model JSON (Model.save)")
    ak.add_argument("data", nargs="+", help="data files (.csv, .dat, .json)")
    ak.add_argument("--request", "-r", default="Fit this data with the guided workflow and report how reliable each "
                    "value is.", help="what to ask Claude")
    ak.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh"])
    ak.add_argument("--out", default="claude_fit", help="output folder: model_fitted.json, transcript.txt")
    sub.add_parser("launcher", help="open the launcher with a button for every app")
    b = sub.add_parser("beamline", help="open the beamline helper (indexer, UB refinement, angles, macros, calculators)")
    b.set_defaults()
    ix = sub.add_parser("index", help="best-guess HKL of Bragg peaks from SPEC psic angles")
    ix.add_argument("peaks", help="text file, one peak per line: del eta chi phi nu mu (or name=value)")
    ix.add_argument("--energy", type=float, required=True, help="photon energy (keV)")
    ix.add_argument("--lattice", default="tio2_surface", help="tio2_surface, tio2, ruo2_surface, or 'a b c al be ga'")
    ix.add_argument("--rule", default=None, choices=["tio2_surface", "all"],
                    help="allowed reflections (default: tio2_surface for TiO2 lattices, else all)")
    ix.add_argument("--normal", default=None, help="surface normal in the phi frame, e.g. '0 0 1'")
    ix.add_argument("--tol-q", type=float, default=0.01, help="relative |Q| tolerance")
    ix.add_argument("--save-ub", default=None, help="write the UB matrix to this JSON file")
    sub.add_parser("echem", help="open the in-situ CV window (stationary SPEC scan + potentiostat)")
    cv = sub.add_parser("cv", help="in-situ CV: stationary SPEC scan vs potential, fits, CTR prediction")
    cv.add_argument("spec", help="SPEC data file")
    cv.add_argument("--scan", type=int, default=None, help="scan number (default: the last stationary scan)")
    cv.add_argument("--ec", default=None, help="potentiostat file (.mpr needs galvani, .mpt, .csv)")
    cv.add_argument("--cv", default=None, metavar="HOLD_V,HOLD_S,LOWER,UPPER,RATE_mVs,CYCLES[,up|down[,DELAY_S]]",
                    help="define the CV by hand instead of --ec, e.g. '0.5,300,0.4,1.4,10,3,up,60'")
    cv.add_argument("--counter", default=None)
    cv.add_argument("--monitor", default=None)
    cv.add_argument("--sync", default="absolute", choices=["absolute", "manual", "events"])
    cv.add_argument("--offset", type=float, default=0.0, help="EC time = SPEC time + offset (s)")
    cv.add_argument("--background", default="linear", choices=["none", "constant", "linear", "exponential"])
    cv.add_argument("--transitions", type=int, default=1, help="sigmoidal transitions per sweep direction")
    cv.add_argument("--cycles", default="all")
    cv.add_argument("--hkl", default=None, help="'H K L' if the scan has no #Q")
    cv.add_argument("--path", default="H2O -> OH -> O", help="composition path for the CTR model")
    cv.add_argument("--path-transitions", default="0.8, 0.03; 1.1, 0.04", help="'E0, width; ...' (V)")
    cv.add_argument("--reference", default="H2O=1", help="surface state during the relaxation period")
    cv.add_argument("--fit-path", action="store_true", help="fit the composition path through the CTR model")
    cv.add_argument("--out", default="insitu_cv", help="output folder: data.csv, results.json, summary.txt, PNGs")
    rf = sub.add_parser("refine", help="refine UB (and lattice, motor offsets) from a list of reflections")
    rf.add_argument("reflections", help="text file, one per line: H K L del eta chi phi nu mu [energy] (or reflex JSON)")
    rf.add_argument("--energy", type=float, default=None, help="photon energy (keV) for lines without one")
    rf.add_argument("--lattice", default="tio2_surface", help="tio2_surface, tio2, ruo2_surface, or 'a b c al be ga'")
    rf.add_argument("--free", default="orientation", choices=["orientation", "scale", "abc", "all", "ub"],
                    help="what to refine besides the orientation")
    rf.add_argument("--offsets", default="", help="motor zero offsets to refine, e.g. 'del eta'")
    rf.add_argument("--save-ub", default=None, help="write UB, lattice and residuals to this JSON file")
    mc = sub.add_parser("macro", help="SPEC macro for rod scans (see the beamline helper for more kinds)")
    mc.add_argument("--rods", required=True, help="e.g. '0 1; 1 0'")
    mc.add_argument("--lmin", type=float, default=0.3)
    mc.add_argument("--lmax", type=float, default=4.0)
    mc.add_argument("--step", type=float, default=0.02)
    mc.add_argument("--time", type=float, default=1.0, help="count time (s)")
    mc.add_argument("--scale-times", action="store_true", help="count times from the model (t ~ 1/I)")
    mc.add_argument("--energy", type=float, default=None, help="photon energy (keV) for the model")
    mc.add_argument("--out", default=None, help="write the macro here (default: print it)")
    return ap


def _lattice(text):
    from .beamline.psic import LATTICES, Lattice
    if text in LATTICES:
        return LATTICES[text]()
    vals = [float(x) for x in text.split()]
    if len(vals) != 6:
        raise SystemExit(f"--lattice: use one of {', '.join(LATTICES)} or 'a b c alpha beta gamma'")
    return Lattice(*vals)


def cmd_index(a):
    import json
    from .beamline.indexing import ALLOWED, index_peaks
    from .beamline.psic import energy_to_wavelength, parse_angle_lines
    peaks = parse_angle_lines(Path(a.peaks).read_text(encoding="utf-8"))
    rule = a.rule or ("tio2_surface" if a.lattice.startswith("tio2") else "all")
    normal = [float(x) for x in a.normal.split()] if a.normal else None
    res = index_peaks(peaks, energy_to_wavelength(a.energy), _lattice(a.lattice), ALLOWED[rule], tol_q=a.tol_q,
                      normal_phi=normal)
    print(res.summary)
    if a.save_ub and res.UB is not None:
        Path(a.save_ub).write_text(json.dumps(dict(UB=res.UB.tolist(), energy_kev=a.energy), indent=1))
        print(f"wrote {a.save_ub}")
    return 0


def cmd_echem(a):
    from .app.echem_window import main as ew_main
    return ew_main([])


def cmd_cv(a):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from .echem.session import Options, Session
    from .project import plots
    o = Options(scan=a.scan, counter=a.counter, monitor=a.monitor, sync_mode=a.sync, offset=a.offset,
                background=a.background, n_transitions=a.transitions, cycles=a.cycles, states=a.path,
                transitions=a.path_transitions, reference=a.reference)
    if a.hkl:
        o.hkl = tuple(float(x) for x in a.hkl.replace(",", " ").split())
    if a.cv:
        parts = [x.strip() for x in a.cv.split(",")]
        if len(parts) < 6:
            raise SystemExit("--cv: give HOLD_V,HOLD_S,LOWER,UPPER,RATE_mVs,CYCLES[,up|down[,DELAY_S]]")
        o.cv_manual = True
        o.cv_hold_V, o.cv_hold_s, o.cv_lower, o.cv_upper, o.cv_rate = map(float, parts[:5])
        o.cv_cycles = int(parts[5])
        if len(parts) > 6:
            o.cv_first = parts[6]
        if len(parts) > 7:
            o.cv_delay = float(parts[7])
    elif not a.ec:
        raise SystemExit("give the potentiostat file (--ec) or define the CV (--cv)")
    s = Session(o)
    try:
        s.load_spec(a.spec)
        if a.ec and not a.cv:
            s.load_ec(a.ec)
        s.run()
        if a.fit_path:
            s.fit_path()
        s.coverage()
    except (ValueError, KeyError, RuntimeError) as ex:
        raise SystemExit(str(ex))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s.export_csv(out / "data.csv")
    s.export_json(out / "results.json")
    (out / "summary.txt").write_text(s.summary() + "\n", encoding="utf-8")
    r = s.res
    for name, fn, args in (("sync", plots.draw_echem_sync, (r, s.ec)), ("background", plots.draw_echem_background, (r,)),
                           ("i_vs_v", plots.draw_echem_iv, (r,)), ("transitions", plots.draw_echem_fit, (r,)),
                           ("ctr_model", plots.draw_echem_model, (r, r.point_model))):
        fig = Figure(figsize=(9, 6), layout="constrained")
        fn(fig, *args)
        fig.savefig(out / f"{name}.png", dpi=110)
    print(s.summary())
    print(f"\nwrote data.csv, results.json, summary.txt and 5 plots to {out}")
    return 0


def cmd_refine(a):
    import json
    from .beamline.refine import ReflectionList, parse_reflection_lines, refine_ub
    path = Path(a.reflections)
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("{"):
        refl = ReflectionList.load(path)
    else:
        refl = ReflectionList()
        try:
            rows = parse_reflection_lines(text, a.energy)
        except ValueError as ex:
            raise SystemExit(f"{ex} (give --energy for lines without an energy)")
        for row in rows:
            if row["hkl"] is None:
                raise SystemExit("every line needs H K L in the file (guessing HKL needs the window)")
            refl.add(row["hkl"], row["angles"], row["energy_kev"], row["label"])
    try:
        res = refine_ub(refl, _lattice(a.lattice), a.free, a.offsets.replace(",", " ").split())
    except ValueError as ex:
        raise SystemExit(str(ex))
    print(res.summary)
    if a.save_ub:
        Path(a.save_ub).write_text(json.dumps(dict(res.to_dict(), **refl.to_dict()), indent=1))
        print(f"wrote {a.save_ub}")
    return 0


def cmd_macro(a):
    from .beamline import macros
    from .core.settings import DEFAULTS, parse_rods
    from .model.model import Model
    settings = dict(DEFAULTS, **({"energy_kev": a.energy} if a.energy else {}))
    m = macros.rod_macro(parse_rods(a.rods), None, a.lmin, a.lmax, a.step, a.time, Model.from_settings(settings),
                         scale_times=a.scale_times)
    text = m.text("rod scans")
    if a.out:
        macros.write_macro(a.out, text)
        print(f"wrote {a.out}: {m.n_points} points, about {macros.duration_text(m.seconds)}")
    else:
        print(text, end="")
    return 0


def cmd_ask(a):
    from .assistant.agent import FitAgent, transcript_text
    from .data.dataset import Dataset
    from .model.model import Model

    def show(kind, data):
        if kind == "text":
            print(f"\nClaude: {data}")
        elif kind == "tool_call":
            print(f"  -> {data['name']} {data['input'] or ''}")
        elif kind == "tool_result" and data["error"]:
            print(f"  <- error: {data['output']}")
        elif kind == "error":
            print(f"  note: {data}")

    agent = FitAgent(Model.load(a.model), [Dataset.load(p) for p in a.data], effort=a.effort, on_event=show)
    agent.ask(a.request)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    agent.model.save(out / "model_fitted.json")
    (out / "transcript.txt").write_text(transcript_text(agent.log), encoding="utf-8")
    if agent.result is not None:
        agent.result.save(out / "result.json")
    print(f"\nabout ${agent.cost_usd:.2f}; wrote {out}/model_fitted.json, transcript.txt")
    return 0


def cmd_launcher(a):
    from .app.launcher_window import main as launcher_main
    return launcher_main(["ctrfit"])


def cmd_beamline(a):
    from .app.beamline_window import main as bl_main
    return bl_main(["ctrfit"])


def cmd_gui(a):
    from .app.fit_window import main as gui_main
    return gui_main(["ctrfit"] + list(a.files))


def cmd_fit(a):
    import matplotlib
    matplotlib.use("Agg")
    from .core.settings import parse_rods
    from .data.dataset import Dataset
    from .fit.fit import Fit
    from .model.model import Model
    from .project.report import write_report

    model = Model.load(a.model)
    datasets = []
    for path in a.data:
        kw = {} if str(path).lower().endswith(".json") else dict(intensities=a.intensities)
        ds = Dataset.load(path, **kw)
        if a.floor is not None:
            ds.sys_floor = a.floor
        if a.rods:
            ds = ds.select_rods(parse_rods(a.rods))
        datasets.append(ds)
    per = [n.strip() for n in a.per_dataset.split(",") if n.strip()]
    if per:
        from .fit.global_fit import global_fit
        fit = global_fit(model, datasets, per_dataset=per, fom=a.fom)
    else:
        fit = Fit(model, datasets, fom=a.fom)
    print(f"{len(fit.names)} free parameters: {', '.join(fit.names)}")
    print(f"{fit.n_data} data points in {len(datasets)} dataset(s)")
    if "de" in a.method:
        st = fit.run("de", maxiter=a.maxiter, popsize=a.popsize, seed=a.seed)
        print(f"DE: {st['nfev']} evaluations in {st['seconds']:.1f} s, {st['fom_name']} = {st['fom']:.6g}")
    if a.profile:
        import numpy as np
        name, lo, hi, step = a.profile.split(":")
        prof = fit.profile(name, np.arange(float(lo), float(hi) + 1e-9, float(step)))
        print(f"profile of {name}: best grid point {min(prof, key=lambda d: d['chi2'])['value']:g}")
    if "lsq" in a.method:
        st = fit.refine()
        print(f"least squares: {st['nfev']} evaluations, chi2 = {st['fom']:.6g}")
    res = fit.report()
    out = Path(a.out)
    if a.no_plots:
        out.mkdir(parents=True, exist_ok=True)
        res.save(out / "result.json")
        (out / "summary.txt").write_text(res.summary + "\n", encoding="utf-8")
        files = [out / "result.json", out / "summary.txt"]
    else:
        files = write_report(fit, res, out)
    model.save(out / "model_fitted.json")
    print(res.summary)
    print(f"\nwrote {len(files) + 1} files to {out}")
    return 0


def main(argv=None):
    a = build_parser().parse_args(argv)
    return {"fit": cmd_fit, "gui": cmd_gui, "beamline": cmd_beamline, "index": cmd_index, "refine": cmd_refine,
            "echem": cmd_echem, "cv": cmd_cv,
            "macro": cmd_macro, "ask": cmd_ask, "launcher": cmd_launcher}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())

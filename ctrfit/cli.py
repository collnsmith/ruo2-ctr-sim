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
    b = sub.add_parser("beamline", help="open the beamline helper (indexer, angles, macros, calculators)")
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
    return {"fit": cmd_fit, "gui": cmd_gui, "beamline": cmd_beamline, "index": cmd_index,
            "macro": cmd_macro}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())

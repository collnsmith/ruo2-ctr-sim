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
    f.add_argument("--no-plots", action="store_true", help="skip the PNG plots")
    return ap


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
    fit = Fit(model, datasets, fom=a.fom)
    print(f"{len(fit.names)} free parameters: {', '.join(fit.names)}")
    print(f"{fit.n_data} data points in {len(datasets)} dataset(s)")
    if "de" in a.method:
        st = fit.run("de", maxiter=a.maxiter, popsize=a.popsize, seed=a.seed)
        print(f"DE: {st['nfev']} evaluations in {st['seconds']:.1f} s, {st['fom_name']} = {st['fom']:.6g}")
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
    return {"fit": cmd_fit}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())

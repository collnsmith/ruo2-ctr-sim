"""Fit the synthetic example and compare with the truth.

    python make_data.py     (once; the files are also in the repository)
    python run_fit.py       film on the even rods, then surface on the odd rods, then everything

The same fit from the command line:
    python -m ctrfit fit model.json data.csv --out fit_results
"""
import json
import sys

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.modules["xraydb"] = None

from ctrfit import Dataset, Fit, Model             # noqa: E402
from ctrfit.project.report import write_report     # noqa: E402

FILM = ["thickness_mean_tl", "roughness_tl", "eps_perp", "scale"]
SURFACE = ["theta", "x_OH", "z_OH", "scale"]


def main(out=HERE / "fit_results"):
    model = Model.load(HERE / "model.json")
    data = Dataset.load(HERE / "data.csv")
    even = data.select_rods(["0 0", "1 1", "0 2", "2 0"], name="even")
    odd = data.select_rods(["0 1", "1 0", "0 3", "0 5"], name="odd")
    free = model.params.free_names

    model.params.fit_only(FILM)                        # step 1: film on the even rods
    film = Fit(model, even)
    film.run("de", maxiter=40, popsize=10, seed=1)
    # thickness fringes give minima about one trilayer apart (traded against eps_perp), so scan it
    prof = film.profile("thickness_mean_tl", np.arange(15.0, 25.01, 0.5))
    print("thickness profile (TL, chi2):", ", ".join(f"{d['value']:g}: {d['chi2']:.0f}" for d in prof))
    film.refine()
    model.params.fit_only(SURFACE)                     # step 2: surface on the odd rods
    Fit(model, odd).run("de", maxiter=30, popsize=10, seed=2)
    model.params.fit_only(free)                        # step 3: all together, refined
    fit = Fit(model, data)
    fit.refine()
    res = fit.report()
    print(res.summary)
    truth = json.loads((HERE / "model_true.json").read_text())
    tv = {p["name"]: p["value"] for p in truth["parameters"]}
    print(f"\n{'parameter':<20} {'true':>9} {'fitted':>10} {'error':>9} {'deviation':>10}")
    for n in res.names:
        print(f"{n:<20} {tv[n]:>9.4g} {res.values[n]:>10.5g} {res.errors[n]:>9.2g} "
              f"{(res.values[n] - tv[n]) / res.errors[n]:>+9.1f}σ")
    write_report(fit, res, out)
    return res


if __name__ == "__main__":
    main()

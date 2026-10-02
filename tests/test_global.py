"""Phase 7: global fits across datasets (shared film, per-dataset surface)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ctrfit import Fit, Model, make_dataset
from ctrfit.fit.global_fit import global_fit, make_per_dataset, series
from ctrfit.project.report import per_dataset_names, write_report
from test_fit import EVEN, FILM_TRUTH, ODD, assert_jointly_recovered, start_model, truth_model

ROOT = Path(__file__).resolve().parents[1]
POTENTIALS = [0.4, 0.8, 1.0, 1.2]
X_OH = [0.2, 0.4, 0.6, 0.8]                 # known trend of the OH share with potential
FREE = list(FILM_TRUTH) + ["x_OH", "scale"]


def potential_series(seed0=200):
    return [make_dataset(truth_model(x_OH=x), rods=EVEN[:2] + ODD[:2], noise_rel=0.04, seed=seed0 + i, step=0.1,
                         name=f"{u:g} V", potential_V=u) for i, (u, x) in enumerate(zip(POTENTIALS, X_OH))]


def start():
    m = start_model(FREE, dict(thickness_mean_tl=19.0, eps_perp=2.1, roughness_tl=0.7, x_OH=0.5, scale=1.05))
    m.params.update(dict(theta=0.85, z_OH=2.05))
    return m


def film_then_refine(fit):
    fit.profile("thickness_mean_tl", np.arange(17.0, 22.01, 0.5))
    fit.refine()
    return fit.report()


def test_make_per_dataset_creates_scoped_copies():
    m = Model.from_settings()
    ds = potential_series()[:2]
    m.params["x_OH"].fit = True
    make_per_dataset(m, ds, ["x_OH"])
    assert [d.scope for d in ds] == ["ds1", "ds2"]
    assert not m.params["x_OH"].fit and m.params["ds1.x_OH"].fit and m.params["ds2.x_OH"].unit == ""
    m.params["ds2.x_OH"].value = 0.9
    assert m.params.view("ds2")["x_OH"] == 0.9 and m.params.view("ds1")["x_OH"] == m.params["x_OH"].value
    m.params.link("ds2.x_OH", "ds1.x_OH + 0.1")              # links across datasets
    assert m.params.view("ds2")["x_OH"] == pytest.approx(m.params["ds1.x_OH"].value + 0.1)
    with pytest.raises(ValueError):
        global_fit(m, ds[:1], per_dataset=["z_OH"])


def test_per_dataset_parameter_changes_only_its_dataset():
    ds = potential_series()[:2]
    m = start()
    fit = global_fit(m, ds, per_dataset=["x_OH"])
    F0 = fit.model_F()
    m.params["ds2.x_OH"].value = 0.9
    F1 = fit.model_F()
    np.testing.assert_array_equal(F0[0], F1[0])
    assert np.max(np.abs(F1[1] / F0[1] - 1)) > 1e-3


@pytest.mark.slow
def test_potential_series_trend_recovered_and_shared_parameters_tighter(tmp_path):
    data = potential_series()
    res = film_then_refine(global_fit(start(), data, per_dataset=["x_OH"]))
    print(res.summary)
    truth = dict(FILM_TRUTH, **{f"ds{i + 1}.x_OH": x for i, x in enumerate(X_OH)})
    assert_jointly_recovered(res, truth)
    pot, x, dx = series(res, "x_OH")
    np.testing.assert_array_equal(pot, POTENTIALS)
    assert np.all(np.abs(x - X_OH) <= 3 * dx) and np.all(np.diff(x) > 0)
    slope = np.polyfit(pot, x, 1, w=1 / dx)[0]
    assert slope == pytest.approx(np.polyfit(POTENTIALS, X_OH, 1)[0], rel=0.1)
    single = film_then_refine(Fit(start(), data[2]))      # one potential alone
    for n in FILM_TRUTH:
        assert res.errors[n] < 0.75 * single.errors[n], (n, res.errors[n], single.errors[n])
    # results and plots
    fit = global_fit(Model.from_dict(res.model), potential_series(), per_dataset=[])
    assert per_dataset_names(res) == ["scale", "x_OH"]
    files = write_report(fit, res, tmp_path)
    assert (tmp_path / "series_x_OH.png").exists() and len(files) > 8


@pytest.mark.slow
def test_cli_global_fit(tmp_path):
    data = potential_series()[:2]
    paths = []
    for i, d in enumerate(data):
        paths.append(tmp_path / f"d{i}.csv")
        d.save(paths[-1])
    start().save(tmp_path / "model.json")
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(ROOT))
    code = "import sys; sys.modules['xraydb'] = None; from ctrfit.cli import main; sys.exit(main())"
    r = subprocess.run([sys.executable, "-c", code, "fit", str(tmp_path / "model.json"), *map(str, paths),
                        "--out", str(tmp_path / "out"), "--method", "lsq", "--per-dataset", "x_OH",
                        "--profile", "thickness_mean_tl:17:22:0.5"],
                       capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((tmp_path / "out" / "result.json").read_text())
    assert {"ds1.x_OH", "ds2.x_OH"} <= set(res["names"])
    assert (tmp_path / "out" / "series_x_OH.png").exists()

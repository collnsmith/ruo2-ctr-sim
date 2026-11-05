"""Phase 7: global fits across datasets (shared film, per-dataset surface)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ctrfit import Fit, Model, make_dataset
from ctrfit.fit.global_fit import film_of, global_fit, make_per_dataset, series
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


def test_per_film_parameters_start_at_each_films_thickness():
    """Two potentials of film A, one of film B: thickness and roughness per film (A's datasets share one
    value, B has its own), x_OH per dataset; each film's thickness starts at its nominal nm."""
    from ctrfit.core.lattice import film_trilayers
    from ctrfit.fit import workflow
    ds = potential_series()[:3]
    for d, (film, nm) in zip(ds, (("A", 7.0), ("A", 7.0), ("B", 3.3))):
        d.meta.update(film=film, thickness_nm=nm)
    assert film_of(ds[0]) == film_of(ds[1]) != film_of(ds[2])
    m = start()
    fit = global_fit(m, ds, per_dataset=["x_OH"], per_film=["thickness_mean_tl", "roughness_tl"])
    p = m.params
    assert p["ds2.thickness_mean_tl"].expr == "ds1.thickness_mean_tl" and not p["ds3.thickness_mean_tl"].linked
    assert p["ds2.roughness_tl"].linked and not p["thickness_mean_tl"].fit
    eps = p["eps_perp"].value
    assert p["ds1.thickness_mean_tl"].value == pytest.approx(film_trilayers(7.0, m.settings["ruo2_a"], eps))
    assert p["ds3.thickness_mean_tl"].value == pytest.approx(film_trilayers(3.3, m.settings["ruo2_a"], eps))
    lo, hi = p["thickness_mean_tl"].min, p["thickness_mean_tl"].max                  # set around film A
    assert p["ds3.thickness_mean_tl"].min < p["ds3.thickness_mean_tl"].value < p["ds3.thickness_mean_tl"].max
    assert p["ds3.thickness_mean_tl"].max - p["ds3.thickness_mean_tl"].min == pytest.approx(hi - lo)
    names = set(fit.names)
    assert {"ds1.thickness_mean_tl", "ds3.thickness_mean_tl", "ds1.x_OH", "ds2.x_OH"} <= names
    assert "ds2.thickness_mean_tl" not in names and "thickness_mean_tl" not in names
    F0 = fit.model_F()
    p["ds1.thickness_mean_tl"].value += 1.0                                          # film A thicker
    F1 = fit.model_F()
    assert all(np.max(np.abs(F1[i] / F0[i] - 1)) > 1e-3 for i in (0, 1)) and np.array_equal(F0[2], F1[2])
    # fitted per dataset after all: the copies are free again (and keep their values)
    make_per_dataset(m, ds, ["thickness_mean_tl"])
    assert not p["ds2.thickness_mean_tl"].linked and p["ds1.thickness_mean_tl"].value == pytest.approx(
        film_trilayers(7.0, m.settings["ruo2_a"], eps) + 1.0)
    with pytest.raises(ValueError, match="not both"):
        global_fit(start(), potential_series()[:2], per_dataset=["x_OH"], per_film=["x_OH"])
    # one film only: per film is the same as shared (no copies)
    one = potential_series()[:2]
    m1 = start()
    workflow.prepare(m1, one, ["x_OH"], ["thickness_mean_tl"])
    assert "ds1.thickness_mean_tl" not in m1.params and "ds2.x_OH" in m1.params
    # the workflow does the same as global_fit for several films
    m2 = start()
    workflow.prepare(m2, ds, ["x_OH"], ["thickness_mean_tl"])
    assert m2.params["ds2.thickness_mean_tl"].linked and not m2.params["ds3.thickness_mean_tl"].linked


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


def test_a_films_dataset_without_its_copy_joins_the_film():
    """Per-film copies made while two films were fitted; then only one film is fitted and a new file of it is
    added: the new dataset follows the film's copy (not the shared value), on regroup and before a run."""
    from ctrfit.fit import workflow
    from ctrfit.fit.global_fit import complete_per_film, make_per_film
    from ctrfit.fit.scans import ScanSelection, regroup
    truth = truth_model()
    mk = lambda name, film, seed: make_dataset(truth, rods=ODD[:1], noise_rel=0.03, seed=seed, step=0.2, name=name,
                                               film=film, thickness_nm=7.0 if film == "7 nm" else 4.1)
    a, b, c = mk("a", "7 nm", 1), mk("b", "7 nm", 2), mk("c", "4 nm", 3)
    model = truth_model()
    sel = ScanSelection([a, b, c])
    datasets, _ = regroup(model, sel)
    make_per_film(model, datasets, ["eps_perp"])
    P = model.params
    lead = next(d.scope for d in datasets if d.name == "a")
    P[f"{lead}.eps_perp"].value = 1.2
    sel.set_use(sel.keys_of("c"), False)                 # one film left: no new per-film copies are made
    sel.add(mk("d", "7 nm", 4))                          # a new file of the 7 nm film
    datasets, notes = regroup(model, sel, datasets)
    d = next(x for x in datasets if x.name == "d")
    assert P[f"{d.scope}.eps_perp"].linked and P.view(d.scope)["eps_perp"] == pytest.approx(1.2)
    assert any("follows its film's" in n for n in notes)
    assert complete_per_film(model, datasets) == []      # nothing more to do
    P.remove(f"{d.scope}.eps_perp")                      # and before a run (prepare) when it is missing
    workflow.prepare(model, datasets)
    assert P[f"{d.scope}.eps_perp"].linked and P.view(d.scope)["eps_perp"] == pytest.approx(1.2)

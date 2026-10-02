"""Guided workflow (film on even rods, surface on odd rods, everything) and the project file."""
import numpy as np
import pytest

from ctrfit import Dataset, Model, make_dataset
from ctrfit.fit import workflow
from ctrfit.project.project import Project
from test_fit import EVEN, FILM_TRUTH, ODD, SURF_TRUTH, assert_recovered, start_model, truth_model

START = dict(thickness_mean_tl=18.4, eps_perp=1.7, roughness_tl=0.9, z_OH=1.85, x_OH=0.35, theta=0.95, scale=1.2)


def test_rod_parity_subsets():
    ds = make_dataset(Model.from_settings(), rods=EVEN[:2] + ODD[:2], step=0.5, seed=1)
    assert workflow.rod_parity_subset(ds, 0).rod_labels == ["0 0", "1 1"]
    assert workflow.rod_parity_subset(ds, 1).rod_labels == ["0 1", "1 0"]
    assert workflow.rod_parity_subset(ds.select_rods(["0 0"]), 1) is None
    with pytest.raises(ValueError, match="odd"):
        workflow.step_surface(Model.from_settings(), ds.select_rods(["0 0"]))


def test_guided_steps_recover_film_and_surface():
    data = make_dataset(truth_model(), rods=EVEN + ODD, noise_rel=0.03, sys_floor=0.02, seed=81)
    model = start_model(["x_OH"], START)          # the steps choose their own free parameters
    before = model.params.free_names
    _, r1 = workflow.step_film(model, data, de=dict(maxiter=25, popsize=10, seed=1))
    assert model.params.free_names == before                     # free set restored after a step
    assert_recovered(r1, FILM_TRUTH)
    _, r2 = workflow.step_surface(model, data, de=dict(maxiter=20, popsize=10, seed=2))
    assert set(r2.names) == {"theta", "x_OH", "z_OH", "scale"} and r2.fom["N"] < len(data)
    _, r3 = workflow.step_all(model, data)
    assert set(r3.names) == set(FILM_TRUTH) | set(SURF_TRUTH) | {"scale"} and r3.fom["N"] == len(data)
    assert_recovered(r3, dict(FILM_TRUTH, **SURF_TRUTH))


def test_steps_with_several_datasets_use_per_dataset_copies():
    data = [make_dataset(truth_model(x_OH=x), rods=EVEN[:1] + ODD[:2], noise_rel=0.03, seed=90 + i, step=0.1,
                         name=f"d{i}", potential_V=v) for i, (x, v) in enumerate([(0.3, 0.5), (0.7, 1.0)])]
    model = start_model([], START)
    model.params.update(dict(FILM_TRUTH, theta=0.85, z_OH=2.05))
    workflow.prepare(model, data, per_dataset=["x_OH"])
    _, res = workflow.step_surface(model, data, de=dict(maxiter=15, popsize=8, seed=3))
    assert {"ds1.x_OH", "ds2.x_OH", "ds1.scale", "ds2.scale"} <= set(res.names) and "x_OH" not in res.names
    assert abs(res.values["ds1.x_OH"] - 0.3) < 3 * res.errors["ds1.x_OH"]
    assert abs(res.values["ds2.x_OH"] - 0.7) < 3 * res.errors["ds2.x_OH"]


def test_cancel_keeps_best_values():
    data = make_dataset(truth_model(), rods=ODD[:2], noise_rel=0.03, seed=5)
    model = start_model(["z_OH", "x_OH", "scale"], START)
    calls = []

    def hook(fit):
        def progress(stage, step, fom):
            calls.append(fom)
            if len(calls) == 3:
                fit.cancel()
        fit.progress = progress
    fit, res = workflow.step_surface(model, data, de=dict(maxiter=500, popsize=10, seed=1), fit_hook=hook)
    assert res.stages[0]["message"] == "stopped by user" and len(res.stages) == 1   # no refine after a stop
    assert res.stages[0]["nit"] <= 4 and np.isfinite(res.fom["chi2"])


def test_project_round_trip(tmp_path):
    model = Model.from_settings(water_layer=True)
    model.params.link("z_H2O", "z_OH + 0.5")
    data = make_dataset(model, rods=ODD[:2], step=0.3, seed=2, name="a", potential_V=1.1)
    model.params.fit_only(["z_OH"])
    fit, res = workflow.step_all(model, data, names=["z_OH"])
    proj = Project(model, [data], [res.to_dict()], name="test", notes="synthetic")
    proj.save(tmp_path / "p.ctrproj.json")
    back = Project.load(tmp_path / "p.ctrproj.json")
    assert back.name == "test" and back.notes == "synthetic"
    assert back.model.params.values() == model.params.values() and back.model.params["z_H2O"].expr == "z_OH + 0.5"
    assert back.model.options == model.options
    d = back.datasets[0]
    np.testing.assert_array_equal(d.F, data.F)
    assert d.meta["potential_V"] == 1.1 and isinstance(d, Dataset)
    assert back.results[0]["values"] == res.values
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(ValueError):
        Project.load(tmp_path / "bad.json")

"""Synthetic recovery: simulate with known parameters plus noise, fit from perturbed starts, and
check each parameter comes back within its reported 2 sigma (fixed seeds, small DE settings)."""
import numpy as np
import pytest

from ctr_params import DEFAULTS
from ctrfit import Fit, Model, make_dataset

ODD = [(0, 1), (1, 0), (0, 3), (0, 5)]
EVEN = [(0, 0), (1, 1), (0, 2), (2, 0)]
SURF_TRUTH = dict(z_OH=2.05, x_OH=0.6, theta=0.85)
FILM_TRUTH = dict(thickness_mean_tl=19.4, eps_perp=2.3, roughness_tl=0.6)
FIXED = dict(z_H2O=2.6)                                  # DFT-like constraint on the H2O height
BOUNDS = dict(thickness_mean_tl=(15, 25), eps_perp=(0.5, 4.0), roughness_tl=(0.05, 2.0), z_OH=(1.6, 2.5),
              scale=(0.5, 2.0))


def truth_model(**extra):
    m = Model.from_settings(DEFAULTS)
    m.params.update({**FIXED, **SURF_TRUTH, **FILM_TRUTH, **extra})
    return m


def start_model(free, start):
    m = Model.from_settings(DEFAULTS)
    m.params.update(FIXED)
    m.params.fit_only(free)
    m.params.update(start)
    for n, (lo, hi) in BOUNDS.items():
        m.params[n].min, m.params[n].max = lo, hi
    return m


def assert_recovered(res, truth, nsig=2.0):
    lines = [f"{n}: true {v:.5g}, fitted {res.values[n]:.5g} +- {res.errors[n]:.2g}" for n, v in truth.items()]
    print("\n".join(lines))
    for n, v in truth.items():
        assert np.isfinite(res.errors[n]) and res.errors[n] > 0, n
        assert abs(res.values[n] - v) <= nsig * res.errors[n], lines


def assert_jointly_recovered(res, truth, prob=0.99):
    """Joint test: Mahalanobis distance of (fitted - true) under the fit covariance below the chi2
    quantile, and every parameter within 3 sigma. With many parameters, a 2 sigma cut on each one
    alone fails by chance in a sizeable share of noise realizations."""
    from scipy.stats import chi2
    idx = [res.names.index(n) for n in truth]
    d = np.array([res.values[n] - truth[n] for n in truth])
    C = res.cov[np.ix_(idx, idx)]
    m2 = float(d @ np.linalg.solve(C, d))
    print(f"Mahalanobis^2 = {m2:.2f} (limit {chi2.ppf(prob, len(d)):.2f})")
    assert m2 < chi2.ppf(prob, len(d))
    assert_recovered(res, truth, nsig=3.0)


def film_fit(fit):
    fit.run("de", maxiter=25, popsize=10, seed=3)
    fit.profile("thickness_mean_tl", np.arange(15.0, 25.01, 0.5))
    fit.refine()


def test_a_surface_only_on_odd_rods():
    data = make_dataset(truth_model(), rods=ODD, noise_rel=0.03, seed=11)
    model = start_model(list(SURF_TRUTH) + ["scale"], dict(z_OH=1.85, x_OH=0.35, theta=0.95, scale=1.1))
    model.params.update(dict(FILM_TRUTH))                # film known from step 1 of the workflow
    fit = Fit(model, data)
    fit.run("de", maxiter=25, popsize=10, seed=2)
    fit.refine()
    res = fit.report()
    assert_recovered(res, SURF_TRUTH)
    assert res.fom["red_chi2"] < 1.3
    assert [h["stage"] for h in res.history].count("de") == 25


def test_b_film_only_on_even_rods():
    data = make_dataset(truth_model(), rods=EVEN, noise_rel=0.03, seed=12, step=0.04)
    model = start_model(list(FILM_TRUTH) + ["scale"], dict(thickness_mean_tl=18.3, eps_perp=1.6, roughness_tl=0.9,
                                                           scale=1.15))
    fit = Fit(model, data)
    film_fit(fit)
    res = fit.report()
    assert_recovered(res, FILM_TRUTH)
    assert res.fom["red_chi2"] < 1.3


def test_c_combined_guided_workflow():
    data = make_dataset(truth_model(), rods=EVEN + ODD, noise_rel=0.03, sys_floor=0.02, seed=13)
    even, odd = data.select_rods(EVEN), data.select_rods(ODD)
    free = list(FILM_TRUTH) + list(SURF_TRUTH) + ["scale"]
    model = start_model(free, dict(thickness_mean_tl=18.4, eps_perp=1.7, roughness_tl=0.9, z_OH=1.85, x_OH=0.35,
                                   theta=0.95, scale=1.2))
    model.params.fit_only(list(FILM_TRUTH) + ["scale"])
    film_fit(Fit(model, even))
    model.params.fit_only(list(SURF_TRUTH) + ["scale"])
    Fit(model, odd).run("de", maxiter=20, popsize=10, seed=4)
    model.params.fit_only(free)
    fit = Fit(model, data)
    fit.refine()
    res = fit.report()
    assert_recovered(res, dict(FILM_TRUTH, **SURF_TRUTH))
    assert 0.7 < res.fom["red_chi2"] < 1.3                    # the floor is in sigma_eff


@pytest.mark.slow
def test_c_combined_direct_de():
    """All free parameters at once, no guided steps: DE, thickness profile, refinement."""
    data = make_dataset(truth_model(), rods=EVEN + ODD, noise_rel=0.03, seed=14)
    free = list(FILM_TRUTH) + list(SURF_TRUTH) + ["scale"]
    model = start_model(free, dict(thickness_mean_tl=18.4, eps_perp=1.7, roughness_tl=0.9, z_OH=1.85, x_OH=0.35,
                                   theta=0.95, scale=1.2))
    fit = Fit(model, data)
    fit.run("de", maxiter=60, popsize=12, seed=5)
    fit.profile("thickness_mean_tl", np.arange(15.0, 25.01, 0.5))
    fit.refine()
    assert_jointly_recovered(fit.report(), dict(FILM_TRUTH, **SURF_TRUTH))


def test_d_degenerate_oh_h2o_mixture_warns():
    """OH and H2O with free heights and B factors on few points: the ratio is not determined, and
    the report must say so rather than give a confident x_OH."""
    truth = truth_model(x_OH=0.5, z_OH=2.0, z_H2O=2.25, B_OH=1.5, B_H2O=2.5)
    pts_rods = [(0, 1), (1, 0)]
    data = make_dataset(truth, rods=pts_rods, noise_rel=0.04, seed=21, L_min=0.5, L_max=3.5, step=0.25)
    free = ["x_OH", "z_OH", "z_H2O", "B_OH", "B_H2O", "theta", "scale"]
    model = start_model(free, dict(x_OH=0.3, z_OH=1.9, z_H2O=2.4, B_OH=3.0, B_H2O=3.0, theta=0.9, scale=1.05))
    model.params.update(FILM_TRUTH)
    model.params["z_H2O"].min, model.params["z_H2O"].max = 1.6, 3.0
    model.params["z_OH"].min, model.params["z_OH"].max = 1.6, 3.0
    for n in ("B_OH", "B_H2O"):
        model.params[n].max = 10.0
    fit = Fit(model, data)
    fit.run("de", maxiter=30, popsize=10, seed=6)
    fit.refine()
    res = fit.report()
    print(res.summary)
    assert res.warnings
    about_ratio = [w for w in res.warnings if "x_OH" in w or "ill-conditioned" in w]
    assert about_ratio or res.errors["x_OH"] > 0.15, res.summary
    assert any("correlated" in w or "relative error" in w or "ill-conditioned" in w for w in res.warnings)


# ---------------------------------------------------------------------------- every parameter
_SWEEP_SETTINGS = dict(DEFAULTS, relax_001=0.5, thickness_nm=6.0, coherent_nm=3.0,
                       comp_default="OH=0.3, H2O=0.3, O=0.2", extra_layers="O cus 3.6 0.5 5.0")
_SWEEP_SKIP = {"coherent_tl"}          # whole trilayers by construction (rounded); kept fixed in fits


@pytest.mark.parametrize("name", [n for n in Model.from_settings(_SWEEP_SETTINGS, water_layer=True).params.names
                                  if n not in _SWEEP_SKIP])
def test_every_parameter_is_recovered(name):
    """Each fit parameter alone: data simulated with the parameter moved, least squares from the
    default value, recovered within 3 sigma (single-parameter fits; 3 sigma keeps the sweep of
    ~35 parameters from failing by chance)."""
    truth = Model.from_settings(_SWEEP_SETTINGS, water_layer=True)
    p = truth.params[name]
    start = p.value
    span = min(p.max - p.min, 4.0)
    step = {"thickness_mean_tl": 0.4, "scale": 0.1}.get(name, 0.04 * span)
    p.value = start + step if start + step <= p.max else start - step
    data = make_dataset(truth, rods=[(0, 0), (0, 1), (1, 0), (1, 1), (0, 3)], noise_rel=0.03, seed=31, step=0.1)
    model = Model.from_settings(_SWEEP_SETTINGS, water_layer=True)
    model.params.fit_only([name])
    fit = Fit(model, data)
    fit.refine()
    res = fit.report()
    assert abs(res.values[name] - p.value) <= 3 * res.errors[name], (name, p.value, res.values[name], res.errors[name])


def test_rod_scales_recovered_with_penalty():
    truth = truth_model()
    data = make_dataset(truth, rods=ODD, noise_rel=0.03, rod_scale_err=0.05, seed=41)
    model = truth_model()
    model.add_rod_scales(ODD, fit=True)
    model.params.fit_only([f"rodscale_{h}_{k}" for h, k in ODD])
    fit = Fit(model, data, rod_scale_penalty=0.1)
    fit.refine()
    res = fit.report()
    for (h, k) in ODD:
        n, true = f"rodscale_{h}_{k}", data.truth["rod_scale"][f"{h} {k}"]
        assert abs(res.values[n] - true) <= 2 * res.errors[n], (n, true, res.values[n], res.errors[n])


def test_two_datasets_get_their_own_scales():
    truth = truth_model()
    d1 = make_dataset(truth, rods=ODD[:2], noise_rel=0.03, seed=51, name="a")
    truth.params["scale"].value = 1.3
    d2 = make_dataset(truth, rods=ODD[:2], noise_rel=0.03, seed=52, name="b")
    model = truth_model()
    model.params.fit_only(["scale", "z_OH"])
    fit = Fit(model, [d1, d2])
    assert set(fit.names) == {"z_OH", "ds1.scale", "ds2.scale"}
    fit.refine()
    res = fit.report()
    assert abs(res.values["ds1.scale"] - 1.0) < 2 * res.errors["ds1.scale"]
    assert abs(res.values["ds2.scale"] - 1.3) < 2 * res.errors["ds2.scale"]
    assert set(res.fom_per_dataset) == {"a", "b"}


def test_result_json_round_trip(tmp_path):
    data = make_dataset(truth_model(), rods=ODD[:2], noise_rel=0.03, seed=61)
    model = truth_model()
    model.params.fit_only(["z_OH", "theta"])
    model.params.link("z_H2O", "z_OH + 0.55")
    fit = Fit(model, data)
    fit.refine()
    res = fit.report()
    assert "z_H2O" in res.errors and res.errors["z_H2O"] == pytest.approx(res.errors["z_OH"], rel=1e-3)
    res.save(tmp_path / "r.json")
    from ctrfit import FitResult
    back = FitResult.load(tmp_path / "r.json")
    assert back.values == res.values and back.errors == res.errors and back.warnings == res.warnings
    np.testing.assert_allclose(back.correlation, res.correlation)
    assert "reduced chi2" in back.summary
    assert Model.from_dict(back.model).params.values() == model.params.values()

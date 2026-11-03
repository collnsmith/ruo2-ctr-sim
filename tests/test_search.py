"""Less starting guess: scales solved exactly in the global search, least squares from several
distinct points of the search, and the neighbouring thickness fringes tried by refine (fixed seeds)."""
import numpy as np
import pytest

from ctr_params import DEFAULTS
from ctrfit import Fit, Model, make_dataset
from ctrfit.fit import workflow

from test_fit import EVEN, FIXED, ODD, SURF_TRUTH, FILM_TRUTH

WIDE = dict(thickness_mean_tl=(5, 40), eps_perp=(-1, 5), roughness_tl=(0.05, 2.0), z_OH=(1.6, 2.5),
            x_OH=(0, 1), theta=(0, 1), scale=(0.05, 20))
TRUTH = {**SURF_TRUTH, **FILM_TRUTH, "scale": 2.7}


def _truth():
    m = Model.from_settings(DEFAULTS)
    m.params.update({**FIXED, **TRUTH})
    return m


def _start(free, start, step_bounds=WIDE):
    m = Model.from_settings(DEFAULTS)
    m.params.update(FIXED)
    m.params.fit_only(free)
    m.params.update(start)
    for n in free:
        m.params[n].min, m.params[n].max = step_bounds[n]
    return m


def test_best_factor_by_figure_of_merit():
    Fc = np.array([1.0, 2.0, 4.0, 8.0])
    s = np.array([0.1, 0.2, 0.2, 0.4])
    assert Fit.best_factor(3 * Fc, Fc, s, "chi2") == pytest.approx(3.0)
    F = np.array([2.0, 4.0, 40.0, 16.0])                   # ratios 2, 2, 10, 2: L1 measures take the median
    assert Fit.best_factor(F, Fc, s, "log") == pytest.approx(2.0)
    assert Fit.best_factor(F, Fc, s, "R1") == pytest.approx(2.0)
    w = 1 / s ** 2
    assert Fit.best_factor(F, Fc, s, "chi2") == pytest.approx(np.sum(w * F * Fc) / np.sum(w * Fc ** 2))
    assert Fit.best_factor(F, np.zeros(4), s) == 1.0      # no model signal: unchanged


def test_scales_are_solved_not_searched():
    data = make_dataset(_truth(), rods=EVEN[:2], noise_rel=0.03, seed=21, step=0.1)
    m = _start(["thickness_mean_tl", "scale"], dict(thickness_mean_tl=19.4, scale=0.2))
    fit = Fit(m, data)
    assert fit.scale_groups() == {"scale": [0]}
    st = fit.run("de", maxiter=3, popsize=6, seed=1)
    assert st["solved_scales"] == ["scale"]
    Fc = fit.model_F()[0]
    assert Fit.best_factor(data.F, Fc, data.sigma_eff) == pytest.approx(1.0, rel=1e-9)   # at its best value
    fit2 = Fit(_start(["thickness_mean_tl", "scale"], dict(thickness_mean_tl=19.4, scale=0.2)), data)
    st2 = fit2.run("de", maxiter=3, popsize=6, seed=1, solve_scales=False)
    assert st2["solved_scales"] == [] and st["nfev"] < st2["nfev"]           # one dimension less to search
    # only scales free: solved directly
    m3 = _start(["scale"], {**TRUTH, "scale": 0.2})
    Fit(m3, data).run("de")
    assert m3.params["scale"].value == pytest.approx(2.7, rel=0.02)


def test_several_datasets_each_scale_solved_and_linked_scales_left_alone():
    t = _truth()
    a = make_dataset(t, rods=ODD[:1], noise_rel=0.03, seed=31, step=0.2, name="a")
    b = make_dataset(t, rods=ODD[:1], noise_rel=0.03, seed=32, step=0.2, name="b")
    b.F *= 2.0
    m = _start(["z_OH", "scale"], dict(z_OH=2.3, scale=1.0))
    fit = Fit(m, [a, b])
    m.params["ds1.scale"].fit = m.params["ds2.scale"].fit = True
    assert fit.scale_groups() == {"ds1.scale": [0], "ds2.scale": [1]}
    fit.run("de", maxiter=6, popsize=6, seed=2)
    v = m.params.values()
    assert v["ds2.scale"] / v["ds1.scale"] == pytest.approx(2.0, rel=0.03)
    m.params.link("ds2.scale", "2 * ds1.scale")             # a link refers to ds1.scale: it stays in the search
    assert fit.scale_groups(fit.names) == {}


def test_refine_tries_the_neighbouring_fringes():
    """Started in the neighbouring thickness fringe (thickness one trilayer off, eps_perp traded), least
    squares stays there; with hops it finds the right one."""
    data = make_dataset(_truth(), rods=EVEN[:2] + ODD[:1], noise_rel=0.03, seed=41, step=0.05)
    free = ["thickness_mean_tl", "eps_perp", "roughness_tl", "scale"]
    wrong = {**TRUTH, "thickness_mean_tl": 20.6, "eps_perp": 2.2}     # the surface as in the truth
    stuck = Fit(_start(free, wrong), data)
    stuck.refine()
    assert abs(stuck.model.params["thickness_mean_tl"].value - 20.6) < 0.3
    fit = Fit(_start(free, wrong), data)
    st = fit.refine(hops=1)
    v = fit.model.params.values()
    assert abs(v["thickness_mean_tl"] - 19.4) < 0.2 and abs(v["eps_perp"] - 2.3) < 0.05
    assert st["starts"] == 3 and "thickness_mean_tl -1 TL" in st["method"] and st["fom"] < stuck.fom()
    res = fit.report()                                       # errors from the chosen start's Jacobian
    assert np.isfinite(res.errors["thickness_mean_tl"]) and res.errors["thickness_mean_tl"] > 0


def test_refine_from_several_points_of_the_search():
    data = make_dataset(_truth(), rods=ODD[:2], noise_rel=0.03, seed=51, step=0.1)
    m = _start(["z_OH", "x_OH", "scale"], dict(z_OH=2.4, x_OH=0.1, scale=0.2))
    fit = Fit(m, data)
    fit.run("de", maxiter=8, popsize=8, seed=3)
    c = fit.candidates(4)
    assert 2 <= len(c) <= 4 and set(c[0]) == {"z_OH", "x_OH", "scale"}
    st = fit.refine(starts=3)
    assert st["method"].startswith("best of ") and st["fom"] <= min(s["fom"] for s in fit.stages[1:-1]) + 1e-9
    assert abs(m.params["z_OH"].value - 2.05) < 0.05
    # history keeps counting least-squares evaluations across the starts (one curve)
    steps = [h["step"] for h in fit.history if h["stage"] == "lsq"]
    assert steps == sorted(steps)


def test_guided_film_step_profile_follows_the_search():
    """Step 1 from a thickness far from the truth: the profile window is centred on what the search found,
    not on the starting value, and the film comes back."""
    data = make_dataset(_truth(), rods=EVEN[:2], noise_rel=0.03, seed=61, step=0.05)
    m = _start(["thickness_mean_tl", "eps_perp", "roughness_tl", "scale"],
               dict(thickness_mean_tl=32.0, eps_perp=0.0, roughness_tl=1.5, scale=0.2),
               dict(WIDE, thickness_mean_tl=(10, 35)))
    fit, res = workflow.step_film(m, data, de=dict(maxiter=20, popsize=8, seed=1))
    assert abs(res.values["thickness_mean_tl"] - 19.4) < 0.3 and abs(res.values["eps_perp"] - 2.3) < 0.1


@pytest.mark.slow
def test_wide_bounds_bad_start_every_seed():
    """Film and surface together over wide bounds from a poor start (scale 0.2 vs 2.7, thickness 30 vs
    19.4 TL): DE with the scales solved, then refine from 3 starts with fringe hops, finds the truth for
    every seed (DE alone ends in the neighbouring fringe for some)."""
    data = make_dataset(_truth(), rods=ODD + EVEN, noise_rel=0.03, seed=11, step=0.05)
    start = dict(thickness_mean_tl=30, eps_perp=0.0, roughness_tl=1.5, z_OH=2.4, x_OH=0.1, theta=0.3, scale=0.2)
    for seed in (1, 4):
        m = _start(list(WIDE), start)
        fit = Fit(m, data)
        fit.run("de", maxiter=30, popsize=10, seed=seed)
        fit.refine(starts=3, hops=1)
        v = m.params.values()
        assert abs(v["thickness_mean_tl"] - 19.4) < 0.2 and abs(v["z_OH"] - 2.05) < 0.05, (seed, v)

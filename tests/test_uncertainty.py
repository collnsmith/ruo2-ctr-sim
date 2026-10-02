"""Phase 5: bootstrap, MCMC, information criteria, and interval coverage over noise realizations."""
import math

import numpy as np
import pytest

from ctrfit import Fit, make_dataset
from ctrfit.fit import bootstrap, compare_models, information_criteria, mcmc
from test_fit import FILM_TRUTH, ODD, SURF_TRUTH, start_model, truth_model

SURF_FREE = list(SURF_TRUTH) + ["scale"]


def surface_fit(seed, noise=0.03, truth_over=None, free=SURF_FREE, start=None):
    data = make_dataset(truth_model(**(truth_over or {})), rods=ODD, noise_rel=noise, seed=seed)
    model = start_model(free, start or dict(z_OH=2.0, x_OH=0.55, theta=0.9, scale=1.02))
    model.params.update(dict(FILM_TRUTH))
    fit = Fit(model, data)
    fit.refine()
    return fit, fit.report()


def test_information_criteria_by_hand():
    ic = information_criteria(100.0, 50, 3)
    assert ic["aic"] == pytest.approx(106.0)
    assert ic["aicc"] == pytest.approx(106.0 + 24.0 / 46.0)
    assert ic["bic"] == pytest.approx(100.0 + 3 * math.log(50))


def test_report_contains_aic_bic():
    _, res = surface_fit(1)
    assert res.fom["aic"] == pytest.approx(res.fom["chi2"] + 2 * res.fom["p"])
    assert "BIC" in res.summary


def test_bootstrap_agrees_with_covariance():
    fit, res = surface_fit(2)
    b = bootstrap(fit, n=40, seed=3)
    assert b.info["failed"] == 0 and b.samples.shape == (40, 4)
    for n in SURF_TRUTH:
        assert 0.5 < b.std[n] / res.errors[n] < 2.0, (n, b.std[n], res.errors[n])
    assert fit.model.params.get_vector(fit.names) == pytest.approx([res.values[n] for n in fit.names])


def test_mcmc_agrees_with_covariance():
    pytest.importorskip("emcee")
    fit, res = surface_fit(4)
    s = mcmc(fit, nsteps=250, burn=100, seed=5)
    assert 0.2 < s.info["acceptance"] < 0.8
    for n in SURF_TRUTH:
        assert 0.6 < s.std[n] / res.errors[n] < 1.6, (n, s.std[n], res.errors[n])
        assert abs(s.mean[n] - res.values[n]) < res.errors[n]
    lo, hi = s.interval("z_OH")
    assert lo < res.values["z_OH"] < hi
    again = mcmc(fit, nsteps=250, burn=100, seed=5)                 # fixed seed: same chain
    np.testing.assert_array_equal(again.samples, s.samples)


def test_mcmc_missing_emcee_message(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "emcee":
            raise ImportError("no emcee")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    fit, _ = surface_fit(1)
    with pytest.raises(ImportError, match="pip install emcee"):
        mcmc(fit, nsteps=10, burn=0)


def test_aic_bic_pick_the_right_surface_model():
    """Mixture data: the OH + H2O model beats OH only. OH-only data: the extra share is not favoured."""
    for truth_x, mixture_should_win in ((0.5, True), (1.0, False)):
        data = make_dataset(truth_model(x_OH=truth_x), rods=ODD, noise_rel=0.03, seed=71)
        results = []
        for free, x0 in ((["z_OH", "theta", "scale"], 1.0), (["z_OH", "theta", "x_OH", "scale"], 0.8)):
            model = start_model(free, dict(z_OH=2.0, theta=0.9, x_OH=x0, scale=1.0))
            model.params.update(dict(FILM_TRUTH))
            fit = Fit(model, data)
            fit.run("de", maxiter=15, popsize=8, seed=1)
            fit.refine()
            results.append(fit.report())
        rows, text = compare_models(results, ["OH only", "OH + H2O"])
        print(text)
        d = {r["label"]: r for r in rows}
        if mixture_should_win:
            assert d["OH only"]["d_bic"] > 6
        else:
            assert d["OH + H2O"]["d_bic"] > -2 and d["OH only"]["d_bic"] < 2


@pytest.mark.slow
def test_interval_coverage_over_20_realizations():
    """1-sigma intervals (covariance and MCMC) contain the true z_OH and x_OH in about 68 % of 20
    noise realizations (wide tolerance: 9 to 19 of 20)."""
    pytest.importorskip("emcee")
    hits = {(k, n): 0 for k in ("cov", "mcmc") for n in ("z_OH", "x_OH")}
    rows = []
    for seed in range(20):
        fit, res = surface_fit(100 + seed)
        s = mcmc(fit, nsteps=220, burn=80, seed=seed)
        for n in ("z_OH", "x_OH"):
            t = SURF_TRUTH[n]
            hits[("cov", n)] += abs(res.values[n] - t) <= res.errors[n]
            lo, hi = s.interval(n)
            hits[("mcmc", n)] += lo <= t <= hi
        rows.append((seed, res.values["z_OH"], res.errors["z_OH"], res.values["x_OH"], res.errors["x_OH"]))
    for r in rows:
        print("seed %3d  z_OH %.4f +- %.4f  x_OH %.4f +- %.4f" % r)
    print({f"{k}/{n}": f"{v}/20" for (k, n), v in hits.items()})
    for key, v in hits.items():
        assert 9 <= v <= 19, (key, v)

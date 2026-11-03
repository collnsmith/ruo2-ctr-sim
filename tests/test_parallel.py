"""Model evaluations on several cores: worker results equal this process's, a parallel fit ends where
one core ends, is reproducible for a seed, and stops when cancelled."""
import numpy as np
import pytest

from ctrfit import Fit, make_dataset
from ctrfit.fit import parallel
from ctrfit.fit.parallel import ParallelFit

from test_search import EVEN, ODD, TRUTH, WIDE, _start, _truth

CORES = 2


@pytest.fixture(scope="module")
def data():
    return make_dataset(_truth(), rods=ODD[:2] + EVEN[:1], noise_rel=0.03, seed=71, step=0.1)


def test_workers_equal_this_process(data):
    m = _start(["thickness_mean_tl", "eps_perp", "z_OH", "x_OH", "scale"], dict(thickness_mean_tl=19.0))
    fit = Fit(m, data)
    names = [n for n in fit.names if n != "scale"]
    groups = fit.scale_groups()
    lo, hi = m.params.bounds(names)
    X = lo + np.random.default_rng(3).random((7, len(names))) * (hi - lo)
    par = ParallelFit(fit, CORES)
    assert par.state and "xraydb" in __import__("pickle").loads(par.state)["blocked"]   # same anomalous terms
    got = par.map("de", X, (names, "chi2", groups))
    for x, g in zip(X, got):
        f, sc = fit._solved_fom(x, names, "chi2", groups)
        assert g[0] == pytest.approx(f, rel=1e-12) and g[1]["scale"] == pytest.approx(sc["scale"], rel=1e-12)
    r = par.map("resid", X[:3], names)
    for x, g in zip(X[:3], r):
        np.testing.assert_allclose(g, fit.residuals(x, names), rtol=1e-12)


def test_least_squares_on_several_cores_equals_one_core(data):
    """Same start: the Jacobian columns computed on the workers give the same refinement and errors."""
    free = ["thickness_mean_tl", "eps_perp", "z_OH", "x_OH", "scale"]
    start = {**TRUTH, "thickness_mean_tl": 19.8, "eps_perp": 2.1, "z_OH": 2.2, "x_OH": 0.5, "scale": 2.5}
    out = []
    for cores in (1, CORES):
        fit = Fit(_start(free, start), data, cores=cores)
        fit.refine()
        out.append(fit.report())
    a, b = out
    assert b.fom["chi2"] == pytest.approx(a.fom["chi2"], rel=1e-7)
    for n in free:
        assert b.values[n] == pytest.approx(a.values[n], rel=1e-5, abs=1e-6)
        assert b.errors[n] == pytest.approx(a.errors[n], rel=1e-3)


def test_parallel_search_is_reproducible_and_finds_the_truth(data):
    free = ["thickness_mean_tl", "eps_perp", "z_OH", "x_OH", "scale"]
    start = {**TRUTH, "thickness_mean_tl": 23.0, "eps_perp": 1.0, "z_OH": 2.4, "x_OH": 0.2, "scale": 0.3}
    out = []
    for _ in range(2):
        m = _start(free, start, dict(WIDE, thickness_mean_tl=(15, 25)))
        fit = Fit(m, data, cores=CORES)
        st = fit.run("de", maxiter=15, popsize=8, seed=4)
        assert st["cores"] == CORES and st["nfev"] > 0
        fit.refine(starts=3, hops=1)
        out.append((fit.fom(), m.params.values()))
    assert out[0] == out[1]                                                   # fixed seed: the same path
    v = out[0][1]
    assert abs(v["thickness_mean_tl"] - 19.4) < 0.2 and abs(v["z_OH"] - 2.05) < 0.05 and abs(v["x_OH"] - 0.6) < 0.05


def test_parallel_search_stops_when_cancelled(data):
    m = _start(["thickness_mean_tl", "z_OH", "scale"], {**TRUTH, "thickness_mean_tl": 21.0})
    fit = Fit(m, data, cores=CORES)
    fit.progress = lambda stage, step, fom: fit.cancel() if step >= 2 else None
    st = fit.run("de", maxiter=500, popsize=8, seed=1)
    assert st["message"] == "stopped by user" and st["nit"] <= 3
    assert np.isfinite(fit.fom())


def test_pool_is_kept_and_replaced_for_another_size():
    p = parallel.get_pool(CORES)
    assert parallel.get_pool(CORES) is p
    assert 1 <= parallel.default_cores() <= parallel.available_cores()

"""In situ cell and the parameters added for it: absorption in the electrolyte film and the window,
per-rod L offsets, adsorbed (bi)sulfate. All off by default (the model unchanged), fast = reference,
and each recovered from synthetic data."""
import numpy as np
import pytest

from ctr_params import DEFAULTS, parse_comp
from ctrfit import Fit, Model, make_dataset
from ctrfit.core import absorption as A
from ctrfit.model.templates import composition, shares, sulfate_share

RODS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _points(rods, L):
    H = np.concatenate([np.full(L.size, h) for h, _ in rods])
    K = np.concatenate([np.full(L.size, k) for _, k in rods])
    return H, K, np.tile(L, len(rods))


def test_off_by_default_and_the_model_is_unchanged():
    m = Model.from_settings(DEFAULTS)
    P = m.params
    assert P["t_liquid"].value == 0 and P["t_window"].value == 0 and P["x_SO4"].value == 0
    assert not any(P[n].fit for n in ("t_liquid", "t_window", "x_SO4", "z_SO4", "B_SO4", "dz_Rucus_SO4"))
    assert not [n for n in P.names if n.startswith("dL_")]
    _, _, comp = m.template.engine_inputs(m.settings, P.view())
    assert "SO4" not in comp


def test_absorption_factor():
    lat = (2.9587, 4.5937 * np.sqrt(2), 4.5937 * np.sqrt(2))
    L = np.array([0.2, 0.5, 1.0, 3.0])
    a, b, _ = A.surface_angles(np.zeros(4), np.ones(4), L, 16.0, lat, 0.5)
    assert np.allclose(a, 0.5) and np.all(np.diff(b) > 0)                     # exit angle grows with L
    path = A.path_factor(np.zeros(4), np.ones(4), L, 16.0, lat, 0.5)
    assert path == pytest.approx(1 / np.sin(np.radians(0.5)) + 1 / np.sin(np.radians(b)))
    T = A.transmission_F(path, 16.0, 10.0, 7.6)
    mu_l, mu_w = A.mu(A.MU_ELECTROLYTE, 16.0), A.mu(A.MU_KAPTON, 16.0)
    assert T == pytest.approx(np.exp(-0.5 * (mu_l * 10e-4 + mu_w * 7.6e-4) * path))
    assert np.all(np.diff(T) > 0) and A.transmission_F(path, 16.0, 0.0, 0.0) == 1.0
    assert 1.4 < mu_l < 1.5 and 1.2 < mu_w < 1.35                              # 1/cm at 16 keV (xraydb)
    # the specular rod: alpha = beta
    a, b, _ = A.surface_angles(np.zeros(2), np.zeros(2), np.array([1.0, 2.0]), 16.0, lat, 0.5)
    assert np.allclose(a, b)


def test_composition_with_sulfate():
    c = composition(0.8, 0.5, 0.25, 0.2)
    assert c["SO4"] == pytest.approx(0.16) and sum(c.values()) == pytest.approx(0.8)
    assert c["O"] == pytest.approx(0.8 * 0.8 * 0.25)
    assert shares(c) == pytest.approx((0.8, 0.5, 0.25)) and sulfate_share(c) == pytest.approx(0.2)
    assert parse_comp("OH=0.3, SO4=0.2") == {"OH": 0.3, "SO4": 0.2}
    m = Model.from_settings(dict(DEFAULTS, comp_default="OH=0.4, H2O=0.4, SO4=0.2"))
    assert m.params["x_SO4"].value == pytest.approx(0.2) and m.params["theta"].value == pytest.approx(1.0)


@pytest.mark.parametrize("over", [{}, dict(mix_mode="domains")])
def test_fast_equals_reference_with_the_new_parameters(over):
    m = Model.from_settings(dict(DEFAULTS, **over))
    m.params.update(dict(t_liquid=25.0, t_window=7.6, x_SO4=0.3, z_SO4=2.2, B_SO4=6.0, dz_Rucus_SO4=0.05))
    m.add_rod_offsets([(0, 1), (1, 1)])
    m.params["dL_0_1"].value, m.params["dL_1_1"].value = 0.012, -0.007
    L = np.linspace(0.25, 3.8, 25)
    ev = m.evaluator(*_points(RODS, L))
    ref = np.concatenate([m.F(h, k, L) for h, k in RODS])
    np.testing.assert_allclose(ev(), ref, rtol=1e-9)
    m.params["dL_0_1"].value = m.params["dL_1_1"].value = 0.0                   # back to the unshifted engine
    np.testing.assert_allclose(ev(), np.concatenate([m.F(h, k, L) for h, k in RODS]), rtol=1e-9)
    assert ev._fast_for(m.params.view()) is ev.fast


def test_absorption_lowers_low_L_most():
    m = Model.from_settings(DEFAULTS)
    L = np.linspace(0.2, 3.0, 15)
    before = m.F(0, 1, L)
    m.params["t_liquid"].value = 20.0
    ratio = m.F(0, 1, L) / before
    assert np.all(ratio < 1) and ratio[0] < ratio[-1] < 1 and np.all(np.diff(ratio) > 0)


def test_electrolyte_thickness_recovered():
    """Data with 30 µm of electrolyte under a 7.6 µm Kapton window: t_liquid comes back from 0 (window
    fixed), with the scale fitted too."""
    truth = Model.from_settings(DEFAULTS)
    truth.params.update(dict(t_liquid=30.0, t_window=7.6))
    data = make_dataset(truth, rods=RODS, noise_rel=0.02, seed=81, step=0.05, L_min=0.15, L_max=3.5)
    model = Model.from_settings(DEFAULTS)
    model.params.update(dict(t_window=7.6))
    model.params.fit_only(["t_liquid", "scale"])
    fit = Fit(model, data)
    fit.refine()
    res = fit.report()
    assert abs(res.values["t_liquid"] - 30.0) <= 3 * res.errors["t_liquid"] and res.errors["t_liquid"] < 10


def test_rod_offset_recovered():
    """A scan whose L is off by 0.015: its dL comes back, the other rods' stay at 0."""
    truth = Model.from_settings(DEFAULTS)
    truth.add_rod_offsets([(0, 1)])
    truth.params["dL_0_1"].value = 0.015
    data = make_dataset(truth, rods=RODS[1:3], noise_rel=0.02, seed=82, step=0.05)
    model = Model.from_settings(DEFAULTS)
    model.add_rod_offsets([(0, 1), (1, 0)], fit=True)
    model.params.fit_only(["dL_0_1", "dL_1_0", "scale"])
    fit = Fit(model, data)
    fit.refine()
    res = fit.report()
    assert abs(res.values["dL_0_1"] - 0.015) <= 3 * res.errors["dL_0_1"] < 0.005
    assert abs(res.values["dL_1_0"]) <= 3 * res.errors["dL_1_0"]


def test_rod_offsets_per_dataset_scope():
    m = Model.from_settings(DEFAULTS)
    m.add_rod_offsets([(0, 1)], scope="ds2")
    p = m.params["ds2.dL_0_1"]
    assert p.value == 0 and not p.fit and p.unit == "r.l.u." and p.group == "instrument"
    assert m.params.view("ds2")["dL_0_1"] == 0 and "dL_0_1" not in m.params.view("ds1")


# ---------------------------------------------------------------------------- L resolution
def test_resolution_off_by_default_and_its_average():
    from ctrfit.model.model import resolution_nodes
    m = Model.from_settings(DEFAULTS)
    assert m.params["res_L"].value == 0 and not m.params["res_L"].fit and m.params["res_L"].group == "instrument"
    x, w = resolution_nodes()
    assert w.sum() == pytest.approx(1) and np.sum(w * x) == pytest.approx(0, abs=1e-12)
    assert np.sum(w * x ** 2) == pytest.approx(1)                      # the nodes are in units of sigma
    L = np.linspace(1.4, 2.6, 49)
    L = L[np.abs(L - 2) > 0.06]
    m7 = Model.from_settings(dict(DEFAULTS, thickness_nm=7.0))
    sharp = m7.F(0, 2, L)
    m7.params["res_L"].value = 0.015
    smooth = m7.F(0, 2, L)
    k = (L > 2.1) & (L < 2.5)
    assert smooth[k].max() / smooth[k].min() < 0.85 * sharp[k].max() / sharp[k].min()   # minima filled


@pytest.mark.parametrize("over", [{}, dict(mix_mode="domains")])
def test_fast_equals_reference_with_resolution(over):
    m = Model.from_settings(dict(DEFAULTS, thickness_nm=7.0, **over))
    m.params.update(dict(res_L=0.012, t_liquid=10.0))
    m.add_rod_offsets([(0, 1)])
    m.params["dL_0_1"].value = 0.01
    L = np.linspace(0.3, 3.7, 31)
    ev = m.evaluator(*_points(RODS, L))
    np.testing.assert_allclose(ev(), np.concatenate([m.F(h, k, L) for h, k in RODS]), rtol=1e-9)
    m.params["res_L"].value = 0.02                                     # a new width: its own spread points
    np.testing.assert_allclose(ev(), np.concatenate([m.F(h, k, L) for h, k in RODS]), rtol=1e-9)
    m.params["res_L"].value = m.params["dL_0_1"].value = 0.0
    assert ev._fast_for(m.params.view()) is ev.fast


def test_resolution_recovered():
    """A 7 nm film's fringes measured with an rms L resolution of 0.015: res_L comes back from 0.006 (a start
    above 0: at 0 its effect, of second order, has no slope)."""
    truth = Model.from_settings(dict(DEFAULTS, thickness_nm=7.0))
    truth.params["res_L"].value = 0.015
    data = make_dataset(truth, rods=[(0, 2), (0, 1)], noise_rel=0.02, seed=83, step=0.01, L_min=1.3, L_max=3.4)
    model = Model.from_settings(dict(DEFAULTS, thickness_nm=7.0))
    model.params["res_L"].value = 0.006
    model.params.fit_only(["res_L", "scale"])
    fit = Fit(model, data)
    fit.refine()
    res = fit.report()
    assert abs(res.values["res_L"] - 0.015) <= 3 * res.errors["res_L"] and res.errors["res_L"] < 0.003

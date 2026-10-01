"""Template rutile110_film: settings -> parameters -> model equals the legacy engine; every parameter
acts on the model; continuous thickness weights."""
from pathlib import Path

import numpy as np
import pytest

from ctr_engine import CTRModel
from ctr_params import DEFAULTS, parse_comp
from ctrfit.core.film import thickness_weights
from ctrfit.model.model import Model
from ctrfit.model.templates import composition, shares

REF = np.load(Path(__file__).resolve().parent / "reference.npz")
RODS = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 5)]


@pytest.mark.parametrize("case, over", [("default", {}), ("relaxed", dict(thickness_nm=10.0, relax_001=0.5))])
def test_settings_to_parameters_reproduce_reference(case, over):
    model = Model.from_settings(dict(DEFAULTS, **over))
    for H, K in RODS:
        np.testing.assert_allclose(model.intensity(H, K, REF["L"]), REF[f"{case}_I_{H}{K}"], rtol=1e-9, atol=0)


@pytest.mark.parametrize("over", [
    dict(eps_perp_mode="elastic"), dict(interface_A=3.1, roughness_nm=0.3, comp_default="OH=0.3, O=0.5"),
    dict(mix_mode="domains", comp_default="H2O=0.4, OH=0.2"), dict(water_on=False, include_H=False),
])
def test_settings_round_trip_other_settings(over):
    p = dict(DEFAULTS, **over)
    model = Model.from_settings(p)
    leg = CTRModel(p)
    comp = parse_comp(p["comp_default"])
    L = np.linspace(0.31, 5.9, 400)
    for H, K in [(0, 0), (0, 1), (1, 1)]:
        np.testing.assert_allclose(model.intensity(H, K, L), leg.intensity(H, K, L, comp), rtol=1e-9, atol=0)


def test_water_layer_option_matches_extra_layer():
    p = dict(DEFAULTS, extra_layers="O cus 3.6 0.7 5.0")
    model = Model.from_settings(p, water_layer=True)
    v = model.params.values()
    assert (v["wl_z"], v["wl_occ"], v["wl_B"]) == (3.6, 0.7, 5.0)
    assert model.settings["extra_layers"] == ""          # taken over by the parameters
    leg = CTRModel(p)
    L = np.linspace(0.31, 5.9, 300)
    np.testing.assert_allclose(model.intensity(0, 1, L), leg.intensity(0, 1, L, parse_comp(p["comp_default"])),
                               rtol=1e-9)


def test_integer_mean_default_spread_equals_legacy_weights():
    for t_nm in (2.0, 6.5, 10.0):
        leg = CTRModel(dict(DEFAULTS, thickness_nm=t_nm))
        new = thickness_weights(float(leg.n_film), leg.film_n_spread)
        old = leg.film_n_weights()
        assert [n for n, _ in new] == [n for n, _ in old]
        np.testing.assert_allclose([w for _, w in new], [w for _, w in old], rtol=1e-14)


def test_thickness_weights_are_continuous_and_centred():
    means = np.linspace(10.0, 12.0, 2001)
    for spread in (0.3, 0.99, 2.0):
        dense = np.zeros((means.size, 40))
        avg = np.zeros(means.size)
        for i, mu in enumerate(means):
            for n, w in thickness_weights(mu, spread):
                dense[i, n] = w
            avg[i] = dense[i] @ np.arange(40)
        assert np.max(np.abs(np.diff(dense, axis=0))) < 0.01      # no jumps as the mean moves
        assert np.all(np.diff(avg) > 0)                             # mean thickness rises with the mean
        if spread >= 0.99:                                          # wide enough: centred on the mean
            np.testing.assert_allclose(avg, means, atol=0.01)
    w0 = dict(thickness_weights(15.0, 0.0))                          # spread floor 0.3 TL
    assert sum(w for n, w in w0.items() if abs(n - 15) > 1) < 1e-9 and w0[14] > 1e-3
    assert thickness_weights(15.0, 0.0) == thickness_weights(15.0, 0.3)


def test_composition_mapping():
    assert composition(1.0, 0.5, 0.0) == {"O": 0.0, "OH": 0.5, "H2O": 0.5}
    c = composition(0.8, 0.25, 0.5)
    assert c == pytest.approx({"O": 0.4, "OH": 0.1, "H2O": 0.3})
    rng = np.random.default_rng(0)
    for theta, x_oh, x_o in rng.uniform(0, 1, (200, 3)):
        c = composition(theta, x_oh, x_o)
        assert min(c.values()) >= 0 and sum(c.values()) == pytest.approx(theta) and sum(c.values()) <= 1 + 1e-12
        assert shares(c) == pytest.approx((theta, x_oh, x_o))
    assert shares({"OH": 0.3, "H2O": 0.6}) == pytest.approx((0.9, 1 / 3, 0.0))


# every user-visible parameter changes the model somewhere (plan rule)
@pytest.mark.parametrize("name", Model.from_settings(DEFAULTS, water_layer=True).params.names)
def test_every_parameter_changes_the_model(name):
    model = Model.from_settings(dict(DEFAULTS, relax_001=0.5, thickness_nm=6.0, coherent_nm=3.0,
                                     comp_default="OH=0.3, H2O=0.3, O=0.2"), water_layer=True)
    p = model.params[name]
    assert p.unit is not None and p.description, name
    model.params["wl_occ"].value = 0.5            # the default layer is empty
    L = np.linspace(0.31, 3.5, 60)
    rods = [(0, 0), (0, 1), (1, 1)]
    before = np.concatenate([model.F(H, K, L) for H, K in rods])
    span = p.max - p.min if np.isfinite(p.max - p.min) else 1.0
    step = {"coherent_tl": -1.0, "thickness_mean_tl": 0.5}.get(name, 0.05 * min(span, 2.0))
    p.value = p.value + step if p.value + step <= p.max else p.value - step
    after = np.concatenate([model.F(H, K, L) for H, K in rods])
    assert np.max(np.abs(after / before - 1)) > 1e-6, name


def test_model_json_round_trip(tmp_path):
    model = Model.from_settings(dict(DEFAULTS, energy_kev=17.0), water_layer=True)
    model.params.link("z_H2O", "z_OH + 0.2")
    model.add_rod_scales([(0, 1), (1, 0)])
    model.save(tmp_path / "m.json")
    back = Model.load(tmp_path / "m.json")
    assert back.settings == model.settings and back.options == model.options
    assert back.params.values() == model.params.values() and back.params["z_H2O"].expr == "z_OH + 0.2"
    L = np.linspace(0.5, 3, 20)
    np.testing.assert_array_equal(back.F(0, 1, L), model.F(0, 1, L))

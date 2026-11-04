"""Where a parameter changes the model curve (the colour of the fitting window's model line)."""
import numpy as np
import pytest
from matplotlib.figure import Figure

from ctrfit.fit.sensitivity import affects, curve_effect, effect_step, target
from ctrfit.model.parameters import Parameter
from ctrfit.project import plots

from test_fit import ODD, truth_model


def _ev(model, rod=(0, 1)):
    L = np.linspace(0.3, 3.5, 120)
    ev = model.evaluator(np.full(L.size, rod[0]), np.full(L.size, rod[1]), L)
    return L, ev


def test_scale_changes_every_point_alike_and_values_are_restored():
    m = truth_model()
    _, ev = _ev(m)
    before = m.params.values()
    s, step, note = curve_effect(m.params, "scale", ev, step=0.01)
    assert np.allclose(s, s[0]) and s[0] == pytest.approx(0.5 * np.log(1.01 / 0.99), rel=1e-6)
    assert m.params.values() == before and "scale ± 0.01" in note


def test_surface_parameter_acts_on_odd_rods_not_even():
    m = truth_model()
    _, odd = _ev(m, (0, 1))
    _, even = _ev(m, (0, 0))
    s_odd = curve_effect(m.params, "x_OH", odd, step=0.05)[0]
    s_even = curve_effect(m.params, "x_OH", even, step=0.05)[0]
    assert s_odd.max() > 5 * s_even.max()                       # the CUS species show on the odd rods


def test_links_follow_and_parameters_that_do_not_act_are_reported():
    m = truth_model()
    P = m.params
    _, ev = _ev(m)
    alone = curve_effect(P, "z_OH", ev, step=0.1)[0]
    P["theta"].value, P["x_OH"].value = 0.9, 0.5
    P.link("z_H2O", "z_OH + 0.5")
    both = curve_effect(P, "z_OH", ev, step=0.1)[0]
    assert not np.allclose(alone, both)                         # the linked H2O height moves too
    assert "linked" in affects(P, "z_H2O")
    P.add(Parameter("ds2.x_OH", 0.4, 0, 1, group="surface"))
    s, _, why = curve_effect(P, "ds2.x_OH", ev, scope="ds1", step=0.05)
    assert s is None and "belongs to ds2" in why
    assert "own copy" in affects(P, "x_OH", "ds2") and affects(P, "x_OH", "ds1") is None
    assert curve_effect(P, "x_OH", ev, scope="ds2")[0] is None


def test_default_step():
    p = Parameter("t", 20.0, 15.0, 25.0)
    assert effect_step(p) == (pytest.approx(0.1), "step")       # 1 % of the range (2 % of the value is more)
    assert effect_step(p, error=0.3) == (0.3, "1σ")
    assert effect_step(Parameter("u", 0.0, -0.03, 0.03))[0] == pytest.approx(6e-4)


def test_plot_colours_the_model_curve():
    L = np.linspace(0.3, 3, 40)
    F = 10 + 5 * np.sin(L)
    Ld = np.linspace(0.3, 3, 100)
    sens = np.linspace(0, 0.02, 100)
    fig = Figure()
    plots.draw_fit_rod(fig, L, F, 0.1 * F, F, "t", Ld, 10 + 5 * np.sin(Ld), sensitivity=sens, sensitivity_label="x",
                       sensitivity_max=0.04)
    assert len(fig.axes) == 3                                   # rod, residuals, colour bar
    lc = [c for c in fig.axes[0].collections if c.get_array() is not None and len(c.get_array()) == 99][0]
    assert lc.get_array().max() == pytest.approx(0.5, abs=0.01)  # scaled to the dataset's largest change
    assert "4 %" in fig.axes[2].get_yticklabels()[-1].get_text()
    fig = Figure()
    plots.draw_fit_rod(fig, L, F, 0.1 * F, F, "t", Ld, 10 + 5 * np.sin(Ld))
    assert len(fig.axes) == 2                                   # no colours: as before


def test_target_of_a_shared_row_and_of_a_per_film_copy():
    m = truth_model()
    P = m.params
    for sc in ("ds1", "ds2", "ds3"):
        P.add(Parameter(f"{sc}.thickness_mean_tl", 20.0, 5, 40, group="film"))
    P.link("ds2.thickness_mean_tl", "ds1.thickness_mean_tl")          # same film as ds1
    assert target(P, "thickness_mean_tl", "ds3") == "ds3.thickness_mean_tl"
    assert target(P, "thickness_mean_tl", "ds2") == "ds1.thickness_mean_tl"    # follows ds1
    assert target(P, "ds2.thickness_mean_tl", "ds2") == "ds1.thickness_mean_tl"
    assert target(P, "z_OH", "ds2") == "z_OH"
    # moving the film's copy changes the dataset that follows it
    _, ev = _ev(m)
    s, _, _ = curve_effect(P, "ds1.thickness_mean_tl", ev, scope="ds2", step=0.3)
    assert s is not None and s.max() > 1e-4

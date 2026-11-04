"""The film / substrate interface: the polar shift (lattice O of the trilayers next to the interface moved
against the cations, Jeong et al. PNAS SI Figs. S10, S11) and the commensurate defaults it sits on.
Off by default (the model unchanged), the substrate part equals an explicit slab, fast = reference in every
geometry, it acts mostly on the odd rods, and each of its parameters is recovered from synthetic data."""
import numpy as np
import pytest

from ctr_params import DEFAULTS, parse_comp
from ctrfit import Fit, Model, make_dataset
from ctrfit.core.ctrmodel import CTRModel
from ctrfit.core.fast import FastCTR
from ctrfit.core.lattice import film_spacing
from ctrfit.core.sf import sf_atoms
from ctrfit.core.structures import add_trilayer, finish_atoms, new_atoms

RODS = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 3), (2, 1)]
PAPER = dict(polar_A=0.085, polar_sub_A=0.063, polar_tl=1.5, polar_sub_tl=5.0)


def _points(rods, L):
    return (np.concatenate([np.full(L.size, h) for h, _ in rods]), np.concatenate([np.full(L.size, k) for _, k in rods]),
            np.tile(L, len(rods)))


def test_defaults_are_the_commensurate_film():
    d = DEFAULTS
    assert d["strain_1m10_pct"] == pytest.approx(100 * (d["tio2_a"] / d["ruo2_a"] - 1), abs=0.005)
    _, eps = film_spacing(dict(d, eps_perp_mode="elastic"))
    assert d["eps_perp_pct"] == pytest.approx(100 * eps, abs=0.05)               # the elastic value, 1.65 %
    m = Model.from_settings(DEFAULTS)
    p = m.params["interface_A"]
    assert (p.min, p.max) == (3.0, 3.5) and p.min < p.value < p.max and not p.fit   # the mean spacing, 3.24 Å
    assert d["polar_A"] == d["polar_sub_A"] == 0 and not any(m.params[n].fit for n in ("polar_A", "polar_sub_A",
                                                                                      "polar_tl", "polar_sub_tl"))


def test_off_by_default_and_the_decay_lengths_alone_do_nothing():
    L = np.linspace(0.3, 4.0, 40)
    comp = parse_comp(DEFAULTS["comp_default"])
    a, b = CTRModel(DEFAULTS), CTRModel(dict(DEFAULTS, polar_tl=7.0, polar_sub_tl=0.5))
    for h, k in RODS:
        np.testing.assert_array_equal(a.intensity(h, k, L, comp), b.intensity(h, k, L, comp))
    assert a.polar_sub_shifts() == [] and a.polar_film_shift(1) == 0.0


def test_the_film_O_move_against_the_cations_and_the_shift_decays():
    p = dict(DEFAULTS, thickness_nm=4.0, roughness_nm=0.0, spread_nm=0.0, polar_A=0.1, polar_tl=1.0)
    on, off = CTRModel(p), CTRModel(dict(p, polar_A=0.0))
    bottom_on, _ = on.build_film_regions({})
    bottom_off, _ = off.build_film_regions({})
    dz = dict(zip(bottom_on["tag"], bottom_on["z"] - bottom_off["z"]))
    e = CTRModel.POLAR_EPS
    expect = lambda n: max(0.0, (np.exp(-n) - e) / (1 - e))
    assert expect(6) > 0 and expect(7) == 0                                    # reaches 0 at exp(-n) = 0.001
    for k, w in ((1, 1.0), (2, expect(1)), (3, expect(2)), (7, expect(6)), (8, 0.0), (11, 0.0)):
        for site in ("O_ip", "O_br", "O_br_lo"):
            assert dz[f"L{k}:{site}"] == pytest.approx(-0.1 * w, abs=1e-12), (k, site)
        assert dz[f"L{k}:M_6f"] == 0 and dz[f"L{k}:M_cus"] == 0                    # the cations stay
    assert on.polar_film_shift(1) == pytest.approx(0.1)
    top = [t for t in dz if t.startswith(f"L{on.n_film}s:")]                    # the exposed top layer: untouched
    assert top and all(dz[t] == 0 for t in top)


def test_substrate_shift_equals_an_explicit_slab():
    """The CTR sum with the top TiO2 trilayers' O moved = an explicit slab of 700 cells (absorbed), built
    straight from the trilayer sites."""
    p = dict(DEFAULTS, alpha_abs=0.05, polar_sub_A=0.07, polar_sub_tl=2.0)
    m = CTRModel(p)
    assert [j for j, _ in m.polar_sub_shifts()] == list(range(14))           # 2 x ln(1000) = 13.8 trilayers
    at = new_atoms()
    shifts = dict(m.polar_sub_shifts())
    for j in range(1400):                                                        # 700 cells of two trilayers
        dz = shifts.get(j, 0.0)
        relax = {n: -dz for n in ("O_ip", "O_br", "O_br_lo")}
        add_trilayer(at, "Ti", j % 2, -j * m.D_SUB, m.D_SUB, p["tio2_u"], float(np.exp(-0.05 * (j // 2))),
                     m.B_SUB["M"], m.B_SUB["O"], f"s{j}", relax=relax)
    slab = finish_atoms(at)
    for h, k in RODS:
        L = np.array([0.37, 1.31, 2.55, 3.72])
        qz, q = m.qvec(h, k, L)
        np.testing.assert_allclose(m.F_sub(h, k, L), sf_atoms(slab, h, k, qz, q, m.f_atom), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("over", [
    {},
    dict(thickness_nm=0.65, roughness_nm=0.3, polar_A=0.12, polar_sub_A=-0.05, polar_tl=2.5),   # interface exposed
    dict(thickness_nm=1.3, ns_tl=3, ns_du=0.01, ns_B_O=2.0, polar_tl=4.0),                     # near-surface O too
    dict(thickness_nm=4.0, coherent_nm=0.4, relax_001=0.5, b_top_extra=1.0, polar_tl=4.0),     # relaxed top reaches it
    dict(mix_mode="domains", polar_sub_tl=0.4),
    dict(polar_A=0.0),                                                                        # substrate only
])
def test_fast_equals_reference(over):
    p = dict(DEFAULTS, **dict(PAPER, **over))
    m = CTRModel(p)
    comp = parse_comp(p["comp_default"])
    L = np.linspace(0.3, 4.2, 37)
    ref = np.concatenate([m.intensity(h, k, L, comp) for h, k in RODS])
    np.testing.assert_allclose(FastCTR(*_points(RODS, L)).intensity(m, comp), ref, rtol=1e-9)


def test_fit_model_evaluator_equals_reference():
    m = Model.from_settings(DEFAULTS)
    m.params.update(dict(PAPER, thickness_mean_tl=12.6))
    L = np.linspace(0.3, 3.6, 30)
    ev = m.evaluator(*_points(RODS, L))
    np.testing.assert_allclose(ev(), np.concatenate([m.F(h, k, L) for h, k in RODS]), rtol=1e-9)


def test_it_acts_on_the_odd_rods():
    """H + K odd: the two cations of a trilayer cancel and the O carry the rod, so moving the O matters
    most there; on the even rods the cations dominate."""
    p = dict(DEFAULTS, thickness_nm=4.1, roughness_nm=0.166)
    comp = parse_comp(p["comp_default"])
    on, off = CTRModel(dict(p, **PAPER)), CTRModel(p)
    L = np.linspace(1.4, 3.2, 120)

    def change(h, k):
        return np.median(np.abs(np.sqrt(on.intensity(h, k, L, comp) / off.intensity(h, k, L, comp)) - 1))
    odd, even = max(change(0, 1), change(1, 0)), max(change(0, 2), change(2, 0))
    assert odd > 0.02 and odd > 3 * even


TRUE = dict(polar_A=0.12, polar_sub_A=0.08, polar_tl=1.5, polar_sub_tl=5.0)


@pytest.mark.parametrize("name, start", [("polar_A", 0.0), ("polar_sub_A", 0.0), ("polar_tl", 3.0),
                                         ("polar_sub_tl", 2.5)])
def test_polar_parameters_are_recovered(name, start):
    """Data with the paper's shift (a bit larger: 0.12 / 0.08 Å, decays 1.5 / 5 trilayers): each parameter
    alone comes back from a start away from it."""
    truth = Model.from_settings(DEFAULTS)
    truth.params.update(TRUE)
    data = make_dataset(truth, rods=[(0, 0), (0, 1), (1, 0), (1, 1), (0, 3)], noise_rel=0.03, seed=91, step=0.05)
    model = Model.from_settings(DEFAULTS)
    model.params.update(TRUE)
    model.params[name].value = start
    model.params.fit_only([name])
    fit = Fit(model, data)
    fit.refine()
    res = fit.report()
    true = truth.params[name].value
    assert abs(res.values[name] - true) <= 3 * res.errors[name], (name, true, res.values[name], res.errors[name])
    assert res.errors[name] < {"polar_A": 0.03, "polar_sub_A": 0.03, "polar_tl": 1.0, "polar_sub_tl": 2.0}[name]

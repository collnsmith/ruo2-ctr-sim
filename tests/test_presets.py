"""Parameter presets: values, bounds and fit flags on the model and on each dataset's copies (by potential);
the shipped Rao et al. 2017 preset applies cleanly."""
import json
from pathlib import Path

import numpy as np
import pytest

from ctrfit.fit.presets import apply_preset, check_preset, read_preset, value_for

from test_fit import truth_model
from test_scans import _file

PRESETS = Path(__file__).resolve().parents[1] / "presets" / "fit"


def _datasets():
    out = []
    for i, (nm, u, film) in enumerate((("A", 0.35, "7 nm"), ("B", 0.7, "7 nm"), ("C", None, "4 nm"))):
        d = _file(nm, 0.5, 60 + i, film=film)
        d.meta["thickness_nm"] = float(film.split()[0])
        if u is None:
            d.meta.pop("potential_V")
            d.meta["condition"] = "OCV"
        else:
            d.meta["potential_V"] = u
        out.append(d)
    return out


def test_values_by_potential_land_on_each_dataset_copy():
    m, ds = truth_model(), _datasets()
    preset = dict(per_dataset="theta, dz_Obr", per_film="thickness_mean_tl", parameters=dict(
        theta=dict(value=0.95, min=0.7, max=1.0, fit=True),
        dz_Obr=dict(value=0.35, min=-0.1, max=0.5, fit=True, by_potential=[[0.6, 0.9, 0.2]]),
        z_OH=dict(value=2.07, fit=False), t_window=dict(value=7.6, fit=False), nope=dict(value=1.0),
        x_OH=dict(value=1.4)))
    notes = apply_preset(m, ds, preset)
    P = m.params
    assert [P[f"ds{i}.dz_Obr"].value for i in (1, 2, 3)] == [0.35, 0.2, 0.35]       # 0.35 V, 0.7 V, OCV
    assert all(P[f"ds{i}.dz_Obr"].fit and P[f"ds{i}.dz_Obr"].min == -0.1 for i in (1, 2, 3))
    assert P["ds2.theta"].value == 0.95 and P["ds2.theta"].max == 1.0 and not P["theta"].fit   # master row
    assert P["z_OH"].value == 2.07 and not P["z_OH"].fit and P["t_window"].value == 7.6
    assert "ds1.z_OH" not in P                                                          # shared
    assert any("nope" in n for n in notes) and any("x_OH" in n and "outside" in n for n in notes)
    assert P["x_OH"].value == 1.0
    assert not P.check()


def test_lists_given_as_lists_or_text():
    for per in (["theta", "dz_Obr"], "theta; dz_Obr"):
        m, ds = truth_model(), _datasets()
        apply_preset(m, ds, dict(parameters=dict(dz_Obr=dict(value=0.3, fit=True))), per_dataset=per, per_film=[])
        assert "ds3.dz_Obr" in m.params


def test_value_for_and_checks():
    spec = dict(value=1.0, by_potential=[[None, 0.2, 3.0], [0.6, None, 2.0]])
    assert value_for(spec, 0.1) == 3.0 and value_for(spec, 0.4) == 1.0 and value_for(spec, 0.8) == 2.0
    assert value_for(spec, None) == 1.0
    with pytest.raises(ValueError, match="parameters"):
        check_preset({})
    with pytest.raises(ValueError, match="unknown fields"):
        check_preset(dict(parameters=dict(theta=dict(valeu=1))))
    with pytest.raises(ValueError, match="min V, max V, value"):
        check_preset(dict(parameters=dict(theta=dict(by_potential=[[0, 1]]))))


def test_shipped_rao2017_preset():
    files = sorted(PRESETS.glob("*.json"))
    assert files, "no parameter presets shipped"
    for f in files:
        preset = read_preset(f)
        m, ds = truth_model(), _datasets()
        notes = apply_preset(m, ds, preset)
        assert not notes, (f.name, notes)                  # every name exists and every value is in bounds
        assert not m.params.check()
    p = read_preset(PRESETS / "rao2017_RuO2_110_acid_low_potential.json")
    m, ds = truth_model(), _datasets()
    apply_preset(m, ds, p)
    P = m.params
    assert P["ds1.z_H2O"].value == 2.65 and P["ds2.z_H2O"].value == 2.4 and P["ds3.z_H2O"].value == 2.65
    assert P["ds1.dz_Obr"].value == 0.35 and P["ds2.dz_Obr"].value == 0.2
    assert P["x_OH"].value == 0 and not any(P[n].fit for n in P.names if n.endswith("x_OH"))
    assert P["ds1.t_liquid"].fit and P["t_window"].value == 7.6 and not P["t_window"].fit
    # a model with it evaluates (the in situ terms on)
    ev = m.evaluator(ds[0].H, ds[0].K, ds[0].L, scope=ds[0].scope)
    assert np.all(np.isfinite(ev()))
    assert json.loads((PRESETS / "rao2017_RuO2_110_acid_low_potential.json").read_text())["notes"]


# ---------------------------------------------------------------------------- rules by film thickness
def test_fields_for_thickness_then_potential():
    from ctrfit.fit.presets import fields_for
    spec = dict(value=1.0, min=0.0, max=5.0, fit=False, by_thickness=[[5.5, None, dict(value=2.0, fit=True)]],
                by_potential=[[0.6, None, 3.0]])
    assert fields_for(spec, None, 4.0) == dict(value=1.0, min=0.0, max=5.0, fit=False)
    assert fields_for(spec, 0.2, 7.0) == dict(value=2.0, min=0.0, max=5.0, fit=True)
    assert fields_for(spec, 0.7, 7.0)["value"] == 3.0 and fields_for(spec, 0.7, 7.0)["fit"]   # potential last
    assert value_for(spec, None, 7.0) == 2.0
    with pytest.raises(ValueError, match="min nm, max nm, value"):
        check_preset(dict(parameters=dict(theta=dict(by_thickness=[[0, 1]]))))
    with pytest.raises(ValueError, match="sets only"):
        check_preset(dict(parameters=dict(theta=dict(by_thickness=[[0, 1, dict(vaule=1)]]))))


def test_values_by_thickness_land_on_each_films_copy():
    """Two 7 nm datasets and a 4 nm one: a fixed parameter whose rule differs between the films gets per-film
    copies; the 7 nm film's copy takes the rule (value, bounds, fit), the other 7 nm dataset follows it."""
    m, ds = truth_model(), _datasets()
    preset = dict(per_film="relax_001, roughness_tl", parameters=dict(
        relax_001=dict(value=0.0, fit=False, by_thickness=[[5.5, None, dict(value=0.3, max=0.8, fit=True)]]),
        roughness_tl=dict(value=0.5, min=0.2, max=1.5, fit=True, by_thickness=[[5.5, None, 0.65]])))
    assert not apply_preset(m, ds, preset)
    P = m.params
    assert P["ds1.relax_001"].value == 0.3 and P["ds1.relax_001"].fit and P["ds1.relax_001"].max == 0.8
    assert P["ds2.relax_001"].linked and P["ds2.relax_001"].expr == "ds1.relax_001"            # same film
    assert P["ds3.relax_001"].value == 0.0 and not P["ds3.relax_001"].fit                       # 4 nm
    assert P["ds1.roughness_tl"].value == 0.65 and P["ds3.roughness_tl"].value == 0.5
    v1, v3 = P.view("ds2"), P.view("ds3")
    assert v1["relax_001"] == 0.3 and v3["relax_001"] == 0.0


def test_one_dataset_takes_its_rules_and_a_shared_varying_parameter_is_noted():
    m, ds = truth_model(), _datasets()
    spec = dict(value=0.1, max=0.5, by_thickness=[[5.5, None, 0.3]], by_potential=[[0.6, None, 0.4]])
    apply_preset(m, [ds[0]], dict(parameters=dict(relax_001=spec)))         # 7 nm at 0.35 V
    assert m.params["relax_001"].value == 0.3
    m = truth_model()
    apply_preset(m, [ds[1]], dict(parameters=dict(relax_001=spec)))         # 7 nm at 0.7 V: potential wins
    assert m.params["relax_001"].value == 0.4
    m, ds = truth_model(), _datasets()
    notes = apply_preset(m, ds, dict(parameters=dict(relax_001=spec)), per_dataset=[], per_film=[])
    assert any("relax_001" in n and "shared" in n for n in notes) and m.params["relax_001"].value == 0.1


def test_shipped_film_preset_after_the_surface_one():
    m, ds = truth_model(), _datasets()
    apply_preset(m, ds, read_preset(PRESETS / "rao2017_RuO2_110_acid_low_potential.json"))
    notes = apply_preset(m, ds, read_preset(PRESETS / "jeong2026_RuO2_TiO2_110_films.json"))
    assert not notes
    P = m.params
    assert P["ds2.z_H2O"].value == 2.4                                       # the surface preset's copies stay
    assert P["ds3.eps_perp"].value == 1.65 and P["ds3.eps_perp"].min == 1.3 and P["ds1.eps_perp"].min == 1.0
    assert P["ds1.relax_001"].fit and P["ds1.coherent_tl"].value == 19 and not P["ds3.relax_001"].fit
    assert P["ds3.roughness_tl"].value == 0.51 and P["ds1.roughness_tl"].value == 0.65
    assert P["interface_A"].value == 3.24 and not P["interface_A"].fit
    assert P["polar_A"].value == 0.085 and P["polar_sub_tl"].value == 5.0 and not P["polar_A"].fit
    for d in ds:
        ev = m.evaluator(d.H, d.K, d.L, scope=d.scope)
        assert np.all(np.isfinite(ev()))

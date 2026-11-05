"""Skip near Bragg peaks: each rod's TiO2 Bragg peaks from the substrate structure factor (the rutile
extinctions included, checked against the conventional cell), the points within ±dL of them left out of a
fit (moved with a rod's L offset), in the fitting workflow, the command line, the processing tool and
the project file."""
import numpy as np
import pytest

from ctr_params import DEFAULTS
from ctrfit import Fit, Model, make_dataset
from ctrfit.cli import main
from ctrfit.core.ctrmodel import CTRModel, near_substrate_bragg, substrate_bragg_L
from ctrfit.data.dataset import Dataset
from ctrfit.fit.workflow import near_bragg, skip_bragg
from ctrfit.project.project import Project

from test_physics import rutile_conventional_F


def _rod(h, k, L, scope="", **meta):
    L = np.asarray(L, float)
    return Dataset(np.full(L.size, h), np.full(L.size, k), L, np.ones(L.size), np.full(L.size, 0.05), scope=scope,
                   name=f"rod {h} {k}", **meta)


@pytest.mark.parametrize("hk, peaks", [((0, 0), [0, 2, 4, 6]), ((0, 1), [3, 5]), ((0, -1), [3, 5]),
                                       ((1, 0), [2, 4, 6]), ((1, 1), [1, 3, 5]), ((0, 2), [0, 2, 4, 6]),
                                       ((0, 3), [1, 5]), ((2, 1), [3, 5])])
def test_each_rods_peaks_with_the_rutile_extinctions(hk, peaks):
    """Not every integer L: e.g. (0 1 1) would be the forbidden (100), (0 3 3) the forbidden (300). The same
    set from the conventional rutile cell, (h k l) = ((K + L) / 2, (L - K) / 2, H)."""
    assert substrate_bragg_L(*hk, 0.0, 6.0).tolist() == peaks
    m = CTRModel(dict(DEFAULTS, anom_mode="off"))
    H, K = hk
    conv = [L for L in range(7) if (K + L) % 2 == 0 and abs(rutile_conventional_F(m, (K + L) // 2, (L - K) // 2, H)) > 1e-4]
    assert conv == peaks


def test_near_substrate_bragg_mixed_rods_and_edges():
    H = np.array([0, 0, 0, 0, 1, 1, 1])
    K = np.array([1, 1, 1, 1, 0, 0, 0])
    L = np.array([1.0, 2.0, 2.95, 3.06, 1.0, 2.05, 2.051])
    assert near_substrate_bragg(H, K, L, 0.05).tolist() == [False, False, True, False, False, True, False]
    assert not near_substrate_bragg(H, K, L, 0.0).any()


def test_near_bragg_moves_with_the_rods_L_offset():
    ds = _rod(0, 1, [2.9, 2.96, 3.0, 3.04, 3.1], scope="ds2")
    assert near_bragg(ds, 0.05).tolist() == [False, True, True, True, False]
    m = Model.from_settings(DEFAULTS)
    m.add_rod_offsets([(0, 1)], scope="ds2")
    m.params["ds2.dL_0_1"].value = 0.06               # the model is taken at L + dL: the data's peak at 2.94
    assert near_bragg(ds, 0.05, m).tolist() == [True, True, False, False, False]
    assert near_bragg(ds, 0.05, values={"dL_0_1": -0.06}).tolist() == [False, False, False, True, True]


def test_skip_bragg_subsets_and_notes():
    a = _rod(0, 1, np.arange(2.5, 3.51, 0.05), scope="ds1", potential_V=0.2)
    b = _rod(1, 0, [1.97, 2.0, 2.03], scope="ds2")
    out, notes = skip_bragg([a, b], 0.05)
    assert len(out) == 1 and out[0].scope == "ds1" and out[0].meta["potential_V"] == 0.2
    assert np.all(np.abs(out[0].L - 3) > 0.05) and len(out[0]) == len(a) - 3
    assert any("3 of 21" in n for n in notes) and any("rod 1 0" in n and "sits this run out" in n for n in notes)
    assert skip_bragg([a], 0.0) == ([a], [])
    with pytest.raises(ValueError, match="every point"):
        skip_bragg([b], 0.05)


def test_a_fit_without_the_points_near_the_peaks():
    """Data whose points near the peaks are spoiled (as a saturated detector would): with the skip the fit
    recovers the truth; without it, it does not."""
    truth = Model.from_settings(DEFAULTS)
    data = make_dataset(truth, rods=[(0, 0), (0, 1), (1, 0)], noise_rel=0.02, seed=7, step=0.02, avoid_bragg=0.0)
    near = near_bragg(data, 0.1)
    data.F[near] *= 0.3
    start = dict(thickness_mean_tl=truth.params["thickness_mean_tl"].value + 0.4)
    res = {}
    for w in (0.0, 0.1):
        m = Model.from_settings(DEFAULTS)
        m.params.update(start)
        m.params.fit_only(["thickness_mean_tl", "scale"])
        fit = Fit(m, skip_bragg([data], w)[0])
        fit.refine()
        res[w] = fit.report()
    assert res[0.1].fom["N"] == len(data) - int(near.sum())
    t = truth.params["thickness_mean_tl"].value
    assert abs(res[0.1].values["thickness_mean_tl"] - t) <= 3 * res[0.1].errors["thickness_mean_tl"]
    assert res[0.1].fom["red_chi2"] < 2 < res[0.0].fom["red_chi2"]


def test_command_line_skip(tmp_path, capsys):
    truth = Model.from_settings(DEFAULTS)
    data = make_dataset(truth, rods=[(0, 1), (1, 0)], noise_rel=0.02, seed=8, step=0.05, avoid_bragg=0.0)
    from ctrfit.data.io import write_csv
    write_csv(data, tmp_path / "d.csv")
    model = Model.from_settings(DEFAULTS)
    model.params.fit_only(["scale"])
    model.save(tmp_path / "m.json")
    main(["fit", str(tmp_path / "m.json"), str(tmp_path / "d.csv"), "--method", "lsq", "--skip-bragg", "0.1",
          "--no-plots", "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    n = int(near_bragg(data, 0.1).sum())
    assert n > 0 and f"{n} of {len(data)} points" in out and f"{len(data) - n} data points" in out


def test_processing_skips_only_each_rods_peaks():
    """The processing tool's Bragg window on the (0 1) rod: around L = 3 only, not around L = 1 or 2."""
    from ctrfit.datatools import processing as pr
    L = np.arange(0.5, 3.5, 0.01)
    near = near_substrate_bragg(np.zeros(L.size), np.ones(L.size), L, 0.05)
    assert np.all(np.abs(L[near] - 3) <= 0.05 + 1e-9) and near.sum() == 11
    assert pr.near_substrate_bragg is near_substrate_bragg


def test_project_keeps_the_skip(tmp_path):
    p = Project(options=dict(skip_bragg=0.07))
    p.save(tmp_path / "p.ctrproj.json")
    assert Project.load(tmp_path / "p.ctrproj.json").options == {"skip_bragg": 0.07}
    assert Project().options == {}

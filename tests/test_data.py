"""Figures of merit and I -> F against hand-computed values; dataset import/export round trips; CLI."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ctr_params import DEFAULTS
from ctrfit import Dataset, Model, make_dataset
from ctrfit.data import intensities_to_F, read_csv, read_dat, write_dat
from ctrfit.fit import fom

ROOT = Path(__file__).resolve().parents[1]
F = np.array([10.0, 20.0, 30.0])
FC = np.array([11.0, 18.0, 30.0])
S = np.array([1.0, 2.0, 3.0])


def test_chi2_by_hand():
    np.testing.assert_allclose(fom.chi_residuals(F, FC, S), [-1.0, 1.0, 0.0])
    assert fom.chi2(F, FC, S) == pytest.approx(2.0)
    assert fom.reduced_chi2(F, FC, S, 1) == pytest.approx(1.0)        # 2 / (3 - 1)


def test_log_fom_by_hand():
    expected = (abs(np.log10(10 / 11)) + abs(np.log10(20 / 18))) / 2  # GenX: / (N - 1)
    assert fom.log_fom(F, FC) == pytest.approx(expected) == pytest.approx(0.0435749, rel=1e-5)


def test_r1_by_hand():
    assert fom.r1(F, FC) == pytest.approx(3.0 / 60.0)


def test_all_foms():
    d = fom.all_foms(F, FC, S, 1)
    assert d == pytest.approx(dict(chi2=2.0, red_chi2=1.0, log=0.0435749, R1=0.05, N=3, p=1), rel=1e-5)


def test_intensity_to_F_by_hand():
    Fv, sF = intensities_to_F([100.0, 4.0, 1.0, -3.0], [10.0, 8.0, 8.0, 8.0])
    np.testing.assert_allclose(Fv, [10.0, 2.0, 1.0, 0.0])
    np.testing.assert_allclose(sF, [0.5, 2.0, np.sqrt(8.0), np.sqrt(8.0)])


def test_sigma_eff_adds_floor_in_quadrature():
    ds = Dataset([0], [1], [1.5], [40.0], [3.0], sys_floor=0.1)
    assert ds.sigma_eff[0] == pytest.approx(5.0)


def test_dataset_validation():
    with pytest.raises(ValueError):
        Dataset([0.5], [1], [1.0], [1.0], [1.0])
    with pytest.raises(ValueError):
        Dataset([0], [1], [1.0], [1.0], [0.0])


@pytest.fixture
def synth():
    m = Model.from_settings(DEFAULTS)
    return make_dataset(m, rods=[(0, 1), (1, 0)], noise_rel=0.05, sys_floor=0.02, seed=7, step=0.2,
                        name="test set", energy_kev=16.0, potential_V=1.2, thickness_nm=6.5)


def _same(a, b, meta=True):
    for k in ("H", "K", "L", "F", "sigma", "rod"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
    assert a.sys_floor == b.sys_floor and a.name == b.name
    if meta:
        assert a.meta == b.meta


def test_synthetic_is_reproducible(synth):
    m = Model.from_settings(DEFAULTS)
    again = make_dataset(m, rods=[(0, 1), (1, 0)], noise_rel=0.05, sys_floor=0.02, seed=7, step=0.2,
                         name="test set", energy_kev=16.0, potential_V=1.2, thickness_nm=6.5)
    _same(synth, again)
    other = make_dataset(m, rods=[(0, 1), (1, 0)], noise_rel=0.05, seed=8, step=0.2)
    assert not np.array_equal(synth.F, other.F)
    assert synth.rod_labels == ["0 1", "1 0"] and synth.meta["potential_V"] == 1.2


@pytest.mark.parametrize("ext", ["json", "csv"])
def test_save_load_round_trip(tmp_path, synth, ext):
    path = tmp_path / f"d.{ext}"
    synth.save(path)
    _same(synth, Dataset.load(path))


def test_dat_round_trip(tmp_path, synth):
    path = tmp_path / "d.dat"
    write_dat(synth, path)
    back = read_dat(path)
    for k in ("H", "K", "L", "F", "sigma"):
        np.testing.assert_array_equal(getattr(back, k), getattr(synth, k))
    assert back.sys_floor == synth.sys_floor and back.meta["energy_kev"] == 16.0


def test_csv_header_variants_and_intensities(tmp_path):
    p = tmp_path / "i.csv"
    p.write_text("# name: my rods\n# sys_floor: 0.05\nl,K,h,I,sigI\n1.5,1,0,100,10\n2.5,0,1,4,8\n", encoding="utf-8")
    ds = read_csv(p, intensities=True)
    assert ds.name == "my rods" and ds.sys_floor == 0.05
    np.testing.assert_array_equal(ds.H, [0, 1])
    np.testing.assert_array_equal(ds.K, [1, 0])
    np.testing.assert_allclose(ds.F, [10.0, 2.0])
    np.testing.assert_allclose(ds.sigma, [0.5, 2.0])
    with pytest.raises(ValueError, match="no column"):
        read_csv(p)                                   # no F column without intensities=True


def test_select_rods(synth):
    one = synth.select_rods([(1, 0)])
    assert one.rod_labels == ["1 0"] and len(one) == len(synth) // 2
    assert synth.select_rods(["0 1"]).rod_labels == ["0 1"]


def test_cli_fit_writes_results_and_plots(tmp_path):
    m = Model.from_settings(DEFAULTS)
    m.params.update(dict(z_OH=2.1))
    ds = make_dataset(m, rods=[(0, 1), (1, 0)], noise_rel=0.03, seed=9, step=0.1)
    ds.save(tmp_path / "data.csv")
    start = Model.from_settings(DEFAULTS)
    start.params.fit_only(["z_OH", "scale"])
    start.params["scale"].min, start.params["scale"].max = 0.5, 2.0
    start.params["z_OH"].min, start.params["z_OH"].max = 1.6, 2.5
    start.save(tmp_path / "model.json")
    out = tmp_path / "out"
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(ROOT))
    code = "import sys; sys.modules['xraydb'] = None; from ctrfit.cli import main; sys.exit(main())"
    r = subprocess.run([sys.executable, "-c", code, "fit", str(tmp_path / "model.json"), str(tmp_path / "data.csv"),
                        "--out", str(out), "--maxiter", "8", "--popsize", "6"],
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((out / "result.json").read_text())
    assert abs(res["values"]["z_OH"] - 2.1) < 3 * res["errors"]["z_OH"]
    for f in ("summary.txt", "fom_history.png", "correlation.png", "model_fitted.json",
              "rod_synthetic_0_1.png", "rod_synthetic_1_0.png"):
        assert (out / f).exists(), f
    assert "reduced chi2" in r.stdout


def test_python_dash_m_entry_point():
    r = subprocess.run([sys.executable, "-m", "ctrfit", "--help"], capture_output=True, text=True, cwd=ROOT,
                       timeout=60)
    assert r.returncode == 0 and "fit" in r.stdout

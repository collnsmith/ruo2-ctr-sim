"""Beamline helper: psic geometry, UB, HKL <-> angles, Bragg peak indexing, macros, calculators."""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ctrfit.beamline import macros as mac
from ctrfit.beamline import xtools
from ctrfit.beamline.indexing import allowed_tio2_surface, candidate_reflections, index_peaks
from ctrfit.beamline.psic import (MODES, MOTORS, Lattice, Psic, angles_for_hkl, as_angles, energy_to_wavelength,
                                  mode_with_alpha, parse_angle_lines, rotation_from_vector, tio2_surface_lattice,
                                  ub_from_two_reflections)

ROOT = Path(__file__).resolve().parents[1]
LAM = energy_to_wavelength(16.0)
LAT = tio2_surface_lattice()
GEO = Psic()


def aligned_ub(miscut_deg=0.3, azimuth_deg=17.0, seed=0):
    """Surface cell with L* close to the phi axis (z of the phi frame), a small miscut and an
    arbitrary azimuth: the usual state after mounting a (110) crystal."""
    rng = np.random.default_rng(seed)
    U = rotation_from_vector([0, 0, math.radians(azimuth_deg)])
    tilt = rng.normal(size=3)
    tilt[2] = 0
    U = rotation_from_vector(math.radians(miscut_deg) * tilt / np.linalg.norm(tilt)) @ U
    return U @ LAT.B


# ---------------------------------------------------------------------------- geometry
def test_lattice_B_orthogonal():
    np.testing.assert_allclose(LAT.B, np.diag(2 * np.pi / np.array([LAT.a, LAT.b, LAT.c])), atol=1e-12)
    assert LAT.d([0, 0, 2])[0] == pytest.approx(4.5937 / math.sqrt(2))       # TiO2 (110) spacing
    hexa = Lattice(3.0, 3.0, 5.0, 90, 90, 120)
    assert hexa.d([1, 0, 0])[0] == pytest.approx(3.0 * math.sqrt(3) / 2)


def test_q_magnitude_and_two_theta():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = as_angles(rng.uniform(-90, 90, 6))
        q = np.linalg.norm(GEO.q_lab(a, LAM))
        tth = math.radians(GEO.two_theta(a))
        assert q == pytest.approx(2 * (2 * math.pi / LAM) * math.sin(tth / 2))
        assert np.linalg.norm(GEO.q_phi(a, LAM)) == pytest.approx(q)


def test_bisecting_q_along_x_and_detector_directions():
    a = as_angles(dict(**{"del": 30.0}, eta=15.0))
    qp = GEO.q_phi(a, LAM)
    assert qp[1] == pytest.approx(0, abs=1e-12) and qp[2] == pytest.approx(0, abs=1e-12) and qp[0] > 0
    ko = GEO.k_out(as_angles({"del": 20.0}), LAM)
    assert ko[0] > 0 and ko[2] == pytest.approx(0)            # delta: vertical plane (x is up)
    ko = GEO.k_out(as_angles({"nu": 20.0}), LAM)
    assert ko[2] > 0 and ko[0] == pytest.approx(0)            # nu: horizontal plane


def test_motor_signs():
    a = as_angles([20, 10, 5, 30, 3, 1])
    flipped = Psic({"eta": -1, "nu": -1})
    b = dict(a, eta=-a["eta"], nu=-a["nu"])
    np.testing.assert_allclose(flipped.q_phi(b, LAM), GEO.q_phi(a, LAM))


@pytest.mark.parametrize("mode_name", list(MODES))
def test_angles_for_hkl_round_trip(mode_name):
    UB = aligned_ub()
    normal = UB @ [0, 0, 1.0]
    mode, fixed = mode_with_alpha(mode_name, 0.4)
    if mode_name.startswith("four-circle"):
        UB = rotation_from_vector([0.3, -0.5, 0.2]) @ LAT.B
    for hkl in ([0, 1, 1.5], [1, 0, 2.3], [1, 1, 0.8], [0, 2, 3.1]):
        sols = angles_for_hkl(hkl, UB, LAM, mode, fixed=fixed, normal_phi=normal)
        assert sols, (mode_name, hkl)
        for s in sols[:2]:
            np.testing.assert_allclose(GEO.hkl(s, UB, LAM), hkl, atol=1e-7)
            for m, v in fixed.items():
                assert s[m] == pytest.approx(v)
            if mode.constraint and mode.constraint[0] == "alpha":
                assert s["alpha"] == pytest.approx(0.4, abs=1e-6)
            if mode.constraint and mode.constraint[0] == "alpha=beta":
                assert s["alpha"] == pytest.approx(s["beta"], abs=1e-6)


def test_unreachable_reflection_has_no_solution():
    mode, fixed = mode_with_alpha("vertical surface, fixed alpha", 0.5)
    assert angles_for_hkl([9, 9, 9], aligned_ub(), LAM, mode, fixed=fixed, normal_phi=[0, 0, 1]) == []


def test_ub_from_two_reflections():
    UB = rotation_from_vector([0.2, 0.4, -0.3]) @ LAT.B
    mode, fixed = MODES["four-circle vertical (del, eta, chi)"]
    h1, h2 = [0, 0, 2], [1, 1, 1]
    a1 = angles_for_hkl(h1, UB, LAM, mode, fixed=fixed)[0]
    a2 = angles_for_hkl(h2, UB, LAM, mode, fixed=fixed)[0]
    np.testing.assert_allclose(ub_from_two_reflections(LAT, h1, a1, h2, a2, LAM), UB, atol=1e-9)


# ---------------------------------------------------------------------------- indexing
def test_tio2_extinctions():
    assert all(allowed_tio2_surface(h) for h in TRUE)
    for hkl in ([0, 0, 2], [1, 1, 1], [0, 2, 2], [1, 0, 2], [0, 1, 3], [2, 0, 0]):
        assert allowed_tio2_surface(hkl), hkl
    for hkl in ([0, 0, 1], [0, 1, 1], [1, 0, 0], [2, 1, 1], [0, 1, 2]):  # K+L odd, or rutile glide/screw
        assert not allowed_tio2_surface(hkl), hkl
    hkl, q = candidate_reflections(LAT, 3.0, allowed_tio2_surface)
    assert all(allowed_tio2_surface(h) for h in hkl) and np.all(q <= 3.0)


TRUE = [(0, 0, 2), (1, 1, 1), (1, 0, 2), (0, 2, 2), (1, 1, 3), (2, 0, 2)]


def synthetic_peaks(UB, noise_deg=0.01, seed=5):
    rng = np.random.default_rng(seed)
    mode, fixed = MODES["four-circle vertical (del, eta, chi)"]
    peaks = []
    for h in TRUE:
        s = angles_for_hkl(h, UB, LAM, mode, fixed=dict(fixed, phi=25.0))[0]
        peaks.append({m: s[m] + (noise_deg * rng.standard_normal() if m in ("del", "eta", "chi") else 0.0)
                      for m in MOTORS})
    return peaks


def test_index_with_surface_normal_recovers_hkl():
    UB = aligned_ub(seed=2)
    order = [3, 0, 5, 1, 4, 2]                                   # peaks in any order
    peaks = [synthetic_peaks(UB)[i] for i in order]
    truth = [TRUE[i] for i in order]
    res = index_peaks(peaks, LAM, LAT, allowed_tio2_surface, normal_phi=[0, 0, 1])
    assert res.n_indexed == len(peaks) and res.rms_dq < 5e-3
    for p, t in zip(res.peaks, truth):
        assert p.hkl[2] == t[2] and abs(p.hkl[0]) == t[0] and abs(p.hkl[1]) == t[1], (p.hkl, t)
    # the remaining choice is the in-plane two-fold (H, K) -> (-H, -K): consistent over all peaks
    s = {(np.sign(p.hkl[0]) if t[0] else 0, np.sign(p.hkl[1]) if t[1] else 0) for p, t in zip(res.peaks, truth)}
    assert len({x for x, _ in s if x}) <= 1 and len({y for _, y in s if y}) <= 1
    assert "UB" in res.summary


def test_index_without_normal_lists_equivalent_indexings():
    UB = aligned_ub(seed=3)
    res = index_peaks(synthetic_peaks(UB), LAM, LAT, allowed_tio2_surface)
    assert res.n_indexed == len(TRUE) and len(res.alternatives) > 1
    assert any(alt == TRUE for alt in res.alternatives)          # the truth is among them
    assert "surface normal" in res.summary


def test_index_rejects_peak_that_is_not_bragg_and_one_peak_mode():
    UB = aligned_ub(seed=4)
    peaks = synthetic_peaks(UB)
    junk = dict(peaks[0])
    junk["del"] += 3.0                                          # not on any reflection
    mode, fixed = MODES["four-circle vertical (del, eta, chi)"]
    forb = angles_for_hkl((1, 2, 1), UB, LAM, mode, fixed=dict(fixed, phi=25.0))[0]   # K + L odd
    res = index_peaks(peaks + [junk, forb], LAM, LAT, allowed_tio2_surface, normal_phi=[0, 0, 1])
    assert res.n_indexed == len(TRUE) and not res.peaks[-2].indexed and not res.peaks[-2].forbidden
    assert res.peaks[-1].forbidden and "forbidden" in res.summary
    one = index_peaks(peaks[:1], LAM, LAT, allowed_tio2_surface)
    assert one.UB is None and (0, 0, 2) in [h for h, _ in one.candidates]


def test_parse_angle_lines():
    text = "# my peaks\n20.1 10.05 90 12 0 0.5  # (0 0 2)\ndelta=30 eta=15 chi=0 phi=0 nu=1 mu=0\n\n"
    p = parse_angle_lines(text)
    assert len(p) == 2 and p[0]["chi"] == 90 and p[0]["label"] == "(0 0 2)" and p[1]["del"] == 30
    with pytest.raises(ValueError):
        parse_angle_lines("foo=1 del=2")
    with pytest.raises(ValueError):
        parse_angle_lines("1 2 3")


# ---------------------------------------------------------------------------- macros
def test_rod_points_skip_bragg_and_refine_near_it():
    L = mac.rod_L_points(0.3, 4.0, 0.1, bragg=[2.0], exclude=0.05, fine_step=0.02, fine_window=0.3)
    assert np.all(np.abs(L - 2.0) >= 0.05) and L.min() >= 0.3 and L.max() <= 4.0
    near = L[(L > 1.7) & (L < 2.3)]
    assert np.allclose(np.diff(near[near < 1.95]), 0.02)
    assert np.allclose(np.round(near / 0.02), near / 0.02)          # fine grid on multiples of the step


def test_rod_macro_with_model_and_times():
    from ctrfit import Model
    model = Model.from_settings()
    s = mac.MacroSettings(before_rod="fon", rock="dscan phi -{hw:g} {hw:g} {n} {t:g}", rock_points=11)
    m = mac.rod_macro([(0, 1), (1, 1)], s, 0.3, 3.5, 0.1, 1.0, model, scale_times=True)
    text = m.text("test")
    assert text.count("fon") == 2 and "br 0 1 0.3000" in text and "dscan phi -0.5 0.5 11" in text
    assert "Bragg peaks at" in text and m.n_points > 40
    assert "br 0 1 3.0000" not in text                             # (0 1 3) Bragg peak skipped
    times = [float(ln.split()[-1]) for ln in text.splitlines() if ln.startswith("dscan")]
    assert min(times) >= 1 and max(times) <= 60 and max(times) > 2 * min(times)
    assert m.seconds > m.count_s > 0


def test_potential_energy_points_and_def(tmp_path):
    inner = mac.points_macro({"P1": (0, 1, 1.1), "P2": (0, 1, 2.85)}, t=5)
    m = mac.potential_macro([0.4, 1.2], inner, wait_s=30)
    lines = m.text().splitlines()
    assert "potset 0.400" in lines and "potset 1.200" in lines and lines.count("sleep(30)") == 2
    assert m.n_points == 4 and m.count_s == 20
    e = mac.energy_macro((0, 1, 1.5), [22.0, 22.1], t=2)
    assert e.lines == ["moveE 22.0000", "br 0 1 1.5000", "ct 2", "moveE 22.1000", "br 0 1 1.5000", "ct 2"]
    d = mac.wrap_def("ctr_run", e.text("e"))
    assert d.splitlines()[-1] == "'" and "def ctr_run '" in d and "    moveE 22.0000" in d
    with pytest.raises(ValueError):
        mac.wrap_def("bad name", "x")
    path = mac.write_macro(tmp_path / "m.mac", e)
    assert Path(path).read_text().startswith("# energy scan")


# ---------------------------------------------------------------------------- calculators
def test_calculators():
    assert xtools.critical_angle("Si", 2.329, 8.048) == pytest.approx(0.22, abs=0.005)   # literature 0.22 deg
    ac = xtools.critical_angle("TiO2", 4.23, 16.0)
    assert 0.12 < ac < 0.18
    below, above = (xtools.penetration_depth(a, "TiO2", 4.23, 16.0) for a in (0.5 * ac, 3 * ac))
    assert below < 100 and above > 50 * below
    fp, frac = xtools.footprint(0.02, 0.5, 5.0)
    assert fp == pytest.approx(0.02 / math.sin(math.radians(0.5))) and frac == 1.0
    assert xtools.footprint(0.1, 0.5, 5.0)[1] == pytest.approx(5 * math.sin(math.radians(0.5)) / 0.1)
    assert xtools.alpha_for_footprint(0.1, 5.0) == pytest.approx(math.degrees(math.asin(0.02)))
    d = 4.5937 / math.sqrt(2)
    assert xtools.two_theta(LAT, [0, 0, 2], 16.0) == pytest.approx(2 * math.degrees(math.asin(LAM / (2 * d))))
    assert xtools.parse_formula("C22H10N2O5") == {"C": 22, "H": 10, "N": 2, "O": 5}


# ---------------------------------------------------------------------------- CLI and window
def test_cli_index_and_macro(tmp_path):
    peaks = synthetic_peaks(aligned_ub(seed=6))
    f = tmp_path / "peaks.txt"
    f.write_text("\n".join(" ".join(f"{p[m]:.4f}" for m in MOTORS) for p in peaks))
    code = "import sys; sys.modules['xraydb'] = None; from ctrfit.cli import main; sys.exit(main())"
    env = dict(__import__("os").environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-c", code, "index", str(f), "--energy", "16", "--normal", "0 0 1",
                        "--save-ub", str(tmp_path / "ub.json")], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "6 of 6 peaks indexed" in r.stdout and (tmp_path / "ub.json").exists()
    r = subprocess.run([sys.executable, "-c", code, "macro", "--rods", "0 1", "--lmax", "2", "--step", "0.1",
                        "--out", str(tmp_path / "r.mac")], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "br 0 1 0.3000" in (tmp_path / "r.mac").read_text()


def test_beamline_window(monkeypatch, tmp_path):
    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets
    from ctrfit.app.beamline_window import BeamlineWindow
    msgs = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: msgs.append(a[2])))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = BeamlineWindow()
    w.show()
    UB = aligned_ub(seed=7)
    peaks = synthetic_peaks(UB)
    w.peaks_in.setPlainText("\n".join(" ".join(f"{p[m]:.4f}" for m in MOTORS) + f"  # peak {i}"
                                      for i, p in enumerate(peaks)))
    w.normal_in.setText("0 0 1")
    w.run_index()
    assert not msgs, msgs
    assert "6 of 6 peaks indexed" in w.index_out.toPlainText()
    w.use_index_ub()
    assert w.UB is not None and w.tabs.currentIndex() == 1
    w.mode.setCurrentText("vertical surface, fixed alpha")
    w.alpha.setValue(0.4)
    w.hkl_in.setText("0 1 1.5")
    w.calc_angles()
    out = w.ang_out.toPlainText()
    assert "umv del" in out and not msgs, (out, msgs)
    best = [float(x) for x in out.split("umv ")[1].split()[1::2]]
    w.where_in.setText(" ".join(map(str, best)))
    w.calc_where()
    assert "H K L = 0.0000 1.0000 1.5000" in w.ang_out.toPlainText().replace("-0.0000", "0.0000")
    w.mac_kind.setCurrentText("potential series (fixed points)")
    w.make_macro()
    assert "potset" in w.mac_out.toPlainText()
    w.save_macro(str(tmp_path / "x.mac"))
    assert (tmp_path / "x.mac").exists()
    w.tabs.setCurrentIndex(3)
    w.update_calc()
    assert "critical angle" in w.calc_out.toPlainText()
    w.peaks_in.setPlainText("1 2 3")
    w.run_index()
    assert msgs                                                   # bad input reported, window alive
    w.close()

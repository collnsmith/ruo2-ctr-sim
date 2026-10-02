"""Diffraction geometry for the X-ray beams in the 3D viewer."""
import math

import numpy as np
import pytest

from ctr_engine import CTRModel
from ctr_params import DEFAULTS
from ctrfit.core.geometry import beam_geometry, q_vector


@pytest.fixture(scope="module")
def m():
    return CTRModel(DEFAULTS)


@pytest.mark.parametrize("hkl", [(0, 1, 1.5), (1, 0, 0.8), (1, 1, 3.2), (0, 3, 2.0), (2, 1, 4.0)])
@pytest.mark.parametrize("alpha", [0.2, 0.5, 2.0])
def test_elastic_and_momentum_transfer(m, hkl, alpha):
    g = beam_geometry(m, *hkl, alpha_deg=alpha)
    assert g["ok"], g["reason"]
    k = 2 * np.pi / m.lam
    np.testing.assert_allclose(np.linalg.norm(g["k_in"]), k, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(g["k_out"]), k, rtol=1e-9)
    np.testing.assert_allclose(g["k_out"] - g["k_in"], q_vector(m, *hkl), atol=1e-9)
    assert g["k_in"][2] < 0 < g["k_out"][2]
    assert math.degrees(math.asin(-g["k_in"][2] / k)) == pytest.approx(alpha)
    cos2t = g["k_in"] @ g["k_out"] / k ** 2
    assert math.degrees(math.acos(cos2t)) == pytest.approx(g["two_theta"], abs=1e-6)


def test_q_matches_engine_convention(m):
    qz, q = m.qvec(1, 2, [1.7])
    v = q_vector(m, 1, 2, 1.7)
    assert v[2] == pytest.approx(qz[0]) and np.linalg.norm(v) == pytest.approx(q[0])


def test_specular_rod(m):
    g = beam_geometry(m, 0, 0, 2.0, alpha_deg=0.3)          # alpha follows from L
    assert g["ok"] and g["alpha"] == pytest.approx(g["beta"]) == pytest.approx(g["two_theta"] / 2)
    assert g["k_in"][:2] == pytest.approx(g["k_out"][:2])
    # TiO2 (110) Bragg peak is (0 0 2): 2 theta from d(110) = a / sqrt(2)
    d = DEFAULTS["tio2_a"] / math.sqrt(2)
    assert g["two_theta"] == pytest.approx(2 * math.degrees(math.asin(m.lam / (2 * d))))


def test_unreachable_cases(m):
    assert not beam_geometry(m, 0, 1, 0.0, alpha_deg=0.5)["ok"]          # exit beam into the sample
    far = beam_geometry(m, 9, 9, 6.0)
    assert not far["ok"] and "beyond" in far["reason"]
    assert not beam_geometry(m, 0, 0, 0.0)["ok"]


def test_viewer_draws_beams(monkeypatch):
    pytest.importorskip("numba")
    pytest.importorskip("PyQt5")
    import ctr_viewer3d as v
    from PyQt5 import QtWidgets
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = v.ViewerWindow()
    for w, val in ((win.nx, 3), (win.ny, 2), (win.nsub, 1)):
        w.setValue(val)
    win.scale.setCurrentIndex(3)
    win.resize(500, 400)
    win.show()
    app.processEvents()
    win.first_build()
    win.timer.stop()
    win.tick()
    win.beam_h.setValue(1)
    win.beam_k.setValue(1)
    win.beam_l.setValue(250)
    assert win.beam is not None and "2θ" in win.beam_info.text()
    img = win.view.grab().toImage()
    cols = {img.pixelColor(x, y).getRgb()[:3] for x in range(0, img.width(), 2) for y in range(0, img.height(), 2)}
    assert v.BEAM_IN in cols and v.BEAM_OUT in cols
    win.beam_l.setValue(0)                                   # exit beam would go into the sample
    assert win.beam is None and "into the sample" in win.beam_info.text()
    win.chk_beam.setChecked(False)
    assert win.beam is None
    cam = win.camera.vectors(1.25)
    xy, front = v.project(cam, cam[0:3] + 10 * cam[3:6], 500, 400)   # point straight ahead -> centre
    assert front[0] and xy[0] == pytest.approx([250, 200])
    win.close()
    app.processEvents()


def test_visibility_behind_a_sphere():
    pytest.importorskip("numba")
    pytest.importorskip("PyQt5")
    import ctr_viewer3d as v
    cam = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.3, 1.0], float)    # at origin, looking along +x
    centers = np.array([[5.0, 0.0, 0.0]], np.float32)
    radii = np.array([1.0], np.float32)
    vis = v.visible(cam, np.array([[10.0, 0, 0], [3.0, 0, 0], [10.0, 3.0, 0]]), centers, radii)
    assert list(vis) == [False, True, True]

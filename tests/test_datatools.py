"""SXRD data tools: the no-Qt logic (ctrfit.datatools) on synthetic files, and the tabbed window offscreen."""
import os
import subprocess
import sys

import numpy as np
import pytest

from ctrfit.datatools import bgfind, ctrmodels, hkl, images, peaks, specfile
from ctrfit.datatools.paths import settings_dir

L = np.linspace(0.5, 2.5, 21)


def write_spec(path):
    lines = ["#F test.spec", "#O0 Two Theta  Theta  Chi", "#O1 Phi  Mu", "",
             "#S 464  hklscan 0 0 0 2 0.5 2.5 20 1", "#P0 10 5 90", "#P1 -3.5 0.4", "#L H  K  L  Epoch  ROI5"]
    lines += [f"0.0003 1.9990 {l:.4f} {i} {100 + i}" for i, l in enumerate(L)]
    lines += ["1 2 3 not-a-number 5", "1 2", "", "#S 465  ascan  th 1 2 2 1", "#L Theta  Detector", "1 10", "2 20"]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_export(path, scan=464, xs=L[::2]):
    lines = ["# Exported from SXRD Plotter", "# X column: L", "# Traces: 1", "",
             f"# Trace: S{scan}_ROI5 (scan={scan}, y=[bg] ROI5)", f"L\tS{scan}_ROI5"]
    lines += [f"{x:.4f}\t{100 + i}" for i, x in enumerate(xs)]
    path.write_text("\n".join(lines) + "\n")
    return path


# ------------------------------------------------------------------ SPEC files and HKL files
def test_spec_file_columns_rows_and_motors(tmp_path):
    scans = specfile.parse_spec_file(write_spec(tmp_path / "s.spec"))
    s = scans[464]
    assert s.columns == ["H", "K", "L", "Epoch", "ROI5"] and s.npoints == 21          # junk rows skipped
    assert s.motor_positions == {"Two Theta": 10.0, "Theta": 5.0, "Chi": 90.0, "Phi": -3.5, "Mu": 0.4}
    assert np.allclose(s.data["L"], L) and scans[465].npoints == 2 and scans[465].motor_positions == {}


def test_hkl_appender_smoother_combiner_chain(tmp_path):
    scans = specfile.parse_spec_file(write_spec(tmp_path / "s.spec"))
    header, traces = hkl.parse_exported_file(write_export(tmp_path / "exp.txt"))
    snap = {c: {"enabled": True, "tol": 0.05, "step": 1.0} for c in "HK"}
    snap["L"] = {"enabled": False, "tol": 0.05, "step": 1.0}
    assert hkl.write_output(tmp_path / "hkl.txt", header, traces, scans, snap)[0] == 1
    (t,) = hkl.parse_hkl_file(tmp_path / "hkl.txt")
    assert np.all(t["H"] == 0) and np.all(t["K"] == 2)                             # snapped
    assert np.allclose(t["L"], L[::2]) and np.allclose(t["I"], 100 + np.arange(11))
    # smoothing: a straight line is unchanged, excluded zone is NaN
    sm = hkl.smooth_trace(t["L"], t["I"], "Savitzky-Golay", 5, 2, [(0.95, 1.05)])
    inside = np.abs(t["L"] - 1.0) < 0.06
    assert np.all(np.isnan(sm[inside])) and np.allclose(sm[~inside], t["I"][~inside])
    # combine two files: dedupe drops the copy, sort by L
    write_export(tmp_path / "exp2.txt")
    hkl.write_output(tmp_path / "hkl2.txt", header, traces, scans, snap)
    rows, log = hkl.combine_hkl_files([tmp_path / "hkl.txt", tmp_path / "hkl2.txt"], sort="By L", dedupe=True)
    assert len(rows) == 11 and any("Deduped" in x for x in log)
    hkl.write_hkl_rows(tmp_path / "all.txt", rows)
    text = (tmp_path / "all.txt").read_text().splitlines()
    assert text[0] == "H\tK\tL\tI" and text[1] == "0\t2\t0.5\t100"
    assert hkl.write_genx_file(tmp_path / "genx.dat", [t]) == 11


# ------------------------------------------------------------------ images and background
def make_image(rng, peak=(40, 30)):
    y, x = np.mgrid[:64, :80]
    img = 10.0 + rng.normal(0, 0.5, (64, 80))
    img += 500 * np.exp(-((x - peak[0]) ** 2 + (y - peak[1]) ** 2) / 8.0)
    return img


def test_detector_files_rois_and_background(tmp_path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(0)
    for scan in (28, 29):
        for im in range(3):
            with h5py.File(tmp_path / f"run_{scan:03d}_{im:03d}_data_000001.h5", "w") as f:
                f["entry/data/data"] = make_image(rng)[None]
    (tmp_path / "notes.txt").write_text("x")
    files = images.scan_directory(str(tmp_path))
    assert len(files) == 6 and images.parse_filename(files[0]["path"])["scan"] == 28
    img = images.load_image(files[0]["path"], "h5")
    sig = images.ROI(35, 25, 10, 10, "signal", "ROI1")
    bgs = [images.ROI(0, 0, 20, 20, "background", "ROI2"), images.ROI(60, 44, 20, 20, "background", "ROI3")]
    for name, fn in images.BG_METHODS.items():
        sub = fn(img, [sig], bgs)
        assert abs(sub[:20, :20].mean()) < 0.3, name                                   # flat 10 background removed
        assert sig.extract(sub).sum() == pytest.approx(sig.extract(img - 10).sum(), rel=0.02), name
    cfg = images.ROIConfig()
    cfg.default_rois = [sig] + bgs
    cfg.set_override(28, 1, [sig.copy(), bgs[0].copy()])
    cfg.save(str(tmp_path / "rois.txt"))
    back = images.ROIConfig()
    back.load(str(tmp_path / "rois.txt"))
    assert [r.to_tuple() for r in back.default_rois] == [r.to_tuple() for r in cfg.default_rois]
    assert len(back.get_rois(28, 1)) == 2 and len(back.get_rois(28, 2)) == 3


def test_background_finder_avoids_signal_and_hot_spots():
    rng = np.random.default_rng(1)
    stack = np.array([make_image(rng, peak=(40 + i, 30)) for i in range(6)])
    stack[:, 5:8, 70:73] = 5000                                                        # a hot spot in every image
    stats = bgfind.compute_stats(stack, use_gpu=False)
    assert stats["backend"] == "numpy" and stats["shape"] == (64, 80)
    mask = bgfind.make_exclusion_mask(stats)
    assert mask[6, 71]
    rois = bgfind.find_bg_rois_near_point(stats, (42, 30), mask, search_radius=30, signal_exclude_radius=8,
                                          box_size=8, n_boxes=3)
    assert rois
    for x, y, bw, bh in rois:                                                         # boxes (x, y, w, h)
        assert np.hypot(x + bw / 2 - 42, y + bh / 2 - 30) > 8 and mask[y:y + bh, x:x + bw].mean() <= 0.3
        assert not (x < 73 and x + bw > 70 and y < 8 and y + bh > 5)                  # not on the hot spot


# ------------------------------------------------------------------ peak fits and quick CTR models
def test_peak_fit_with_a_centre_hint():
    x = np.linspace(-1, 1, 81)
    y = peaks.gaussian(x, 200, 0.1, 0.08, 20, 5) + np.random.default_rng(2).normal(0, 1, x.size)
    fit = peaks.fit_peak(x, y, "Gaussian + linear bg", center_hint=0.05, fit_window=0.6)
    assert fit is not None                                                              # ptp() is gone in NumPy 2
    popt, _, model, xu, yu = fit
    assert popt[1] == pytest.approx(0.1, abs=0.005) and abs(popt[2]) == pytest.approx(0.08, rel=0.05)
    area = peaks.integrate_peak(xu, yu, popt, model)
    assert area == pytest.approx(200 * 0.08 * np.sqrt(2 * np.pi), rel=0.05)
    assert peaks.numerical_area(x, y) == pytest.approx(area, rel=0.15)


def test_bgsub_sum_files_for_the_plotter(tmp_path):
    (tmp_path / "sums.txt").write_text("scan\timage\tROI1\n28\t0\t5\n28\t1\t7\n29\t0\t9\n")
    data, header = peaks.parse_bgsub_file(tmp_path / "sums.txt")
    assert header == ["scan", "image", "ROI1"] and list(data[28]["ROI1"]) == [5, 7] and list(data[29]["image"]) == [0]


def test_quick_ctr_models():
    Ls = np.array([0.5, 1.5, 0.98])
    for name, m in ctrmodels.CTR_MODELS.items():
        assert len(m["params"]) == len(m["defaults"]) == len(m["bounds_lo"]) == len(m["bounds_hi"]), name
        I = m["func"](Ls, m["defaults"])
        assert np.all(np.isfinite(I)) and np.all(I > 0), name
    sub = ctrmodels.CTR_MODELS["1: Substrate only"]
    I = sub["func"](np.array([0.5, 0.99]), [1.0, 3.248, 0.99, 1.0, 0.0])           # beta: layer-to-layer amplitude ratio
    assert I[1] > 100 * I[0]                                                            # rises toward the Bragg peak


def test_datatools_logic_imports_without_qt():
    code = ("import sys, ctrfit.datatools.images, ctrfit.datatools.bgfind, ctrfit.datatools.hkl, "
            "ctrfit.datatools.peaks, ctrfit.datatools.ctrmodels, ctrfit.datatools.h5meta; "
            "print([m for m in sys.modules if m.split('.')[0] in ('PyQt5', 'PySide6')])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(__file__)))
    assert out.stdout.strip() == "[]", out.stderr


# ------------------------------------------------------------------ the window
@pytest.fixture
def qapp():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def no_dialogs(monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    said = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(lambda *a, k=kind, **kw: said.append((k, a[2]))))
    return said


def test_data_tools_window(qapp, no_dialogs, tmp_path, monkeypatch):
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent
    from ctrfit.app.data_tools_window import TOOLS, DataToolsWindow
    monkeypatch.setenv("CTRFIT_SETTINGS_DIR", str(tmp_path / "settings"))
    w = DataToolsWindow()
    w.show()
    qapp.processEvents()
    assert [w.tabs.tabText(i) for i in range(w.tabs.count())] == [t[1] for t in TOOLS]
    w.tools["plotter"]._launch_ctr_fit()                                               # the plotter's CTR button
    assert w.tabs.currentWidget() is w.tools["ctr_fit"]
    # the image tool filters arrow keys for the whole application, but only while it is on screen
    bg = w.tools["bgsub"]
    calls = []
    monkeypatch.setattr(bg, "_handle_arrow", lambda *a: calls.append(a) or True)
    key = QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.NoModifier)
    assert not bg._key_filter.eventFilter(w.tabs, key) and not calls
    w.show_tool("bgsub")
    qapp.processEvents()
    assert bg._key_filter.eventFilter(w.tabs, key) and calls
    # the combiner tab end to end
    scans = specfile.parse_spec_file(write_spec(tmp_path / "s.spec"))
    header, traces = hkl.parse_exported_file(write_export(tmp_path / "exp.txt"))
    hkl.write_output(tmp_path / "hkl.txt", header, traces, scans)
    c = w.show_tool("combiner")
    c._files = [str(tmp_path / "hkl.txt")]
    c._out.setText(str(tmp_path / "combined.txt"))
    c._combine()
    assert c._status_label.text().startswith("Wrote 11 rows") and not no_dialogs                # status bar, no pop-up
    assert len((tmp_path / "combined.txt").read_text().splitlines()) == 12
    c._files = []
    c._combine()
    assert no_dialogs[-1][0] == "warning"                                              # errors still pop up
    w.close()
    assert {"sxrd_bgsub_settings.ini", "sxrd_plotter_settings.ini", "sxrd_ctr_settings.ini"} <= set(
        os.listdir(settings_dir()))


def test_launcher_entry_point_runs_offscreen():
    root = os.path.dirname(os.path.dirname(__file__))
    code = ("import sys; sys.argv=['x']; from PyQt5.QtCore import QTimer; from PyQt5.QtWidgets import QApplication; "
            "app = QApplication(sys.argv); QTimer.singleShot(300, app.quit); "
            "import runpy; runpy.run_path('ctr_data_tools.py', run_name='__main__')")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", CTRFIT_SETTINGS_DIR=settings_dir())
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root, env=env, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]


def test_background_finder_opens_from_the_subtraction_tab(qapp, no_dialogs, tmp_path):
    h5py = pytest.importorskip("h5py")
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication
    from ctrfit.app.datatools.bg_finder import BgFinderWindow
    from ctrfit.app.datatools.bgsub import MainWindow
    rng = np.random.default_rng(3)
    for im in range(2):
        with h5py.File(tmp_path / f"run_028_{im:03d}_data_000001.h5", "w") as f:
            f["entry/data/data"] = make_image(rng)[None]
    w = MainWindow()
    w._dir.setText(str(tmp_path))
    w._reload()
    assert len(w._files) == 2
    seen = []

    def finder_saves_and_closes():
        (f,) = [x for x in QApplication.topLevelWidgets() if isinstance(x, BgFinderWindow) and x.isVisible()]
        seen.append(f._save_to)
        cfg = images.ROIConfig()
        cfg.default_rois = [images.ROI(1, 2, 8, 8, "background", "BG1")]
        cfg.save(f._save_to)
        f.close()
    QTimer.singleShot(200, finder_saves_and_closes)
    w._launch_bg_finder()                                                              # modal until the finder closes
    assert seen and not os.path.exists(seen[0])                                        # temp file read and removed
    bg = [r for r in w._roi_config.default_rois if r.roi_type == "background"]
    assert [(r.x, r.y, r.w, r.h) for r in bg] == [(1, 2, 8, 8)]
    w._save_settings()


def test_tools_follow_the_window_standard(qapp, no_dialogs, monkeypatch, tmp_path):
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QPushButton, QShortcut
    from ctrfit.app.data_tools_window import DataToolsWindow
    from ctrfit.app.datatools.common import PRIMARY_STYLE
    monkeypatch.setenv("CTRFIT_SETTINGS_DIR", str(tmp_path / "settings"))
    w = DataToolsWindow()
    w.show()
    qapp.processEvents()
    assert qapp.palette().window().color().lightness() > 150                          # the system (light) theme
    for key, tool in w.tools.items():
        primary = [b for b in tool.findChildren(QPushButton) if b.styleSheet() == PRIMARY_STYLE]
        assert primary or key == "meta", key                                         # a blue main action (the viewer has none)
        assert tool._status_label is not None and tool._status_label.text(), key        # a first hint in the status bar
        for sc in tool.findChildren(QShortcut):
            assert sc.context() == Qt.WidgetWithChildrenShortcut, key                   # shortcuts stay in their tab
    # a dialog of the plotter reports in the plotter's status bar
    plotter = w.tools["plotter"]
    plotter._peak_fit_dialog.say("fitted 3 scans")
    assert plotter._status_label.text() == "fitted 3 scans" and not no_dialogs
    # the ROI copy grid is wired to the window
    bg = w.tools["bgsub"]
    calls = []
    monkeypatch.setattr(bg, "_apply_rois_scan", lambda: calls.append("scan"))
    bg._roi_w._btn_apply_scan.clicked.disconnect()
    bg._roi_w.set_apply_scan_callback(bg._apply_rois_scan)
    bg._roi_w._btn_apply_scan.click()
    assert calls == ["scan"]
    w.close()

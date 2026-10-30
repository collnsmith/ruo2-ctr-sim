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


# ------------------------------------------------------------------ background subtraction: speed, undo, batch
def _old_display(img, log, vmin, vmax, lut):
    """The viewer's display transform before the speed-up (reference)."""
    img = img.copy()
    if log:
        img = np.where(img > 0, np.log10(img + 1), 0)
    pos = img[img > 0] if np.any(img > 0) else img.ravel()
    vmin = (np.log10(vmin + 1) if log and vmin > 0 else vmin) if vmin is not None else np.percentile(pos, 1)
    vmax = (np.log10(vmax + 1) if log and vmax > 0 else vmax) if vmax is not None else np.percentile(pos, 99.5)
    idx = np.clip((img - vmin) / (vmax - vmin + 1e-10) * 255, 0, 255).astype(np.uint8)
    return lut[idx]


def test_display_and_polynomial_background_match_the_old_code():
    rng = np.random.default_rng(4)
    img = rng.poisson(30, (120, 200)).astype(float)
    img[5, 5] = -3
    for log, lims in ((True, (None, None)), (False, (None, None)), (True, (5.0, 200.0))):
        new = images.display_rgb(img, log, *lims, "gray", exact=True)                   # gray: RGB = the level
        with np.errstate(invalid="ignore"):
            old = _old_display(img, log, *lims, images.colormap_lut("gray"))
        assert np.array_equal(new, old)                                               # batch PNGs unchanged
        view = images.display_rgb(img, log, *lims, "gray")                              # the viewer (float32)
        assert np.abs(view.astype(int) - old.astype(int)).max() <= 1
    big = rng.poisson(30, (600, 1000)).astype(float)
    fast = images.display_limits(np.log10(big + 1))
    exact = images.display_limits(np.log10(big + 1), exact=True)
    assert np.allclose(fast, exact, rtol=0.01)                                         # subsampled percentiles
    # the separable polynomial surface equals the term-by-term one
    yy, xx = np.mgrid[:120, :200]
    surf = 3 + 0.01 * yy + 0.02 * xx + 1e-4 * yy * xx + 2e-4 * xx ** 2
    bgs = [images.ROI(0, 0, 40, 120, "background"), images.ROI(160, 0, 40, 120, "background")]
    sub = images.subtract_bg_polyfit(surf, [], bgs)
    assert np.abs(sub).max() < 1e-9


def test_image_cache_reads_ahead_and_sees_changed_files(tmp_path):
    h5py = pytest.importorskip("h5py")
    paths = []
    for im in range(4):
        p = tmp_path / f"run_028_{im:03d}_data_000001.h5"
        with h5py.File(p, "w") as f:
            f["entry/data/data"] = np.full((1, 4, 5), float(im))
        paths.append({"path": str(p), "ext": "h5"})
    cache = images.ImageCache(maxsize=2)
    cache.prefetch(paths[1:3])
    assert cache.get(paths[1]["path"], "h5")[0, 0] == 1 and cache.get(paths[2]["path"], "h5")[0, 0] == 2
    a = cache.get(paths[0]["path"], "h5")
    assert a is cache.get(paths[0]["path"], "h5") and not a.flags.writeable                # cached, read-only
    import time
    time.sleep(0.01)
    with h5py.File(paths[0]["path"], "w") as f:                                         # rewritten (a transfer)
        f["entry/data/data"] = np.full((1, 4, 5), 9.0)
    assert cache.get(paths[0]["path"], "h5")[0, 0] == 9
    assert len(cache._items) <= 2
    with pytest.raises(Exception):
        cache.get(str(tmp_path / "missing.h5"), "h5")


def test_process_image_and_edit_history():
    img = np.full((20, 30), 10.0)
    img[5:10, 5:10] += 100
    rois = [images.ROI(5, 5, 5, 5, "signal", "S"), images.ROI(20, 10, 8, 8, "background", "B")]
    mask = np.zeros_like(img, bool)
    mask[6, 6] = True
    sub, sums = images.process_image(img, rois, images.BG_METHODS["Average"], mask)
    assert sums["S"] == pytest.approx(24 * 100) and sums["B"] == pytest.approx(0) and sub[6, 6] == 0
    cfg = images.ROIConfig()
    cfg.default_rois = [rois[0].copy()]
    hist = images.EditHistory()
    hist.checkpoint(images.EditState(cfg, rois, mask))
    hist.checkpoint(images.EditState(cfg, rois, mask))                                  # same state kept once
    assert len(hist.undo_stack) == 1
    moved = [r.copy() for r in rois]
    moved[0].x = 12
    back = hist.undo(images.EditState(cfg, moved, mask))
    cur, m = back.apply_to(cfg)
    assert cur[0].x == 5 and m[6, 6] and hist.can_redo
    fwd = hist.redo(images.EditState(cfg, cur, m))
    assert fwd.apply_to(cfg)[0][0].x == 12
    assert hist.redo(images.EditState(cfg, moved, mask)) is None


@pytest.fixture
def bgsub_window(qapp, tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    from ctrfit.app.datatools.bgsub import MainWindow
    monkeypatch.setenv("CTRFIT_SETTINGS_DIR", str(tmp_path / "settings"))
    rng = np.random.default_rng(5)
    folder = tmp_path / "imgs"
    folder.mkdir()
    for im in range(3):
        with h5py.File(folder / f"run_028_{im:03d}_data_000001.h5", "w") as f:
            f["entry/data/data"] = make_image(rng)[None]
    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    w._dir.setText(str(folder))
    w._reload()
    w._roi_w.set_rois([images.ROI(30, 20, 20, 20, "signal", "S"), images.ROI(0, 0, 15, 15, "background", "B")])
    w._on_rois_changed()
    qapp.processEvents()
    yield w
    w.close()


def _drag(qapp, cv, start, steps, dx=1.0):
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent

    def ev(kind, x, y, btns):
        qapp.sendEvent(cv, QMouseEvent(kind, QPointF(x, y), Qt.LeftButton, btns, Qt.NoModifier))
    x0, y0 = start
    ev(QEvent.MouseButtonPress, x0, y0, Qt.LeftButton)
    for k in range(1, steps + 1):
        ev(QEvent.MouseMove, x0 + k * dx, y0, Qt.LeftButton)
    ev(QEvent.MouseButtonRelease, x0 + steps * dx, y0, Qt.NoButton)
    qapp.processEvents()


def test_roi_drag_is_throttled_and_undoable(qapp, bgsub_window, no_dialogs, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    w = bgsub_window
    cv = w._cv_orig
    calls = []
    real = w._update_sub
    monkeypatch.setattr(w, "_update_sub", lambda: (calls.append(1), real()))
    renders = []
    real_render = w._cv_sub._render
    monkeypatch.setattr(w._cv_sub, "_render", lambda: (renders.append(1), real_render()))
    x0, y0 = cv._img_to_widget(40, 30)                                                   # inside S
    _drag(qapp, cv, (x0, y0), 40, dx=cv._scale)                                         # 40 moves of one pixel
    moved = w._roi_w.get_rois()[0].x
    assert abs(moved - 70) <= 1                                                         # the ROI followed the mouse
    assert len(calls) <= 3 and len(renders) <= 3                                        # not once per mouse move
    # undo / redo of the drag (Ctrl+Z / Ctrl+Y)
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    w.setFocus()
    QTest.keyClick(w, Qt.Key_Z, Qt.ControlModifier)
    assert w._roi_w.get_rois()[0].x == 30 and w._cv_orig._rois[0].x == 30
    QTest.keyClick(w, Qt.Key_Y, Qt.ControlModifier)
    assert w._roi_w.get_rois()[0].x == moved
    # adding a ROI and copying ROIs to all images are undone too
    w._roi_w._add_roi()
    assert len(w._roi_w.get_rois()) == 3
    w.undo()
    assert len(w._roi_w.get_rois()) == 2
    w._roi_w._btn_apply.click()
    assert len(w._roi_config.overrides) == 3
    w.undo()
    assert len(w._roi_config.overrides) < 3
    w.undo()
    w.undo()
    w.undo()
    assert "Nothing to undo" in w._status_label.text()


def test_mask_paint_undo(qapp, bgsub_window, no_dialogs):
    w = bgsub_window
    w._chk_auto.setChecked(False)
    w._mask = np.zeros((64, 80), bool)
    w._cv_orig.set_mask_data(w._mask)
    w._chk_mask_paint.setChecked(True)
    x0, y0 = w._cv_orig._img_to_widget(60.2, 10.2)
    _drag(qapp, w._cv_orig, (x0, y0), 3, dx=w._cv_orig._scale)
    assert w._mask.sum() >= 3
    w.undo()
    assert w._mask.sum() == 0 and w._cv_orig._mask_data is w._mask
    w.redo()
    assert w._mask.sum() >= 3


def test_batch_png_images_are_optional(qapp, bgsub_window, no_dialogs, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog
    w = bgsub_window
    for save, name in ((False, "a_sums.txt"), (True, "b_sums.txt")):
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, n=name, **k: (str(tmp_path / n), "")))
        w._chk_save_png.setChecked(save)
        w._run_batch()
        lines = (tmp_path / name).read_text().splitlines()
        assert lines[0].split("\t") == ["scan", "image", "B", "S", "S_err"]
        assert len([l for l in lines[1:] if l and not l.startswith("#")]) == 3
        folder = tmp_path / (name[:-4] + "_images")
        assert folder.exists() == save
        if save:
            assert len(list(folder.glob("*_sub.png"))) == 3


def test_batch_saves_roi_positions(qapp, bgsub_window, no_dialogs, tmp_path, monkeypatch):
    """The batch sums file ends with its own ROI positions section: one row per scan for a ROI that
    stays put, one per image for a ROI that moves; every reader of the table skips it."""
    from PyQt5.QtWidgets import QFileDialog
    from ctrfit.datatools import processing
    from ctrfit.datatools.peaks import parse_bgsub_file
    from ctrfit.echem.external import parse_roi_sums
    w = bgsub_window
    for im, x in ((0, 30), (1, 34), (2, 30)):
        w._roi_config.set_override(28, im, [images.ROI(x, 22, 20, 20, "signal", "S"),
                                            images.ROI(0, 0, 15, 15, "background", "B")])
    path = tmp_path / "pos_sums.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))
    w._run_batch()
    text = path.read_text()
    assert images.ROI_SECTION in text
    pos = images.read_roi_positions(path)
    assert pos[(28, "all", "B")] == ("background", 0, 0, 15, 15) and (28, "all", "S") not in pos
    assert images.roi_position(pos, 28, 1, "S") == ("signal", 34, 22, 20, 20)
    assert {k[1] for k in pos if k[2] == "S"} == {0, 1, 2} and pos[(28, 2, "S")] == ("signal", 30, 22, 20, 20)
    assert images.roi_position(pos, 28, 2, "B") == ("background", 0, 0, 15, 15)
    # the sums table reads as before
    data, header = parse_bgsub_file(str(path))
    assert header == ["scan", "image", "B", "S", "S_err"] and data[28]["S"].size == 3
    assert parse_roi_sums(text)[28]["image"].tolist() == [0, 1, 2]
    assert processing.sum_files(str(tmp_path)) == [str(path)] and processing.load_sums([str(path)])[28]["S"].size == 3


# ------------------------------------------------------------------ plotter: potential next to the scans
def test_potential_helpers():
    from ctrfit.datatools import potential as pot
    assert pot.time_zero([np.nan, 5.0, 6.0]) == 5.0
    with pytest.raises(ValueError):
        pot.time_zero([np.nan])
    ec = pot.hand_cv(0.0, 10.0, 0.7, 0.0, 10.0, 1, "up")                                # 10 s hold, 70 s up, 70 s down
    assert ec.time[-1] == pytest.approx(150.0) and ec.potential.max() == pytest.approx(0.7)
    E = pot.potential_at(ec, [0.0, 105.0, 125.0, 500.0], offset=20.0)                   # EC t = -20, 85, 105, 480
    assert np.isnan(E[0])
    assert E[1] == pytest.approx(0.7 - (85 - 80) * 0.01, abs=1e-6) and E[2] == pytest.approx(0.45, abs=1e-6)
    assert np.isnan(E[3])
    t, Ev = pot.trace_between(ec, 20.0, 50.0, 100.0)
    assert t[0] == 50.0 and t[-1] == 100.0 and Ev[0] == pytest.approx(0.2, abs=1e-6)
    for bad in ((0, 0, 0.7, 0, 0, 1, "up"), (0, 0, 0.0, 0.7, 5, 1, "up"), (0, 0, 0.7, 0, 5, 1, "sideways")):
        with pytest.raises(ValueError):
            pot.hand_cv(*bad)


def test_plotter_potential_axis_and_time_zero(qapp, no_dialogs, tmp_path, monkeypatch):
    from ctrfit.app.datatools.plotter import MainWindow
    from ctrfit.echem.synthetic import write_experiment
    monkeypatch.setenv("CTRFIT_SETTINGS_DIR", str(tmp_path / "settings"))
    spec, mpt, truth = write_experiment(tmp_path / "exp")
    w = MainWindow()
    w.show()
    w._load_spec_file(str(spec))
    for i in range(w._scan_list.count()):
        w._scan_list.item(i).setSelected("loopscan" in w._scan_list.item(i).text())
    qapp.processEvents()
    w._col_sel.set_selection("Epoch", ["Detector"], "Monitor")
    w._update_plot()
    w._zero_mode.setCurrentIndex(1)                                                      # t = 0 at each scan's start
    epoch = w._spec_scans[2].data["Epoch"]
    assert w._last_traces[0]["t"][0] == pytest.approx(0.0)
    # EC-Lab file, aligned by the offset: E on a second axis over the data's time range
    w._ec_source.setCurrentText("EC-Lab file")
    w._ec_path.setText(str(mpt))
    w._rebuild_ec()
    w._ec_offset.setValue(truth["cv_delay"] - 1.1)      # CV 60 s after the scan's #D; the first point ends 1.1 s in
    w._update_plot()
    assert len(w._canvas.fig.axes) == 2 and "Ag/AgCl" in w._canvas.fig.axes[1].get_ylabel()
    w._chk_show_E.setChecked(False)                                                      # hide the potential trace
    assert len(w._canvas.fig.axes) == 1 and w._last_traces[0]["E"] is not None          # still known per point
    w._chk_show_E.setChecked(True)
    assert len(w._canvas.fig.axes) == 2
    E = w._last_traces[0]["E"]
    assert np.nanmin(E) == pytest.approx(truth["lower"], abs=0.01) and np.nanmax(E) == pytest.approx(truth["upper"], abs=0.01)
    # alignment by SPEC point number: the first sweep (after the 300 s hold) starts at point #326
    w._ec_offset.setValue(0.0)
    w._align_point.setValue(326)
    w._align_marks.setCurrentIndex(0)
    w._align_at_point()
    assert w._ec_offset.value() == pytest.approx(truth["cv_delay"] - 1.1, abs=1.5)
    w._align_marks.setCurrentIndex(1)                                                    # or the record start there
    w._align_point.setValue(54)
    w._align_at_point()
    assert w._ec_offset.value() == pytest.approx(54 * 1.1, abs=1e-6)
    w._align_point.setValue(10 ** 6)
    w._align_at_point()
    assert "not in the scan" in no_dialogs[-1][1]
    w._ec_offset.setValue(truth["cv_delay"] - 1.1)
    w._update_plot()
    # intensity vs potential: x is the potential, no second axis (the old one is removed)
    w._ec_mode.setCurrentIndex(1)
    assert len(w._canvas.fig.axes) == 1 and w._canvas.ax.get_xlabel().startswith("E (")
    x = w._canvas.ax.lines[0].get_xdata()
    assert np.nanmax(x) == pytest.approx(truth["upper"], abs=0.01)
    # the intensity drops at high potential (the synthetic H2O -> OH -> O path)
    y = w._canvas.ax.lines[0].get_ydata()
    assert np.nanmean(y[x > 1.3]) < 0.3 * np.nanmean(y[x < 0.6])
    # a sweep defined by hand, export, settings round trip
    w._ec_source.setCurrentText("Defined by hand")
    w._cv_upper.setValue(1.2)
    w._rebuild_ec()
    assert w._ec.potential.max() == pytest.approx(1.2)
    from PyQt5.QtWidgets import QFileDialog
    out = tmp_path / "ive.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    w._export_potential_data()
    lines = out.read_text().splitlines()
    assert any(l.startswith("t_s\tE_V_vs_Ag/AgCl") for l in lines)
    w.close()
    w2 = MainWindow()
    assert w2._ec_source.currentText() == "Defined by hand" and w2._cv_upper.value() == pytest.approx(1.2)
    assert w2._zero_mode.currentIndex() == 1 and w2._ec is not None
    w2.close()


# ------------------------------------------------------------------ plotter: export figure
def test_figure_spec_style_and_save(tmp_path):
    from matplotlib.figure import Figure
    from ctrfit.datatools import figure as fg
    f = Figure()
    ax = f.add_subplot(111)
    ax.plot([1, 2, 3], [1, 4, 9], "o-", color="#d62728", label="S1 Det")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("I")
    ax.set_yscale("log")
    a2 = ax.twinx()
    a2.plot([1, 3], [0.1, 0.7], "--", color="k", label="E")
    a2.set_ylabel("E (V)")
    spec = fg.FigureSpec.from_axes(f.axes)
    assert [(t.label, t.axis, t.linestyle, t.marker, t.color) for t in spec.traces] == [
        ("S1 Det", "left", "-", "o", "#d62728"), ("E", "right", "--", "none", "#000000")]
    assert spec.ylog and spec.y2label == "E (V)" and spec.has_right
    st = fg.FigureStyle.from_spec(spec)
    st.width, st.height, st.dpi = 4.0, 3.0, 100
    st.traces[0].label, st.traces[0].color, st.traces[1].visible = "Intensity", "#00ff00", False
    for fmt in fg.FORMATS:
        path = fg.save(spec, st, tmp_path / f"fig.{fmt}")
        assert path.stat().st_size > 1000, fmt
    from PIL import Image
    assert Image.open(tmp_path / "fig.png").size == (400, 300)                          # size x DPI
    rendered = fg.render(Figure(figsize=(4, 3)), spec, st)
    assert len(rendered.axes) == 1                                                       # hidden right trace: no twin
    assert rendered.axes[0].get_lines()[0].get_color() == "#00ff00"
    assert rendered.axes[0].get_legend().get_texts()[0].get_text() == "Intensity"
    back = fg.FigureStyle.from_json(st.to_json())
    assert back == st
    # a refresh keeps edited traces (matched by their plot label) and the text; a new plot takes new labels
    again = fg.FigureStyle.from_spec(spec, base=st, keep_text=True, keep_traces=True)
    assert again.traces[0].label == "Intensity" and again.traces[0].color == "#00ff00" and not again.traces[1].visible
    fresh = fg.FigureStyle.from_spec(spec, base=st)
    assert fresh.width == 4.0 and fresh.traces[0].color == "#d62728"                     # look kept, traces new
    st.xmin = "oops"
    with pytest.raises(ValueError):
        fg.render(Figure(), spec, st)
    with pytest.raises(ValueError):
        fg.FigureSpec.from_axes(Figure().subplots(1, 1).figure.axes)


def test_plotter_export_window(qapp, no_dialogs, tmp_path, monkeypatch):
    from ctrfit.app.datatools.plotter import MainWindow
    from ctrfit.echem.synthetic import write_experiment
    monkeypatch.setenv("CTRFIT_SETTINGS_DIR", str(tmp_path / "settings"))
    spec, mpt, truth = write_experiment(tmp_path / "exp")
    w = MainWindow()
    w.show()
    assert w._export_figure() is None and "Nothing to export" in no_dialogs[-1][1]     # empty plot
    w._load_spec_file(str(spec))
    for i in range(w._scan_list.count()):
        w._scan_list.item(i).setSelected("loopscan" in w._scan_list.item(i).text())
    qapp.processEvents()
    w._col_sel.set_selection("Epoch", ["Detector"], "Monitor")
    w._zero_mode.setCurrentIndex(1)                                                      # hand CV from t = 0
    w._ec_source.setCurrentText("Defined by hand")
    w._rebuild_ec()
    working_title = w._canvas.ax.get_title()
    ex = w._export_figure()
    qapp.processEvents()
    assert len(ex.style.traces) == 2 and ex.style.traces[1].axis == "right"
    # edit in the window: preset, a label, a colour, hide nothing; the working plot is untouched
    ex._apply_preset(ex._preset.findText("Paper, one column (8.5 cm)"))
    ex._title.setText("RuO2 (0 1 1.5)")
    ex._rows[0][1].setText("I")
    ex._rows[0][2].set_color("#123456")
    ex._units.setCurrentText("cm")
    assert ex._w.value() == pytest.approx(8.5, abs=0.01)
    ex._render()
    out = ex.save_to(str(tmp_path / "final"))                                            # format from the combo
    assert out.endswith(".png")
    from PIL import Image
    assert Image.open(out).size == (int(8.5 / 2.54 * 600), int(6.5 / 2.54 * 600))         # exact size kept
    assert w._canvas.ax.get_title() == working_title                                     # working view unchanged
    # refresh after the working plot changed keeps the edits; the style is remembered for next time
    w._ec_source.setCurrentText("None")
    w._rebuild_ec()
    ex.refresh_from_plot()
    assert len(ex.style.traces) == 1 and ex.style.traces[0].label == "I" and ex.style.title == "RuO2 (0 1 1.5)"
    ex.close()
    ex2 = w._export_figure()
    assert ex2.style.width == pytest.approx(8.5 / 2.54) and ex2.style.dpi == 600
    ex2.close()
    w.close()


def test_apply_rois_copies_positions_unless_masters_are_kept():
    cfg = images.ROIConfig()
    sig = images.ROI(10, 10, 5, 5, "signal", "S")
    bg = images.ROI(30, 10, 5, 5, "background", "B")
    bg.follow_id, bg.offset_dx, bg.offset_dy = 0, 20, 0
    cfg.overrides[(1, 0)] = [images.ROI(2, 2, 5, 5, "signal", "S"), images.ROI(22, 2, 5, 5, "background", "B")]
    cfg.overrides[(9, 9)] = [images.ROI(1, 1, 1, 1, "signal", "X")]                     # not in keys: untouched
    keys = [(1, 0), (1, 1)]
    assert images.apply_rois(cfg, keys, [sig, bg], set_default=True) == 2
    assert [r.to_tuple() for r in cfg.overrides[(1, 0)]] == [(10, 10, 5, 5), (30, 10, 5, 5)]   # exact copy
    assert [r.to_tuple() for r in cfg.default_rois] == [(10, 10, 5, 5), (30, 10, 5, 5)]
    assert cfg.overrides[(9, 9)][0].name == "X"
    cfg.overrides[(1, 0)][0].x, cfg.overrides[(1, 0)][0].y = 3, 4                        # tracked signal there
    images.apply_rois(cfg, keys, [sig, bg], keep_masters=True)
    r0 = cfg.overrides[(1, 0)]
    assert (r0[0].x, r0[0].y) == (3, 4) and r0[1].x == 3 + 20                            # follower placed from it
    assert cfg.overrides[(1, 1)][0].to_tuple() == (10, 10, 5, 5)


def test_apply_rois_to_all_and_to_scan_in_the_window(qapp, bgsub_window, no_dialogs, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    w = bgsub_window
    # every image already has its own (old) ROIs: visit them all
    for _ in range(2):
        w._nav(1)
    w._nav(-2)
    w._roi_config.overrides = {(28, i): [images.ROI(30, 20, 20, 20, "signal", "S"),
                                         images.ROI(0, 0, 15, 15, "background", "B")] for i in range(3)}
    w._load_current()
    rois = w._roi_w.get_rois()
    rois[0].x, rois[0].y = 44, 33                                                         # move the signal here
    w._roi_w._btn_apply.click()                                                           # All ROIs -> all images
    for i in range(3):
        assert w._roi_config.get_rois(28, i)[0].to_tuple() == (44, 33, 20, 20), i
    w._nav(1)
    assert w._roi_w.get_rois()[0].to_tuple() == (44, 33, 20, 20)                         # shown on the next image
    w._roi_w.get_rois()[0].x = 50
    w._roi_w._btn_apply_scan.click()                                                      # All ROIs -> this scan
    assert all(w._roi_config.get_rois(28, i)[0].x == 50 for i in range(3))
    w.undo()
    assert w._roi_config.get_rois(28, 0)[0].x == 44



def test_plotter_several_scans_time_zero_and_shifts(qapp, no_dialogs, tmp_path, monkeypatch):
    from ctrfit.app.datatools.plotter import MainWindow
    from ctrfit.echem.synthetic import write_experiment
    monkeypatch.setenv("CTRFIT_SETTINGS_DIR", str(tmp_path / "settings"))
    spec, mpt, truth = write_experiment(tmp_path / "exp")
    w = MainWindow()
    w.show()
    w._load_spec_file(str(spec))
    for i in range(w._scan_list.count()):
        w._scan_list.item(i).setSelected(True)                                           # scans 1 and 2
    qapp.processEvents()
    w._col_sel.set_selection("Epoch", ["Detector"], "")
    w._update_plot()
    e1, e2 = w._spec_scans[1].data["Epoch"], w._spec_scans[2].data["Epoch"]
    assert w._shift_table.rowCount() == 2
    w._zero_mode.setCurrentIndex(1)                                                      # every scan from 0
    assert [tr["t"][0] for tr in w._last_traces] == [pytest.approx(0.0), pytest.approx(0.0)]
    w._zero_mode.setCurrentIndex(2)                                                      # one clock, from scan 1
    assert w._last_traces[1]["t"][0] == pytest.approx(e2[0] - e1[0])
    w._zero_mode.setCurrentIndex(1)
    w._shift_table.item(1, 2).setText("12.5")                                            # shift scan 2
    assert w._last_traces[1]["t"][0] == pytest.approx(12.5) and w._last_traces[0]["t"][0] == pytest.approx(0)
    w._shift_table.item(1, 2).setText("abc")
    assert "not a number" in no_dialogs[-1][1]
    # t = 0 at a chosen point of a scan (e.g. where its CV started)
    w._shift_table.item(1, 1).setText("10")
    assert w._last_traces[1]["t"][10] == pytest.approx(12.5)
    assert w._last_traces[1]["t"][0] == pytest.approx(12.5 - (e2[10] - e2[0]))
    w._shift_table.item(1, 1).setText("100000")
    assert "not in the scan" in no_dialogs[-1][1]
    w._shift_table.item(1, 1).setText("0")
    # both scans against one hand-defined potential
    w._ec_source.setCurrentText("Defined by hand")
    w._rebuild_ec()
    w._ec_mode.setCurrentIndex(1)
    assert len(w._canvas.ax.lines) == 2 and all(np.isfinite(tr["E"]).any() for tr in w._last_traces)
    # 'At point' uses the scan selected in the shift table
    w._shift_table.selectRow(1)
    w._align_point.setValue(0)
    w._align_marks.setCurrentIndex(1)
    w._align_at_point()
    assert w._ec_offset.value() == pytest.approx(12.5)
    w.close()
    w2 = MainWindow()
    assert w2._scan_shift.get(2) == 12.5 and w2._scan_point.get(2) == 0
    w2.close()


def test_smoothing_takes_every_plotted_trace(qapp, no_dialogs, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog
    from ctrfit.app.datatools.plotter import MainWindow
    from ctrfit.echem.synthetic import write_experiment
    monkeypatch.setenv("CTRFIT_SETTINGS_DIR", str(tmp_path / "settings"))
    spec, mpt, truth = write_experiment(tmp_path / "exp")
    w = MainWindow()
    w.show()
    w._load_spec_file(str(spec))
    for i in range(w._scan_list.count()):
        w._scan_list.item(i).setSelected(True)
    qapp.processEvents()
    w._col_sel.set_selection("Epoch", ["Detector"], "")
    w._zero_mode.setCurrentIndex(1)
    w._ec_source.setCurrentText("Defined by hand")                                      # E on a second axis
    w._rebuild_ec()
    assert len(w._canvas.ax.get_lines()) == 2                                           # one per scan
    d = w._smooth_dialog
    d._grab_from_plot()
    assert len(d._sources) == 2 and "2 trace(s)" in d._source_info.text()               # not the potential
    d._apply_smoothing()
    assert len(d._smoothed) == 2
    d._send_to_main()
    lines = w._canvas.ax.get_lines()
    assert len(lines) == 2 and all(ln.get_label().endswith("(smoothed)") for ln in lines)
    assert len(w._canvas.fig.axes) == 2                                                  # potential axis kept
    # the smoothed traces still go against the potential
    w._ec_mode.setCurrentIndex(1)
    lines = w._canvas.ax.get_lines()
    assert len(lines) == 2 and len(w._canvas.fig.axes) == 1 and w._canvas.ax.get_xlabel().startswith("E (")
    E = np.concatenate([ln.get_xdata() for ln in lines])
    assert np.nanmin(E) >= w._ec.potential.min() - 1e-9 and np.nanmax(E) <= w._ec.potential.max() + 1e-9
    d._grab_from_plot()                                                                   # again: along time
    assert d._on_time and max(np.nanmax(src["x"]) for src in d._sources) > 100          # seconds, not volts
    d._apply_smoothing()
    w._ec_mode.setCurrentIndex(0)
    out = tmp_path / "smooth.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    d._export_smooth()
    assert out.read_text().count("# Trace:") == 2
    w._export_plot_data()                                                                # overlay with 2 traces
    assert out.read_text().count("# Trace:") == 2
    w.close()


def test_column_table_contrast_follows_the_theme(qapp):
    from PyQt5.QtGui import QColor, QPalette
    from ctrfit.app.datatools.plotter import ColumnSelectorWidget
    w = ColumnSelectorWidget()
    w.set_columns(["L", "Detector"], ["ROI1"])
    light = w._table.styleSheet()
    pal = QPalette(w.palette())
    pal.setColor(QPalette.Base, QColor(30, 30, 30))
    w.setPalette(pal)                                                                     # system turns dark
    qapp.processEvents()
    assert "gridline-color" in w._table.styleSheet() and light == ""
    assert w._table.item(2, 0).foreground().color().lightness() > 128                     # [bg] column readable


def test_plotter_refresh_rereads_the_files(qapp, no_dialogs, tmp_path, monkeypatch):
    from ctrfit.app.datatools.plotter import MainWindow
    monkeypatch.setenv("CTRFIT_SETTINGS_DIR", str(tmp_path / "settings"))
    spec = write_spec(tmp_path / "s.spec")
    sums = tmp_path / "sums"
    sums.mkdir()
    (sums / "a_sums.txt").write_text("scan\timage\tROI1\n" + "".join(f"464\t{i}\t{10 + i}\n" for i in range(5)))
    w = MainWindow()
    w.show()
    w._spec_path.setText(str(spec))
    w._bgsub_path.setText(str(sums))
    w.refresh_sources()
    for i in range(w._scan_list.count()):
        w._scan_list.item(i).setSelected(w._scan_list.item(i).text().startswith("464"))
    qapp.processEvents()
    w._col_sel.set_selection("L", ["[bg] ROI1"], "")
    w._update_plot()
    assert len(w._last_traces[0]["y"]) == 5
    # background subtraction writes more images, the beamtime adds a scan
    (sums / "a_sums.txt").write_text("scan\timage\tROI1\n" + "".join(f"464\t{i}\t{99}\n" for i in range(8)))
    with open(spec, "a") as f:
        f.write("\n#S 466  ascan  th 1 2 2 1\n#L Theta  Detector\n1 10\n2 20\n")
    w.refresh_sources()
    assert w._get_selected_scans() == [464] and w._col_sel.get_selection()[1] == ["[bg] ROI1"]
    assert len(w._last_traces[0]["y"]) == 8 and np.all(w._last_traces[0]["y"] == 99)
    assert 466 in w._spec_scans and "Refreshed" in w._status_label.text()
    w.close()


def test_logit_scale_and_errors_do_not_crash(qapp, no_dialogs, tmp_path, monkeypatch):
    import sys
    from ctrfit.app.datatools import common
    from ctrfit.app.datatools.plotter import MainWindow
    monkeypatch.setenv("CTRFIT_SETTINGS_DIR", str(tmp_path / "settings"))
    w = MainWindow()
    w.show()
    ax = w._canvas.ax
    ax.plot(np.arange(10), np.arange(10) * 100.0)                                     # includes 0: logit breaks
    ax.set_yscale("logit")                                                               # as the toolbar's axes dialog does
    w._canvas.draw()                                                                     # used to raise inside the paint
    assert ax.get_yscale() == "linear" and "back to linear" in w._status_label.text()
    # an unexpected error in a button or a timer is logged and shown; the program keeps running
    monkeypatch.setattr(common, "_log_file", None)
    old = sys.excepthook
    try:
        path = common.install_crash_handler(w)
        try:
            raise RuntimeError("boom in a slot")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())
        assert "boom in a slot" in open(path).read() and "boom in a slot" in no_dialogs[-1][1]
    finally:
        sys.excepthook = old
    w.close()

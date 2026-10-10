"""Smoke tests: the GUI completes one run offscreen, the notebook runs as a script."""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_gui_completes_one_run(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    import ctr_gui
    from ctr_gui import QtCore, QtWidgets

    monkeypatch.setattr(ctr_gui, "SETTINGS_FILE", tmp_path / "settings.ini")   # never touch the real one
    errors = []
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name,
                            staticmethod(lambda *a, **k: errors.append(a[1:3]) or QtWidgets.QMessageBox.Ok))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.No))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", staticmethod(lambda *a, **k: ("", False)))

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = ctr_gui.MainWindow()
    win.show()
    t0 = time.time()
    while (win.result is None or win.run_worker is not None) and time.time() - t0 < 90:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        time.sleep(0.01)
    assert not errors, errors
    assert win.result is not None, "first run did not finish"
    out = win.result
    assert out["rod"] is not None and out["compare"] is not None
    assert "scan" in out and win.rank_table.rowCount() > 0       # default: scan on when no settings.ini
    assert "ERROR" not in win.log.toPlainText()
    assert (tmp_path / "settings.ini").exists()
    win.close()
    app.processEvents()


def test_notebook_runs_as_script(tmp_path):
    src = (ROOT / "ctr_notebook.py").read_text(encoding="utf-8")
    assert "RUN_STUDY = True" in src
    runner = (
        "import sys; from pathlib import Path\n"
        f"path = Path({str(ROOT / 'ctr_notebook.py')!r})\n"
        "src = path.read_text(encoding='utf-8').replace('RUN_STUDY = True', 'RUN_STUDY = False')\n"
        "exec(compile(src, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})\n"
    )
    env = dict(os.environ, MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, "-c", runner], cwd=tmp_path, env=env, capture_output=True, text=True,
                       timeout=180)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "rank" in r.stdout and "Wavelength" in r.stdout
    assert "relaxation 0" not in r.stdout          # study skipped


def test_parameter_study_engine():
    from ctr_engine import parameter_study
    from ctr_params import DEFAULTS, STUDY_KEYS, format_study_rows, parse_comp, parse_study_rows
    rows = parse_study_rows("roughness_nm, 0.05, 0.4, 4; oh_dz, -0.2, 0.2, 3; h2o_dz, -0.2, 0.2, 3")
    assert parse_study_rows(format_study_rows(rows)) == rows
    assert "roughness_nm" in STUDY_KEYS and "L_step" not in STUDY_KEYS
    for bad in ("roughness_nm, 0.1, 0.1, 3", "roughness_nm, 0, 9, 3", "L_step, 0.01, 0.02, 3", "oh_dz, -0.1, 0.1, 1",
                "oh_dz, -0.1, 0.1"):
        with pytest.raises(ValueError):
            parse_study_rows(bad)
    seen = []
    out = parameter_study(DEFAULTS, rows, [(0, 0), (1, 1)], parse_comp("OH=1"), progress=lambda i, n, m: seen.append(i))
    by = {d["key"]: d for d in out}
    assert [d["key"] for d in out][:2] == sorted(["roughness_nm", "oh_dz"], key=lambda k: -by[k]["max_rel"])
    assert by["h2o_dz"]["max_rel"] == 0 and out[-1]["key"] == "h2o_dz"          # no H2O on the surface: no effect
    r = by["roughness_nm"]
    assert r["max_rel"] > 0.2 and r["at_hk"] in [(0, 0), (1, 1)] and r["I"][(0, 0)].shape == (4, r["L"][(0, 0)].size)
    # rougher surfaces lose intensity between the Bragg peaks
    L = r["L"][(0, 0)]
    i = np.argmin(np.abs(L - 1.0))
    assert r["I"][(0, 0)][0, i] > r["I"][(0, 0)][-1, i]
    assert seen and max(seen) == 10


def test_parameter_study_tab(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    import ctr_gui
    from ctr_gui import QtCore, QtWidgets
    monkeypatch.setattr(ctr_gui, "SETTINGS_FILE", tmp_path / "settings.ini")
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: errors.append(a[1:3])))
    monkeypatch.setattr(ctr_gui.QtCore.QTimer, "singleShot", staticmethod(lambda *a, **k: None))   # no first run
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = ctr_gui.MainWindow()
    win.show()
    p = win.panel.values()
    win.panel.set_values(dict(p, rods="0 0; 0 1", L_step=0.02))
    assert win.ps_in.rowCount() == 2                                            # the default rows
    win._ps_add_row()
    win.ps_in.cellWidget(2, 0).setCurrentIndex(ctr_gui.STUDY_KEYS.index("b_surf_O"))
    lo, hi = float(win.ps_in.item(2, 1).text()), float(win.ps_in.item(2, 2).text())
    assert lo < p["b_surf_O"] < hi                                              # range around the current value
    win.ps_in.item(2, 3).setText("3")
    rows = win.param_study_rows()
    assert [r[0] for r in rows] == ["roughness_nm", "oh_dz", "b_surf_O"]
    win.run_param_study()
    t0 = time.time()
    while win.ps_worker is not None and time.time() - t0 < 120:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        time.sleep(0.01)
    assert not errors, errors
    assert win.ps_table.rowCount() == 3 and win.ps_result is not None
    assert win.ps_table.item(0, 1).text() == ctr_gui.study_label(win.ps_result[0]["key"])
    assert win.ps_rod.count() == 2 and win.p_ps.figure.axes
    win.ps_rod.setCurrentIndex(1 - win.ps_rod.currentIndex())                 # the other rod
    assert win.p_ps.figure.axes
    win.ps_in.item(0, 3).setText("x")                                            # bad input: message, no crash
    win.run_param_study()
    assert errors and win.ps_worker is None
    win.ps_in.item(0, 3).setText("4")
    win.save_settings()
    assert "roughness_nm, 0.05, 0.4, 4" in (tmp_path / "settings.ini").read_text(encoding="utf-8")
    win.set_param_study_rows([])
    win.load_settings()
    assert win.ps_in.rowCount() == 3
    win.close()
    app.processEvents()

"""Smoke tests: the GUI completes one run offscreen, the notebook runs as a script."""
import os
import subprocess
import sys
import time
from pathlib import Path

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

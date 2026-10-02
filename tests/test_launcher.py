"""Launcher: app tags, discovery, and the window (offscreen)."""
import time
from pathlib import Path

import pytest

from ctrfit.apps import discover, parse_tag, read_app

ROOT = Path(__file__).resolve().parents[1]


def test_parse_tag():
    f = parse_tag(" title: Fitting | group: Fit | order: 10 | kind: gui | needs: PyQt5, numba | desc: a: b")
    assert f == dict(title="Fitting", group="Fit", order="10", kind="gui", needs="PyQt5, numba", desc="a: b")


def test_repository_apps_are_found():
    apps = discover(ROOT)
    titles = [a.title for a in apps]
    assert titles == ["CTR simulator", "3D film viewer", "Simulation notebook", "Fitting", "Example fit",
                      "Make example data", "Beamline helper"]
    assert [a.group for a in apps][:3] == ["Simulate"] * 3
    byname = {a.path.name: a for a in apps}
    assert byname["ctr_viewer3d.py"].needs == ["PyQt5", "numba"] and byname["run_fit.py"].kind == "script"
    assert all(a.desc for a in apps)
    assert "apps.py" not in byname                     # its docstring example is indented, not a tag


def _write(path, tag, body="print('hello from the script')\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((tag + "\n" if tag else "") + body, encoding="utf-8")


@pytest.fixture
def tree(tmp_path):
    _write(tmp_path / "a.py", "# @app title: Script A | group: Tools | order: 2 | kind: script | desc: prints")
    _write(tmp_path / "b.py", "# @app title: Needs X | group: Tools | order: 1 | needs: surely_not_a_module_xyz")
    _write(tmp_path / "g.py", "# @app title: Window G | group: Simulate | kind: gui",
           "from pathlib import Path\nPath('started.txt').write_text('yes')\n")
    _write(tmp_path / "plain.py", None)
    _write(tmp_path / "notitle.py", "# @app group: Tools")
    _write(tmp_path / "tests" / "t.py", "# @app title: hidden test")
    _write(tmp_path / "late.py", "\n" * 50 + "# @app title: too late")
    return tmp_path


def test_discover_rules(tree):
    apps = discover(tree)
    assert [a.title for a in apps] == ["Window G", "Needs X", "Script A"]       # Simulate first, then by order
    assert apps[1].missing == ["surely_not_a_module_xyz"] and apps[2].missing == []
    assert read_app(tree / "plain.py") is None and read_app(tree / "notitle.py") is None
    assert apps[0].kind == "gui" and apps[2].kind == "script"


def test_launcher_window(tree):
    pytest.importorskip("PyQt5")
    from PyQt5 import QtCore, QtWidgets
    from ctrfit.app.launcher_window import LauncherWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = LauncherWindow(root=tree)
    w.show()
    app.processEvents()
    assert [c.app.title for c in w.cards] == ["Window G", "Needs X", "Script A"]
    needs = w.cards[1]
    assert needs.missing and "pip install" in needs.toolTip()
    w.search.setText("script")
    assert [c.isVisible() for c in w.cards] == [False, False, True]
    w.search.setText("")
    script = w.cards[2].app
    assert w.launch(script)
    assert not w.launch(script)                                    # already running
    t0 = time.time()
    while script.path in w.procs and time.time() - t0 < 30:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    app.processEvents()
    out = w.console.toPlainText()
    assert "hello from the script" in out and "finished (exit code 0)" in out
    assert w.cards[2].status.text() == "done"
    assert w.launch(w.cards[0].app)                                # detached window app
    t0 = time.time()
    while not (tree / "started.txt").exists() and time.time() - t0 < 30:
        time.sleep(0.1)
    assert (tree / "started.txt").read_text() == "yes"
    _write(tree / "new.py", "# @app title: Brand new | group: Tools | order: 9")
    w.scan()
    assert "Brand new" in [c.app.title for c in w.cards]
    w.close()

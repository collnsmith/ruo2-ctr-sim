"""Fitting window, offscreen: load data and model, edit the table, run the guided steps and a fit,
stop a run, save and reopen a project. Dialogs are patched so nothing blocks."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5 import QtCore, QtWidgets  # noqa: E402

from ctrfit.app.fit_window import FitWindow  # noqa: E402

EX = Path(__file__).resolve().parents[1] / "examples" / "synthetic_ruo2"


@pytest.fixture
def win(monkeypatch, tmp_path):
    msgs = []
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name,
                            staticmethod(lambda *a, **k: msgs.append(a[1:3]) or QtWidgets.QMessageBox.Ok))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = FitWindow(settings_dir=tmp_path)
    w.messages = msgs
    w.resize(1200, 800)
    w.show()
    app.processEvents()
    w.open_model(str(EX / "model.json"))
    w.add_data(str(EX / "data.csv"))
    w.maxiter.setValue(12)
    w.popsize.setValue(8)
    yield w
    w.close()
    app.processEvents()


def _cell(w, name, col):
    row = w.table.params.names.index(name)
    return w.table.index(row, w.table.COLS.index(col))


def test_window_loads_and_plots(win):
    assert len(win.datasets) == 1 and win.rod_pick.count() == 8
    assert win.table.rowCount() == len(win.model.params)
    assert "7 free" in win.n_free.text()
    assert win.p_rod.figure.axes, "rod plot not drawn"
    win.group.setCurrentText("surface")
    assert 0 < win.proxy.rowCount() < win.table.rowCount()
    win.group.setCurrentText("free")
    assert win.proxy.rowCount() == 7


def test_table_edits_values_bounds_links(win):
    t = win.table
    assert t.setData(_cell(win, "z_H2O", "Value"), "2.5")
    assert win.model.params["z_H2O"].value == 2.5
    assert not t.setData(_cell(win, "theta", "Value"), "1.5")                   # outside bounds
    assert win.messages and "outside" in win.messages[-1][1]
    assert t.setData(_cell(win, "z_H2O", "Link"), "z_OH + 0.4")
    assert win.model.params["z_H2O"].linked
    assert not (t.flags(_cell(win, "z_H2O", "Value")) & QtCore.Qt.ItemIsEditable)
    assert not t.setData(_cell(win, "z_H2O", "Link"), "nonsense_name * 2")
    assert t.setData(_cell(win, "z_H2O", "Link"), "")
    assert not win.model.params["z_H2O"].linked
    assert t.setData(_cell(win, "B_OH", "Fit"), QtCore.Qt.Checked, QtCore.Qt.CheckStateRole)
    assert "B_OH" in win.model.params.free_names and "8 free" in win.n_free.text()
    assert t.setData(_cell(win, "B_OH", "Max"), "5")
    assert win.model.params["B_OH"].max == 5


@pytest.mark.slow
def test_guided_workflow_and_fit(win, tmp_path):
    win.run_step("film")
    assert win.worker is not None and win.table.locked
    assert win.wait(240)
    r = win.result
    assert set(r.names) == {"thickness_mean_tl", "roughness_tl", "eps_perp", "scale"}
    assert abs(r.values["thickness_mean_tl"] - 19.4) < 3 * r.errors["thickness_mean_tl"]
    assert "reduced chi2" in win.report.toPlainText() and win.p_fom.figure.axes
    assert win.table.errors["eps_perp"] > 0 and not win.table.locked
    win.run_step("surface")
    assert win.wait(240) and "x_OH" in win.result.names
    win.run_step("all")
    assert win.wait(240) and len(win.result.names) == 7
    assert win.p_corr.figure.axes
    assert len(win.project.results) == 3
    path = tmp_path / "p.ctrproj.json"
    win.save_project(str(path))
    values = win.model.params.values()
    win.new_model()
    assert win.model.params.values() != values
    win.open_project(str(path))
    assert win.model.params.values() == values and len(win.datasets) == 1
    assert win.result is not None and "reduced chi2" in win.report.toPlainText()
    win.export_report(str(tmp_path / "rep"))
    assert (tmp_path / "rep" / "result.json").exists()


def test_run_and_stop(win):
    win.maxiter.setValue(2000)
    win.run_or_stop()
    app = QtWidgets.QApplication.instance()
    while not win.history:                       # wait for the first progress update
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    win.run_or_stop()                            # Stop
    assert win.wait(60)
    assert win.result.stages[0]["message"] == "stopped by user"
    assert win.run_button.text() == "Run fit"


def test_global_fit_from_window(win, tmp_path):
    from ctrfit import make_dataset
    from test_fit import ODD, truth_model
    win.datasets.clear()
    for i, x in enumerate((0.3, 0.7)):
        d = make_dataset(truth_model(x_OH=x), rods=ODD[:2], noise_rel=0.03, seed=300 + i, step=0.1, name=f"d{i}",
                         potential_V=0.5 + 0.5 * i)
        d.save(tmp_path / f"d{i}.csv")
        win.add_data(str(tmp_path / f"d{i}.csv"))
    win.model.params.fit_only(["x_OH", "scale"])
    win.model.params.update(dict(thickness_mean_tl=19.4, roughness_tl=0.6, eps_perp=2.3, theta=0.85, z_OH=2.05))
    win.per_dataset.setText("x_OH")
    win.method.setCurrentText("refine only")
    win.run_or_stop()
    assert win.wait(120)
    assert {"ds1.x_OH", "ds2.x_OH"} <= set(win.result.names)
    assert win.series_pick.count() >= 1 and win.p_series.figure.axes
    v = [win.result.values[f"ds{i}.x_OH"] for i in (1, 2)]
    np.testing.assert_allclose(v, [0.3, 0.7], atol=0.05)

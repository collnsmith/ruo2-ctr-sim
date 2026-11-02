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


def test_several_datasets_at_once_with_film_and_potential(win, tmp_path):
    """Datasets exported per film and potential: added in one go, listed with film and potential; a run
    gives each film its own thickness (Per film) and each dataset its own x_OH (Per dataset)."""
    from ctrfit import make_dataset
    from ctrfit.fit import workflow
    from test_fit import ODD, truth_model
    win.datasets.clear()
    paths = []
    for i, (film, nm, u) in enumerate((("7 nm", 7.0, 0.7), ("7 nm", 7.0, 0.0), ("4.1 nm", 4.1, 0.7))):
        d = make_dataset(truth_model(), rods=ODD[:1], noise_rel=0.03, seed=400 + i, step=0.2, name=f"{film} {u:g} V",
                         potential_V=u, film=film, thickness_nm=nm)
        d.save(tmp_path / f"f{i}.csv")
        paths.append(str(tmp_path / f"f{i}.csv"))
    added = win.add_datasets(paths)
    assert len(added) == 3 and win.data_list.count() == 3
    assert win.data_list.item(2).text().endswith("rods 1 0, 4.1 nm") or "0.7 V" in win.data_list.item(2).text()
    assert win.per_film.text().startswith("thickness_mean_tl") and win._per_film()
    win.per_dataset.setText("x_OH")
    workflow.prepare(win.model, win.datasets, win._per_dataset(), win._per_film())
    p = win.model.params
    assert p["ds2.thickness_mean_tl"].linked and not p["ds3.thickness_mean_tl"].linked and "ds3.x_OH" in p
    assert p["ds3.thickness_mean_tl"].value < p["ds1.thickness_mean_tl"].value          # 4.1 nm vs 7 nm


def test_window_keeps_its_size_while_a_fit_runs(win, qapp=None):
    """Messages that change during a fit (progress, the free-parameter list, the assistant's counts) and
    the live rod plot must not resize the window or rescale the axes from one trial to the next."""
    from PyQt5.QtWidgets import QApplication
    from ctrfit.project import plots
    app = QApplication.instance()
    win.resize(1500, 900)
    win.show()
    app.processEvents()
    size = (win.width(), win.height())
    names = [f"ds{i}.{n}" for i in range(1, 9) for n in ("theta", "x_OH", "z_OH", "thickness_mean_tl", "scale")]
    win.n_free.setText(f"{len(names)} free: {', '.join(names)}")
    win.status.setText("differential evolution, generation 12: figure of merit 1.23456 " + "long " * 80)
    win.assistant.cost.setText("Tokens: " + "9" * 400)
    win.assistant.send.setText("Claude is working… (Stop: Run fit button)")
    app.processEvents()
    assert (win.width(), win.height()) == size
    assert win.n_free.text().startswith("40 free: ds1.theta") and win.n_free.toolTip() == win.n_free.text()
    # the live rod plot keeps the data's axes whatever the trial model does
    from matplotlib.figure import Figure
    L = np.linspace(0.3, 2, 30)
    F = 10 + 5 * np.sin(L)
    lims = []
    for scale in (1e-3, 1e3):
        fig = Figure()
        plots.draw_fit_rod(fig, L, F, 0.1 * F, F * scale, "live", L, F * scale, hold=True)
        lims.append((fig.axes[0].get_ylim(), fig.axes[1].get_ylim()))
    assert lims[0] == lims[1] and lims[0][1] == (-10.0, 10.0)
    fig = Figure()
    plots.draw_fit_rod(fig, L, F, 0.1 * F, F * 1e3, "final", L, F * 1e3)          # after the fit: autoscale
    assert fig.axes[0].get_ylim()[1] > 1e3


def test_live_update_every_generation(win, monkeypatch):
    """The model curve and the table follow the best values after every DE generation."""
    seen = []
    orig = win._on_best

    def spy(stage, step, fom, values):
        seen.append((stage, step, fom, dict(values)))
        orig(stage, step, fom, values)
        if stage == "de":
            title = win.p_rod.figure.axes[0].get_title()
            assert f"generation {step}" in title
            assert win.table.data(_cell(win, "z_OH", "Value")) == f"{values['z_OH']:.6g}"
    monkeypatch.setattr(win, "_on_best", spy)
    win.worker = None
    win.maxiter.setValue(6)
    win.method.setCurrentText("DE only")
    start = win.model.params["z_OH"].value
    win.run_or_stop()
    assert win.wait(120)
    de = [s for s in seen if s[0] == "de"]
    assert [s[1] for s in de] == list(range(1, len(de) + 1)) and len(de) >= 6
    assert all(b[2] <= a[2] for a, b in zip(de, de[1:]))                 # best FOM never gets worse
    assert len({round(s[3]["z_OH"], 6) for s in de} | {round(start, 6)}) > 1   # the values move
    assert not win.table.live and "generation" not in win.p_rod.figure.axes[0].get_title()

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
    win.clear_data()
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
    win.clear_data()
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
    """Messages that change during a fit (progress, the free-parameter list) and
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


def test_scan_table_leave_out_and_groups(win, tmp_path):
    """Each scan of the loaded files is listed; unticking leaves it out of every fit (drawn hollow), a
    group typed for scans of two files fits them as one dataset, and the project keeps the choices."""
    from PyQt5.QtCore import Qt
    from test_scans import _file
    win.clear_data()
    paths = []
    for i, (nm, u) in enumerate((("A", 0.5), ("A repeat", 0.5), ("B", 0.9))):
        d = _file(nm, u, 700 + i, scans=(f"#{2 * i}", f"#{2 * i + 1}"))
        d.save(tmp_path / f"s{i}.csv")
        paths.append(str(tmp_path / f"s{i}.csv"))
    win.add_datasets(paths)
    t = win.scan_table
    assert t.rowCount() == 6 and win.data_list.count() == 3 and "6 of 6 scans" in win.scan_info.text()
    assert t.item(0, 1).text() == "#0" and t.item(0, 7).text() == "A" and "2 scans" in win.data_list.item(0).text()
    n0 = len(win.datasets[0])
    t.item(1, 0).setCheckState(Qt.Unchecked)                       # tick box: leave scan #1 out
    QtWidgets.QApplication.processEvents()
    assert len(win.datasets[0]) < n0 and "5 of 6" in win.scan_info.text()
    assert t.item(1, 0).checkState() == Qt.Unchecked
    assert win.rod_pick.itemText(1).endswith("(left out)")         # A's second rod: only the left-out scan
    win.rod_pick.setCurrentIndex(1)
    labels = [x.get_text() for x in win.p_rod.figure.axes[0].get_legend().get_texts()]
    assert "left out" in labels
    t.item(2, 8).setText("A")                                       # typed group: the repeat's first scan with A
    QtWidgets.QApplication.processEvents()
    assert [d.name for d in win.datasets] == ["A", "A repeat", "B"] and win.datasets[0].scan_labels == ["#0", "#2"]
    t.selectRow(3)
    win.group_selected("A")                                         # the repeat's other scan too: file gone
    assert [d.name for d in win.datasets] == ["A", "B"] and win.data_list.count() == 2
    win.rod_pick.setCurrentIndex(0)
    labels = [x.get_text() for x in win.p_rod.figure.axes[0].get_legend().get_texts()]
    assert {"#0", "#2"} <= set(labels)                              # one colour per scan
    win.data_list.setCurrentRow(1)                                  # a dataset picks its scans in the table
    assert win.selected_scans() == win.scan_keys("B")
    win.data_list.setCurrentRow(0)
    assert len(win.selected_scans()) == 4
    win.save_project(str(tmp_path / "p.ctrproj.json"))
    win.clear_data()
    assert t.rowCount() == 0 and not win.datasets
    win.open_project(str(tmp_path / "p.ctrproj.json"))
    assert [d.name for d in win.datasets] == ["A", "B"] and t.rowCount() == 6
    assert t.item(1, 0).checkState() == Qt.Unchecked and t.item(3, 8).text() == "A"
    win.set_scan_group(win.scan_keys("A repeat"), "")                # back to the file's own group
    assert [d.name for d in win.datasets] == ["A", "A repeat", "B"]
    win.set_scan_use(win.scan_keys(), False)                         # nothing to fit
    win.run_or_stop()
    assert win.worker is None and "left out" in win.messages[-1][1]
    win.set_scan_use(win.scan_keys(), True)
    win.remove_data(["B"])
    assert [r.source for r in win.project.scans.rows()].count("B") == 0 and len(win.datasets) == 2


def _palette(base, text):
    from PyQt5 import QtGui
    pal = QtGui.QPalette()
    for role in (QtGui.QPalette.Base, QtGui.QPalette.Window):
        pal.setColor(role, QtGui.QColor(base))
    for role in (QtGui.QPalette.Text, QtGui.QPalette.WindowText):
        pal.setColor(role, QtGui.QColor(text))
    return pal


@pytest.mark.parametrize("base, text", [("#ffffff", "#000000"), ("#2a2a2a", "#f0f0f0"), ("#1e1e1e", "#dcdcdc")])
def test_table_colours_readable_in_light_and_dark_themes(win, base, text):
    """Flagged and linked rows (and the library's to-fill / wrong cells) take their background from the
    theme: the theme's text stays readable on them (contrast >= 4.5), in light and dark themes."""
    from PyQt5 import QtCore
    from ctrfit.app import widgets as W
    app = QtWidgets.QApplication.instance()
    old = app.palette()
    try:
        app.setPalette(_palette(base, text))
        win.model.params.link("z_H2O", "z_OH + 0.5")
        win.table.flagged = {"z_OH"}
        win.table.refresh()
        for name in ("z_OH", "z_H2O"):
            row = win.table.params.names.index(name)
            bg = win.table.data(win.table.index(row, 1), QtCore.Qt.BackgroundRole)
            fg = win.table.data(win.table.index(row, 1), QtCore.Qt.ForegroundRole)
            assert W.contrast(bg, fg) >= 4.5 and W.contrast(bg, text) >= 4.5, (name, bg.name())
            assert (bg.lightness() < 128) == W.is_dark()
        for accent, amount in ((W.TODO, 0.14), (W.BAD, 0.155), (W.FLAG, 0.4)):
            assert W.contrast(W.tint(accent, amount), text) >= 4.5
        assert W.contrast(W.alert_text(), base) >= 4.5 and W.contrast(W.muted_text(), base) > 1.5
        if base == "#ffffff":                                       # light theme: the colours as before
            assert W.tint(W.FLAG, 0.245).name() == "#fff0c2" and W.tint(W.LINK, 0.085).name() == "#eef3fb"
    finally:
        app.setPalette(old)


def test_tick_and_untick_many_scans_and_parameters(win, tmp_path):
    """Fit all / none for the scans and for the parameters the Show filter lists, the selected rows,
    Space toggles the selection; shared parameters every dataset overrides are greyed and skipped."""
    from PyQt5 import QtCore, QtGui
    from ctrfit.fit import workflow
    from test_scans import _file
    win.clear_data()
    paths = []
    for i, (nm, film) in enumerate((("A", "7 nm"), ("B", "4 nm"))):
        d = _file(nm, 0.5 + 0.4 * i, 800 + i, film=film, scans=(f"#{2 * i}", f"#{2 * i + 1}"))
        d.meta["thickness_nm"] = float(film.split()[0])
        d.save(tmp_path / f"t{i}.csv")
        paths.append(str(tmp_path / f"t{i}.csv"))
    win.add_datasets(paths)
    names = [b.text() for b in win.scan_buttons]
    win.scan_buttons[names.index("Leave all out")].click()
    assert "0 of 4" in win.scan_info.text() and not win.datasets
    win.scan_buttons[names.index("Fit all")].click()
    assert "4 of 4" in win.scan_info.text() and len(win.datasets) == 2
    win.scan_table.selectRow(0)
    win.scan_table.setFocus()
    QtWidgets.QApplication.sendEvent(win.scan_table, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Space,
                                                                      QtCore.Qt.NoModifier))
    assert "3 of 4" in win.scan_info.text()                                  # Space: the selected scan out
    win.scan_buttons[names.index("Fit selected")].click()
    assert "4 of 4" in win.scan_info.text()
    # parameters: the Show filter's rows
    P = win.model.params
    pb = {b.text(): b for b in win.param_buttons}
    win.group.setCurrentText("free")
    pb["Fix shown"].click()
    assert not P.free_names
    win.group.setCurrentText("surface")
    shown = win.shown_params()
    pb["Fit shown"].click()
    assert {n for n in shown if not P[n].linked} == set(P.free_names)
    assert all(P[n].group == "surface" for n in P.free_names)
    win.group.setCurrentText("all")
    for name in ("theta", "x_OH"):
        win.view.selectionModel().select(win.proxy.mapFromSource(win.table.index(P.names.index(name), 0)),
                                         QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    pb["Fix selected"].click()
    assert not P["theta"].fit and not P["x_OH"].fit and set(win.selected_params()) == {"theta", "x_OH"}
    QtWidgets.QApplication.sendEvent(win.view, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Space,
                                                                QtCore.Qt.NoModifier))
    assert P["theta"].fit and P["x_OH"].fit                                  # Space: both back on
    # once each film has its own thickness, the shared one is greyed and skipped by Fit all
    workflow.prepare(win.model, win.datasets, ["x_OH"], win._per_film())
    win._refresh_all()
    assert "thickness_mean_tl" in win.table.inactive and "x_OH" in win.table.inactive
    row = P.names.index("thickness_mean_tl")
    assert win.table.data(win.table.index(row, 1), QtCore.Qt.ForegroundRole) is not None
    assert "Not used" in win.table.data(win.table.index(row, 1), QtCore.Qt.ToolTipRole)
    win.group.setCurrentText("film")
    pb["Fit shown"].click()
    assert not P["thickness_mean_tl"].fit and P["ds1.thickness_mean_tl"].fit and P["ds2.thickness_mean_tl"].fit


def test_run_on_several_cores(win):
    """The Cores setting: a fit from the window runs DE and the Jacobian on worker processes."""
    assert 1 <= win.cores.value() <= win.cores.maximum()
    win.cores.setValue(2)
    win.maxiter.setValue(4)
    win.run_or_stop()
    assert not win.cores.isEnabled()                     # not changed during a run
    assert win.wait(120)
    assert win.result is not None and win.result.stages[0]["cores"] == 2 and win.cores.isEnabled()


def test_model_colour_follows_the_selected_parameter(win):
    """Selecting a parameter colours the model curve by its effect (a colour bar appears); the box turns it
    off; a linked parameter says so."""
    P = win.model.params

    def select(name):
        idx = win.proxy.mapFromSource(win.table.index(P.names.index(name), 1))
        win.view.setCurrentIndex(idx)
        win.view.selectRow(idx.row())
        QtWidgets.QApplication.processEvents()

    assert win.sens_box.isChecked()
    select("x_OH")
    assert len(win.p_rod.figure.axes) == 3 and "x_OH" in win.p_rod.figure.axes[2].get_ylabel()
    win.sens_box.setChecked(False)
    assert len(win.p_rod.figure.axes) == 2
    win.sens_box.setChecked(True)
    assert len(win.p_rod.figure.axes) == 3
    P.link("z_H2O", "z_OH + 0.4")
    win._refresh_all()
    select("z_H2O")
    assert len(win.p_rod.figure.axes) == 2 and "linked" in win.status.text()

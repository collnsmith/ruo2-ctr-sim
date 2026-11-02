"""Scan library: scans of ROI sums sorted by film and CV / CA in a text file (data read only), and
comparisons by group (CVs at one H K L, CAs at one potential on one rod), logic and tab."""
import os
import shutil

import numpy as np
import pytest

from ctrfit.datatools import images
from ctrfit.datatools import library as lb
from ctrfit.echem.external import parse_traces


def rod_I(H, K, L):
    return 1e4 / (0.05 + np.sin(np.pi * L) ** 2) + 100 * (abs(H) + abs(K))


def write_beamtime(folder, name, scale, seed=0):
    """SPEC file and ROI sums of one film, `scale` times the same structure:
        1, 2  rod 0 1 (scan 2 is 1.2x scan 1)       4  rod 1 1
        3     loopscan at 0 1 0.5 (#Q, no H K L columns), intensity changing in time
        5     hklscan that stays at 0 1 1.5 (stationary from its columns)
        6     a rocking scan (H and K move): other
    ROI1 with its error and a fixed position, ROI2 = 2 x ROI1 without an error, moving.
    Returns (spec path, sums path, {scan: normalised intensity per point})."""
    rng = np.random.default_rng(seed)
    spec = [f"#F {name}.spec", "#E 1772892000", "#D Sat Mar 07 14:00:00 2026", ""]
    sums = ["scan\timage\tROI1\tROI2\tROI1_err"]
    truth, used = {}, []

    def counts(sn, I0, n):
        mon = rng.uniform(9e4, 1.1e5, n)
        tr = np.where(np.arange(n) % 4 == 0, 0.1, 1.0)
        ct = np.full(n, 0.5)
        truth[sn] = I0
        return mon, tr, ct, I0 * (mon / 1e5) * tr * ct

    def add_sums(sn, I):
        nonlocal sums
        sums += [f"{sn:03d}\t{i:03d}\t{v:.10g}\t{2 * v:.10g}\t{np.sqrt(v):.10g}" for i, v in enumerate(I)]
        for i in range(I.size):
            used.append((sn, i, [images.ROI(500, 300, 40, 30, "signal", "ROI1"),
                                 images.ROI(100 + i, 50, 20, 20, "signal", "ROI2")]))

    for sn, (H, K, L0, L1, n, f) in {1: (0, 1, 0.3, 2.0, 30, 1.0), 2: (0, 1, 0.3, 2.0, 30, 1.2),
                                     4: (1, 1, 0.5, 1.5, 20, 1.0)}.items():
        L = np.linspace(L0, L1, n)
        mon, tr, ct, I = counts(sn, scale * f * rod_I(H, K, L), n)
        spec += [f"#S {sn}  hklscan {H} {H} {K} {K} {L0} {L1} {n - 1} 0.5", "#D Sat Mar 07 14:10:00 2026",
                 "#L H  K  L  Epoch  Seconds  Monitor  transm  Detector"]
        spec += [f"{H} {K} {l:.6f} {100 * sn + i} {c:g} {m:.3f} {x:g} {d:.3f}" for i, (l, c, m, x, d) in
                 enumerate(zip(L, ct, mon, tr, I))] + [""]
        add_sums(sn, I)
    n = 40
    mon, tr, ct, I = counts(3, scale * rod_I(0, 1, 0.5) * (1 + 0.3 * np.sin(np.arange(n) / 5)), n)
    spec += ["#S 3  loopscan 40 0.5 0", "#D Sat Mar 07 14:20:00 2026", "#Q 0 1 0.5",
             "#L Time  Epoch  Seconds  Monitor  transm  Detector"]
    spec += [f"{2.0 * i:g} {1000 + 2.0 * i:g} {c:g} {m:.3f} {x:g} {d:.3f}" for i, (c, m, x, d) in
             enumerate(zip(ct, mon, tr, I))] + [""]
    add_sums(3, I)
    mon, tr, ct, I = counts(5, scale * rod_I(0, 1, 1.5) * np.linspace(1, 0.5, n), n)
    spec += ["#S 5  hklscan 0 0 1 1 1.5 1.5 39 0.5", "#D Sat Mar 07 14:30:00 2026",
             "#L H  K  L  Epoch  Seconds  Monitor  transm  Detector"]
    spec += [f"0 1 1.5 {2000 + i} {c:g} {m:.3f} {x:g} {d:.3f}" for i, (c, m, x, d) in
             enumerate(zip(ct, mon, tr, I))] + [""]
    add_sums(5, I)
    th = np.linspace(-1, 1, 10)
    mon, tr, ct, I = counts(6, scale * 1e3 * np.exp(-th ** 2), 10)
    spec += ["#S 6  ascan th -1 1 9 0.5", "#D Sat Mar 07 14:40:00 2026",
             "#L th  H  K  L  Epoch  Seconds  Monitor  transm  Detector"]
    spec += [f"{t:g} {0.05 * t:.5f} {1 - 0.05 * t:.5f} 0.5 {3000 + i} {c:g} {m:.3f} {x:g} {d:.3f}"
             for i, (t, c, m, x, d) in enumerate(zip(th, ct, mon, tr, I))] + [""]
    add_sums(6, I)
    sums += [""] + images.roi_positions_section(used)
    (folder / f"{name}.spec").write_text("\n".join(spec) + "\n")
    (folder / f"{name}_sums.txt").write_text("\n".join(sums) + "\n")
    return folder / f"{name}.spec", folder / f"{name}_sums.txt", truth


def two_films(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    a = write_beamtime(data, "a", 1.0)
    b = write_beamtime(data, "b", 2.0, seed=1)
    return a, b


def assigned_library(tmp_path):
    (sa, ua, ta), (sb, ub, tb) = two_films(tmp_path)
    lib = lb.Library()
    lib.add_source(sa, [ua])
    lib.add_source(sb, [ub], "film B run")
    lib.settings.update(monitor="Monitor", transmission="transm", count_time="Seconds")
    S1 = [e for e in lib.entries if e.source == "S1"]
    S2 = [e for e in lib.entries if e.source == "S2"]
    lib.assign(S1, film="10 nm")
    lib.assign(S2, film="2 nm")
    E = lib.entry
    lib.assign([E("S1", 1), E("S2", 1), E("S1", 4), E("S2", 4)], mode="CA", potential=1.0)
    lib.assign([E("S1", 2), E("S2", 2)], potential=1.4)                # a potential alone: CA
    lib.assign([E("S1", 3), E("S2", 3), E("S1", 5)], mode="CV")
    return lib, (sa, ua, ta), (sb, ub, tb)


def file_state(paths):
    return {str(p): (open(p, "rb").read(), os.stat(p).st_mtime_ns) for p in paths}


def test_library_reads_the_scans(tmp_path):
    (sa, ua, ta), _ = two_films(tmp_path)
    lib = lb.Library()
    src = lib.add_source(sa, [ua])
    assert src.id == "S1" and src.label == "a" and lib.notes == []
    by = {e.scan: e for e in lib.entries}
    assert [(e.scan, e.kind) for e in lib.entries] == [(1, "rod"), (2, "rod"), (3, "stationary"), (4, "rod"),
                                                       (5, "stationary"), (6, "other")]
    assert (by[1].rod, by[1].L_min, by[1].L_max, by[1].points) == ((0, 1), 0.3, 2.0, 30)
    assert (by[3].H, by[3].K, by[3].L) == (0, 1, 0.5) and by[3].command.startswith("loopscan")   # from #Q
    assert by[5].rod == (0, 1) and by[5].L == pytest.approx(1.5)
    assert by[1].rois == ["ROI1", "ROI2"] and all(not e.assigned for e in lib.entries)
    assert lib.effective_roi(by[1]) == "ROI1"                                    # the first with an error
    assert lib.roi_description(by[1], "ROI1") == "ROI1: signal ROI 40 x 30 px at (500, 300) in a_sums.txt"
    assert "moves from image to image" in lib.roi_description(by[1], "ROI2")
    assert lib.spec_columns()[:3] == ["H", "K", "L"] and "Time" in lib.spec_columns()
    # the same SPEC file again with a second background subtraction run: its ROIs are kept apart
    second = tmp_path / "data" / "a_rerun.txt"
    shutil.copy(ua, second)
    assert lib.add_source(sa, [second]) is src and len(src.sums) == 2 and len(lib.sources) == 1
    assert by[1].rois == ["ROI1", "ROI2", "ROI1 (a_rerun)", "ROI2 (a_rerun)"]
    with pytest.raises(ValueError, match="not a ROI sums file"):
        lib.add_source(sa, [sa])


def test_library_file_round_trip_and_the_data_stay_untouched(tmp_path):
    lib, (sa, ua, _), (sb, ub, _) = assigned_library(tmp_path)
    before = file_state([sa, ua, sb, ub])
    lib.entry("S1", 3).roi = "ROI2"
    lib.entry("S2", 6).use = False
    lib.assign([lib.entry("S1", 1)], notes="after\tthe\nstep")                       # tabs never break the file
    (tmp_path / "lib").mkdir()
    path = lb.write_library(lib, tmp_path / "lib" / "scans.txt")
    text = open(path).read()
    assert text.startswith(lb.FORMAT) and "S1\tspec\t../data/a.spec\t" in text      # relative to the library
    assert "S1\t1\t10 nm\tCA\t1\t\t1\trod\t0\t1\t" in text and "after the step" in text
    # move library and data together: the relative paths still find the data
    shutil.move(str(tmp_path / "lib"), str(tmp_path / "moved" / "lib"))
    shutil.move(str(tmp_path / "data"), str(tmp_path / "moved" / "data"))
    back = lb.read_library(tmp_path / "moved" / "lib" / "scans.txt")
    assert back.notes == [] and back.settings == lib.settings
    key = lambda L: [(e.key, e.film, e.mode, e.potential, e.roi, e.use, e.kind, e.points, e.notes) for e in L.entries]
    assert key(back) == key(lib)
    assert back.source("S2").label == "film B run" and back.source("S1").spec.endswith(os.path.join("moved", "data",
                                                                                                    "a.spec"))
    moved = [tmp_path / "moved" / "data" / n for n in ("a.spec", "a_sums.txt", "b.spec", "b_sums.txt")]
    assert [v[0] for v in file_state(moved).values()] == [v[0] for v in before.values()]  # bytes unchanged
    # the library never writes over a data file, and a crash while writing leaves no half file
    with pytest.raises(ValueError, match="read only"):
        lb.write_library(back, moved[1])
    assert open(moved[1], "rb").read() == before[str(ua)][0]
    assert not any(n.endswith(".tmp") for n in os.listdir(tmp_path / "moved" / "lib"))
    # a data file that changed since it was read is reported when the library is opened
    with open(moved[2], "a") as f:
        f.write("\n")
    os.utime(moved[2], ns=(1, 1))
    assert any("b.spec changed" in n for n in lb.read_library(tmp_path / "moved" / "lib" / "scans.txt").notes)
    # edited by hand: a potential without a mode is a CA, a bad potential is an error
    p = tmp_path / "moved" / "lib" / "scans.txt"
    lines = open(p).read().splitlines()
    i = next(i for i, l in enumerate(lines) if l.startswith("S2\t6\t"))
    parts = lines[i].split("\t")
    parts[3], parts[4] = "", "0.8"
    lines[i] = "\t".join(parts)
    open(p, "w").write("\n".join(lines) + "\n")
    e = lb.read_library(p).entry("S2", 6)
    assert (e.mode, e.potential, e.use) == ("CA", 0.8, False)
    parts[4] = "high"
    lines[i] = "\t".join(parts)
    open(p, "w").write("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="not a number"):
        lb.read_library(p)
    with pytest.raises(ValueError, match="not a scan library"):
        lb.read_library(moved[1])


def test_comparisons_by_group(tmp_path):
    lib, (_, _, ta), (_, _, tb) = assigned_library(tmp_path)
    # all CAs at 1.0 V on rod 0 1: one scan per film, normalised by monitor, transmission, count time
    sel = lb.select(lib, "rod", {"CA"}, potential=1.0, rod=(0, 1))
    assert sorted(e.key for e in sel) == [("S1", 1), ("S2", 1)]
    trs, notes = lb.traces(lib, sel, colour_by="film")
    assert notes == [] and [t.group for t in trs] == ["2 nm", "10 nm"]                 # natural order
    t2, t10 = trs
    assert t10.xlabel == "L (r.l.u.)" and np.allclose(t10.x, np.linspace(0.3, 2.0, 30))
    q = t10.y / ta[1]
    assert np.allclose(q, q[0], rtol=1e-6)                                      # the truth times one constant
    assert np.allclose(t2.y / t10.y, 2.0, rtol=1e-6)                             # one monitor reference
    assert t10.sigma is not None and np.all(t10.sigma > 0)
    # one film, every potential on the rod
    sel = lb.select(lib, "rod", {"CA"}, films={"10 nm"}, rod=(0, 1))
    trs, _ = lb.traces(lib, sel, colour_by="potential")
    assert [t.group for t in trs] == ["CA 1 V vs RHE", "CA 1.4 V vs RHE"]
    assert np.allclose(trs[1].y / trs[0].y, 1.2, rtol=1e-6)
    assert lib.potentials() == [1.0, 1.4]
    # all CVs at 0 1 0.5: time from the scan start, H K L of the loopscan from #Q
    st = lb.select(lib, "stationary")
    assert lb.positions(st) == [(0, 1, 0.5), (0, 1, 1.5)]
    sel = lb.select(lib, "stationary", {"CV"}, L=0.5, rod=(0, 1))
    trs, _ = lb.traces(lib, sel, lb.X_TIME)
    assert [t.entry.key for t in trs] == [("S2", 3), ("S1", 3)]
    assert trs[0].xlabel == "time from scan start (s)" and np.allclose(trs[0].x, 2.0 * np.arange(40))
    assert np.allclose(trs[1].y / ta[3], (trs[1].y / ta[3])[0], rtol=1e-6)
    y, s = lb.scaled(trs[1], lb.NORMS[2])
    assert y[0] == 1 and np.allclose(y, ta[3] / ta[3][0], rtol=1e-6)            # I / I0
    assert lb.traces(lib, sel, lb.X_POINT)[0][0].xlabel == "SPEC point"
    assert np.allclose(lb.traces(lib, sel, "Epoch")[0][0].x, 1000 + 2.0 * np.arange(40))
    # the ROI per scan: ROI2 (2 x ROI1, no error column)
    lib.entry("S1", 3).roi = "ROI2"
    t = lb.trace(lib, lib.entry("S1", 3), lb.X_TIME, lb.monitor_reference(lib, sel))
    assert t.roi == "ROI2" and t.sigma is None and np.allclose(t.y, 2 * trs[1].y, rtol=1e-6)
    lib.settings["roi"] = "ROI2"                                                 # the default ROI
    assert lib.effective_roi(lib.entry("S2", 3)) == "ROI2"
    lib.settings["roi"] = ""
    # filters: unused scans and other kinds stay out, films, CV scans pass a potential filter
    assert [e.key for e in lb.select(lib, "other")] == [("S1", 6), ("S2", 6)]
    lib.entry("S2", 6).use = False
    assert [e.key for e in lb.select(lib, "other")] == [("S1", 6)]
    assert {e.key for e in lb.select(lib, "stationary", {"CV", "CA"}, potential=1.0)} == {("S1", 3), ("S2", 3),
                                                                                          ("S1", 5)}
    assert lb.select(lib, "stationary", {"CV"}, films={""}) == []
    assert lb.group_label(lib, lib.entry("S2", 3), "scan") == "film B run #3"
    assert lb.group_label(lib, lib.entry("S2", 5), "film and potential") == "2 nm, no condition"
    # export: the traces read back with the CV tool's reader
    trs, _ = lb.traces(lib, sel, lb.X_TIME)
    out = lb.write_traces(tmp_path / "tr.txt", trs, "CVs")
    blocks = parse_traces(open(out).read())
    assert [b["label"] for b in blocks] == ["S2_3", "S1_3"] and blocks[0]["columns"] == ["time_from_scan_start", "I",
                                                                                         "I_err"]
    assert np.allclose(blocks[1]["data"][:, 1], trs[1].y, rtol=1e-7)


def test_refresh_keeps_assignments(tmp_path):
    lib, (sa, ua, _), _ = assigned_library(tmp_path)
    lib.entry("S1", 1).roi = "ROI2"
    # a new scan in the data (a later batch run) is added unassigned; assignments stay
    lines = open(ua).read().split("\n\n")[0].splitlines()
    extra = [l.replace("001\t", "007\t", 1) for l in lines if l.startswith("001\t")]
    open(ua, "w").write("\n".join(lines + extra) + "\n")
    open(sa, "a").write("#S 7  hklscan 0 0 1 1 0.3 2 29 0.5\n#L H  K  L  Epoch  Seconds  Monitor  transm  Detector\n"
                        + "".join(f"0 1 {0.3 + i * 1.7 / 29:.6f} {i} 0.5 1e5 1 1\n" for i in range(30)))
    notes = lib.load()
    e7 = lib.entry("S1", 7)
    assert e7 is not None and e7.kind == "rod" and not e7.assigned
    assert (lib.entry("S1", 1).film, lib.entry("S1", 1).mode, lib.entry("S1", 1).roi) == ("10 nm", "CA", "ROI2")
    assert any("a_sums.txt changed" in n for n in notes)
    # a scan that left the ROI sums keeps its entry (kind missing) and is never plotted
    open(ua, "w").write("\n".join(l for l in lines if not l.startswith("003\t")) + "\n")
    notes = lib.load()
    e3 = lib.entry("S1", 3)
    assert e3.kind == "missing" and e3.mode == "CV" and any("scan 3: no ROI sums" in n for n in notes)
    assert ("S1", 3) not in {e.key for e in lb.select(lib, "stationary")}
    # undo state: remove a source and bring it back
    snap = lib.snapshot()
    lib.remove_source("S2")
    assert {e.source for e in lib.entries} == {"S1"}
    lib.restore(snap)
    assert lib.entry("S2", 1).film == "2 nm" and lb.traces(lib, [lib.entry("S2", 1)])[0]
    assert lib.add_source(sa, [ua]).id == "S1" and lib._next_id() == "S3"


# ------------------------------------------------------------------ the tab
@pytest.fixture
def qapp():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_library_tab(qapp, tmp_path, monkeypatch):
    from PyQt5.QtCore import QCoreApplication, QEvent, Qt
    from PyQt5.QtWidgets import QMessageBox
    said = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(lambda *a, k=kind, **kw: said.append((k, a[2]))))
    monkeypatch.setenv("CTRFIT_SETTINGS_DIR", str(tmp_path / "settings"))
    from ctrfit.app.datatools import library as ui

    def settle():
        qapp.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        w._timer.stop()
        w.draw_plot()

    (sa, ua, ta), (sb, ub, tb) = two_films(tmp_path)
    before = file_state([sa, ua, sb, ub])
    w = ui.LibraryWindow()
    w.show()
    assert "Add a SPEC file" in w._status_label.text()
    w.add_source(sa, [ua])
    w.add_source(sb, [ub], "film B")
    assert w._table.rowCount() == 12 and w._count.text() == "12 of 12 shown, 12 to assign"
    assert "Not saved yet" in w._state.text()
    # assign with the panel: film per source, CA at a potential, CV
    w.select_scans([("S1", i) for i in range(1, 7)])
    assert w.assign_selected(film="10 nm") == 6
    w.select_scans([("S2", i) for i in range(1, 7)])
    w._film.setEditText("2 nm")
    w.assign_selected(film=w._film.currentText())
    assert w._film.findText("10 nm") >= 0                                         # known films offered
    w.select_scans([("S1", 1), ("S2", 1)])
    w._rb_ca.setChecked(True)
    w._pot.setValue(1.0)
    w.assign_selected(**w._mode_fields())
    w.select_scans([("S1", 3), ("S2", 3)])
    w._rb_cv.setChecked(True)
    assert not w._pot.isEnabled()                                                # CV has nothing to set
    w.assign_selected(**w._mode_fields())
    assert w._status_label.text() == "2 scan(s): mode = 'CV'"
    # typing a potential with two rows selected sets both and makes them CA
    w.select_scans([("S1", 2), ("S2", 2)])
    w._table.item(w._row_of[("S1", 2)], ui.C_E).setText("1.4")
    settle()
    assert [(e.mode, e.potential) for e in (w.lib.entry("S1", 2), w.lib.entry("S2", 2))] == [("CA", 1.4)] * 2
    assert w._table.cellWidget(w._row_of[("S2", 2)], ui.C_MODE).currentText() == "CA"
    w._table.item(w._row_of[("S1", 2)], ui.C_E).setText("high")
    settle()
    assert said[-1][0] == "warning" and w.lib.entry("S1", 2).potential == 1.4
    said.clear()
    # undo / redo
    w.undo()
    settle()
    assert w.lib.entry("S2", 2).mode == ""
    w.redo()
    settle()
    assert w.lib.entry("S2", 2).potential == 1.4
    # the measurement and ROI boxes of a row
    w._table.cellWidget(w._row_of[("S1", 4)], ui.C_MODE).setCurrentText("CA")
    settle()
    e4 = w.lib.entry("S1", 4)
    assert e4.mode == "CA" and e4.potential is None
    assert w._table.item(w._row_of[("S1", 4)], ui.C_E).background().color() == ui.BAD    # potential missing
    roi = w._table.cellWidget(w._row_of[("S1", 3)], ui.C_ROI)
    assert [roi.itemText(i) for i in range(roi.count())] == ["(default)", "ROI1", "ROI2"]
    assert "40 x 30 px" in roi.itemData(1, Qt.ToolTipRole)
    roi.setCurrentText("ROI2")
    settle()
    assert w.lib.entry("S1", 3).roi == "ROI2" and "Shown with: ROI2" in roi.toolTip()
    # filter the rows
    w._filter.setText("stationary CV")
    assert w._count.text().startswith("2 of 12 shown")
    w._filter.setText("")
    w._chk_todo.setChecked(True)
    todo = [w._entry_at(r).key for r in range(w._table.rowCount()) if not w._table.isRowHidden(r)]
    assert ("S1", 4) in todo and ("S1", 1) not in todo                              # CA without potential: to do
    w._chk_todo.setChecked(False)
    # normalisation columns
    for combo, text in ((w._monitor, "Monitor"), (w._transm, "transm"), (w._ctime, "Seconds")):
        combo.setCurrentText(text)
    assert w.lib.settings["count_time"] == "Seconds"
    # compare: CAs on one rod at one potential, one colour per film
    w.preset("ca")
    settle()
    ax = w._fig.axes[0]
    assert ax.get_title() == "CA at 1 V vs RHE, rod 0 1" and ax.get_yscale() == "log"
    lines = [l for l in ax.get_lines()]
    assert [l.get_label() for l in lines] == ["2 nm", "10 nm"]
    assert np.allclose(lines[0].get_ydata() / lines[1].get_ydata(), 2.0, rtol=1e-6)
    assert "S2:1 (ROI1), S1:1 (ROI1)" in w._plot_info.text()
    w._stack.setValue(10)
    settle()
    lines = w._fig.axes[0].get_lines()
    assert np.allclose(lines[1].get_ydata() / lines[0].get_ydata(), 5.0, rtol=1e-6)    # 10 nm group x 10
    w._stack.setValue(1)
    # one film unticked
    item = w._films.findItems("2 nm", Qt.MatchExactly)[0]
    item.setCheckState(Qt.Unchecked)
    settle()
    assert [l.get_label() for l in w._fig.axes[0].get_lines()] == ["10 nm"]
    item.setCheckState(Qt.Checked)
    # CVs at one H K L, against time, then against a SPEC column
    w.preset("cv")
    settle()
    ax = w._fig.axes[0]
    assert ax.get_title() == "CV at (0 1 0.5)" and ax.get_xlabel() == "time from scan start (s)"
    assert [l.get_label() for l in ax.get_lines()] == ["2 nm", "10 nm"] and ax.get_yscale() == "linear"
    assert np.allclose(ax.get_lines()[1].get_ydata(), ax.get_lines()[0].get_ydata(), rtol=1e-6)   # ROI2 = 2 x ROI1
    w._x.setCurrentText("Epoch")
    settle()
    assert w._fig.axes[0].get_xlabel() == "Epoch"
    w._norm.setCurrentText(lb.NORMS[2])
    settle()
    assert all(l.get_ydata()[0] == pytest.approx(1.0) for l in w._fig.axes[0].get_lines())
    # selected rows only
    w._chk_sel.setChecked(True)
    w.select_scans([("S1", 5), ("S1", 1)])                                          # a rod scan is not stationary
    settle()
    assert len(w._fig.axes[0].get_lines()) == 1 and "1 selected stationary" in w._fig.axes[0].get_title()
    w._chk_sel.setChecked(False)
    # nothing matches
    w._mode_f.setCurrentIndex(0)
    w._where.setCurrentIndex(w._where.count() - 1)
    settle()
    assert not w._fig.axes and "No scans match" in w._fig.texts[0].get_text()
    w.preset("cv")
    settle()
    # exports
    out = w.export_data(str(tmp_path / "tr.txt"))
    assert [b["label"] for b in parse_traces(open(out).read())] == ["S2_3", "S1_3"]
    fw = w.export_figure()
    assert fw is not None and len(fw.spec.traces) == 2
    fw.close()
    # save: never over a data file; then to the library file
    assert w.save_as(str(ua)) is None and said[-1][0] == "warning" and "read only" in said[-1][1]
    said.clear()
    path = w.save_as(str(tmp_path / "scans_library.txt"))
    assert path and w._state.text() == "Saved" and w._path.text() == path
    w.select_scans([("S1", 5)])
    w.assign_selected(notes="dry")
    assert "Unsaved changes" in w._state.text()
    w.close()                                                                     # saved on closing
    assert lb.read_library(path).entry("S1", 5).notes == "dry"
    # the next window opens the library again, with the comparison settings
    w = ui.LibraryWindow()
    settle()
    assert w.lib.path == path and w._table.rowCount() == 12 and w._kind.currentIndex() == 1
    assert w._x.currentText() == "Epoch" and w._state.text() == "Saved"
    assert w._table.cellWidget(w._row_of[("S1", 3)], ui.C_ROI).currentText() == "ROI2"
    # removing a source: undo brings it back
    w._sources.setCurrentRow(1)
    w.remove_source()
    assert w._table.rowCount() == 6
    w.undo()
    assert w._table.rowCount() == 12 and w.lib.entry("S2", 1).film == "2 nm"
    # a new library that is never saved comes back as a draft
    w.new_library(ask=False)
    assert w._table.rowCount() == 0 and w._state.text() == ""
    w.add_source(sa, [ua])
    w.close()
    w = ui.LibraryWindow()
    settle()
    assert w.lib.path is None and w._table.rowCount() == 6 and "Not saved yet" in w._state.text()
    w.close()
    assert not said
    # the data files were only read
    assert file_state([sa, ua, sb, ub]) == before

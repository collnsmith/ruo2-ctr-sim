"""CTR processing (steps 2-5) on synthetic data with a known |F|, the counting error of ROI sums, the
processing window and the plotter's several normalisation columns."""
import numpy as np
import pytest

from ctrfit.datatools import images
from ctrfit.datatools import processing as pr


def true_F(H, K, L):
    """A smooth fake rod (any positive function will do)."""
    return 50.0 / (0.2 + np.sin(np.pi * L) ** 2) + 10 * (abs(H) + abs(K))


def write_experiment(tmp_path, opt, rods, seed=0):
    """SPEC file with one hklscan per entry of rods [(scan, H, K, L0, L1, n, scale)], and a ROI sums file
    whose ROI1 is the counts the corrections predict for true_F, times the scale, monitor, transmission
    and count time (with Poisson-like errors)."""
    rng = np.random.default_rng(seed)
    spec = ["#F synth.spec", "#E 1772892000", "#D Sat Mar 07 14:00:00 2026", ""]
    sums = ["scan\timage\tROI1\tROI1_err"]
    truth = {}
    for sn, H, K, L0, L1, n, scale in rods:
        L = np.linspace(L0, L1, n)
        mon = rng.uniform(9e4, 1.1e5, n)
        tr = np.where(np.arange(n) % 3 == 0, 0.1, 1.0)
        ct = np.full(n, 2.0)
        a, b, c = pr.surface_angles(np.full(n, H), np.full(n, K), L, opt.energy_kev, opt.lattice, opt.alpha_deg)
        P, Cr, Ca = pr.corrections(a, b, c, opt)
        I = scale * true_F(H, K, L) ** 2 * P * Cr * Ca * (mon / 1e5) * tr * ct
        sig = np.sqrt(I)
        I = I + rng.normal(0, 1, n) * sig * 0.0          # exact counts: the recovery test checks the algebra
        spec += [f"#S {sn}  hklscan {H} {H} {K} {K} {L0} {L1} {n - 1} 2", "#D Sat Mar 07 14:10:00 2026",
                 "#L H  K  L  Epoch  Seconds  Monitor  transm  Detector"]
        spec += [f"{H} {K} {l:.6f} {i} {t:g} {m:.3f} {x:g} {d:.3f}" for i, (l, t, m, x, d) in
                 enumerate(zip(L, ct, mon, tr, I))]
        spec.append("")
        sums += [f"{sn}\t{i}\t{v:.10g}\t{s:.10g}" for i, (v, s) in enumerate(zip(I, sig))]
        truth[sn] = (H, K, L, true_F(H, K, L))
    (tmp_path / "s.spec").write_text("\n".join(spec) + "\n")
    folder = tmp_path / "sums"
    folder.mkdir(exist_ok=True)
    (folder / "scan_sums.txt").write_text("\n".join(sums) + "\n")
    (folder / "notes.txt").write_text("not a sums file\n")
    return tmp_path / "s.spec", folder, truth


def opts(**kw):
    base = dict(roi="ROI1", monitor="Monitor", transmission="transm", count_time="Seconds")
    return pr.Options(**dict(base, **kw))


def test_roi_sigma_matches_the_scatter_of_poisson_images():
    rng = np.random.default_rng(1)
    sig = images.ROI(10, 10, 6, 6, "signal", "S")
    bgs = [images.ROI(0, 0, 8, 30, "background", "B1"), images.ROI(30, 0, 8, 30, "background", "B2")]
    mean = np.full((30, 40), 20.0)
    mean[10:16, 10:16] += 15.0
    for name in ("Average", "Median"):
        method = images.BG_METHODS[name]
        vals, errs = [], []
        for _ in range(400):
            img = rng.poisson(mean).astype(float)
            _, sums = images.process_image(img, [sig] + bgs, method)
            vals.append(sums["S"])
            errs.append(sums["S_err"])
        assert np.std(vals) == pytest.approx(np.mean(errs), rel=0.12), name
        assert np.mean(vals) == pytest.approx(36 * 15, rel=0.05), name


def test_processing_recovers_the_structure_factor(tmp_path):
    opt = opts(match_overlaps=True)
    rods = [(10, 1, 0, 0.3, 2.0, 60, 1.0), (11, 1, 0, 1.8, 3.5, 60, 3.0),   # one rod in two scans, other scale
            (12, -1, 0, 0.5, 1.5, 40, 1.0), (13, 0, 0, 0.6, 2.8, 50, 1.0)]  # an equivalent, the specular rod
    spec, folder, truth = write_experiment(tmp_path, opt, rods)
    paths = pr.sum_files(folder)
    assert len(paths) == 1                                                       # notes.txt skipped
    results, pts, notes = pr.run_files(paths, str(spec), opt)
    assert [r.scan for r in results] == [10, 11, 12, 13]
    # every scan's F is the truth times one constant (the scale), and scan 11 is matched to scan 10
    ratios = {}
    for r in results:
        Ft = truth[r.scan][3]
        q = r.F * np.sqrt(r.scale) / Ft
        assert np.allclose(q, q[0], rtol=1e-6), r.scan
        ratios[r.scan] = q[0]
    assert ratios[11] == pytest.approx(ratios[10], rel=5e-3) and ratios[12] == pytest.approx(ratios[10], rel=5e-3)
    assert any("scan 11 scaled" in n for n in notes)
    # nothing merged: every kept point of every scan, overlaps and the (-1 0) equivalent as separate points
    assert pts.L.size == sum(int(r.keep.sum()) for r in results) == sum(r[5] for r in rods)
    assert set(pts.rod) == {"1 0", "-1 0", "0 0"} and [int((pts.scan == sn).sum()) for sn in (10, 11, 12, 13)] == \
        [60, 60, 40, 50]
    sel = np.array([x == "1 0" for x in pts.rod])
    assert np.allclose(pts.F[sel] / true_F(1, 0, pts.L[sel]), ratios[10], rtol=5e-3)
    overlap = sel & (pts.L >= 1.8) & (pts.L <= 2.0)
    assert set(pts.scan[overlap]) == {10, 11}                                       # both scans kept there
    assert not hasattr(pr, "merge") and not hasattr(pr, "EQUIVALENTS")
    # corrections: specular rod at alpha = beta, horizontal polarisation untouched there
    spec_r = results[3]
    assert np.allclose(spec_r.alpha, spec_r.beta) and np.all(spec_r.P > 0.99)          # only the 2 % vertical part
    full = pr.Options(pol_fraction=1.0)
    cos_psi = np.ones(spec_r.L.size)
    assert np.allclose(pr.corrections(spec_r.alpha, spec_r.beta, cos_psi, full)[0], 1.0)
    # outputs: the fitting app reads the CSV with the systematic floor
    from ctrfit.data.io import read_csv
    out = pr.write_points(tmp_path / "p.csv", pts, opt)
    ds = read_csv(out)
    assert ds.sys_floor == pytest.approx(0.05) and ds.L.size == pts.L.size
    pr.write_scans(tmp_path / "p_scans.txt", results)
    pr.write_genx(tmp_path / "p_genx.dat", pts)
    assert len((tmp_path / "p_genx.dat").read_text().splitlines()) == 1 + pts.L.size
    assert len((tmp_path / "p_scans.txt").read_text().splitlines()) == 2 + sum(r[5] for r in rods)


def test_options_change_the_result(tmp_path):
    opt = opts()
    spec, folder, truth = write_experiment(tmp_path, opt, [(20, 1, 1, 0.3, 3.0, 80, 1.0)])
    base, _, _ = pr.run_files(pr.sum_files(folder), str(spec), opt)
    off, _, _ = pr.run_files(pr.sum_files(folder), str(spec), opts(polarization=False, rod_interception=False))
    assert not np.allclose(base[0].F, off[0].F)
    skip, pts, _ = pr.run_files(pr.sum_files(folder), str(spec), opts(exclude_bragg=0.05))
    assert pts.L.size == int(skip[0].keep.sum())
    assert (~skip[0].keep).any() and np.all(np.abs(skip[0].L[~skip[0].keep] - np.round(skip[0].L[~skip[0].keep])) < 0.05)
    # no _err column: sqrt(|I|) with a note
    text = (folder / "scan_sums.txt").read_text().splitlines()
    (folder / "scan_sums.txt").write_text("\n".join("\t".join(l.split("\t")[:3]) for l in text) + "\n")
    r, _, notes = pr.run_files(pr.sum_files(folder), str(spec), opt)
    assert any("lower bound" in n for n in notes)
    with pytest.raises(ValueError):
        pr.run_files(pr.sum_files(folder), str(spec), opts(monitor="nope"))


def test_stationary_scans_are_processed_point_by_point(tmp_path):
    """A CV-style scan at fixed HKL: normalised and corrected per point, kept out of the rod points, not
    overlap-scaled against a rod scan of the same rod, not cut by the Bragg window; its file is read by
    the in-situ CV tool with the propagated sigma."""
    opt = opts(exclude_bragg=0.1)
    spec, folder, truth = write_experiment(tmp_path, opt, [(50, 1, 0, 0.3, 2.0, 40, 1.0),
                                                           (51, 1, 0, 1.0, 1.0, 30, 2.0)])
    results, pts, notes = pr.run_files(pr.sum_files(folder), str(spec), opt)
    rod, st = results
    assert not rod.stationary and st.stationary and st.keep.all() and st.scale == 1.0
    # monitor, transmission and count time vary point by point; the corrected intensity does not
    assert np.allclose(st.I_corr, st.I_corr[0], rtol=1e-6) and np.ptp(st.I_raw) > 0
    assert np.allclose(st.F, 2 ** 0.5 * true_F(1, 0, 1.0) * rod.F[0] / true_F(1, 0, 0.3), rtol=1e-6)
    assert pts.L.size == int(rod.keep.sum()) and pr.stationary(results) == [st]
    out = pr.write_stationary(tmp_path / "cv.dat", pr.stationary(results))
    from ctrfit.echem import external as ex
    assert ex.choices(out)[1][:2] == ["ROI1_corr", "ROI1_norm"]
    y, sig, note = ex.read_intensity_sigma(str(out), 51, 30, column="ROI1_corr")
    assert np.allclose(y, st.I_corr, rtol=1e-5) and np.allclose(sig, st.s_corr, rtol=1e-5)
    assert "sigma from" in note
    # only stationary scans: no rod points
    (tmp_path / "b").mkdir()
    spec2, folder2, _ = write_experiment(tmp_path / "b", opt, [(52, 0, 1, 0.5, 0.5, 10, 1.0)])
    res2, pts2, _ = pr.run_files(pr.sum_files(folder2), str(spec2), opt)
    assert res2[0].stationary and pts2 is None


def test_groups_keep_conditions_apart(tmp_path):
    """Same rod at two potentials (different structure: scale 1 and 3 stands in for it): by default every
    scan keeps its own intensity; with overlap scaling in one group the second scan is scaled onto the
    first (never averaged); in two groups both keep their own intensity."""
    opt = opts()
    spec, folder, truth = write_experiment(tmp_path, opt, [(70, 1, 0, 0.3, 2.0, 30, 1.0), (71, 1, 0, 0.3, 2.0, 30, 3.0),
                                                           (72, 1, 0, 1.8, 3.0, 20, 3.0)])
    paths = pr.sum_files(folder)
    plain, m0, _ = pr.run_files(paths, str(spec), opt)
    assert [r.scale for r in plain] == [1.0, 1.0, 1.0] and m0.L.size == 80          # off by default, all points
    one, m1, _ = pr.run_files(paths, str(spec), opts(match_overlaps=True))
    assert one[1].scale == pytest.approx(1 / 3, rel=1e-3) and m1.groups() == [""] and m1.L.size == 80
    groups = {70: "0.4 V", 71: "1.2 V", 72: "1.2 V"}
    two, m2, notes = pr.run_files(paths, str(spec), opts(scan_group=groups, match_overlaps=True))
    assert two[1].scale == 1.0 and two[2].scale == pytest.approx(1.0, rel=0.05)     # 72 matched to 71 only (short overlap at a Bragg peak)
    assert m2.groups() == ["0.4 V", "1.2 V"]
    a, b = m2.select("0.4 V"), m2.select("1.2 V")
    assert b.L.max() > 2.9 and a.L.max() < 2.01 and (a.L.size, b.L.size) == (30, 50)
    k71 = b.scan == 71                                                               # same L grid as scan 70
    assert np.allclose(b.L[k71], a.L) and np.allclose(b.I[k71] / a.I, 3.0, rtol=1e-6)   # the difference survives
    assert pr.safe_name("1.2 V / 5nm") == "1.2_V_5nm"
    pr.write_scans(tmp_path / "s.txt", two, opts(scan_group=groups))
    assert "1.2 V" in (tmp_path / "s.txt").read_text()


def test_roi_per_scan(tmp_path):
    opt = opts(roi=None)
    spec, folder, truth = write_experiment(tmp_path, opt, [(60, 1, 0, 0.3, 2.0, 20, 1.0), (61, 0, 1, 0.3, 2.0, 20, 1.0)])
    sums = pr.load_sums(pr.sum_files(folder))
    sums[61]["B"], sums[61]["B_err"] = 2 * sums[61]["ROI1"], sums[61]["ROI1_err"]
    del sums[60]["ROI1_err"]
    from ctrfit.datatools.specfile import parse_spec_file
    scans = parse_spec_file(str(spec))
    base, _, _ = pr.process(sums, scans, opts())
    picked, _, notes = pr.process(sums, scans, opts(scan_roi={61: "B"}))
    assert [r.roi for r in picked] == ["ROI1", "B"] and np.allclose(picked[1].I_raw, 2 * base[1].I_raw)
    # a global ROI missing in a scan: that scan's default, with a note; a missing per-scan ROI is an error
    other, _, notes = pr.process(sums, scans, opts(roi="B"))
    assert [r.roi for r in other] == ["ROI1", "B"] and any("no ROI 'B' in this scan" in n for n in notes)
    left, _, notes = pr.process(sums, scans, opts(scan_roi={60: "B"}))      # a missing per-scan ROI: left out
    assert [r.scan for r in left] == [61] and any(n.startswith("scan 60 left out: no ROI column 'B'") for n in notes)


def test_time_scans_take_hkl_from_q_and_bad_scans_are_left_out(tmp_path):
    """A loopscan has no H K L columns: its H K L is the #Q line. A scan with neither is left out with a
    note and the others are processed; only when every scan fails is it an error."""
    opt = opts()
    spec, folder, truth = write_experiment(tmp_path, opt, [(10, 1, 0, 0.3, 2.0, 30, 1.0)])
    rows = [f"{2 * i} {1000 + 2 * i} 2 1e5 1 {500 + 10 * i}" for i in range(20)]
    extra = ["", "#S 11  loopscan 20 2 0", "#D Sat Mar 07 14:20:00 2026", "#Q 1 0 0.5",
             "#L Time  Epoch  Seconds  Monitor  transm  Detector"] + rows + \
            ["", "#S 12  timescan 2 0", "#D Sat Mar 07 14:30:00 2026",
             "#L Time  Epoch  Seconds  Monitor  transm  Detector"] + rows
    with open(spec, "a") as f:
        f.write("\n".join(extra) + "\n")
    with open(folder / "scan_sums.txt", "a") as f:
        f.write("".join(f"{sn}\t{i}\t{500 + 10 * i}\t{np.sqrt(500 + 10 * i):.6g}\n" for sn in (11, 12)
                        for i in range(20)))
    results, merged, notes = pr.run_files(pr.sum_files(folder), str(spec), opt)
    assert [r.scan for r in results] == [10, 11]
    r11 = results[1]
    assert r11.stationary and np.all(r11.H == 1) and np.all(r11.K == 0) and np.all(r11.L == 0.5)
    assert np.all(np.isfinite(r11.F)) and np.all(r11.sF > 0)
    assert any(n.startswith("scan 12 left out: no H column and no #Q line") for n in notes)
    assert merged is not None and set(merged.rod) == {"1 0"}                          # the rod scan as before
    with pytest.raises(ValueError, match="and 2 more scan"):                          # every scan fails
        pr.run_files(pr.sum_files(folder), str(spec), opts(monitor="nope"))


@pytest.fixture
def qapp():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_processing_window(qapp, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    said = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(lambda *a, k=kind, **kw: said.append((k, a[2]))))
    monkeypatch.setenv("CTRFIT_SETTINGS_DIR", str(tmp_path / "settings"))
    from PyQt5.QtCore import QItemSelectionModel
    from ctrfit.app.datatools.processing import ProcessingWindow
    import json
    from ctrfit.datatools.paths import settings_file
    with open(settings_file("sxrd_processing_settings.json"), "w") as f:                # settings of the old layout
        json.dump({"_chk_overlap": True, "_equiv": "±H ±K (p2mm)"}, f)
    opt = opts()
    spec, folder, truth = write_experiment(tmp_path, opt, [(30, 1, 0, 0.3, 2.0, 40, 1.0), (31, 1, 0, 1.8, 3.0, 30, 2.0),
                                                       (32, 1, 0, 0.5, 0.5, 20, 1.0)])
    lines = (folder / "scan_sums.txt").read_text().splitlines()                     # a second ROI, 4x ROI1
    lines = [lines[0] + "\tROI2\tROI2_err"] + [l + f"\t{4 * float(l.split()[2]):.10g}\t{2 * float(l.split()[3]):.10g}"
                                               for l in lines[1:]]
    (folder / "scan_sums.txt").write_text("\n".join(lines) + "\n")
    w = ProcessingWindow()
    assert not w._chk_scale_overlap.isChecked()                                     # the old 'on' is not restored
    w.show()
    w.open_spec(str(spec))
    w.open_sums_folder(str(folder))
    for combo, text in ((w._monitor, "Monitor"), (w._transm, "transm"), (w._ctime, "Seconds")):
        combo.setCurrentText(text)
    w.reprocess()
    assert w._table.rowCount() == 3 and w._roi.currentText() == "ROI1" and not said
    w._table.selectRow(1)                                                          # preview of scan 31
    assert len(w._fig.axes) == 2 and "Scan 31" in w._fig.axes[0].get_title()
    assert float(w._table.item(1, 5).text()) == 1.0                                  # not scaled by default
    assert "other scans of this rod" in [t.get_text() for t in w._fig.axes[1].get_legend().get_texts()]
    w._chk_scale_overlap.setChecked(True)
    w.reprocess()
    assert float(w._table.item(1, 5).text()) == pytest.approx(0.5, rel=0.03)         # matched to scan 30
    assert w._points.L.size == int(w._results[0].keep.sum() + w._results[1].keep.sum())   # nothing merged
    raw = w._results[0].I_raw.copy()
    combo = w._table.cellWidget(0, 6)                                               # scan 30 uses ROI2
    assert combo.currentText() == "(default)" and combo.findText("ROI2") > 0
    combo.setCurrentText("ROI2")
    w.reprocess()
    assert w._results[0].roi == "ROI2" and np.allclose(w._results[0].I_raw, 4 * raw)
    assert w._results[1].roi == "ROI1" and w._table.cellWidget(0, 6).currentText() == "ROI2"
    out = w.save_to(str(tmp_path / "one.csv"))
    assert [p.split("/")[-1] for p in out] == ["one_scans.txt", "one.csv", "one_genx.dat", "one_stationary.dat"]
    # groups: select scans 30 and 31, type a group once -> both get it; scan 31 alone in another
    sm = w._table.selectionModel()          # not selectRow: it follows the keyboard state an earlier test left
    sm.select(w._table.model().index(0, 0), QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
    sm.select(w._table.model().index(1, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
    w._table.item(0, 7).setText("0.4 V")
    assert w._table.item(1, 7).text() == "0.4 V"
    w._table.clearSelection()
    w._table.item(1, 7).setText("1.2 V")
    w.reprocess()
    assert w._points.groups() == ["0.4 V", "1.2 V"] and w._results[1].scale == 1.0
    out = w.save_to(str(tmp_path / "res.csv"))
    assert [p.split("/")[-1] for p in out] == ["res_scans.txt", "res_0.4_V.csv", "res_0.4_V_genx.dat", "res_1.2_V.csv",
                                               "res_1.2_V_genx.dat", "res_stationary.dat"]
    assert "group '1.2 V'" in w._summary.toPlainText()
    w._table.item(1, 0).setCheckState(0)                                             # leave scan 31 out
    w.reprocess()
    assert all(L <= 2.0 + 1e-6 for L in w._points.L) and w._points.groups() == ["0.4 V"]
    assert "stationary" in w._table.item(2, 2).text() and "1 stationary scan" in w._summary.toPlainText()
    w._table.selectRow(2)
    assert "stationary" in w._fig.axes[0].get_title() and w._fig.axes[1].get_xlabel() == "SPEC point"
    w.close()
    w2 = ProcessingWindow()                                                         # settings remembered
    assert w2._spec_path.text() == str(spec)
    qapp.processEvents()
    w2.refresh()
    assert w2._monitor.currentText() == "Monitor" and w2._table.rowCount() == 3 and w2._chk_scale_overlap.isChecked()
    assert w2._table.cellWidget(0, 6).currentText() == "ROI2" and w2._table.item(1, 0).checkState() == 0
    assert w2._table.item(1, 7).text() == "1.2 V"
    w2.close()


def test_plotter_divides_by_several_normalisation_columns(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("CTRFIT_SETTINGS_DIR", str(tmp_path / "settings"))
    from ctrfit.app.datatools.plotter import MainWindow
    opt = opts()
    spec, folder, truth = write_experiment(tmp_path, opt, [(40, 1, 0, 0.3, 2.0, 20, 1.0)])
    w = MainWindow()
    w._load_spec_file(str(spec))
    w._col_sel.set_selection("L", ["Detector"], "Monitor × transm")
    assert w._col_sel.get_selection()[2] == "Monitor × transm"
    w._update_plot()
    s = w._spec_scans[40]
    y = w._canvas.ax.get_lines()[0].get_ydata()
    assert np.allclose(y, s.data["Detector"] / (s.data["Monitor"] * s.data["transm"]))
    assert "Monitor × transm" in w._canvas.ax.get_ylabel()
    w.close()

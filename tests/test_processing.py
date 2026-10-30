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
    opt = opts(match_overlaps=True, equivalents=pr.EQUIVALENTS[1])
    rods = [(10, 1, 0, 0.3, 2.0, 60, 1.0), (11, 1, 0, 1.8, 3.5, 60, 3.0),   # one rod in two scans, other scale
            (12, -1, 0, 0.5, 1.5, 40, 1.0), (13, 0, 0, 0.6, 2.8, 50, 1.0)]  # an equivalent, the specular rod
    spec, folder, truth = write_experiment(tmp_path, opt, rods)
    paths = pr.sum_files(folder)
    assert len(paths) == 1                                                       # notes.txt skipped
    results, merged, notes = pr.run_files(paths, str(spec), opt)
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
    # merged: (1 0) and (-1 0) are one rod, the overlap and the equivalent agree (R_sym ~ 0)
    assert set(merged.rod) == {"1 0", "0 0"} and merged.n_multiple > 0 and merged.r_sym < 0.03   # L grids differ by < l_tol
    sel = np.array([x == "1 0" for x in merged.rod])
    assert np.allclose(merged.F[sel] / true_F(1, 0, merged.L[sel]), ratios[10], rtol=0.03)
    # corrections: specular rod at alpha = beta, horizontal polarisation untouched there
    spec_r = results[3]
    assert np.allclose(spec_r.alpha, spec_r.beta) and np.all(spec_r.P > 0.99)          # only the 2 % vertical part
    full = pr.Options(pol_fraction=1.0)
    cos_psi = np.ones(spec_r.L.size)
    assert np.allclose(pr.corrections(spec_r.alpha, spec_r.beta, cos_psi, full)[0], 1.0)
    # outputs: the fitting app reads the CSV with the systematic floor
    from ctrfit.data.io import read_csv
    out = pr.write_merged(tmp_path / "p.csv", merged, opt)
    ds = read_csv(out)
    assert ds.sys_floor == pytest.approx(0.05) and ds.L.size == merged.L.size
    pr.write_scans(tmp_path / "p_scans.txt", results)
    pr.write_genx(tmp_path / "p_genx.dat", merged)
    assert len((tmp_path / "p_scans.txt").read_text().splitlines()) == 2 + sum(r[5] for r in rods)


def test_options_change_the_result(tmp_path):
    opt = opts()
    spec, folder, truth = write_experiment(tmp_path, opt, [(20, 1, 1, 0.3, 3.0, 80, 1.0)])
    base, _, _ = pr.run_files(pr.sum_files(folder), str(spec), opt)
    off, _, _ = pr.run_files(pr.sum_files(folder), str(spec), opts(polarization=False, rod_interception=False))
    assert not np.allclose(base[0].F, off[0].F)
    skip, merged, _ = pr.run_files(pr.sum_files(folder), str(spec), opts(exclude_bragg=0.05))
    assert (~skip[0].keep).any() and np.all(np.abs(skip[0].L[~skip[0].keep] - np.round(skip[0].L[~skip[0].keep])) < 0.05)
    # no _err column: sqrt(|I|) with a note
    text = (folder / "scan_sums.txt").read_text().splitlines()
    (folder / "scan_sums.txt").write_text("\n".join("\t".join(l.split("\t")[:3]) for l in text) + "\n")
    r, _, notes = pr.run_files(pr.sum_files(folder), str(spec), opt)
    assert any("lower bound" in n for n in notes)
    with pytest.raises(ValueError):
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
    from ctrfit.app.datatools.processing import ProcessingWindow
    opt = opts()
    spec, folder, truth = write_experiment(tmp_path, opt, [(30, 1, 0, 0.3, 2.0, 40, 1.0), (31, 1, 0, 1.8, 3.0, 30, 2.0)])
    w = ProcessingWindow()
    w.show()
    w.open_spec(str(spec))
    w.open_sums_folder(str(folder))
    for combo, text in ((w._monitor, "Monitor"), (w._transm, "transm"), (w._ctime, "Seconds")):
        combo.setCurrentText(text)
    w.reprocess()
    assert w._table.rowCount() == 2 and w._roi.currentText() == "ROI1" and not said
    w._table.selectRow(1)                                                          # preview of scan 31
    assert len(w._fig.axes) == 2 and "Scan 31" in w._fig.axes[0].get_title()
    assert float(w._table.item(1, 5).text()) == pytest.approx(0.5, rel=0.03)         # matched to scan 30
    w._table.item(1, 0).setCheckState(0)                                             # leave scan 31 out
    w.reprocess()
    assert all(L <= 2.0 + 1e-6 for L in w._merged.L)
    out = w.save_to(str(tmp_path / "res.csv"))
    assert [p.split("/")[-1] for p in out] == ["res_scans.txt", "res.csv", "res_genx.dat"]
    w.close()
    w2 = ProcessingWindow()                                                         # settings remembered
    assert w2._spec_path.text() == str(spec)
    qapp.processEvents()
    w2.refresh()
    assert w2._monitor.currentText() == "Monitor" and w2._table.rowCount() == 2
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

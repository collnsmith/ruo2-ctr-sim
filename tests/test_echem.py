"""In-situ CV: SPEC reader, potentiostat readers, hand-defined CV, sync modes, background, cycles,
sigmoid fits, CTR-model prediction and fits, session, CLI and window (offscreen)."""
import json
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from ctrfit.echem import analysis as an
from ctrfit.echem import predict as pr
from ctrfit.echem import sync as sy
from ctrfit.echem.ec import cv_knots, make_cv, read_ec, read_mpr, read_mpt, read_table, write_mpt
from ctrfit.echem.session import Options, Session
from ctrfit.echem.spec import SpecFile, parse_spec_date, to_seconds
from ctrfit.echem.synthetic import TRUTH, make_experiment, write_experiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("insitu")
    out = {}
    for kind in ("loopscan", "phi"):
        spec, mpt, tr = write_experiment(d / kind, kind)
        out[kind] = (spec, mpt, tr)
    return out


def session(files, kind="loopscan", **opt):
    spec, mpt, tr = files[kind]
    s = Session(Options(monitor="Monitor", n_transitions=2, **opt))
    s.load_spec(spec)
    if not opt.get("cv_manual"):
        s.load_ec(mpt)
    return s, tr


# ---------------------------------------------------------------------------- readers
def test_spec_file_scans_and_times():
    text, ec, tr = make_experiment("loopscan")
    f = SpecFile(text=text)
    assert [s.number for s in f.scans] == [1, 2] and [s.number for s in f.stationary_scans()] == [2]
    s = f.scan(2)
    assert s.kind == "loopscan" and s.stationary and s.hkl == (0, 1, 1.5) and s.count_time == 1
    assert s.default_time_column() == ("Time", "elapsed") and s.default_counter() == "Detector"
    assert s.default_monitor() == "Monitor" and f.motor_names[:2] == ["Two Theta", "Theta"]
    t_el, t_ep = s.times("Time", "elapsed"), s.times("Epoch", "epoch")
    np.testing.assert_allclose(t_el, t_ep, atol=1e-6)                 # #D + Time == #D(file) + Epoch
    assert t_el[0] == pytest.approx(to_seconds(tr["scan_start"]) + 0.5 + TRUTH["dead"])     # middle of the count
    assert s.times("Epoch", "epoch", "middle")[0] - t_ep[0] == pytest.approx(0.5)     # stamp already mid-count
    phi = SpecFile(text=make_experiment("phi")[0]).scan(2)
    assert phi.stationary and phi.kind == "ascan" and phi.default_time_column() == ("Epoch", "epoch")
    assert not SpecFile(text=text).scan(1).stationary                  # ascan phi 10 30 moves
    assert parse_spec_date("Sat Mar 07 14:00:00 2026") == datetime(2026, 3, 7, 14)
    with pytest.raises(KeyError):
        s.column("nope")


def test_potentiostat_readers(tmp_path, monkeypatch):
    ec = make_cv(0.5, 10, 1.0, 0.4, 0.05, 2, start=datetime(2026, 3, 7, 14, 30, 1, 250000))
    write_mpt(tmp_path / "a.mpt", ec)
    back = read_ec(tmp_path / "a.mpt")
    assert back.start == ec.start and back.time.size == ec.time.size
    np.testing.assert_allclose(back.potential, ec.potential, atol=1e-6)
    (tmp_path / "b.csv").write_text("time/s,Ewe/V\n0,0.1\n1,0.2\n2,0.3\n")
    t = read_table(tmp_path / "b.csv")
    assert t.potential.tolist() == [0.1, 0.2, 0.3] and t.start is None
    comma = (tmp_path / "a.mpt").read_text(encoding="latin-1").replace(".", ",").replace("14:30:01,250",
                                                                                          "14:30:01.250")
    assert read_mpt(text=comma).potential[5] == pytest.approx(ec.potential[5], abs=1e-6)    # decimal commas
    # .mpr through galvani (faked here: the real one needs a binary file)
    data = np.zeros(4, dtype=[("time/s", "f8"), ("Ewe/V", "f4"), ("cycle number", "f8")])
    data["time/s"], data["Ewe/V"] = [0, 1, 2, 3], [0.5, 0.6, 0.7, 0.6]
    fake = types.SimpleNamespace(MPRfile=lambda p: types.SimpleNamespace(data=data, timestamp=ec.start))
    monkeypatch.setitem(sys.modules, "galvani", types.SimpleNamespace(BioLogic=fake))
    monkeypatch.setitem(sys.modules, "galvani.BioLogic", fake)
    m = read_mpr(tmp_path / "x.mpr")
    assert m.start == ec.start and m.potential.tolist() == pytest.approx([0.5, 0.6, 0.7, 0.6])
    monkeypatch.setitem(sys.modules, "galvani", None)
    with pytest.raises(RuntimeError, match="galvani"):
        read_mpr(tmp_path / "x.mpr")


def test_hand_defined_cv():
    t, E, c = cv_knots(0.5, 100, 1.4, 0.4, 0.01, 2, "up")
    assert E.tolist() == [0.5, 0.5, 1.4, 0.4, 0.5, 1.4, 0.4, 0.5] and t[1] == 100
    assert t[-1] == pytest.approx(100 + 2 * (0.9 + 1.0 + 0.1) / 0.01)
    ec = make_cv(0.5, 100, 1.4, 0.4, 0.01, 2, "down")
    assert ec.potential.min() == pytest.approx(0.4) and ec.potential[np.searchsorted(ec.time, 110)] < 0.5
    assert set(np.unique(ec.cycle)) == {0, 1, 2}
    for bad in (dict(rate=0), dict(E_upper=0.3), dict(n_cycles=0), dict(first="left")):
        kw = dict(E_hold=0.5, t_hold=1, E_upper=1.4, E_lower=0.4, rate=0.01, n_cycles=1, first="up")
        kw.update(bad)
        with pytest.raises(ValueError):
            cv_knots(**kw)


# ---------------------------------------------------------------------------- sync
def test_sync_modes_put_the_right_potential_on_each_point(files):
    truth_V = None
    for mode, off in (("absolute", TRUTH["clock_offset"]), ("manual", -TRUTH["cv_delay"])):
        s, tr = session(files, sync_mode=mode, offset=off)
        r = s.run(upto="sync")
        ok = np.isfinite(r.V)
        if truth_V is None:
            truth_V = r.V
        np.testing.assert_allclose(r.V[ok], truth_V[ok], atol=1e-9)      # both modes agree exactly
        assert ok.sum() > 0.85 * ok.size and not ok[:50].any()            # the first 60 s are before the EC file
    # sweep start of the hand-defined CV and the intensity onset
    s, _ = session(files)
    t0, rate = sy.detect_sweep_start(s.ec.time, s.ec.potential)
    assert t0 == pytest.approx(TRUTH["hold"], abs=0.5) and rate == pytest.approx(TRUTH["rate"], rel=0.05)
    # events: the same lag on both axes, so the two offsets differ by the true difference
    s_abs, _ = session(files, sync_mode="absolute")
    s_man, _ = session(files, sync_mode="manual")
    d = s_abs.events_offset() - s_man.events_offset()
    assert d == pytest.approx(TRUTH["clock_offset"] + TRUTH["cv_delay"], abs=0.2)
    lag = TRUTH["clock_offset"] - s_abs.events_offset()
    assert 5 < lag < 40                                                   # intensity changes after the sweep starts
    s_ev, _ = session(files, sync_mode="events", onset=TRUTH["cv_delay"] + TRUTH["hold"])  # typed onset: exact
    assert s_ev.run(upto="sync").alignment.offset == pytest.approx(TRUTH["clock_offset"] - 0.6, abs=0.6)


def test_branches_and_cycles(files):
    s, _ = session(files, offset=TRUTH["clock_offset"])
    r = s.run(upto="sync")
    assert set(np.unique(r.cycle)) == {0, 1, 2, 3}
    assert np.all(r.branch[r.cycle == 0] == 0)
    assert an.parse_cycles("2-3", r.cycle) == [2, 3] and an.parse_cycles("all", r.cycle) == [1, 2, 3]
    with pytest.raises(ValueError):
        an.parse_cycles("7", r.cycle)
    up = np.flatnonzero((r.branch == 1) & (r.cycle == 2))
    runs = np.split(up, np.flatnonzero(np.diff(up) > 1) + 1)          # 0.5 -> 1.4 V, then 0.4 -> 0.5 V
    assert len(runs) == 2 and all(np.all(np.diff(r.V[run]) > 0) for run in runs)


# ---------------------------------------------------------------------------- analysis
def test_background_fits_and_corrections():
    t = np.linspace(0, 300, 301)
    y = 1000 * (1 + 0.05 * np.exp(-t / 100.0))
    bg = an.fit_background(t, y, 300, "exponential")
    assert bg.params[2] == pytest.approx(100, rel=1e-3) and bg.rms < 1e-6
    corr, _ = bg.correct(t, y)
    np.testing.assert_allclose(corr, y[-1], rtol=1e-6)                  # flat, at the level of the window end
    lin = an.fit_background(t, 5 + 0.1 * t, 50, "linear", "subtract")
    np.testing.assert_allclose(lin.correct(t, 5 + 0.1 * t)[0], 10.0, atol=1e-9)
    assert lin.notes                                                     # extrapolated far beyond the window
    with pytest.raises(ValueError):
        an.fit_background(t, y, 1, "exponential")


def test_sigmoid_fit_recovers_transitions():
    rng = np.random.default_rng(4)
    V = np.linspace(0.4, 1.4, 300)
    y = 100 - 60 * an.sigmoid((V - 0.8) / 0.03) - 30 * an.sigmoid((V - 1.1) / 0.05)
    s = np.full(V.size, 0.5)
    f = an.fit_sigmoids(V, y + rng.normal(0, 0.5, V.size), s, n=2)
    for t, (E, w, st) in zip(f.transitions, ((0.8, 0.03, -60), (1.1, 0.05, -30))):
        assert abs(t.E0 - E) < 4 * t.E0_err + 1e-4 and abs(t.width - w) < 4 * t.width_err + 1e-4
        assert t.step == pytest.approx(st, rel=0.03)
    assert 0.7 < f.red_chi2 < 1.4 and f.transitions[0].n_apparent == pytest.approx(an.RT_F / 0.03, rel=0.05)
    other = an.fit_sigmoids(V - 0.02, y, s, n=2, branch=-1)
    h = an.hysteresis(f, other)
    assert [round(x, 3) for x, _ in h] == [0.02, 0.02]
    b = an.bin_by_potential(V, y, s, np.ones(V.size, int), np.ones(V.size, int), [1], 0.05)
    assert len(b) == 1 and b[0].n.sum() == V.size and b[0].V.size == 20


# ---------------------------------------------------------------------------- CTR model
def test_point_model_is_exact_and_paths_are_physical():
    pm = pr.PointModel(hkl=(1, 0, 2.3))
    assert pm.exact
    rng = np.random.default_rng(2)
    for _ in range(5):
        c = rng.dirichlet(np.ones(4))[:3]
        assert pm.intensity(c)[0] == pytest.approx(pm.direct(c), rel=1e-9)
    V = np.linspace(0, 2, 200)
    C = pr.path_fractions(V, ["H2O", "OH", "O"], [1.0, 0.9], [0.05, 0.05], theta=0.8)   # overlapping, reversed
    assert np.all(C >= 0) and np.allclose(C.sum(axis=1), 0.8)
    assert pr.parse_states("h2o > OH, empty") == ["H2O", "OH", "empty"]
    with pytest.raises(ValueError):
        pr.parse_states("H2O -> CO")
    with pytest.raises(ValueError):
        pr.parse_transitions("0.8, 0.03", 2)
    # two-state path: the inversion is exact where the intensity is monotonic
    pm0 = pr.PointModel(hkl=(0, 1, 1.5))
    V = np.linspace(0.6, 1.0, 80)
    Ctrue = pr.path_fractions(V, ["H2O", "OH"], [0.8], [0.03])
    cov = pr.coverage_from_intensity(pm0, pm0.intensity(Ctrue), ["H2O", "OH"])
    np.testing.assert_allclose(cov.fractions, Ctrue, atol=2e-3)
    assert not cov.out_of_range.any()


def test_session_fits_the_path_through_the_ctr_model(files):
    for kind in ("loopscan", "phi"):
        s, tr = session(files, kind, offset=TRUTH["clock_offset"])
        s.run()
        p = s.fit_path()
        np.testing.assert_allclose(p.E0, TRUTH["E0"], atol=2e-3)
        np.testing.assert_allclose(p.widths, TRUTH["widths"], rtol=0.05)
        assert p.cathodic_shift == pytest.approx(TRUTH["cathodic_shift"], abs=3e-3) and p.red_chi2 < 1.5
        cov = s.coverage()
        ok = np.isfinite(s.res.V)
        assert np.nanmean(cov.fractions[ok][:, 2][s.res.V[ok] > 1.3]) > 0.9           # O at the top
        assert np.nanmean(cov.fractions[ok][:, 0][s.res.V[ok] < 0.6]) > 0.9           # H2O at the bottom
        assert set(s.res.fits) == {1, -1} and len(s.res.hysteresis) == 2
    # the same with the CV typed in by hand instead of the .mpt
    m, _ = session(files, cv_manual=True, cv_delay=TRUTH["cv_delay"], cv_hold_s=TRUTH["hold"], cv_rate=10.0,
                   cv_cycles=3, cv_lower=0.4, cv_upper=1.4, cv_hold_V=0.5)
    m.run()
    np.testing.assert_allclose(m.fit_path().E0, TRUTH["E0"], atol=2e-3)
    assert "hand-defined CV" in m.summary()


def test_session_outputs(files, tmp_path):
    s, _ = session(files, offset=TRUTH["clock_offset"], background="exponential")
    s.run()
    s.fit_path()
    s.coverage()
    s.export_csv(tmp_path / "d.csv")
    head = (tmp_path / "d.csv").read_text().splitlines()[0].split(",")
    assert {"t_s", "V", "I_corr", "branch", "cycle", "frac_OH"} <= set(head)
    s.export_json(tmp_path / "r.json")
    d = json.loads((tmp_path / "r.json").read_text())
    assert d["path_fit"]["E0"][0] == pytest.approx(0.8, abs=2e-3) and "anodic" in d["fits"]
    o = Options.from_dict(json.loads(json.dumps(s.opt.to_dict())))
    assert o == s.opt
    text = s.summary()
    assert "Path fit" in text and "Hysteresis" in text and "Background: exponential" in text
    with pytest.raises(ValueError):
        Session().run()


def test_cli_cv(files, tmp_path):
    spec, mpt, _ = files["phi"]
    code = "import sys; sys.modules['xraydb'] = None; from ctrfit.cli import main; sys.exit(main())"
    env = dict(__import__("os").environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, "-c", code, "cv", str(spec), "--cv", "0.5,300,0.4,1.4,10,3,up,60",
                        "--monitor", "Monitor", "--transitions", "2", "--fit-path", "--out", str(tmp_path / "o")],
                       capture_output=True, text=True, env=env, timeout=180)
    assert r.returncode == 0, r.stderr
    assert "Path fit" in r.stdout and (tmp_path / "o" / "ctr_model.png").exists()
    r = subprocess.run([sys.executable, "-c", code, "cv", str(spec)], capture_output=True, text=True, env=env,
                       timeout=120)
    assert r.returncode != 0 and "--ec" in r.stderr


def test_echem_window(files, tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets
    from ctrfit.app.echem_window import EchemWindow
    msgs = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: msgs.append(a[2])))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = EchemWindow()
    w.show()
    spec, mpt, _ = files["phi"]
    w.open_spec(str(spec))
    assert w.scan_combo.count() == 1 and w.counter.currentText() == "Detector"
    w.only_stationary.setChecked(False)
    assert w.scan_combo.count() == 2 and w.scan_combo.currentData() == 2
    w.monitor.setCurrentText("Monitor")
    w.open_ec(str(mpt))
    w.offset.setValue(TRUTH["clock_offset"])
    w.n_trans.setValue(2)
    w.update_all()
    assert not msgs, msgs
    assert w.p_iv.figure.axes and "Anodic sweep" in w.summary.toPlainText()
    w.fit_path()
    assert not msgs, msgs
    assert abs(w.session.res.path_fit.E0[0] - 0.8) < 2e-3
    w.use_fitted_transitions()
    assert w.transitions.text().startswith("0.80")
    w.coverage()
    w.offset_from_events()
    assert not msgs and w.offset.value() < TRUTH["clock_offset"]           # the event lag
    w.sync_mode.setCurrentIndex(w.sync_mode.findData("events"))
    assert not w.offset.isEnabled() and w.onset.isEnabled()
    w.use_manual.setChecked(True)                                          # the CV typed in
    w.cv_delay.setValue(TRUTH["cv_delay"])
    w.sync_mode.setCurrentIndex(w.sync_mode.findData("absolute"))
    w.offset.setValue(0.0)
    w.fit_path()
    assert not msgs and abs(w.session.res.path_fit.E0[1] - 1.1) < 2e-3
    w.export_csv(str(tmp_path / "d.csv"))
    w.export_results(str(tmp_path / "r.json"))
    w.save_session(str(tmp_path / "s.json"))
    assert (tmp_path / "d.csv").exists() and (tmp_path / "r.json").exists()
    w2 = EchemWindow()
    w2.open_session(str(tmp_path / "s.json"))
    assert not msgs, msgs
    assert w2.use_manual.isChecked() and w2.cv_delay.value() == TRUTH["cv_delay"] and w2.session.res.t is not None
    w.hkl.setText("1 2")
    w.update_all()
    assert msgs and "HKL" in msgs[-1]                                     # bad input: message, window alive
    w.close()
    w2.close()


# ---------------------------------------------------------------------------- real-file features
def _cut_out(text):
    """The scans without the file header (#E, #D, #O), as when scans are copied out of a SPEC file."""
    return text[text.index("#S 1"):]


def test_scans_without_file_header_point_numbers_and_loop_shift():
    text, ec, tr = make_experiment("phi", cathodic_shift=0.0)
    f = SpecFile(text=_cut_out(text))
    assert f.file_date is None and f.scan(2).stationary
    full = SpecFile(text=text).scan(2).times()
    np.testing.assert_allclose(f.scan(2).times(), full, atol=TRUTH["dead"] + 1e-6)   # #D-anchored epoch
    s = Session(Options(monitor="Monitor", n_transitions=2, cv_manual=True, cv_hold_V=0.5, cv_hold_s=TRUTH["hold"],
                        cv_lower=0.4, cv_upper=1.4, cv_rate=10.0, cv_cycles=0))
    s.spec = f
    s.opt.scan = 2
    t = s.scan.times()
    sweep0 = to_seconds(tr["cv0"]) + TRUTH["hold"]
    k = int(np.argmin(np.abs(t - sweep0)))
    s.opt.cv_start_point, s.opt.cv_end_point = k, s.scan.npts - 20
    r = s.run(upto="fit")
    assert r.alignment.ec_t[-1] - r.alignment.ec_t[0] == pytest.approx(t[-20] - t[k] + TRUTH["hold"], abs=0.01)
    assert set(np.unique(r.cycle)) >= {1, 2, 3}                                       # cycles until stopped
    assert r.period_I == pytest.approx(200.0, rel=0.03)                              # 2 x 1.0 V / 10 mV/s
    hold_marked = Session(Options(**dict(s.opt.to_dict(), cv_start_point=k - int(TRUTH["hold"] / 1.1),
                                         cv_start_marks="hold")))
    hold_marked.spec, hold_marked.opt.hkl = f, None
    assert hold_marked.run(upto="sync").alignment.ec_t[0] == pytest.approx(r.alignment.ec_t[0], abs=1.5)
    # a deliberate timing error is found by closing the anodic / cathodic loop (no true hysteresis here)
    s.opt.offset = 12.0
    best, m_best, m0, _ = s.loop_offset(-30, 30, 2)
    assert best + 12.0 == pytest.approx(t[k] - sweep0, abs=1.5) and m_best < 0.5 * m0
    assert s.opt.offset == 12.0                                                       # options restored


def test_transmission_and_off_rod_hkl():
    text, ec, tr = make_experiment("loopscan")
    lines = text.splitlines()
    i = next(j for j, ln in enumerate(lines) if ln.startswith("#L Time"))
    lines[i] += "  transm"
    j = i + 1
    while j < len(lines) and lines[j] and not lines[j].startswith("#"):
        lines[j] += " 0.25"
        j += 1
    lines = [ln.replace("#Q 0 1 1.5", "#Q -0.0003 1.0057 1.5") for ln in lines]
    f = SpecFile(text="\n".join(lines))
    s = Session(Options(transmission="transm", per_second=False))
    s.spec, s.opt.scan = f, 2
    raw, I, sig = s.intensity()
    np.testing.assert_allclose(I, raw / 0.25)
    np.testing.assert_allclose(sig, np.sqrt(np.maximum(raw, 1)) / 0.25)
    assert s.scan.default_transmission() == "transm" and s.hkl() == (0.0, 1.0, 1.5)
    s.opt.hkl = (0.3, 1, 2)
    with pytest.raises(ValueError, match="rod"):
        s.hkl()


# ---------------------------------------------------------------------------- near-surface oxygen
def test_near_surface_oxygen_in_the_cv_model(files, tmp_path):
    pm = pr.PointModel(hkl=(0, 1, 3.25))
    c = np.array([0.3, 0.4, 0.1])
    I = pm.intensity(np.tile(c, (3, 1)), dB=np.array([0.0, 1.3, 2.7]), du=np.array([0.0, 0.001, -0.002]))
    assert I == pytest.approx([pm.direct(c, b, u) for b, u in ((0, 0), (1.3, 0.001), (2.7, -0.002))], rel=1e-4)
    assert pm.intensity(c, 2.0)[0] < 0.8 * pm.intensity(c)[0]                     # odd rod near Bragg: drops
    with pytest.raises(ValueError, match="ns_tl"):
        pr.PointModel(dict(ns_tl=0), hkl=(0, 1, 3.25)).intensity(c, 1.0)
    ns = pr.NearSurface(2.0)
    V = np.linspace(0.4, 1.4, 50)
    np.testing.assert_allclose(ns.extent(V, [0.8], [0.03]), an.sigmoid((V - 0.8) / 0.03))
    # synthetic CV on an odd rod near Bragg with a known near-surface change: fit recovers it
    spec, mpt, tr = write_experiment(tmp_path, "loopscan", hkl=(0, 1, 3.25), ns_dB=2.0, states=("H2O", "OH"),
                                     E0=(0.8,), widths=(0.04,))
    s = Session(Options(monitor="Monitor", offset=TRUTH["clock_offset"], states="H2O -> OH",
                        transitions="0.75, 0.05", fit_ns_B=True))
    s.load_spec(spec)
    s.load_ec(mpt)
    s.run()
    p = s.fit_path()
    assert p.ns.dB == pytest.approx(2.0, abs=max(4 * p.ns_err["dB"], 0.1))
    assert abs(p.E0[0] - 0.8) < 3 * p.E0_err[0]
    plain = Session(Options(**dict(s.opt.to_dict(), fit_ns_B=False)))
    plain.spec, plain.ec = s.spec, s.ec
    plain.run()
    assert plain.fit_path().red_chi2 > 3 * p.red_chi2                               # without it the fit is poor
    assert "near-surface O" in s.summary()
    s.coverage()
    assert np.nanmean(s.res.coverage.fractions[s.res.V > 1.2, 1]) > 0.9               # OH at the top
    d = s.export_json(tmp_path / "r.json")
    assert "near_surface" in json.loads(Path(d).read_text())["path_fit"]


def test_window_near_surface_fields(files, tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets
    from ctrfit.app.echem_window import EchemWindow
    msgs = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: msgs.append(a[2])))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = EchemWindow()
    spec, mpt, _ = files["loopscan"]
    w.open_spec(str(spec))
    w.open_ec(str(mpt))
    w.offset.setValue(TRUTH["clock_offset"])
    w.ns_dB.setValue(1.5)
    w.ns_tl.setText("4")
    w.ns_E0.setText("0.9")
    w.update_all()
    assert not msgs, msgs
    o = w.session.opt
    assert (o.ns_dB, o.ns_tl, o.ns_E0) == (1.5, 4, 0.9) and w.session.point_model().model.ns_tl == 4
    assert "Near-surface O change" in w.summary.toPlainText()
    assert w.session.res.sim["ns_extent"] is not None
    w.save_session(str(tmp_path / "s.json"))
    w2 = EchemWindow()
    w2.open_session(str(tmp_path / "s.json"))
    assert w2.ns_dB.value() == 1.5 and w2.ns_tl.text() == "4" and w2.ns_E0.text() == "0.9"
    w.ns_width.setText("-1")
    w.update_all()
    assert msgs and "width" in msgs[-1]
    w.close()
    w2.close()


# ---------------------------------------------------------------------------- joint fit of several scans
@pytest.fixture(scope="module")
def joint_scans(tmp_path_factory):
    """Four stationary CVs at different L on (0 1) from one truth: H2O -> OH at 0.8 V, near-surface O
    B +2 A^2 in the top 2 trilayers, OH oxygen 2.1 A above Ru (default 2.0)."""
    from ctrfit.echem.joint import JointScan
    d = tmp_path_factory.mktemp("joint")
    scans = []
    for i, L in enumerate((2.75, 3.16, 3.25, 1.25)):
        sp, mpt, _ = write_experiment(d / str(i), "loopscan", seed=10 + i, hkl=(0, 1, L), states=("H2O", "OH"),
                                      E0=(0.8,), widths=(0.04,), ns_dB=2.0, sim=dict(oh_z=2.1, ns_tl=2))
        s = Session(Options(monitor="Monitor", offset=TRUTH["clock_offset"], states="H2O -> OH",
                            transitions="0.8, 0.04"))
        s.load_spec(sp)
        s.load_ec(mpt)
        s.run()
        scans.append(JointScan.from_session(s, f"L{L}"))
    return scans, d


def test_structure_point_is_exact():
    from ctrfit.core.ctrmodel import CTRModel
    from ctrfit.core.settings import DEFAULTS
    from ctrfit.echem.joint import StructurePoint
    for over, hkl in ((dict(ns_tl=2), (0, 1, 3.16)), (dict(ns_tl=3, mix_mode="domains"), (1, 0, 2.25)),
                      (dict(ns_tl=4, relax_001=0.5, thickness_nm=10.0), (1, 0, 2.25))):
        sp = StructurePoint(CTRModel(dict(DEFAULTS, **over)), hkl)
        c = np.array([0.2, 0.5, 0.1])
        assert sp.intensity(c, 1.7)[0] == pytest.approx(sp.direct(c, 1.7), rel=1e-8)
        assert sp.check_error < 1e-8
    assert len(StructurePoint(CTRModel(dict(DEFAULTS, ns_tl=4, relax_001=0.5, thickness_nm=10.0)), (1, 0, 2.25)).Q) == 2


def test_joint_fit_recovers_shared_parameters(joint_scans, tmp_path):
    from ctrfit.echem.joint import JointFit
    scans, _ = joint_scans
    j = JointFit(scans, dict(ns_tl=2), states=("H2O", "OH"), E0=(0.75,), widths=(0.05,))
    j.set("cathodic_shift", free=True)
    j.set("ns_dB", 1.0, free=True)
    j.set("oh_z", free=True)
    r = j.fit()
    truth = dict(E0_1=0.8, width_1=0.04, cathodic_shift=TRUTH["cathodic_shift"], ns_dB=2.0, oh_z=2.1)
    for k, v in truth.items():
        assert abs(r.values[k] - v) < max(4 * r.errors[k], 1e-3), (k, r.values[k], r.errors[k])
    assert 0.8 < r.red_chi2 < 1.3 and set(r.chi2_per_scan) == {s.label for s in scans}
    assert "tied to transition 1" in r.summary()
    prof = j.profile_ns_tl((1, 2, 3))
    assert min(prof, key=lambda p: p[1].red_chi2)[0] == 2                      # the true depth
    j.apply(r)
    back = JointFit.load(j.save(tmp_path / "set.json"))
    assert back.params["oh_z"].value == pytest.approx(r.values["oh_z"]) and len(back.scans) == 4
    assert back.fit().red_chi2 == pytest.approx(r.red_chi2, rel=1e-3)
    j2 = JointFit(scans, dict(ns_tl=0), states=("H2O", "OH"), E0=(0.8,), widths=(0.04,))
    j2.set("ns_dB", 1.0, free=True)
    with pytest.raises(ValueError, match="ns_tl"):
        j2.fit()


def test_cli_cv_joint(joint_scans, tmp_path):
    from ctrfit.echem.joint import JointFit
    scans, _ = joint_scans
    j = JointFit(scans[:2], dict(ns_tl=2), states=("H2O", "OH"), E0=(0.78,), widths=(0.05,))
    j.save(tmp_path / "set.json")
    code = "import sys; sys.modules['xraydb'] = None; from ctrfit.cli import main; sys.exit(main())"
    env = dict(__import__("os").environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-c", code, "cv-joint", str(tmp_path / "set.json"), "--free", "ns_dB",
                        "--profile", "1,2", "--out", str(tmp_path / "fit.json")], capture_output=True, text=True,
                       env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "ns_tl  2" in r.stdout and "Joint fit of 2 scans" in r.stdout and (tmp_path / "fit.json").exists()
    r = subprocess.run([sys.executable, "-c", code, "cv-joint", str(tmp_path / "set.json"), "--free", "nope"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode != 0 and "unknown parameters" in r.stderr


def test_joint_fit_tab(joint_scans, tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets
    from ctrfit.app.echem_window import EchemWindow
    _, d = joint_scans
    msgs = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: msgs.append(a[2])))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = EchemWindow()
    w.joint_fit()
    assert msgs and "add scans" in msgs[-1]
    msgs.clear()
    w.states.setText("H2O -> OH")
    w.transitions.setText("0.78, 0.05")
    w.monitor.setCurrentText("Monitor")
    for i in (0, 1):
        w.open_spec(str(d / str(i) / "insitu.spec"))
        w.monitor.setCurrentText("Monitor")
        w.open_ec(str(d / str(i) / "cv.mpt"))
        w.offset.setValue(TRUTH["clock_offset"])
        w.update_all()
        w.session.opt.hkl = None
        w.joint_add_scan()
    assert not msgs, msgs
    assert w.j_scans.rowCount() == 2 and "oh_z" in w.joint.params
    w.joint_add_param("relax_Obr")
    assert "relax_Obr" in w.joint.params
    w.j_tl.setValue(2)
    w.joint_fit()
    assert not msgs, msgs
    assert "Joint fit of 2 scans" in w.j_out.toPlainText() and w.p_joint.figure.axes
    w.j_profile.setText("1, 2")
    w.joint_profile()
    assert "ns_tl profile" in w.j_out.toPlainText()
    w.joint_apply()
    w.joint_save(str(tmp_path / "j.json"))
    w.joint_clear()
    w.joint_load(str(tmp_path / "j.json"))
    assert w.j_scans.rowCount() == 2 and not msgs
    w.close()

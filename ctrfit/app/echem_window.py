"""In-situ CV window: a stationary SPEC scan (loopscan, or a phi scan that does not move) lined up
with the potentiostat (EC-Lab .mpr / .mpt, or a CV defined by hand); intensity vs potential,
background, sigmoid fits, and the CTR-model prediction at the HKL.

Run:  python ctr_echem.py   (or python -m ctrfit echem)

The computation is in ctrfit.echem (testable without Qt); this file only holds the window.
"""
import json
import sys
import traceback
from pathlib import Path

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:  # pragma: no cover
    from PySide6 import QtCore, QtGui, QtWidgets

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
except ImportError:  # pragma: no cover
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

QShortcut = getattr(QtWidgets, "QShortcut", None) or QtGui.QShortcut

from ..echem.session import Options, Session
from ..project import plots

SYNC_LABELS = [("absolute", "Absolute clocks + offset"), ("manual", "Manual offset"), ("events", "Align by events")]
SYNC_HELP = {
    "absolute": "Both files' own clocks (SPEC #D/#E, EC-Lab start time). Offset = EC clock − SPEC clock (0 if "
                "synchronised). 'Offset from events' fills it from the intensity onset and the sweep start.",
    "manual": "Times from each file's start (the scan's #D, the first potentiostat sample). Offset = minus the delay "
              "of the potentiostat start after the scan start (started 60 s later: −60).",
    "events": "The detected (or typed) intensity onset is put on the detected (or typed) sweep start, plus the fine "
              "shift. Biased if the intensity only starts to change some time into the CV.",
}


def _dspin(lo, hi, val, dec=3, step=0.1, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


class Plot(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(NavToolbar(self.canvas, self))
        lay.addWidget(self.canvas, 1)

    def draw(self, fn, *args):
        try:
            fn(self.figure, *args)
        except Exception:  # keep the window alive
            self.figure.clear()
            plots.message(self.figure, "plot failed:\n" + traceback.format_exc().splitlines()[-1])
        self.canvas.draw_idle()


class EchemWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ctrfit in-situ CV analysis")
        self.session = Session(Options())
        self.spec_path = None
        self.ec_path = None

        side = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(side)
        sl.addWidget(self._files_box())
        sl.addWidget(self._potential_box())
        sl.addWidget(self._sync_box())
        sl.addWidget(self._analysis_box())
        sl.addWidget(self._model_box())
        sl.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setMinimumWidth(500)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.b_update = QtWidgets.QPushButton("Update (Ctrl+R)")
        self.b_update.setMinimumHeight(32)
        self.b_update.setStyleSheet("QPushButton { background: #1f77b4; color: white; font-weight: 600; "
                                    "border-radius: 4px; }")
        b_csv = QtWidgets.QPushButton("Export data CSV…")
        b_json = QtWidgets.QPushButton("Export results…")
        b_save = QtWidgets.QPushButton("Save session…")
        b_load = QtWidgets.QPushButton("Open session…")
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.addWidget(scroll, 1)
        ll.addWidget(self.b_update)
        g = QtWidgets.QGridLayout()
        for i, b in enumerate((b_csv, b_json, b_save, b_load)):
            g.addWidget(b, i // 2, i % 2)
        ll.addLayout(g)

        self.tabs = QtWidgets.QTabWidget()
        self.p_sync, self.p_bg, self.p_iv, self.p_fit, self.p_model = Plot(), Plot(), Plot(), Plot(), Plot()
        self.summary = QtWidgets.QPlainTextEdit(readOnly=True)
        self.summary.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        for w, name in ((self.p_sync, "Sync"), (self.p_bg, "Background"), (self.p_iv, "I vs V"),
                        (self.p_fit, "Transition fit"), (self.p_model, "CTR model"), (self.summary, "Summary")):
            self.tabs.addTab(w, name)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setSizes([520, 980])
        self.setCentralWidget(split)
        self.status = QtWidgets.QLabel("Open a SPEC file and a potentiostat file (or define the CV by hand).")
        self.statusBar().addWidget(self.status, 1)

        self.b_update.clicked.connect(lambda: self.update_all())
        b_csv.clicked.connect(lambda: self.export_csv())
        b_json.clicked.connect(lambda: self.export_results())
        b_save.clicked.connect(lambda: self.save_session())
        b_load.clicked.connect(lambda: self.open_session())
        for seq in ("Ctrl+R", "F5"):
            QShortcut(QtGui.QKeySequence(seq), self, activated=self.update_all)
        self._sync_mode_changed()
        self._cv_mode_changed()
        self.resize(1500, 950)

    # ------------------------------------------------------------------ side panel
    def _files_box(self):
        box = QtWidgets.QGroupBox("SPEC scan")
        f = QtWidgets.QFormLayout(box)
        row = QtWidgets.QHBoxLayout()
        self.spec_label = QtWidgets.QLabel("no file")
        b = QtWidgets.QPushButton("Open SPEC file…")
        b.clicked.connect(lambda: self.open_spec())
        row.addWidget(b)
        row.addWidget(self.spec_label, 1)
        f.addRow(row)
        self.scan_combo = QtWidgets.QComboBox()
        self.only_stationary = QtWidgets.QCheckBox("only stationary scans")
        self.only_stationary.setChecked(True)
        self.only_stationary.setToolTip("loopscans / timescans, and motor scans whose start equals the end")
        f.addRow("Scan", self.scan_combo)
        f.addRow("", self.only_stationary)
        self.counter = QtWidgets.QComboBox()
        self.monitor = QtWidgets.QComboBox()
        self.transm = QtWidgets.QComboBox()
        self.transm.setToolTip("Attenuator transmission column: the intensity is divided by it")
        self.per_second = QtWidgets.QCheckBox("divide by the count time")
        self.per_second.setChecked(True)
        f.addRow("Counter", self.counter)
        f.addRow("Monitor", self.monitor)
        f.addRow("Transmission", self.transm)
        f.addRow("", self.per_second)
        self.time_col = QtWidgets.QComboBox()
        self.time_kind = QtWidgets.QComboBox()
        self.time_kind.addItems(["epoch", "elapsed", "unix"])
        self.time_kind.setToolTip("epoch: seconds since #E (phi scans); elapsed: seconds since the scan's #D "
                                  "(loopscan Time); unix: absolute Unix time")
        self.time_mark = QtWidgets.QComboBox()
        self.time_mark.addItems(["end", "middle", "start"])
        self.time_mark.setToolTip("Which moment of the count the time stamp marks (SPEC: the end)")
        tr = QtWidgets.QHBoxLayout()
        tr.addWidget(self.time_col, 1)
        tr.addWidget(self.time_kind)
        tr.addWidget(self.time_mark)
        f.addRow("Time column", tr)
        self.hkl = QtWidgets.QLineEdit()
        self.hkl.setPlaceholderText("from the scan's #Q, or type H K L")
        f.addRow("HKL", self.hkl)
        self.scan_combo.currentIndexChanged.connect(lambda *_: self._scan_changed())
        self.only_stationary.toggled.connect(lambda *_: self._fill_scans())
        return box

    def _potential_box(self):
        box = QtWidgets.QGroupBox("Potential")
        f = QtWidgets.QFormLayout(box)
        self.use_file = QtWidgets.QRadioButton("Potentiostat file")
        self.use_manual = QtWidgets.QRadioButton("Define the CV by hand")
        self.use_file.setChecked(True)
        f.addRow(self.use_file)
        row = QtWidgets.QHBoxLayout()
        self.ec_label = QtWidgets.QLabel("no file")
        self.b_ec = QtWidgets.QPushButton("Open .mpr / .mpt…")
        self.b_ec.clicked.connect(lambda: self.open_ec())
        row.addWidget(self.b_ec)
        row.addWidget(self.ec_label, 1)
        f.addRow(row)
        self.ec_start = QtWidgets.QLineEdit()
        self.ec_start.setPlaceholderText("from the file; or e.g. 2026-03-07 14:31:07.3")
        f.addRow("Potentiostat start", self.ec_start)
        f.addRow(self.use_manual)
        o = Options()
        self.cv_hold_V = _dspin(-5, 5, o.cv_hold_V, 3, 0.05, " V")
        self.cv_hold_s = _dspin(0, 1e6, o.cv_hold_s, 1, 10, " s")
        self.cv_lower = _dspin(-5, 5, o.cv_lower, 3, 0.05, " V")
        self.cv_upper = _dspin(-5, 5, o.cv_upper, 3, 0.05, " V")
        self.cv_rate = _dspin(0.001, 1e5, o.cv_rate, 3, 1, " mV/s")
        self.cv_cycles = QtWidgets.QSpinBox()
        self.cv_cycles.setRange(0, 1000)
        self.cv_cycles.setSpecialValueText("until stopped")
        self.cv_cycles.setToolTip("0: as many cycles as fit until the 'stopped at point'")
        self.cv_cycles.setValue(o.cv_cycles)
        self.cv_first = QtWidgets.QComboBox()
        self.cv_first.addItems(["up", "down"])
        self.cv_delay = _dspin(-1e6, 1e6, o.cv_delay, 2, 1, " s")
        self.cv_delay.setToolTip("The hold starts this long after the scan start (#D); not used with a start point")
        self.cv_start_point = QtWidgets.QLineEdit()
        self.cv_start_point.setPlaceholderText("point # (0-based); blank: use the delay")
        self.cv_start_marks = QtWidgets.QComboBox()
        self.cv_start_marks.addItem("start of the first sweep", "sweep")
        self.cv_start_marks.addItem("start of the hold", "hold")
        self.cv_end_point = QtWidgets.QLineEdit()
        self.cv_end_point.setPlaceholderText("point # where the CV stopped (optional)")
        self.cv_widgets = []
        for lab, w in (("Hold (relaxation) potential", self.cv_hold_V), ("Hold time", self.cv_hold_s),
                       ("Lower vertex", self.cv_lower), ("Upper vertex", self.cv_upper), ("Scan rate", self.cv_rate),
                       ("Cycles", self.cv_cycles), ("First sweep", self.cv_first),
                       ("Hold starts after the scan start", self.cv_delay), ("CV starts at point", self.cv_start_point),
                       ("That point marks the", self.cv_start_marks), ("CV stopped at point", self.cv_end_point)):
            f.addRow(lab, w)
            self.cv_widgets.append(w)
        self.use_manual.toggled.connect(lambda *_: self._cv_mode_changed())
        return box

    def _sync_box(self):
        box = QtWidgets.QGroupBox("Time sync")
        f = QtWidgets.QFormLayout(box)
        self.sync_mode = QtWidgets.QComboBox()
        for key, lab in SYNC_LABELS:
            self.sync_mode.addItem(lab, key)
        self.sync_help = QtWidgets.QLabel()
        self.sync_help.setWordWrap(True)
        self.sync_help.setStyleSheet("color: #555;")
        self.offset = _dspin(-1e7, 1e7, 0.0, 2, 1, " s")
        self.offset.setToolTip("EC time = SPEC time + offset")
        self.onset = QtWidgets.QLineEdit()
        self.onset.setPlaceholderText("detect (s from the first SPEC point)")
        self.onset_point = QtWidgets.QLineEdit()
        self.onset_point.setPlaceholderText("or the point # where the CV starts")
        self.sweep_start = QtWidgets.QLineEdit()
        self.sweep_start.setPlaceholderText("detect (s from the potentiostat start)")
        self.fine_shift = _dspin(-1e5, 1e5, 0.0, 2, 0.5, " s")
        b_auto = QtWidgets.QPushButton("Offset from events")
        b_auto.setToolTip("Detect the intensity onset and the sweep start and set the offset that lines them up")
        b_auto.clicked.connect(lambda: self.offset_from_events())
        f.addRow("Mode", self.sync_mode)
        f.addRow(self.sync_help)
        f.addRow("Offset", self.offset)
        b_loop = QtWidgets.QPushButton("Shift that closes the loop")
        b_loop.setToolTip("Find the time shift that makes the anodic and cathodic I(V) agree best. A real hysteresis "
                          "is closed by it too, so it tests a timing error; it does not prove one")
        b_loop.clicked.connect(lambda: self.close_loop())
        row = QtWidgets.QHBoxLayout()
        row.addWidget(b_auto)
        row.addWidget(b_loop)
        f.addRow("", row)
        f.addRow("Intensity onset", self.onset)
        f.addRow("Onset at point", self.onset_point)
        f.addRow("Sweep start", self.sweep_start)
        f.addRow("Fine shift (events)", self.fine_shift)
        self.sync_mode.currentIndexChanged.connect(lambda *_: self._sync_mode_changed())
        return box

    def _analysis_box(self):
        box = QtWidgets.QGroupBox("Background and I vs V")
        f = QtWidgets.QFormLayout(box)
        self.bg_model = QtWidgets.QComboBox()
        self.bg_model.addItems(["none", "constant", "linear", "exponential"])
        self.bg_model.setCurrentText("linear")
        self.bg_corr = QtWidgets.QComboBox()
        self.bg_corr.addItems(["divide", "subtract"])
        self.relax_end = QtWidgets.QLineEdit()
        self.relax_end.setPlaceholderText("auto: just before the onset (s from the first point)")
        self.cycles = QtWidgets.QLineEdit("all")
        self.cycles.setToolTip("e.g. all, 2, 2-4, 1,3")
        self.bin_width = _dspin(0.1, 1000, 10, 1, 1, " mV")
        self.average = QtWidgets.QCheckBox("average cycles in potential bins")
        self.average.setChecked(True)
        self.n_trans = QtWidgets.QSpinBox()
        self.n_trans.setRange(1, 4)
        self.slope = QtWidgets.QCheckBox("sloped baseline")
        b_fit = QtWidgets.QPushButton("Fit transitions")
        b_fit.clicked.connect(lambda: self.update_all(tab=self.p_fit))
        for lab, w in (("Background model", self.bg_model), ("Correction", self.bg_corr),
                       ("Relaxation ends at", self.relax_end), ("Cycles", self.cycles), ("Bin width", self.bin_width),
                       ("", self.average), ("Sigmoidal transitions", self.n_trans), ("", self.slope), ("", b_fit)):
            f.addRow(lab, w)
        return box

    def _model_box(self):
        box = QtWidgets.QGroupBox("CTR model at the HKL")
        f = QtWidgets.QFormLayout(box)
        o = Options()
        row = QtWidgets.QHBoxLayout()
        self.sim_label = QtWidgets.QLabel("simulator defaults")
        b_ini = QtWidgets.QPushButton("Settings .ini…")
        b_ini.setToolTip("Simulator settings (settings.ini or a preset from the CTR simulator)")
        b_def = QtWidgets.QPushButton("Defaults")
        b_ini.clicked.connect(lambda: self.load_sim_settings())
        b_def.clicked.connect(lambda: self.load_sim_settings(clear=True))
        row.addWidget(b_ini)
        row.addWidget(b_def)
        row.addWidget(self.sim_label, 1)
        f.addRow(row)
        self.states = QtWidgets.QLineEdit(o.states)
        self.transitions = QtWidgets.QLineEdit(o.transitions)
        self.transitions.setToolTip("One 'E0 (V), width (V)' per state change, separated by ';'")
        self.theta = _dspin(0, 1, o.theta, 3, 0.05)
        self.cath_shift = _dspin(-1000, 1000, 1e3 * o.cathodic_shift, 1, 5, " mV")
        self.reference = QtWidgets.QLineEdit(o.reference)
        self.reference.setToolTip("Surface state during the relaxation period; sets the intensity scale")
        self.fit_shift = QtWidgets.QCheckBox("fit the cathodic shift")
        self.fit_shift.setChecked(True)
        self.fit_theta = QtWidgets.QCheckBox("fit the coverage")
        b_sim = QtWidgets.QPushButton("Simulate")
        b_fitp = QtWidgets.QPushButton("Fit path to data")
        b_cov = QtWidgets.QPushButton("Coverage from data")
        b_use = QtWidgets.QPushButton("Use fitted transitions")
        b_use.setToolTip("Copy the path fit's E0 and widths into the fields above")
        b_sim.clicked.connect(lambda: self.simulate())
        b_fitp.clicked.connect(lambda: self.fit_path())
        b_cov.clicked.connect(lambda: self.coverage())
        b_use.clicked.connect(lambda: self.use_fitted_transitions())
        for lab, w in (("Path (states)", self.states), ("Transitions", self.transitions), ("Coverage θ", self.theta),
                       ("Cathodic shift", self.cath_shift), ("Relaxation state", self.reference), ("", self.fit_shift),
                       ("", self.fit_theta)):
            f.addRow(lab, w)
        g = QtWidgets.QGridLayout()
        for i, b in enumerate((b_sim, b_fitp, b_cov, b_use)):
            g.addWidget(b, i // 2, i % 2)
        f.addRow(g)
        return box

    # ------------------------------------------------------------------ helpers
    def say(self, text, error=False):
        self.status.setText(text.splitlines()[0][:220])
        if error:
            QtWidgets.QMessageBox.warning(self, "In-situ CV", text)

    def _run(self, fn):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            return fn()
        except (ValueError, KeyError, RuntimeError, OSError) as ex:
            self.say(str(ex), error=True)
        except Exception:  # keep the window alive
            self.say(traceback.format_exc(), error=True)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        return None

    @staticmethod
    def _opt_float(text, what):
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{what}: '{text}' is not a number (leave it empty to detect)") from None

    @staticmethod
    def _opt_int(text, what):
        text = text.strip().lstrip("#").strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"{what}: '{text}' is not a point number") from None

    def _sync_mode_changed(self):
        mode = self.sync_mode.currentData()
        self.sync_help.setText(SYNC_HELP[mode])
        ev = mode == "events"
        for w in (self.onset, self.onset_point, self.sweep_start, self.fine_shift):
            w.setEnabled(ev)
        self.offset.setEnabled(not ev)

    def _cv_mode_changed(self):
        manual = self.use_manual.isChecked()
        for w in self.cv_widgets:
            w.setEnabled(manual)
        self.b_ec.setEnabled(not manual)
        self.ec_start.setEnabled(not manual)

    # ------------------------------------------------------------------ files
    def open_spec(self, path=None):
        path = path or QtWidgets.QFileDialog.getOpenFileName(self, "SPEC file", "", "SPEC files (*)")[0]
        if not path:
            return

        def go():
            self.session.spec = None
            self.session.opt.scan = None
            self.session.load_spec(path)
            self.spec_path = str(path)
            self.spec_label.setText(Path(path).name)
            self._fill_scans()
            self.say(f"{Path(path).name}: {len(self.session.spec.scans)} scans, "
                     f"{len(self.session.spec.stationary_scans())} stationary")
        self._run(go)

    def _fill_scans(self):
        if self.session.spec is None:
            return
        scans = self.session.spec.stationary_scans() if self.only_stationary.isChecked() else self.session.spec.scans
        cur = self.session.opt.scan
        self.scan_combo.blockSignals(True)
        self.scan_combo.clear()
        for s in scans:
            self.scan_combo.addItem(f"#{s.number}  {s.command}  ({s.npts} pts)", s.number)
        idx = self.scan_combo.findData(cur)
        self.scan_combo.setCurrentIndex(idx if idx >= 0 else self.scan_combo.count() - 1)
        self.scan_combo.blockSignals(False)
        self._scan_changed()

    def _scan_changed(self):
        n = self.scan_combo.currentData()
        if n is None or self.session.spec is None:
            return
        self.session.opt.scan = n
        s = self.session.scan
        for combo, items, pick in ((self.counter, s.labels, s.default_counter()),
                                   (self.monitor, ["(none)"] + s.labels, s.default_monitor() or "(none)"),
                                   (self.transm, ["(none)"] + s.labels, s.default_transmission() or "(none)"),
                                   (self.time_col, s.labels, s.default_time_column()[0] if any(
                                       x in s.labels for x in ("Epoch", "Time")) else s.labels[0])):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            combo.setCurrentText(pick)
            combo.blockSignals(False)
        try:
            self.time_kind.setCurrentText(s.default_time_column()[1])
        except KeyError:
            pass
        self.hkl.setPlaceholderText(f"from #Q: {' '.join(f'{x:g}' for x in s.hkl)}" if s.hkl else
                                    "no #Q in this scan: type H K L")

    def open_ec(self, path=None):
        path = path or QtWidgets.QFileDialog.getOpenFileName(
            self, "Potentiostat file", "", "EC-Lab (*.mpr *.mpt *.txt);;CSV (*.csv);;All files (*)")[0]
        if not path:
            return

        def go():
            ec = self.session.load_ec(path)
            self.ec_path = str(path)
            self.ec_label.setText(Path(path).name)
            self.use_file.setChecked(True)
            self.ec_start.setPlaceholderText(str(ec.start) if ec.start else "no start time in the file: type it")
            self.say(f"{Path(path).name}: {ec.time.size} samples" + ("; " + "; ".join(ec.notes) if ec.notes else ""))
        self._run(go)

    def load_sim_settings(self, path=None, clear=False):
        if clear:
            self.session.opt.sim_settings = None
            self.sim_label.setText("simulator defaults")
            return
        path = path or QtWidgets.QFileDialog.getOpenFileName(self, "Simulator settings", "", "INI (*.ini)")[0]
        if path:
            self.session.opt.sim_settings = str(path)
            self.sim_label.setText(Path(path).name)

    # ------------------------------------------------------------------ options <-> widgets
    def gather(self):
        o = self.session.opt
        o.counter = self.counter.currentText() or None
        o.monitor = None if self.monitor.currentText() in ("", "(none)") else self.monitor.currentText()
        o.transmission = None if self.transm.currentText() in ("", "(none)") else self.transm.currentText()
        o.per_second = self.per_second.isChecked()
        o.time_column = self.time_col.currentText() or None
        o.time_kind = self.time_kind.currentText()
        o.time_mark = self.time_mark.currentText()
        text = self.hkl.text().replace(",", " ").split()
        if text:
            if len(text) != 3:
                raise ValueError("HKL: give three numbers")
            o.hkl = tuple(float(x) for x in text)
        else:
            o.hkl = None
        o.cv_manual = self.use_manual.isChecked()
        o.cv_hold_V, o.cv_hold_s = self.cv_hold_V.value(), self.cv_hold_s.value()
        o.cv_lower, o.cv_upper, o.cv_rate = self.cv_lower.value(), self.cv_upper.value(), self.cv_rate.value()
        o.cv_cycles, o.cv_first, o.cv_delay = self.cv_cycles.value(), self.cv_first.currentText(), self.cv_delay.value()
        o.cv_start_point = self._opt_int(self.cv_start_point.text(), "CV start point")
        o.cv_end_point = self._opt_int(self.cv_end_point.text(), "CV stop point")
        o.cv_start_marks = self.cv_start_marks.currentData()
        o.onset_point = self._opt_int(self.onset_point.text(), "onset point")
        o.ec_start = self.ec_start.text().strip() or None
        o.sync_mode = self.sync_mode.currentData()
        o.offset = self.offset.value()
        o.onset = self._opt_float(self.onset.text(), "intensity onset")
        o.sweep_start = self._opt_float(self.sweep_start.text(), "sweep start")
        o.fine_shift = self.fine_shift.value()
        o.background, o.correction = self.bg_model.currentText(), self.bg_corr.currentText()
        o.relax_end = self._opt_float(self.relax_end.text(), "relaxation end")
        o.cycles = self.cycles.text().strip() or "all"
        o.bin_width = self.bin_width.value() / 1e3
        o.average = self.average.isChecked()
        o.n_transitions, o.slope = self.n_trans.value(), self.slope.isChecked()
        o.states, o.transitions = self.states.text(), self.transitions.text()
        o.theta, o.cathodic_shift = self.theta.value(), self.cath_shift.value() / 1e3
        o.reference = self.reference.text()
        o.fit_shift, o.fit_theta = self.fit_shift.isChecked(), self.fit_theta.isChecked()
        return o

    def apply_options(self, o):
        self.session.opt = o
        sets = ((self.per_second, o.per_second), (self.average, o.average), (self.slope, o.slope),
                (self.fit_shift, o.fit_shift), (self.fit_theta, o.fit_theta))
        for w, v in sets:
            w.setChecked(bool(v))
        self.use_manual.setChecked(bool(o.cv_manual))
        self.use_file.setChecked(not o.cv_manual)
        for w, v in ((self.cv_hold_V, o.cv_hold_V), (self.cv_hold_s, o.cv_hold_s), (self.cv_lower, o.cv_lower),
                     (self.cv_upper, o.cv_upper), (self.cv_rate, o.cv_rate), (self.cv_delay, o.cv_delay),
                     (self.offset, o.offset), (self.fine_shift, o.fine_shift), (self.theta, o.theta),
                     (self.bin_width, 1e3 * o.bin_width), (self.cath_shift, 1e3 * o.cathodic_shift)):
            w.setValue(float(v))
        self.cv_cycles.setValue(int(o.cv_cycles))
        self.n_trans.setValue(int(o.n_transitions))
        self.cv_first.setCurrentText(o.cv_first)
        self.sync_mode.setCurrentIndex(self.sync_mode.findData(o.sync_mode))
        self.bg_model.setCurrentText(o.background)
        self.bg_corr.setCurrentText(o.correction)
        self.time_mark.setCurrentText(o.time_mark)
        for w, v in ((self.onset, o.onset), (self.sweep_start, o.sweep_start), (self.relax_end, o.relax_end),
                     (self.cv_start_point, o.cv_start_point), (self.cv_end_point, o.cv_end_point),
                     (self.onset_point, o.onset_point)):
            w.setText("" if v is None else f"{v:g}")
        self.cv_start_marks.setCurrentIndex(max(0, self.cv_start_marks.findData(o.cv_start_marks)))
        self.ec_start.setText(o.ec_start or "")
        self.hkl.setText("" if o.hkl is None else " ".join(f"{x:g}" for x in o.hkl))
        self.cycles.setText(o.cycles)
        self.states.setText(o.states)
        self.transitions.setText(o.transitions)
        self.reference.setText(o.reference)
        self.sim_label.setText(Path(o.sim_settings).name if o.sim_settings else "simulator defaults")

    # ------------------------------------------------------------------ actions
    def update_all(self, tab=None):
        def go():
            self.gather()
            self.session.run()
            self._redraw()
            if tab is not None:
                self.tabs.setCurrentWidget(tab)
            r = self.session.res
            n_v = int(sum(1 for v in r.V if v == v))
            self.say(f"{r.t.size} points, {n_v} with a potential; offset {r.alignment.offset:+.2f} s")
        self._run(go)

    def _redraw(self):
        s, r = self.session, self.session.res
        self.p_sync.draw(plots.draw_echem_sync, r, s.ec)
        self.p_bg.draw(plots.draw_echem_background, r)
        self.p_iv.draw(plots.draw_echem_iv, r)
        self.p_fit.draw(plots.draw_echem_fit, r)
        self.p_model.draw(plots.draw_echem_model, r, r.point_model)
        self.summary.setPlainText(s.summary())

    def offset_from_events(self):
        def go():
            self.gather()
            if self.session.opt.sync_mode == "events":
                raise ValueError("in 'Align by events' the offset is found from the events directly; pick "
                                 "'Absolute clocks + offset' or 'Manual offset' to fill the offset from them")
            off = self.session.events_offset()
            self.offset.setValue(off)
            self.say(f"offset from events: {off:+.2f} s (check it: the intensity may lag the sweep start)")
            self.update_all(tab=self.p_sync)
        self._run(go)

    def close_loop(self):
        def go():
            o = self.gather()
            best, m_best, m0, _ = self.session.loop_offset()
            target = self.fine_shift if o.sync_mode == "events" else self.offset
            target.setValue(target.value() + best)
            self.update_all(tab=self.p_iv)
            self.say(f"shift {best:+.1f} s: anodic/cathodic mismatch {100 * m0:.1f}% -> {100 * m_best:.1f}%. A real "
                     "hysteresis is closed too; keep it only if a timing error is plausible")
        self._run(go)

    def simulate(self):
        def go():
            self.gather()
            if self.session.res.I_corr is None and self.session.spec is not None and (
                    self.session.ec is not None or self.session.opt.cv_manual):
                self.session.run()
            else:
                self.session.simulate()
            self._redraw()
            self.tabs.setCurrentWidget(self.p_model)
        self._run(go)

    def fit_path(self):
        def go():
            self.gather()
            self.session.run()
            self.session.fit_path()
            self._redraw()
            self.tabs.setCurrentWidget(self.p_model)
            p = self.session.res.path_fit
            self.say("path fit: " + ", ".join(f"E0 {e:.3f} V" for e in p.E0) + f", reduced chi2 {p.red_chi2:.2f}")
        self._run(go)

    def coverage(self):
        def go():
            self.gather()
            self.session.run()
            self.session.coverage()
            self._redraw()
            self.tabs.setCurrentWidget(self.p_model)
        self._run(go)

    def use_fitted_transitions(self):
        p = self.session.res.path_fit
        if p is None:
            self.say("Fit the path first.", error=True)
            return
        self.transitions.setText("; ".join(f"{e:.4f}, {w:.4f}" for e, w in zip(p.E0, p.widths)))
        self.cath_shift.setValue(1e3 * p.cathodic_shift)
        if self.fit_theta.isChecked():
            self.theta.setValue(p.theta)

    # ------------------------------------------------------------------ files out
    def export_csv(self, path=None):
        if self.session.res.t is None:
            self.say("Press Update first.", error=True)
            return
        path = path or QtWidgets.QFileDialog.getSaveFileName(self, "Export data", "insitu_cv.csv", "CSV (*.csv)")[0]
        if path:
            self.session.export_csv(path)
            self.say(f"Saved {Path(path).name}")

    def export_results(self, path=None):
        if self.session.res.t is None:
            self.say("Press Update first.", error=True)
            return
        path = path or QtWidgets.QFileDialog.getSaveFileName(self, "Export results", "insitu_cv.json",
                                                             "JSON (*.json)")[0]
        if path:
            self.session.export_json(path)
            self.say(f"Saved {Path(path).name}")

    def save_session(self, path=None):
        def go():
            o = self.gather()
            p = path or QtWidgets.QFileDialog.getSaveFileName(self, "Save session", "insitu_session.json",
                                                              "JSON (*.json)")[0]
            if p:
                Path(p).write_text(json.dumps(dict(options=o.to_dict(), spec=self.spec_path, ec=self.ec_path),
                                              indent=1), encoding="utf-8")
                self.say(f"Saved {Path(p).name}")
        self._run(go)

    def open_session(self, path=None):
        path = path or QtWidgets.QFileDialog.getOpenFileName(self, "Open session", "", "JSON (*.json)")[0]
        if not path:
            return

        def go():
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            o = Options.from_dict(d["options"])
            if d.get("spec"):
                self.open_spec(d["spec"])
            if d.get("ec") and not o.cv_manual:
                self.open_ec(d["ec"])
            self.apply_options(o)
            self._fill_scans()
            self.update_all()
        self._run(go)


def main(argv=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv if argv is None else argv)
    app.setStyle("Fusion")
    win = EchemWindow()
    win.show()
    return app.exec() if hasattr(app, "exec") else app.exec_()


if __name__ == "__main__":
    sys.exit(main())


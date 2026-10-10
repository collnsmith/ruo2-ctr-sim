# @app title: CTR simulator | group: Simulate | order: 10 | kind: gui | needs: PyQt5 | desc: Rods, OH vs H2O comparison, sensitivity map and ranking, parameter study, thickness and relaxation study
"""RuO2/TiO2 CTR simulator: desktop GUI.

Run:  python ctr_gui.py
Files are kept next to this script: settings.ini (last configuration) and presets/*.ini.
"""
import configparser
import csv
import math
import re
import sys
import traceback
import warnings
from pathlib import Path

try:  # PyQt5 preferred, PySide6 as fallback
    from PyQt5 import QtCore, QtGui, QtWidgets
    Signal = QtCore.pyqtSignal
    QT_API = "PyQt5"
except ImportError:  # pragma: no cover
    from PySide6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.Signal
    QT_API = "PySide6"
QShortcut = getattr(QtWidgets, "QShortcut", None) or QtGui.QShortcut

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
except ImportError:  # older matplotlib
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
import matplotlib
from matplotlib.figure import Figure

APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import ctr_plots as plots                                                     # noqa: E402
from ctr_engine import (CTRModel, film_spacing, oh_h2o, parameter_study, parse_inputs,  # noqa: E402
                        relaxation_study)
from ctr_params import (DEFAULTS, GROUPS, SPECS, STUDY_DEFAULT, STUDY_KEYS, comp_label,  # noqa: E402
                        format_study_rows, parse_study_rows, read_ini, study_label, write_ini)

SETTINGS_FILE = APP_DIR / "settings.ini"
PRESET_DIR = APP_DIR / "presets"
ACCENT = "#1f77b4"          # same blue as Ru in the plots

matplotlib.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})
warnings.filterwarnings("ignore", message=".*constrained_layout.*")
warnings.filterwarnings("ignore", message=".*layout engine.*")
warnings.filterwarnings("ignore", message="Attempt to set non-positive")


class Cancelled(Exception):
    pass


# ==============================================================================================
# small widgets
# ==============================================================================================
class _NoWheelMixin:
    """Spin boxes and combos ignore the mouse wheel unless focused, so scrolling the sidebar
    never changes a value by accident."""

    def wheelEvent(self, ev):
        if self.hasFocus():
            super().wheelEvent(ev)
        else:
            ev.ignore()


class DoubleSpin(_NoWheelMixin, QtWidgets.QDoubleSpinBox):
    pass


class IntSpin(_NoWheelMixin, QtWidgets.QSpinBox):
    pass


class Combo(_NoWheelMixin, QtWidgets.QComboBox):
    pass


class Section(QtWidgets.QWidget):
    """Collapsible group of parameters."""

    def __init__(self, title, expanded):
        super().__init__()
        self.button = QtWidgets.QToolButton(text=title, checkable=True, checked=expanded)
        self.button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.button.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.button.setStyleSheet("QToolButton { border: none; font-weight: 600; padding: 6px 2px; "
                                  "text-align: left; }")
        self.body = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self.body)
        self.form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        self.form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.form.setContentsMargins(14, 0, 4, 8)
        self.body.setVisible(expanded)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.button)
        lay.addWidget(self.body)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        lay.addWidget(line)
        self.button.toggled.connect(self._toggle)

    def _toggle(self, on):
        self.body.setVisible(on)
        self.button.setArrowType(QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)


class ParamPanel(QtWidgets.QWidget):
    """Sidebar built from the parameter schema."""
    changed = Signal()

    DERIVED = ("thickness_nm", "coherent_nm", "spread_nm", "roughness_nm")

    def __init__(self):
        super().__init__()
        self.widgets, self.sections, self.derived = {}, {}, {}
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(0)
        for title, expanded, specs in GROUPS:
            sec = Section(title, expanded)
            self.sections[title] = sec
            for spec in specs:
                w = self._make_widget(spec)
                self.widgets[spec["key"]] = w
                label = QtWidgets.QLabel(spec["label"])
                if spec.get("tip"):
                    label.setToolTip(spec["tip"])
                    w.setToolTip(spec["tip"])
                sec.form.addRow(label, w)
                if spec["key"] in self.DERIVED:
                    info = QtWidgets.QLabel()
                    info.setStyleSheet("color: #555; font-size: 11px;")
                    self.derived[spec["key"]] = info
                    sec.form.addRow("", info)
            lay.addWidget(sec)
        lay.addStretch(1)
        self.changed.connect(self._update_derived)
        self.changed.connect(self._update_enabled)

    def _make_widget(self, spec):
        t = spec["type"]
        if t == "float":
            w = DoubleSpin()
            w.setDecimals(spec.get("decimals", 3))
            w.setRange(spec.get("min", -1e9), spec.get("max", 1e9))
            w.setSingleStep(spec.get("step", 0.1))
            if spec.get("unit"):
                w.setSuffix(" " + spec["unit"])
            w.setKeyboardTracking(False)
            w.setFocusPolicy(QtCore.Qt.StrongFocus)
            w.valueChanged.connect(self.changed.emit)
        elif t == "int":
            w = IntSpin()
            w.setRange(spec.get("min", 0), spec.get("max", 1000))
            w.setKeyboardTracking(False)
            w.setFocusPolicy(QtCore.Qt.StrongFocus)
            w.valueChanged.connect(self.changed.emit)
        elif t == "bool":
            w = QtWidgets.QCheckBox()
            w.toggled.connect(self.changed.emit)
        elif t == "choice":
            w = Combo()
            w.addItems(spec["choices"])
            w.setFocusPolicy(QtCore.Qt.StrongFocus)
            w.currentIndexChanged.connect(self.changed.emit)
        elif spec["key"] in ("study_points", "extra_layers"):
            w = QtWidgets.QPlainTextEdit()
            w.setFixedHeight(64)
            w.setTabChangesFocus(True)
            w.textChanged.connect(self.changed.emit)
        else:
            w = QtWidgets.QLineEdit()
            w.textEdited.connect(self.changed.emit)
        return w

    def values(self):
        out = {}
        for key, w in self.widgets.items():
            t = SPECS[key]["type"]
            if t in ("float", "int"):
                out[key] = w.value()
            elif t == "bool":
                out[key] = w.isChecked()
            elif t == "choice":
                out[key] = w.currentText()
            elif isinstance(w, QtWidgets.QPlainTextEdit):
                out[key] = w.toPlainText().strip()
            else:
                out[key] = w.text().strip()
        return out

    def set_values(self, vals):
        for key, w in self.widgets.items():
            v = vals.get(key, DEFAULTS[key])
            w.blockSignals(True)
            t = SPECS[key]["type"]
            if t in ("float", "int"):
                w.setValue(v)
            elif t == "bool":
                w.setChecked(bool(v))
            elif t == "choice":
                i = w.findText(str(v))
                w.setCurrentIndex(max(i, 0))
            elif isinstance(w, QtWidgets.QPlainTextEdit):
                w.setPlainText(str(v))
            else:
                w.setText(str(v))
            w.blockSignals(False)
        self._update_derived()
        self._update_enabled()

    def _update_derived(self):
        v = self.values()
        try:
            d, _ = film_spacing(v)
        except (ZeroDivisionError, ValueError):
            return
        n = max(1, int(round(10 * v["thickness_nm"] / d)))
        nc = max(0, int(round(10 * v["coherent_nm"] / d)))
        self.derived["thickness_nm"].setText(f"→ {n} trilayers = {n * d / 10:.2f} nm  (d = {d:.3f} Å)")
        self.derived["coherent_nm"].setText(f"→ {nc} trilayers = {nc * d / 10:.2f} nm")
        self.derived["spread_nm"].setText(f"≈ {10 * v['spread_nm'] / d:.2f} trilayers")
        self.derived["roughness_nm"].setText(f"≈ {10 * v['roughness_nm'] / d:.2f} trilayers")

    def _update_enabled(self):
        v = self.values()
        self.widgets["eps_perp_pct"].setEnabled(v["eps_perp_mode"] == "manual")
        for k in ("fp_Ru", "fpp_Ru", "fp_Ti", "fpp_Ti", "fp_O", "fpp_O"):
            self.widgets[k].setEnabled(v["anom_mode"] == "manual")
        for k in ("water_rho", "water_z0", "water_sigma"):
            self.widgets[k].setEnabled(v["water_on"])

    def section_state(self):
        return {t: int(s.button.isChecked()) for t, s in self.sections.items()}

    def set_section_state(self, state):
        for t, s in self.sections.items():
            if t in state:
                s.button.setChecked(bool(int(state[t])))


class PlotPanel(QtWidgets.QWidget):
    """Matplotlib figure with toolbar; resizes with the window."""

    def __init__(self, placeholder="Press Run (Ctrl+R) to compute."):
        super().__init__()
        try:
            self.figure = Figure(layout="constrained")
        except TypeError:  # matplotlib < 3.5
            self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(240, 180)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.toolbar = NavToolbar(self.canvas, self)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)
        plots.message(self.figure, placeholder)
        self.canvas.draw_idle()
        self._last, self._shape = None, None
        self._timer = QtCore.QTimer(self, singleShot=True, interval=250)
        self._timer.timeout.connect(self._reflow)
        self.canvas.installEventFilter(self)

    def _shape_class(self):
        w, h = self.canvas.width(), max(self.canvas.height(), 1)
        return tuple(w / h >= r for r in (1.25, 1.3, 1.6))

    def eventFilter(self, obj, ev):
        if obj is self.canvas and ev.type() == QtCore.QEvent.Resize and self._last is not None:
            self._timer.start()
        return False

    def _reflow(self):
        if self._last is not None and self._shape_class() != self._shape:
            self.draw(*self._last[0], **self._last[1])

    def draw(self, fn, *args, **kw):
        """Draw with fn(figure, ...). Returns None, or the traceback text if drawing failed."""
        err = None
        self._last, self._shape = ((fn, *args), kw), self._shape_class()
        try:
            fn(self.figure, *args, **kw)
        except Exception as ex:  # keep the GUI alive, show the problem in the plot area
            err = traceback.format_exc()
            plots.message(self.figure, f"Could not draw this plot:\n{ex}")
        self.canvas.draw_idle()
        return err


class RodPicker(QtWidgets.QWidget):
    """Editable list of rods plus a show button."""
    requested = Signal(int, int)

    def __init__(self, label="Rod (H K)"):
        super().__init__()
        self.combo = QtWidgets.QComboBox()
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.combo.setMinimumContentsLength(8)
        self.combo.setToolTip("Pick a rod or type H and K, then press Enter")
        btn = QtWidgets.QPushButton("Show rod")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 0)
        lay.addWidget(QtWidgets.QLabel(label))
        lay.addWidget(self.combo)
        lay.addWidget(btn)
        lay.addStretch(1)
        btn.clicked.connect(self._emit)
        self.combo.activated.connect(lambda *_: self._emit())
        self.combo.lineEdit().returnPressed.connect(self._emit)

    def set_rods(self, rods):
        current = self.combo.currentText()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems([f"{h} {k}" for h, k in rods])
        i = self.combo.findText(current)
        self.combo.setCurrentIndex(i if i >= 0 else 0)
        self.combo.blockSignals(False)

    def current(self):
        parts = re.split(r"[\s,]+", self.combo.currentText().strip())
        try:
            h, k = int(parts[0]), int(parts[1])
            return h, k
        except (ValueError, IndexError):
            return None

    def _emit(self):
        hk = self.current()
        if hk is not None:
            self.requested.emit(*hk)


# ==============================================================================================
# background work
# ==============================================================================================
class WorkerSignals(QtCore.QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)


class Worker(QtCore.QRunnable):
    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args
        self.signals = WorkerSignals()
        self.cancelled = False

    def _progress(self, i, n, msg):
        if self.cancelled:
            raise Cancelled()
        self.signals.progress.emit(i, n, msg)

    def run(self):
        try:
            out = self.fn(*self.args, progress=self._progress)
        except Cancelled:
            self.signals.failed.emit("cancelled")
        except Exception:
            self.signals.failed.emit(traceback.format_exc())
        else:
            self.signals.done.emit(out)


def compute_all(params, inputs, flags, rod_hk, cmp_hk, progress):
    """Everything behind one Run. Runs in a worker thread (no Qt or drawing here)."""
    progress(0, 1, "building model")
    m = CTRModel(params)
    out = dict(model=m, inputs=inputs, params=params)
    progress(0, 1, "rods")
    out["rod"] = m.rod_parts(*rod_hk, inputs["comp_default"])
    comps = {f"x_OH = {x:g}": oh_h2o(x, params["theta"]) for x in inputs["x_oh_list"]}
    out["compare"] = m.compare(*cmp_hk, comps)
    if flags["scan"]:
        res, peaks = m.sensitivity_scan(inputs["sens_a"], inputs["sens_b"], params["scan_h_max"],
                                        params["scan_k_max"], params["metric"], params["bragg_excl"],
                                        params["rel_min"], params["i_bg"], progress=progress)
        hq = m.q_max * m.A1 / (2 * math.pi)
        kq = m.q_max * m.A2 / (2 * math.pi)
        progress(0, 1, "map")
        out["scan"] = (res, peaks)
        out["counts"] = m.bragg_counts(max(params["scan_h_max"], int(hq)), max(params["scan_k_max"], int(kq)))
    if flags["study"]:
        out["study"] = relaxation_study(params, inputs["study_points"], inputs["study_thick"],
                                        inputs["study_relax"], inputs["sens_a"], inputs["sens_b"],
                                        progress=progress)
    return out


# ==============================================================================================
# main window
# ==============================================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RuO2 / TiO2(110) CTR simulator")
        self.pool = QtCore.QThreadPool()
        self.pool.setMaxThreadCount(1)
        self.workers = set()
        self.run_worker = None
        self.result = None
        self.run_snapshot = None
        PRESET_DIR.mkdir(exist_ok=True)

        # ---------------- sidebar
        self.panel = ParamPanel()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.setToolTip("Saved configurations in the presets folder")
        b_load = QtWidgets.QPushButton("Load")
        b_save = QtWidgets.QPushButton("Save as…")
        b_del = QtWidgets.QPushButton("Delete")
        for b, tip in ((b_load, "Load the selected preset"), (b_save, "Save the current settings as a preset"),
                       (b_del, "Delete the selected preset")):
            b.setToolTip(tip)
        preset_row = QtWidgets.QGridLayout()
        preset_row.addWidget(QtWidgets.QLabel("Preset"), 0, 0)
        preset_row.addWidget(self.preset_combo, 0, 1, 1, 3)
        preset_row.addWidget(b_load, 1, 1)
        preset_row.addWidget(b_save, 1, 2)
        preset_row.addWidget(b_del, 1, 3)
        preset_row.setColumnStretch(1, 1)
        preset_row.setColumnStretch(2, 1)
        preset_row.setColumnStretch(3, 1)

        self.chk_scan = QtWidgets.QCheckBox("Sensitivity scan")
        self.chk_scan.setToolTip("Scan all rods for the A vs B difference (a few seconds)")
        self.chk_study = QtWidgets.QCheckBox("Thickness and relaxation study")
        self.chk_study.setToolTip("Recompute the fixed points for each thickness and relaxation (a few seconds)")
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setToolTip("Compute with the current settings (Ctrl+R)")
        self.run_button.setMinimumHeight(34)
        self.run_button.setStyleSheet(f"QPushButton {{ background: {ACCENT}; color: white; font-weight: 600; "
                                      f"border-radius: 4px; }} QPushButton:disabled {{ background: #9bb8d3; }}")
        b_reset = QtWidgets.QPushButton("Reset to defaults")

        side = QtWidgets.QWidget()
        side_lay = QtWidgets.QVBoxLayout(side)
        side_lay.setContentsMargins(6, 6, 6, 6)
        side_lay.addLayout(preset_row)
        side_lay.addWidget(scroll, 1)
        side_lay.addWidget(self.chk_scan)
        side_lay.addWidget(self.chk_study)
        side_lay.addWidget(self.run_button)
        side_lay.addWidget(b_reset)
        side.setMinimumWidth(300)

        # ---------------- output tabs
        self.stale = QtWidgets.QLabel("Settings changed since the last run. Press Run (Ctrl+R) to update the results.")
        self.stale.setStyleSheet("background: #fff3cd; color: #5c4400; padding: 6px; border-radius: 3px;")
        self.stale.setWordWrap(True)
        self.stale.hide()

        self.tabs = QtWidgets.QTabWidget()
        self.summary = QtWidgets.QPlainTextEdit(readOnly=True)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.summary.setFont(mono)
        self.summary.setPlainText("Press Run (Ctrl+R) to compute.")
        self.tabs.addTab(self.summary, "Summary")

        self.p_struct = PlotPanel()
        self.tabs.addTab(self.p_struct, "Structure")

        self.rod_pick = RodPicker()
        self.p_rod = PlotPanel()
        self.tabs.addTab(self._stack(self.rod_pick, self.p_rod), "Rods")

        self.cmp_pick = RodPicker()
        self.p_cmp = PlotPanel()
        self.tabs.addTab(self._stack(self.cmp_pick, self.p_cmp), "OH vs H2O")

        self.p_map = PlotPanel("Tick 'Sensitivity scan' and press Run to see the map.")
        self.tabs.addTab(self.p_map, "Sensitivity map")

        self.rank_table = self._table(["Rank", "H", "K", "L", "|dI|/I", "SNR", "I mean (e²)",
                                       "Brighter", "H+K", "Next to Bragg"])
        self.p_ab = PlotPanel("Select a row in the ranking to see that rod.")
        b_rank_csv = QtWidgets.QPushButton("Export ranking as CSV…")
        rank_top = QtWidgets.QWidget()
        rt = QtWidgets.QVBoxLayout(rank_top)
        rt.setContentsMargins(0, 0, 0, 0)
        rt.addWidget(self.rank_table)
        rt.addWidget(b_rank_csv, 0, QtCore.Qt.AlignLeft)
        split_rank = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split_rank.addWidget(rank_top)
        split_rank.addWidget(self.p_ab)
        split_rank.setSizes([300, 400])
        self.tabs.addTab(split_rank, "Ranking")
        self.tabs.addTab(self._param_study_tab(), "Parameter study")

        self.p_study = PlotPanel("Tick 'Thickness and relaxation study' and press Run.")
        self.study_table = self._table(["Relaxation", "Point"])
        b_study_csv = QtWidgets.QPushButton("Export study as CSV…")
        study_bottom = QtWidgets.QWidget()
        sb = QtWidgets.QVBoxLayout(study_bottom)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.addWidget(self.study_table)
        sb.addWidget(b_study_csv, 0, QtCore.Qt.AlignLeft)
        split_study = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split_study.addWidget(self.p_study)
        split_study.addWidget(study_bottom)
        split_study.setSizes([420, 260])
        self.tabs.addTab(split_study, "Thickness study")

        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setFont(mono)
        self.tabs.addTab(self.log, "Log")

        right = QtWidgets.QWidget()
        r_lay = QtWidgets.QVBoxLayout(right)
        r_lay.setContentsMargins(4, 6, 6, 6)
        r_lay.addWidget(self.stale)
        r_lay.addWidget(self.tabs, 1)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.addWidget(side)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([380, 1000])
        self.setCentralWidget(self.splitter)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.status_msg = QtWidgets.QLabel("Ready")
        self.statusBar().addWidget(self.status_msg, 1)
        self.statusBar().addPermanentWidget(self.progress)

        # ---------------- signals
        self.panel.changed.connect(self._mark_stale)
        self.run_button.clicked.connect(self.run_or_stop)
        b_reset.clicked.connect(self.reset_defaults)
        b_load.clicked.connect(self.load_preset)
        b_save.clicked.connect(self.save_preset)
        b_del.clicked.connect(self.delete_preset)
        self.rod_pick.requested.connect(self.show_rod)
        self.cmp_pick.requested.connect(self.show_compare)
        self.rank_table.itemSelectionChanged.connect(self.show_ranked_rod)
        b_rank_csv.clicked.connect(self.export_ranking)
        b_study_csv.clicked.connect(self.export_study)
        self.p_map.canvas.mpl_connect("motion_notify_event", self._map_hover)
        self.p_map.canvas.mpl_connect("button_press_event", self._map_click)
        for seq in ("Ctrl+R", "F5"):
            QShortcut(QtGui.QKeySequence(seq), self, activated=self.run_or_stop)

        self.resize(1400, 900)
        self.refresh_presets()
        self.load_settings()
        QtCore.QTimer.singleShot(200, self.run_or_stop)  # first results right after start-up

    # ------------------------------------------------------------------ parameter study tab
    PS_COLS = ["Parameter", "Min", "Max", "Steps"]

    def _param_study_tab(self):
        self.ps_in = QtWidgets.QTableWidget(0, len(self.PS_COLS))
        self.ps_in.setHorizontalHeaderLabels(self.PS_COLS)
        self.ps_in.verticalHeader().setVisible(False)
        self.ps_in.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.ps_in.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.ps_in.setToolTip("One setting per row, varied from Min to Max in Steps values while the other settings "
                              "stay as in the sidebar")
        b_add = QtWidgets.QPushButton("Add parameter")
        b_del = QtWidgets.QPushButton("Remove selected")
        self.ps_run = QtWidgets.QPushButton("Run parameter study")
        self.ps_run.setStyleSheet(f"QPushButton {{ background: {ACCENT}; color: white; font-weight: 600; "
                                  f"border-radius: 4px; padding: 4px 10px; }} "
                                  f"QPushButton:disabled {{ background: #9bb8d3; }}")
        b_csv = QtWidgets.QPushButton("Export ranking as CSV…")
        self.ps_table = self._table(["Rank", "Parameter", "Min", "Max", "Steps", "Max change (%)", "Rod", "L",
                                     "Median change (%)"])
        self.ps_rod = QtWidgets.QComboBox()
        self.ps_rod.setToolTip("Rod shown for the selected parameter (default: where its change is largest)")
        self.p_ps = PlotPanel("Fill the table, then press 'Run parameter study'. Select a row to see its rods.")
        buttons = QtWidgets.QHBoxLayout()
        for b in (b_add, b_del, self.ps_run):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(b_csv)
        top = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(top)
        tl.addWidget(QtWidgets.QLabel("<b>Parameters to vary</b> (one at a time; rods, surface state for plots and "
                                      "'Ignore near Bragg peaks' come from the sidebar). Change = (max − min) / mean "
                                      "of |F|² over the values."))
        tl.addWidget(self.ps_in, 2)
        tl.addLayout(buttons)
        tl.addWidget(self.ps_table, 3)
        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Rod"))
        row.addWidget(self.ps_rod)
        row.addStretch(1)
        bl.addLayout(row)
        bl.addWidget(self.p_ps, 1)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(top)
        split.addWidget(bottom)
        split.setSizes([330, 420])
        self.ps_result = None
        self.ps_worker = None
        b_add.clicked.connect(lambda: self._ps_add_row())
        b_del.clicked.connect(self._ps_remove_rows)
        self.ps_run.clicked.connect(self.run_param_study)
        b_csv.clicked.connect(lambda: self._export(self.ps_table, "parameter_ranking.csv"))
        self.ps_table.itemSelectionChanged.connect(self._ps_selected)
        self.ps_rod.currentIndexChanged.connect(lambda *_: self.show_param_study())
        self.set_param_study_rows(parse_study_rows(STUDY_DEFAULT))
        return split

    def _ps_add_row(self, key=None, lo=None, hi=None, n=5):
        used = {r[0] for r in self._ps_cells()}
        key = key or next((k for k in STUDY_KEYS if k not in used), STUDY_KEYS[0])
        t = self.ps_in
        i = t.rowCount()
        t.insertRow(i)
        combo = Combo()
        for k in STUDY_KEYS:
            combo.addItem(study_label(k), k)
        combo.setCurrentIndex(STUDY_KEYS.index(key))
        t.setCellWidget(i, 0, combo)
        for j, v in ((1, lo), (2, hi), (3, n)):
            t.setItem(i, j, QtWidgets.QTableWidgetItem("" if v is None else f"{v:g}"))
        combo.currentIndexChanged.connect(lambda *_, c=combo: self._ps_default_range(c))
        if lo is None or hi is None:
            self._ps_default_range(combo)

    def _ps_default_range(self, combo):
        """A range around the current sidebar value: 4 spin-box steps each way, within the limits."""
        for i in range(self.ps_in.rowCount()):
            if self.ps_in.cellWidget(i, 0) is combo:
                break
        else:
            return
        key = combo.currentData()
        s = SPECS[key]
        v = float(self.panel.values().get(key, s["default"]))
        d = 4 * s.get("step", max(abs(v) * 0.1, 0.01))
        lo, hi = max(s.get("min", -math.inf), v - d), min(s.get("max", math.inf), v + d)
        self.ps_in.item(i, 1).setText(f"{lo:.6g}")
        self.ps_in.item(i, 2).setText(f"{hi:.6g}")

    def _ps_remove_rows(self):
        for i in sorted({ix.row() for ix in self.ps_in.selectedIndexes()}, reverse=True):
            self.ps_in.removeRow(i)

    def _ps_cells(self):
        out = []
        for i in range(self.ps_in.rowCount()):
            combo = self.ps_in.cellWidget(i, 0)
            out.append((combo.currentData(), *[(self.ps_in.item(i, j).text() if self.ps_in.item(i, j) else "").strip()
                                               for j in (1, 2, 3)]))
        return out

    def param_study_rows(self):
        """Rows of the input table, checked (raises ValueError with a readable message)."""
        text = "; ".join(", ".join(c) for c in self._ps_cells())
        rows = parse_study_rows(text)
        if not rows:
            raise ValueError("parameter study: add at least one parameter")
        return rows

    def set_param_study_rows(self, rows):
        self.ps_in.setRowCount(0)
        for key, lo, hi, n in rows:
            self._ps_add_row(key, lo, hi, n)

    def run_param_study(self):
        if self.ps_worker is not None:
            self.ps_worker.cancelled = True
            self.status_msg.setText("Stopping…")
            return
        params, inputs = self._gather()
        if params is None:
            return
        try:
            rows = self.param_study_rows()
        except ValueError as ex:
            QtWidgets.QMessageBox.warning(self, "Check the parameter study", str(ex))
            self.say(str(ex), error=True)
            return

        def done(out):
            self._ps_finished()
            self.ps_result = out
            self._fill_param_study(out)
            self.say(f"Parameter study done: {len(out)} parameter(s) on {len(inputs['rods'])} rod(s)")

        def fn(progress):
            return parameter_study(params, rows, inputs["rods"], inputs["comp_default"], params["bragg_excl"],
                                   progress=progress)
        self.ps_run.setText("Stop")
        self.say(f"Parameter study: {sum(r[3] for r in rows)} models…")
        self.ps_worker = self._start(fn, on_done=done)
        self.ps_worker.signals.failed.connect(lambda *_: self._ps_finished())

    def _ps_finished(self):
        self.ps_worker = None
        self.ps_run.setText("Run parameter study")
        self.progress.hide()

    def _fill_param_study(self, out):
        t = self.ps_table
        t.setSortingEnabled(False)
        t.setRowCount(len(out))
        for i, d in enumerate(out):
            found = d["at_hk"] is not None and d["max_rel"] > 0
            rod = f"{d['at_hk'][0]} {d['at_hk'][1]}" if found else "no change"
            row = [i + 1, study_label(d["key"]), (d["lo"], ".6g"), (d["hi"], ".6g"), d["steps"],
                   (100 * d["max_rel"], ".1f"), rod, (d["at_L"], ".3f") if found else "",
                   (100 * d["median_rel"], ".1f")]
            for j, v in enumerate(row):
                it = self._item(*v) if isinstance(v, tuple) else self._item(v)
                if j == 0:
                    it.setData(QtCore.Qt.UserRole, i)
                t.setItem(i, j, it)
        t.setSortingEnabled(True)
        t.sortItems(0, QtCore.Qt.AscendingOrder)
        if out:
            t.selectRow(0)

    def _ps_current(self):
        if not self.ps_result:
            return None
        rows = self.ps_table.selectionModel().selectedRows()
        if not rows:
            return None
        return self.ps_result[self.ps_table.item(rows[0].row(), 0).data(QtCore.Qt.UserRole)]

    def _ps_selected(self):
        d = self._ps_current()
        if d is None:
            return
        self.ps_rod.blockSignals(True)
        self.ps_rod.clear()
        for hk in d["I"]:
            self.ps_rod.addItem(f"({hk[0]} {hk[1]} L)", hk)
        if d["at_hk"] in d["I"]:
            self.ps_rod.setCurrentIndex(list(d["I"]).index(d["at_hk"]))
        self.ps_rod.blockSignals(False)
        self.show_param_study()

    def show_param_study(self):
        d = self._ps_current()
        hk = self.ps_rod.currentData()
        if d is None or hk is None:
            return
        hk = tuple(hk)
        mark = d["at_L"] if hk == d["at_hk"] and d["max_rel"] > 0 else None
        excl = self.result["params"]["bragg_excl"] if self.result else DEFAULTS["bragg_excl"]
        err = self.p_ps.draw(plots.draw_param_study, d, hk, study_label(d["key"]), excl, mark)
        if err:
            self.say(err, error=True)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _stack(top, bottom):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(top)
        lay.addWidget(bottom, 1)
        return w

    @staticmethod
    def _table(headers):
        t = QtWidgets.QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        t.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        return t

    @staticmethod
    def _item(value, fmt=None):
        it = QtWidgets.QTableWidgetItem()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            it.setData(QtCore.Qt.DisplayRole, float(f"{value:{fmt}}") if fmt else value)
        else:
            it.setText(str(value))
        it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return it

    def say(self, text, error=False):
        self.status_msg.setText(text.splitlines()[0][:160])
        self.log.appendPlainText(("ERROR: " if error else "") + text)

    def _mark_stale(self):
        if self.result is not None:
            self.stale.show()

    def _start(self, fn, *args, on_done, track_run=False):
        w = Worker(fn, *args)
        self.workers.add(w)

        def done(out):
            self.workers.discard(w)
            on_done(out)

        def failed(msg):
            self.workers.discard(w)
            if track_run:
                self._run_finished()
            if msg == "cancelled":
                self.say("Run stopped.")
            else:
                self.say(msg, error=True)
                QtWidgets.QMessageBox.warning(self, "Calculation failed",
                                              msg.strip().splitlines()[-1] + "\n\nSee the Log tab for details.")

        w.signals.done.connect(done)
        w.signals.failed.connect(failed)
        w.signals.progress.connect(self._on_progress)
        self.pool.start(w)
        return w

    def _on_progress(self, i, n, msg):
        self.progress.show()
        self.progress.setRange(0, max(n, 1))
        self.progress.setValue(i)
        self.status_msg.setText(msg)

    # ------------------------------------------------------------------ run
    def _gather(self):
        params = self.panel.values()
        try:
            inputs = parse_inputs(params)
            CTRModel(params)  # validates numbers before the thread starts
        except (ValueError, KeyError) as ex:
            QtWidgets.QMessageBox.warning(self, "Check the settings", str(ex))
            self.say(str(ex), error=True)
            return None, None
        return params, inputs

    def run_or_stop(self):
        if self.run_worker is not None:
            self.run_worker.cancelled = True
            self.status_msg.setText("Stopping…")
            return
        params, inputs = self._gather()
        if params is None:
            return
        self.rod_pick.set_rods(inputs["rods"])
        self.cmp_pick.set_rods(inputs["rods"])
        rod_hk = self.rod_pick.current() or inputs["rods"][0]
        cmp_hk = self.cmp_pick.current() or inputs["rods"][0]
        flags = dict(scan=self.chk_scan.isChecked(), study=self.chk_study.isChecked())
        self.run_snapshot = params
        self.run_button.setText("Stop")
        self.say("Running…")
        self.run_worker = self._start(compute_all, params, inputs, flags, rod_hk, cmp_hk,
                                      on_done=self._show_results, track_run=True)

    def _run_finished(self):
        self.run_worker = None
        self.run_button.setText("Run")
        self.progress.hide()

    def _show_results(self, out):
        self._run_finished()
        self.result = out
        m, inp, prm = out["model"], out["inputs"], out["params"]
        self.summary.setPlainText(m.summary())
        errors = []
        errors.append(self.p_struct.draw(plots.draw_structure, m, inp["comp_default"]))
        errors.append(self.p_rod.draw(plots.draw_rod_parts, out["rod"], comp_label(inp["comp_default"])))
        errors.append(self.p_cmp.draw(plots.draw_compare, out["compare"], prm["rel_min"], prm["bragg_excl"]))
        labels = (comp_label(inp["sens_a"]), comp_label(inp["sens_b"]))
        if "scan" in out:
            res, peaks = out["scan"]
            errors.append(self.p_map.draw(plots.draw_hk_map, m, res, out["counts"], prm["metric"],
                                          prm["physical_aspect"], labels,
                                          hint="hover for values, click for curves"))
            self._fill_ranking(peaks[:prm["top_n"]])
        if "study" in out:
            errors.append(self.p_study.draw(plots.draw_study, out["study"], inp["study_points"],
                                            inp["study_thick"], inp["study_relax"], m.n_coherent * m.D_FILM / 10))
            self._fill_study(out["study"], inp)
        errors = [e for e in errors if e]
        for e in errors:
            self.say(e, error=True)
        self.stale.setVisible(self.panel.values() != self.run_snapshot)
        self.say(f"Done: {m.n_film} trilayers ({m.n_film * m.D_FILM / 10:.2f} nm)"
                 + (", relaxed top" if m.relaxed else "") + (f", {len(errors)} plot error(s)" if errors else ""))
        for n in m.notes:
            self.say("Note: " + n)
        self.save_settings()

    def _fill_ranking(self, peaks):
        t = self.rank_table
        t.setSortingEnabled(False)
        t.setRowCount(len(peaks))
        for i, p in enumerate(peaks):
            row = [i + 1, p["H"], p["K"], (p["L"], ".3f"), (p["rel"], ".3f"), (p["snr"], ".3f"), (p["I"], ".4g"),
                   "A" if p["A_gt_B"] else "B", "odd" if (p["H"] + p["K"]) % 2 else "even",
                   "yes" if p["edge"] else ""]
            for j, v in enumerate(row):
                t.setItem(i, j, self._item(*v) if isinstance(v, tuple) else self._item(v))
        t.setSortingEnabled(True)
        t.sortItems(0, QtCore.Qt.AscendingOrder)
        if peaks:
            t.selectRow(0)

    def _fill_study(self, study, inp):
        thick, relax, pts = inp["study_thick"], inp["study_relax"], inp["study_points"]
        t = self.study_table
        t.setSortingEnabled(False)
        t.setColumnCount(3 + len(thick))
        t.setHorizontalHeaderLabels(["Relaxation", "Point", "H K L"] + [f"{x:g} nm (%)" for x in thick])
        t.setRowCount(len(relax) * len(pts))
        r = 0
        for R in relax:
            for lab, (h, k, l) in pts.items():
                t.setItem(r, 0, self._item(R))
                t.setItem(r, 1, QtWidgets.QTableWidgetItem(lab))
                t.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{h} {k} {l:g}"))
                for j, x in enumerate(thick):
                    t.setItem(r, 3 + j, self._item(100 * study[(lab, R, x)], ".1f"))
                r += 1

    # ------------------------------------------------------------------ rod views
    def _need_result(self):
        if self.result is None:
            self.say("Press Run first.")
            return False
        return True

    def show_rod(self, h, k):
        if not self._need_result():
            return
        m, inp = self.result["model"], self.result["inputs"]
        self.say(f"Computing ({h} {k} L)…")
        def done(d):
            err = self.p_rod.draw(plots.draw_rod_parts, d, comp_label(inp["comp_default"]))
            if err:
                self.say(err, error=True)
            else:
                self.say(f"Showing ({h} {k} L)" if d else f"({h} {k} L) is beyond the reachable q range")
        self._start(lambda progress: m.rod_parts(h, k, inp["comp_default"]), on_done=done)

    def show_compare(self, h, k):
        if not self._need_result():
            return
        m, prm, inp = self.result["model"], self.result["params"], self.result["inputs"]
        comps = {f"x_OH = {x:g}": oh_h2o(x, prm["theta"]) for x in inp["x_oh_list"]}
        self.say(f"Computing ({h} {k} L) comparison…")
        def done(d):
            err = self.p_cmp.draw(plots.draw_compare, d, prm["rel_min"], prm["bragg_excl"])
            if err:
                self.say(err, error=True)
            else:
                self.say(f"Showing ({h} {k} L) comparison" if d else f"({h} {k} L) is beyond the reachable q range")
        self._start(lambda progress: m.compare(h, k, comps), on_done=done)

    def show_ranked_rod(self):
        if self.result is None or "scan" not in self.result:
            return
        rows = self.rank_table.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[0].row()
        try:
            h = int(self.rank_table.item(r, 1).data(QtCore.Qt.DisplayRole))
            k = int(self.rank_table.item(r, 2).data(QtCore.Qt.DisplayRole))
            L = float(self.rank_table.item(r, 3).data(QtCore.Qt.DisplayRole))
        except (AttributeError, TypeError, ValueError):
            return
        res, _ = self.result["scan"]
        inp, prm = self.result["inputs"], self.result["params"]
        labels = (comp_label(inp["sens_a"]), comp_label(inp["sens_b"]))
        err = self.p_ab.draw(plots.draw_ab, res.get((h, k)), h, k, labels, L, prm["bragg_excl"])
        if err:
            self.say(err, error=True)

    def _map_rod(self, event):
        if (self.result is None or "scan" not in self.result or event.inaxes is None
                or not getattr(event.inaxes, "_ctr_map", False) or event.xdata is None):
            return None
        return int(round(event.xdata)), int(round(event.ydata))

    def _map_hover(self, event):
        hk = self._map_rod(event)
        if hk is None:
            return
        h, k = hk
        res, _ = self.result["scan"]
        n = self.result["counts"].get((abs(h), abs(k)))
        r = res.get((abs(h), abs(k)))
        if n is None:
            text = f"({h} {k}): beyond the reachable q range"
        elif r is None:
            text = f"({h} {k}): {n} TiO2 Bragg peaks in range; not scanned (raise 'Scan H/K up to')"
        else:
            text = (f"({h} {k}): {n} TiO2 Bragg peaks; max |dI|/I {r['rel_max']:.2f} at L {r['L_rel']:.2f}; "
                    f"max SNR {r['snr_max']:.2f} at L {r['L_snr']:.2f}. Click to see the curves")
        self.status_msg.setText(text)

    def _map_click(self, event):
        if str(getattr(self.p_map.toolbar, "mode", "")):   # zoom or pan active
            return
        hk = self._map_rod(event)
        if hk is None:
            return
        h, k = hk
        res, _ = self.result["scan"]
        r = res.get((abs(h), abs(k)))
        if r is None:
            return
        inp, prm = self.result["inputs"], self.result["params"]
        labels = (comp_label(inp["sens_a"]), comp_label(inp["sens_b"]))
        Lb = r["L_snr" if prm["metric"] == "snr" else "L_rel"]
        self.rank_table.clearSelection()
        err = self.p_ab.draw(plots.draw_ab, r, abs(h), abs(k), labels, Lb, prm["bragg_excl"])
        if err:
            self.say(err, error=True)
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.p_ab.parentWidget()))
        self.say(f"Showing ({abs(h)} {abs(k)} L), clicked on the map")

    # ------------------------------------------------------------------ export
    def _export(self, table, default_name):
        if table.rowCount() == 0:
            self.say("Nothing to export yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", str(APP_DIR / default_name),
                                                        "CSV files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([table.horizontalHeaderItem(j).text() for j in range(table.columnCount())])
            for i in range(table.rowCount()):
                w.writerow([table.item(i, j).text() if table.item(i, j) else ""
                            for j in range(table.columnCount())])
        self.say(f"Exported {Path(path).name}")

    def export_ranking(self):
        self._export(self.rank_table, "ranking.csv")

    def export_study(self):
        self._export(self.study_table, "thickness_study.csv")

    # ------------------------------------------------------------------ presets and settings
    def refresh_presets(self, select=None):
        names = sorted(p.stem for p in PRESET_DIR.glob("*.ini"))
        self.preset_combo.clear()
        self.preset_combo.addItems(names)
        if hasattr(self.preset_combo, "setPlaceholderText"):
            self.preset_combo.setPlaceholderText("Choose a preset")
        self.preset_combo.setCurrentIndex(names.index(select) if select in names else -1)

    def load_preset(self):
        name = self.preset_combo.currentText()
        if not name:
            self.say("No preset selected.")
            return
        vals, problems = read_ini(PRESET_DIR / f"{name}.ini")
        self.panel.set_values(vals)
        for p in problems:
            self.say(f"Preset {name}: {p}")
        self._mark_stale()
        self.say(f"Loaded preset '{name}'. Press Run to update the results.")

    def save_preset(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save preset", "Preset name:",
                                                  text=self.preset_combo.currentText())
        name = re.sub(r"[^\w\- ]", "", name or "").strip()
        if not ok or not name:
            return
        path = PRESET_DIR / f"{name}.ini"
        if path.exists() and QtWidgets.QMessageBox.question(
                self, "Replace preset", f"Replace the preset '{name}'?") != QtWidgets.QMessageBox.Yes:
            return
        try:
            write_ini(path, self.panel.values())
        except OSError as ex:
            self.say(f"Could not save preset: {ex}", error=True)
            return
        self.refresh_presets(select=name)
        self.say(f"Saved preset '{name}'")

    def delete_preset(self):
        name = self.preset_combo.currentText()
        if not name:
            return
        if QtWidgets.QMessageBox.question(self, "Delete preset",
                                          f"Delete the preset '{name}'?") != QtWidgets.QMessageBox.Yes:
            return
        try:
            (PRESET_DIR / f"{name}.ini").unlink()
        except OSError as ex:
            self.say(f"Could not delete preset: {ex}", error=True)
        self.refresh_presets()
        self.say(f"Deleted preset '{name}'")

    def reset_defaults(self):
        if QtWidgets.QMessageBox.question(self, "Reset settings",
                                          "Replace all settings with the defaults?") != QtWidgets.QMessageBox.Yes:
            return
        self.panel.set_values(DEFAULTS)
        self._mark_stale()
        self.say("Settings reset to defaults. Press Run to update the results.")

    def load_settings(self):
        if not SETTINGS_FILE.exists():
            self.panel.set_values(DEFAULTS)
            self.chk_scan.setChecked(True)
            return
        vals, problems = read_ini(SETTINGS_FILE)
        self.panel.set_values(vals)
        for p in problems:
            self.say(f"settings.ini: {p}")
        cp = configparser.ConfigParser(interpolation=None)
        try:
            cp.read(SETTINGS_FILE, encoding="utf-8")
            win = cp["window"] if cp.has_section("window") else {}
            if "geometry" in win:
                self.restoreGeometry(QtCore.QByteArray.fromBase64(win["geometry"].encode()))
            if "splitter" in win:
                self.splitter.restoreState(QtCore.QByteArray.fromBase64(win["splitter"].encode()))
            self.tabs.setCurrentIndex(int(win.get("tab", 0)))
            run = cp["run"] if cp.has_section("run") else {}
            self.chk_scan.setChecked(run.get("scan", "true") == "true")
            self.chk_study.setChecked(run.get("study", "false") == "true")
            if cp.has_section("sections"):
                titles = list(self.panel.sections)
                self.panel.set_section_state({titles[int(k[1:])]: v for k, v in cp["sections"].items()
                                              if k[1:].isdigit() and int(k[1:]) < len(titles)})
            if cp.has_section("param_study") and cp["param_study"].get("rows", "").strip():
                try:
                    self.set_param_study_rows(parse_study_rows(cp["param_study"]["rows"]))
                except ValueError as ex:
                    self.say(f"settings.ini: parameter study rows not restored ({ex})")
        except (configparser.Error, ValueError) as ex:
            self.say(f"settings.ini: window layout not restored ({ex})")

    def _ps_rows_lenient(self):
        """Input rows that parse, for saving (a half-typed row is skipped, not an error)."""
        rows = []
        for cells in self._ps_cells():
            try:
                rows += parse_study_rows(", ".join(cells))
            except ValueError:
                pass
        return rows

    def save_settings(self):
        b64 = lambda qba: bytes(qba.toBase64()).decode()  # noqa: E731
        extra = {
            "window": dict(geometry=b64(self.saveGeometry()), splitter=b64(self.splitter.saveState()),
                           tab=self.tabs.currentIndex()),
            "run": dict(scan="true" if self.chk_scan.isChecked() else "false",
                        study="true" if self.chk_study.isChecked() else "false"),
            "sections": {f"s{i}": on for i, on in enumerate(self.panel.section_state().values())},
            "param_study": dict(rows=format_study_rows(self._ps_rows_lenient())),
        }
        try:
            write_ini(SETTINGS_FILE, self.panel.values(), extra)
        except OSError as ex:
            self.say(f"Could not save settings.ini: {ex}", error=True)

    def closeEvent(self, ev):
        for w in list(self.workers):
            w.cancelled = True
        self.pool.clear()
        self.pool.waitForDone(5000)
        self.save_settings()
        super().closeEvent(ev)


def main():
    if QT_API == "PyQt5":
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()

    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        sys.__stderr__.write(text)
        win.say(text, error=True)
    sys.excepthook = hook
    win.show()
    sys.exit(app.exec() if hasattr(app, "exec") else app.exec_())


if __name__ == "__main__":
    main()

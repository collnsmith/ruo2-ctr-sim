"""Scan library tab: sort background-subtracted scans by film and CV / CA, keep the sorting in a text
file (the library), and compare groups of scans in one plot, e.g. every CV at one H K L, or every CA
at one potential on one rod. The logic is in ctrfit.datatools.library; this file only holds the
window. The data files (SPEC, ROI sums) are only read; the library file is the only file written.
"""
import json
import os

import matplotlib
import numpy as np
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import QItemSelectionModel, Qt, QTimer
from PyQt5.QtGui import QBrush, QColor, QKeySequence
from PyQt5.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                             QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                             QPushButton, QRadioButton, QShortcut, QSpinBox, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from ctrfit.app.datatools.common import (SafeCanvas, StatusMixin, group, hint, primary_button, row, run_tool,
                                         sidebar)
from ctrfit.datatools import library as lb
from ctrfit.datatools.paths import settings_file

try:
    from PyQt5.sip import isdeleted
except ImportError:  # pragma: no cover
    from sip import isdeleted

SETTINGS = "sxrd_library_settings.json"
DRAFT = "sxrd_library_unsaved.txt"           # a library that has no file yet, kept when the window closes
DEFAULT_ROI = "(default)"
AUTO_ROI = "(auto: first with errors)"
NONE = "(none)"
ANY = "(any)"
NO_FILM = "(no film)"
KIND_ITEMS = ("Rod scans (I vs L)", "Stationary scans (I vs time)")
MODE_ITEMS = ("CA / OCV", "CV", "all assigned", "any (also unassigned)")
MODE_SETS = ({"CA", "OCV"}, {"CV"}, {"CV", "CA", "OCV"}, None)
COLUMNS = ("Use", "Source", "Scan", "Kind", "H K", "L", "Points", "Film", "Measurement", "Potential", "CV start", "ROI",
           "Notes", "SPEC command")
(C_USE, C_SRC, C_SCAN, C_KIND, C_HK, C_L, C_PTS, C_FILM, C_MODE, C_E, C_START, C_ROI, C_NOTES,
 C_CMD) = range(len(COLUMNS))
TYPED = (C_FILM, C_E, C_START, C_NOTES)      # typed with several rows selected: set for all of them
TODO = QColor("#fff1db")                     # a field still to fill
BAD = QColor("#fde2e4")
PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#7f7f7f",
           "#bcbd22")


def _fmt(v, d=3):
    return "" if v is None or not np.isfinite(v) else f"{v:.{d}g}"


class LibraryWindow(StatusMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SXRD scan library")
        self.resize(1500, 950)
        self.lib = lb.Library()
        self._dirty = False
        self._undo, self._redo = [], []
        self._row_of = {}
        self._films_off = set()               # films unticked in the comparison
        self._last_traces = []
        self._pending_x = None                # X axis from the settings, picked once its column is there
        self._loading = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.draw_plot)
        self._build()
        self._load_settings()
        self.say("Add a SPEC file and its ROI sums (Data), assign film and CV / CA to the scans, compare them "
                 "below. The data files are only read.")

    # ------------------------------------------------------------------ layout
    def _build(self):
        # library file
        fg, ff = group("Library file")
        self._path = QLineEdit()
        self._path.setReadOnly(True)
        self._path.setPlaceholderText("not saved yet")
        self._state = QLabel("")
        b_new, b_open, b_saveas = QPushButton("New"), QPushButton("Open…"), QPushButton("Save as…")
        b_new.clicked.connect(lambda: self.new_library())
        b_open.clicked.connect(lambda: self.open_library())
        b_saveas.clicked.connect(lambda: self.save_as())
        ff.addRow(self._path)
        ff.addRow(row(b_new, b_open, b_saveas, 1, stretch_first=False))
        ff.addRow(self._state)
        self._unit = QLineEdit(lb.DEFAULT_SETTINGS["unit"])
        self._unit.setToolTip("Unit of the CA potentials (written to the library)")
        self._unit.editingFinished.connect(self._unit_changed)
        ff.addRow("Potential unit", self._unit)

        # data
        dg, dl = group("Data (read only)", form=False)
        self._sources = QListWidget()
        self._sources.setMaximumHeight(110)
        self._sources.setToolTip("One SPEC file and its ROI sums files per entry. Double-click to rename.")
        self._sources.itemChanged.connect(self._source_renamed)
        dl.addWidget(self._sources)
        b_add, b_more = QPushButton("Add SPEC + ROI sums…"), QPushButton("More ROI sums…")
        b_add.setToolTip("A SPEC file, then the ROI sums files of its scans (background subtraction output)")
        b_more.setToolTip("Add ROI sums files to the selected SPEC file (e.g. another background subtraction run)")
        b_add.clicked.connect(lambda: self.add_source_dialog())
        b_more.clicked.connect(lambda: self.more_sums_dialog())
        b_rm, b_ref = QPushButton("Remove"), QPushButton("Refresh  (F5)")
        b_rm.setToolTip("Take the selected SPEC file and its scans out of the library (the files stay; Ctrl+Z undoes)")
        b_ref.setToolTip("Read the data files again (new scans are added, assignments kept)")
        b_rm.clicked.connect(lambda: self.remove_source())
        b_ref.clicked.connect(lambda: self.refresh())
        b_imp = QPushButton("Import assignments…")
        b_imp.setToolTip("Film, measurement, potential, CV start, ROI and notes from another library or "
                         "assignment file, onto the scans of the selected SPEC file (by scan number)")
        b_imp.clicked.connect(lambda: self.import_dialog())
        dl.addWidget(row(b_add, b_more, stretch_first=False))
        dl.addWidget(row(b_rm, b_ref, b_imp, stretch_first=False))
        dl.addWidget(hint("Files are opened for reading only. The library refers to them by path (relative to "
                          "the library file), so keep the folders together when moving them."))

        # assign
        ag, af = group("Assign to the selected scans")
        self._film = QComboBox()
        self._film.setEditable(True)
        self._film.setInsertPolicy(QComboBox.NoInsert)
        self._film.lineEdit().setPlaceholderText("e.g. 10 nm RuO2")
        b_film = QPushButton("Set")
        b_film.clicked.connect(lambda: self.assign_selected(film=self._film.currentText()))
        af.addRow("Film", row(self._film, b_film))
        self._rb_cv, self._rb_ca, self._rb_ocv = QRadioButton("CV"), QRadioButton("CA at"), QRadioButton("OCV")
        self._rb_cv.setToolTip("Measured during a cyclic voltammogram; the start point is optional")
        self._rb_ca.setToolTip("Chronoamperometry: the potential held at this value")
        self._rb_ocv.setToolTip("At open circuit (no potential applied)")
        self._rb_ca.setChecked(True)
        bg = QButtonGroup(self)
        for b in (self._rb_cv, self._rb_ca, self._rb_ocv):
            bg.addButton(b)
        self._start = QSpinBox()
        self._start.setRange(-1, 1000000)
        self._start.setValue(-1)
        self._start.setSpecialValueText("start point not set")
        self._start.setPrefix("starts at point ")
        self._start.setToolTip("SPEC point (0-based) where the CV starts: t = 0 on the 'time from CV start' axis. "
                               "Not set: the scans keep the start they have.")
        self._start.setEnabled(False)
        self._rb_cv.toggled.connect(self._start.setEnabled)
        self._pot = QDoubleSpinBox()
        self._pot.setRange(-5, 5)
        self._pot.setDecimals(3)
        self._pot.setSingleStep(0.05)
        self._pot.setValue(1.0)
        self._pot.setSuffix(" " + lb.DEFAULT_SETTINGS["unit"])
        self._rb_ca.toggled.connect(self._pot.setEnabled)
        b_mode, b_clear = QPushButton("Set"), QPushButton("Clear")
        b_mode.clicked.connect(lambda: self.assign_selected(**self._mode_fields()))
        b_clear.setToolTip("Back to no measurement assigned")
        b_clear.clicked.connect(lambda: self.assign_selected(mode=""))
        af.addRow("Measurement", row(self._rb_cv, self._start, stretch_first=False))
        af.addRow("", row(self._rb_ca, self._pot, stretch_first=False))
        af.addRow("", row(self._rb_ocv, 1, b_mode, b_clear, stretch_first=False))
        self._roi_set = QComboBox()
        b_roi = QPushButton("Set")
        b_roi.clicked.connect(lambda: self.assign_selected(roi=self._roi_value(self._roi_set.currentText())))
        af.addRow("ROI", row(self._roi_set, b_roi))
        self._notes = QLineEdit()
        b_notes = QPushButton("Set")
        b_notes.clicked.connect(lambda: self.assign_selected(notes=self._notes.text()))
        af.addRow("Notes", row(self._notes, b_notes))
        b_use, b_skip = QPushButton("Use"), QPushButton("Skip")
        b_use.setToolTip("Selected scans take part in the comparisons")
        b_skip.setToolTip("Selected scans are left out of the comparisons (kept in the library)")
        b_use.clicked.connect(lambda: self.assign_selected(use=True))
        b_skip.clicked.connect(lambda: self.assign_selected(use=False))
        af.addRow("", row(b_use, b_skip, 1, stretch_first=False))
        af.addRow(hint("Or edit the table: typing a film, potential, CV start or note with several rows selected "
                       "sets them all; a potential makes the scan a CA, a start point a CV. Ctrl+Z undoes, "
                       "Ctrl+Y redoes."))

        # intensity
        ig, iff = group("Intensity")
        self._roi_def = QComboBox()
        self._roi_def.setToolTip("ROI of the scans whose ROI is (default) in the table")
        self._monitor, self._transm, self._ctime = QComboBox(), QComboBox(), QComboBox()
        self._monitor.setToolTip("I is divided by monitor / (mean monitor of the plotted scans)")
        self._transm.setToolTip("Attenuator transmission per point (I is divided by it)")
        self._ctime.setToolTip("Count time per point (I is divided by it)")
        for w in (self._roi_def, self._monitor, self._transm, self._ctime):
            w.currentIndexChanged.connect(self._settings_changed)
        self._sys = QDoubleSpinBox()
        self._sys.setRange(0, 100)
        self._sys.setDecimals(1)
        self._sys.setSingleStep(1)
        self._sys.setSuffix(" % of I")
        self._sys.setToolTip("Systematic error added in quadrature to the counting error (error band and exported "
                             "I_err); usually 5 to 10 %")
        self._sys.valueChanged.connect(self._settings_changed)
        iff.addRow("Default ROI", self._roi_def)
        iff.addRow("Monitor", self._monitor)
        iff.addRow("Transmission", self._transm)
        iff.addRow("Count time", self._ctime)
        iff.addRow("Systematic error", self._sys)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(sidebar(fg, dg, ag, ig, width=400), 1)
        lv.addWidget(primary_button("Save library", self.save, "Ctrl+S", self))
        for key, slot in (("F5", self.refresh), ("Ctrl+Z", self.undo), ("Ctrl+Y", self.redo),
                          ("Ctrl+Shift+Z", self.redo)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)

        # table
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter rows: words that must all appear (e.g. 'rod CA 10 nm', 'S2 CV')")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        self._chk_todo = QCheckBox("Only scans still to assign")
        self._chk_todo.toggled.connect(self._apply_filter)
        self._count = QLabel("")
        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
                                    | QAbstractItemView.AnyKeyPressed)
        self._table.horizontalHeader().setSectionResizeMode(C_CMD, QHeaderView.Stretch)
        self._table.horizontalHeaderItem(C_KIND).setToolTip(
            "From the SPEC scan: rod (L moves), stationary (H K L fixed, e.g. during a CV or a CA hold), other "
            "(H or K moves), no SPEC (scan not in the SPEC file), missing (no ROI sums any more)")
        self._table.horizontalHeaderItem(C_ROI).setToolTip("The ROI this scan is shown with; (default): the default "
                                                           "ROI under Intensity")
        self._table.itemChanged.connect(self._table_edited)
        self._table.itemSelectionChanged.connect(self._selection_changed)
        top = QWidget()
        tv = QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.addWidget(row(self._filter, self._chk_todo, self._count))
        tv.addWidget(self._table, 1)

        # comparison
        cg, cf = group("Compare")
        b_cv = QPushButton("CVs at one H K L")
        b_cv.setToolTip("Stationary scans during CVs at one H K L, one colour per film")
        b_ca = QPushButton("CAs on one rod")
        b_ca.setToolTip("Rod scans held at one potential, one colour per film")
        b_cv.clicked.connect(lambda: self.preset("cv"))
        b_ca.clicked.connect(lambda: self.preset("ca"))
        cf.addRow(row(b_cv, b_ca, stretch_first=False))
        self._kind = QComboBox()
        self._kind.addItems(KIND_ITEMS)
        self._mode_f = QComboBox()
        self._mode_f.addItems(MODE_ITEMS)
        self._where = QComboBox()
        self._where.setToolTip("Rod (H K) of rod scans, or H K L of stationary scans")
        self._pot_f = QComboBox()
        self._pot_f.setToolTip("CA scans held at this potential (CV scans are not filtered by it)")
        self._films = QListWidget()
        self._films.setMaximumHeight(100)
        self._films.setToolTip("Films in the plot")
        self._films.itemChanged.connect(self._film_ticked)
        self._colour = QComboBox()
        self._colour.addItems(lb.COLOUR_BY)
        self._colour.setToolTip("One colour (and legend entry) per group; scans of one group (e.g. a rod in two "
                                "L ranges) share it")
        self._x = QComboBox()
        self._x.setToolTip("X axis of stationary scans: time from each scan's start, the SPEC point, or a SPEC "
                           "column (e.g. a potential counter)")
        self._norm = QComboBox()
        self._norm.addItems(lb.NORMS)
        self._norm.setToolTip("Divide each scan by its maximum or its first point (I / I0) to compare shapes")
        self._chk_log = QCheckBox("Log intensity")
        self._chk_log.setChecked(True)
        self._chk_err = QCheckBox("Error band")
        self._chk_err.setToolTip("± the counting error of the ROI ('_err' column of the ROI sums)")
        self._chk_sel = QCheckBox("Selected rows only")
        self._chk_sel.setToolTip("Plot the scans selected in the table (of the kind above) instead of the filters")
        self._stack = QDoubleSpinBox()
        self._stack.setRange(1, 1e4)
        self._stack.setDecimals(1)
        self._stack.setValue(1.0)
        self._stack.setToolTip("Log intensity: each group drawn this factor above the one before (1: overlaid)")
        for w in (self._kind, self._mode_f, self._where, self._pot_f, self._colour, self._x, self._norm):
            w.currentIndexChanged.connect(self._compare_changed)
        for w in (self._chk_log, self._chk_err, self._chk_sel):
            w.toggled.connect(self._compare_changed)
        self._stack.valueChanged.connect(self._compare_changed)
        cf.addRow("Scans", self._kind)
        cf.addRow("Measurement", self._mode_f)
        cf.addRow("Rod / point", self._where)
        cf.addRow("Potential", self._pot_f)
        cf.addRow("Films", self._films)
        cf.addRow("Colour by", self._colour)
        cf.addRow("X (stationary)", self._x)
        cf.addRow("Scale", self._norm)
        cf.addRow(row(self._chk_log, self._chk_err, stretch_first=False))
        cf.addRow("Stack ×", self._stack)
        cf.addRow(self._chk_sel)
        b_exp, b_fig = QPushButton("Export data…"), QPushButton("Export figure…")
        b_exp.setToolTip("The plotted traces as text (x, I, I_err per scan)")
        b_fig.setToolTip("Style and save the plot as an image")
        b_exp.clicked.connect(lambda: self.export_data_dialog())
        b_fig.clicked.connect(lambda: self.export_figure())
        cf.addRow(row(b_exp, b_fig, stretch_first=False))

        self._fig = Figure(layout="constrained")
        self._canvas = SafeCanvas(self._fig)
        plot = QWidget()
        pv = QVBoxLayout(plot)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(NavigationToolbar(self._canvas, plot))
        pv.addWidget(self._canvas, 1)
        self._plot_info = QLabel("")
        self._plot_info.setWordWrap(True)
        self._plot_info.setStyleSheet("color: #666;")
        pv.addWidget(self._plot_info)
        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(sidebar(cg, width=320))
        bottom.addWidget(plot)
        bottom.setSizes([340, 760])
        right = QSplitter(Qt.Vertical)
        right.addWidget(top)
        right.addWidget(bottom)
        right.setSizes([360, 560])
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([420, 1080])
        self.setCentralWidget(split)
        self._fill_compare()

    # ------------------------------------------------------------------ state
    def _alive(self):
        return not (isdeleted(self._table) or isdeleted(self._canvas))

    def _set_dirty(self, dirty=True):
        self._dirty = dirty
        if dirty:
            where = "Ctrl+S saves" if self.lib.path else "Save library… to keep it"
            self._state.setText(f"Unsaved changes ({where}; saved on closing)" if self.lib.path else
                                f"Not saved yet ({where})")
            self._state.setStyleSheet("color: #b26a00;")
        else:
            self._state.setText("Saved" if self.lib.path else "")
            self._state.setStyleSheet("color: #2e7d32;")

    def _checkpoint(self):
        self._undo.append(self.lib.snapshot())
        del self._undo[:-100]
        self._redo.clear()

    def undo(self):
        if not self._undo:
            self.say("Nothing to undo")
            return
        self._redo.append(self.lib.snapshot())
        self.lib.restore(self._undo.pop())
        self._after_change("Undone")

    def redo(self):
        if not self._redo:
            self.say("Nothing to redo")
            return
        self._undo.append(self.lib.snapshot())
        self.lib.restore(self._redo.pop())
        self._after_change("Redone")

    def _after_change(self, message=None, table=True):
        """Everything that follows a change of the library: table, choices, plot, unsaved mark."""
        if table:
            self._fill_table()
        self._show_unit()
        self._fill_sources()
        self._fill_columns()
        self._fill_compare()
        self._set_dirty(True)
        self._schedule_plot()
        if message:
            self.say(message)

    # ------------------------------------------------------------------ library file
    def _keep_current(self):
        """Before the library is replaced: save it if it has a file; ask before dropping an unsaved one."""
        if not self._dirty or not self.lib.sources:
            return True
        if self.lib.path:
            return bool(self.guarded(lambda: self.save() or True))
        ans = QMessageBox.question(self, "Scan library", "The current library has not been saved. Discard it?",
                                   QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel)
        return ans == QMessageBox.Discard

    def new_library(self, ask=True):
        if ask and not self._keep_current():
            return
        self.lib = lb.Library()
        self._undo.clear()
        self._redo.clear()
        self._path.setText("")
        self._after_change("New library: add a SPEC file and its ROI sums")
        self._set_dirty(False)

    def open_library(self, path=None, ask=True):
        if ask and not self._keep_current():
            return
        path = path or QFileDialog.getOpenFileName(self, "Open scan library", self._folder(),
                                                   "Scan library (*.txt);;All files (*)")[0]
        if not path:
            return

        def go():
            lib = lb.read_library(path)
            self.lib = lib
            self._undo.clear()
            self._redo.clear()
            self._path.setText(lib.path)
            self._after_change()
            self._set_dirty(False)
            n = len(lib.entries)
            msg = f"Opened {os.path.basename(path)}: {n} scans from {len(lib.sources)} SPEC file(s)"
            if lib.notes:
                msg += f"; {len(lib.notes)} note(s): {lib.notes[0]}"
            self.say(msg)
            return lib
        return self.guarded(go)

    def save(self):
        if not self.lib.path:
            return self.save_as()
        return self.guarded(lambda: self._write(self.lib.path))

    def save_as(self, path=None):
        path = path or QFileDialog.getSaveFileName(self, "Save scan library", os.path.join(self._folder(),
                                                   "scan_library.txt"), "Scan library (*.txt);;All files (*)")[0]
        if not path:
            return None
        return self.guarded(lambda: self._write(path))

    def _write(self, path):
        out = lb.write_library(self.lib, path)
        self._path.setText(out)
        self._set_dirty(False)
        self.say(f"Saved {os.path.basename(out)} ({len(self.lib.entries)} scans)")
        self._save_settings(draft=False)
        return out

    def _folder(self):
        for p in (self.lib.path, *(s.spec for s in self.lib.sources)):
            if p:
                return os.path.dirname(p)
        return ""

    # ------------------------------------------------------------------ data
    def add_source_dialog(self):
        spec = QFileDialog.getOpenFileName(self, "SPEC file", self._folder(),
                                           "SPEC (*.spec *.dat *.txt);;All files (*)")[0]
        if not spec:
            return None
        sums = QFileDialog.getOpenFileNames(self, "ROI sums files of this SPEC file", os.path.dirname(spec),
                                            "ROI sums (*.txt);;All files (*)")[0]
        if not sums:
            return None
        return self.add_source(spec, sums)

    def add_source(self, spec, sums, label=""):
        def go():
            self._checkpoint()
            n0 = len(self.lib.entries)
            src = self.lib.add_source(spec, sums, label)
            new = len(self.lib.entries) - n0
            self._after_change()
            msg = f"{src.id} {src.label}: {new} new scan(s) with ROI sums"
            if self.lib.notes:
                msg += f"; {self.lib.notes[0]}" + (f" (+{len(self.lib.notes) - 1} more)" if len(self.lib.notes) > 1
                                                   else "")
            self.say(msg)
            return src
        return self.guarded(go)

    def more_sums_dialog(self):
        src = self._current_source()
        if src is None:
            self.say("Pick the SPEC file in the Data list first", error=True)
            return None
        sums = QFileDialog.getOpenFileNames(self, f"More ROI sums for {os.path.basename(src.spec)}",
                                            os.path.dirname(src.spec), "ROI sums (*.txt);;All files (*)")[0]
        return self.add_source(src.spec, sums) if sums else None

    def _current_source(self):
        it = self._sources.currentItem()
        return self.lib.source(it.data(Qt.UserRole)) if it is not None else None

    def remove_source(self, sid=None):
        src = self.lib.source(sid) if sid else self._current_source()
        if src is None:
            self.say("Pick the SPEC file to remove in the Data list", error=True)
            return
        self._checkpoint()
        self.lib.remove_source(src.id)
        self._after_change(f"Removed {src.id} {src.label} from the library (the files are untouched; Ctrl+Z undoes)")

    def import_dialog(self):
        path = QFileDialog.getOpenFileName(self, "Import assignments (a library or assignment file)", self._folder(),
                                           "Scan library (*.txt);;All files (*)")[0]
        return self.import_assignments(path) if path else None

    def import_assignments(self, path, sid=None):
        """Film, measurement, potential, CV start, ROI, use and notes of another library file onto the
        scans of source sid (default: the selected one, or the only one), by scan number."""
        src = self.lib.source(sid) if sid else (self._current_source() or
                                                (self.lib.sources[0] if len(self.lib.sources) == 1 else None))
        if src is None:
            self.say("Add the SPEC file first (or pick it in the Data list), then import its assignments", error=True)
            return None

        def go():
            other = lb.read_library(path, load=False)
            self._checkpoint()
            matched, added = lb.apply_assignments(self.lib, other, src.id)
            for key in ("unit", "sys_error", "roi"):              # the file's settings if this library has none
                if other.settings.get(key) and self.lib.settings.get(key) in ("", "0", None):
                    self.lib.settings[key] = other.settings[key]
            self._after_change(f"Imported {matched + added} assignment(s) onto {src.id} {src.label}: {matched} "
                               f"scan(s) in the data, {added} not (yet)")
            return matched, added
        return self.guarded(go)

    def refresh(self):
        def go():
            self.lib.load()
            self._after_change(table=True)
            msg = f"Read {len(self.lib.sources)} SPEC file(s): {len(self.lib.entries)} scans"
            if self.lib.notes:
                msg += f"; {self.lib.notes[0]}" + (f" (+{len(self.lib.notes) - 1} more)" if len(self.lib.notes) > 1
                                                   else "")
            self.say(msg)
        self.guarded(go)

    def _fill_sources(self):
        self._sources.blockSignals(True)
        cur = self._current_source()
        self._sources.clear()
        for s in self.lib.sources:
            n = sum(1 for e in self.lib.entries if e.source == s.id)
            it = QListWidgetItem(s.label or s.id)
            it.setData(Qt.UserRole, s.id)
            it.setFlags(it.flags() | Qt.ItemIsEditable)
            it.setToolTip(f"{s.id}: {n} scans\nSPEC: {s.spec}\nROI sums:\n" + "\n".join(s.sums))
            self._sources.addItem(it)
            if cur is not None and cur.id == s.id:
                self._sources.setCurrentItem(it)
        self._sources.blockSignals(False)

    def _source_renamed(self, item):
        src = self.lib.source(item.data(Qt.UserRole))
        if src is None or item.text().strip() == src.label:
            return
        self._checkpoint()
        src.label = item.text().strip()
        self._after_change(f"{src.id} is now '{src.label}'")

    # ------------------------------------------------------------------ settings of the library
    def _fill_columns(self):
        """ROI and SPEC column lists, showing the library's settings."""
        self._loading = True
        rois, cols = self.lib.roi_names(), self.lib.spec_columns()
        st = self.lib.settings
        for combo, items, value in ((self._roi_def, [AUTO_ROI] + rois, st.get("roi") or AUTO_ROI),
                                    (self._monitor, [NONE] + cols, st.get("monitor") or NONE),
                                    (self._transm, [NONE] + cols, st.get("transmission") or NONE),
                                    (self._ctime, [NONE] + cols, st.get("count_time") or NONE),
                                    (self._roi_set, [DEFAULT_ROI] + rois, self._roi_set.currentText())):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            if value and value not in items:
                combo.addItem(value)                   # a setting the data no longer have stays visible
            combo.setCurrentText(value)
            combo.blockSignals(False)
        self._sys.blockSignals(True)
        try:
            self._sys.setValue(float(st.get("sys_error") or 0))
        except ValueError:
            self._sys.setValue(0)
        self._sys.blockSignals(False)
        films = self.lib.films()
        text = self._film.currentText()
        self._film.blockSignals(True)
        self._film.clear()
        self._film.addItems(films)
        self._film.setEditText(text)
        self._film.blockSignals(False)
        self._loading = False

    def _settings_changed(self, *_):
        if self._loading:
            return
        col = lambda c: "" if c.currentText() in (NONE, AUTO_ROI, "") else c.currentText()
        new = {"roi": col(self._roi_def), "monitor": col(self._monitor), "transmission": col(self._transm),
               "count_time": col(self._ctime), "sys_error": f"{self._sys.value():g}"}
        if all(self.lib.settings.get(k, "") == v for k, v in new.items()):
            return
        self._checkpoint()
        self.lib.settings.update(new)
        self._refresh_rows(self.lib.entries)               # the (default) ROI tooltips
        self._set_dirty(True)
        self._schedule_plot()

    def _show_unit(self):
        """The unit box, the potential box's suffix and the table header from the library's unit."""
        unit = self.lib.settings.get("unit", "")
        if self._unit.text().strip() != unit:
            self._unit.setText(unit)
        self._pot.setSuffix(" " + unit if unit else "")
        self._table.horizontalHeaderItem(C_E).setText(f"Potential ({unit})" if unit else "Potential")

    def _unit_changed(self):
        unit = self._unit.text().strip()
        self._pot.setSuffix(" " + unit if unit else "")
        self._table.horizontalHeaderItem(C_E).setText(f"Potential ({unit})" if unit else "Potential")
        if unit != self.lib.settings.get("unit", ""):
            self._checkpoint()
            self.lib.settings["unit"] = unit
            self._fill_compare()
            self._set_dirty(True)
            self._schedule_plot()

    # ------------------------------------------------------------------ table
    def _fill_table(self):
        t = self._table
        t.blockSignals(True)
        t.clearSelection()
        t.setRowCount(len(self.lib.entries))
        self._row_of = {}
        for r, e in enumerate(self.lib.entries):
            self._row_of[e.key] = r
            self._set_row(r, e)
        t.resizeColumnsToContents()
        t.horizontalHeader().setSectionResizeMode(C_CMD, QHeaderView.Stretch)
        t.blockSignals(False)
        self._apply_filter()

    def _item(self, text, editable=False, tip=None):
        it = QTableWidgetItem(text)
        if not editable:
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        if tip:
            it.setToolTip(tip)
        return it

    def _set_row(self, r, e):
        t = self._table
        src = self.lib.source(e.source)
        use = QTableWidgetItem()
        use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        use.setCheckState(Qt.Checked if e.use else Qt.Unchecked)
        use.setToolTip("Unticked: left out of the comparisons")
        if e.kind == "rod":
            lr = f"{_fmt(e.L_min)} to {_fmt(e.L_max)}"
        else:
            lr = _fmt(e.L) or "?"
        hk = f"{_fmt(e.H)} {_fmt(e.K)}" if np.isfinite(e.H) else "?"
        scan = self._item(str(e.scan))
        scan.setData(Qt.UserRole, list(e.key))
        cells = {C_USE: use, C_SRC: self._item(src.label if src else e.source,
                                                tip=f"{e.source}: {src.spec}" if src else None),
                 C_SCAN: scan, C_KIND: self._item(e.kind), C_HK: self._item(hk), C_L: self._item(lr),
                 C_PTS: self._item(str(e.points)), C_FILM: self._item(e.film, True),
                 C_MODE: self._item(""), C_E: self._item(lb.format_potential(e.potential) if e.potential is not None
                                                          else "", True),
                 C_START: self._item("" if e.cv_start is None else str(e.cv_start), True,
                                     "SPEC point where the CV starts (t = 0 on 'time from CV start'); typing one "
                                     "makes the scan a CV"),
                 C_ROI: self._item(""), C_NOTES: self._item(e.notes, True), C_CMD: self._item(e.command)}
        if not e.film:
            cells[C_FILM].setBackground(QBrush(TODO))
            cells[C_FILM].setToolTip("No film yet")
        if e.mode == "CA" and e.potential is None:
            cells[C_E].setBackground(QBrush(BAD))
            cells[C_E].setToolTip("CA without a potential: type the held potential")
        elif e.mode != "CA":
            cells[C_E].setToolTip("Typing a potential makes the scan a CA")
        if e.kind in ("missing", "no SPEC"):
            for c in (C_KIND, C_HK, C_L):
                cells[c].setForeground(QBrush(QColor("#b00020")))
        for c, it in cells.items():
            t.setItem(r, c, it)
        # the two boxes of a row are made once and updated after (replaced widgets linger until deleted)
        mode = t.cellWidget(r, C_MODE)
        if not isinstance(mode, QComboBox):
            mode = QComboBox()
            mode.addItems(list(lb.MODES))
            mode.setToolTip("CV: during a cyclic voltammogram; CA: potential held (type it under Potential); "
                            "OCV: at open circuit")
            mode.currentTextChanged.connect(self._row_mode)
            t.setCellWidget(r, C_MODE, mode)
        mode.blockSignals(True)
        mode.setProperty("key", list(e.key))
        mode.setCurrentText(e.mode)
        mode.setStyleSheet("" if e.mode else "QComboBox { background: #fff1db; }")
        mode.blockSignals(False)
        roi = t.cellWidget(r, C_ROI)
        if not isinstance(roi, QComboBox):
            roi = QComboBox()
            roi.currentTextChanged.connect(self._row_roi)
            t.setCellWidget(r, C_ROI, roi)
        roi.blockSignals(True)
        roi.setProperty("key", list(e.key))
        roi.clear()
        roi.addItems([DEFAULT_ROI] + list(e.rois))
        if e.roi and e.roi not in e.rois:
            roi.addItem(e.roi)                       # picked before, no longer in the data
        roi.setCurrentText(e.roi or DEFAULT_ROI)
        for i, name in enumerate(e.rois, start=1):
            roi.setItemData(i, self.lib.roi_description(e, name), Qt.ToolTipRole)
        shown = self.lib.effective_roi(e)
        roi.setToolTip(f"Shown with: {shown or 'no ROI'}" + (f"\n{self.lib.roi_description(e, shown)}" if shown else ""))
        roi.blockSignals(False)

    def _refresh_rows(self, entries):
        self._table.blockSignals(True)
        for e in entries:
            r = self._row_of.get(e.key)
            if r is not None:
                self._set_row(r, e)
        for c in (C_FILM, C_E, C_NOTES):                # typed text wider than the empty column
            self._table.resizeColumnToContents(c)
        self._table.blockSignals(False)
        self._apply_filter()

    def _entry_at(self, r):
        it = self._table.item(r, C_SCAN)
        if it is None:
            return None
        sid, sn = it.data(Qt.UserRole)
        return self.lib.entry(sid, sn)

    def _selected_rows(self):
        sm = self._table.selectionModel()
        rows = sorted({i.row() for i in sm.selectedRows()}) if sm is not None else []
        return [r for r in rows if not self._table.isRowHidden(r)]

    def selected_entries(self):
        return [e for e in (self._entry_at(r) for r in self._selected_rows()) if e is not None]

    def select_scans(self, keys):
        """Select the rows of (source, scan) keys (for scripts and tests)."""
        t = self._table
        t.clearSelection()
        for k in keys:                       # the selection model, not selectRow (which follows Ctrl / Shift)
            r = self._row_of.get(tuple(k))
            if r is not None:
                t.selectionModel().select(t.model().index(r, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)

    def _roi_value(self, text):
        return "" if text in (DEFAULT_ROI, "") else text

    def _mode_fields(self):
        if self._rb_cv.isChecked():
            return {"mode": "CV"} if self._start.value() < 0 else {"mode": "CV", "cv_start": self._start.value()}
        if self._rb_ocv.isChecked():
            return {"mode": "OCV"}
        return {"mode": "CA", "potential": self._pot.value()}

    def assign_selected(self, **fields):
        """Set fields (film, mode, potential, roi, use, notes) of the selected scans."""
        es = self.selected_entries()
        if not es:
            self.say("Select scans in the table first (Ctrl+A: every row shown)", error=True)
            return 0
        self._checkpoint()
        self.lib.assign(es, **fields)
        self._refresh_rows(es)
        self._after_change(table=False)
        what = ", ".join(f"{k} = {v!r}" if k != "potential" else f"potential = {lb.format_potential(v)}"
                         for k, v in fields.items())
        self.say(f"{len(es)} scan(s): {what}")
        return len(es)

    def _sender_entry(self):
        key = self.sender().property("key") if self.sender() is not None else None
        return self.lib.entry(*key) if key else None

    def _row_mode(self, text):
        e = self._sender_entry()
        if e is None or e.mode == text:
            return
        self._checkpoint()
        self.lib.assign([e], mode=text)
        QTimer.singleShot(0, lambda: self._refresh_rows([e]))   # not while the combo is still emitting
        self._after_change(table=False)

    def _row_roi(self, text):
        e = self._sender_entry()
        if e is None or e.roi == self._roi_value(text):
            return
        self._checkpoint()
        self.lib.assign([e], roi=self._roi_value(text))
        QTimer.singleShot(0, lambda: self._refresh_rows([e]))    # the 'shown with' tooltip
        self._after_change(table=False)

    def _table_edited(self, item):
        if not self._alive():
            return
        c, r = item.column(), item.row()
        e = self._entry_at(r)
        if e is None:
            return
        if c == C_USE:
            use = item.checkState() == Qt.Checked
            if use != e.use:
                self._checkpoint()
                e.use = use
                self._after_change(table=False)
            return
        if c not in TYPED:
            return
        text = item.text().strip()
        rows = self._selected_rows()
        targets = [x for x in (self._entry_at(i) for i in rows) if x is not None] if r in rows and len(rows) > 1 \
            else [e]
        if c == C_START:
            try:
                fields = {"cv_start": int(text) if text else None}
                if fields["cv_start"] is not None and fields["cv_start"] < 0:
                    raise ValueError
            except ValueError:
                self.say(f"CV start '{text}' is not a SPEC point number (0 or more)", error=True)
                QTimer.singleShot(0, lambda: self._refresh_rows([e]))
                return
        elif c == C_E:
            try:
                fields = {"potential": float(text.split()[0].replace(",", ".")) if text else None}
            except ValueError:
                self.say(f"Potential '{text}' is not a number", error=True)
                QTimer.singleShot(0, lambda: self._refresh_rows([e]))    # not inside the item's own signal
                return
        else:
            fields = {"film" if c == C_FILM else "notes": text}
        self._checkpoint()
        self.lib.assign(targets, **fields)
        QTimer.singleShot(0, lambda: self._refresh_rows(targets))
        self._after_change(table=False)

    def _apply_filter(self, *_):
        if not self._alive():
            return
        words = self._filter.text().lower().split()
        todo = self._chk_todo.isChecked()
        shown = 0
        for r in range(self._table.rowCount()):
            e = self._entry_at(r)
            if e is None:
                continue
            src = self.lib.source(e.source)
            text = " ".join(str(x) for x in (e.source, src.label if src else "", e.scan, e.kind, e.film, e.mode,
                                             lb.format_potential(e.potential) if e.potential is not None else "",
                                             e.roi, e.notes, e.command, f"{_fmt(e.H)} {_fmt(e.K)}",
                                             "" if e.cv_start is None else f"start {e.cv_start}")).lower()
            hide = (todo and e.assigned and not (e.mode == "CA" and e.potential is None)) or \
                any(w not in text for w in words)
            self._table.setRowHidden(r, hide)
            shown += not hide
        n = self._table.rowCount()
        todo_n = sum(1 for e in self.lib.entries if not e.assigned)
        self._count.setText(f"{shown} of {n} shown, {todo_n} to assign")

    def _selection_changed(self):
        if self._chk_sel.isChecked() or self._last_traces:
            self._schedule_plot()

    # ------------------------------------------------------------------ comparison
    def _kind_value(self):
        return "rod" if self._kind.currentIndex() == 0 else "stationary"

    def _fill_compare(self):
        """Rod / point, potential, film and X lists from the library, keeping the picks."""
        if not hasattr(self, "_where"):
            return
        self._loading = True
        kind = self._kind_value()
        of_kind = [e for e in self.lib.entries if e.kind == kind and e.use]
        modes = MODE_SETS[self._mode_f.currentIndex()]
        shown = [e for e in of_kind if modes is None or e.mode in modes]
        unit = self.lib.settings.get("unit", "")
        cur = self._where.currentText()
        self._where.clear()
        self._where.addItem(ANY, None)
        if kind == "rod":
            for h, k in lb.rods(shown):
                self._where.addItem(f"{h} {k}", (h, k))
        else:
            for h, k, L in lb.positions(shown):
                self._where.addItem(f"{h} {k} {L:.3g}", (h, k, L))
        self._where.setCurrentText(cur if self._where.findText(cur) >= 0 else ANY)
        cur = self._pot_f.currentText()
        self._pot_f.clear()
        self._pot_f.addItem(ANY, None)
        for E in self.lib.holds(shown):
            self._pot_f.addItem(lb.OCV if E == lb.OCV else lb.format_potential(E, unit), E)
        self._pot_f.setCurrentText(cur if self._pot_f.findText(cur) >= 0 else ANY)
        self._pot_f.setEnabled(modes is None or "CA" in modes)
        self._films.blockSignals(True)
        self._films.clear()
        names = self.lib.films() + ([NO_FILM] if any(not e.film for e in self.lib.entries) else [])
        for f in names:
            it = QListWidgetItem(f)
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Unchecked if f in self._films_off else Qt.Checked)
            self._films.addItem(it)
        self._films.blockSignals(False)
        cur = self._x.currentText()
        self._x.clear()
        self._x.addItems([lb.X_TIME, lb.X_CV, lb.X_POINT] + [c for c in self.lib.spec_columns()
                                                    if c not in ("H", "K", "L")])
        if self._pending_x and self._x.findText(self._pending_x) >= 0:
            self._x.setCurrentText(self._pending_x)
            self._pending_x = None
        elif self._x.findText(cur) >= 0:
            self._x.setCurrentText(cur)
        self._x.setEnabled(kind == "stationary")
        self._stack.setEnabled(self._chk_log.isChecked())
        self._loading = False

    def _film_ticked(self, item):
        name = item.text()
        if item.checkState() == Qt.Checked:
            self._films_off.discard(name)
        else:
            self._films_off.add(name)
        self._schedule_plot()

    def _compare_changed(self, *_):
        if self._loading:
            return
        sender = self.sender()
        if sender in (self._kind, self._mode_f):
            self._fill_compare()
        self._stack.setEnabled(self._chk_log.isChecked())
        self._x.setEnabled(self._kind_value() == "stationary")
        self._schedule_plot()

    def preset(self, which):
        """'cv': CVs at one H K L; 'ca': CAs on one rod at one potential. One colour per film."""
        self._loading = True
        if which == "cv":
            self._kind.setCurrentIndex(1)
            self._mode_f.setCurrentIndex(1)
            self._chk_log.setChecked(False)
            self._pending_x = lb.X_CV                   # CVs lined up at their start points
        else:
            self._kind.setCurrentIndex(0)
            self._mode_f.setCurrentIndex(0)
            self._chk_log.setChecked(True)
        self._colour.setCurrentText("film")
        self._chk_sel.setChecked(False)
        self._loading = False
        self._fill_compare()
        self._loading = True
        if self._where.currentIndex() == 0 and self._where.count() > 1:
            self._where.setCurrentIndex(1)
        if which == "ca" and self._pot_f.currentIndex() == 0 and self._pot_f.count() > 1:
            self._pot_f.setCurrentIndex(1)
        self._loading = False
        self._schedule_plot()

    def comparison(self):
        """(entries, title) of the current comparison."""
        kind = self._kind_value()
        unit = self.lib.settings.get("unit", "")
        if self._chk_sel.isChecked():
            es = [e for e in self.selected_entries() if e.kind == kind and e.kind not in ("missing", "no SPEC")]
            return es, f"{len(es)} selected {kind} scan(s)"
        modes = MODE_SETS[self._mode_f.currentIndex()]
        films = None
        if self._films_off:
            films = {("" if f == NO_FILM else f) for f in
                     (self._films.item(i).text() for i in range(self._films.count()))
                     if f not in self._films_off}
        where = self._where.currentData()
        E = self._pot_f.currentData() if self._pot_f.isEnabled() else None
        rod = tuple(where[:2]) if where else None
        L = where[2] if where and kind == "stationary" else None
        es = lb.select(self.lib, kind, modes, films, E, rod, L)
        what = {0: "CA / OCV", 1: "CV", 2: "All assigned", 3: "All"}[self._mode_f.currentIndex()]
        if E == lb.OCV:
            head = "OCV"
        elif E is not None:
            head = ("CA" if self._mode_f.currentIndex() == 0 else what) + f" at {lb.format_potential(E, unit)}"
        else:
            head = what
        parts = [head]
        if kind == "rod":
            parts.append(f"rod {rod[0]} {rod[1]}" if rod else "every rod")
        else:
            parts.append(f"at ({rod[0]} {rod[1]} {L:.3g})" if where else "(stationary scans)")
            return es, " ".join(parts)
        return es, ", ".join(parts)

    def _schedule_plot(self):
        if self._alive():
            self._timer.start()

    def draw_plot(self):
        if not self._alive():
            return
        fig = self._fig
        fig.clear()
        es, title = self.comparison()
        trs, notes = lb.traces(self.lib, es, self._x.currentText() or lb.X_TIME, self._colour.currentText())
        self._last_traces = trs
        self._plot_info.setText("")
        self._plot_info.setToolTip("")
        if not trs:
            msg = "No scans match: " + title
            if es:
                msg = (f"{len(es)} scan(s) match ({title}) but have no ROI sums loaded yet:\n"
                       "Add SPEC + ROI sums (Data) to read them")
            if not self.lib.entries:
                msg = "Add a SPEC file and its ROI sums, then assign film and CV / CA to the scans"
            elif not any(e.assigned for e in self.lib.entries):
                msg += "\n(assign a film and CV / CA to the scans in the table)"
            fig.text(0.5, 0.5, msg, ha="center", va="center", color="#888", wrap=True)
            self._canvas.draw_idle()
            return
        ax = fig.add_subplot(111)
        log = self._chk_log.isChecked()
        norm = self._norm.currentText()
        stack = self._stack.value() if log else 1.0
        sel = {e.key for e in self.selected_entries()} if not self._chk_sel.isChecked() else set()
        groups = []
        for t in trs:
            if t.group not in groups:
                groups.append(t.group)
        colours = PALETTE if len(groups) <= len(PALETTE) else list(matplotlib.colormaps["tab20"].colors)
        seen = set()
        for t in trs:
            gi = groups.index(t.group)
            c = colours[gi % len(colours)]
            y, s = lb.scaled(t, norm)
            f = stack ** gi
            y = y * f
            s = s * f if s is not None else None
            if log:
                y = np.where(y > 0, y, np.nan)
            lw = 2.4 if t.entry.key in sel else 1.2
            label = t.group if t.group not in seen else "_nolegend_"
            seen.add(t.group)
            marker = "o" if t.x.size < 400 else "none"
            ax.plot(t.x, y, marker=marker, ms=2.5, lw=lw, color=c, label=label)
            if self._chk_err.isChecked() and s is not None:
                lo = y - s
                ax.fill_between(t.x, np.where(lo > 0, lo, np.nan) if log else lo, y + s, color=c, alpha=0.2, lw=0)
        ax.set_xlabel(trs[0].xlabel)
        ylab = {lb.NORMS[0]: "I (normalised)", lb.NORMS[1]: "I / I_max", lb.NORMS[2]: "I / I(first point)"}[norm]
        if stack > 1:
            ylab += f" (groups × {stack:g}^n)"
        ax.set_ylabel(ylab)
        if log:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, title=f"{self._colour.currentText()}", title_fontsize=8)
        self._canvas.draw_idle()
        info = f"{len(trs)} scan(s) in {len(groups)} group(s): " + ", ".join(
            f"{t.entry.source}:{t.entry.scan} ({t.roi})" for t in trs[:12]) + (" ..." if len(trs) > 12 else "")
        if notes:
            info += f"\n{len(notes)} note(s): {notes[0]}"
            self._plot_info.setToolTip("\n".join(notes))
        self._plot_info.setText(info)

    def export_data(self, path):
        if not self._last_traces:
            raise ValueError("nothing is plotted")
        _, title = self.comparison()
        out = lb.write_traces(path, self._last_traces, title)
        self.say(f"Wrote {len(self._last_traces)} trace(s) to {os.path.basename(out)}")
        return out

    def export_data_dialog(self):
        path = QFileDialog.getSaveFileName(self, "Export plotted data", os.path.join(self._folder(),
                                           "library_traces.txt"), "Text (*.txt);;All files (*)")[0]
        if path:
            self.guarded(lambda: self.export_data(path))

    def export_figure(self):
        from .figure_export import FigureExportWindow, spec_from_canvas
        self._timer.stop()
        self.draw_plot()
        try:
            spec = spec_from_canvas(self._fig)
        except ValueError as e:
            self.say(f"Nothing to export: {e}", error=True)
            return None
        win = getattr(self, "_export_window", None)
        if win is not None and not isdeleted(win):
            win.close()
        self._export_window = FigureExportWindow(spec, lambda: spec_from_canvas(self._fig), self, "library")
        self._export_window.show()
        self._export_window.raise_()
        return self._export_window

    # ------------------------------------------------------------------ settings
    WIDGETS = ("_kind", "_mode_f", "_colour", "_x", "_norm", "_chk_log", "_chk_err", "_stack")

    def _save_settings(self, draft=True):
        """Remember the library and the comparison; a library with changes is saved (to its file, or
        as a draft in the settings folder if it has none)."""
        if draft and self._dirty and self.lib.sources:
            try:
                if self.lib.path:
                    lb.write_library(self.lib, self.lib.path)
                    self._set_dirty(False)
                else:
                    lb.write_library(self.lib, settings_file(DRAFT), remember=False)
            except (OSError, ValueError):
                pass
        d = {"library": self.lib.path, "films_off": sorted(self._films_off),
             "draft": bool(draft and self._dirty and not self.lib.path and self.lib.sources)}
        for name in self.WIDGETS:
            w = getattr(self, name)
            d[name] = (w.currentText() if isinstance(w, QComboBox) else w.isChecked() if isinstance(w, QCheckBox)
                       else w.value())
        try:
            with open(settings_file(SETTINGS), "w") as f:
                json.dump(d, f, indent=1)
        except OSError:
            pass

    def _load_settings(self):
        try:
            with open(settings_file(SETTINGS)) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return
        self._loading = True
        for name in self.WIDGETS:
            if name not in d:
                continue
            w = getattr(self, name)
            if isinstance(w, QComboBox):
                if name == "_x":
                    self._pending_x = d[name]
                if w.findText(d[name]) >= 0:
                    w.setCurrentText(d[name])
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(d[name]))
            else:
                w.setValue(float(d[name]))
        self._films_off = set(d.get("films_off") or [])
        self._loading = False
        path = d.get("library")
        if path and os.path.isfile(path):
            QTimer.singleShot(0, lambda: self.open_library(path, ask=False))
        elif d.get("draft") and os.path.isfile(settings_file(DRAFT)):
            QTimer.singleShot(0, self._open_draft)

    def _open_draft(self):
        def go():
            self.lib = lb.read_library(settings_file(DRAFT))
            self.lib.path = None
            self._after_change(f"Restored the unsaved library ({len(self.lib.entries)} scans): Save library… to keep it")
        self.guarded(go)

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


def main(argv=None):
    return run_tool(LibraryWindow)


if __name__ == '__main__':
    main()

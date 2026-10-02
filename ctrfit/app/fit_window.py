"""Fitting window: data, parameter table, guided workflow, live figure of merit, results.

Run:  python -m ctrfit gui   (or python ctr_fit_gui.py)

All computation goes through ctrfit (Fit, workflow, plots); this file only holds the window.
"""
import sys
import time
import traceback
from pathlib import Path

try:  # same Qt choice as the simulator GUI
    from PyQt5 import QtCore, QtGui, QtWidgets
    Signal = QtCore.pyqtSignal
except ImportError:  # pragma: no cover
    from PySide6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.Signal

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
except ImportError:  # older matplotlib
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
import numpy as np
from matplotlib.figure import Figure

from ..core.settings import DEFAULTS, read_ini
from ..data.dataset import Dataset
from ..fit import workflow
from ..fit.fit import Fit, FitResult
from ..fit.global_fit import series
from ..model.expr import ExprError
from ..model.parameters import scoped_view
from ..model.model import Model
from ..project import plots
from ..project.project import Project
from ..project.report import per_dataset_names, write_report

GROUPS = ("all", "film", "surface", "scale", "instrument", "free")


# ==============================================================================================
# parameter table
# ==============================================================================================
class ParamTable(QtCore.QAbstractTableModel):
    COLS = ["Fit", "Name", "Value", "Error", "Min", "Max", "Unit", "Link", "Description"]
    EDITABLE = {"Value", "Min", "Max", "Link"}
    edited = Signal(str)          # error message, or "" after a good edit

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.errors, self.flagged = {}, set()
        self.locked = False
        self.live = {}                 # best values during a running fit (shown instead of the parameters)

    def set_params(self, params):
        self.beginResetModel()
        self.params, self.errors, self.flagged = params, {}, set()
        self.endResetModel()

    def refresh(self):
        self.beginResetModel()
        self.endResetModel()

    def refresh_values(self):
        col = self.COLS.index("Value")
        self.dataChanged.emit(self.index(0, col), self.index(self.rowCount() - 1, col))

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.params)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.COLS)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.COLS[section]
        return None

    def param(self, row):
        return self.params[self.params.names[row]]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        p, col = self.param(index.row()), self.COLS[index.column()]
        if role == QtCore.Qt.CheckStateRole and col == "Fit":
            return QtCore.Qt.Checked if p.fit else QtCore.Qt.Unchecked
        if role == QtCore.Qt.ToolTipRole:
            return f"{p.name} ({p.group}): {p.description}" + (f"\nlinked: {p.expr}" if p.linked else "")
        if role == QtCore.Qt.BackgroundRole:
            if p.name in self.flagged:
                return QtGui.QColor("#fff0c2")
            if p.linked:
                return QtGui.QColor("#eef3fb")
            return None
        if role == QtCore.Qt.UserRole:          # for the group filter
            return "free " + p.group if p.free else p.group
        if role not in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            return None
        if col == "Name":
            return p.name
        if col == "Value":
            v = self.live.get(p.name, p.value) if self.locked else p.value
            return f"{v:.6g}" if role == QtCore.Qt.DisplayRole else repr(v)
        if col == "Error":
            e = self.errors.get(p.name)
            return "" if e is None else f"{e:.3g}"
        if col in ("Min", "Max"):
            v = p.min if col == "Min" else p.max
            return "" if not np.isfinite(v) else (f"{v:g}" if role == QtCore.Qt.DisplayRole else repr(v))
        if col == "Unit":
            return p.unit
        if col == "Link":
            return p.expr or ""
        if col == "Description":
            return p.description
        return None

    def flags(self, index):
        f = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if self.locked:
            return f
        col = self.COLS[index.column()]
        if col == "Fit":
            return f | QtCore.Qt.ItemIsUserCheckable
        if col in self.EDITABLE and not (col == "Value" and self.param(index.row()).linked):
            return f | QtCore.Qt.ItemIsEditable
        return f

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if self.locked or not index.isValid():
            return False
        p, col = self.param(index.row()), self.COLS[index.column()]
        try:
            if col == "Fit" and role == QtCore.Qt.CheckStateRole:
                p.fit = QtCore.Qt.CheckState(value) == QtCore.Qt.Checked if not isinstance(value, bool) else value
            elif col == "Value":
                v = float(value)
                if not p.min <= v <= p.max:
                    raise ValueError(f"{p.name}: {v:g} is outside [{p.min:g}, {p.max:g}]")
                p.value = v
            elif col in ("Min", "Max"):
                v = float(value) if str(value).strip() else (-np.inf if col == "Min" else np.inf)
                lo, hi = (v, p.max) if col == "Min" else (p.min, v)
                if lo > hi:
                    raise ValueError(f"{p.name}: min must not exceed max")
                p.min, p.max = lo, hi
            elif col == "Link":
                text = str(value).strip()
                if text:
                    self.params.link(p.name, text)
                    self.params.values()
                elif p.linked:
                    self.params.unlink(p.name)
            else:
                return False
        except (ValueError, ExprError, KeyError) as ex:
            self.edited.emit(str(ex))
            return False
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))
        self.edited.emit("")
        return True


class GroupFilter(QtCore.QSortFilterProxyModel):
    def __init__(self):
        super().__init__()
        self.group = "all"

    def set_group(self, g):
        self.group = g
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        if self.group == "all":
            return True
        tag = self.sourceModel().data(self.sourceModel().index(row, 0, parent), QtCore.Qt.UserRole) or ""
        return ("free" in tag.split()) if self.group == "free" else tag.endswith(self.group)


# ==============================================================================================
# plots and background work
# ==============================================================================================
class PlotPanel(QtWidgets.QWidget):
    def __init__(self, placeholder=""):
        super().__init__()
        try:
            self.figure = Figure(layout="constrained")
        except TypeError:  # matplotlib < 3.5
            self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(240, 180)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(NavToolbar(self.canvas, self))
        lay.addWidget(self.canvas, 1)
        if placeholder:
            plots.message(self.figure, placeholder)

    def draw(self, fn, *args, **kw):
        try:
            fn(self.figure, *args, **kw)
        except Exception as ex:  # keep the window alive
            plots.message(self.figure, f"Could not draw this plot:\n{ex}")
            self.canvas.draw_idle()
            return traceback.format_exc()
        self.canvas.draw_idle()
        return None


class WorkerSignals(QtCore.QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(str, int, float)
    best = Signal(str, int, float, object)


class Worker(QtCore.QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()
        self.fit = None
        self._last = 0.0
        self._last_best = 0.0

    def hook(self, fit):
        self.fit = fit
        fit.progress = self.progress
        fit.on_best = self.best

    def best(self, stage, step, fom, values):
        """New best values: every DE generation; least squares at most every 0.1 s."""
        now = time.perf_counter()
        if stage == "de" or now - self._last_best > 0.1:
            self._last_best = now
            self.signals.best.emit(stage, step, fom, values)

    def progress(self, stage, step, fom):
        now = time.perf_counter()
        if now - self._last > 0.15:          # throttle redraws
            self._last = now
            self.signals.progress.emit(stage, step, fom)

    def stop(self):
        if self.fit is not None:
            self.fit.cancel()

    def run(self):
        try:
            out = self.fn(self)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())
        else:
            self.signals.done.emit(out)


# ==============================================================================================
# main window
# ==============================================================================================
class FitWindow(QtWidgets.QMainWindow):
    def __init__(self, settings_dir=None):
        super().__init__()
        self.setWindowTitle("ctrfit: RuO2 / TiO2(110) CTR fitting")
        self.settings_dir = Path(settings_dir) if settings_dir else Path.cwd()
        self.project = Project(Model.from_settings(DEFAULTS), [])
        self.result = None
        self.fit = None
        self.history = []
        self.worker = None
        self._rod_cache = {}
        self.pool = QtCore.QThreadPool()
        self.pool.setMaxThreadCount(1)

        # ---------------- left: datasets and parameters
        self.data_list = QtWidgets.QListWidget()
        self.data_list.setMaximumHeight(110)
        self.floor = QtWidgets.QDoubleSpinBox(decimals=3, maximum=1.0, singleStep=0.005)
        self.floor.setToolTip("Systematic error floor (relative), added in quadrature to every point's sigma")
        b_add, b_del = QtWidgets.QPushButton("Add data…"), QtWidgets.QPushButton("Remove")
        self.table = ParamTable(self.model.params)
        self.proxy = GroupFilter()
        self.proxy.setSourceModel(self.table)
        self.view = QtWidgets.QTableView()
        self.view.setModel(self.proxy)
        self.view.verticalHeader().setVisible(False)
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.group = QtWidgets.QComboBox()
        self.group.addItems(GROUPS)
        self.n_free = QtWidgets.QLabel()

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.addWidget(QtWidgets.QLabel("<b>Datasets</b>"))
        ll.addWidget(self.data_list)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(b_add)
        row.addWidget(b_del)
        row.addWidget(QtWidgets.QLabel("Error floor"))
        row.addWidget(self.floor)
        ll.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("<b>Parameters</b>"))
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("Show"))
        row.addWidget(self.group)
        ll.addLayout(row)
        ll.addWidget(self.view, 1)
        ll.addWidget(self.n_free)

        # ---------------- right: fit controls and guided workflow
        self.method = QtWidgets.QComboBox()
        self.method.addItems(["DE + refine", "DE only", "refine only"])
        self.fom = QtWidgets.QComboBox()
        self.fom.addItems(["chi2", "log", "R1"])
        self.maxiter = QtWidgets.QSpinBox(minimum=1, maximum=10000, value=40)
        self.popsize = QtWidgets.QSpinBox(minimum=4, maximum=200, value=12)
        self.seed = QtWidgets.QSpinBox(minimum=0, maximum=10 ** 6, value=1)
        self.per_dataset = QtWidgets.QLineEdit("x_OH, z_OH")
        self.per_dataset.setToolTip("With several datasets: parameters fitted separately per dataset "
                                    "(everything else free is shared)")
        self.run_button = QtWidgets.QPushButton("Run fit")
        self.run_button.setMinimumHeight(32)
        self.step_buttons = [QtWidgets.QPushButton(t) for t in
                             ("1. Film on the even rods", "2. Surface on the odd rods", "3. Everything together")]
        form = QtWidgets.QFormLayout()
        form.addRow("Method", self.method)
        form.addRow("Figure of merit", self.fom)
        form.addRow("DE generations", self.maxiter)
        form.addRow("DE population", self.popsize)
        form.addRow("Seed", self.seed)
        form.addRow("Per dataset", self.per_dataset)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.addWidget(QtWidgets.QLabel("<b>Fit</b>"))
        rl.addLayout(form)
        rl.addWidget(self.run_button)
        rl.addSpacing(10)
        rl.addWidget(QtWidgets.QLabel("<b>Guided workflow</b>"))
        guide = QtWidgets.QLabel("Film first (thickness, roughness, strain, with a thickness scan), then the "
                                 "CUS surface on the rods it changes most, then refine everything.")
        guide.setWordWrap(True)
        rl.addWidget(guide)
        for b in self.step_buttons:
            rl.addWidget(b)
        rl.addStretch(1)
        right.setMaximumWidth(300)

        # ---------------- centre: plots and report
        self.rod_pick = QtWidgets.QComboBox()
        self.p_rod = PlotPanel("Add data to see the rods.")
        self.p_fom = PlotPanel("The figure of merit appears here during a fit.")
        self.p_corr = PlotPanel("Run a fit to see the correlation matrix.")
        self.series_pick = QtWidgets.QComboBox()
        self.p_series = PlotPanel("A global fit (several datasets) shows per-dataset parameters here.")
        self.report = QtWidgets.QPlainTextEdit(readOnly=True)
        self.report.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._stack(self.rod_pick, self.p_rod), "Data and model")
        self.tabs.addTab(self.p_fom, "Figure of merit")
        self.tabs.addTab(self.report, "Report")
        self.tabs.addTab(self.p_corr, "Correlations")
        self.tabs.addTab(self._stack(self.series_pick, self.p_series), "Series")
        self.tabs.addTab(self.log, "Log")

        split = QtWidgets.QSplitter()
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([480, 800, 280])
        self.setCentralWidget(split)
        self.status = QtWidgets.QLabel("Ready")
        self.statusBar().addWidget(self.status, 1)
        self._menus()
        from .assistant_panel import AssistantPanel
        self.assistant = AssistantPanel(self)
        dock = QtWidgets.QDockWidget("Ask Claude", self)
        dock.setObjectName("assistant")
        dock.setWidget(self.assistant)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable
                         | QtWidgets.QDockWidget.DockWidgetClosable)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self.assistant_dock = dock
        self.menuBar().addMenu("&View").addAction(dock.toggleViewAction())

        # ---------------- signals
        b_add.clicked.connect(lambda: self.add_data())
        b_del.clicked.connect(self.remove_data)
        self.floor.valueChanged.connect(self._set_floor)
        self.data_list.currentRowChanged.connect(lambda *_: self._sync_floor())
        self.group.currentTextChanged.connect(self.proxy.set_group)
        self.table.edited.connect(self._on_edit)
        self.rod_pick.currentIndexChanged.connect(lambda *_: self.show_rod())
        self.series_pick.currentIndexChanged.connect(lambda *_: self.show_series())
        self.run_button.clicked.connect(self.run_or_stop)
        for i, b in enumerate(self.step_buttons):
            b.clicked.connect(lambda _=False, s=("film", "surface", "all")[i]: self.run_step(s))
        self.resize(1500, 900)
        self._refresh_all()

    # ------------------------------------------------------------------ helpers
    @property
    def model(self):
        return self.project.model

    @property
    def datasets(self):
        return self.project.datasets

    @staticmethod
    def _stack(top, bottom):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(top)
        lay.addWidget(bottom, 1)
        return w

    def _menus(self):
        m = self.menuBar().addMenu("&File")
        for text, fn, key in (("New model from defaults", self.new_model, None),
                              ("New model from settings or preset…", self.new_model_from_ini, None),
                              ("Open model…", self.open_model, None), ("Save model…", self.save_model, None),
                              ("Add data…", self.add_data, "Ctrl+D"), (None, None, None),
                              ("Open project…", self.open_project, "Ctrl+O"),
                              ("Save project…", self.save_project, "Ctrl+S"),
                              ("Export report folder…", self.export_report, None)):
            if text is None:
                m.addSeparator()
                continue
            a = m.addAction(text)
            a.triggered.connect(lambda _=False, f=fn: f())
            if key:
                a.setShortcut(QtGui.QKeySequence(key))

    def say(self, text, error=False):
        self.status.setText(text.splitlines()[0][:160])
        self.log.appendPlainText(("ERROR: " if error else "") + text)

    def warn(self, title, text):
        self.say(text, error=True)
        QtWidgets.QMessageBox.warning(self, title, text.strip().splitlines()[-1])

    def _on_edit(self, msg):
        if msg:
            self.warn("Parameter not changed", msg)
        else:
            self._update_free()
            self.show_rod()

    def _size_columns(self):
        cols = self.table.COLS
        for c in ("Name", "Value", "Error", "Min", "Max", "Unit"):
            self.view.resizeColumnToContents(cols.index(c))
        self.view.setColumnWidth(cols.index("Fit"), 34)

    def _update_free(self):
        names = self.model.params.free_names
        self.n_free.setText(f"{len(names)} free: {', '.join(names) or 'none'}")

    def _refresh_all(self):
        self.table.set_params(self.model.params)
        if self.result is not None:
            self.table.errors = dict(self.result.errors)
        self._size_columns()
        self._update_free()
        self.data_list.clear()
        for ds in self.datasets:
            pot = ds.meta.get("potential_V")
            self.data_list.addItem(f"{ds.name}: {len(ds)} points, rods {', '.join(ds.rod_labels)}"
                                   + (f", {pot:g} V" if pot is not None else ""))
        if self.datasets:
            self.data_list.setCurrentRow(0)
        self.rod_pick.blockSignals(True)
        self.rod_pick.clear()
        for i, ds in enumerate(self.datasets):
            for lab in ds.rod_labels:
                self.rod_pick.addItem(f"{ds.name}: ({lab} L)", (i, lab))
        self.rod_pick.blockSignals(False)
        self.show_rod()

    def _sync_floor(self):
        i = self.data_list.currentRow()
        if 0 <= i < len(self.datasets):
            self.floor.blockSignals(True)
            self.floor.setValue(self.datasets[i].sys_floor)
            self.floor.blockSignals(False)

    def _set_floor(self, v):
        i = self.data_list.currentRow()
        if 0 <= i < len(self.datasets):
            self.datasets[i].sys_floor = float(v)
            self.show_rod()

    # ------------------------------------------------------------------ files
    def _open_name(self, title, filt):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, str(self.settings_dir), filt)
        return path or None

    def _save_name(self, title, name, filt):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, str(self.settings_dir / name), filt)
        return path or None

    def add_data(self, path=None, **kw):
        path = path or self._open_name("Add data", "Data (*.csv *.dat *.json);;All files (*)")
        if not path:
            return None
        try:
            ds = Dataset.load(path, **kw)
        except (OSError, ValueError, KeyError) as ex:
            self.warn("Could not read the data", f"{path}: {ex}")
            return None
        self.datasets.append(ds)
        self._refresh_all()
        self.say(f"Added {ds.name}: {len(ds)} points on {len(ds.rod_labels)} rods")
        return ds

    def remove_data(self):
        i = self.data_list.currentRow()
        if 0 <= i < len(self.datasets):
            ds = self.datasets.pop(i)
            self._refresh_all()
            self.say(f"Removed {ds.name}")

    def new_model(self, settings=None):
        self.project.model = Model.from_settings(dict(DEFAULTS, **(settings or {})))
        self.result = None
        self._refresh_all()
        self.say("New model (template rutile110_film)")

    def new_model_from_ini(self, path=None):
        path = path or self._open_name("Settings or preset", "Settings (*.ini)")
        if path:
            vals, problems = read_ini(Path(path))
            for p in problems:
                self.say(p)
            self.new_model(vals)

    def open_model(self, path=None):
        path = path or self._open_name("Open model", "Model (*.json)")
        if path:
            try:
                self.project.model = Model.load(path)
            except (OSError, ValueError, KeyError) as ex:
                self.warn("Could not read the model", f"{path}: {ex}")
                return
            self.result = None
            self._refresh_all()
            self.say(f"Opened model {Path(path).name}")

    def save_model(self, path=None):
        path = path or self._save_name("Save model", "model.json", "Model (*.json)")
        if path:
            self.model.save(path)
            self.say(f"Saved model {Path(path).name}")

    def open_project(self, path=None):
        path = path or self._open_name("Open project", "ctrfit project (*.ctrproj.json *.json)")
        if not path:
            return
        try:
            self.project = Project.load(path)
        except (OSError, ValueError, KeyError) as ex:
            self.warn("Could not open the project", f"{path}: {ex}")
            return
        self.result = FitResult(**{k: v for k, v in self.project.results[-1].items() if k != "summary"}) \
            if self.project.results else None
        self._refresh_all()
        if self.result is not None:
            self._show_result()
        self.say(f"Opened project {Path(path).name}")

    def save_project(self, path=None):
        path = path or self._save_name("Save project", "fit.ctrproj.json", "ctrfit project (*.ctrproj.json)")
        if path:
            self.project.save(path)
            self.say(f"Saved project {Path(path).name}")

    def export_report(self, folder=None):
        if self.result is None or self.fit is None:
            self.say("Run a fit first.")
            return
        folder = folder or QtWidgets.QFileDialog.getExistingDirectory(self, "Report folder", str(self.settings_dir))
        if folder:
            files = write_report(self.fit, self.result, folder)
            self.say(f"Wrote {len(files)} files to {folder}")

    # ------------------------------------------------------------------ plots
    def _rod_evaluators(self, i, lab):
        """(data-point evaluator, dense-curve evaluator, L of the dense curve) for one rod, cached.
        Built on the GUI thread; called with explicit values, so a running fit is never read."""
        ds = self.datasets[i]
        scope = ds.scope if len(self.datasets) > 1 else ""
        key = (id(self.model), id(ds), lab, scope)
        hit = self._rod_cache.get(key)
        if hit is None:
            idx = ds.rods()[lab]
            H, K = ds.rod_hk(lab)
            Ld = np.linspace(ds.L[idx].min(), ds.L[idx].max(), 400)
            hit = (self.model.evaluator(ds.H[idx], ds.K[idx], ds.L[idx], scope=scope),
                   self.model.evaluator(np.full(Ld.size, H), np.full(Ld.size, K), Ld, scope=scope), Ld, scope)
            self._rod_cache = {key: hit}
        return hit

    def show_rod(self, values=None, note=""):
        """Draw the selected rod. values: all parameter values (e.g. the live best of a running fit)."""
        data = self.rod_pick.currentData()
        if data is None:
            return
        i, lab = data
        ds = self.datasets[i]
        idx = ds.rods()[lab]
        H, K = ds.rod_hk(lab)
        try:
            ev_pts, ev_dense, Ld, scope = self._rod_evaluators(i, lab)
            v = self.model.params.view(scope) if values is None else scoped_view(values, scope)
            Fc, Fd = ev_pts(v), ev_dense(v)
        except (ValueError, KeyError) as ex:
            self.say(f"Model not evaluated: {ex}", error=True)
            return
        self.p_rod.draw(plots.draw_fit_rod, ds.L[idx], ds.F[idx], ds.sigma_eff[idx], Fc,
                        f"{ds.name}: ({H} {K} L)" + (f"  {note}" if note else ""), Ld, Fd)

    def show_series(self):
        name = self.series_pick.currentText()
        if self.result is None or not name:
            return
        x, v, e = series(self.result, name)
        has_pot = all("potential_V" in m for m in self.result.dataset_meta)
        self.p_series.draw(plots.draw_series, x, v, e, name, xlabel="potential (V)" if has_pot else "dataset")

    def _on_progress(self, stage, step, fom):
        self.history.append(dict(stage=stage, step=step, fom=fom))
        self.status.setText(f"{'differential evolution, generation' if stage == 'de' else 'least squares, evaluation'} "
                            f"{step}: figure of merit {fom:.6g}")
        self.p_fom.draw(plots.draw_fom_history, self.history)

    def _on_best(self, stage, step, fom, values):
        """Live update during a fit: parameter values and the model curve of the shown rod."""
        self.table.live = values
        self.table.refresh_values()
        what = f"generation {step}" if stage == "de" else f"least squares, evaluation {step}"
        self.show_rod(values, note=f"[{what}]")

    def _show_result(self):
        r = self.result
        self.table.errors = dict(r.errors)
        flagged = set()
        for w in r.warnings:
            flagged |= {n for n in r.names if n in w}
        self.table.flagged = flagged
        self.table.refresh()
        self._size_columns()
        self.report.setPlainText(r.summary)
        self.p_fom.draw(plots.draw_fom_history, r.history)
        if len(r.names) > 1:
            self.p_corr.draw(plots.draw_correlation, r.names, r.correlation)
        names = per_dataset_names(r)
        self.series_pick.blockSignals(True)
        self.series_pick.clear()
        self.series_pick.addItems(names)
        self.series_pick.blockSignals(False)
        self.show_series()
        self._update_free()
        self.show_rod()

    # ------------------------------------------------------------------ running
    def _busy(self, on):
        self.table.locked = on
        self.run_button.setText("Stop" if on else "Run fit")
        for b in self.step_buttons:
            b.setEnabled(not on)
        for w in (self.floor, self.data_list, self.rod_pick):
            w.setEnabled(not on)

    def _per_dataset(self):
        return [n.strip() for n in self.per_dataset.text().replace(";", ",").split(",") if n.strip()]

    def _check_ready(self):
        if not self.datasets:
            self.warn("No data", "Add a dataset first.")
            return False
        problems = self.model.params.check()
        if problems:
            self.warn("Check the parameters", "; ".join(problems))
            return False
        return True

    def _de_settings(self):
        return dict(maxiter=self.maxiter.value(), popsize=self.popsize.value(), seed=self.seed.value())

    def run_or_stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.say("Stopping… (keeping the best values so far)")
            return
        if not self._check_ready():
            return
        if not self.model.params.free_names:
            self.warn("Nothing to fit", "Tick the Fit box of at least one parameter.")
            return
        method, fom, de = self.method.currentText(), self.fom.currentText(), self._de_settings()
        per = self._per_dataset() if len(self.datasets) > 1 else []

        workflow.prepare(self.model, self.datasets, per)     # on this thread: the table shows the new rows
        self._refresh_all()

        def job(worker):
            fit = Fit(self.model, self.datasets, fom=fom)
            worker.hook(fit)
            if method != "refine only":
                fit.run("de", **de)
            if method != "DE only" and not fit.cancelled:
                fit.refine()
            return fit, fit.report()
        self._start(job, "Fitting")

    def run_step(self, step):
        if self.worker is not None or not self._check_ready():
            return
        de = self._de_settings()
        if len(self.datasets) > 1:
            workflow.prepare(self.model, self.datasets, self._per_dataset() if step != "film" else ())
            self._refresh_all()

        def job(worker):
            if step == "film":
                return workflow.step_film(self.model, self.datasets, de=de, fit_hook=worker.hook)
            if step == "surface":
                return workflow.step_surface(self.model, self.datasets, de=de, fit_hook=worker.hook)
            return workflow.step_all(self.model, self.datasets, fit_hook=worker.hook)
        self._start(job, {"film": "Step 1: film", "surface": "Step 2: surface", "all": "Step 3: everything"}[step])

    def _start(self, job, label):
        self.history = []
        w = Worker(job)
        self.worker = w
        w.signals.progress.connect(self._on_progress)
        w.signals.best.connect(self._on_best)
        w.signals.done.connect(lambda out, lab=label: self._done(out, lab))
        w.signals.failed.connect(self._failed)
        self._busy(True)
        self.say(f"{label} running…")
        self.pool.start(w)

    def _done(self, out, label):
        self.worker = None
        self.table.live = {}
        self._busy(False)
        self.fit, self.result = out
        self.project.results.append(self.result.to_dict())
        self._refresh_all()
        self._show_result()
        stopped = any(s.get("message") == "stopped by user" for s in self.result.stages)
        self.say(f"{label} {'stopped' if stopped else 'done'}: reduced chi2 {self.result.fom['red_chi2']:.4g}"
                 + (f", {len(self.result.warnings)} warning(s)" if self.result.warnings else ""))
        self.tabs.setCurrentIndex(2 if self.result.warnings else 0)

    def _failed(self, msg):
        self.worker = None
        self.table.live = {}
        self._busy(False)
        self.warn("Fit failed", msg)

    def wait(self, timeout=300.0):
        """Process events until the running fit has finished (for scripts and tests)."""
        app = QtWidgets.QApplication.instance()
        t0 = time.time()
        while self.worker is not None and time.time() - t0 < timeout:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.01)
        app.processEvents()
        return self.worker is None

    def closeEvent(self, ev):
        if self.worker is not None:
            self.worker.stop()
        self.pool.waitForDone(10000)
        super().closeEvent(ev)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setStyle("Fusion")
    win = FitWindow()
    for path in argv[1:]:
        if path.endswith(".ctrproj.json"):
            win.open_project(path)
        elif path.endswith(".json") and "model" in Path(path).name:
            win.open_model(path)
        else:
            win.add_data(path)
    win.show()
    return app.exec() if hasattr(app, "exec") else app.exec_()


if __name__ == "__main__":
    sys.exit(main())

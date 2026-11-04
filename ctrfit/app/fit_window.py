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
from .widgets import FLAG, LINK, ElidedLabel, muted_text, text_color, tint
from ..data.dataset import Dataset
from ..fit import workflow
from ..fit.fit import Fit, FitResult
from ..fit import parallel
from ..fit.global_fit import series, shadowed
from ..fit.sensitivity import curve_effect
from ..fit.scans import SEP, regroup
from ..model.expr import ExprError
from ..model.parameters import scoped_view
from ..model.model import Model
from ..project import plots
from ..project.project import Project
from ..project.report import per_dataset_names, write_report

GROUPS = ("all", "film", "surface", "scale", "instrument", "free", "selected dataset")
SCAN_COLS = ("Fit", "Scan", "Rod", "Points", "L", "Film", "Condition", "File", "Group")


# ==============================================================================================
# parameter table
# ==============================================================================================
class ParamTable(QtCore.QAbstractTableModel):
    """The parameters, each shared parameter followed by its copies (one per dataset or film), with the
    dataset each row belongs to. A shared parameter that every dataset replaces with its own copy is a
    master row: its Fit box fits all the copies that are not linked (one per film for per-film
    parameters), its Min / Max set their bounds; values are edited in the copies' own rows."""
    COLS = ["Fit", "Name", "Dataset", "Value", "Error", "Min", "Max", "Unit", "Link", "Description"]
    EDITABLE = {"Value", "Min", "Max", "Link"}
    edited = Signal(str)          # error message, or "" after a good edit

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.errors, self.flagged = {}, set()
        self.inactive = {}             # global name -> its copies, when every dataset uses its own copy
        self.labels = {}               # dataset scope -> dataset name, in dataset order
        self.locked = False
        self.live = {}                 # best values during a running fit (shown instead of the parameters)
        self.order = list(params.names)

    def set_context(self, labels, inactive):
        """The datasets (scope -> name) and the shared parameters every dataset overrides."""
        self.labels, self.inactive = dict(labels), dict(inactive)

    def _reorder(self):
        P = self.params
        rank = {sc: i for i, sc in enumerate(self.labels)}
        copies = {}
        for p in P:
            if p.scope:
                copies.setdefault(p.base, []).append(p.name)
        order = []
        for p in P:
            if not p.scope:
                order.append(p.name)
                order += sorted(copies.pop(p.name, []), key=lambda n: (rank.get(n.rpartition(".")[0], 1e9), n))
        self.order = order + [n for names in copies.values() for n in names]

    def set_params(self, params):
        self.beginResetModel()
        self.params, self.errors, self.flagged = params, {}, set()
        self._reorder()
        self.endResetModel()

    def refresh(self):
        self.beginResetModel()
        self._reorder()
        self.endResetModel()

    def refresh_cells(self):
        """Redraw every cell, keeping the selection (a reset would clear it)."""
        if self.rowCount():
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))

    def refresh_values(self):
        col = self.COLS.index("Value")
        self.dataChanged.emit(self.index(0, col), self.index(self.rowCount() - 1, col))

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.order)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.COLS)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.COLS[section]
        return None

    def name_at(self, row):
        return self.order[row]

    def row_of(self, name):
        return self.order.index(name)

    def param(self, row):
        return self.params[self.order[row]]

    # ------------------------------------------------------------------ master rows
    def copies(self, name, free_only=False):
        """The copies of a master row (those not linked, with free_only: the ones its Fit box sets)."""
        out = [self.params[n] for n in self.inactive.get(name, ()) if n in self.params]
        return [c for c in out if not c.linked] if free_only else out

    def _master_state(self, name):
        cs = self.copies(name, free_only=True)
        n = sum(c.fit for c in cs)
        return QtCore.Qt.Checked if cs and n == len(cs) else QtCore.Qt.PartiallyChecked if n else QtCore.Qt.Unchecked

    def _scope_label(self, scope):
        return self.labels[scope] if scope in self.labels else "(no dataset)"

    def _dataset_text(self, p):
        if p.scope:
            text = self._scope_label(p.scope)
            ref = (p.expr or "").strip()
            if p.linked and ref.rpartition(".")[2] == p.base and ref.rpartition(".")[0] in self.labels:
                text += f"  (follows {ref.rpartition('.')[0]})"
            return text
        if p.name in self.inactive:
            cs = self.copies(p.name)
            lead = [c for c in cs if not c.linked]
            return f"per film: {len(lead)} films, {len(cs)} datasets" if len(lead) < len(cs) else \
                f"per dataset: {len(cs)}"
        return "shared" if len(self.labels) > 1 else ""

    # ------------------------------------------------------------------ cells
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        p, col = self.param(index.row()), self.COLS[index.column()]
        master = p.name in self.inactive
        if role == QtCore.Qt.CheckStateRole and col == "Fit":
            if master:
                return self._master_state(p.name)
            return QtCore.Qt.Checked if p.fit else QtCore.Qt.Unchecked
        if role == QtCore.Qt.ToolTipRole:
            tip = f"{p.name} ({p.group}): {p.description}" + (f"\nlinked: {p.expr}" if p.linked else "")
            if master:
                tip += ("\nEvery dataset has its own copy (the rows below). Fit here fits all of them that are not "
                        "linked (one per film for per-film parameters); Min / Max here set all their bounds; "
                        "edit a value in its own row.")
            return tip
        if role == QtCore.Qt.FontRole and master:
            f = QtGui.QFont()
            f.setBold(True)
            return f
        if role == QtCore.Qt.BackgroundRole:            # from the palette: readable in a dark theme too
            if p.name in self.flagged:
                return tint(FLAG, 0.245)
            if p.linked:
                return tint(LINK, 0.085)
            return None
        if role == QtCore.Qt.ForegroundRole:
            if master and col == "Value":
                return muted_text()
            if p.name in self.flagged or p.linked:
                return text_color()
            return None
        if role == QtCore.Qt.UserRole:          # for the group filter
            free = p.free or (master and self._master_state(p.name) != QtCore.Qt.Unchecked)
            return "free " + p.group if free else p.group
        if role not in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            return None
        if col == "Name":
            return f"    ↳ {p.name}" if p.scope and role == QtCore.Qt.DisplayRole else p.name
        if col == "Dataset":
            return self._dataset_text(p)
        if col == "Value":
            if master:                           # the copies' values, edited in their rows
                vals = [self.live.get(c.name, c.value) if self.locked else c.value for c in self.copies(p.name)]
                lo, hi = (min(vals), max(vals)) if vals else (p.value, p.value)
                return f"{lo:.6g}" if np.isclose(lo, hi) else f"{lo:.4g} … {hi:.4g}"
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
        p = self.param(index.row())
        if col == "Fit":
            return f | QtCore.Qt.ItemIsUserCheckable
        if p.name in self.inactive and col in ("Value", "Link"):
            return f                             # values in the copies' rows; a master is not linked
        if col in self.EDITABLE and not (col == "Value" and p.linked):
            return f | QtCore.Qt.ItemIsEditable
        return f

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if self.locked or not index.isValid():
            return False
        p, col = self.param(index.row()), self.COLS[index.column()]
        master = p.name in self.inactive
        try:
            if col == "Fit" and role == QtCore.Qt.CheckStateRole:
                on = QtCore.Qt.CheckState(value) == QtCore.Qt.Checked if not isinstance(value, bool) else value
                if master:                           # the copies, not the unused shared parameter
                    for c in self.copies(p.name, free_only=True):
                        c.fit = on
                else:
                    p.fit = on
            elif col == "Value":
                v = float(value)
                if not p.min <= v <= p.max:
                    raise ValueError(f"{p.name}: {v:g} is outside [{p.min:g}, {p.max:g}]")
                p.value = v
            elif col in ("Min", "Max"):
                v = float(value) if str(value).strip() else (-np.inf if col == "Min" else np.inf)
                for q in [p] + (self.copies(p.name) if master else []):
                    lo, hi = (v, q.max) if col == "Min" else (q.min, v)
                    if lo > hi:
                        raise ValueError(f"{q.name}: min must not exceed max")
                for q in [p] + (self.copies(p.name) if master else []):
                    if col == "Min":
                        q.min = v
                    else:
                        q.max = v
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
    """Rows of one parameter group, the free ones, or those the selected dataset uses."""

    def __init__(self):
        super().__init__()
        self.group = "all"
        self.only = None               # names the selected dataset uses ("selected dataset")

    def set_group(self, g):
        self.group = g
        self.invalidateFilter()

    def set_only(self, names):
        self.only = None if names is None else set(names)
        if self.group == "selected dataset":
            self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        if self.group == "all":
            return True
        if self.group == "selected dataset":
            return self.only is None or self.sourceModel().name_at(row) in self.only
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

        # ---------------- left: datasets, their scans, parameters
        self.data_list = QtWidgets.QListWidget()
        self.data_list.setToolTip("The fitted datasets: one per group of scans (by default one per file), each with "
                                  "its own scale and its own copy of the Per dataset parameters")
        self.floor = QtWidgets.QDoubleSpinBox(decimals=3, maximum=1.0, singleStep=0.005)
        self.floor.setToolTip("Systematic error floor (relative) of the selected dataset, added in quadrature to "
                              "every point's sigma")
        b_add, b_del = QtWidgets.QPushButton("Add data…"), QtWidgets.QPushButton("Remove file")
        b_del.setToolTip("Remove the loaded file(s) of the selected scans (or of the selected dataset)")
        self.b_add, self.b_del = b_add, b_del
        self.scan_table = QtWidgets.QTableWidget(0, len(SCAN_COLS))
        self.scan_table.setHorizontalHeaderLabels(SCAN_COLS)
        self.scan_table.verticalHeader().setVisible(False)
        self.scan_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.scan_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.scan_table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked
                                        | QtWidgets.QAbstractItemView.EditKeyPressed)
        self.scan_table.horizontalHeader().setStretchLastSection(True)
        self.scan_table.setToolTip("Every scan of the loaded files. Untick Fit to leave a scan out. Scans with the same "
                                   "Group are fitted together as one dataset (double-click a Group to type one, or "
                                   "select rows and use Group…); by default a file is one group.")
        self._scan_rows = []
        self._filling = False
        self.scan_buttons = [QtWidgets.QPushButton(t) for t in ("Fit all", "Leave all out", "Group…",
                                                                "Fit selected", "Leave out selected", "Own file")]
        tips = ("Tick Fit for every scan", "Untick Fit for every scan (then tick the few you want)",
                "Fit the selected scans together as one dataset (type a group name)",
                "Fit the selected scans (Space toggles them)", "Leave the selected scans out of the fit (Space "
                "toggles them)", "Put the selected scans back in their file's own group")
        for b, t in zip(self.scan_buttons, tips):
            b.setToolTip(t)
        self.scan_info = ElidedLabel("No data loaded")
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
        self.n_free = ElidedLabel()            # one line: a long list of free parameters never widens the window
        self.param_buttons = [QtWidgets.QPushButton(t) for t in ("Fit shown", "Fix shown", "Fit selected",
                                                                 "Fix selected")]
        tips = ("Tick Fit for every parameter the Show filter lists (not linked ones, nor shared ones that every "
                "dataset overrides with its own copy)", "Untick Fit for every parameter the Show filter lists",
                "Tick Fit for the selected rows (Space toggles them)", "Untick Fit for the selected rows (Space "
                "toggles them)")
        for b, t in zip(self.param_buttons, tips):
            b.setToolTip(t)

        top = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(QtWidgets.QLabel("<b>Datasets</b> (fitted)"))
        self.data_list.setMaximumHeight(96)        # a few rows (it scrolls): room for the scan table
        tl.addWidget(self.data_list)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(b_add)
        row.addWidget(b_del)
        row.addWidget(QtWidgets.QLabel("Error floor"))
        row.addWidget(self.floor)
        tl.addLayout(row)
        info = QtWidgets.QLabel("<b>Scans</b>: untick to leave out; the same group is fitted together")
        info.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        tl.addWidget(info)
        self.scan_table.setMinimumHeight(110)
        tl.addWidget(self.scan_table, 1)
        grid = QtWidgets.QGridLayout()
        for i, b in enumerate(self.scan_buttons):
            grid.addWidget(b, i // 3, i % 3)
        tl.addLayout(grid)
        tl.addWidget(self.scan_info)
        left = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left.addWidget(top)
        bottom = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(bottom)
        ll.setContentsMargins(0, 0, 0, 0)
        left.addWidget(bottom)
        left.setSizes([540, 380])
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("<b>Parameters</b>"))
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("Show"))
        row.addWidget(self.group)
        ll.addLayout(row)
        ll.addWidget(self.view, 1)
        row = QtWidgets.QHBoxLayout()
        for b in self.param_buttons:
            row.addWidget(b)
        ll.addLayout(row)
        ll.addWidget(self.n_free)

        # ---------------- right: fit controls and guided workflow
        self.method = QtWidgets.QComboBox()
        self.method.addItems(["DE + refine", "DE only", "refine only"])
        self.fom = QtWidgets.QComboBox()
        self.fom.addItems(["chi2", "log", "R1"])
        self.maxiter = QtWidgets.QSpinBox(minimum=1, maximum=10000, value=40)
        self.popsize = QtWidgets.QSpinBox(minimum=4, maximum=200, value=12)
        self.seed = QtWidgets.QSpinBox(minimum=0, maximum=10 ** 6, value=1)
        self.solve_scales = QtWidgets.QCheckBox("Solve scales exactly")
        self.solve_scales.setChecked(True)
        self.solve_scales.setToolTip("Differential evolution does not search the dataset scales: for every trial "
                                     "structure their best values are computed directly (|F| is proportional to the "
                                     "scale). The search then needs no starting scale and converges faster.")
        self.starts = QtWidgets.QSpinBox(minimum=1, maximum=10, value=3)
        self.starts.setToolTip("Least squares from this many best distinct points of the global search (not only "
                               "the best one); the lowest chi2 is kept")
        self.cores = QtWidgets.QSpinBox(minimum=1, maximum=parallel.available_cores(), value=parallel.default_cores())
        self.cores.setToolTip("Model evaluations of differential evolution and of the least-squares Jacobian on "
                              "this many processes (1: all in this process, as before). The workers start once, "
                              "about a second, and are kept for later fits. Several cores give the same kind "
                              "of result but not the identical DE path as one core (fixed seed: reproducible).")
        self.hops = QtWidgets.QSpinBox(minimum=0, maximum=3, value=1)
        self.hops.setSuffix(" TL")
        self.hops.setToolTip("Also refine from the film thickness 1 ... n trilayers up and down (per film): the "
                             "thickness fringes give separate minima about one trilayer apart that a global search "
                             "often confuses. 0: off.")
        self.per_dataset = QtWidgets.QLineEdit("theta, x_OH, z_OH")
        self.per_dataset.setToolTip("With several datasets (e.g. one per potential): what the potential changes, "
                                    "fitted separately per dataset: the CUS coverage and species (theta, x_OH, "
                                    "x_O), their heights (z_OH, ...), near-surface O (ns_B_O, ns_du). Everything "
                                    "else free is shared.")
        self.per_film = QtWidgets.QLineEdit("thickness_mean_tl, roughness_tl, eps_perp")
        self.per_film.setToolTip("With several films: one value per film, shared by all potentials of that film "
                                 "(datasets with the same 'film' metadata, e.g. from the scan library's 'Export "
                                 "for fitting'). One film: the same as shared.")
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
        form.addRow(self.solve_scales)
        form.addRow("Refine starts", self.starts)
        form.addRow("Fringe hops", self.hops)
        form.addRow("Cores", self.cores)
        form.addRow("Per dataset", self.per_dataset)
        form.addRow("Per film", self.per_film)
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
        self.rod_pick.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.rod_pick.setMinimumContentsLength(24)
        self.p_rod = PlotPanel("Add data to see the rods.")
        self.sens_box = QtWidgets.QCheckBox("Colour by selected parameter")
        self.sens_box.setChecked(True)
        self.sens_box.setToolTip("Colour the model curve by how much |F| changes when the parameter selected in the "
                                 "table moves by ±1σ (its fit error; before a fit, a small step): red = no change, "
                                 "blue = the largest change on this rod (the colour bar gives it in %). Untick for "
                                 "the plain red curve.")
        rod_row = QtWidgets.QWidget()
        rl_ = QtWidgets.QHBoxLayout(rod_row)
        rl_.setContentsMargins(0, 0, 0, 0)
        rl_.addWidget(self.rod_pick, 1)
        rl_.addWidget(self.sens_box)
        self.p_fom = PlotPanel("The figure of merit appears here during a fit.")
        self.p_corr = PlotPanel("Run a fit to see the correlation matrix.")
        self.series_pick = QtWidgets.QComboBox()
        self.series_pick.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.series_pick.setMinimumContentsLength(16)
        self.p_series = PlotPanel("A global fit (several datasets) shows per-dataset parameters here.")
        self.report = QtWidgets.QPlainTextEdit(readOnly=True)
        self.report.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._stack(rod_row, self.p_rod), "Data and model")
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
        self.status = ElidedLabel("Ready")     # progress text changes every generation: fixed width
        self.statusBar().addWidget(self.status, 1)
        self._menus()

        # ---------------- signals
        b_add.clicked.connect(lambda: self.add_data())
        b_del.clicked.connect(lambda: self.remove_data())
        self.floor.valueChanged.connect(self._set_floor)
        self.data_list.currentRowChanged.connect(lambda *_: self._on_dataset_picked())
        self.data_list.itemClicked.connect(lambda *_: self._on_dataset_picked())
        self.scan_table.itemChanged.connect(self._on_scan_item)
        self.scan_table.currentCellChanged.connect(lambda r, *_: self._show_scan(r))
        for b, fn in zip(self.scan_buttons, (lambda: self.set_scan_use(self.scan_keys(), True),
                                             lambda: self.set_scan_use(self.scan_keys(), False),
                                             lambda: self.group_selected(),
                                             lambda: self.set_scan_use(self.selected_scans(), True),
                                             lambda: self.set_scan_use(self.selected_scans(), False),
                                             lambda: self.set_scan_group(self.selected_scans(), ""))):
            b.clicked.connect(lambda _=False, f=fn: f())
        for b, fn in zip(self.param_buttons, (lambda: self.set_param_fit(self.shown_params(), True),
                                              lambda: self.set_param_fit(self.shown_params(), False),
                                              lambda: self.set_param_fit(self.selected_params(), True),
                                              lambda: self.set_param_fit(self.selected_params(), False))):
            b.clicked.connect(lambda _=False, f=fn: f())
        self.scan_table.installEventFilter(self)          # Space toggles the selected rows
        self.view.installEventFilter(self)
        self.group.currentTextChanged.connect(self.proxy.set_group)
        self.table.edited.connect(self._on_edit)
        self.rod_pick.currentIndexChanged.connect(lambda *_: self.show_rod())
        self.sens_box.toggled.connect(lambda *_: self.show_rod())
        self.view.selectionModel().currentRowChanged.connect(
            lambda *_: self.show_rod() if self.sens_box.isChecked() else None)
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

    def _table_context(self):
        """The table's datasets (scope -> name) and master rows (shared parameters every dataset overrides)."""
        self.table.set_context({ds.scope: ds.name for ds in self.datasets if ds.scope},
                               shadowed(self.model.params, self.datasets))

    def _dataset_params(self, i):
        """The parameters dataset i uses: its own copies and the shared ones it does not override."""
        if not 0 <= i < len(self.datasets):
            return None
        sc, P = self.datasets[i].scope, self.model.params
        return [p.name for p in P if (sc and p.scope == sc) or (not p.scope and not (sc and f"{sc}.{p.name}" in P))]

    def _on_edit(self, msg):
        if msg:
            self.warn("Parameter not changed", msg)
        else:
            self._table_context()
            self.table.refresh_cells()
            self._update_free()
            self.show_rod()

    def _size_columns(self):
        cols = self.table.COLS
        for c in ("Name", "Dataset", "Value", "Error", "Min", "Max", "Unit"):
            self.view.resizeColumnToContents(cols.index(c))
        self.view.setColumnWidth(cols.index("Fit"), 34)

    def _update_free(self):
        names = self.model.params.free_names
        self.n_free.setText(f"{len(names)} free: {', '.join(names) or 'none'}")

    # ------------------------------------------------------------------ fit flags of many rows at once
    def shown_params(self):
        """Names of the parameters the Show filter lists, in table order."""
        return [self.table.name_at(self.proxy.mapToSource(self.proxy.index(r, 0)).row())
                for r in range(self.proxy.rowCount())]

    def _current_param(self):
        """The parameter the table's current row shows, if that row is selected (else the first selected)."""
        rows = self.selected_params()
        idx = self.view.currentIndex()
        cur = self.table.name_at(self.proxy.mapToSource(idx).row()) if idx.isValid() else None
        return cur if cur in rows else (rows[0] if rows else None)

    def selected_params(self):
        rows = {self.proxy.mapToSource(i).row() for i in self.view.selectionModel().selectedRows()}
        return [self.table.name_at(r) for r in sorted(rows) if 0 <= r < self.table.rowCount()]

    def _is_fit(self, name):
        """Fit flag as the table shows it (a master row: all its free copies)."""
        if name in self.table.inactive:
            cs = self.table.copies(name, free_only=True)
            return bool(cs) and all(c.fit for c in cs)
        return self.model.params[name].fit

    def set_param_fit(self, names, fit=True):
        """Tick (fit=True) or untick Fit for these parameters. A master row (a shared parameter every
        dataset overrides) stands for its copies that are not linked; ticking skips linked parameters
        (a link sets their value)."""
        if self.worker is not None or not names:
            return
        P = self.model.params
        names = list(dict.fromkeys(c.name for n in names for c in (
            self.table.copies(n, free_only=True) if n in self.table.inactive else [P[n]])))
        skip = [n for n in names if fit and P[n].linked]
        changed = [n for n in names if n not in skip and P[n].fit != fit]
        for n in changed:
            P[n].fit = fit
        self.table.refresh_cells()
        self._update_free()
        self.say(f"{'Fitting' if fit else 'Fixed'} {len(changed)} parameter(s)"
                 + (f"; skipped {len(skip)} linked or not used ({', '.join(skip[:4])}"
                    f"{', …' if len(skip) > 4 else ''})" if skip else "") + f": {len(P.free_names)} free")

    def eventFilter(self, obj, ev):
        """Space in the scan or parameter table toggles Fit of all selected rows (ticked if any was not)."""
        if ev.type() == QtCore.QEvent.KeyPress and ev.key() == QtCore.Qt.Key_Space and self.worker is None:
            if obj is self.scan_table and self.selected_scans():
                keys = self.selected_scans()
                use = {r.key: r.use for r in self._scan_rows}
                self.set_scan_use(keys, not all(use.get(k, True) for k in keys))
                return True
            if obj is self.view and self.selected_params():
                names = self.selected_params()
                self.set_param_fit(names, not all(self._is_fit(n) for n in names))
                return True
        return super().eventFilter(obj, ev)

    def _refresh_all(self):
        self._table_context()
        self.table.set_params(self.model.params)
        if self.result is not None:
            self.table.errors = dict(self.result.errors)
        self._size_columns()
        self._update_free()
        self.data_list.blockSignals(True)
        self.data_list.clear()
        for ds in self.datasets:
            pot = ds.meta.get("potential_V")
            film = ds.meta.get("film")
            cond = ds.meta.get("condition")
            extra = ([f"film {film}"] if film and film not in ds.name else []) + \
                ([f"{pot:g} V"] if pot is not None else [cond] if cond and cond not in ds.name else [])
            n = len(ds.scan_labels)
            self.data_list.addItem(f"{ds.name}: {len(ds)} points, {n} scan{'s' if n != 1 else ''}, "
                                   f"rods {', '.join(ds.rod_labels)}" + "".join(f", {x}" for x in extra))
        if self.datasets:
            self.data_list.setCurrentRow(0)
        self.data_list.blockSignals(False)
        self._sync_floor()
        self.proxy.set_only(self._dataset_params(self.data_list.currentRow()))
        self._fill_scans()
        self.rod_pick.blockSignals(True)
        self.rod_pick.clear()
        for i, ds in enumerate(self.datasets):
            out = self.project.scans.left_out(ds.name)
            extra = [lab for lab in (out.rod_labels if out is not None else []) if lab not in ds.rod_labels]
            for lab in ds.rod_labels + extra:          # a rod whose scans are all left out: still shown
                self.rod_pick.addItem(f"{ds.name}: ({lab} L)" + (" (left out)" if lab in extra else ""), (i, lab))
        self.rod_pick.blockSignals(False)
        self.show_rod()

    # ------------------------------------------------------------------ scans
    def _fill_scans(self):
        """The scan table from the loaded files and the choice per scan."""
        sel = self.project.scans
        keep = set(self.selected_scans())          # the selection survives a refill (same scans)
        self._scan_rows = sel.rows()
        t = self.scan_table
        self._filling = True
        t.clearSelection()
        t.setRowCount(len(self._scan_rows))
        grey = QtGui.QBrush(muted_text())
        for i, r in enumerate(self._scan_rows):
            cond = r.condition or (f"{r.potential_V:g} V" if r.potential_V is not None else "")
            vals = ("", r.scan, ", ".join(r.rods), str(r.n), f"{r.L_min:.2f}–{r.L_max:.2f}", r.film or "", cond,
                    r.source, r.group)
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
                if SCAN_COLS[c] == "Fit":
                    flags |= QtCore.Qt.ItemIsUserCheckable
                    it.setCheckState(QtCore.Qt.Checked if r.use else QtCore.Qt.Unchecked)
                elif SCAN_COLS[c] == "Group":
                    flags |= QtCore.Qt.ItemIsEditable
                    it.setToolTip("Scans with the same group are fitted together (one scale, one copy of the per-"
                                  "dataset parameters). Empty: the file's own group.")
                    if not r.custom:
                        it.setForeground(grey)
                if not r.use and SCAN_COLS[c] != "Fit":
                    it.setForeground(grey)
                it.setFlags(flags)
                t.setItem(i, c, it)
        t.resizeColumnsToContents()
        t.setColumnWidth(SCAN_COLS.index("Fit"), 34)
        sm = t.selectionModel()
        for row, r in enumerate(self._scan_rows):
            if r.key in keep:
                sm.select(t.model().index(row, 0), QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
        self._filling = False
        self.scan_info.setText(sel.summary())

    def selected_scans(self):
        """Keys of the scans selected in the scan table."""
        rows = sorted({i.row() for i in self.scan_table.selectionModel().selectedRows()})
        return [self._scan_rows[r].key for r in rows if r < len(self._scan_rows)]

    def scan_keys(self, source=None, scan=None):
        """Keys of the loaded scans, optionally of one file and / or one scan label (for scripts and tests)."""
        return [r.key for r in self.project.scans.rows()
                if (source is None or r.source == source) and (scan is None or r.scan == scan)]

    def _regroup(self):
        """Rebuild the fitted datasets from the files and the scan choices; parameters follow (fit.scans)."""
        self.project.datasets, notes = regroup(self.model, self.project.scans, self.project.datasets)
        for n in notes:
            self.log.appendPlainText(n)
        self._rod_cache = {}
        self._refresh_all()
        return notes

    def set_scan_use(self, keys, use=True):
        """Fit (use=True) or leave out these scans."""
        if self.worker is not None or not keys:
            return
        self.project.scans.set_use(keys, use)
        self._regroup()
        self.say(f"{len(keys)} scan(s) {'fitted' if use else 'left out'}: {self.project.scans.summary()}")

    def set_scan_group(self, keys, group=""):
        """Fit these scans together as the dataset `group` ('' = back to their file's group)."""
        if self.worker is not None or not keys:
            return
        self.project.scans.set_group(keys, group)
        notes = self._regroup()
        what = f"grouped as {group.strip()}" if group.strip() else "back in their file's group"
        self.say(f"{len(keys)} scan(s) {what}: {self.project.scans.summary()}" + (f" ({notes[0]})" if notes else ""))

    def group_selected(self, name=None):
        keys = self.selected_scans()
        if not keys:
            self.say("Select the scans to group in the scan table first.")
            return
        if name is None:
            rows = {r.key: r for r in self._scan_rows}
            name, ok = QtWidgets.QInputDialog.getText(self, "Group scans", "Fit the selected scans together as:",
                                                      text=rows[keys[0]].group)
            if not ok:
                return
        self.set_scan_group(keys, name)

    def _on_scan_item(self, item):
        if self._filling or item.row() >= len(self._scan_rows):
            return
        r = self._scan_rows[item.row()]
        col = SCAN_COLS[item.column()]
        if col == "Fit":
            use = item.checkState() == QtCore.Qt.Checked
            if use != r.use:
                QtCore.QTimer.singleShot(0, lambda: self.set_scan_use([r.key], use))   # not inside the item signal
        elif col == "Group" and item.text().strip() != r.group:
            QtCore.QTimer.singleShot(0, lambda t=item.text(): self.set_scan_group([r.key], t))

    def _show_scan(self, row):
        """Show the clicked scan's rod in its dataset."""
        if not 0 <= row < len(self._scan_rows):
            return
        r = self._scan_rows[row]
        for j in range(self.rod_pick.count()):
            i, lab = self.rod_pick.itemData(j)
            if self.datasets[i].name == r.group and lab == r.rods[0]:
                self.rod_pick.setCurrentIndex(j)
                return

    def _on_dataset_picked(self):
        """A dataset picked in the list: its floor, its parameters (Show: selected dataset), and the scans of
        its group selected in the scan table."""
        self._sync_floor()
        i = self.data_list.currentRow()
        self.proxy.set_only(self._dataset_params(i))
        if not 0 <= i < len(self.datasets):
            return
        name = self.datasets[i].name
        t = self.scan_table
        sm = t.selectionModel()
        sm.clearSelection()
        for row, r in enumerate(self._scan_rows):
            if r.group == name:                    # its scans, the left-out ones too
                sm.select(t.model().index(row, 0), QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

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
        """Add one dataset (path) or, from the dialog, several at once (e.g. every condition the scan
        library exported for fitting); returns the (last) dataset added."""
        if path is None:
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add data (one or several datasets)",
                                                              str(self.settings_dir),
                                                              "Data (*.csv *.dat *.json);;All files (*)")
            if not paths:
                return None
            added = self.add_datasets(paths, **kw)
            return added[-1] if added else None
        return (self.add_datasets([path], **kw) or [None])[-1]

    def add_datasets(self, paths, **kw):
        """Add several datasets; returns those that could be read."""
        added = []
        for path in paths:
            try:
                ds = Dataset.load(path, **kw)
            except (OSError, ValueError, KeyError) as ex:
                self.warn("Could not read the data", f"{path}: {ex}")
                continue
            added.append(self.project.scans.add(ds))
        if added:
            self._regroup()
            ds = added[-1]
            self.say(f"Added {ds.name}: {len(ds)} points, {len(ds.scan_labels)} scans on {len(ds.rod_labels)} rods"
                     if len(added) == 1 else f"Added {len(added)} datasets ({', '.join(d.name for d in added)})")
        return added

    def remove_data(self, files=None):
        """Remove loaded files: given by name, else those of the selected scans, else of the selected dataset."""
        if self.worker is not None:
            return
        if files is None:
            keys = self.selected_scans()
            i = self.data_list.currentRow()
            if not keys and 0 <= i < len(self.datasets):
                keys = getattr(self.datasets[i], "parts", [])
            files = list(dict.fromkeys(k.split(SEP, 1)[0] for k in keys))
        if not files:
            return
        self.project.scans.remove(files)
        self._regroup()
        self.say(f"Removed {', '.join(files)}")

    def clear_data(self):
        """Remove every loaded file."""
        self.project.scans.clear()
        self._regroup()

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
        scope = ds.scope                     # what the fit uses for this dataset
        key = (id(self.model), id(ds), lab, scope)
        hit = self._rod_cache.get(key)
        if hit is None:
            idx = ds.rods().get(lab, np.zeros(0, int))
            out = self.project.scans.left_out(ds.name)
            k = None if out is None else out.rod == lab
            left = None if k is None or not k.any() else (out.L[k], out.F[k], out.sigma_eff[k])
            H, K = ds.rod_hk(lab) if idx.size else out.rod_hk(lab)
            Lall = ds.L[idx] if left is None else np.concatenate([ds.L[idx], left[0]])
            Ld = np.linspace(Lall.min(), Lall.max(), 400)
            hit = (self.model.evaluator(ds.H[idx], ds.K[idx], ds.L[idx], scope=scope) if idx.size else
                   (lambda v: np.zeros(0)),
                   self.model.evaluator(np.full(Ld.size, H), np.full(Ld.size, K), Ld, scope=scope), Ld, scope,
                   left)
            self._rod_cache = {key: hit}
        return hit

    def _dataset_evaluator(self, i):
        """|F| at every point of dataset i (for the sensitivity colour scale), cached."""
        ds = self.datasets[i]
        key = (id(self.model), id(ds), ds.scope)
        if getattr(self, "_ds_eval", (None,))[0] != key:
            self._ds_eval = (key, self.model.evaluator(ds.H, ds.K, ds.L, scope=ds.scope))
        return self._ds_eval[1]

    def show_rod(self, values=None, note=""):
        """Draw the selected rod. values: all parameter values (e.g. the live best of a running fit)."""
        data = self.rod_pick.currentData()
        if data is None:
            return
        i, lab = data
        ds = self.datasets[i]
        idx = ds.rods().get(lab, np.zeros(0, int))
        try:
            ev_pts, ev_dense, Ld, scope, left = self._rod_evaluators(i, lab)
            v = self.model.params.view(scope) if values is None else scoped_view(values, scope)
            Fc, Fd = ev_pts(v), ev_dense(v)
            H, K = ev_dense.rods[0]
        except (ValueError, KeyError) as ex:
            self.say(f"Model not evaluated: {ex}", error=True)
            return
        sens, sens_label, sens_max = None, "", None
        name = self._current_param() if values is None and self.sens_box.isChecked() else None
        if name in self.table.inactive and f"{scope}.{name}" in self.model.params:
            name = f"{scope}.{name}"             # a master row: this dataset's own copy
        if name is not None:                   # not during a fit: it moves the parameter for a moment
            err = self.result.errors.get(name) if self.result is not None else None
            try:
                sens, step, sens_label = curve_effect(self.model.params, name, ev_dense, scope, error=err)
                if sens is not None:           # one colour scale for every rod of the dataset
                    every, _, _ = curve_effect(self.model.params, name, self._dataset_evaluator(i), scope, step=step)
                    sens_max = float(np.nanmax(every)) if every is not None and every.size else None
                    sens_label += " (scale: all rods)"
            except (ValueError, KeyError) as ex:
                sens, sens_label = None, str(ex)
            if sens is None:
                self.status.setText(f"Colour: {sens_label}")
                sens_label = ""
        self.p_rod.draw(plots.draw_fit_rod, ds.L[idx], ds.F[idx], ds.sigma_eff[idx], Fc,
                        f"{ds.name}: ({H} {K} L)" + (f"  {note}" if note else ""), Ld, Fd,
                        hold=values is not None,       # a running fit: the axes stay with the data
                        scans=ds.point_scans()[idx], left_out=left, sensitivity=sens, sensitivity_label=sens_label,
                        sensitivity_max=sens_max)

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
        for w in (self.floor, self.data_list, self.rod_pick, self.scan_table, self.b_add, self.b_del, self.cores,
                  *self.scan_buttons, *self.param_buttons):
            w.setEnabled(not on)

    def _per_dataset(self):
        return [n.strip() for n in self.per_dataset.text().replace(";", ",").split(",") if n.strip()]

    def _per_film(self):
        per = set(self._per_dataset())
        return [n.strip() for n in self.per_film.text().replace(";", ",").split(",") if n.strip() and n.strip() not in per]

    def _check_ready(self):
        if not self.datasets:
            self.warn("No data", "Every scan is left out: tick Fit for at least one in the scan table."
                      if self.project.scans.sources else "Add a dataset first.")
            return False
        problems = self.model.params.check()
        if problems:
            self.warn("Check the parameters", "; ".join(problems))
            return False
        return True

    def _de_settings(self):
        return dict(maxiter=self.maxiter.value(), popsize=self.popsize.value(), seed=self.seed.value(),
                    solve_scales=self.solve_scales.isChecked())

    def _refine_settings(self, after_de=True):
        return dict(starts=self.starts.value() if after_de else 1, hops=self.hops.value())

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
        ref = self._refine_settings(after_de=method != "refine only")
        per = self._per_dataset() if len(self.datasets) > 1 else []

        workflow.prepare(self.model, self.datasets, per,      # on this thread: the table shows the new rows
                         self._per_film() if len(self.datasets) > 1 else [])
        self._refresh_all()

        cores = self.cores.value()

        def job(worker):
            fit = Fit(self.model, self.datasets, fom=fom, cores=cores)
            worker.hook(fit)
            if method != "refine only":
                fit.run("de", **de)
            if method != "DE only" and not fit.cancelled:
                fit.refine(**ref)
            return fit, fit.report()
        self._start(job, "Fitting")

    def run_step(self, step):
        if self.worker is not None or not self._check_ready():
            return
        de = self._de_settings()
        ref = dict(self._refine_settings(), cores=self.cores.value())
        if len(self.datasets) > 1:
            workflow.prepare(self.model, self.datasets, self._per_dataset() if step != "film" else (),
                             self._per_film())
            self._refresh_all()

        def job(worker):
            if step == "film":
                return workflow.step_film(self.model, self.datasets, de=de, fit_hook=worker.hook, **ref)
            if step == "surface":
                return workflow.step_surface(self.model, self.datasets, de=de, fit_hook=worker.hook, **ref)
            return workflow.step_all(self.model, self.datasets, fit_hook=worker.hook, hops=ref["hops"],
                                     cores=ref["cores"])
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
        parallel.shutdown()                     # the worker processes of a parallel fit
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

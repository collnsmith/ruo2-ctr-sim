"""SXRD Spec Data Plotter
Plot spec file scan data with background-subtracted ROI integration.
Includes publication-quality figure options dialog and peak fitting
for rocking curve integration.

Qt window of the data tools (ctr_data_tools.py); the logic is in ctrfit.datatools.
"""
import configparser
import json
import os
from collections import OrderedDict

import matplotlib
import numpy as np
from PyQt5.QtCore import Qt, QEvent, QObject
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import (QFormLayout, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
                             QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QFileDialog,
                             QCheckBox, QSplitter, QScrollArea, QListWidget, QListWidgetItem, QGridLayout,
                             QColorDialog, QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
                             QAbstractItemView, QDialogButtonBox)
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit

from ctrfit.app.datatools.common import (HINT_STYLE, PRIMARY_STYLE, StatusMixin, group, hint, row, run_tool,
                                         sidebar)
from ctrfit.datatools.paths import settings_dir, settings_file
from ctrfit.datatools.peaks import (FIT_MODELS, display_label, fit_peak, integrate_peak, load_label_map,
                                    numerical_area, parse_bgsub_dir, save_label_map)
from ctrfit.datatools.potential import (FIRST_SWEEPS, MARKS, MODES, hand_cv, load_potential,
                                        offset_for_point, point_time, potential_at, time_zero, trace_between)
from ctrfit.datatools.specfile import parse_spec_file


class _ScrollBlockFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            if isinstance(obj, (QSpinBox, QDoubleSpinBox, QComboBox)):
                if not obj.hasFocus():
                    event.ignore()
                    return True
        return False


class ColumnSelectorWidget(QWidget):
    def __init__(self, on_change=None, parent=None):
        super().__init__(parent)
        self._on_change = on_change
        self._columns = []
        self._source_tags = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Column", "X", "Y", "Norm"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            self._table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Fixed)
            self._table.setColumnWidth(c, 34 if c < 3 else 50)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)
        self._apply_contrast()

    def _dark(self):
        return self.palette().color(QPalette.Base).lightness() < 128

    def _apply_contrast(self):
        """Dark themes: bright grid lines and check boxes with a clear frame and a filled checked state
        (the native ones nearly vanish on a dark table)."""
        if self._dark():
            self._table.setStyleSheet(
                "QTableWidget { gridline-color: #9aa0a6; alternate-background-color: #2f3337; }"
                "QHeaderView::section { border: 1px solid #9aa0a6; padding: 2px 4px; }"
                "QCheckBox::indicator { width: 14px; height: 14px; border: 2px solid #d0d4d8; border-radius: 3px;"
                " background: #1b1d1f; }"
                "QCheckBox::indicator:hover { border-color: #ffffff; }"
                "QCheckBox::indicator:checked { background: #4ea3ff; border-color: #cfe6ff; }")
        else:
            self._table.setStyleSheet("")
        bg = QColor(110, 210, 130) if self._dark() else QColor(30, 130, 60)
        for i, col in enumerate(self._columns):
            item = self._table.item(i, 0)
            if item is not None and self._source_tags.get(col) == 'bgsub':
                item.setForeground(bg)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.PaletteChange:          # the system switched between light and dark
            self._apply_contrast()

    def set_columns(self, spec_cols, bgsub_cols=None):
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        self._columns = []; self._source_tags = {}
        all_cols = []
        for c in spec_cols:
            all_cols.append(c); self._source_tags[c] = 'spec'
        if bgsub_cols:
            for c in bgsub_cols:
                tag = f"[bg] {c}"; all_cols.append(tag); self._source_tags[tag] = 'bgsub'
        self._columns = all_cols
        self._table.setRowCount(len(all_cols))
        for i, col in enumerate(all_cols):
            item = QTableWidgetItem(col)
            if self._source_tags.get(col) == 'bgsub':
                item.setForeground(QColor(110, 210, 130) if self._dark() else QColor(30, 130, 60))
            self._table.setItem(i, 0, item)
            for c_idx in (1, 2, 3):
                chk = QCheckBox()
                chk.setStyleSheet("QCheckBox { margin-left: 6px; }")
                if c_idx == 1:
                    chk.toggled.connect(lambda checked, r=i: self._on_x_toggled(r, checked))
                elif c_idx == 2:
                    chk.toggled.connect(self._emit_change)
                else:
                    chk.toggled.connect(lambda checked, r=i: self._on_norm_toggled(r, checked))
                self._table.setCellWidget(i, c_idx, chk)
        self._table.blockSignals(False)

    def _on_x_toggled(self, row, checked):
        if checked:
            for i in range(self._table.rowCount()):
                if i != row:
                    w = self._table.cellWidget(i, 1)
                    if w: w.blockSignals(True); w.setChecked(False); w.blockSignals(False)
        self._emit_change()

    def _on_norm_toggled(self, row, checked):
        if checked:
            for i in range(self._table.rowCount()):
                if i != row:
                    w = self._table.cellWidget(i, 3)
                    if w: w.blockSignals(True); w.setChecked(False); w.blockSignals(False)
        self._emit_change()

    def _emit_change(self, *a):
        if self._on_change: self._on_change()

    def get_selection(self):
        x_col = None; y_cols = []; norm_col = None
        for i in range(self._table.rowCount()):
            name = self._columns[i]
            if self._table.cellWidget(i, 1) and self._table.cellWidget(i, 1).isChecked(): x_col = name
            if self._table.cellWidget(i, 2) and self._table.cellWidget(i, 2).isChecked(): y_cols.append(name)
            if self._table.cellWidget(i, 3) and self._table.cellWidget(i, 3).isChecked(): norm_col = name
        return x_col, y_cols, norm_col

    def set_selection(self, x_col, y_cols, norm_col):
        for i in range(self._table.rowCount()):
            name = self._columns[i]
            for c_idx, target in [(1, x_col), (2, y_cols), (3, norm_col)]:
                w = self._table.cellWidget(i, c_idx)
                if w:
                    w.blockSignals(True)
                    if c_idx == 2:
                        w.setChecked(name in target)
                    else:
                        w.setChecked(name == target)
                    w.blockSignals(False)


class FigureOptionsDialog(QDialog):
    """Non-modal dialog for figure styling. Lives as long as the main window."""
    def __init__(self, main_win, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Figure Options")
        self.setMinimumSize(480, 600)
        self._mw = main_win
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        inner = QWidget()
        vl = QVBoxLayout(inner)

        # Presets
        pr = QHBoxLayout()
        pr.addWidget(QLabel("Preset:"))
        b1 = QPushButton("Screen (dark)"); b1.clicked.connect(self._mw._preset_dark); pr.addWidget(b1)
        b2 = QPushButton("Paper (light)"); b2.clicked.connect(self._mw._preset_light); pr.addWidget(b2)
        vl.addLayout(pr)

        # Colors
        cg = QGroupBox("Colors")
        cl = QGridLayout(cg)
        cl.addWidget(QLabel("Fig bg:"), 0, 0)
        self._mw._btn_fig_bg = QPushButton(); self._mw._btn_fig_bg.setFixedSize(40, 20)
        self._mw._btn_fig_bg.clicked.connect(lambda: self._mw._pick_color('fig_bg'))
        cl.addWidget(self._mw._btn_fig_bg, 0, 1)
        cl.addWidget(QLabel("Axes bg:"), 0, 2)
        self._mw._btn_axes_bg = QPushButton(); self._mw._btn_axes_bg.setFixedSize(40, 20)
        self._mw._btn_axes_bg.clicked.connect(lambda: self._mw._pick_color('axes_bg'))
        cl.addWidget(self._mw._btn_axes_bg, 0, 3)
        cl.addWidget(QLabel("Text:"), 0, 4)
        self._mw._btn_text_color = QPushButton(); self._mw._btn_text_color.setFixedSize(40, 20)
        self._mw._btn_text_color.clicked.connect(lambda: self._mw._pick_color('text'))
        cl.addWidget(self._mw._btn_text_color, 0, 5)
        vl.addWidget(cg)

        # Figure size
        sg = QGroupBox("Figure size")
        sl = QGridLayout(sg)
        sl.addWidget(QLabel("Width (in):"), 0, 0)
        self._mw._fig_w = QDoubleSpinBox(); self._mw._fig_w.setRange(2, 30); self._mw._fig_w.setValue(8)
        self._mw._fig_w.setSingleStep(0.5); self._mw._fig_w.setDecimals(1)
        self._mw._fig_w.editingFinished.connect(self._mw._apply_figsize); sl.addWidget(self._mw._fig_w, 0, 1)
        sl.addWidget(QLabel("Height (in):"), 0, 2)
        self._mw._fig_h = QDoubleSpinBox(); self._mw._fig_h.setRange(2, 30); self._mw._fig_h.setValue(6)
        self._mw._fig_h.setSingleStep(0.5); self._mw._fig_h.setDecimals(1)
        self._mw._fig_h.editingFinished.connect(self._mw._apply_figsize); sl.addWidget(self._mw._fig_h, 0, 3)
        sl.addWidget(QLabel("DPI:"), 1, 0)
        self._mw._fig_dpi = QSpinBox(); self._mw._fig_dpi.setRange(50, 600); self._mw._fig_dpi.setValue(100)
        self._mw._fig_dpi.editingFinished.connect(self._mw._apply_figsize); sl.addWidget(self._mw._fig_dpi, 1, 1)
        vl.addWidget(sg)

        # Fonts
        fg = QGroupBox("Fonts")
        fl = QGridLayout(fg)
        fl.addWidget(QLabel("Family:"), 0, 0)
        self._mw._font_family = QComboBox()
        self._mw._font_family.addItems(["DejaVu Sans", "serif", "sans-serif", "monospace",
                                         "Times New Roman", "Arial", "Helvetica", "CMU Serif", "STIXGeneral"])
        self._mw._font_family.currentTextChanged.connect(self._mw._update_plot); fl.addWidget(self._mw._font_family, 0, 1, 1, 3)
        for row, (lbl, attr, val) in enumerate(
            [("Title size:", "_font_title", 14), ("Axis label:", "_font_axis", 12),
             ("Tick label:", "_font_tick", 10), ("Legend:", "_font_legend", 9)], start=1):
            fl.addWidget(QLabel(lbl), row, 0)
            sb = QSpinBox(); sb.setRange(4, 40); sb.setValue(val)
            sb.editingFinished.connect(self._mw._update_plot)
            setattr(self._mw, attr, sb); fl.addWidget(sb, row, 1)
        vl.addWidget(fg)

        # Lines & markers
        lmg = QGroupBox("Lines && markers")
        lml = QGridLayout(lmg)
        lml.addWidget(QLabel("Line style:"), 0, 0)
        self._mw._line_combo = QComboBox(); self._mw._line_combo.addItems(["-", "--", "-.", ":", "none"])
        self._mw._line_combo.currentTextChanged.connect(self._mw._update_plot); lml.addWidget(self._mw._line_combo, 0, 1)
        lml.addWidget(QLabel("Width:"), 0, 2)
        self._mw._line_width = QDoubleSpinBox(); self._mw._line_width.setRange(0.1, 10); self._mw._line_width.setValue(1.2)
        self._mw._line_width.setSingleStep(0.2); self._mw._line_width.setDecimals(1)
        self._mw._line_width.editingFinished.connect(self._mw._update_plot); lml.addWidget(self._mw._line_width, 0, 3)
        lml.addWidget(QLabel("Marker:"), 1, 0)
        self._mw._marker_combo = QComboBox(); self._mw._marker_combo.addItems(["o", "s", "^", "v", "D", "x", "+", ".", "none"])
        self._mw._marker_combo.currentTextChanged.connect(self._mw._update_plot); lml.addWidget(self._mw._marker_combo, 1, 1)
        lml.addWidget(QLabel("Size:"), 1, 2)
        self._mw._marker_size = QDoubleSpinBox(); self._mw._marker_size.setRange(0.5, 20); self._mw._marker_size.setValue(4)
        self._mw._marker_size.setSingleStep(0.5); self._mw._marker_size.setDecimals(1)
        self._mw._marker_size.editingFinished.connect(self._mw._update_plot); lml.addWidget(self._mw._marker_size, 1, 3)
        self._mw._chk_markers = QCheckBox("Show markers"); self._mw._chk_markers.setChecked(True)
        self._mw._chk_markers.toggled.connect(self._mw._update_plot); lml.addWidget(self._mw._chk_markers, 2, 0, 1, 2)
        vl.addWidget(lmg)

        # Grid, spines, ticks
        gg = QGroupBox("Grid / spines / ticks")
        gl = QGridLayout(gg)
        self._mw._chk_grid = QCheckBox("Grid"); self._mw._chk_grid.setChecked(True)
        self._mw._chk_grid.toggled.connect(self._mw._update_plot); gl.addWidget(self._mw._chk_grid, 0, 0)
        gl.addWidget(QLabel("Style:"), 0, 1)
        self._mw._grid_style = QComboBox(); self._mw._grid_style.addItems(["-", "--", "-.", ":"]); self._mw._grid_style.setCurrentIndex(3)
        self._mw._grid_style.currentTextChanged.connect(self._mw._update_plot); gl.addWidget(self._mw._grid_style, 0, 2)
        gl.addWidget(QLabel("Alpha:"), 0, 3)
        self._mw._grid_alpha = QDoubleSpinBox(); self._mw._grid_alpha.setRange(0, 1); self._mw._grid_alpha.setValue(0.3)
        self._mw._grid_alpha.setSingleStep(0.1); self._mw._grid_alpha.setDecimals(2)
        self._mw._grid_alpha.editingFinished.connect(self._mw._update_plot); gl.addWidget(self._mw._grid_alpha, 0, 4)
        gl.addWidget(QLabel("Grid W:"), 1, 0)
        self._mw._grid_width = QDoubleSpinBox(); self._mw._grid_width.setRange(0.1, 5); self._mw._grid_width.setValue(0.5)
        self._mw._grid_width.setSingleStep(0.1); self._mw._grid_width.setDecimals(1)
        self._mw._grid_width.editingFinished.connect(self._mw._update_plot); gl.addWidget(self._mw._grid_width, 1, 1)
        gl.addWidget(QLabel("Spine W:"), 1, 2)
        self._mw._spine_width = QDoubleSpinBox(); self._mw._spine_width.setRange(0, 5); self._mw._spine_width.setValue(1)
        self._mw._spine_width.setSingleStep(0.5); self._mw._spine_width.setDecimals(1)
        self._mw._spine_width.editingFinished.connect(self._mw._update_plot); gl.addWidget(self._mw._spine_width, 1, 3)
        gl.addWidget(QLabel("Tick W:"), 2, 0)
        self._mw._tick_width = QDoubleSpinBox(); self._mw._tick_width.setRange(0, 5); self._mw._tick_width.setValue(1)
        self._mw._tick_width.setSingleStep(0.5); self._mw._tick_width.setDecimals(1)
        self._mw._tick_width.editingFinished.connect(self._mw._update_plot); gl.addWidget(self._mw._tick_width, 2, 1)
        gl.addWidget(QLabel("Tick L:"), 2, 2)
        self._mw._tick_length = QDoubleSpinBox(); self._mw._tick_length.setRange(0, 20); self._mw._tick_length.setValue(4)
        self._mw._tick_length.setSingleStep(1); self._mw._tick_length.setDecimals(1)
        self._mw._tick_length.editingFinished.connect(self._mw._update_plot); gl.addWidget(self._mw._tick_length, 2, 3)
        self._mw._chk_tick_in = QCheckBox("Ticks in"); self._mw._chk_tick_in.toggled.connect(self._mw._update_plot)
        gl.addWidget(self._mw._chk_tick_in, 3, 0, 1, 2)
        self._mw._chk_minor_ticks = QCheckBox("Minor ticks"); self._mw._chk_minor_ticks.toggled.connect(self._mw._update_plot)
        gl.addWidget(self._mw._chk_minor_ticks, 3, 2, 1, 2)
        vl.addWidget(gg)

        # Axes
        ag = QGroupBox("Axes")
        al = QGridLayout(ag)
        self._mw._chk_logx = QCheckBox("Log X"); self._mw._chk_logx.toggled.connect(self._mw._update_plot); al.addWidget(self._mw._chk_logx, 0, 0)
        self._mw._chk_logy = QCheckBox("Log Y"); self._mw._chk_logy.toggled.connect(self._mw._update_plot); al.addWidget(self._mw._chk_logy, 0, 1)
        self._mw._chk_legend = QCheckBox("Legend"); self._mw._chk_legend.setChecked(True)
        self._mw._chk_legend.toggled.connect(self._mw._update_plot); al.addWidget(self._mw._chk_legend, 0, 2)
        al.addWidget(QLabel("Title:"), 1, 0)
        self._mw._plot_title = QLineEdit(); self._mw._plot_title.setPlaceholderText("Auto")
        self._mw._plot_title.editingFinished.connect(self._mw._update_plot); al.addWidget(self._mw._plot_title, 1, 1, 1, 2)
        for row, (lbl, attr) in enumerate([("X min:", "_xlim_min"), ("X max:", "_xlim_max"),
                                            ("Y min:", "_ylim_min"), ("Y max:", "_ylim_max")], start=2):
            al.addWidget(QLabel(lbl), row, 0)
            le = QLineEdit(); le.setPlaceholderText("auto"); le.editingFinished.connect(self._mw._update_plot)
            setattr(self._mw, attr, le); al.addWidget(le, row, 1, 1, 2)
        vl.addWidget(ag)

        vl.addStretch()
        scroll.setWidget(inner); scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

    def _sync_color_buttons(self):
        """Update button colors from main window state."""
        self._mw._btn_fig_bg.setStyleSheet(f"background: {self._mw._fig_bg_color}; border: 1px solid #888;")
        self._mw._btn_axes_bg.setStyleSheet(f"background: {self._mw._axes_bg_color}; border: 1px solid #888;")
        self._mw._btn_text_color.setStyleSheet(f"background: {self._mw._text_color}; border: 1px solid #888;")


class PeakFitDialog(StatusMixin, QDialog):
    """Dialog for fitting peaks across a range of scans (rocking curves)."""
    def __init__(self, main_win, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Peak Fitting — Rocking Curve Integration")
        self.setMinimumSize(900, 650)
        self._mw = main_win
        self._fit_results = {}  # scan_num -> {popt, area, model_name, ...}
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # --- Left: controls ---
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)

        # Scan range
        sg = QGroupBox("Scan range")
        sl = QGridLayout(sg)
        sl.addWidget(QLabel("From:"), 0, 0)
        self._scan_from = QSpinBox(); self._scan_from.setRange(1, 99999)
        self._scan_from.editingFinished.connect(self._on_range_changed)
        sl.addWidget(self._scan_from, 0, 1)
        sl.addWidget(QLabel("To:"), 0, 2)
        self._scan_to = QSpinBox(); self._scan_to.setRange(1, 99999)
        self._scan_to.editingFinished.connect(self._on_range_changed)
        sl.addWidget(self._scan_to, 0, 3)
        lv.addWidget(sg)

        # Column selection
        cg = QGroupBox("Data columns")
        cl = QGridLayout(cg)
        cl.addWidget(QLabel("X (motor):"), 0, 0)
        self._x_col = QComboBox(); cl.addWidget(self._x_col, 0, 1)
        cl.addWidget(QLabel("Y (intensity):"), 1, 0)
        self._y_col = QComboBox(); cl.addWidget(self._y_col, 1, 1)
        cl.addWidget(QLabel("Normalize by:"), 2, 0)
        self._norm_col = QComboBox(); self._norm_col.addItem("(none)")
        cl.addWidget(self._norm_col, 2, 1)
        # Transmittance correction (per image)
        self._chk_pf_transm = QCheckBox("Also correct by transm")
        self._chk_pf_transm.setToolTip(
            "Divide Y by the per-image transmittance value from spec.\n"
            "Applied IN ADDITION to the normalize-by dropdown above.\n"
            "For rocking scans with auto-attenuators this varies per image.")
        cl.addWidget(self._chk_pf_transm, 3, 0, 1, 2)
        cl.addWidget(QLabel("transm column:"), 4, 0)
        self._pf_transm_col = QLineEdit("transm")
        cl.addWidget(self._pf_transm_col, 4, 1)
        # Wire to update preview when toggled
        self._chk_pf_transm.toggled.connect(self._update_preview)
        self._pf_transm_col.editingFinished.connect(self._update_preview)
        lv.addWidget(cg)

        # Fit model
        fg = QGroupBox("Fit model")
        fl = QVBoxLayout(fg)
        self._model_combo = QComboBox()
        self._model_combo.addItems(list(FIT_MODELS.keys()))
        fl.addWidget(self._model_combo)
        # Fit window
        fw_row = QHBoxLayout()
        self._chk_fit_window = QCheckBox("Fit window ±")
        self._chk_fit_window.setToolTip("Restrict fit to center ± this half-width.\nRequires a center guide to be set.")
        fw_row.addWidget(self._chk_fit_window)
        self._fit_window_val = QDoubleSpinBox()
        self._fit_window_val.setRange(0.001, 9999); self._fit_window_val.setValue(1.0)
        self._fit_window_val.setDecimals(3); self._fit_window_val.setSingleStep(0.1)
        fw_row.addWidget(self._fit_window_val)
        fl.addLayout(fw_row)
        lv.addWidget(fg)

        # Actions
        btn_fit_one = QPushButton("Fit current scan")
        btn_fit_one.clicked.connect(self._fit_current)
        lv.addWidget(btn_fit_one)
        btn_fit_all = QPushButton("Fit all scans in range")
        btn_fit_all.setStyleSheet(PRIMARY_STYLE)
        btn_fit_all.clicked.connect(self._fit_range)
        lv.addWidget(btn_fit_all)

        # Browse scans
        bg = QGroupBox("Preview scan")
        bl = QHBoxLayout(bg)
        btn_prev = QPushButton("◀"); btn_prev.setFixedWidth(40)
        btn_prev.clicked.connect(self._prev_scan); bl.addWidget(btn_prev)
        self._preview_label = QLabel("—")
        self._preview_label.setAlignment(Qt.AlignCenter)
        bl.addWidget(self._preview_label)
        btn_next = QPushButton("▶"); btn_next.setFixedWidth(40)
        btn_next.clicked.connect(self._next_scan); bl.addWidget(btn_next)
        lv.addWidget(bg)

        # Fit info
        self._fit_info = QLabel("No fit yet")
        self._fit_info.setWordWrap(True)
        self._fit_info.setStyleSheet(HINT_STYLE)
        lv.addWidget(self._fit_info)

        # Display options
        self._chk_show_fit = QCheckBox("Overlay fit on data"); self._chk_show_fit.setChecked(True)
        self._chk_show_fit.toggled.connect(self._update_preview)
        lv.addWidget(self._chk_show_fit)
        self._chk_show_bg = QCheckBox("Show background"); self._chk_show_bg.setChecked(True)
        self._chk_show_bg.toggled.connect(self._update_preview)
        lv.addWidget(self._chk_show_bg)

        # Center hint
        hint_row = QHBoxLayout()
        self._chk_click_guide = QCheckBox("Guide center:")
        self._chk_click_guide.setToolTip("Click near the peak you want to fit,\n"
                                          "or type a value. Snaps to nearest data point.")
        hint_row.addWidget(self._chk_click_guide)
        self._hint_edit = QLineEdit()
        self._hint_edit.setPlaceholderText("click or type")
        self._hint_edit.setFixedWidth(90)
        self._hint_edit.editingFinished.connect(self._on_hint_edited)
        hint_row.addWidget(self._hint_edit)
        btn_clear_hint = QPushButton("✕")
        btn_clear_hint.setFixedWidth(24)
        btn_clear_hint.clicked.connect(self._clear_hint)
        hint_row.addWidget(btn_clear_hint)
        lv.addLayout(hint_row)

        # Apply guide from current fit center to remaining scans
        btn_apply_guide = QPushButton("Apply fitted center to remaining scans")
        btn_apply_guide.setToolTip("Uses the fitted center from the current scan\n"
                                    "as the guide for all subsequent scans in range.\n"
                                    "Each scan updates the guide from its own fit result.")
        btn_apply_guide.clicked.connect(self._apply_guide_forward)
        lv.addWidget(btn_apply_guide)

        # Numerical area comparison
        self._chk_use_numerical = QCheckBox("Use numerical area (model-free)")
        self._chk_use_numerical.setToolTip("Use trapezoidal integration instead of fit area.\n"
                                            "Subtracts a linear background from edges.")
        lv.addWidget(self._chk_use_numerical)

        # --- Parameter editor ---
        pg = QGroupBox("Edit parameters")
        p_layout = QGridLayout(pg)
        self._param_spins = {}
        param_labels = [('amp', 'Amplitude'), ('cen', 'Center'), ('sigma', 'Width (σ)'),
                        ('bg_const', 'BG offset'), ('bg_slope', 'BG slope')]
        for row, (key, label) in enumerate(param_labels):
            p_layout.addWidget(QLabel(label + ":"), row, 0)
            sb = QDoubleSpinBox()
            sb.setRange(-1e12, 1e12); sb.setDecimals(7); sb.setSingleStep(0.001)
            sb.setSpecialValueText("—")
            sb.editingFinished.connect(self._on_param_edited)
            p_layout.addWidget(sb, row, 1)
            self._param_spins[key] = sb
        btn_refit = QPushButton("Refit from edited params")
        btn_refit.clicked.connect(self._refit_from_params)
        p_layout.addWidget(btn_refit, len(param_labels), 0, 1, 2)
        self._chk_drag_handles = QCheckBox("Drag handles on plot")
        self._chk_drag_handles.setChecked(True)
        self._chk_drag_handles.toggled.connect(self._update_preview)
        p_layout.addWidget(self._chk_drag_handles, len(param_labels)+1, 0, 1, 2)
        lv.addWidget(pg)

        # Plot integrated area
        btn_plot_area = QPushButton("Plot integrated area vs scan #")
        btn_plot_area.clicked.connect(self._plot_integrated)
        lv.addWidget(btn_plot_area)

        # Export / Import
        exp_row = QHBoxLayout()
        btn_export = QPushButton("Export results")
        btn_export.clicked.connect(self._export_results)
        exp_row.addWidget(btn_export)
        btn_import = QPushButton("Import results")
        btn_import.clicked.connect(self._import_results)
        exp_row.addWidget(btn_import)
        lv.addLayout(exp_row)

        lv.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(320)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(left_scroll)

        # --- Right: plot ---
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(0, 0, 0, 0)
        self._fig = Figure(figsize=(6, 4), dpi=100, facecolor='white')
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor('white')
        # Small nav toolbar for Home/Save buttons
        self._nav_toolbar = NavigationToolbar(self._canvas, right_w)
        right_l.addWidget(self._nav_toolbar)
        right_l.addWidget(self._canvas, stretch=1)
        layout.addWidget(right_w, stretch=1)

        self._current_preview_scan = None
        self._center_hint = None

        # Drag handle state
        self._drag_active = False
        self._drag_handle_name = None
        self._handle_artists = {}  # name -> scatter artist

        # Pan state (right-click drag)
        self._pan_active = False
        self._pan_start = None  # (x, y) in data coords

        # Connect matplotlib events
        self._canvas.mpl_connect('button_press_event', self._on_canvas_press)
        self._canvas.mpl_connect('button_release_event', self._on_canvas_release)
        self._canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
        self._canvas.mpl_connect('scroll_event', self._on_canvas_scroll)

    def showEvent(self, event):
        super().showEvent(event)
        if self._mw._spec_scans:
            sn_list = sorted(self._mw._spec_scans.keys())
            self._scan_from.setValue(sn_list[0])
            self._scan_to.setValue(sn_list[-1])
            if self._current_preview_scan is None:
                self._current_preview_scan = sn_list[0]
            self._populate_columns(self._current_preview_scan)
            self._update_preview()

    def _populate_columns(self, scan_num=None):
        """Populate column dropdowns from the given scan (or union of range)."""
        if not self._mw._spec_scans:
            return
        # Collect all unique columns across the scan range
        s_from = self._scan_from.value()
        s_to = self._scan_to.value()
        all_cols = OrderedDict()  # preserve order from first scan
        for sn in sorted(self._mw._spec_scans.keys()):
            if sn < s_from or sn > s_to:
                continue
            for c in self._mw._spec_scans[sn].columns:
                if c not in all_cols:
                    all_cols[c] = True
        if not all_cols:
            # Fallback to first scan
            first = next(iter(self._mw._spec_scans.values()))
            for c in first.columns:
                all_cols[c] = True

        # Remember current selections
        prev_x = self._x_col.currentText()
        prev_y = self._y_col.currentText()
        prev_n = self._norm_col.currentText()

        self._x_col.blockSignals(True); self._y_col.blockSignals(True); self._norm_col.blockSignals(True)
        self._x_col.clear(); self._y_col.clear()
        self._norm_col.clear(); self._norm_col.addItem("(none)")
        for c in all_cols:
            self._x_col.addItem(c); self._y_col.addItem(c); self._norm_col.addItem(c)

        # Add bgsub columns for scans in range
        bgsub_cols = OrderedDict()
        for sn in sorted(self._mw._bgsub_data.keys()):
            if sn < s_from or sn > s_to:
                continue
            for c in self._mw._bgsub_data[sn]:
                if c not in ('scan', 'image') and c not in bgsub_cols:
                    bgsub_cols[c] = True
        for c in bgsub_cols:
            tag = f"[bg] {c}"
            self._y_col.addItem(tag)
            self._norm_col.addItem(tag)

        # Try to set X to the scanned motor of the current preview scan
        if scan_num and scan_num in self._mw._spec_scans:
            scan = self._mw._spec_scans[scan_num]
            # First column is typically the scanned motor
            motor_col = scan.columns[0] if scan.columns else None
            if motor_col:
                idx = self._x_col.findText(motor_col)
                if idx >= 0:
                    self._x_col.setCurrentIndex(idx)

        # Restore previous selections if they still exist
        if prev_x:
            idx = self._x_col.findText(prev_x)
            if idx >= 0: self._x_col.setCurrentIndex(idx)
        if prev_y:
            idx = self._y_col.findText(prev_y)
            if idx >= 0: self._y_col.setCurrentIndex(idx)
        if prev_n:
            idx = self._norm_col.findText(prev_n)
            if idx >= 0: self._norm_col.setCurrentIndex(idx)

        self._x_col.blockSignals(False); self._y_col.blockSignals(False); self._norm_col.blockSignals(False)

    def _on_range_changed(self):
        """Jump preview to the first available scan in the new range."""
        s_from = self._scan_from.value()
        s_to = self._scan_to.value()
        # Find first scan in range
        for sn in sorted(self._mw._spec_scans.keys()):
            if s_from <= sn <= s_to:
                self._current_preview_scan = sn
                break
        self._populate_columns(self._current_preview_scan)
        self._update_preview()

    def _get_scan_data(self, scan_num):
        scan = self._mw._spec_scans.get(scan_num)
        if scan is None:
            return None, None
        bgsub = self._mw._bgsub_data.get(scan_num, {})

        def resolve(col_name):
            if col_name.startswith('[bg] '):
                return bgsub.get(col_name[5:])
            return scan.data.get(col_name)

        xc = self._x_col.currentText()
        yc = self._y_col.currentText()
        nc = self._norm_col.currentText()
        x = resolve(xc)
        y = resolve(yc)
        if x is None or y is None:
            return None, None

        # Get transm column if enabled (per-image)
        transm_arr = None
        if self._chk_pf_transm.isChecked():
            tcol = self._pf_transm_col.text().strip()
            if tcol:
                transm_arr = scan.data.get(tcol)

        def apply_t(y_arr, idx_arr):
            """Divide y_arr by transm at given indices (or sequential if None)."""
            if transm_arr is None:
                return y_arr
            if idx_arr is not None:
                valid_mask = idx_arr < len(transm_arr)
                t_vals = np.ones(len(idx_arr))
                if valid_mask.any():
                    safe_idx = idx_arr[valid_mask]
                    t_vals[valid_mask] = transm_arr[safe_idx]
            else:
                n = min(len(y_arr), len(transm_arr))
                t_vals = np.ones(len(y_arr))
                t_vals[:n] = transm_arr[:n]
            bad = (t_vals <= 0) | ~np.isfinite(t_vals)
            t_vals = np.where(bad, 1.0, t_vals)
            return y_arr / t_vals

        x_is_bg = xc.startswith('[bg] ')
        y_is_bg = yc.startswith('[bg] ')

        if not x_is_bg and y_is_bg and 'image' in bgsub:
            # X from spec, Y from bgsub: use image indices
            img_idx = bgsub['image'].astype(int)
            valid = img_idx < len(x)
            img_idx = img_idx[valid]
            x = x[img_idx]; y = y[valid].copy()
            if nc and nc != "(none)":
                nd = resolve(nc)
                if nd is not None:
                    if nc.startswith('[bg] '):
                        nd = nd[valid].copy()
                    else:
                        nd = nd[img_idx].copy()
                    nd[nd == 0] = 1; y = y / nd
            y = apply_t(y, img_idx)
        else:
            n = min(len(x), len(y))
            x = x[:n]; y = y[:n].copy()
            if nc and nc != "(none)":
                nd = resolve(nc)
                if nd is not None:
                    nd = nd[:n].copy(); nd[nd == 0] = 1; y = y / nd
            y = apply_t(y, None)
        return x, y

    def _on_canvas_press(self, event):
        """Handle click: left=drag handle or set center hint, right=start pan."""
        if event.inaxes != self._ax:
            return

        # Right-click: start panning
        if event.button == 3:
            self._pan_active = True
            self._pan_start = (event.xdata, event.ydata)
            return

        if event.button != 1:
            return

        # Check if clicking near a drag handle
        if self._chk_drag_handles.isChecked() and self._handle_artists:
            for name, artist in self._handle_artists.items():
                if artist is None:
                    continue
                offsets = artist.get_offsets()
                if len(offsets) == 0:
                    continue
                hx, hy = offsets[0]
                # Check proximity in display coords
                disp_h = self._ax.transData.transform((hx, hy))
                disp_click = self._ax.transData.transform((event.xdata, event.ydata))
                dist = np.sqrt((disp_h[0]-disp_click[0])**2 + (disp_h[1]-disp_click[1])**2)
                if dist < 15:  # 15 pixels
                    self._drag_active = True
                    self._drag_handle_name = name
                    return

        # Not near a handle: set center hint
        if self._chk_click_guide.isChecked():
            click_x = event.xdata
            sn = self._current_preview_scan
            if sn is not None:
                x, y = self._get_scan_data(sn)
                if x is not None and len(x) > 0:
                    idx = np.argmin(np.abs(x - click_x))
                    click_x = float(x[idx])
            self._center_hint = click_x
            self._hint_edit.setText(f"{click_x:.5g}")
            self._chk_click_guide.setChecked(True)
            self._update_preview()

    def _on_canvas_release(self, event):
        if self._drag_active:
            self._drag_active = False
            self._drag_handle_name = None
        if self._pan_active:
            self._pan_active = False
            self._pan_start = None

    def _on_canvas_motion(self, event):
        # Right-click pan
        if self._pan_active and event.inaxes == self._ax and self._pan_start is not None:
            dx = self._pan_start[0] - event.xdata
            dy = self._pan_start[1] - event.ydata
            xlim = self._ax.get_xlim()
            ylim = self._ax.get_ylim()
            self._ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
            self._ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
            self._canvas.draw_idle()
            return

        # Left-click drag handles
        if not self._drag_active or event.inaxes != self._ax:
            return
        sn = self._current_preview_scan
        if sn is None or sn not in self._fit_results:
            return
        fr = self._fit_results[sn]
        popt = fr.get('popt')
        if popt is None:
            return

        model = FIT_MODELS[fr['model_name']]
        params = model['params']
        name = self._drag_handle_name

        # Map drag to parameter changes
        if name == 'peak_top':
            # Drag vertically → amplitude
            popt[0] = max(0, event.ydata - (popt[-2] + popt[-1] * popt[1] if len(popt) >= 5 else 0))
        elif name == 'peak_center':
            # Drag horizontally → center
            popt[1] = event.xdata
        elif name == 'sigma_left':
            # Distance from center → sigma
            popt[2] = max(0.001, abs(popt[1] - event.xdata))
        elif name == 'sigma_right':
            popt[2] = max(0.001, abs(event.xdata - popt[1]))
        elif name == 'bg_left':
            # Drag background left endpoint vertically
            if len(popt) >= 5:
                x_data, _ = self._get_scan_data(sn)
                if x_data is not None:
                    x_left = x_data.min()
                    # bg at x_left = bg_const + bg_slope * x_left = event.ydata
                    # bg at x_right stays fixed
                    x_right = x_data.max()
                    bg_right = popt[-2] + popt[-1] * x_right
                    if abs(x_right - x_left) > 0:
                        popt[-1] = (bg_right - event.ydata) / (x_right - x_left)
                        popt[-2] = event.ydata - popt[-1] * x_left
        elif name == 'bg_right':
            if len(popt) >= 5:
                x_data, _ = self._get_scan_data(sn)
                if x_data is not None:
                    x_left = x_data.min()
                    x_right = x_data.max()
                    bg_left = popt[-2] + popt[-1] * x_left
                    if abs(x_right - x_left) > 0:
                        popt[-1] = (event.ydata - bg_left) / (x_right - x_left)
                        popt[-2] = bg_left - popt[-1] * x_left

        # Update fit result and redraw
        fr['popt'] = popt
        x_data, y_data = self._get_scan_data(sn)
        if x_data is not None:
            fr['fit_area'] = integrate_peak(x_data, y_data, popt, model)
            fr['area'] = fr['fit_area']
        self._sync_params_to_spins(popt, params)
        self._update_preview()

    def _on_hint_edited(self):
        """Parse the text box value as center hint."""
        text = self._hint_edit.text().strip()
        if not text:
            self._center_hint = None
            return
        try:
            self._center_hint = float(text)
            self._chk_click_guide.setChecked(True)
            self._update_preview()
        except ValueError:
            pass

    def _clear_hint(self):
        self._center_hint = None
        self._hint_edit.clear()
        self._chk_click_guide.setChecked(False)
        self._update_preview()

    def _on_canvas_scroll(self, event):
        """Scroll wheel zoom centered at cursor position."""
        if event.inaxes != self._ax:
            return
        factor = 0.8 if event.button == 'up' else 1.25
        xdata, ydata = event.xdata, event.ydata
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        # Zoom both axes toward the cursor
        new_xlim = [xdata + (x - xdata) * factor for x in xlim]
        new_ylim = [ydata + (y - ydata) * factor for y in ylim]
        self._ax.set_xlim(new_xlim)
        self._ax.set_ylim(new_ylim)
        self._canvas.draw_idle()

    def _sync_params_to_spins(self, popt, param_names):
        """Update spinboxes from fit parameters."""
        mapping = {'amp': 0, 'cen': 1, 'sigma': 2, 'gamma': 2, 'eta': 3,
                   'bg_const': -2, 'bg_slope': -1}
        for key, sb in self._param_spins.items():
            sb.blockSignals(True)
            if key in param_names:
                idx = param_names.index(key)
                sb.setValue(popt[idx])
                sb.setEnabled(True)
            elif key == 'sigma' and 'gamma' in param_names:
                idx = param_names.index('gamma')
                sb.setValue(popt[idx])
                sb.setEnabled(True)
            else:
                sb.setValue(0)
                sb.setEnabled(False)
            sb.blockSignals(False)

    def _on_param_edited(self):
        """Update fit result from spinbox values."""
        sn = self._current_preview_scan
        if sn is None or sn not in self._fit_results:
            return
        fr = self._fit_results[sn]
        popt = fr.get('popt')
        if popt is None:
            return
        model = FIT_MODELS[fr['model_name']]
        params = model['params']
        for key, sb in self._param_spins.items():
            if not sb.isEnabled():
                continue
            if key in params:
                idx = params.index(key)
                popt[idx] = sb.value()
            elif key == 'sigma' and 'gamma' in params:
                idx = params.index('gamma')
                popt[idx] = sb.value()
        # Recalculate area
        x_data, y_data = self._get_scan_data(sn)
        if x_data is not None:
            fr['fit_area'] = integrate_peak(x_data, y_data, popt, model)
            if not self._chk_use_numerical.isChecked():
                fr['area'] = fr['fit_area']
        self._update_preview()

    def _refit_from_params(self):
        """Re-run curve_fit using current spinbox values as initial guess."""
        sn = self._current_preview_scan
        if sn is None:
            return
        x, y = self._get_scan_data(sn)
        if x is None:
            return
        fr = self._fit_results.get(sn)
        if fr is None or fr.get('popt') is None:
            self.say("No existing fit to refit from", error=True); return

        model_name = fr['model_name']
        model = FIT_MODELS[model_name]
        p0 = list(fr['popt'])
        window = self._get_fit_window()
        hint = p0[1]  # use current center

        x_fit, y_fit = x, y
        if window is not None and window > 0:
            mask = (x >= hint - window) & (x <= hint + window)
            if mask.sum() >= 4:
                x_fit = x[mask]; y_fit = y[mask]

        try:
            bounds = model['bounds'](x_fit, y_fit)
            popt, pcov = curve_fit(model['func'], x_fit, y_fit, p0=p0, bounds=bounds, maxfev=5000)
            fit_area = integrate_peak(x_fit, y_fit, popt, model)
            num_area = numerical_area(x, y, center=hint, window=window)
            area = num_area if self._chk_use_numerical.isChecked() else fit_area
            self._fit_results[sn] = {'popt': popt, 'pcov': pcov, 'model_name': model_name,
                                      'area': area, 'num_area': num_area, 'fit_area': fit_area}
            self._sync_params_to_spins(popt, model['params'])
            param_str = ", ".join(f"{n}={v:.7g}" for n, v in zip(model['params'], popt))
            self._fit_info.setText(f"Scan {sn} (refit): area={area:.7g}\n{param_str}")
        except Exception as e:
            self._fit_info.setText(f"Refit failed: {e}")
        self._update_preview()

    def _draw_handles(self, popt, model_name, x_data):
        """Draw draggable handle markers on the fit curve."""
        self._handle_artists = {}
        if popt is None:
            return

        model = FIT_MODELS[model_name]
        params = model['params']
        cen = popt[1]
        sigma = popt[2]
        amp = popt[0]

        # Background at center
        bg_at_cen = 0
        if len(popt) >= 5:
            bg_at_cen = popt[-2] + popt[-1] * cen

        handle_props = {
            'peak_top': (cen, bg_at_cen + amp, '#d62728', 'v', 12),           # ▼ at peak top
            'peak_center': (cen, bg_at_cen, '#2ca02c', 'D', 10),               # ◆ at base center
            'sigma_left': (cen - sigma, model['func'](np.array([cen - sigma]), *popt)[0],
                           '#ff7f0e', '<', 10),                                  # ◀ at -σ
            'sigma_right': (cen + sigma, model['func'](np.array([cen + sigma]), *popt)[0],
                            '#ff7f0e', '>', 10),                                 # ▶ at +σ
        }

        # Background line endpoints
        if len(popt) >= 5 and x_data is not None and len(x_data) > 0:
            x_lo, x_hi = x_data.min(), x_data.max()
            handle_props['bg_left'] = (x_lo, popt[-2] + popt[-1] * x_lo, '#8c564b', 's', 9)
            handle_props['bg_right'] = (x_hi, popt[-2] + popt[-1] * x_hi, '#8c564b', 's', 9)

        for name, (hx, hy, color, marker, size) in handle_props.items():
            artist = self._ax.scatter([hx], [hy], color=color, marker=marker,
                                       s=size**2, zorder=10, edgecolors='white', linewidths=0.5)
            self._handle_artists[name] = artist

    def _get_fit_window(self):
        if self._chk_fit_window.isChecked():
            return self._fit_window_val.value()
        return None

    def _fit_current(self):
        sn = self._current_preview_scan
        if sn is None:
            return
        x, y = self._get_scan_data(sn)
        if x is None:
            self.say("No data for this scan", error=True); return
        model_name = self._model_combo.currentText()
        hint = self._center_hint if self._chk_click_guide.isChecked() else None
        window = self._get_fit_window()
        result = fit_peak(x, y, model_name, center_hint=hint, fit_window=window)

        # Numerical area for comparison
        num_area = numerical_area(x, y, center=hint, window=window)

        if result is None:
            # Fit failed. Create a manual starting popt so the user can still
            # use drag handles and spinboxes to manually position the curve.
            model = FIT_MODELS[model_name]
            n_params = len(model['params'])
            # Build a reasonable starting guess
            y_range = float(y.max() - y.min()) if len(y) > 0 else 1.0
            x_center = float(hint) if hint is not None else float(x.mean())
            x_span = float(x.max() - x.min()) if len(x) > 1 else 1.0
            manual_popt = np.zeros(n_params)
            for i, pname in enumerate(model['params']):
                if pname == 'amp':
                    manual_popt[i] = y_range * 0.5
                elif pname == 'cen':
                    manual_popt[i] = x_center
                elif pname in ('sigma', 'gamma'):
                    manual_popt[i] = x_span * 0.02
                elif pname == 'eta':
                    manual_popt[i] = 0.5
                elif pname == 'bg_const':
                    manual_popt[i] = float(np.median(y)) if len(y) > 0 else 0.0
                elif pname == 'bg_slope':
                    manual_popt[i] = 0.0
            self._fit_results[sn] = {
                'popt': manual_popt, 'pcov': None, 'model_name': model_name,
                'area': num_area if self._chk_use_numerical.isChecked() else 0,
                'num_area': num_area, 'fit_area': 0}
            self._fit_info.setText(
                f"<span style='color:#ff4444;font-weight:bold;'>"
                f"Scan {sn}: fit FAILED</span> — use drag handles or "
                f"spinboxes to adjust manually (numerical area={num_area:.7g})")
            self._update_preview(); return

        popt, pcov, model, x_used, y_used = result
        fit_area = integrate_peak(x_used, y_used, popt, model)
        area = num_area if self._chk_use_numerical.isChecked() else fit_area

        # Area comparison
        ratio = fit_area / num_area if num_area > 0 else float('inf')
        flag = ""
        if ratio < 0.5 or ratio > 2.0:
            flag = " ⚠ fit/num mismatch!"

        self._fit_results[sn] = {'popt': popt, 'pcov': pcov, 'model_name': model_name,
                                  'area': area, 'num_area': num_area, 'fit_area': fit_area}
        param_str = ", ".join(f"{n}={v:.7g}" for n, v in zip(model['params'], popt))
        self._fit_info.setText(
            f"Scan {sn}: area={area:.7g}{flag}\n"
            f"fit={fit_area:.4g}, numerical={num_area:.4g} (ratio={ratio:.2f})\n"
            f"{param_str}")
        self._update_preview()

    def _fit_range(self):
        s_from = self._scan_from.value()
        s_to = self._scan_to.value()
        model_name = self._model_combo.currentText()
        hint = self._center_hint if self._chk_click_guide.isChecked() else None
        window = self._get_fit_window()
        count = 0; failed = 0; flagged = 0
        for sn in sorted(self._mw._spec_scans.keys()):
            if sn < s_from or sn > s_to: continue
            x, y = self._get_scan_data(sn)
            if x is None: failed += 1; continue

            num_area = numerical_area(x, y, center=hint, window=window)
            result = fit_peak(x, y, model_name, center_hint=hint, fit_window=window)

            if result is None:
                # Create a manual popt so user can edit later
                model = FIT_MODELS[model_name]
                n_params = len(model['params'])
                y_range = float(y.max() - y.min()) if len(y) > 0 else 1.0
                x_center = float(hint) if hint is not None else float(x.mean())
                x_span = float(x.max() - x.min()) if len(x) > 1 else 1.0
                manual_popt = np.zeros(n_params)
                for pi, pname in enumerate(model['params']):
                    if pname == 'amp': manual_popt[pi] = y_range * 0.5
                    elif pname == 'cen': manual_popt[pi] = x_center
                    elif pname in ('sigma', 'gamma'): manual_popt[pi] = x_span * 0.02
                    elif pname == 'eta': manual_popt[pi] = 0.5
                    elif pname == 'bg_const': manual_popt[pi] = float(np.median(y)) if len(y) > 0 else 0
                    elif pname == 'bg_slope': manual_popt[pi] = 0
                self._fit_results[sn] = {
                    'popt': manual_popt, 'pcov': None, 'model_name': model_name,
                    'area': num_area if self._chk_use_numerical.isChecked() else 0,
                    'num_area': num_area, 'fit_area': 0}
                failed += 1
                count += 1  # still counts as processed (editable)
                continue

            popt, pcov, model, x_used, y_used = result
            fit_area = integrate_peak(x_used, y_used, popt, model)
            area = num_area if self._chk_use_numerical.isChecked() else fit_area

            ratio = fit_area / num_area if num_area > 0 else float('inf')
            if ratio < 0.5 or ratio > 2.0:
                flagged += 1

            self._fit_results[sn] = {'popt': popt, 'pcov': pcov, 'model_name': model_name,
                                      'area': area, 'num_area': num_area, 'fit_area': fit_area}
            count += 1

        msg = f"Fitted {count} scans, {failed} failed"
        if flagged > 0:
            msg += f", {flagged} flagged (fit/numerical mismatch)"
        self._fit_info.setText(msg)
        self._update_preview()

    def _apply_guide_forward(self):
        """Fit remaining scans in range, propagating the center from each fit to the next."""
        sn = self._current_preview_scan
        if sn is None: return
        s_to = self._scan_to.value()
        s_from = self._scan_from.value()
        model_name = self._model_combo.currentText()
        window = self._get_fit_window()

        # Start with current hint or current fit center
        guide = self._center_hint
        if guide is None and sn in self._fit_results:
            popt = self._fit_results[sn].get('popt')
            if popt is not None:
                guide = popt[1]  # fitted center
        if guide is None:
            self.say("Set a center guide first (click on plot or type a value)", error=True); return

        sn_list = [s for s in sorted(self._mw._spec_scans.keys()) if s_from <= s <= s_to and s >= sn]
        count = 0; failed = 0
        for scan_num in sn_list:
            x, y = self._get_scan_data(scan_num)
            if x is None: failed += 1; continue

            num_area = numerical_area(x, y, center=guide, window=window)
            result = fit_peak(x, y, model_name, center_hint=guide, fit_window=window)

            if result is None:
                if self._chk_use_numerical.isChecked():
                    self._fit_results[scan_num] = {'popt': None, 'pcov': None, 'model_name': model_name,
                                                    'area': num_area, 'num_area': num_area, 'fit_area': 0}
                    count += 1
                else:
                    failed += 1
                continue

            popt, pcov, model, x_used, y_used = result
            fit_area = integrate_peak(x_used, y_used, popt, model)
            area = num_area if self._chk_use_numerical.isChecked() else fit_area

            self._fit_results[scan_num] = {'popt': popt, 'pcov': pcov, 'model_name': model_name,
                                            'area': area, 'num_area': num_area, 'fit_area': fit_area}
            # Propagate: use this scan's fitted center as guide for next scan
            guide = popt[1]
            count += 1

        self._center_hint = guide
        self._hint_edit.setText(f"{guide:.5g}")
        self._fit_info.setText(f"Forward-guided: {count} fitted, {failed} failed")
        self._update_preview()

    def _prev_scan(self):
        s_from = self._scan_from.value(); s_to = self._scan_to.value()
        sn_list = [sn for sn in sorted(self._mw._spec_scans.keys()) if s_from <= sn <= s_to]
        if not sn_list: return
        if self._current_preview_scan is None:
            self._current_preview_scan = sn_list[0]
        else:
            idx = sn_list.index(self._current_preview_scan) if self._current_preview_scan in sn_list else 0
            self._current_preview_scan = sn_list[max(0, idx - 1)]
        self._populate_columns(self._current_preview_scan)
        self._update_preview()

    def _next_scan(self):
        s_from = self._scan_from.value(); s_to = self._scan_to.value()
        sn_list = [sn for sn in sorted(self._mw._spec_scans.keys()) if s_from <= sn <= s_to]
        if not sn_list: return
        if self._current_preview_scan is None:
            self._current_preview_scan = sn_list[0]
        else:
            idx = sn_list.index(self._current_preview_scan) if self._current_preview_scan in sn_list else 0
            self._current_preview_scan = sn_list[min(len(sn_list)-1, idx + 1)]
        self._populate_columns(self._current_preview_scan)
        self._update_preview()

    def _update_preview(self):
        self._ax.clear()
        self._ax.set_facecolor('white')
        self._ax.tick_params(colors='#222')
        for sp in self._ax.spines.values(): sp.set_color('#999')

        sn = self._current_preview_scan
        if sn is None:
            self._canvas.draw(); return

        self._preview_label.setText(f"Scan {sn}")
        x, y = self._get_scan_data(sn)
        if x is None:
            self._ax.set_title(f"Scan {sn}: no data", color='#888')
            self._canvas.draw(); return

        self._ax.plot(x, y, 'o-', color='#1f77b4', markersize=3, linewidth=1, label='data')

        # Show fit window region
        hint = self._center_hint if self._chk_click_guide.isChecked() else None
        window = self._get_fit_window()
        if hint is not None and window is not None:
            self._ax.axvspan(hint - window, hint + window, alpha=0.08, color='#ff7f0e')

        # Overlay fit if available
        if sn in self._fit_results and self._chk_show_fit.isChecked():
            fr = self._fit_results[sn]
            popt = fr.get('popt')
            if popt is not None:
                model = FIT_MODELS[fr['model_name']]
                x_fit = np.linspace(x.min(), x.max(), 500)
                y_fit = model['func'](x_fit, *popt)
                self._ax.plot(x_fit, y_fit, '-', color='#d62728', linewidth=1.5, label='fit')
                if self._chk_show_bg.isChecked() and len(popt) >= 5:
                    bg_line = popt[-2] + popt[-1] * x_fit
                    self._ax.plot(x_fit, bg_line, '--', color='#ff7f0e', linewidth=1, alpha=0.7, label='background')
                # Draw drag handles
                if self._chk_drag_handles.isChecked():
                    self._draw_handles(popt, fr['model_name'], x)
                # Sync param spinboxes
                self._sync_params_to_spins(popt, model['params'])
            else:
                # Clear spinboxes when no fit params
                for sb in self._param_spins.values():
                    sb.blockSignals(True); sb.setValue(0); sb.setEnabled(False); sb.blockSignals(False)
            # Title with both areas
            fit_a = fr.get('fit_area', fr['area'])
            num_a = fr.get('num_area', 0)
            title = f"Scan {sn} — area={fr['area']:.4g}"
            if num_a > 0 and fit_a > 0:
                ratio = fit_a / num_a
                if ratio < 0.5 or ratio > 2.0:
                    title += f" ⚠"
            self._ax.set_title(title, color='#222')
            self._ax.legend(facecolor='white', edgecolor='#999', labelcolor='#222', fontsize=8)
        else:
            self._ax.set_title(f"Scan {sn}", color='#222')
            self._handle_artists = {}
            for sb in self._param_spins.values():
                sb.blockSignals(True); sb.setValue(0); sb.setEnabled(False); sb.blockSignals(False)

        xc = self._x_col.currentText()
        self._ax.set_xlabel(display_label(xc, self._mw._label_map), color='#222')
        self._ax.set_ylabel("Intensity", color='#222')
        # Show center hint marker
        if self._chk_click_guide.isChecked() and self._center_hint is not None:
            self._ax.axvline(self._center_hint, color='#ff7f0e', linestyle='--', alpha=0.5, linewidth=1)
        self._fig.tight_layout()
        self._canvas.draw()

    def _plot_integrated(self):
        """Send integrated area vs scan number to the main plot as persistent overlay."""
        if not self._fit_results:
            self.say("No fit results", error=True); return
        scans = sorted(self._fit_results.keys())
        areas = [self._fit_results[s]['area'] for s in scans]
        self._mw._overlay = {
            'x': np.array(scans, dtype=float),
            'y': np.array(areas),
            'xlabel': 'Scan number',
            'ylabel': 'Integrated intensity',
            'title': 'Rocking curve integrated area',
            'label': 'Integrated area',
        }
        self._mw._update_plot()

    def _export_results(self):
        if not self._fit_results:
            self.say("No fit results", error=True); return
        path, _ = QFileDialog.getSaveFileName(self, "Export fit results", "peak_fits.txt", "Text (*.txt)")
        if not path: return
        with open(path, 'w') as f:
            f.write("scan\tarea\tfit_area\tnum_area\tmodel\tparams\n")
            for sn in sorted(self._fit_results.keys()):
                fr = self._fit_results[sn]
                fit_a = fr.get('fit_area', fr['area'])
                num_a = fr.get('num_area', 0)
                params_str = ""
                if fr.get('popt') is not None:
                    model = FIT_MODELS.get(fr['model_name'])
                    if model:
                        params_str = ",".join(f"{n}={v:.6g}" for n, v in zip(model['params'], fr['popt']))
                f.write(f"{sn}\t{fr['area']:.6g}\t{fit_a:.6g}\t{num_a:.6g}\t{fr['model_name']}\t{params_str}\n")
        self.say(f"Exported {len(self._fit_results)} results to {path}")

    def _import_results(self):
        """Import previously exported fit results.
        Matches the export format: scan, area, fit_area, num_area, model, params.
        Restores per-scan fit entries so you can keep editing where you left off."""
        path, _ = QFileDialog.getOpenFileName(self, "Import fit results", "", "Text (*.txt);;All (*)")
        if not path:
            return
        try:
            scans = []
            areas = []
            n_with_fit = 0
            header = None
            col_idx = {}
            with open(path, 'r') as f:
                for line in f:
                    line = line.rstrip('\r\n')
                    if not line.strip() or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if header is None:
                        header = [h.strip() for h in parts]
                        col_idx = {h: i for i, h in enumerate(header)}
                        continue
                    if len(parts) < 2:
                        continue
                    try:
                        sn = int(parts[col_idx.get('scan', 0)])
                    except (ValueError, IndexError):
                        continue

                    def get_col(name, default=None, cast=float):
                        i = col_idx.get(name)
                        if i is None or i >= len(parts):
                            return default
                        try:
                            return cast(parts[i])
                        except (ValueError, TypeError):
                            return default

                    area = get_col('area', 0.0)
                    fit_area = get_col('fit_area', area)
                    num_area = get_col('num_area', 0.0)
                    model_name = get_col('model', 'Gaussian + linear bg', cast=str)
                    params_str = get_col('params', '', cast=str)

                    scans.append(sn)
                    areas.append(area)

                    entry = {
                        'popt': None, 'pcov': None,
                        'model_name': model_name,
                        'area': area,
                        'fit_area': fit_area,
                        'num_area': num_area,
                    }

                    # Parse params string into popt array in the right order
                    if params_str:
                        parsed = {}
                        for kv in params_str.split(','):
                            kv = kv.strip()
                            if '=' not in kv:
                                continue
                            k, v = kv.split('=', 1)
                            try:
                                parsed[k.strip()] = float(v.strip())
                            except ValueError:
                                pass
                        # Map to the model's parameter order
                        model = FIT_MODELS.get(model_name)
                        if model:
                            popt = []
                            all_found = True
                            for pname in model['params']:
                                if pname in parsed:
                                    popt.append(parsed[pname])
                                else:
                                    all_found = False
                                    break
                            if all_found and popt:
                                entry['popt'] = np.array(popt)
                                n_with_fit += 1

                    self._fit_results[sn] = entry

            # Refresh UI to show the imported fits
            # If the scan selector has imported scans, jump to the first one
            if scans:
                # Update preview if current selected scan has a restored fit
                try:
                    self._update_preview()
                except Exception:
                    pass

                # Also overlay on the main canvas
                self._mw._overlay = {
                    'x': np.array(scans, dtype=float),
                    'y': np.array(areas),
                    'xlabel': 'Scan number',
                    'ylabel': 'Integrated intensity',
                    'title': f"Imported: {os.path.basename(path)}",
                    'label': 'Integrated area',
                }
                self._mw._update_plot()
                self._fit_info.setText(
                    f"Imported {len(scans)} results ({n_with_fit} with params) "
                    f"from {os.path.basename(path)}")
            else:
                self.say("No valid rows in file", error=True)
        except Exception as e:
            self.say(str(e), error=True)
            return


class SmoothingDialog(StatusMixin, QDialog):
    """Smooth CTR data with exclusion zones for Bragg peaks."""
    def __init__(self, main_win, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CTR Smoothing")
        self.setMinimumSize(950, 650)
        self._mw = main_win
        self._x_data = None
        self._y_data = None
        self._smoothed = None
        self._exclusions = []  # list of (x_min, x_max)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # --- Left: controls ---
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)

        # Data source
        dg = QGroupBox("Data source")
        dl = QVBoxLayout(dg)
        dl.addWidget(QLabel("Load X/Y data from:"))
        btn_from_plot = QPushButton("Current main plot")
        btn_from_plot.clicked.connect(self._load_from_plot)
        dl.addWidget(btn_from_plot)
        btn_from_file = QPushButton("Import from text file")
        btn_from_file.clicked.connect(self._load_from_file)
        dl.addWidget(btn_from_file)
        btn_from_fits = QPushButton("Peak fit results (area vs scan)")
        btn_from_fits.clicked.connect(self._load_from_fits)
        dl.addWidget(btn_from_fits)
        self._data_info = QLabel("No data loaded")
        self._data_info.setStyleSheet(HINT_STYLE)
        dl.addWidget(self._data_info)
        lv.addWidget(dg)

        # Smoothing method
        sg = QGroupBox("Smoothing method")
        sl = QGridLayout(sg)
        sl.addWidget(QLabel("Method:"), 0, 0)
        self._method = QComboBox()
        self._method.addItems(["Savitzky-Golay", "Moving average", "Gaussian kernel",
                                "Lowess", "Spline"])
        self._method.currentTextChanged.connect(self._apply_smoothing)
        sl.addWidget(self._method, 0, 1)
        sl.addWidget(QLabel("Window/pts:"), 1, 0)
        self._window_size = QSpinBox()
        self._window_size.setRange(3, 999); self._window_size.setValue(5)
        self._window_size.setSingleStep(2)
        self._window_size.editingFinished.connect(self._apply_smoothing)
        sl.addWidget(self._window_size, 1, 1)
        sl.addWidget(QLabel("Poly order:"), 2, 0)
        self._poly_order = QSpinBox()
        self._poly_order.setRange(1, 10); self._poly_order.setValue(2)
        self._poly_order.editingFinished.connect(self._apply_smoothing)
        sl.addWidget(self._poly_order, 2, 1)
        lv.addWidget(sg)

        # Exclusion zones
        eg = QGroupBox("Exclusion zones (Bragg peaks)")
        el = QVBoxLayout(eg)
        el.addWidget(QLabel("Click two points on plot to define zone,\nor enter manually:"))
        er = QGridLayout()
        er.addWidget(QLabel("X min:"), 0, 0)
        self._exc_min = QDoubleSpinBox()
        self._exc_min.setRange(-1e6, 1e6); self._exc_min.setDecimals(4)
        er.addWidget(self._exc_min, 0, 1)
        er.addWidget(QLabel("X max:"), 0, 2)
        self._exc_max = QDoubleSpinBox()
        self._exc_max.setRange(-1e6, 1e6); self._exc_max.setDecimals(4)
        er.addWidget(self._exc_max, 0, 3)
        el.addLayout(er)
        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add zone")
        btn_add.clicked.connect(self._add_exclusion)
        btn_row.addWidget(btn_add)
        btn_clear = QPushButton("Clear all")
        btn_clear.clicked.connect(self._clear_exclusions)
        btn_row.addWidget(btn_clear)
        el.addLayout(btn_row)
        self._chk_click_exc = QCheckBox("Click-to-define (2 clicks per zone)")
        el.addWidget(self._chk_click_exc)
        self._exc_list = QListWidget()
        self._exc_list.setMaximumHeight(80)
        el.addWidget(self._exc_list)
        btn_remove = QPushButton("Remove selected zone")
        btn_remove.clicked.connect(self._remove_exclusion)
        el.addWidget(btn_remove)
        lv.addWidget(eg)

        # Actions
        self._chk_log_y = QCheckBox("Log Y")
        self._chk_log_y.toggled.connect(self._update_preview)
        lv.addWidget(self._chk_log_y)

        btn_to_main = QPushButton("Send smoothed to main plot")
        btn_to_main.clicked.connect(self._send_to_main)
        lv.addWidget(btn_to_main)
        btn_export = QPushButton("Export smoothed data")
        btn_export.clicked.connect(self._export)
        lv.addWidget(btn_export)

        lv.addStretch()
        left_scroll = QScrollArea()
        left_scroll.setWidget(left); left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(310); left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(left_scroll)

        # --- Right: plot ---
        self._fig = Figure(figsize=(6, 4), dpi=100, facecolor='white')
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor('white')
        layout.addWidget(self._canvas, stretch=1)

        self._click_points = []  # for click-to-define exclusion
        self._canvas.mpl_connect('button_press_event', self._on_click)

    # ---- Data loading ----
    def _load_from_plot(self):
        """Grab the first plotted line from main canvas."""
        lines = self._mw._canvas.ax.get_lines()
        if not lines:
            self.say("No data on main plot", error=True); return
        line = lines[0]
        self._x_data = np.array(line.get_xdata(), dtype=float)
        self._y_data = np.array(line.get_ydata(), dtype=float)
        self._data_info.setText(f"{len(self._x_data)} points from main plot")
        self._apply_smoothing()

    def _load_from_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load data", "", "Text (*.txt *.dat *.csv);;All (*)")
        if not path: return
        try:
            data = np.loadtxt(path, comments='#')
            if data.ndim == 1 or data.shape[1] < 2:
                self.say("Need at least 2 columns", error=True); return
            self._x_data = data[:, 0]
            self._y_data = data[:, 1]
            self._data_info.setText(f"{len(self._x_data)} pts from {os.path.basename(path)}")
            self._apply_smoothing()
        except Exception as e:
            self.say(str(e), error=True)

    def _load_from_fits(self):
        """Load from peak fit dialog results."""
        fr = self._mw._peak_fit_dialog._fit_results
        if not fr:
            self.say("No peak fit results", error=True); return
        scans = sorted(fr.keys())
        self._x_data = np.array(scans, dtype=float)
        self._y_data = np.array([fr[s]['area'] for s in scans], dtype=float)
        self._data_info.setText(f"{len(scans)} points from peak fits")
        self._apply_smoothing()

    # ---- Exclusion zones ----
    def _add_exclusion(self):
        xmin = self._exc_min.value()
        xmax = self._exc_max.value()
        if xmin >= xmax:
            self.say("X min must be < X max", error=True); return
        self._exclusions.append((xmin, xmax))
        self._exc_list.addItem(f"{xmin:.4g} — {xmax:.4g}")
        self._apply_smoothing()

    def _remove_exclusion(self):
        idx = self._exc_list.currentRow()
        if 0 <= idx < len(self._exclusions):
            self._exclusions.pop(idx)
            self._exc_list.takeItem(idx)
            self._apply_smoothing()

    def _clear_exclusions(self):
        self._exclusions.clear()
        self._exc_list.clear()
        self._click_points.clear()
        self._apply_smoothing()

    def _on_click(self, event):
        if not self._chk_click_exc.isChecked():
            return
        if event.inaxes != self._ax or event.button != 1:
            return
        self._click_points.append(event.xdata)
        if len(self._click_points) == 2:
            xmin = min(self._click_points)
            xmax = max(self._click_points)
            self._exclusions.append((xmin, xmax))
            self._exc_list.addItem(f"{xmin:.4g} — {xmax:.4g}")
            self._exc_min.setValue(xmin); self._exc_max.setValue(xmax)
            self._click_points.clear()
            self._apply_smoothing()
        else:
            # Draw temporary marker
            self._ax.axvline(event.xdata, color='#d62728', linestyle=':', alpha=0.7)
            self._canvas.draw()

    # ---- Smoothing ----
    def _get_mask(self):
        """Boolean mask: True = include, False = excluded."""
        if self._x_data is None:
            return None
        mask = np.ones(len(self._x_data), dtype=bool)
        for xmin, xmax in self._exclusions:
            mask &= ~((self._x_data >= xmin) & (self._x_data <= xmax))
        return mask

    def _apply_smoothing(self):
        if self._x_data is None or self._y_data is None:
            return
        mask = self._get_mask()
        method = self._method.currentText()
        win = self._window_size.value()
        poly = self._poly_order.value()

        # Ensure window is odd for savgol
        if win % 2 == 0:
            win += 1

        # Work on included segments separately
        x_inc = self._x_data[mask]
        y_inc = self._y_data[mask]

        if len(x_inc) < 4:
            self._smoothed = self._y_data.copy()
            self._update_preview(); return

        # Find contiguous segments
        segments = []
        seg_start = 0
        inc_indices = np.where(mask)[0]
        for i in range(1, len(inc_indices)):
            if inc_indices[i] - inc_indices[i-1] > 1:
                segments.append(inc_indices[seg_start:i])
                seg_start = i
        segments.append(inc_indices[seg_start:])

        self._smoothed = self._y_data.copy()
        self._smoothed[~mask] = np.nan  # excluded regions

        for seg_idx in segments:
            if len(seg_idx) < 3:
                continue
            x_seg = self._x_data[seg_idx]
            y_seg = self._y_data[seg_idx]
            y_smooth = self._smooth_segment(y_seg, x_seg, method, win, poly)
            self._smoothed[seg_idx] = y_smooth

        self._update_preview()

    def _smooth_segment(self, y, x, method, window, poly_order):
        """Apply smoothing to a single contiguous segment."""
        n = len(y)
        win = min(window, n)
        if win % 2 == 0: win = max(win - 1, 3)
        if win < 3: return y.copy()

        if method == "Savitzky-Golay":
            from scipy.signal import savgol_filter
            order = min(poly_order, win - 1)
            return savgol_filter(y, win, order)
        elif method == "Moving average":
            kernel = np.ones(win) / win
            padded = np.pad(y, win//2, mode='edge')
            return np.convolve(padded, kernel, mode='valid')[:n]
        elif method == "Gaussian kernel":
            from scipy.ndimage import gaussian_filter1d
            sigma = win / 4
            return gaussian_filter1d(y, sigma)
        elif method == "Lowess":
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                frac = min(win / n, 0.9)
                result = lowess(y, x, frac=frac, return_sorted=False)
                return result
            except ImportError:
                # Fallback to savgol
                from scipy.signal import savgol_filter
                return savgol_filter(y, min(win, n if n % 2 == 1 else n-1), min(poly_order, win-1))
        elif method == "Spline":
            from scipy.interpolate import UnivariateSpline
            try:
                s_factor = len(y) * np.var(y) * (1.0 / max(poly_order, 1))
                spl = UnivariateSpline(x, y, s=s_factor, k=min(poly_order, 5))
                return spl(x)
            except Exception:
                return y.copy()
        return y.copy()

    def _update_preview(self):
        self._ax.clear()
        self._ax.set_facecolor('white')
        self._ax.tick_params(colors='#222')
        for sp in self._ax.spines.values(): sp.set_color('#999')

        if self._x_data is None:
            self._ax.set_title("Load data first", color='#888')
            self._canvas.draw(); return

        # Draw exclusion zones
        for xmin, xmax in self._exclusions:
            self._ax.axvspan(xmin, xmax, alpha=0.15, color='#d62728')

        # Original data
        self._ax.plot(self._x_data, self._y_data, 'o', color='#1f77b4',
                      markersize=3, alpha=0.6, label='original')

        # Smoothed (only non-NaN)
        if self._smoothed is not None:
            valid = ~np.isnan(self._smoothed)
            # Plot each contiguous valid segment
            mask = self._get_mask()
            inc_indices = np.where(mask)[0]
            if len(inc_indices) > 0:
                seg_start = 0
                for i in range(1, len(inc_indices)):
                    if inc_indices[i] - inc_indices[i-1] > 1:
                        seg = inc_indices[seg_start:i]
                        self._ax.plot(self._x_data[seg], self._smoothed[seg], '-',
                                      color='#2ca02c', linewidth=2)
                        seg_start = i
                seg = inc_indices[seg_start:]
                self._ax.plot(self._x_data[seg], self._smoothed[seg], '-',
                              color='#2ca02c', linewidth=2, label='smoothed')

        if self._chk_log_y.isChecked():
            self._ax.set_yscale('log')

        self._ax.set_title("CTR Smoothing", color='#222')
        self._ax.set_xlabel("X", color='#222')
        self._ax.set_ylabel("Intensity", color='#222')
        self._ax.legend(facecolor='white', edgecolor='#999', labelcolor='#222', fontsize=8)
        self._fig.tight_layout()
        self._canvas.draw()

    def _send_to_main(self):
        """Plot smoothed data on the main canvas."""
        if self._x_data is None or self._smoothed is None:
            self.say("No smoothed data", error=True); return

        tc = self._mw._text_color
        fig = self._mw._canvas.fig; fig.set_facecolor(self._mw._fig_bg_color)
        ax = self._mw._canvas.ax
        self._mw._canvas.clear_and_setup()
        ax.set_facecolor(self._mw._axes_bg_color)
        ax.tick_params(colors=tc, labelsize=self._mw._font_tick.value())
        for sp in ax.spines.values(): sp.set_color(tc); sp.set_linewidth(self._mw._spine_width.value())

        # Original
        ax.plot(self._x_data, self._y_data, 'o', color='#1f77b4', markersize=4, alpha=0.5, label='data')

        # Smoothed segments
        mask = self._get_mask()
        inc_indices = np.where(mask)[0]
        if len(inc_indices) > 0:
            seg_start = 0
            for i in range(1, len(inc_indices)):
                if inc_indices[i] - inc_indices[i-1] > 1:
                    seg = inc_indices[seg_start:i]
                    ax.plot(self._x_data[seg], self._smoothed[seg], '-', color='#2ca02c', linewidth=2)
                    seg_start = i
            seg = inc_indices[seg_start:]
            ax.plot(self._x_data[seg], self._smoothed[seg], '-', color='#2ca02c', linewidth=2, label='smoothed')

        # Exclusion zones
        for xmin, xmax in self._exclusions:
            ax.axvspan(xmin, xmax, alpha=0.1, color='#d62728')

        ax.set_xlabel("X", color=tc, fontsize=self._mw._font_axis.value())
        ax.set_ylabel("Intensity", color=tc, fontsize=self._mw._font_axis.value())
        ax.set_title("CTR Smoothed", color=tc, fontsize=self._mw._font_title.value())
        if self._mw._chk_logy.isChecked(): ax.set_yscale('log')
        if self._mw._chk_grid.isChecked():
            ax.grid(True, alpha=self._mw._grid_alpha.value(),
                    linestyle=self._mw._grid_style.currentText(), color=tc)
        ax.legend(facecolor=self._mw._axes_bg_color, edgecolor=tc, labelcolor=tc,
                  fontsize=self._mw._font_legend.value())
        fig.tight_layout()
        self._mw._canvas.draw()

    def _export(self):
        if self._x_data is None or self._smoothed is None:
            self.say("No data", error=True); return
        path, _ = QFileDialog.getSaveFileName(self, "Export", "smoothed.txt", "Text (*.txt)")
        if not path: return
        mask = self._get_mask()
        with open(path, 'w') as f:
            f.write("x\ty_original\ty_smoothed\tincluded\n")
            for i in range(len(self._x_data)):
                sm = self._smoothed[i] if not np.isnan(self._smoothed[i]) else ""
                inc = "1" if mask[i] else "0"
                f.write(f"{self._x_data[i]:.6g}\t{self._y_data[i]:.6g}\t{sm}\t{inc}\n")
        self.say(f"Exported to {path}")


class CTRSmoothDialog(StatusMixin, QDialog):
    """Dialog for smoothing CTR or any plotted data."""
    def __init__(self, main_win, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Smoothing")
        self.setMinimumSize(850, 550)
        self._mw = main_win
        self._sources = []          # traces to smooth: {'label', 'x', 'y', 'color'}
        self._on_time = False       # sources are on the plot's time axis (they can follow the potential)
        self._smoothed = []         # (x, y) per source, after Apply
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # --- Left controls ---
        left = QWidget()
        lv = QVBoxLayout(left); lv.setContentsMargins(4, 4, 4, 4)

        # Source
        sg = QGroupBox("Source data")
        sl = QVBoxLayout(sg)
        btn_from_plot = QPushButton("Grab from main plot")
        btn_from_plot.setToolTip("Takes every trace plotted on the main window's left axis")
        btn_from_plot.clicked.connect(self._grab_from_plot)
        sl.addWidget(btn_from_plot)
        btn_from_file = QPushButton("Load from text file")
        btn_from_file.clicked.connect(self._load_from_file)
        sl.addWidget(btn_from_file)
        self._source_info = QLabel("No data loaded")
        self._source_info.setStyleSheet(HINT_STYLE); self._source_info.setWordWrap(True)
        sl.addWidget(self._source_info)
        lv.addWidget(sg)

        # Method
        mg = QGroupBox("Smoothing method")
        ml = QVBoxLayout(mg)
        self._method_combo = QComboBox()
        self._method_combo.addItems(["Savitzky-Golay", "Moving average",
                                      "Spline interpolation", "Gaussian convolution"])
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        ml.addWidget(self._method_combo)

        # Parameters (stacked — different per method)
        param_grid = QGridLayout()

        param_grid.addWidget(QLabel("Window:"), 0, 0)
        self._window_size = QSpinBox()
        self._window_size.setRange(3, 999); self._window_size.setValue(5)
        self._window_size.setSingleStep(2)
        self._window_size.setToolTip("Must be odd for Savitzky-Golay")
        param_grid.addWidget(self._window_size, 0, 1)

        param_grid.addWidget(QLabel("Poly order:"), 1, 0)
        self._poly_order = QSpinBox()
        self._poly_order.setRange(1, 10); self._poly_order.setValue(3)
        param_grid.addWidget(self._poly_order, 1, 1)

        param_grid.addWidget(QLabel("Spline pts:"), 2, 0)
        self._spline_pts = QSpinBox()
        self._spline_pts.setRange(10, 10000); self._spline_pts.setValue(500)
        param_grid.addWidget(self._spline_pts, 2, 1)

        param_grid.addWidget(QLabel("Spline k:"), 3, 0)
        self._spline_k = QSpinBox()
        self._spline_k.setRange(1, 5); self._spline_k.setValue(3)
        self._spline_k.setToolTip("Spline degree (1=linear, 3=cubic)")
        param_grid.addWidget(self._spline_k, 3, 1)

        param_grid.addWidget(QLabel("Gauss σ:"), 4, 0)
        self._gauss_sigma = QDoubleSpinBox()
        self._gauss_sigma.setRange(0.1, 100); self._gauss_sigma.setValue(2.0)
        self._gauss_sigma.setSingleStep(0.5); self._gauss_sigma.setDecimals(2)
        param_grid.addWidget(self._gauss_sigma, 4, 1)

        ml.addLayout(param_grid)
        lv.addWidget(mg)

        # Actions
        btn_apply = QPushButton("Apply smoothing")
        btn_apply.clicked.connect(self._apply_smoothing)
        btn_apply.setStyleSheet(PRIMARY_STYLE)
        lv.addWidget(btn_apply)

        self._smooth_info = QLabel("—")
        self._smooth_info.setStyleSheet(HINT_STYLE); self._smooth_info.setWordWrap(True)
        lv.addWidget(self._smooth_info)

        # Display
        self._chk_show_orig = QCheckBox("Show original data"); self._chk_show_orig.setChecked(True)
        self._chk_show_orig.toggled.connect(self._update_preview)
        lv.addWidget(self._chk_show_orig)
        self._chk_logy = QCheckBox("Log Y")
        self._chk_logy.toggled.connect(self._update_preview)
        lv.addWidget(self._chk_logy)

        # Send / export
        btn_send = QPushButton("Send smoothed to main plot")
        btn_send.clicked.connect(self._send_to_main)
        lv.addWidget(btn_send)
        btn_export = QPushButton("Export smoothed data")
        btn_export.clicked.connect(self._export_smooth)
        lv.addWidget(btn_export)

        lv.addStretch()
        left.setFixedWidth(280)
        layout.addWidget(left)

        # --- Right: preview plot ---
        self._fig = Figure(figsize=(6, 4), dpi=100, facecolor='white')
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor('white')
        layout.addWidget(self._canvas, stretch=1)

        self._on_method_changed()

    def _on_method_changed(self):
        m = self._method_combo.currentText()
        self._window_size.setEnabled(m in ("Savitzky-Golay", "Moving average"))
        self._poly_order.setEnabled(m == "Savitzky-Golay")
        self._spline_pts.setEnabled(m == "Spline interpolation")
        self._spline_k.setEnabled(m == "Spline interpolation")
        self._gauss_sigma.setEnabled(m == "Gaussian convolution")

    def _set_sources(self, sources, info):
        self._sources = sources
        self._smoothed = []
        self._source_info.setText(info)
        self._update_preview()

    def _grab_from_plot(self):
        """Every trace on the main plot's left axis (the potential on a second axis is not smoothed)."""
        self._on_time = False
        if self._mw._last_traces:                  # scan traces: smooth along time, whatever the X shown
            sources = [{'label': tr['label'], 'x': np.asarray(tr['t'], float), 'y': np.asarray(tr['y'], float),
                        'color': tr.get('color', '#1f77b4')} for tr in self._mw._last_traces]
            self._on_time = True
            n = sum(len(src['x']) for src in sources)
            self._set_sources(sources, f"From plot: {len(sources)} trace(s), {n} points (along time)")
            return
        lines = [ln for ln in self._mw._canvas.ax.get_lines() if np.asarray(ln.get_xdata()).size]
        if not lines:
            self.say("No data on main plot", error=True)
            return
        sources = []
        for k, ln in enumerate(lines):
            label = ln.get_label()
            sources.append({'label': label if label and not label.startswith('_') else f"trace {k + 1}",
                            'x': np.array(ln.get_xdata(), dtype=float), 'y': np.array(ln.get_ydata(), dtype=float),
                            'color': ln.get_color()})
        n = sum(len(src['x']) for src in sources)
        self._set_sources(sources, f"From plot: {len(sources)} trace(s), {n} points")

    def _load_from_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load data", "", "Text (*.txt *.dat *.csv);;All (*)")
        if not path: return
        try:
            data = np.loadtxt(path, comments='#')
            if data.ndim == 1:
                x, y = np.arange(len(data), dtype=float), data
            else:
                x, y = data[:, 0], data[:, 1]
            self._on_time = False
            self._set_sources([{'label': os.path.basename(path), 'x': x, 'y': y, 'color': '#1f77b4'}],
                              f"From file: {len(x)} pts")
        except Exception as e:
            self.say(str(e), error=True)

    def _smooth_one(self, x, y):
        """(x, y) smoothed with the chosen method; non-finite points are left out first."""
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        method = self._method_combo.currentText()
        if method == "Savitzky-Golay":
            from scipy.signal import savgol_filter
            win = self._window_size.value()
            if win % 2 == 0: win += 1  # must be odd
            win = min(win, len(y) if len(y) % 2 else len(y) - 1)
            order = min(self._poly_order.value(), win - 1)
            return x.copy(), savgol_filter(y, win, order)
        if method == "Moving average":
            win = self._window_size.value()
            return x.copy(), np.convolve(y, np.ones(win) / win, mode='same')
        if method == "Spline interpolation":
            from scipy.interpolate import UnivariateSpline
            sort_idx = np.argsort(x)
            xs, ys = x[sort_idx], y[sort_idx]
            unique_mask = np.diff(xs, prepend=-np.inf) > 0
            xs, ys = xs[unique_mask], ys[unique_mask]
            if len(xs) < 4:
                raise ValueError("not enough unique points for a spline")
            k = min(self._spline_k.value(), len(xs) - 1)
            spl = UnivariateSpline(xs, ys, k=k, s=len(xs))
            sx = np.linspace(xs.min(), xs.max(), self._spline_pts.value())
            return sx, spl(sx)
        from scipy.ndimage import gaussian_filter1d
        return x.copy(), gaussian_filter1d(y, sigma=self._gauss_sigma.value())

    def _apply_smoothing(self):
        if not self._sources:
            self.say("Load data first", error=True); return
        smoothed = []
        for src in self._sources:
            try:
                smoothed.append(self._smooth_one(src['x'], src['y']))
            except Exception as e:
                self.say(f"Smoothing {src['label']} failed:\n{e}", error=True)
                return
        self._smoothed = smoothed
        n = sum(len(sx) for sx, _ in smoothed)
        self._smooth_info.setText(f"{self._method_combo.currentText()}: {len(smoothed)} trace(s), {n} pts output")
        self._update_preview()

    def _update_preview(self):
        self._ax.clear()
        self._ax.set_facecolor('white')
        self._ax.tick_params(colors='#222')
        for sp in self._ax.spines.values(): sp.set_color('#999')
        for k, src in enumerate(self._sources):
            if self._chk_show_orig.isChecked():
                self._ax.plot(src['x'], src['y'], 'o', color=src['color'], markersize=3, alpha=0.4,
                              label=f"{src['label']}")
            if k < len(self._smoothed):
                sx, sy = self._smoothed[k]
                self._ax.plot(sx, sy, '-', color=src['color'], linewidth=1.6, label=f"{src['label']} (smoothed)")
        if self._chk_logy.isChecked():
            self._ax.set_yscale('log')
        if self._sources:
            self._ax.legend(facecolor='white', edgecolor='#999', labelcolor='#222', fontsize=8)
        self._ax.set_title("Smoothing preview", color='#222')
        self._fig.tight_layout()
        self._canvas.draw()

    def _send_to_main(self):
        if not self._smoothed:
            self.say("Apply smoothing first", error=True); return
        ax = self._mw._canvas.ax
        self._mw._overlay = {
            'traces': [{'x': sx.copy(), 'y': sy.copy(), 'label': f"{src['label']} (smoothed)", 'color': src['color']}
                       for src, (sx, sy) in zip(self._sources, self._smoothed)],
            'xlabel': (getattr(self._mw, '_time_xlabel', None) if self._on_time else None) or ax.get_xlabel() or 'X',
            'ylabel': (ax.get_ylabel() or 'Y') + ' (smoothed)',
            'title': f"Smoothed ({self._method_combo.currentText()})",
            'time_axis': self._on_time,
        }
        self._mw._update_plot()

    def _export_smooth(self):
        if not self._smoothed:
            self.say("Apply smoothing first", error=True); return
        path, _ = QFileDialog.getSaveFileName(self, "Export smoothed", "smoothed.txt", "Text (*.txt)")
        if not path: return
        with open(path, 'w') as f:
            f.write(f"# Smoothed ({self._method_combo.currentText()}), {len(self._smoothed)} trace(s)\n")
            for src, (sx, sy) in zip(self._sources, self._smoothed):
                f.write(f"\n# Trace: {src['label']}\nx\t{src['label']}_smoothed\n")
                for xi, yi in zip(sx, sy):
                    f.write(f"{xi:.6g}\t{yi:.6g}\n")
        self.say(f"Exported {len(self._smoothed)} smoothed trace(s) to {path}")


class PerScanYDialog(QDialog):
    """Dialog to assign per-scan Y column overrides."""
    def __init__(self, main_win, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Per-Scan Y Columns")
        self.setMinimumSize(500, 450)
        self._mw = main_win
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Override Y columns for specific scans.\n"
                                "Scans not listed use the global Y selection from the main window."))

        # Table: scan number | Y columns (checkboxes shown via edit)
        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Scan", "Y Columns", ""])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self._table.setColumnWidth(0, 60)
        self._table.setColumnWidth(2, 60)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table)

        # Add row
        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Scan #:"))
        self._add_scan = QSpinBox()
        self._add_scan.setRange(1, 99999)
        add_row.addWidget(self._add_scan)
        btn_add = QPushButton("Add override")
        btn_add.clicked.connect(self._add_override)
        add_row.addWidget(btn_add)
        btn_clear = QPushButton("Clear all")
        btn_clear.clicked.connect(self._clear_all)
        add_row.addWidget(btn_clear)
        layout.addLayout(add_row)

        # Apply
        btn_apply = QPushButton("Apply && replot")
        btn_apply.setStyleSheet(PRIMARY_STYLE)
        btn_apply.clicked.connect(self._apply)
        layout.addWidget(btn_apply)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_table()

    def _get_all_bgsub_cols(self):
        """Get all available bgsub column names."""
        cols = OrderedDict()
        for sn, bd in self._mw._bgsub_data.items():
            for k in bd:
                if k not in ('scan', 'image'):
                    cols[k] = True
        # Also add spec Eiger columns
        if self._mw._spec_scans:
            first = next(iter(self._mw._spec_scans.values()))
            for c in first.columns:
                if 'eiger' in c.lower() or 'Eiger' in c:
                    cols[c] = True
        return list(cols.keys())

    def _refresh_table(self):
        self._table.setRowCount(0)
        overrides = self._mw._per_scan_y
        for sn in sorted(overrides.keys()):
            y_cols = overrides[sn]
            self._add_table_row(sn, y_cols)

    def _add_table_row(self, scan_num, y_cols):
        row = self._table.rowCount()
        self._table.setRowCount(row + 1)

        # Scan number
        item = QTableWidgetItem(str(scan_num))
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, 0, item)

        # Y columns as clickable button that opens selector
        btn = QPushButton(", ".join(y_cols) if y_cols else "(click to set)")
        btn.setStyleSheet("text-align: left; padding: 2px 6px; color: #1f5f9f;")
        btn.clicked.connect(lambda checked, r=row, s=scan_num: self._edit_y_cols(r, s))
        self._table.setCellWidget(row, 1, btn)

        # Remove button
        btn_rm = QPushButton("✕")
        btn_rm.setFixedWidth(40)
        btn_rm.clicked.connect(lambda checked, s=scan_num: self._remove_override(s))
        self._table.setCellWidget(row, 2, btn_rm)

    def _edit_y_cols(self, row, scan_num):
        """Open a checkable list of available columns."""
        all_cols = self._get_all_bgsub_cols()
        current = self._mw._per_scan_y.get(scan_num, [])

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Y columns for scan {scan_num}")
        dlg.setMinimumWidth(250)
        vl = QVBoxLayout(dlg)
        lst = QListWidget()
        for c in all_cols:
            item = QListWidgetItem(c)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if c in current or f"[bg] {c}" in current else Qt.Unchecked)
            lst.addItem(item)
        vl.addWidget(lst)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vl.addWidget(btns)

        if dlg.exec_() == QDialog.Accepted:
            selected = []
            for i in range(lst.count()):
                item = lst.item(i)
                if item.checkState() == Qt.Checked:
                    name = item.text()
                    # Add [bg] prefix for bgsub columns
                    if name in [k for bd in self._mw._bgsub_data.values() for k in bd if k not in ('scan', 'image')]:
                        selected.append(f"[bg] {name}")
                    else:
                        selected.append(name)
            self._mw._per_scan_y[scan_num] = selected
            # Update button text
            btn = self._table.cellWidget(row, 1)
            if btn:
                btn.setText(", ".join(selected) if selected else "(none)")

    def _add_override(self):
        sn = self._add_scan.value()
        if sn not in self._mw._per_scan_y:
            self._mw._per_scan_y[sn] = []
        self._refresh_table()

    def _remove_override(self, scan_num):
        self._mw._per_scan_y.pop(scan_num, None)
        self._refresh_table()

    def _clear_all(self):
        self._mw._per_scan_y.clear()
        self._refresh_table()

    def _apply(self):
        self._mw._update_plot()


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor='white')
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('white')
        self.ax.tick_params(colors='#222')
        for sp in self.ax.spines.values(): sp.set_color('#999')
        self.fig.tight_layout()

    def clear_and_setup(self):
        for extra in self.fig.axes[1:]:          # e.g. the potential axis of the last plot
            extra.remove()
        self.ax.clear()
        self.ax.set_facecolor('white')
        self.ax.tick_params(colors='#222')
        for sp in self.ax.spines.values(): sp.set_color('#999')


def overlay_traces(ov):
    """The traces of a main-plot overlay: a list under 'traces', or one trace as 'x' / 'y'."""
    if 'traces' in ov:
        return ov['traces']
    return [{'x': ov['x'], 'y': ov['y'], 'label': ov.get('label', 'data'), 'color': '#2ca02c'}]


ZERO_MODES = ('off (raw X)', 'start of each scan', 'start of the first scan')

DEFAULT_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
                  '#9467bd', '#8c564b', '#17becf', '#e377c2']


class MainWindow(StatusMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SXRD Spec Data Plotter")
        self.setMinimumSize(1400, 800)

        self._spec_scans = {}
        self._bgsub_data = {}
        self._bgsub_header = []
        self._label_map = load_label_map()
        self._plot_colors = list(DEFAULT_COLORS)

        # Color state (set before dialogs reference them)
        self._fig_bg_color = 'white'
        self._axes_bg_color = 'white'
        self._text_color = '#222222'

        # Overlay data from peak fitting / import (persists through style changes)
        # Format: {'x': array, 'y': array, 'xlabel': str, 'ylabel': str, 'title': str, 'label': str}
        self._overlay = None

        # Per-scan Y column overrides: {scan_number: [y_col_names]}
        # If a scan is in this dict, it uses these Y columns instead of the global selection
        self._per_scan_y = {}

        self._setup_ui()
        self._install_scroll_blocker()

        # Create dialogs (they create widgets on self)
        self._fig_opts_dialog = FigureOptionsDialog(self, self)
        self._peak_fit_dialog = PeakFitDialog(self, self)
        self._smooth_dialog = CTRSmoothDialog(self, self)
        self._perscan_y_dialog = PerScanYDialog(self, self)
        self._fig_opts_dialog._sync_color_buttons()

        self._load_settings()

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    # ---- Settings ----
    def _save_settings(self):
        c = configparser.ConfigParser()
        c['files'] = {'spec_path': self._spec_path.text(), 'bgsub_path': self._bgsub_path.text()}
        c['scan'] = {'selected': json.dumps(self._get_selected_scans())}
        c['attenuator'] = {
            'transmittance': str(self._att_trans.value()),
            'att1': str(self._att[0].isChecked()), 'att2': str(self._att[1].isChecked()),
            'att3': str(self._att[2].isChecked()), 'att4': str(self._att[3].isChecked()),
        }
        c['transm'] = {
            'enabled': str(self._chk_transm.isChecked()),
            'column': self._transm_col.text(),
        }
        c['display'] = {
            'log_x': str(self._chk_logx.isChecked()), 'log_y': str(self._chk_logy.isChecked()),
            'grid': str(self._chk_grid.isChecked()), 'legend': str(self._chk_legend.isChecked()),
            'markers': str(self._chk_markers.isChecked()),
            'line_style': self._line_combo.currentText(), 'marker_style': self._marker_combo.currentText(),
            'line_width': str(self._line_width.value()), 'marker_size': str(self._marker_size.value()),
            'fig_bg': self._fig_bg_color, 'axes_bg': self._axes_bg_color, 'text_color': self._text_color,
            'fig_w': str(self._fig_w.value()), 'fig_h': str(self._fig_h.value()), 'fig_dpi': str(self._fig_dpi.value()),
            'font_title': str(self._font_title.value()), 'font_axis': str(self._font_axis.value()),
            'font_tick': str(self._font_tick.value()), 'font_legend': str(self._font_legend.value()),
            'font_family': self._font_family.currentText(),
            'grid_style': self._grid_style.currentText(), 'grid_alpha': str(self._grid_alpha.value()),
            'grid_width': str(self._grid_width.value()), 'spine_width': str(self._spine_width.value()),
            'tick_width': str(self._tick_width.value()), 'tick_length': str(self._tick_length.value()),
            'tick_in': str(self._chk_tick_in.isChecked()), 'minor_ticks': str(self._chk_minor_ticks.isChecked()),
            'title': self._plot_title.text(),
            'xlim_min': self._xlim_min.text(), 'xlim_max': self._xlim_max.text(),
            'ylim_min': self._ylim_min.text(), 'ylim_max': self._ylim_max.text(),
        }
        c['potential'] = {
            'zero': self._zero_mode.currentText(), 'shifts': json.dumps({str(k): v for k, v in self._scan_shift.items()}),
            'zero_points': json.dumps({str(k): v for k, v in self._scan_point.items()}),
            'source': self._ec_source.currentText(), 'file': self._ec_path.text(),
            'cv_start': repr(self._cv_start.value()), 'cv_hold': repr(self._cv_hold.value()),
            'cv_upper': repr(self._cv_upper.value()), 'cv_lower': repr(self._cv_lower.value()),
            'cv_rate': repr(self._cv_rate.value()), 'cv_cycles': str(self._cv_cycles.value()),
            'cv_first': self._cv_first.currentText(), 'offset': repr(self._ec_offset.value()),
            'mode': self._ec_mode.currentText(), 'label': self._ec_ref.text(),
            'show_trace': str(self._chk_show_E.isChecked()),
            'align_point': str(self._align_point.value()), 'align_marks': self._align_marks.currentText(),
        }
        c['window'] = {
            'geometry': self.saveGeometry().toBase64().data().decode(),
            'hsplitter': self._hsplitter.saveState().toBase64().data().decode(),
            'left_vsplitter': self._left_vsplitter.saveState().toBase64().data().decode(),
        }
        try:
            with open(settings_file("sxrd_plotter_settings.ini"), 'w') as f: c.write(f)
        except Exception: pass

    def _load_settings(self):
        if not os.path.isfile(settings_file("sxrd_plotter_settings.ini")): return
        c = configparser.ConfigParser()
        try: c.read(settings_file("sxrd_plotter_settings.ini"))
        except Exception: return
        def get(s, k, d=''):
            try: return c.get(s, k)
            except: return d
        def getbool(s, k, d=False): return get(s, k, str(d)).lower() in ('true', '1', 'yes')

        sp = get('files', 'spec_path')
        if sp and os.path.isfile(sp): self._spec_path.setText(sp); self._load_spec_file(sp)
        bp = get('files', 'bgsub_path')
        if bp and os.path.isdir(bp): self._bgsub_path.setText(bp); self._load_bgsub_dir(bp)
        st = get('scan', 'selected', '[]')
        try:
            saved_scans = json.loads(st)
        except (json.JSONDecodeError, ValueError):
            saved_scans = []
        if saved_scans:
            self._scan_list.blockSignals(True)
            for i in range(self._scan_list.count()):
                item = self._scan_list.item(i)
                try:
                    sn = int(item.text().split(':')[0])
                    item.setSelected(sn in saved_scans)
                except ValueError:
                    pass
            self._scan_list.blockSignals(False)
            self._on_scan_changed()
        self._att_trans.setValue(float(get('attenuator', 'transmittance', '0.1')))
        for i, k in enumerate(['att1', 'att2', 'att3', 'att4']): self._att[i].setChecked(getbool('attenuator', k))
        self._chk_transm.setChecked(getbool('transm', 'enabled', False))
        self._transm_col.setText(get('transm', 'column', 'transm'))

        # Display
        self._chk_logx.setChecked(getbool('display', 'log_x'))
        self._chk_logy.setChecked(getbool('display', 'log_y'))
        self._chk_grid.setChecked(getbool('display', 'grid', True))
        self._chk_legend.setChecked(getbool('display', 'legend', True))
        self._chk_markers.setChecked(getbool('display', 'markers', True))
        for attr, key, default in [('_line_combo', 'line_style', '-'), ('_marker_combo', 'marker_style', 'o'),
                                    ('_grid_style', 'grid_style', ':'), ('_font_family', 'font_family', 'DejaVu Sans')]:
            val = get('display', key, default)
            w = getattr(self, attr); idx = w.findText(val)
            if idx >= 0: w.setCurrentIndex(idx)
        for attr, key, default in [('_line_width', 'line_width', '1.2'), ('_marker_size', 'marker_size', '4'),
                                    ('_grid_alpha', 'grid_alpha', '0.3'), ('_grid_width', 'grid_width', '0.5'),
                                    ('_spine_width', 'spine_width', '1'), ('_tick_width', 'tick_width', '1'),
                                    ('_tick_length', 'tick_length', '4')]:
            getattr(self, attr).setValue(float(get('display', key, default)))
        for attr, key, default in [('_fig_w', 'fig_w', '8'), ('_fig_h', 'fig_h', '6')]:
            getattr(self, attr).setValue(float(get('display', key, default)))
        self._fig_dpi.setValue(int(float(get('display', 'fig_dpi', '100'))))
        for attr, key, default in [('_font_title', 'font_title', '14'), ('_font_axis', 'font_axis', '12'),
                                    ('_font_tick', 'font_tick', '10'), ('_font_legend', 'font_legend', '9')]:
            getattr(self, attr).setValue(int(float(get('display', key, default))))
        self._chk_tick_in.setChecked(getbool('display', 'tick_in'))
        self._chk_minor_ticks.setChecked(getbool('display', 'minor_ticks'))
        self._plot_title.setText(get('display', 'title', ''))
        self._xlim_min.setText(get('display', 'xlim_min', ''))
        self._xlim_max.setText(get('display', 'xlim_max', ''))
        self._ylim_min.setText(get('display', 'ylim_min', ''))
        self._ylim_max.setText(get('display', 'ylim_max', ''))
        for attr, key, default in [('_fig_bg_color', 'fig_bg', 'white'),
                                    ('_axes_bg_color', 'axes_bg', 'white'),
                                    ('_text_color', 'text_color', '#222222')]:
            setattr(self, attr, get('display', key, default))
        self._fig_opts_dialog._sync_color_buttons()

        try:
            self._load_potential_settings(get, getbool)
        except ValueError:
            pass

        from PyQt5.QtCore import QByteArray
        geo = get('window', 'geometry')
        if geo: self.restoreGeometry(QByteArray.fromBase64(geo.encode()))
        hs = get('window', 'hsplitter')
        if hs: self._hsplitter.restoreState(QByteArray.fromBase64(hs.encode()))
        vs = get('window', 'left_vsplitter')
        if vs: self._left_vsplitter.restoreState(QByteArray.fromBase64(vs.encode()))

    # ---- UI ----
    def _setup_ui(self):
        self._hsplitter = QSplitter(Qt.Horizontal)

        # ---- left, top: files, scans, corrections, labels
        fg, fl = group("Files")
        self._spec_path = QLineEdit()
        self._spec_path.setReadOnly(True)
        sb = QPushButton("Browse…")
        sb.clicked.connect(self._browse_spec)
        fl.addRow("SPEC", row(self._spec_path, sb))
        self._bgsub_path = QLineEdit()
        self._bgsub_path.setReadOnly(True)
        self._bgsub_path.setPlaceholderText("optional: ROI sums from background subtraction")
        bb = QPushButton("Browse…")
        bb.clicked.connect(self._browse_bgsub)
        fl.addRow("ROI sums", row(self._bgsub_path, bb))

        sg, sl = group("Scans", form=False)
        self._scan_list = QListWidget()
        self._scan_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._scan_list.setMinimumHeight(140)
        self._scan_list.itemSelectionChanged.connect(self._on_scan_changed)
        sl.addWidget(self._scan_list)
        self._scan_info = hint("No scan loaded")
        sl.addWidget(self._scan_info)

        ag, al = group("Corrections")
        self._att_trans = QDoubleSpinBox()
        self._att_trans.setRange(0.0001, 1)
        self._att_trans.setValue(0.1)
        self._att_trans.setDecimals(6)
        self._att_trans.setSingleStep(0.01)
        self._att_trans.setToolTip("Transmission of one attenuator sheet")
        self._att_trans.editingFinished.connect(self._update_plot)
        al.addRow("Sheet transmission", self._att_trans)
        self._att = []
        sheets = []
        for n in [1, 2, 4, 8]:
            chk = QCheckBox(f"{n}")
            chk.toggled.connect(self._update_plot)
            self._att.append(chk)
            sheets.append(chk)
        self._lbl_att = hint("×1.000")
        al.addRow("Sheets in", row(*sheets, self._lbl_att, stretch_first=False))
        self._chk_transm = QCheckBox("Divide Y by")
        self._chk_transm.setToolTip("Divides each Y point by the corresponding transm value from spec.\n"
                                    "For rocking scans with auto-attenuators this varies per image.\n"
                                    "Applies on top of the normalize column (typically the ion chamber).")
        self._chk_transm.toggled.connect(self._update_plot)
        self._transm_col = QLineEdit("transm")
        self._transm_col.setToolTip("Name of the per-image transmission column in the SPEC file")
        self._transm_col.editingFinished.connect(self._update_plot)
        al.addRow("Per image", row(self._chk_transm, self._transm_col, stretch_first=False))

        pg = self._potential_group()

        lg, ll = group("Axis labels")
        self._lbl_from = QLineEdit()
        self._lbl_from.setPlaceholderText("VTH")
        self._lbl_to = QLineEdit()
        self._lbl_to.setPlaceholderText("η")
        btn_map = QPushButton("Set")
        btn_map.clicked.connect(self._add_label_mapping)
        ll.addRow("Column → label", row(self._lbl_from, QLabel("→"), self._lbl_to, btn_map, stretch_first=False))
        self._lbl_map_info = hint("")
        ll.addRow(self._lbl_map_info)
        self._refresh_label_map_display()

        # ---- left, bottom: the column table (X / Y / normalise), resizable
        cg, cl = group("Columns: X / Y / normalise", form=False)
        self._col_sel = ColumnSelectorWidget(on_change=self._update_plot)
        cl.addWidget(self._col_sel)
        left_vs = QSplitter(Qt.Vertical)
        left_vs.addWidget(sidebar(fg, sg, ag, pg, lg))
        left_vs.addWidget(cg)
        left_vs.setStretchFactor(0, 1)
        left_vs.setStretchFactor(1, 1)
        left_vs.setSizes([420, 360])
        self._left_vsplitter = left_vs

        btns = [("Export data…", self._export_plot_data, "Plot data as text (one '# Trace:' block per curve)"),
                ("Export for GenX…", self._export_genx, "Plot data in GenX format"),
                ("Export figure…", self._export_figure, "Final image: preview, size, fonts, colours, labels, traces"),
                ("Export t, E, I…", self._export_potential_data, "Time, potential and intensity of every trace"),
                ("Save config…", self._save_plot_config, "Columns, styles and options as JSON"),
                ("Load config…", self._load_plot_config, None)]
        grid = QGridLayout()
        for k, (text, fn, tip) in enumerate(btns):
            b = QPushButton(text)
            b.clicked.connect(lambda *_, f=fn: f())
            if tip:
                b.setToolTip(tip)
            grid.addWidget(b, k // 2, k % 2)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(left_vs, 1)
        lv.addLayout(grid)
        self._hsplitter.addWidget(left)

        # ---- right: the plot with its tools
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self._canvas = PlotCanvas()
        self._toolbar = NavigationToolbar(self._canvas, self)
        tools = []
        for text, fn, tip in (("Export figure…", self._export_figure, "Final image: preview, size, fonts, colours, "
                                                                       "labels, traces (the working view stays as is)"),
                              ("View options…", self._show_fig_opts, "Look of this working view: log axes, limits, "
                                                                     "markers, grid, fonts"),
                              ("Peak fitting…", self._show_peak_fit, "Fit rocking curves, integrated intensities"),
                              ("Smoothing…", self._show_smooth, "Smooth a CTR built from the peak fits"),
                              ("Per-scan Y…", self._show_perscan_y, "Different Y columns for some scans"),
                              ("CTR fitting", self._launch_ctr_fit, "Quick CTR models (the CTR fit tab)")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda *_, f=fn: f())
            tools.append(b)
        rv.addWidget(row(self._toolbar, *tools))
        rv.addWidget(self._canvas, stretch=1)
        self._hsplitter.addWidget(right)
        self._hsplitter.setStretchFactor(0, 0)
        self._hsplitter.setStretchFactor(1, 1)
        self._hsplitter.setSizes([400, 1000])
        self.setCentralWidget(self._hsplitter)
        self.say("Open a SPEC file (and the ROI sums folder), pick scans and the X / Y columns.")

    # ---- Potential (EC-Lab) and time zero ----
    def _potential_group(self):
        self._ec = None                 # ECData shown with the scans (file or defined by hand)
        self._export_window = None      # the Export figure window, while open
        self._last_traces = []          # what the last plot drew (for 'Export I vs E')
        box, f = group("Potential and time")
        # time axis: zero and a shift per scan
        self._zero_mode = QComboBox()
        self._zero_mode.addItems(list(ZERO_MODES))
        self._zero_mode.setToolTip("Each scan's start: every scan begins at t = 0 (scans with their own CV).\n"
                                   "The first scan's start: one clock for all scans (one long potentiostat record).")
        self._zero_mode.currentIndexChanged.connect(self._update_plot)
        f.addRow("t = 0 at", self._zero_mode)
        self._scan_shift = {}           # scan number -> extra time shift (s)
        self._scan_point = {}           # scan number -> point that is t = 0 (0 = the first point)
        self._shift_table = QTableWidget(0, 3)
        self._shift_table.setHorizontalHeaderLabels(["Scan", "t = 0 at point #", "Shift (s)"])
        self._shift_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._shift_table.verticalHeader().setVisible(False)
        self._shift_table.setMaximumHeight(120)
        self._shift_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._shift_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._shift_table.setToolTip("t = 0 at point #: the SPEC point (0-based) of each scan that is time zero, "
                                     "e.g. where its CV started.\nShift: seconds added to the scan's time.\n"
                                     "The selected row is the scan used by 'At point'.")
        self._shift_table.itemChanged.connect(self._shift_edited)
        f.addRow(self._shift_table)
        # potentiostat record
        self._ec_source = QComboBox()
        self._ec_source.addItems(["None", "EC-Lab file", "Defined by hand"])
        self._ec_source.currentIndexChanged.connect(self._ec_source_changed)
        f.addRow("Potential", self._ec_source)
        self._ec_file_row = QWidget()
        fl = QHBoxLayout(self._ec_file_row)
        fl.setContentsMargins(0, 0, 0, 0)
        self._ec_path = QLineEdit()
        self._ec_path.setReadOnly(True)
        self._ec_path.setPlaceholderText(".mpr (needs galvani), .mpt / .txt export or .csv")
        b_ec = QPushButton("Browse…")
        b_ec.clicked.connect(self._browse_ec)
        fl.addWidget(self._ec_path, 1)
        fl.addWidget(b_ec)
        f.addRow(self._ec_file_row)
        self._cv_box = QWidget()
        cf = QFormLayout(self._cv_box)
        cf.setContentsMargins(0, 0, 0, 0)

        def dspin(lo, hi, val, dec, suffix, tip):
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setDecimals(dec)
            w.setValue(val)
            w.setSuffix(suffix)
            w.setToolTip(tip)
            w.editingFinished.connect(self._rebuild_ec)
            return w
        self._cv_start = dspin(-5, 5, 0.0, 3, " V", "Potential of the hold before the first sweep")
        self._cv_hold = dspin(0, 1e6, 0.0, 1, " s", "Hold at the start potential before sweeping")
        self._cv_upper = dspin(-5, 5, 0.7, 3, " V", "Upper vertex")
        self._cv_lower = dspin(-5, 5, 0.0, 3, " V", "Lower vertex")
        self._cv_rate = dspin(0.001, 1e5, 5.0, 3, " mV/s", "Sweep rate")
        self._cv_cycles = QSpinBox()
        self._cv_cycles.setRange(1, 10000)
        self._cv_cycles.setValue(3)
        self._cv_cycles.editingFinished.connect(self._rebuild_ec)
        self._cv_first = QComboBox()
        self._cv_first.addItems(list(FIRST_SWEEPS))
        self._cv_first.currentIndexChanged.connect(self._rebuild_ec)
        cf.addRow("Start, hold", row(self._cv_start, self._cv_hold, stretch_first=False))
        cf.addRow("Lower, upper", row(self._cv_lower, self._cv_upper, stretch_first=False))
        cf.addRow("Rate", self._cv_rate)
        cf.addRow("Cycles, first", row(self._cv_cycles, self._cv_first, stretch_first=False))
        f.addRow(self._cv_box)
        self._ec_offset = QDoubleSpinBox()
        self._ec_offset.setRange(-1e12, 1e12)
        self._ec_offset.setDecimals(3)
        self._ec_offset.setSuffix(" s")
        self._ec_offset.setToolTip("Where the potentiostat's first sample lies on the plotted time axis "
                                   "(after t0). Raise it if the potential leads the intensity.")
        self._ec_offset.editingFinished.connect(self._update_plot)
        f.addRow("EC start at t =", self._ec_offset)
        self._align_point = QSpinBox()
        self._align_point.setRange(0, 10 ** 7)
        self._align_point.setPrefix("#")
        self._align_point.setToolTip("SPEC point number (0-based, as SPEC counts) in the scan chosen above")
        self._align_marks = QComboBox()
        self._align_marks.addItems(list(MARKS))
        self._align_marks.setToolTip("What happens at that point: the first sweep starts (after any hold; found "
                                     "from the potential) or the potentiostat record starts")
        b_align = QPushButton("Align")
        b_align.setToolTip("Set 'EC start at t' so that this happens at the chosen point's X (time) value")
        b_align.clicked.connect(self._align_at_point)
        f.addRow("At point", row(self._align_point, self._align_marks, b_align, stretch_first=False))
        self._ec_mode = QComboBox()
        self._ec_mode.addItems(list(MODES))
        self._ec_mode.currentIndexChanged.connect(self._update_plot)
        self._chk_show_E = QCheckBox("Potential trace")
        self._chk_show_E.setChecked(True)
        self._chk_show_E.setToolTip("Show or hide the potential curve on the second axis (vs time)")
        self._chk_show_E.toggled.connect(self._update_plot)
        f.addRow("Show", row(self._ec_mode, self._chk_show_E))
        self._ec_ref = QLineEdit("V vs Ag/AgCl")
        self._ec_ref.setToolTip("Unit / reference electrode in the potential axis label")
        self._ec_ref.editingFinished.connect(self._update_plot)
        f.addRow("Label", self._ec_ref)
        self._ec_info = hint("No potential record")
        f.addRow(self._ec_info)
        self._ec_file_row.setVisible(False)
        self._cv_box.setVisible(False)
        return box

    def _load_potential_settings(self, get, getbool):
        widgets = [self._zero_mode, self._chk_show_E, self._ec_source, self._ec_path, self._cv_start, self._cv_hold,
                   self._cv_upper, self._cv_lower, self._cv_rate, self._cv_cycles, self._cv_first,
                   self._ec_offset, self._ec_mode, self._ec_ref]
        for w in widgets:
            w.blockSignals(True)
        try:
            i = self._zero_mode.findText(get('potential', 'zero', ZERO_MODES[0]))
            self._zero_mode.setCurrentIndex(max(i, 0))
            try:
                self._scan_shift = {int(k): float(v) for k, v in json.loads(get('potential', 'shifts', '{}')).items()}
            except (ValueError, TypeError, AttributeError):
                self._scan_shift = {}
            try:
                self._scan_point = {int(k): int(v) for k, v in json.loads(get('potential', 'zero_points', '{}')).items()}
            except (ValueError, TypeError, AttributeError):
                self._scan_point = {}
            for spin, key, d in ((self._cv_start, 'cv_start', '0'), (self._cv_hold, 'cv_hold', '0'),
                                 (self._cv_upper, 'cv_upper', '0.7'), (self._cv_lower, 'cv_lower', '0'),
                                 (self._cv_rate, 'cv_rate', '5'), (self._ec_offset, 'offset', '0')):
                spin.setValue(float(get('potential', key, d)))
            self._cv_cycles.setValue(int(get('potential', 'cv_cycles', '3')))
            for combo, key in ((self._cv_first, 'cv_first'), (self._ec_mode, 'mode'), (self._ec_source, 'source')):
                i = combo.findText(get('potential', key, combo.currentText()))
                if i >= 0:
                    combo.setCurrentIndex(i)
            self._ec_path.setText(get('potential', 'file', ''))
            self._ec_ref.setText(get('potential', 'label', 'V vs Ag/AgCl'))
            self._chk_show_E.setChecked(getbool('potential', 'show_trace', True))
            self._align_point.setValue(int(get('potential', 'align_point', '0')))
            i = self._align_marks.findText(get('potential', 'align_marks', MARKS[0]))
            if i >= 0:
                self._align_marks.setCurrentIndex(i)
        finally:
            for w in widgets:
                w.blockSignals(False)
        src = self._ec_source.currentText()
        self._ec_file_row.setVisible(src == "EC-Lab file")
        self._cv_box.setVisible(src == "Defined by hand")
        if src == "Defined by hand" or (src == "EC-Lab file" and os.path.isfile(self._ec_path.text())):
            self._rebuild_ec()

    def _ec_source_changed(self, *_):
        src = self._ec_source.currentText()
        self._ec_file_row.setVisible(src == "EC-Lab file")
        self._cv_box.setVisible(src == "Defined by hand")
        self._rebuild_ec()

    def _browse_ec(self):
        path, _ = QFileDialog.getOpenFileName(self, "Potentiostat file", "",
                                              "EC-Lab (*.mpr *.mpt *.txt);;Tables (*.csv *.dat *.txt);;All (*)")
        if path:
            self._ec_path.setText(path)
            self._rebuild_ec()

    def _rebuild_ec(self, *_):
        """Read the file or build the hand-defined CV; errors go to the status bar."""
        src = self._ec_source.currentText()
        self._ec = None
        try:
            if src == "EC-Lab file" and self._ec_path.text():
                self._ec = load_potential(self._ec_path.text())
            elif src == "Defined by hand":
                self._ec = hand_cv(self._cv_start.value(), self._cv_hold.value(), self._cv_upper.value(),
                                   self._cv_lower.value(), self._cv_rate.value(), self._cv_cycles.value(),
                                   self._cv_first.currentText())
        except ImportError as e:
            self.say(f"Cannot read .mpr files: {e}. Install galvani or export the file as .mpt from EC-Lab.", error=True)
        except Exception as e:
            self.say(f"Potential: {e}", error=True)
        if self._ec is not None:
            t, E = self._ec.time, self._ec.potential
            self._ec_info.setText(f"{t.size} points over {t[-1] - t[0]:.0f} s, E {E.min():.3f} to {E.max():.3f} V")
        else:
            self._ec_info.setText("No potential record")
        if hasattr(self, "_fig_opts_dialog"):          # not while the window is being built
            self._update_plot()

    def _x_values(self, sn, x_col):
        scan = self._spec_scans.get(sn)
        if scan is None:
            return None
        if x_col.startswith('[bg] '):
            return self._bgsub_data.get(sn, {}).get(x_col[5:])
        return scan.data.get(x_col)

    def _align_at_point(self):
        """Put the first sweep (or the record start) at the time of a SPEC point of the chosen scan."""
        if self._ec is None:
            self.say("Choose a potential (EC-Lab file or defined by hand) first", error=True)
            return
        x_col = self._col_sel.get_selection()[0]
        sn = self._chosen_scan()
        x = self._x_values(sn, x_col) if (x_col and sn is not None) else None
        if x is None:
            self.say("Pick the X (time) column and a scan (select its row in the shift table)", error=True)
            return
        try:
            t = point_time(self._shift_x(np.asarray(x, dtype=float), sn, x_col), self._align_point.value())
            off = offset_for_point(self._ec, t, self._align_marks.currentText())
        except ValueError as e:
            self.say(str(e), error=True)
            return
        self._ec_offset.setValue(off)
        what = "First sweep starts" if self._align_marks.currentIndex() == 0 else "EC record starts"
        self.say(f"{what} at point #{self._align_point.value()} of scan {sn} (t = {t:.6g} s): "
                 f"EC start at t = {off:.6g} s")
        self._update_plot()

    def _draw_potential_axis(self, ax, t_lo, t_hi, tc, lw, tick_dir):
        """E(t) on a second y axis over the plotted time range."""
        if self._ec is None or not self._chk_show_E.isChecked() or not (np.isfinite(t_lo) and t_hi > t_lo):
            return
        ref = self._ec_ref.text().strip() or "V"
        ax2 = ax.twinx()
        te, Ee = trace_between(self._ec, self._ec_offset.value(), t_lo, t_hi)
        ax2.plot(te, Ee, color=tc, linestyle='--', linewidth=max(lw * 0.8, 0.8), label='E')
        ax2.set_ylabel(f"E ({ref})", color=tc, fontsize=self._font_axis.value())
        ax2.tick_params(colors=tc, labelsize=self._font_tick.value(), direction=tick_dir)
        for sp in ax2.spines.values(): sp.set_color(tc); sp.set_linewidth(self._spine_width.value())
        if te.size == 0:
            ax2.text(0.5, 0.5, "potential record outside the plotted time range\n(check t0 and EC start)",
                     transform=ax2.transAxes, ha='center', va='center', color='#b00020')

    def _refresh_shift_table(self):
        """One row per selected scan in the shift table (keeps the selection and the shifts)."""
        cur = self._chosen_scan()
        t = self._shift_table
        t.blockSignals(True)
        t.setRowCount(0)
        for sn in self._get_selected_scans():
            r = t.rowCount()
            t.insertRow(r)
            name = QTableWidgetItem(f"S{sn}")
            name.setFlags(name.flags() & ~Qt.ItemIsEditable)
            name.setData(Qt.UserRole, sn)
            t.setItem(r, 0, name)
            t.setItem(r, 1, QTableWidgetItem(str(self._scan_point.get(sn, 0))))
            t.setItem(r, 2, QTableWidgetItem(f"{self._scan_shift.get(sn, 0.0):g}"))
            if sn == cur:
                t.selectRow(r)
        if t.rowCount() and not t.selectedItems():
            t.selectRow(0)
        t.blockSignals(False)

    def _chosen_scan(self):
        """The scan of the selected row of the shift table (else the first selected scan)."""
        rows = self._shift_table.selectionModel().selectedRows() if self._shift_table.selectionModel() else []
        if rows:
            item = self._shift_table.item(rows[0].row(), 0)
            if item is not None:
                return item.data(Qt.UserRole)
        sel = self._get_selected_scans()
        return sel[0] if sel else None

    def _shift_edited(self, item):
        if item.column() == 0:
            return
        sn = self._shift_table.item(item.row(), 0).data(Qt.UserRole)
        text = item.text().strip() or "0"
        try:
            if item.column() == 1:
                point = int(text)
                x_col = self._col_sel.get_selection()[0]
                x = self._x_values(sn, x_col) if x_col else None
                if x is not None:
                    point_time(x, point)                       # raises if the point is not in the scan
                self._scan_point[sn] = point
            else:
                self._scan_shift[sn] = float(text)
        except ValueError as e:
            what = "point #" if item.column() == 1 else "shift"
            self.say(f"S{sn} {what}: {e if 'point' in str(e) else repr(item.text()) + ' is not a number'}",
                     error=True)
            return
        if self._zero_mode.currentIndex() == 0 and item.column() == 1:
            self._zero_mode.setCurrentIndex(1)                 # a zero point means: zero at each scan
            return
        self._update_plot()

    def _zero_of(self, sn, x_col):
        """The time subtracted from scan sn's X: its own start, the first selected scan's start, or 0."""
        mode = self._zero_mode.currentIndex()
        if mode == 0 or not x_col:
            return 0.0
        ref = sn if mode == 1 else (self._get_selected_scans() or [sn])[0]
        x = self._x_values(ref, x_col)
        if x is None:
            return 0.0
        try:
            point = self._scan_point.get(ref, 0)
            return point_time(x, point) if point else time_zero(x)
        except ValueError:
            return 0.0

    def _shift_x(self, xd, sn, x_col):
        """X on the plot's time axis: minus the time zero, plus the scan's own shift."""
        return xd - self._zero_of(sn, x_col) + self._scan_shift.get(sn, 0.0)

    def _export_potential_data(self):
        if not self._last_traces:
            self.say("Nothing plotted to export", error=True)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export t, E, I", "intensity_vs_potential.txt", "Text (*.txt)")
        if not path:
            return
        ref = self._ec_ref.text().strip() or "V"
        with open(path, 'w') as f:
            f.write("# Exported from SXRD Plotter: time on the plot's axis, potential there, intensity\n")
            f.write(f"# t = 0 at: {self._zero_mode.currentText()}; scan shifts (s): "
                    f"{ {k: v for k, v in self._scan_shift.items() if v} }; t = 0 at points "
                    f"{ {k: v for k, v in self._scan_point.items() if v} }; EC start at t = {self._ec_offset.value():.6g} s\n")
            for tr in self._last_traces:
                f.write(f"\n# Trace: {tr['label']}\nt_s\tE_{ref.replace(' ', '_')}\t{tr['label']}\n")
                E = tr['E'] if tr['E'] is not None else np.full(len(tr['t']), np.nan)
                for ti, ei, yi in zip(tr['t'], E, tr['y']):
                    f.write(f"{ti:.6g}\t{ei:.6g}\t{yi:.6g}\n")
        self.say(f"Exported {len(self._last_traces)} traces to {path}")

    def _show_fig_opts(self):
        self._fig_opts_dialog._sync_color_buttons()
        self._fig_opts_dialog.show(); self._fig_opts_dialog.raise_()

    def _show_peak_fit(self):
        self._peak_fit_dialog.show(); self._peak_fit_dialog.raise_()

    def _show_smooth(self):
        self._smooth_dialog.show(); self._smooth_dialog.raise_()

    def _show_perscan_y(self):
        self._perscan_y_dialog.show(); self._perscan_y_dialog.raise_()

    def _launch_ctr_fit(self):
        """Show the CTR fitting tool: its tab in the data tools window, or a window of its own."""
        host = getattr(self, "host", None)
        if host is not None:
            host.show_tool("ctr_fit")
            return
        from .ctr_fit import CTRFitWindow
        self._ctr_window = CTRFitWindow()
        self._ctr_window.show()

    def _install_scroll_blocker(self):
        self._scroll_filter = _ScrollBlockFilter(self)
        for w in self.findChildren(QSpinBox): w.setFocusPolicy(Qt.StrongFocus); w.installEventFilter(self._scroll_filter)
        for w in self.findChildren(QDoubleSpinBox): w.setFocusPolicy(Qt.StrongFocus); w.installEventFilter(self._scroll_filter)
        for w in self.findChildren(QComboBox): w.setFocusPolicy(Qt.StrongFocus); w.installEventFilter(self._scroll_filter)

    # ---- Color picker ----
    def _pick_color(self, target):
        cur = QColor({'fig_bg': self._fig_bg_color, 'axes_bg': self._axes_bg_color, 'text': self._text_color}[target])
        color = QColorDialog.getColor(cur, self, f"Pick {target} color")
        if not color.isValid(): return
        h = color.name()
        if target == 'fig_bg': self._fig_bg_color = h
        elif target == 'axes_bg': self._axes_bg_color = h
        elif target == 'text': self._text_color = h
        self._fig_opts_dialog._sync_color_buttons()
        self._update_plot()

    def _apply_figsize(self):
        # Don't resize the embedded canvas — it should fill available space.
        # Figure size and DPI are applied only during export (savefig).
        self._update_plot()

    def _preset_dark(self):
        self._fig_bg_color = '#353535'; self._axes_bg_color = '#2a2a2a'; self._text_color = '#dddddd'
        self._spine_width.setValue(1); self._tick_width.setValue(1); self._tick_length.setValue(4)
        self._grid_alpha.setValue(0.3); self._chk_tick_in.setChecked(False); self._chk_minor_ticks.setChecked(False)
        idx = self._font_family.findText("DejaVu Sans")
        if idx >= 0: self._font_family.setCurrentIndex(idx)
        self._fig_opts_dialog._sync_color_buttons(); self._update_plot()

    def _preset_light(self):
        self._fig_bg_color = '#ffffff'; self._axes_bg_color = '#ffffff'; self._text_color = '#000000'
        self._font_title.setValue(14); self._font_axis.setValue(12); self._font_tick.setValue(10); self._font_legend.setValue(9)
        self._spine_width.setValue(1.5); self._tick_width.setValue(1.5); self._tick_length.setValue(6)
        self._chk_tick_in.setChecked(True); self._chk_minor_ticks.setChecked(True); self._grid_alpha.setValue(0.15)
        for name in ["Times New Roman", "serif"]:
            idx = self._font_family.findText(name)
            if idx >= 0: self._font_family.setCurrentIndex(idx); break
        self._fig_opts_dialog._sync_color_buttons(); self._update_plot()

    # ---- File loading ----
    def _browse_spec(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open spec file", "", "Spec files (*.spec *.txt *.dat);;All (*)")
        if path: self._spec_path.setText(path); self._load_spec_file(path)

    def _browse_bgsub(self):
        path = QFileDialog.getExistingDirectory(self, "Select BGsub sums directory")
        if path: self._bgsub_path.setText(path); self._load_bgsub_dir(path)

    def _load_spec_file(self, path):
        try:
            self._spec_scans = parse_spec_file(path)
            self._scan_list.blockSignals(True); self._scan_list.clear()
            for sn in sorted(self._spec_scans.keys()):
                self._scan_list.addItem(f"{sn}: {self._spec_scans[sn].scan_cmd}")
            self._scan_list.blockSignals(False)
            if self._scan_list.count() > 0:
                self._scan_list.item(0).setSelected(True)
                self._on_scan_changed()
            self.say(f"Loaded {len(self._spec_scans)} scans")
        except Exception as e: self.say(str(e), error=True)

    def _load_bgsub_dir(self, dirpath):
        try:
            self._bgsub_data, self._bgsub_header, n_files = parse_bgsub_dir(dirpath)
            self._on_scan_changed()
            self.say(f"BGsub: {n_files} files, scans {sorted(self._bgsub_data.keys())}")
        except Exception as e: self.say(str(e), error=True)

    def _get_selected_scans(self):
        """Return list of selected scan numbers."""
        scans = []
        for item in self._scan_list.selectedItems():
            try:
                sn = int(item.text().split(':')[0])
                scans.append(sn)
            except ValueError:
                pass
        return sorted(scans)

    def _on_scan_changed(self):
        selected = self._get_selected_scans()
        if not selected: return
        # Clear overlay when user switches to scan data
        self._overlay = None
        self._refresh_shift_table()
        sn = selected[0]
        scan = self._spec_scans.get(sn)
        if not scan: return
        if len(selected) == 1:
            self._scan_info.setText(f"Scan {sn}: {scan.scan_cmd}\n{scan.npoints} pts, {len(scan.columns)} cols")
        else:
            self._scan_info.setText(f"{len(selected)} scans selected\nColumns from scan {sn}")
        # Remember current selection before rebuilding column list
        prev_x, prev_y, prev_norm = self._col_sel.get_selection()
        # Collect bgsub columns from ALL selected scans (union)
        bgsub_cols_set = OrderedDict()
        for s in selected:
            if s in self._bgsub_data:
                for k in self._bgsub_data[s]:
                    if k not in ('scan', 'image') and k not in bgsub_cols_set:
                        bgsub_cols_set[k] = True
        # Also include any bgsub columns from non-selected scans so they're available
        for s, bd in self._bgsub_data.items():
            for k in bd:
                if k not in ('scan', 'image') and k not in bgsub_cols_set:
                    bgsub_cols_set[k] = True
        bgsub_cols = list(bgsub_cols_set.keys())
        self._col_sel.set_columns(scan.columns, bgsub_cols)
        # Restore previous selection if columns still exist
        if prev_x or prev_y or prev_norm:
            self._col_sel.set_selection(prev_x, prev_y, prev_norm)
        self._update_plot()

    def _get_att_factor(self):
        T1 = self._att_trans.value()
        total = sum(sheets for i, sheets in enumerate([1, 2, 4, 8]) if self._att[i].isChecked())
        factor = 1.0 / (T1 ** total) if total > 0 else 1.0
        self._lbl_att.setText(f"×{factor:.4g} ({total} sheets)")
        return factor

    def _apply_transm(self, scan, y, img_idx=None):
        """Divide y by the transm column from spec, per-image.
        If img_idx is given, picks transm[img_idx]; else uses transm[:len(y)].
        Zero or negative transm values are skipped (left uncorrected for that point)."""
        if not self._chk_transm.isChecked():
            return y
        col_name = self._transm_col.text().strip()
        if not col_name:
            return y
        td = scan.data.get(col_name)
        if td is None:
            return y
        if img_idx is not None:
            valid_mask = img_idx < len(td)
            if not valid_mask.all():
                # Pad: for any out-of-range image, use 1 (no correction)
                t_vals = np.ones(len(img_idx))
                t_vals[valid_mask] = td[img_idx[valid_mask]]
            else:
                t_vals = td[img_idx]
        else:
            n = min(len(y), len(td))
            t_vals = np.ones(len(y))
            t_vals[:n] = td[:n]
        # Avoid div by zero / negative
        bad = (t_vals <= 0) | ~np.isfinite(t_vals)
        t_vals = np.where(bad, 1.0, t_vals)
        return y / t_vals

    # ---- Plotting ----
    def _update_plot(self):
        tc = self._text_color
        fig = self._canvas.fig; fig.set_facecolor(self._fig_bg_color)
        matplotlib.rcParams['font.family'] = self._font_family.currentText()
        self._canvas.clear_and_setup()
        ax = self._canvas.ax; ax.set_facecolor(self._axes_bg_color)
        tick_dir = 'in' if self._chk_tick_in.isChecked() else 'out'
        ax.tick_params(colors=tc, labelsize=self._font_tick.value(),
                       width=self._tick_width.value(), length=self._tick_length.value(), direction=tick_dir, which='both')
        for sp in ax.spines.values(): sp.set_color(tc); sp.set_linewidth(self._spine_width.value())

        ls = self._line_combo.currentText(); ms_style = self._marker_combo.currentText()
        if ls == 'none': ls = ''
        if ms_style == 'none': ms_style = ''
        show_m = self._chk_markers.isChecked(); lw = self._line_width.value(); ms_sz = self._marker_size.value()

        has_data = False
        self._last_traces = []                             # filled again by what this plot draws

        # If overlay data exists, render it
        if self._overlay is not None:
            ov = self._overlay
            kw = {'linestyle': ls if ls else 'None'}
            if show_m and ms_style: kw['marker'] = ms_style; kw['markersize'] = ms_sz
            elif not ls: kw['marker'] = 'o'; kw['markersize'] = ms_sz
            ec = self._ec
            on_time = bool(ov.get('time_axis'))            # e.g. smoothed scans: x is the plot's time axis
            vs_E = on_time and ec is not None and self._ec_mode.currentIndex() == 1
            self._last_traces = []
            t_lo, t_hi = np.inf, -np.inf
            for tr in overlay_traces(ov):
                t = np.asarray(tr['x'], dtype=float)
                E_pts = potential_at(ec, t, self._ec_offset.value()) if (on_time and ec is not None) else None
                if t.size and np.isfinite(t).any():
                    t_lo, t_hi = min(t_lo, np.nanmin(t)), max(t_hi, np.nanmax(t))
                ax.plot(E_pts if vs_E else t, tr['y'], color=tr.get('color', '#2ca02c'), label=tr.get('label', 'data'),
                        linewidth=lw, **kw)
                if on_time:
                    self._last_traces.append({'label': tr.get('label', 'data'), 't': t, 'E': E_pts,
                                              'y': np.asarray(tr['y'], float), 'color': tr.get('color', '#2ca02c')})
            ref = self._ec_ref.text().strip() or "V"
            ax.set_xlabel(f"E ({ref})" if vs_E else ov.get('xlabel', ''), color=tc, fontsize=self._font_axis.value())
            if on_time and ec is not None and not vs_E:
                self._draw_potential_axis(ax, t_lo, t_hi, tc, lw, tick_dir)
            ax.set_ylabel(ov.get('ylabel', ''), color=tc, fontsize=self._font_axis.value())
            title = self._plot_title.text() or ov.get('title', '')
            ax.set_title(title, color=tc, fontsize=self._font_title.value())
            has_data = True
        else:
            # Normal scan data plot
            selected = self._get_selected_scans()
            if not selected:
                ax.set_title("Select a scan", color='#888')
                self._canvas.draw(); return
            x_col, y_cols, norm_col = self._col_sel.get_selection()
            if not x_col or not y_cols:
                ax.set_title("Select X and Y columns", color='#888')
                self._canvas.draw(); return

            att = self._get_att_factor()
            color_idx = 0
            multi_scan = len(selected) > 1
            ec = self._ec
            vs_E = ec is not None and self._ec_mode.currentIndex() == 1
            self._last_traces = []
            t_lo, t_hi = np.inf, -np.inf

            for sn in selected:
                scan = self._spec_scans.get(sn)
                if not scan: continue
                bgsub = self._bgsub_data.get(sn, {})
                def gd(c, _scan=scan, _bg=bgsub):
                    return _bg.get(c[5:]) if c.startswith('[bg] ') else _scan.data.get(c)

                x_data = gd(x_col)
                if x_data is None: continue
                norm_data = gd(norm_col) if norm_col else None

                # Use per-scan Y override if defined, otherwise global
                scan_y_cols = self._per_scan_y.get(sn, y_cols)

                for yc in scan_y_cols:
                    yd = gd(yc)
                    if yd is None: continue

                    # If mixing spec X with bgsub Y (or vice versa), align by image index
                    x_is_bg = x_col.startswith('[bg] ')
                    y_is_bg = yc.startswith('[bg] ')

                    if not x_is_bg and y_is_bg and 'image' in bgsub:
                        # Y from bgsub, X from spec: use bgsub image indices to pick spec rows
                        img_idx = bgsub['image'].astype(int)
                        valid = img_idx < len(x_data)
                        img_idx = img_idx[valid]
                        xd = x_data[img_idx].copy()
                        yd2 = yd[valid].copy()
                        if norm_data is not None:
                            nd = norm_data[img_idx].copy(); nd[nd == 0] = 1; yd2 /= nd
                        yd2 = self._apply_transm(scan, yd2, img_idx)
                    elif x_is_bg and not y_is_bg and 'image' in bgsub:
                        # X from bgsub, Y from spec: use bgsub image indices to pick spec rows
                        img_idx = bgsub['image'].astype(int)
                        valid = img_idx < len(yd)
                        img_idx = img_idx[valid]
                        xd = x_data[valid].copy()
                        yd2 = yd[img_idx].copy()
                        if norm_data is not None:
                            # norm could be from either source
                            if norm_col and norm_col.startswith('[bg] '):
                                nd = norm_data[valid].copy()
                            else:
                                nd = norm_data[img_idx].copy()
                            nd[nd == 0] = 1; yd2 /= nd
                        yd2 = self._apply_transm(scan, yd2, img_idx)
                    else:
                        # Both from same source: simple truncation
                        n = min(len(x_data), len(yd))
                        xd = x_data[:n].copy(); yd2 = yd[:n].copy()
                        if norm_data is not None:
                            nd = norm_data[:n].copy(); nd[nd == 0] = 1; yd2 /= nd
                        yd2 = self._apply_transm(scan, yd2, None)

                    if att != 1: yd2 *= att
                    xd = self._shift_x(np.asarray(xd, dtype=float), sn, x_col)
                    E_pts = potential_at(ec, xd, self._ec_offset.value()) if ec is not None else None
                    if xd.size:
                        t_lo, t_hi = min(t_lo, np.nanmin(xd)), max(t_hi, np.nanmax(xd))
                    t_plot = xd
                    if vs_E:
                        xd = E_pts
                    color = self._plot_colors[color_idx % len(self._plot_colors)]
                    col_label = display_label(yc.replace('[bg] ', ''), self._label_map)
                    if yc.startswith('[bg] '): col_label = f"[bg] {col_label}"
                    label = f"S{sn} {col_label}" if multi_scan else col_label
                    kw = {'linestyle': ls if ls else 'None'}
                    if show_m and ms_style: kw['marker'] = ms_style; kw['markersize'] = ms_sz
                    elif not ls: kw['marker'] = 'o'; kw['markersize'] = ms_sz
                    ax.plot(xd, yd2, color=color, label=label, linewidth=lw, **kw)
                    self._last_traces.append({'label': f"S{sn}_{yc.replace('[bg] ', '')}",
                                              't': t_plot, 'E': E_pts, 'y': yd2, 'color': color})
                    color_idx += 1
                    has_data = True

            xlab = display_label(x_col.replace('[bg] ', ''), self._label_map)
            if self._zero_mode.currentIndex() == 1:
                xlab = f"{xlab} from each scan's start"
            elif self._zero_mode.currentIndex() == 2:
                xlab = f"{xlab} from the start of S{selected[0]}"
            ref = self._ec_ref.text().strip() or "V"
            self._time_xlabel = xlab                    # the X label on the time axis (for smoothed traces)
            if vs_E:
                xlab = f"E ({ref})"
            ax.set_xlabel(xlab, color=tc, fontsize=self._font_axis.value())
            if ec is not None and not vs_E:
                self._draw_potential_axis(ax, t_lo, t_hi, tc, lw, tick_dir)
            ylabel = f"Intensity / {display_label(norm_col.replace('[bg] ',''), self._label_map)}" if norm_col else "Intensity"
            if self._chk_transm.isChecked():
                tcol = self._transm_col.text().strip() or "transm"
                ylabel = f"{ylabel} / {tcol}"
            ax.set_ylabel(ylabel, color=tc, fontsize=self._font_axis.value())
            if multi_scan:
                title = self._plot_title.text() or f"Scans {', '.join(str(s) for s in selected)}"
            else:
                sn = selected[0]; scan = self._spec_scans.get(sn)
                title = self._plot_title.text() or (f"Scan {sn}: {scan.scan_cmd}" if scan else "")
            ax.set_title(title, color=tc, fontsize=self._font_title.value())

        if not has_data:
            self._canvas.draw(); return

        if self._chk_logx.isChecked(): ax.set_xscale('log')
        if self._chk_logy.isChecked(): ax.set_yscale('log')

        def plim(t):
            t = t.strip()
            if not t or t.lower() == 'auto': return None
            try: return float(t)
            except: return None
        xmn, xmx = plim(self._xlim_min.text()), plim(self._xlim_max.text())
        ymn, ymx = plim(self._ylim_min.text()), plim(self._ylim_max.text())
        if xmn is not None or xmx is not None: ax.set_xlim(left=xmn, right=xmx)
        if ymn is not None or ymx is not None: ax.set_ylim(bottom=ymn, top=ymx)
        if self._chk_grid.isChecked():
            ax.grid(True, alpha=self._grid_alpha.value(), linestyle=self._grid_style.currentText(),
                    linewidth=self._grid_width.value(), color=tc)
        if self._chk_minor_ticks.isChecked():
            ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
        if self._chk_legend.isChecked():
            ax.legend(facecolor=self._axes_bg_color, edgecolor=tc, labelcolor=tc, fontsize=self._font_legend.value())
        fig.tight_layout(); self._canvas.draw()

    # ---- Label map ----
    def _add_label_mapping(self):
        frm = self._lbl_from.text().strip(); to = self._lbl_to.text().strip()
        if not frm: return
        if to: self._label_map[frm] = to
        else: self._label_map.pop(frm, None)
        save_label_map(self._label_map); self._refresh_label_map_display(); self._update_plot()

    def _refresh_label_map_display(self):
        self._lbl_map_info.setText(
            "Current: " + ", ".join(f"{k}→{v}" for k, v in sorted(self._label_map.items()))
            if self._label_map else "No mappings")

    # ---- Config ----
    def _save_plot_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save plot config", os.path.join(settings_dir(), "plot_config.json"), "JSON (*.json)")
        if not path: return
        x, y, n = self._col_sel.get_selection()
        config = {
            'spec_path': self._spec_path.text(), 'bgsub_path': self._bgsub_path.text(),
            'scans': self._get_selected_scans(), 'x_col': x, 'y_cols': y, 'norm_col': n,
            'log_x': self._chk_logx.isChecked(), 'log_y': self._chk_logy.isChecked(),
            'grid': self._chk_grid.isChecked(), 'legend': self._chk_legend.isChecked(),
            'markers': self._chk_markers.isChecked(),
            'line_style': self._line_combo.currentText(), 'marker_style': self._marker_combo.currentText(),
            'line_width': self._line_width.value(), 'marker_size': self._marker_size.value(),
            'title': self._plot_title.text(), 'att_trans': self._att_trans.value(),
            'att': [c.isChecked() for c in self._att],
            'fig_bg': self._fig_bg_color, 'axes_bg': self._axes_bg_color, 'text_color': self._text_color,
            'fig_w': self._fig_w.value(), 'fig_h': self._fig_h.value(), 'fig_dpi': self._fig_dpi.value(),
            'font_title': self._font_title.value(), 'font_axis': self._font_axis.value(),
            'font_tick': self._font_tick.value(), 'font_legend': self._font_legend.value(),
            'font_family': self._font_family.currentText(),
            'grid_style': self._grid_style.currentText(), 'grid_alpha': self._grid_alpha.value(),
            'grid_width': self._grid_width.value(), 'spine_width': self._spine_width.value(),
            'tick_width': self._tick_width.value(), 'tick_length': self._tick_length.value(),
            'tick_in': self._chk_tick_in.isChecked(), 'minor_ticks': self._chk_minor_ticks.isChecked(),
            'xlim_min': self._xlim_min.text(), 'xlim_max': self._xlim_max.text(),
            'ylim_min': self._ylim_min.text(), 'ylim_max': self._ylim_max.text(),
        }
        with open(path, 'w') as f: json.dump(config, f, indent=2)
        self.say(f"Config saved: {path}")

    def _load_plot_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", settings_dir(), "JSON (*.json)")
        if not path: return
        try:
            with open(path) as f: cfg = json.load(f)
        except Exception as e: self.say(str(e), error=True); return
        sp = cfg.get('spec_path', '')
        if sp and os.path.isfile(sp) and sp != self._spec_path.text():
            self._spec_path.setText(sp); self._load_spec_file(sp)
        bp = cfg.get('bgsub_path', '')
        if bp and os.path.isdir(bp) and bp != self._bgsub_path.text():
            self._bgsub_path.setText(bp); self._load_bgsub_dir(bp)
        saved_scans = cfg.get('scans', [])
        # Backward compat: old configs had 'scan' as a string
        if not saved_scans and 'scan' in cfg:
            try:
                sn = int(cfg['scan'].split(':')[0])
                saved_scans = [sn]
            except (ValueError, AttributeError):
                pass
        if saved_scans:
            self._scan_list.blockSignals(True)
            for i in range(self._scan_list.count()):
                item = self._scan_list.item(i)
                try:
                    sn = int(item.text().split(':')[0])
                    item.setSelected(sn in saved_scans)
                except ValueError:
                    pass
            self._scan_list.blockSignals(False)
        self._col_sel.set_selection(cfg.get('x_col'), cfg.get('y_cols', []), cfg.get('norm_col'))
        self._chk_logx.setChecked(cfg.get('log_x', False)); self._chk_logy.setChecked(cfg.get('log_y', False))
        self._chk_grid.setChecked(cfg.get('grid', True)); self._chk_legend.setChecked(cfg.get('legend', True))
        self._chk_markers.setChecked(cfg.get('markers', True))
        for attr, key, default in [('_line_combo', 'line_style', '-'), ('_marker_combo', 'marker_style', 'o'),
                                    ('_grid_style', 'grid_style', ':'), ('_font_family', 'font_family', 'DejaVu Sans')]:
            v = cfg.get(key, default); w = getattr(self, attr); i = w.findText(v)
            if i >= 0: w.setCurrentIndex(i)
        for attr, key, default in [('_line_width', 'line_width', 1.2), ('_marker_size', 'marker_size', 4),
                                    ('_grid_alpha', 'grid_alpha', 0.3), ('_grid_width', 'grid_width', 0.5),
                                    ('_spine_width', 'spine_width', 1), ('_tick_width', 'tick_width', 1),
                                    ('_tick_length', 'tick_length', 4), ('_fig_w', 'fig_w', 8), ('_fig_h', 'fig_h', 6)]:
            getattr(self, attr).setValue(cfg.get(key, default))
        self._fig_dpi.setValue(cfg.get('fig_dpi', 100))
        for attr, key, default in [('_font_title', 'font_title', 14), ('_font_axis', 'font_axis', 12),
                                    ('_font_tick', 'font_tick', 10), ('_font_legend', 'font_legend', 9)]:
            getattr(self, attr).setValue(cfg.get(key, default))
        self._chk_tick_in.setChecked(cfg.get('tick_in', False)); self._chk_minor_ticks.setChecked(cfg.get('minor_ticks', False))
        self._plot_title.setText(cfg.get('title', ''))
        self._xlim_min.setText(cfg.get('xlim_min', '')); self._xlim_max.setText(cfg.get('xlim_max', ''))
        self._ylim_min.setText(cfg.get('ylim_min', '')); self._ylim_max.setText(cfg.get('ylim_max', ''))
        self._att_trans.setValue(cfg.get('att_trans', 0.1))
        for i, v in enumerate(cfg.get('att', [False]*4)):
            if i < 4: self._att[i].setChecked(v)
        for attr, key, default in [('_fig_bg_color', 'fig_bg', 'white'), ('_axes_bg_color', 'axes_bg', 'white'),
                                    ('_text_color', 'text_color', '#222222')]:
            setattr(self, attr, cfg.get(key, default))
        self._fig_opts_dialog._sync_color_buttons(); self._update_plot()
        self.say(f"Config loaded: {path}")

    def _export_figure(self):
        """Open the export window on a snapshot of the working plot."""
        from .figure_export import FigureExportWindow, spec_from_canvas
        try:
            spec = spec_from_canvas(self._canvas.fig)
        except ValueError as e:
            self.say(f"Nothing to export: {e}", error=True)
            return None
        sel = self._get_selected_scans()
        name = ("S" + "_".join(str(s) for s in sel[:4])) if sel else "plot"
        if self._export_window is not None:
            self._export_window.close()
        self._export_window = FigureExportWindow(spec, lambda: spec_from_canvas(self._canvas.fig), self, name)
        self._export_window.show()
        self._export_window.raise_()
        return self._export_window

    def _save_plot(self, fmt):
        path, _ = QFileDialog.getSaveFileName(self, f"Save {fmt.upper()}", f"plot.{fmt}", f"{fmt.upper()} (*.{fmt})")
        if path:
            old_size = self._canvas.fig.get_size_inches()
            old_dpi = self._canvas.fig.get_dpi()
            self._canvas.fig.set_size_inches(self._fig_w.value(), self._fig_h.value())
            self._canvas.fig.savefig(path, dpi=self._fig_dpi.value(), facecolor=self._fig_bg_color, bbox_inches='tight')
            self._canvas.fig.set_size_inches(old_size)
            self._canvas.fig.set_dpi(old_dpi)
            self.say(f"Saved: {path}")

    def _export_plot_data(self):
        """Export all currently plotted traces as tab-separated text.
        Output format: H K L y [x_orig]
        If X column is H, K, or L, it's used for that axis (not duplicated).
        Otherwise X is appended at the end."""
        # Check overlay first
        if self._overlay is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Export plot data", "plot_data.txt", "Text (*.txt)")
            if not path: return
            with open(path, 'w') as f:
                traces = overlay_traces(self._overlay)
                for tr in traces:
                    if len(traces) > 1:
                        f.write(f"\n# Trace: {tr.get('label', 'data')}\n")
                    f.write(f"{self._overlay.get('xlabel','x')}\t{self._overlay.get('ylabel','y')}\n")
                    for xi, yi in zip(tr['x'], tr['y']):
                        f.write(f"{xi:.6g}\t{yi:.6g}\n")
            self.say(f"Exported overlay to {path}")
            return

        selected = self._get_selected_scans()
        if not selected: return
        x_col, y_cols, norm_col = self._col_sel.get_selection()
        if not x_col: return
        att = self._get_att_factor()

        def find_hkl_col(scan, target):
            """Find H, K, or L column in scan data (case-insensitive)."""
            for c in scan.columns:
                if c.strip().upper() == target:
                    return c
            return None

        # Collect all traces with H, K, L
        traces = []
        for sn in selected:
            scan = self._spec_scans.get(sn)
            if not scan: continue
            bgsub = self._bgsub_data.get(sn, {})
            def gd(c, _scan=scan, _bg=bgsub):
                return _bg.get(c[5:]) if c.startswith('[bg] ') else _scan.data.get(c)

            # Find H, K, L in spec data
            h_col = find_hkl_col(scan, 'H')
            k_col = find_hkl_col(scan, 'K')
            l_col = find_hkl_col(scan, 'L')
            h_spec = scan.data.get(h_col) if h_col else None
            k_spec = scan.data.get(k_col) if k_col else None
            l_spec = scan.data.get(l_col) if l_col else None

            x_data = gd(x_col)
            if x_data is None: continue
            norm_data = gd(norm_col) if norm_col else None
            scan_y_cols = self._per_scan_y.get(sn, y_cols)

            for yc in scan_y_cols:
                yd = gd(yc)
                if yd is None: continue
                x_is_bg = x_col.startswith('[bg] ')
                y_is_bg = yc.startswith('[bg] ')
                if not x_is_bg and y_is_bg and 'image' in bgsub:
                    img_idx = bgsub['image'].astype(int)
                    valid = img_idx < len(x_data)
                    img_idx = img_idx[valid]
                    xd = x_data[img_idx].copy(); yd2 = yd[valid].copy()
                    # Index HKL by img_idx
                    h = h_spec[img_idx].copy() if h_spec is not None else None
                    k = k_spec[img_idx].copy() if k_spec is not None else None
                    l = l_spec[img_idx].copy() if l_spec is not None else None
                    if norm_data is not None:
                        nd = norm_data[img_idx].copy(); nd[nd == 0] = 1; yd2 /= nd
                    yd2 = self._apply_transm(scan, yd2, img_idx)
                elif x_is_bg and not y_is_bg and 'image' in bgsub:
                    img_idx = bgsub['image'].astype(int)
                    valid = img_idx < len(yd)
                    img_idx = img_idx[valid]
                    xd = x_data[valid].copy(); yd2 = yd[img_idx].copy()
                    # HKL uses valid mask (spec row indices = valid positions)
                    h = h_spec[valid].copy() if h_spec is not None else None
                    k = k_spec[valid].copy() if k_spec is not None else None
                    l = l_spec[valid].copy() if l_spec is not None else None
                    if norm_data is not None:
                        if norm_col and norm_col.startswith('[bg] '):
                            nd = norm_data[valid].copy()
                        else:
                            nd = norm_data[img_idx].copy()
                        nd[nd == 0] = 1; yd2 /= nd
                    yd2 = self._apply_transm(scan, yd2, img_idx)
                else:
                    n = min(len(x_data), len(yd))
                    xd = x_data[:n].copy(); yd2 = yd[:n].copy()
                    h = h_spec[:n].copy() if h_spec is not None else None
                    k = k_spec[:n].copy() if k_spec is not None else None
                    l = l_spec[:n].copy() if l_spec is not None else None
                    if norm_data is not None:
                        nd = norm_data[:n].copy(); nd[nd == 0] = 1; yd2 /= nd
                    yd2 = self._apply_transm(scan, yd2, None)
                if att != 1: yd2 *= att
                col_label = yc.replace('[bg] ', '')
                traces.append({
                    'label': f"S{sn}_{col_label}",
                    'scan': sn, 'y_col': yc,
                    'x': xd, 'y': yd2,
                    'H': h, 'K': k, 'L': l})

        if not traces:
            self.say("Nothing plotted to export", error=True); return

        path, _ = QFileDialog.getSaveFileName(self, "Export plot data", "plot_data.txt", "Text (*.txt)")
        if not path: return

        x_label_upper = x_col.strip().upper().replace('[BG] ', '')
        x_is_hkl = x_label_upper in ('H', 'K', 'L')

        with open(path, 'w') as f:
            f.write(f"# Exported from SXRD Plotter\n")
            f.write(f"# X column: {x_col}\n")
            if norm_col: f.write(f"# Normalized by: {norm_col}\n")
            if self._per_scan_y:
                f.write(f"# Per-scan Y overrides:\n")
                for sn, ylist in sorted(self._per_scan_y.items()):
                    f.write(f"#   Scan {sn}: {', '.join(ylist)}\n")
            f.write(f"# Traces: {len(traces)}\n")

            # Each trace as a block
            for t in traces:
                f.write(f"\n# Trace: {t['label']} (scan={t['scan']}, y={t['y_col']})\n")
                has_hkl = t['H'] is not None and t['K'] is not None and t['L'] is not None
                if has_hkl and x_is_hkl:
                    # H K L y  (X is one of H/K/L, already included)
                    f.write(f"H\tK\tL\t{t['label']}\n")
                    for h, k, l, yi in zip(t['H'], t['K'], t['L'], t['y']):
                        f.write(f"{h:.6g}\t{k:.6g}\t{l:.6g}\t{yi:.6g}\n")
                elif has_hkl:
                    # H K L y x_orig  (X is something else, append at end)
                    f.write(f"H\tK\tL\t{t['label']}\t{x_col}\n")
                    for h, k, l, yi, xi in zip(t['H'], t['K'], t['L'], t['y'], t['x']):
                        f.write(f"{h:.6g}\t{k:.6g}\t{l:.6g}\t{yi:.6g}\t{xi:.6g}\n")
                else:
                    # Fallback: no HKL available in spec, write X y
                    f.write(f"# WARNING: H/K/L not found in spec for this scan\n")
                    f.write(f"{x_col}\t{t['label']}\n")
                    for xi, yi in zip(t['x'], t['y']):
                        f.write(f"{xi:.6g}\t{yi:.6g}\n")

        self.say(f"Exported {len(traces)} traces to {path}")

    def _export_genx(self):
        """Export currently plotted data in GenX sls_sxrd format.
        Format: H  K  L  I  Ierr  (whitespace-separated)
        Each trace (scan × y_col) is written as a rod with constant H,K.

        A dialog asks for H,K per trace and optional Bragg peak info for the
        beta roughness model (LB, dL columns)."""

        # Build traces the same way as _export_plot_data
        if self._overlay is not None:
            # Overlay: single trace, ask for HK
            traces = [{'label': tr.get('label', 'overlay'), 'x': tr['x'], 'y': tr['y']}
                      for tr in overlay_traces(self._overlay)]
        else:
            selected = self._get_selected_scans()
            if not selected: return
            x_col, y_cols, norm_col = self._col_sel.get_selection()
            if not x_col: return
            att = self._get_att_factor()
            traces = []
            for sn in selected:
                scan = self._spec_scans.get(sn)
                if not scan: continue
                bgsub = self._bgsub_data.get(sn, {})
                def gd(c, _scan=scan, _bg=bgsub):
                    return _bg.get(c[5:]) if c.startswith('[bg] ') else _scan.data.get(c)
                x_data = gd(x_col)
                if x_data is None: continue
                norm_data = gd(norm_col) if norm_col else None
                scan_y_cols = self._per_scan_y.get(sn, y_cols)
                for yc in scan_y_cols:
                    yd = gd(yc)
                    if yd is None: continue
                    x_is_bg = x_col.startswith('[bg] ')
                    y_is_bg = yc.startswith('[bg] ')
                    if not x_is_bg and y_is_bg and 'image' in bgsub:
                        img_idx = bgsub['image'].astype(int)
                        valid = img_idx < len(x_data)
                        img_idx = img_idx[valid]
                        xd = x_data[img_idx].copy(); yd2 = yd[valid].copy()
                        if norm_data is not None:
                            nd = norm_data[img_idx].copy(); nd[nd == 0] = 1; yd2 /= nd
                        yd2 = self._apply_transm(scan, yd2, img_idx)
                    elif x_is_bg and not y_is_bg and 'image' in bgsub:
                        img_idx = bgsub['image'].astype(int)
                        valid = img_idx < len(yd)
                        img_idx = img_idx[valid]
                        xd = x_data[valid].copy(); yd2 = yd[img_idx].copy()
                        if norm_data is not None:
                            if norm_col and norm_col.startswith('[bg] '):
                                nd = norm_data[valid].copy()
                            else:
                                nd = norm_data[img_idx].copy()
                            nd[nd == 0] = 1; yd2 /= nd
                        yd2 = self._apply_transm(scan, yd2, img_idx)
                    else:
                        n = min(len(x_data), len(yd))
                        xd = x_data[:n].copy(); yd2 = yd[:n].copy()
                        if norm_data is not None:
                            nd = norm_data[:n].copy(); nd[nd == 0] = 1; yd2 /= nd
                        yd2 = self._apply_transm(scan, yd2, None)
                    if att != 1: yd2 *= att
                    traces.append({'label': f"S{sn}_{yc.replace('[bg] ','')}", 'scan': sn,
                                   'x': xd, 'y': yd2})

        if not traces:
            self.say("Nothing plotted to export", error=True); return

        # Dialog: ask for HK per trace and optional beta roughness columns
        dlg = QDialog(self)
        dlg.setWindowTitle("GenX Export Settings")
        dlg.setMinimumWidth(450)
        vl = QVBoxLayout(dlg)
        vl.addWidget(QLabel("Assign H, K indices per trace.\n"
                            "X axis is used as L. Intensity error estimated as sqrt(I)."))

        tbl = QTableWidget(len(traces), 4)
        tbl.setHorizontalHeaderLabels(["Trace", "H", "K", "Include"])
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        hk_spins = []
        include_chks = []
        for i, t in enumerate(traces):
            tbl.setItem(i, 0, QTableWidgetItem(t['label']))
            h_sb = QDoubleSpinBox(); h_sb.setRange(-20, 20); h_sb.setValue(0); h_sb.setDecimals(1)
            k_sb = QDoubleSpinBox(); k_sb.setRange(-20, 20); k_sb.setValue(0); k_sb.setDecimals(1)
            tbl.setCellWidget(i, 1, h_sb)
            tbl.setCellWidget(i, 2, k_sb)
            chk = QCheckBox(); chk.setChecked(True)
            tbl.setCellWidget(i, 3, chk)
            hk_spins.append((h_sb, k_sb))
            include_chks.append(chk)
        vl.addWidget(tbl)

        # Beta roughness extras
        chk_beta = QCheckBox("Include LB and dL columns (for beta roughness model)")
        vl.addWidget(chk_beta)
        beta_row = QHBoxLayout()
        beta_row.addWidget(QLabel("dL (Bragg spacing):"))
        dl_spin = QDoubleSpinBox(); dl_spin.setRange(0.01, 100); dl_spin.setValue(1.0); dl_spin.setDecimals(3)
        beta_row.addWidget(dl_spin)
        vl.addLayout(beta_row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vl.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export GenX data", "rods.dat", "Data (*.dat *.txt);;All (*)")
        if not path: return

        include_beta = chk_beta.isChecked()
        dl_val = dl_spin.value()

        with open(path, 'w') as f:
            if include_beta:
                f.write("# GenX sls_sxrd format with beta roughness columns\n")
                f.write("# H  K  L  I  Ierr  LB  dL\n")
            else:
                f.write("# GenX sls_sxrd format\n")
                f.write("# H  K  L  I  Ierr\n")

            for i, t in enumerate(traces):
                if not include_chks[i].isChecked():
                    continue
                h_val = hk_spins[i][0].value()
                k_val = hk_spins[i][1].value()
                x = t['x']; y = t['y']
                # Sort by L
                sort_idx = np.argsort(x)
                x = x[sort_idx]; y = y[sort_idx]
                # Intensity error: sqrt(I) for counting statistics, minimum 1
                yerr = np.sqrt(np.maximum(np.abs(y), 1.0))

                for j in range(len(x)):
                    L_val = x[j]; I_val = y[j]; Ie_val = yerr[j]
                    if include_beta:
                        # LB = nearest integer L (Bragg peak position)
                        LB = round(L_val / dl_val) * dl_val
                        f.write(f"{h_val:.3e}  {k_val:.3e}  {L_val:.3e}  "
                                f"{I_val:.3e}  {Ie_val:.3e}  {LB:.3e}  {dl_val:.3e}\n")
                    else:
                        f.write(f"{h_val:.3e}  {k_val:.3e}  {L_val:.3e}  "
                                f"{I_val:.3e}  {Ie_val:.3e}\n")

        n_exported = sum(1 for c in include_chks if c.isChecked())
        self.say(f"Exported {n_exported} rods to {path} (GenX format)")


def main(argv=None):
    return run_tool(MainWindow)


if __name__ == '__main__':
    main()

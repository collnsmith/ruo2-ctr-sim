"""SXRD HKL Smoother
=================
Standalone tool to apply 1D smoothing to HKL-format files.

Input: tab-separated files with columns H K L I (the format produced by
the SXRD plotter Export Plot Data button, also accepted by the appender's
GenX Export tab). Multi-trace files are supported via "# Trace:" comment
markers; each trace is smoothed independently.

Output: same HKL format with the intensity column replaced by the smoothed
values. Excluded points (Bragg zones) are written as NaN by default, or
optionally as the original (un-smoothed) value.

Smoothing methods: Savitzky-Golay, Moving average, Gaussian kernel,
Lowess, Spline. Multiple Bragg exclusion zones can be defined per trace
or applied globally (e.g. integer L).

Qt window of the data tools (ctr_data_tools.py); the logic is in ctrfit.datatools.
"""
import os

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
                             QPushButton, QFileDialog, QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox,
                             QMessageBox, QCheckBox, QListWidget, QSplitter, QScrollArea)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ctrfit.app.datatools.common import run_tool
from ctrfit.datatools.hkl import parse_hkl_file, smooth_trace


class HKLSmoother(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SXRD HKL Smoother")
        self.resize(1300, 800)

        self._files = []
        self._traces = []  # list of trace dicts (from parse_hkl_file)
        self._smoothed_traces = {}  # label -> smoothed_y array
        self._exclusions_per_trace = {}  # label -> list of (xmin, xmax)
        self._current_trace_idx = -1
        self._click_points = []

        cw = QWidget()
        self.setCentralWidget(cw)
        outer = QHBoxLayout(cw)

        # Left: controls
        ctrl = self._build_controls()
        outer.addWidget(ctrl)

        # Right: trace list + plot
        right = QSplitter(Qt.Vertical)
        right.addWidget(self._build_trace_list())
        right.addWidget(self._build_plot_area())
        right.setSizes([180, 620])
        outer.addWidget(right, stretch=1)

    # ---- UI building ----
    def _build_controls(self):
        ctrl = QWidget()
        ctrl.setMaximumWidth(360)
        cv = QVBoxLayout(ctrl)

        # Files
        fg = QGroupBox("Input files")
        fl = QVBoxLayout(fg)
        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(80)
        fl.addWidget(self._file_list)
        bh = QHBoxLayout()
        b_add = QPushButton("Add files...")
        b_add.clicked.connect(self._add_files)
        bh.addWidget(b_add)
        b_clr = QPushButton("Clear")
        b_clr.clicked.connect(self._clear_files)
        bh.addWidget(b_clr)
        fl.addLayout(bh)
        cv.addWidget(fg)

        # Smoothing method
        sg = QGroupBox("Smoothing method")
        sl = QGridLayout(sg)
        sl.addWidget(QLabel("Method:"), 0, 0)
        self._method = QComboBox()
        self._method.addItems(["Savitzky-Golay", "Moving average",
                               "Gaussian kernel", "Lowess", "Spline"])
        self._method.currentTextChanged.connect(self._reapply_current)
        sl.addWidget(self._method, 0, 1)
        sl.addWidget(QLabel("Window/pts:"), 1, 0)
        self._window_size = QSpinBox()
        self._window_size.setRange(3, 999)
        self._window_size.setValue(11)
        self._window_size.setSingleStep(2)
        self._window_size.editingFinished.connect(self._reapply_current)
        sl.addWidget(self._window_size, 1, 1)
        sl.addWidget(QLabel("Poly order:"), 2, 0)
        self._poly_order = QSpinBox()
        self._poly_order.setRange(1, 10)
        self._poly_order.setValue(2)
        self._poly_order.editingFinished.connect(self._reapply_current)
        sl.addWidget(self._poly_order, 2, 1)
        cv.addWidget(sg)

        # Smooth in log space
        self._chk_log_space = QCheckBox("Smooth in log space")
        self._chk_log_space.setToolTip(
            "Take log10 of intensity, smooth, then exponentiate. "
            "Better for CTR data spanning many orders of magnitude.")
        self._chk_log_space.setChecked(True)
        self._chk_log_space.toggled.connect(self._reapply_current)
        cv.addWidget(self._chk_log_space)

        # Exclusion zones
        eg = QGroupBox("Bragg exclusion zones (current trace)")
        el = QVBoxLayout(eg)

        # Auto-add at integer L
        auto_row = QHBoxLayout()
        auto_row.addWidget(QLabel("Auto-exclude integer L ±"))
        self._auto_halfwidth = QDoubleSpinBox()
        self._auto_halfwidth.setRange(0.001, 1.0)
        self._auto_halfwidth.setDecimals(3)
        self._auto_halfwidth.setSingleStep(0.01)
        self._auto_halfwidth.setValue(0.05)
        auto_row.addWidget(self._auto_halfwidth)
        b_auto = QPushButton("Add")
        b_auto.clicked.connect(self._auto_add_bragg)
        auto_row.addWidget(b_auto)
        el.addLayout(auto_row)

        # Manual zone
        m_row = QGridLayout()
        m_row.addWidget(QLabel("X min:"), 0, 0)
        self._exc_min = QDoubleSpinBox()
        self._exc_min.setRange(-1e6, 1e6)
        self._exc_min.setDecimals(4)
        m_row.addWidget(self._exc_min, 0, 1)
        m_row.addWidget(QLabel("X max:"), 0, 2)
        self._exc_max = QDoubleSpinBox()
        self._exc_max.setRange(-1e6, 1e6)
        self._exc_max.setDecimals(4)
        m_row.addWidget(self._exc_max, 0, 3)
        el.addLayout(m_row)

        b_row = QHBoxLayout()
        b_add_zone = QPushButton("Add zone")
        b_add_zone.clicked.connect(self._add_exclusion)
        b_row.addWidget(b_add_zone)
        b_clear_zones = QPushButton("Clear all")
        b_clear_zones.clicked.connect(self._clear_exclusions)
        b_row.addWidget(b_clear_zones)
        el.addLayout(b_row)

        self._chk_click_exc = QCheckBox("Click-to-define (2 clicks per zone)")
        el.addWidget(self._chk_click_exc)

        self._exc_list = QListWidget()
        self._exc_list.setMaximumHeight(80)
        el.addWidget(self._exc_list)
        b_remove = QPushButton("Remove selected zone")
        b_remove.clicked.connect(self._remove_exclusion)
        el.addWidget(b_remove)

        # Apply same zones to all traces
        b_propagate = QPushButton("Apply current zones to all traces")
        b_propagate.setToolTip(
            "Copy the exclusion zone list from the current trace to every "
            "other loaded trace.")
        b_propagate.clicked.connect(self._propagate_zones)
        el.addWidget(b_propagate)

        cv.addWidget(eg)

        # Display
        dg = QGroupBox("Display")
        dl = QHBoxLayout(dg)
        self._chk_log_y = QCheckBox("Log Y")
        self._chk_log_y.setChecked(True)
        self._chk_log_y.toggled.connect(self._update_plot)
        dl.addWidget(self._chk_log_y)
        cv.addWidget(dg)

        # Export
        xg = QGroupBox("Export")
        xl = QVBoxLayout(xg)
        self._exc_handling = QComboBox()
        self._exc_handling.addItems(["NaN at excluded points",
                                      "Original (un-smoothed) at excluded points",
                                      "Skip excluded points entirely"])
        xl.addWidget(QLabel("Excluded point handling:"))
        xl.addWidget(self._exc_handling)
        b_export = QPushButton("Export smoothed (current trace)")
        b_export.clicked.connect(self._export_current)
        xl.addWidget(b_export)
        b_export_all = QPushButton("Export all traces")
        b_export_all.clicked.connect(self._export_all)
        xl.addWidget(b_export_all)
        cv.addWidget(xg)

        cv.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(ctrl)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(380)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        return scroll

    def _build_trace_list(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        v.addWidget(QLabel("Traces (click to select):"))
        self._trace_list = QListWidget()
        self._trace_list.currentRowChanged.connect(self._on_trace_selected)
        v.addWidget(self._trace_list)
        return w

    def _build_plot_area(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        self._fig = Figure(figsize=(7, 5), dpi=100, facecolor='#353535')
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor('#2a2a2a')
        nav = NavigationToolbar(self._canvas, w)
        v.addWidget(nav)
        v.addWidget(self._canvas, stretch=1)
        self._canvas.mpl_connect('button_press_event', self._on_canvas_click)
        return w

    # ---- File loading ----
    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add HKL files", "", "Text (*.txt *.dat);;All (*)")
        for p in paths:
            if p in self._files:
                continue
            try:
                new_traces = parse_hkl_file(p)
            except Exception as e:
                QMessageBox.warning(self, "Parse error",
                                    f"{os.path.basename(p)}: {e}")
                continue
            if not new_traces:
                continue
            # Tag traces with source file (for multi-file output)
            for t in new_traces:
                t['_source_file'] = p
            self._files.append(p)
            self._file_list.addItem(os.path.basename(p))
            for t in new_traces:
                self._traces.append(t)
                self._exclusions_per_trace[t['label']] = []
                self._trace_list.addItem(
                    f"{t['label']}  ({len(t['L'])} pts, "
                    f"H={t['H'][0]:.3g}, K={t['K'][0]:.3g}, "
                    f"L=[{t['L'].min():.3g},{t['L'].max():.3g}])")
        if self._traces and self._current_trace_idx < 0:
            self._trace_list.setCurrentRow(0)

    def _clear_files(self):
        self._files.clear()
        self._traces.clear()
        self._smoothed_traces.clear()
        self._exclusions_per_trace.clear()
        self._file_list.clear()
        self._trace_list.clear()
        self._exc_list.clear()
        self._current_trace_idx = -1
        self._update_plot()

    # ---- Trace selection ----
    def _on_trace_selected(self, row):
        if not (0 <= row < len(self._traces)):
            return
        self._current_trace_idx = row
        # Refresh exclusion list display
        self._refresh_exc_list()
        self._click_points.clear()
        self._reapply_current()

    def _current_trace(self):
        if 0 <= self._current_trace_idx < len(self._traces):
            return self._traces[self._current_trace_idx]
        return None

    def _refresh_exc_list(self):
        self._exc_list.clear()
        t = self._current_trace()
        if t is None:
            return
        for xmin, xmax in self._exclusions_per_trace.get(t['label'], []):
            self._exc_list.addItem(f"{xmin:.4g} — {xmax:.4g}")

    # ---- Exclusion zones ----
    def _add_exclusion(self):
        t = self._current_trace()
        if t is None:
            return
        xmin = self._exc_min.value()
        xmax = self._exc_max.value()
        if xmin >= xmax:
            QMessageBox.warning(self, "Error", "X min must be < X max")
            return
        self._exclusions_per_trace.setdefault(t['label'], []).append(
            (xmin, xmax))
        self._refresh_exc_list()
        self._reapply_current()

    def _remove_exclusion(self):
        t = self._current_trace()
        if t is None:
            return
        idx = self._exc_list.currentRow()
        zones = self._exclusions_per_trace.get(t['label'], [])
        if 0 <= idx < len(zones):
            zones.pop(idx)
            self._refresh_exc_list()
            self._reapply_current()

    def _clear_exclusions(self):
        t = self._current_trace()
        if t is None:
            return
        self._exclusions_per_trace[t['label']] = []
        self._click_points.clear()
        self._refresh_exc_list()
        self._reapply_current()

    def _auto_add_bragg(self):
        """Add exclusion zones at every integer L within the trace's L range."""
        t = self._current_trace()
        if t is None:
            return
        hw = self._auto_halfwidth.value()
        L_min = float(t['L'].min())
        L_max = float(t['L'].max())
        first = int(np.ceil(L_min - 0.5))
        last = int(np.floor(L_max + 0.5))
        zones = self._exclusions_per_trace.setdefault(t['label'], [])
        added = 0
        for L_int in range(first, last + 1):
            if L_int < L_min - hw or L_int > L_max + hw:
                continue
            zone = (L_int - hw, L_int + hw)
            if zone not in zones:
                zones.append(zone)
                added += 1
        self._refresh_exc_list()
        self._reapply_current()

    def _propagate_zones(self):
        """Copy current trace's zones to all other traces."""
        t = self._current_trace()
        if t is None:
            return
        zones = list(self._exclusions_per_trace.get(t['label'], []))
        n = 0
        for other in self._traces:
            if other['label'] == t['label']:
                continue
            self._exclusions_per_trace[other['label']] = list(zones)
            n += 1
        QMessageBox.information(
            self, "Done", f"Copied {len(zones)} zone(s) to {n} other trace(s)")

    def _on_canvas_click(self, event):
        if not self._chk_click_exc.isChecked():
            return
        if event.inaxes != self._ax or event.button != 1:
            return
        if event.xdata is None:
            return
        t = self._current_trace()
        if t is None:
            return
        self._click_points.append(event.xdata)
        if len(self._click_points) == 2:
            xmin = min(self._click_points)
            xmax = max(self._click_points)
            self._exclusions_per_trace.setdefault(t['label'], []).append(
                (xmin, xmax))
            self._exc_min.setValue(xmin)
            self._exc_max.setValue(xmax)
            self._click_points.clear()
            self._refresh_exc_list()
            self._reapply_current()
        else:
            self._ax.axvline(event.xdata, color='#ff6b6b',
                             linestyle=':', alpha=0.7)
            self._canvas.draw()

    # ---- Smoothing ----
    def _reapply_current(self):
        t = self._current_trace()
        if t is None:
            self._update_plot()
            return
        x = t['L']
        y = t['I']
        zones = self._exclusions_per_trace.get(t['label'], [])
        method = self._method.currentText()
        win = self._window_size.value()
        poly = self._poly_order.value()

        if self._chk_log_space.isChecked():
            # log10 in safe regions, smooth, exp back
            with np.errstate(divide='ignore', invalid='ignore'):
                y_for_smooth = np.where(y > 0, np.log10(np.maximum(y, 1e-30)),
                                         np.nan)
            sm_log = smooth_trace(x, y_for_smooth, method, win, poly, zones)
            smoothed = np.power(10.0, sm_log)
        else:
            smoothed = smooth_trace(x, y, method, win, poly, zones)

        self._smoothed_traces[t['label']] = smoothed
        self._update_plot()

    def _update_plot(self):
        self._ax.clear()
        self._ax.set_facecolor('#2a2a2a')
        self._ax.tick_params(colors='#ddd')
        for sp in self._ax.spines.values():
            sp.set_color('#555')
        self._ax.grid(True, alpha=0.2, color='#555')

        t = self._current_trace()
        if t is None:
            self._ax.set_title("Load HKL files and select a trace",
                               color='#888')
            self._canvas.draw()
            return

        x = t['L']
        y = t['I']

        # Exclusion zone shading
        for xmin, xmax in self._exclusions_per_trace.get(t['label'], []):
            self._ax.axvspan(xmin, xmax, alpha=0.15, color='#ff6b6b')

        # Original
        self._ax.plot(x, y, 'o', color='#2a9fd6', markersize=3,
                      alpha=0.5, label='original')
        # Smoothed
        sm = self._smoothed_traces.get(t['label'])
        if sm is not None:
            valid = ~np.isnan(sm) & np.isfinite(sm) & (sm > 0 if self._chk_log_y.isChecked() else np.ones_like(sm, dtype=bool))
            self._ax.plot(x[valid], sm[valid], '-', color='#ffd43b',
                          linewidth=1.5, label='smoothed')

        if self._chk_log_y.isChecked():
            try:
                self._ax.set_yscale('log')
            except Exception:
                pass
        else:
            self._ax.set_yscale('linear')

        self._ax.set_xlabel('L', color='#ddd')
        self._ax.set_ylabel('I', color='#ddd')
        self._ax.set_title(t['label'], color='#ddd')
        leg = self._ax.legend(facecolor='#353535', edgecolor='#555')
        for txt in leg.get_texts():
            txt.set_color('#ddd')
        self._fig.tight_layout()
        self._canvas.draw()

    # ---- Export ----
    def _build_export_rows(self, trace, handling):
        """Return list of (H, K, L, I_out) for output, applying excluded
        point handling rule."""
        x = trace['L']
        y = trace['I']
        H = trace['H']
        K = trace['K']
        sm = self._smoothed_traces.get(trace['label'])
        zones = self._exclusions_per_trace.get(trace['label'], [])
        excluded_mask = np.zeros(len(x), dtype=bool)
        for xmin, xmax in zones:
            excluded_mask |= (x >= xmin) & (x <= xmax)

        rows = []
        for i in range(len(x)):
            if sm is None or np.isnan(sm[i]) or not np.isfinite(sm[i]):
                # Smoother failed at this point
                if handling == "Skip excluded points entirely":
                    if excluded_mask[i]:
                        continue
                    rows.append((H[i], K[i], x[i], float('nan')))
                elif handling == "Original (un-smoothed) at excluded points":
                    rows.append((H[i], K[i], x[i], y[i]))
                else:  # NaN
                    rows.append((H[i], K[i], x[i], float('nan')))
                continue
            if excluded_mask[i]:
                if handling == "NaN at excluded points":
                    rows.append((H[i], K[i], x[i], float('nan')))
                elif handling == "Original (un-smoothed) at excluded points":
                    rows.append((H[i], K[i], x[i], y[i]))
                else:  # Skip
                    continue
            else:
                rows.append((H[i], K[i], x[i], sm[i]))
        return rows

    def _write_hkl_file(self, path, traces_to_write, handling):
        """Write one or more traces to a single HKL file in the same format
        as the plotter's export."""
        with open(path, 'w') as f:
            f.write("# Smoothed by SXRD HKL Smoother\n")
            f.write(f"# Method: {self._method.currentText()}\n")
            f.write(f"# Window: {self._window_size.value()}, "
                    f"poly order: {self._poly_order.value()}\n")
            f.write(f"# Log-space smoothing: {self._chk_log_space.isChecked()}\n")
            f.write(f"# Excluded point handling: {handling}\n")
            f.write(f"# Traces: {len(traces_to_write)}\n\n")
            for t in traces_to_write:
                rows = self._build_export_rows(t, handling)
                zones = self._exclusions_per_trace.get(t['label'], [])
                f.write(f"# Trace: {t['label']}\n")
                if zones:
                    f.write(f"# Exclusion zones: "
                            f"{', '.join(f'[{a:.4g},{b:.4g}]' for a,b in zones)}\n")
                f.write(f"H\tK\tL\t{t['label']}\n")
                for H, K, L, I in rows:
                    if I != I:  # NaN
                        I_str = "nan"
                    else:
                        I_str = f"{I:.6g}"
                    f.write(f"{H:.6g}\t{K:.6g}\t{L:.6g}\t{I_str}\n")
                f.write("\n")

    def _export_current(self):
        t = self._current_trace()
        if t is None:
            QMessageBox.warning(self, "Error", "No trace selected")
            return
        if t['label'] not in self._smoothed_traces:
            QMessageBox.warning(self, "Error",
                                "Smoothing has not been computed for this trace")
            return
        suggested = os.path.splitext(t.get('_source_file',
                                           f"{t['label']}.txt"))[0] + "_smoothed.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export smoothed", suggested, "Text (*.txt *.dat);;All (*)")
        if not path:
            return
        try:
            self._write_hkl_file(path, [t], self._exc_handling.currentText())
            QMessageBox.information(self, "Done", f"Wrote {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _export_all(self):
        if not self._traces:
            QMessageBox.warning(self, "Error", "No traces loaded")
            return

        # Make sure every trace has been smoothed (apply current settings to
        # any not yet computed)
        method = self._method.currentText()
        win = self._window_size.value()
        poly = self._poly_order.value()
        log_space = self._chk_log_space.isChecked()
        for t in self._traces:
            if t['label'] in self._smoothed_traces:
                continue
            x = t['L']; y = t['I']
            zones = self._exclusions_per_trace.get(t['label'], [])
            if log_space:
                with np.errstate(divide='ignore', invalid='ignore'):
                    y_log = np.where(y > 0, np.log10(np.maximum(y, 1e-30)),
                                     np.nan)
                sm = np.power(10.0, smooth_trace(x, y_log, method, win, poly, zones))
            else:
                sm = smooth_trace(x, y, method, win, poly, zones)
            self._smoothed_traces[t['label']] = sm

        # Write per-input-file (preserves multi-trace structure of original)
        handling = self._exc_handling.currentText()
        wrote = 0
        for src_path in self._files:
            traces_for_file = [t for t in self._traces
                               if t.get('_source_file') == src_path]
            if not traces_for_file:
                continue
            base, ext = os.path.splitext(src_path)
            out_path = base + "_smoothed" + (ext or ".txt")
            try:
                self._write_hkl_file(out_path, traces_for_file, handling)
                wrote += 1
            except Exception as e:
                QMessageBox.warning(self, "Error",
                                    f"{os.path.basename(src_path)}: {e}")
        QMessageBox.information(self, "Done",
                                f"Wrote {wrote} smoothed file(s)")


def main(argv=None):
    return run_tool(HKLSmoother)


if __name__ == '__main__':
    main()

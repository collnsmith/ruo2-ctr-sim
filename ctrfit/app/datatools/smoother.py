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
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QSpinBox,
                             QDoubleSpinBox, QComboBox, QCheckBox, QListWidget, QSplitter)
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from ctrfit.app.datatools.common import StatusMixin, group, hint, primary_button, row, run_tool, sidebar
from ctrfit.datatools.hkl import parse_hkl_file, smooth_trace


class HKLSmoother(StatusMixin, QMainWindow):
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

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_controls())
        split.addWidget(self._build_plot_area())
        split.setSizes([400, 900])
        self.setCentralWidget(split)
        self.say("Add HKL files, pick a trace, set the exclusion zones, then export.")

    # ---- UI building ----
    def _build_controls(self):
        fg, fl = group("Files and traces", form=False)
        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(90)
        fl.addWidget(self._file_list)
        b_add, b_clr = QPushButton("Add files…"), QPushButton("Clear")
        b_add.clicked.connect(self._add_files)
        b_clr.clicked.connect(self._clear_files)
        fl.addWidget(row(b_add, b_clr, 1, stretch_first=False))
        fl.addWidget(hint("Traces (click one to edit its zones and see it):"))
        self._trace_list = QListWidget()
        self._trace_list.setMinimumHeight(120)
        self._trace_list.currentRowChanged.connect(self._on_trace_selected)
        fl.addWidget(self._trace_list)

        sg, sl = group("Smoothing")
        self._method = QComboBox()
        self._method.addItems(["Savitzky-Golay", "Moving average", "Gaussian kernel", "Lowess", "Spline"])
        self._method.currentTextChanged.connect(self._reapply_current)
        self._window_size = QSpinBox()
        self._window_size.setRange(3, 999)
        self._window_size.setValue(11)
        self._window_size.setSingleStep(2)
        self._window_size.setSuffix(" points")
        self._window_size.editingFinished.connect(self._reapply_current)
        self._poly_order = QSpinBox()
        self._poly_order.setRange(1, 10)
        self._poly_order.setValue(2)
        self._poly_order.setToolTip("Polynomial order (Savitzky-Golay), spline degree")
        self._poly_order.editingFinished.connect(self._reapply_current)
        self._chk_log_space = QCheckBox("Smooth log10(I)")
        self._chk_log_space.setToolTip("Take log10 of intensity, smooth, then exponentiate. "
                                       "Better for CTR data spanning many orders of magnitude.")
        self._chk_log_space.setChecked(True)
        self._chk_log_space.toggled.connect(self._reapply_current)
        sl.addRow("Method", self._method)
        sl.addRow("Window", self._window_size)
        sl.addRow("Order", self._poly_order)
        sl.addRow(self._chk_log_space)

        eg, el = group("Bragg exclusion zones (current trace)")
        self._auto_halfwidth = QDoubleSpinBox()
        self._auto_halfwidth.setRange(0.001, 1.0)
        self._auto_halfwidth.setDecimals(3)
        self._auto_halfwidth.setSingleStep(0.01)
        self._auto_halfwidth.setValue(0.05)
        self._auto_halfwidth.setPrefix("± ")
        b_auto = QPushButton("Add")
        b_auto.clicked.connect(self._auto_add_bragg)
        el.addRow("Integer L", row(self._auto_halfwidth, b_auto))
        self._exc_min = QDoubleSpinBox()
        self._exc_min.setRange(-1e6, 1e6)
        self._exc_min.setDecimals(4)
        self._exc_max = QDoubleSpinBox()
        self._exc_max.setRange(-1e6, 1e6)
        self._exc_max.setDecimals(4)
        b_add_zone = QPushButton("Add")
        b_add_zone.clicked.connect(self._add_exclusion)
        el.addRow("From, to", row(self._exc_min, self._exc_max, b_add_zone, stretch_first=False))
        self._chk_click_exc = QCheckBox("Click on the plot to add a zone (two clicks)")
        el.addRow(self._chk_click_exc)
        self._exc_list = QListWidget()
        self._exc_list.setMaximumHeight(90)
        el.addRow(self._exc_list)
        b_remove, b_clear_zones = QPushButton("Remove selected"), QPushButton("Clear all")
        b_remove.clicked.connect(self._remove_exclusion)
        b_clear_zones.clicked.connect(self._clear_exclusions)
        b_propagate = QPushButton("Copy to all traces")
        b_propagate.setToolTip("Copy the exclusion zone list from the current trace to every other loaded trace.")
        b_propagate.clicked.connect(self._propagate_zones)
        el.addRow(row(b_remove, b_clear_zones, 1, stretch_first=False))
        el.addRow(b_propagate)

        xg, xl = group("Export")
        self._exc_handling = QComboBox()
        self._exc_handling.addItems(["NaN at excluded points",
                                      "Original (un-smoothed) at excluded points",
                                      "Skip excluded points entirely"])
        xl.addRow("Excluded points", self._exc_handling)
        b_export = QPushButton("Export current trace…")
        b_export.clicked.connect(self._export_current)
        xl.addRow(b_export)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(sidebar(fg, sg, eg, xg), 1)
        lv.addWidget(primary_button("Export all traces…", self._export_all, "Ctrl+S", self))
        return left

    def _build_plot_area(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        self._fig = Figure(figsize=(7, 5), dpi=100, layout="constrained")
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        nav = NavigationToolbar(self._canvas, w)
        self._chk_log_y = QCheckBox("Log I")
        self._chk_log_y.setChecked(True)
        self._chk_log_y.toggled.connect(self._update_plot)
        v.addWidget(row(nav, self._chk_log_y))
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
                self.say(f"{os.path.basename(p)}: {e}", error=True)
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
            self.say("X min must be < X max", error=True)
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
        self.say(f"Copied {len(zones)} zone(s) to {n} other trace(s)")

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
            self._ax.axvline(event.xdata, color='#d62728',
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
        self._ax.set_facecolor('white')
        self._ax.tick_params(colors='#222')
        for sp in self._ax.spines.values():
            sp.set_color('#999')
        self._ax.grid(True, alpha=0.2, color='#999')

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
            self._ax.axvspan(xmin, xmax, alpha=0.15, color='#d62728')

        # Original
        self._ax.plot(x, y, 'o', color='#1f77b4', markersize=3,
                      alpha=0.5, label='original')
        # Smoothed
        sm = self._smoothed_traces.get(t['label'])
        if sm is not None:
            valid = ~np.isnan(sm) & np.isfinite(sm) & (sm > 0 if self._chk_log_y.isChecked() else np.ones_like(sm, dtype=bool))
            self._ax.plot(x[valid], sm[valid], '-', color='#ff7f0e',
                          linewidth=1.5, label='smoothed')

        if self._chk_log_y.isChecked():
            try:
                self._ax.set_yscale('log')
            except Exception:
                pass
        else:
            self._ax.set_yscale('linear')

        self._ax.set_xlabel('L', color='#222')
        self._ax.set_ylabel('I', color='#222')
        self._ax.set_title(t['label'], color='#222')
        leg = self._ax.legend(facecolor='white', edgecolor='#999')
        for txt in leg.get_texts():
            txt.set_color('#222')
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
            self.say("No trace selected", error=True)
            return
        if t['label'] not in self._smoothed_traces:
            self.say("Smoothing has not been computed for this trace", error=True)
            return
        suggested = os.path.splitext(t.get('_source_file',
                                           f"{t['label']}.txt"))[0] + "_smoothed.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export smoothed", suggested, "Text (*.txt *.dat);;All (*)")
        if not path:
            return
        try:
            self._write_hkl_file(path, [t], self._exc_handling.currentText())
            self.say(f"Wrote {path}")
        except Exception as e:
            self.say(str(e), error=True)

    def _export_all(self):
        if not self._traces:
            self.say("No traces loaded", error=True)
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
                self.say(f"{os.path.basename(src_path)}: {e}", error=True)
        self.say(f"Wrote {wrote} smoothed file(s)")


def main(argv=None):
    return run_tool(HKLSmoother)


if __name__ == '__main__':
    main()

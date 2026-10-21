"""SXRD Plotter HKL Appender
=========================
Adds H, K, L columns from a spec file to plotter-exported data files.

Input file format (from sxrd_plotter "Export plot data to text"):
    # Trace: S<scan>_<col> (scan=<n>, y=<col>)
    L   S<n>_<col>
    <x> <y>
    ...

Output format (per trace):
    # Trace: S<scan>_<col> (scan=<n>, y=<col>)
    H   K       L       <y_col>[        <X_orig>]
    <h> <k>     <l>     <y>[    <x>]

If the X axis is H, K, or L, that column is used as the matching column
and not duplicated. Otherwise X is appended at the end.

H, K, L are interpolated from the spec scan columns 'H', 'K', 'L'
(or 'h', 'k', 'l') against the matching axis (usually L from a hklscan).

Qt window of the data tools (ctr_data_tools.py); the logic is in ctrfit.datatools.
"""
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFormLayout, QSplitter, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton,
                             QFileDialog, QListWidget, QLineEdit, QTextEdit, QCheckBox, QTabWidget,
                             QDoubleSpinBox, QTableWidget, QTableWidgetItem, QComboBox, QHeaderView)

from ctrfit.app.datatools.common import (StatusMixin, group, hint, mono_font, primary_button, row, run_tool,
                                         sidebar)
from ctrfit.datatools.hkl import (parse_exported_file, parse_hkl_file, process_peakfit_file, write_genx_file,
                                  write_output)
from ctrfit.datatools.specfile import parse_spec_file


class MainWindow(StatusMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SXRD HKL Appender")
        self.resize(1200, 760)

        self._spec_path = None
        self._spec_scans = {}
        self._data_paths = []
        self._peakfit_paths = []
        self._gx_files = []
        self._gx_traces_cache = {}  # path -> list of traces

        # left: what every mode shares (the SPEC file, snapping of H K L)
        spec, sf = group("SPEC file")
        self._spec_label = QLineEdit()
        self._spec_label.setReadOnly(True)
        btn_spec = QPushButton("Browse…")
        btn_spec.clicked.connect(self._load_spec)
        sf.addRow(row(self._spec_label, btn_spec))
        sf.addRow(hint("H, K, L are interpolated from this file's scans (columns H K L) along the data's x axis."))

        snap, sn = group("Snap nearly constant H K L")
        sn.addRow(hint("If a column's spread is below the tolerance, every value becomes the nearest multiple "
                       "of the step (H = 1.975 → 2). Step 0 keeps the mean. Used by plotter exports and peak fits."))
        self._snap_widgets = {}
        for axis in ['H', 'K', 'L']:
            chk = QCheckBox(axis)
            chk.setChecked(axis in ('H', 'K'))
            tol = QDoubleSpinBox()
            tol.setRange(0.0, 1.0); tol.setSingleStep(0.01); tol.setValue(0.1); tol.setDecimals(3)
            tol.setPrefix("tol ")
            step = QDoubleSpinBox()
            step.setRange(0.0, 10.0); step.setSingleStep(0.5); step.setValue(1.0); step.setDecimals(2)
            step.setPrefix("step ")
            step.setToolTip("0 = use the mean, 1 = nearest integer, 0.5 = nearest half-integer")
            sn.addRow(row(chk, tol, step, stretch_first=False))
            self._snap_widgets[axis] = {'enabled': chk, 'tol': tol, 'step': step}
        self._pf_snap_widgets = self._snap_widgets

        # right: one tab per input kind, the log below
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_plotter_tab(), "Plotter exports")
        self._tabs.addTab(self._build_peakfit_tab(), "Peak fits")
        self._tabs.addTab(self._build_genx_tab(), "GenX export")
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(mono_font())
        right = QSplitter(Qt.Vertical)
        right.addWidget(self._tabs)
        right.addWidget(self._log)
        right.setSizes([520, 200])

        split = QSplitter(Qt.Horizontal)
        split.addWidget(sidebar(spec, snap))
        split.addWidget(right)
        split.setSizes([400, 800])
        self.setCentralWidget(split)
        self._log_msg("Ready. Load a SPEC file, then use one of the tabs.")
        self.say("Load a SPEC file, then add files in one of the tabs.")

    def _file_box(self, title, add, clear, extra=()):
        """A titled file list with Add / (extra) / Clear buttons; returns (box, list)."""
        box, lay = group(title, form=False)
        lst = QListWidget()
        lst.setMinimumHeight(120)
        lay.addWidget(lst, 1)
        b_add, b_clr = QPushButton("Add files…"), QPushButton("Clear")
        b_add.clicked.connect(add)
        b_clr.clicked.connect(clear)
        lay.addWidget(row(b_add, *extra, b_clr, 1, stretch_first=False))
        return box, lst

    def _build_plotter_tab(self):
        tab = QWidget()
        lv = QVBoxLayout(tab)
        lv.addWidget(hint("Files from the plotter's 'Export plot data' (one or more '# Trace:' blocks). "
                          "Each trace gets H K L columns from its SPEC scan."))
        box, self._file_list = self._file_box("Plotter export files", self._add_files, self._clear_files)
        lv.addWidget(box, 1)
        self._chk_inplace = QCheckBox("Overwrite the originals (otherwise <name>_hkl.txt next to each)")
        lv.addWidget(self._chk_inplace)
        lv.addWidget(primary_button("Add H K L", self._process))
        return tab

    def _build_peakfit_tab(self):
        tab = QWidget()
        lv = QVBoxLayout(tab)
        lv.addWidget(hint("Peak-fit tables from the plotter's peak fitting: the scan and an area column; H K L "
                          "are the scan means. One row per scan, written to <name>_hkl.txt."))
        box, self._pf_list = self._file_box("Peak fit files", self._pf_add_files, self._pf_clear_files)
        lv.addWidget(box, 1)
        self._pf_area_choice = QComboBox()
        self._pf_area_choice.setEditable(True)
        self._pf_area_choice.addItems(['area', 'fit_area', 'num_area'])
        self._pf_area_choice.setToolTip("Column of the peak fit file used as the intensity (any column name works)")
        form = QFormLayout()
        form.addRow("Area column", self._pf_area_choice)
        lv.addLayout(form)
        lv.addWidget(primary_button("Add H K L", self._pf_process))
        return tab

    def _log_msg(self, msg):
        self._log.append(msg)

    def _load_spec(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select spec file", "",
            "Spec files (*.spec *.dat *.txt);;All (*)")
        if not path:
            return
        try:
            self._spec_scans = parse_spec_file(path)
            self._spec_path = path
            self._spec_label.setText(path)
            self.say(f"{len(self._spec_scans)} scans in {os.path.basename(path)}")
            self._log_msg(f"Loaded spec: {os.path.basename(path)} "
                          f"({len(self._spec_scans)} scans)")
            scan_nums = sorted(self._spec_scans.keys())
            if scan_nums:
                self._log_msg(f"  Scan range: {scan_nums[0]} - {scan_nums[-1]}")
        except Exception as e:
            self.say(f"Could not read the SPEC file: {e}", error=True)

    # ---- Plotter tab ----
    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select data files", "",
            "Text files (*.txt *.dat);;All (*)")
        for p in paths:
            if p not in self._data_paths:
                self._data_paths.append(p)
                self._file_list.addItem(p)

    def _clear_files(self):
        self._data_paths = []
        self._file_list.clear()

    def _process(self):
        if not self._spec_scans:
            self.say("Load a SPEC file first.", error=True)
            return
        if not self._data_paths:
            self.say("Add at least one data file.", error=True)
            return

        self._log_msg("\n" + "=" * 50)
        self._log_msg(f"Processing {len(self._data_paths)} file(s)...")

        total_ok = 0
        total_skip = 0
        for path in self._data_paths:
            try:
                self._log_msg(f"\n{os.path.basename(path)}:")
                header_lines, traces = parse_exported_file(path)
                self._log_msg(f"  Found {len(traces)} trace(s)")

                if self._chk_inplace.isChecked():
                    out_path = path
                else:
                    base, ext = os.path.splitext(path)
                    out_path = f"{base}_hkl{ext}"

                snap_settings = {}
                for axis in ['H', 'K', 'L']:
                    w = self._snap_widgets[axis]
                    snap_settings[axis] = {
                        'enabled': w['enabled'].isChecked(),
                        'tol': w['tol'].value(),
                        'step': w['step'].value(),
                    }

                n_ok, n_skip = write_output(
                    out_path, header_lines, traces, self._spec_scans,
                    snap_settings=snap_settings,
                    log_fn=self._log_msg)
                total_ok += n_ok
                total_skip += n_skip
                self._log_msg(f"  Wrote: {out_path}")
            except Exception as e:
                self._log_msg(f"  ERROR: {e}")

        self._log_msg("\n" + "=" * 50)
        self._log_msg(f"Done. {total_ok} traces augmented, {total_skip} skipped.")
        self.say(f"H K L added to {total_ok} traces ({total_skip} skipped)")

    # ---- Peak Fits tab ----
    def _pf_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select peak fit files", "",
            "Text files (*.txt *.dat);;All (*)")
        for p in paths:
            if p not in self._peakfit_paths:
                self._peakfit_paths.append(p)
                self._pf_list.addItem(p)

    def _pf_clear_files(self):
        self._peakfit_paths = []
        self._pf_list.clear()

    def _pf_process(self):
        if not self._spec_scans:
            self.say("Load a SPEC file first.", error=True)
            return
        if not self._peakfit_paths:
            self.say("Add at least one peak fit file.", error=True)
            return

        area_col = self._pf_area_choice.currentText().strip() or 'area'

        snap_settings = {}
        for axis in ['H', 'K', 'L']:
            w = self._pf_snap_widgets[axis]
            snap_settings[axis] = {
                'enabled': w['enabled'].isChecked(),
                'tol': w['tol'].value(),
                'step': w['step'].value(),
            }

        self._log_msg("\n" + "=" * 50)
        self._log_msg(f"Peak fits: processing {len(self._peakfit_paths)} file(s), "
                      f"area column = '{area_col}'")

        for path in self._peakfit_paths:
            try:
                self._log_msg(f"\n{os.path.basename(path)}:")
                base, ext = os.path.splitext(path)
                out_path = f"{base}_hkl{ext}"
                n = process_peakfit_file(
                    path, out_path, self._spec_scans, area_col,
                    snap_settings, log_fn=self._log_msg)
                self._log_msg(f"  Wrote {n} rows to {out_path}")
            except Exception as e:
                self._log_msg(f"  ERROR: {e}")


    # ============================================================
    # GENX EXPORT TAB
    # ============================================================
    def _build_genx_tab(self):
        tab = QWidget()
        lv = QVBoxLayout(tab)
        lv.addWidget(hint("HKL files (output of the other tabs, or any 'H K L I' file) to GenX sls_sxrd format. "
                          "Every '# Trace:' block becomes a rod; Preview lists them so H and K can be set per trace."))
        b_pre = QPushButton("Preview traces")
        b_pre.clicked.connect(self._gx_preview)
        box, self._gx_list = self._file_box("Input HKL files", self._gx_add_files, self._gx_clear_files, (b_pre,))
        self._gx_hk_table = QTableWidget(0, 4)
        self._gx_hk_table.setHorizontalHeaderLabels(['Trace', 'Points', 'H', 'K'])
        self._gx_hk_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._gx_hk_table.setToolTip("H and K per trace; edit to override the file's values")
        top = QSplitter(Qt.Horizontal)
        top.addWidget(box)
        hk, hl = group("H K per trace (edit to override)", form=False)
        hl.addWidget(self._gx_hk_table)
        top.addWidget(hk)
        lv.addWidget(top, 1)

        og, form = group("Output")
        self._gx_err_mode = QComboBox()
        self._gx_err_mode.addItems(['sqrt(I)  (counting stats)', '5% of I', 'constant (use floor)'])
        self._gx_err_floor = QDoubleSpinBox()
        self._gx_err_floor.setRange(0.0, 1e12)
        self._gx_err_floor.setDecimals(2)
        self._gx_err_floor.setValue(1.0)
        form.addRow("Error", row(self._gx_err_mode, QLabel("floor"), self._gx_err_floor))
        self._gx_chk_lb = QCheckBox("Add LB and dL columns (beta roughness model)")
        self._gx_dl = QDoubleSpinBox()
        self._gx_dl.setRange(0.001, 100.0)
        self._gx_dl.setDecimals(4)
        self._gx_dl.setValue(1.0)
        self._gx_dl.setPrefix("dL ")
        self._gx_dl.setToolTip("Bragg spacing in reciprocal lattice units. For TiO2(110) in (110) coordinates, "
                               "dL = 1 (each integer L is a Bragg peak).")
        form.addRow(row(self._gx_chk_lb, self._gx_dl))
        self._gx_chk_dropneg = QCheckBox("Drop I ≤ 0 (GenX fits in log space)")
        self._gx_chk_dropneg.setChecked(True)
        self._gx_chk_combine = QCheckBox("Combine all inputs into one file (otherwise <name>_genx.dat per input)")
        self._gx_chk_combine.setChecked(True)
        form.addRow(self._gx_chk_dropneg)
        form.addRow(self._gx_chk_combine)
        self._gx_out = QLineEdit()
        self._gx_out.setPlaceholderText("next to the first input: <name>_genx.dat")
        b_browse = QPushButton("Browse…")
        b_browse.clicked.connect(self._gx_pick_output)
        form.addRow("Combined file", row(self._gx_out, b_browse))
        self._gx_chk_combine.toggled.connect(lambda on: (self._gx_out.setEnabled(on), b_browse.setEnabled(on)))
        lv.addWidget(og)
        lv.addWidget(primary_button("Export to GenX", self._gx_process))
        return tab

    def _gx_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add HKL files", "", "Text (*.txt *.dat);;All (*)")
        for p in paths:
            if p not in self._gx_files:
                self._gx_files.append(p)
                self._gx_list.addItem(os.path.basename(p))

    def _gx_clear_files(self):
        self._gx_files.clear()
        self._gx_traces_cache.clear()
        self._gx_list.clear()
        self._gx_hk_table.setRowCount(0)

    def _gx_pick_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "GenX output file", "genx_data.dat",
            "GenX data (*.dat *.txt);;All (*)")
        if path:
            self._gx_out.setText(path)

    def _gx_preview(self):
        if not self._gx_files:
            self.say("Add at least one HKL file.", error=True)
            return

        self._gx_traces_cache.clear()
        all_traces = []
        for p in self._gx_files:
            try:
                traces = parse_hkl_file(p)
                self._gx_traces_cache[p] = traces
                all_traces.extend(traces)
            except Exception as e:
                self._log_msg(f"Error parsing {os.path.basename(p)}: {e}")

        # Populate table
        self._gx_hk_table.setRowCount(len(all_traces))
        for i, t in enumerate(all_traces):
            label = t['label']
            npts = len(t['L'])
            h_val = t['H'][0] if len(t['H']) else 0
            k_val = t['K'][0] if len(t['K']) else 0
            self._gx_hk_table.setItem(i, 0, QTableWidgetItem(label))
            self._gx_hk_table.setItem(i, 1, QTableWidgetItem(str(npts)))
            self._gx_hk_table.setItem(i, 2, QTableWidgetItem(f"{h_val:.4g}"))
            self._gx_hk_table.setItem(i, 3, QTableWidgetItem(f"{k_val:.4g}"))
            # Make label and counts read-only
            for col in (0, 1):
                item = self._gx_hk_table.item(i, col)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

        self._log_msg(f"Preview: found {len(all_traces)} trace(s) across "
                      f"{len(self._gx_files)} file(s)")

    def _gx_process(self):
        if not self._gx_files:
            self.say("Add input files first.", error=True)
            return

        # Make sure traces are cached
        if not self._gx_traces_cache:
            self._gx_preview()

        # Read HK overrides from table
        hk_overrides = {}
        for i in range(self._gx_hk_table.rowCount()):
            label = self._gx_hk_table.item(i, 0).text()
            h_text = self._gx_hk_table.item(i, 2).text().strip()
            k_text = self._gx_hk_table.item(i, 3).text().strip()
            try:
                h = float(h_text) if h_text else None
            except ValueError:
                h = None
            try:
                k = float(k_text) if k_text else None
            except ValueError:
                k = None
            if h is not None or k is not None:
                hk_overrides[label] = (h, k)

        # Decode error mode
        em = self._gx_err_mode.currentText()
        if em.startswith('sqrt'): error_mode = 'sqrt'
        elif em.startswith('5%'): error_mode = 'percent_5'
        else: error_mode = 'constant'

        kwargs = dict(
            hk_overrides=hk_overrides,
            error_mode=error_mode,
            error_floor=self._gx_err_floor.value(),
            add_lb_dl=self._gx_chk_lb.isChecked(),
            dl_value=self._gx_dl.value(),
            drop_negative=self._gx_chk_dropneg.isChecked(),
            log_fn=self._log_msg,
        )

        self._log_msg("\n" + "=" * 50)

        if self._gx_chk_combine.isChecked():
            out_path = self._gx_out.text().strip()
            if not out_path:
                # Default next to the first input file
                base = os.path.splitext(self._gx_files[0])[0]
                out_path = base + "_genx.dat"
            all_traces = []
            for p in self._gx_files:
                all_traces.extend(self._gx_traces_cache.get(p, []))
            self._log_msg(f"GenX export: combining {len(all_traces)} trace(s) "
                          f"into {os.path.basename(out_path)}")
            try:
                n = write_genx_file(out_path, all_traces, **kwargs)
                self._log_msg(f"  Total: {n} rows")
            except Exception as e:
                self._log_msg(f"  ERROR: {e}")
        else:
            for p in self._gx_files:
                base, ext = os.path.splitext(p)
                out_path = f"{base}_genx.dat"
                traces = self._gx_traces_cache.get(p, [])
                self._log_msg(f"GenX export: {os.path.basename(p)} -> "
                              f"{os.path.basename(out_path)} ({len(traces)} traces)")
                try:
                    n = write_genx_file(out_path, traces, **kwargs)
                    self._log_msg(f"  Total: {n} rows")
                except Exception as e:
                    self._log_msg(f"  ERROR: {e}")


def main(argv=None):
    return run_tool(MainWindow)


if __name__ == '__main__':
    main()

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
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QListWidget, QGroupBox, QLineEdit, QTextEdit, QMessageBox,
                             QCheckBox, QTabWidget, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
                             QComboBox, QHeaderView)

from ctrfit.app.datatools.common import run_tool
from ctrfit.datatools.hkl import (parse_exported_file, parse_hkl_file, process_peakfit_file, write_genx_file,
                                  write_output)
from ctrfit.datatools.specfile import parse_spec_file


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SXRD HKL Appender")
        self.resize(900, 700)

        self._spec_path = None
        self._spec_scans = {}
        self._data_paths = []
        self._peakfit_paths = []

        cw = QWidget(); self.setCentralWidget(cw)
        lv = QVBoxLayout(cw)

        # Spec file group (shared across tabs)
        sg = QGroupBox("Spec file (shared)")
        sl = QHBoxLayout(sg)
        self._spec_label = QLineEdit(); self._spec_label.setReadOnly(True)
        sl.addWidget(self._spec_label)
        btn_spec = QPushButton("Browse...")
        btn_spec.clicked.connect(self._load_spec)
        sl.addWidget(btn_spec)
        lv.addWidget(sg)

        # Tabs
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_plotter_tab(), "Plotter Data")
        self._tabs.addTab(self._build_peakfit_tab(), "Peak Fits")
        self._tabs.addTab(self._build_genx_tab(), "GenX Export")
        lv.addWidget(self._tabs)

        # Shared log
        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.setStyleSheet("font-family:monospace;")
        self._log.setMaximumHeight(200)
        lv.addWidget(self._log)

        self._log_msg("Ready. Load a spec file, then use either tab.")

    def _build_plotter_tab(self):
        tab = QWidget()
        lv = QVBoxLayout(tab)

        dg = QGroupBox("Data files (from plotter export)")
        dl = QVBoxLayout(dg)
        self._file_list = QListWidget()
        dl.addWidget(self._file_list)
        bh = QHBoxLayout()
        btn_add = QPushButton("Add files...")
        btn_add.clicked.connect(self._add_files)
        bh.addWidget(btn_add)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_files)
        bh.addWidget(btn_clear)
        dl.addLayout(bh)
        lv.addWidget(dg)

        og = QGroupBox("Options")
        ol = QVBoxLayout(og)
        self._chk_inplace = QCheckBox(
            "Overwrite originals (otherwise saves as <n>_hkl.txt next to original)")
        ol.addWidget(self._chk_inplace)

        snap_group = QGroupBox("Snap nearly-constant columns to round values")
        sg_layout = QVBoxLayout(snap_group)
        sg_layout.addWidget(QLabel(
            "If a column's spread is below the tolerance, all values are\n"
            "replaced with the nearest multiple of step (e.g. H=1.975 -> H=2)."))

        self._snap_widgets = {}
        for axis in ['H', 'K', 'L']:
            row = QHBoxLayout()
            chk = QCheckBox(f"Snap {axis}")
            chk.setChecked(axis in ('H', 'K'))
            row.addWidget(chk)
            row.addWidget(QLabel("tol:"))
            tol = QDoubleSpinBox()
            tol.setRange(0.0, 1.0); tol.setSingleStep(0.01); tol.setValue(0.1)
            tol.setDecimals(3)
            row.addWidget(tol)
            row.addWidget(QLabel("step:"))
            step = QDoubleSpinBox()
            step.setRange(0.0, 10.0); step.setSingleStep(0.5); step.setValue(1.0)
            step.setDecimals(2)
            step.setToolTip("0=use mean, 1=nearest integer, 0.5=nearest half-integer")
            row.addWidget(step)
            row.addStretch()
            sg_layout.addLayout(row)
            self._snap_widgets[axis] = {'enabled': chk, 'tol': tol, 'step': step}

        ol.addWidget(snap_group)
        lv.addWidget(og)

        btn_process = QPushButton("Process")
        btn_process.setStyleSheet(
            "QPushButton{background:#2d5a27;color:white;font-weight:bold;padding:10px;}"
            "QPushButton:hover{background:#3a7a33;}")
        btn_process.clicked.connect(self._process)
        lv.addWidget(btn_process)
        return tab

    def _build_peakfit_tab(self):
        tab = QWidget()
        lv = QVBoxLayout(tab)

        info = QLabel(
            "Processes peak-fit files (from plotter peak fitting dialog).\n"
            "Extracts 'scan' and a chosen area column, looks up H, K, L\n"
            "from the spec file (mean of each scan), writes one row per scan."
        )
        info.setStyleSheet("color:#aaa;")
        lv.addWidget(info)

        fg = QGroupBox("Peak fit files")
        fl = QVBoxLayout(fg)
        self._pf_list = QListWidget()
        fl.addWidget(self._pf_list)
        bh = QHBoxLayout()
        btn_add = QPushButton("Add files...")
        btn_add.clicked.connect(self._pf_add_files)
        bh.addWidget(btn_add)
        btn_clr = QPushButton("Clear")
        btn_clr.clicked.connect(self._pf_clear_files)
        bh.addWidget(btn_clr)
        fl.addLayout(bh)
        lv.addWidget(fg)

        pg = QGroupBox("Options")
        pl = QVBoxLayout(pg)

        row = QHBoxLayout()
        row.addWidget(QLabel("Area column:"))
        self._pf_area_choice = QListWidget()
        self._pf_area_choice.setMaximumHeight(75)
        for name in ['area', 'fit_area', 'num_area']:
            self._pf_area_choice.addItem(name)
        self._pf_area_choice.setCurrentRow(0)
        row.addWidget(self._pf_area_choice)
        pl.addLayout(row)

        snap_group = QGroupBox("Snap nearly-constant HKL columns")
        sg_layout = QVBoxLayout(snap_group)
        self._pf_snap_widgets = {}
        for axis in ['H', 'K', 'L']:
            row = QHBoxLayout()
            chk = QCheckBox(f"Snap {axis}")
            chk.setChecked(axis in ('H', 'K'))
            row.addWidget(chk)
            row.addWidget(QLabel("tol:"))
            tol = QDoubleSpinBox()
            tol.setRange(0.0, 1.0); tol.setSingleStep(0.01); tol.setValue(0.1)
            tol.setDecimals(3)
            row.addWidget(tol)
            row.addWidget(QLabel("step:"))
            step = QDoubleSpinBox()
            step.setRange(0.0, 10.0); step.setSingleStep(0.5); step.setValue(1.0)
            step.setDecimals(2)
            row.addWidget(step)
            row.addStretch()
            sg_layout.addLayout(row)
            self._pf_snap_widgets[axis] = {'enabled': chk, 'tol': tol, 'step': step}
        pl.addWidget(snap_group)
        lv.addWidget(pg)

        btn = QPushButton("Process")
        btn.setStyleSheet(
            "QPushButton{background:#2d5a27;color:white;font-weight:bold;padding:10px;}"
            "QPushButton:hover{background:#3a7a33;}")
        btn.clicked.connect(self._pf_process)
        lv.addWidget(btn)
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
            self._log_msg(f"Loaded spec: {os.path.basename(path)} "
                          f"({len(self._spec_scans)} scans)")
            scan_nums = sorted(self._spec_scans.keys())
            if scan_nums:
                self._log_msg(f"  Scan range: {scan_nums[0]} - {scan_nums[-1]}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse spec: {e}")

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
            QMessageBox.warning(self, "No spec", "Load a spec file first.")
            return
        if not self._data_paths:
            QMessageBox.warning(self, "No files", "Add at least one data file.")
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
            QMessageBox.warning(self, "No spec", "Load a spec file first.")
            return
        if not self._peakfit_paths:
            QMessageBox.warning(self, "No files", "Add at least one peak fit file.")
            return

        area_col = self._pf_area_choice.currentItem().text()

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

        info = QLabel(
            "Convert HKL-augmented files (output of the Plotter Data tab,\n"
            "or any 'H K L Intensity' file) to GenX sls_sxrd format.\n"
            "Each file may contain one or more 'Trace:' blocks; each becomes\n"
            "a rod in the output. Set H K per trace if needed."
        )
        info.setStyleSheet("color:#aaa;")
        lv.addWidget(info)

        # File list
        fg = QGroupBox("Input HKL files")
        fl = QVBoxLayout(fg)
        self._gx_list = QListWidget()
        fl.addWidget(self._gx_list)
        bh = QHBoxLayout()
        b_add = QPushButton("Add files...")
        b_add.clicked.connect(self._gx_add_files)
        bh.addWidget(b_add)
        b_pre = QPushButton("Preview traces")
        b_pre.clicked.connect(self._gx_preview)
        bh.addWidget(b_pre)
        b_clr = QPushButton("Clear")
        b_clr.clicked.connect(self._gx_clear_files)
        bh.addWidget(b_clr)
        fl.addLayout(bh)
        lv.addWidget(fg)

        # HK override table - populated after preview
        hg = QGroupBox("H K override per trace (leave blank to keep file value)")
        hl = QVBoxLayout(hg)
        self._gx_hk_table = QTableWidget(0, 4)
        self._gx_hk_table.setHorizontalHeaderLabels(
            ['Trace label', 'N points', 'H override', 'K override'])
        self._gx_hk_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        hl.addWidget(self._gx_hk_table)
        lv.addWidget(hg)

        # Options
        og = QGroupBox("GenX output options")
        ol = QVBoxLayout(og)

        row = QHBoxLayout()
        row.addWidget(QLabel("Error model:"))
        self._gx_err_mode = QComboBox()
        self._gx_err_mode.addItems(['sqrt(I)  (counting stats)',
                                     '5% of I',
                                     'constant (use floor)'])
        row.addWidget(self._gx_err_mode)
        row.addWidget(QLabel("Error floor:"))
        self._gx_err_floor = QDoubleSpinBox()
        self._gx_err_floor.setRange(0.0, 1e12)
        self._gx_err_floor.setDecimals(2)
        self._gx_err_floor.setValue(1.0)
        row.addWidget(self._gx_err_floor)
        ol.addLayout(row)

        row2 = QHBoxLayout()
        self._gx_chk_lb = QCheckBox("Add LB and dL columns (for beta roughness model)")
        self._gx_chk_lb.setChecked(False)
        row2.addWidget(self._gx_chk_lb)
        row2.addWidget(QLabel("dL:"))
        self._gx_dl = QDoubleSpinBox()
        self._gx_dl.setRange(0.001, 100.0)
        self._gx_dl.setDecimals(4)
        self._gx_dl.setValue(1.0)
        self._gx_dl.setToolTip(
            "Bragg spacing in reciprocal lattice units. For TiO2(110) "
            "in (110) coordinates, dL = 1 (each L=integer is a Bragg peak).")
        row2.addWidget(self._gx_dl)
        ol.addLayout(row2)

        self._gx_chk_dropneg = QCheckBox(
            "Drop non-positive intensities (recommended - GenX fits in log space)")
        self._gx_chk_dropneg.setChecked(True)
        ol.addWidget(self._gx_chk_dropneg)

        self._gx_chk_combine = QCheckBox(
            "Combine all input files into one output (otherwise one output per input)")
        self._gx_chk_combine.setChecked(True)
        ol.addWidget(self._gx_chk_combine)

        lv.addWidget(og)

        # Output
        outr = QHBoxLayout()
        outr.addWidget(QLabel("Output file (if combining):"))
        self._gx_out = QLineEdit()
        self._gx_out.setPlaceholderText("genx_data.dat")
        outr.addWidget(self._gx_out)
        b_browse = QPushButton("Browse...")
        b_browse.clicked.connect(self._gx_pick_output)
        outr.addWidget(b_browse)
        lv.addLayout(outr)

        b_proc = QPushButton("Export to GenX format")
        b_proc.setStyleSheet(
            "QPushButton{background:#5a3a8a;color:white;font-weight:bold;padding:8px;}")
        b_proc.clicked.connect(self._gx_process)
        lv.addWidget(b_proc)

        # State
        self._gx_files = []
        self._gx_traces_cache = {}  # path -> list of traces

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
            QMessageBox.warning(self, "No files", "Add at least one HKL file.")
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
            QMessageBox.warning(self, "No files", "Add input files first.")
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

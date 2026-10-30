"""CTR processing window: ROI sums from background subtraction + the SPEC file -> normalised,
corrected structure factors with errors, merged and scaled (steps 2 to 5). The logic is in
ctrfit.datatools.processing; this file only holds the window.
"""
import json
import os

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHeaderView, QLineEdit,
                             QMainWindow, QPushButton, QShortcut, QSplitter, QTableWidget, QTableWidgetItem,
                             QTextEdit, QVBoxLayout, QWidget)
from matplotlib.figure import Figure

from ctrfit.app.datatools.common import (SafeCanvas, StatusMixin, group, mono_font, primary_button, row,
                                         run_tool, sidebar)
from ctrfit.datatools import processing as pr
from ctrfit.datatools.paths import settings_file
from ctrfit.datatools.specfile import parse_spec_file

try:
    from PyQt5.sip import isdeleted
except ImportError:  # pragma: no cover
    from sip import isdeleted

NONE = "(none)"
SETTINGS = "sxrd_processing_settings.json"


def _spin(lo, hi, val, dec=3, step=0.01, suffix=""):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


class ProcessingWindow(StatusMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SXRD CTR processing")
        self.resize(1450, 900)
        self._sums = {}
        self._sum_paths = []
        self._spec = {}
        self._results = []
        self._merged = None
        self._notes = []
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self.reprocess)
        self._loading = False
        self._build()
        self._load_settings()
        self.say("Open the SPEC file and the ROI sums folder (background subtraction output).")

    # ------------------------------------------------------------------ layout
    def _build(self):
        ch = self._changed
        fg, ff = group("Files")
        self._spec_path = QLineEdit()
        self._spec_path.setReadOnly(True)
        b_spec = QPushButton("Browse…")
        b_spec.clicked.connect(lambda: self.open_spec())
        ff.addRow("SPEC", row(self._spec_path, b_spec))
        self._sums_label = QLineEdit()
        self._sums_label.setReadOnly(True)
        self._sums_label.setPlaceholderText("folder of ROI sums .txt files, or chosen files")
        b_dir, b_files = QPushButton("Folder…"), QPushButton("Files…")
        b_dir.clicked.connect(lambda: self.open_sums_folder())
        b_files.clicked.connect(lambda: self.open_sums_files())
        ff.addRow("ROI sums", row(self._sums_label, b_dir, b_files))
        b_refresh = QPushButton("Refresh files  (F5)")
        b_refresh.setToolTip("Read the SPEC file and the ROI sums again (new scans, new images)")
        b_refresh.clicked.connect(lambda: self.refresh())
        sc = QShortcut(QKeySequence("F5"), self)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(self.refresh)
        ff.addRow(b_refresh)

        ng, nf = group("2. Normalise")
        self._roi = QComboBox()
        self._roi.setToolTip("Signal ROI of the ROI sums (its '_err' column gives the counting error)")
        self._monitor, self._transm, self._ctime = QComboBox(), QComboBox(), QComboBox()
        self._monitor.setToolTip("I is divided by monitor / (mean monitor of all scans)")
        self._transm.setToolTip("Attenuator transmission per point (I is divided by it)")
        self._ctime.setToolTip("Count time per point, e.g. Seconds (I is divided by it)")
        for w in (self._roi, self._monitor, self._transm, self._ctime):
            w.currentIndexChanged.connect(ch)
        nf.addRow("Intensity", self._roi)
        nf.addRow("Monitor", self._monitor)
        nf.addRow("Transmission", self._transm)
        nf.addRow("Count time", self._ctime)

        cg, cf = group("3. Correct")
        self._geometry = QComboBox()
        self._geometry.addItems(list(pr.GEOMETRIES))
        self._geometry.setToolTip("Horizontal surface: the normal is vertical (vertical scattering plane for "
                                  "the specular rod); vertical surface: the normal is horizontal")
        self._energy = _spin(3, 60, 16.0, 4, 0.1, " keV")
        self._alpha = _spin(0, 30, 0.5, 3, 0.05, "°")
        self._alpha.setToolTip("Incidence angle (fixed). The specular rod uses alpha = beta.")
        self._alpha_col, self._beta_col = QComboBox(), QComboBox()
        self._alpha_col.setToolTip("Incidence angle from a SPEC column instead of the fixed value")
        self._beta_col.setToolTip("Exit angle from a SPEC column instead of from L")
        cf.addRow("Geometry", self._geometry)
        cf.addRow("Energy", self._energy)
        cf.addRow("Incidence", row(self._alpha, self._alpha_col))
        cf.addRow("Exit angle", self._beta_col)
        self._a1, self._a2, self._a3 = (_spin(0.5, 50, v, 4, 0.01, " Å") for v in pr.TIO2_SURFACE)
        cf.addRow("Surface cell", row(self._a1, self._a2, self._a3, stretch_first=False))
        self._chk_pol = QCheckBox("Polarisation, horizontal fraction")
        self._pol = _spin(0, 1, 0.98, 3, 0.01)
        self._chk_rod = QCheckBox("Rod interception 1/sin(β)")
        self._chk_area = QCheckBox("Illuminated area")
        for c in (self._chk_pol, self._chk_rod, self._chk_area):
            c.setChecked(True)
        self._beam_h = _spin(0.001, 10, 0.05, 3, 0.01, " mm")
        self._beam_w = _spin(0.001, 50, 0.5, 3, 0.05, " mm")
        self._sample = _spin(0.01, 100, 5.0, 2, 0.5, " mm")
        self._beam_h.setToolTip("Beam size in the plane of incidence")
        self._sample.setToolTip("Sample length along the beam")
        cf.addRow(row(self._chk_pol, self._pol))
        cf.addRow(self._chk_rod)
        cf.addRow(self._chk_area)
        cf.addRow("Beam h, w", row(self._beam_h, self._beam_w, stretch_first=False))
        cf.addRow("Sample length", self._sample)
        for w in (self._geometry, self._alpha_col, self._beta_col):
            w.currentIndexChanged.connect(ch)
        for w in (self._energy, self._alpha, self._a1, self._a2, self._a3, self._pol, self._beam_h, self._beam_w,
                  self._sample):
            w.valueChanged.connect(ch)
        for w in (self._chk_pol, self._chk_rod, self._chk_area):
            w.toggled.connect(ch)

        mg, mf = group("5. Merge and scale")
        self._equiv = QComboBox()
        self._equiv.addItems(list(pr.EQUIVALENTS))
        self._equiv.setToolTip("Rods averaged together: none, or (±H, ±K) at the same L (p2mm surface)")
        self._ltol = _spin(0.0001, 0.5, 0.01, 4, 0.005)
        self._ltol.setToolTip("Points of the same (or equivalent) rod closer than this in L are averaged")
        self._chk_overlap = QCheckBox("Scale scans of a rod on their overlap")
        self._chk_overlap.setChecked(True)
        self._bragg = _spin(0, 0.5, 0.0, 3, 0.01)
        self._bragg.setToolTip("Leave out points with |L - integer| below this (0: keep all)")
        self._floor = _spin(0, 50, 5.0, 1, 0.5, " %")
        self._floor.setToolTip("Systematic error floor written for the fit (sys_floor); set it from R_sym")
        for w in (self._equiv,):
            w.currentIndexChanged.connect(ch)
        for w in (self._ltol, self._bragg, self._floor):
            w.valueChanged.connect(ch)
        self._chk_overlap.toggled.connect(ch)
        mf.addRow("Equivalents", self._equiv)
        mf.addRow("L tolerance", self._ltol)
        mf.addRow(self._chk_overlap)
        mf.addRow("Skip near integer L", self._bragg)
        mf.addRow("Systematic floor", self._floor)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(sidebar(fg, ng, cg, mg, width=400), 1)
        lv.addWidget(primary_button("Process and save…", self.save_dialog, "Ctrl+S", self))

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(["Use", "Scan", "Rod", "L range", "Points", "Scale", "Notes"])
        self._table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.itemSelectionChanged.connect(self._draw_preview)
        self._table.itemChanged.connect(self._table_edited)
        self._fig = Figure(layout="constrained")
        self._canvas = SafeCanvas(self._fig)
        prev = QWidget()
        pv = QVBoxLayout(prev)
        pv.setContentsMargins(0, 0, 0, 0)
        self._chk_logy = QCheckBox("Log intensity")
        self._chk_logy.setChecked(True)
        self._chk_logy.toggled.connect(self._draw_preview)
        pv.addWidget(row(self._chk_logy))
        pv.addWidget(self._canvas, 1)
        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setFont(mono_font())
        right = QSplitter(Qt.Vertical)
        right.addWidget(self._table)
        right.addWidget(prev)
        right.addWidget(self._summary)
        right.setSizes([220, 560, 120])
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([430, 1020])
        self.setCentralWidget(split)

    # ------------------------------------------------------------------ files
    def open_spec(self, path=None):
        path = path or QFileDialog.getOpenFileName(self, "SPEC file", "", "SPEC (*.spec *.dat *.txt);;All files (*)")[0]
        if path:
            self._spec_path.setText(str(path))
            self.refresh()

    def open_sums_folder(self, folder=None):
        folder = folder or QFileDialog.getExistingDirectory(self, "ROI sums folder")
        if folder:
            self._sum_paths = pr.sum_files(folder)
            self._sums_label.setText(f"{folder}  ({len(self._sum_paths)} files)")
            self._sums_folder = folder
            self.refresh()

    def open_sums_files(self, paths=None):
        paths = paths or QFileDialog.getOpenFileNames(self, "ROI sums files", "", "Text (*.txt);;All files (*)")[0]
        if paths:
            self._sum_paths = [str(p) for p in paths]
            self._sums_folder = None
            self._sums_label.setText(f"{len(self._sum_paths)} file(s)")
            self.refresh()

    def refresh(self):
        """Read the SPEC file and the ROI sums again, keep the options, process."""
        def go():
            if getattr(self, "_sums_folder", None):
                self._sum_paths = pr.sum_files(self._sums_folder)
            if self._spec_path.text() and os.path.isfile(self._spec_path.text()):
                self._spec = parse_spec_file(self._spec_path.text())
            self._sums = pr.load_sums(self._sum_paths) if self._sum_paths else {}
            self._fill_columns()
            self.reprocess()
            self.say(f"{len(self._sums)} scans in the ROI sums, {len(self._spec)} in the SPEC file")
        self.guarded(go)

    def _fill_columns(self):
        self._loading = True
        names, with_err = pr.roi_columns(self._sums)
        spec_cols = []
        for sn in self._sums:
            if sn in self._spec:
                spec_cols += [c for c in self._spec[sn].columns if c not in spec_cols]
        for combo, items, default in ((self._roi, names, (with_err or names or [""])[0]),
                                      (self._monitor, [NONE] + spec_cols, None),
                                      (self._transm, [NONE] + spec_cols, None),
                                      (self._ctime, [NONE] + spec_cols, None),
                                      (self._alpha_col, ["(fixed)"] + spec_cols, None),
                                      (self._beta_col, ["(from L)"] + spec_cols, None)):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            pick = cur if cur in items else (self._pending.get(id(combo)) if hasattr(self, "_pending") else None)
            if pick in items:
                combo.setCurrentText(pick)
            elif default in items:
                combo.setCurrentText(default)
            combo.blockSignals(False)
        self._loading = False

    # ------------------------------------------------------------------ processing
    def options(self):
        def col(combo):
            t = combo.currentText()
            return None if t in ("", NONE, "(fixed)", "(from L)") else t
        o = pr.Options(roi=self._roi.currentText() or None, monitor=col(self._monitor), transmission=col(self._transm),
                       count_time=col(self._ctime), energy_kev=self._energy.value(),
                       lattice=(self._a1.value(), self._a2.value(), self._a3.value()),
                       geometry=self._geometry.currentText(), alpha_deg=self._alpha.value(),
                       alpha_column=col(self._alpha_col), beta_column=col(self._beta_col),
                       polarization=self._chk_pol.isChecked(), pol_fraction=self._pol.value(),
                       rod_interception=self._chk_rod.isChecked(), area=self._chk_area.isChecked(),
                       beam_h_mm=self._beam_h.value(), beam_w_mm=self._beam_w.value(), sample_mm=self._sample.value(),
                       equivalents=self._equiv.currentText(), l_tol=self._ltol.value(),
                       match_overlaps=self._chk_overlap.isChecked(), exclude_bragg=self._bragg.value(),
                       floor_pct=self._floor.value())
        o.scan_scale, o.scan_use = self._table_scales()
        return o

    def _table_scales(self):
        scale, use = {}, {}
        for r in range(self._table.rowCount()):
            sn = int(self._table.item(r, 1).text())
            try:
                scale[sn] = float(self._table.item(r, 5).data(Qt.UserRole))
            except (TypeError, ValueError):
                scale[sn] = 1.0
            use[sn] = self._table.item(r, 0).checkState() == Qt.Checked
        return scale, use

    def _changed(self, *_):
        if not self._loading:
            self._timer.start()

    def reprocess(self):
        if not self._sums or not self._spec:
            return
        def go():
            opt = self.options()
            self._results, self._merged, self._notes = pr.process(self._sums, self._spec, opt)
            self._fill_table(opt)
            self._write_summary()
            self._draw_preview()
        self.guarded(go)

    def _fill_table(self, opt):
        sel = self._selected_scan()
        t = self._table
        t.blockSignals(True)
        t.setRowCount(len(self._results))
        for i, r in enumerate(self._results):
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            use.setCheckState(Qt.Checked if opt.scan_use.get(r.scan, True) else Qt.Unchecked)
            items = [use, QTableWidgetItem(str(r.scan)), QTableWidgetItem(f"{r.rod[0]} {r.rod[1]}"),
                     QTableWidgetItem(f"{np.nanmin(r.L):.3f} to {np.nanmax(r.L):.3f}"),
                     QTableWidgetItem(f"{int(r.keep.sum())} / {r.L.size}"),
                     QTableWidgetItem(f"{r.scale:.4g}"), QTableWidgetItem("; ".join(r.notes))]
            items[5].setData(Qt.UserRole, opt.scan_scale.get(r.scan, 1.0))
            items[5].setToolTip("Applied scale (hand scale × overlap match); type a number to set the hand scale")
            for c, it in enumerate(items):
                if c not in (0, 5):
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                t.setItem(i, c, it)
            if r.scan == sel:
                t.selectRow(i)
        if t.rowCount() and not t.selectedItems():
            t.selectRow(0)
        t.resizeColumnsToContents()
        t.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        t.blockSignals(False)

    def _table_edited(self, item):
        if not self._alive():
            return
        if item.column() == 5:
            try:
                val = float(item.text())
            except ValueError:
                self.say(f"Scale '{item.text()}' is not a number", error=True)
                return
            # the typed value is the applied scale; keep the overlap match out of the hand scale
            r = self._result(int(self._table.item(item.row(), 1).text()))
            hand = float(item.data(Qt.UserRole) or 1.0)
            match = (r.scale / hand) if (r is not None and hand) else 1.0
            self._table.blockSignals(True)
            item.setData(Qt.UserRole, val / match if match else val)
            self._table.blockSignals(False)
        if item.column() in (0, 5):
            self._changed()

    def _result(self, sn):
        return next((r for r in self._results if r.scan == sn), None)

    def _selected_scan(self):
        rows = self._table.selectionModel().selectedRows() if self._table.selectionModel() else []
        if rows:
            it = self._table.item(rows[0].row(), 1)
            return int(it.text()) if it else None
        return None

    def _write_summary(self):
        m = self._merged
        lines = []
        if m is not None:
            rods = sorted(set(m.rod))
            lines.append(f"{m.L.size} merged points on {len(rods)} rod(s): {', '.join(rods)}")
            if m.n_multiple:
                lines.append(f"R_sym = {100 * m.r_sym:.1f} % over {m.n_multiple} points measured more than once "
                             f"(a guide for the systematic floor)")
            else:
                lines.append("R_sym: no point measured more than once (no equivalents or overlaps)")
        lines += self._notes
        self._summary.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------ preview
    def _alive(self):
        """False while Qt tears the window down (the table still emits signals then)."""
        return not (isdeleted(self._table) or isdeleted(self._canvas))

    def _draw_preview(self):
        if not self._alive():
            return
        fig = self._fig
        fig.clear()
        r = self._result(self._selected_scan())
        if r is None:
            fig.text(0.5, 0.5, "Open the SPEC file and the ROI sums", ha="center", va="center", color="#888")
            self._canvas.draw_idle()
            return
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        k = r.keep
        raw = r.I_raw / np.nanmax(np.abs(r.I_raw[k])) if k.any() and np.nanmax(np.abs(r.I_raw[k])) > 0 else r.I_raw
        corr = r.I_corr * r.scale
        cmax = np.nanmax(np.abs(corr[k])) if k.any() else np.nan
        corr_n = corr / cmax if np.isfinite(cmax) and cmax > 0 else corr
        ax1.plot(r.L, raw, "o", ms=3, color="#9a9a9a", label="before: ROI sum")
        ax1.plot(r.L[k], corr_n[k], "o-", ms=3, lw=1, color="#1f77b4", label="after: normalised and corrected")
        if (~k).any():
            ax1.plot(r.L[~k], corr_n[~k], "x", ms=5, color="#d62728", label="left out")
        ax1.set_ylabel("I (each scaled to its max)")
        if self._chk_logy.isChecked():
            ax1.set_yscale("log")
        ax1.legend(fontsize=8)
        ax1.set_title(f"Scan {r.scan}, rod {r.rod[0]} {r.rod[1]}")
        F, sF = r.F * np.sqrt(r.scale), r.sF * np.sqrt(r.scale)
        ax2.errorbar(r.L[k], F[k], yerr=sF[k], fmt="o", ms=3, lw=0.8, capsize=1.5, color="#1f77b4",
                     label=f"scan {r.scan}")
        m = self._merged
        if m is not None:
            key = pr._rod_key(r.rod, self.options())
            sel = (m.H == key[0]) & (m.K == key[1])
            if sel.any():
                ax2.errorbar(m.L[sel], m.F[sel], yerr=m.sF[sel], fmt="s", ms=2.5, lw=0.6, capsize=1, mfc="none",
                             color="#d62728", label="merged rod")
        ax2.set_xlabel("L (r.l.u.)")
        ax2.set_ylabel("|F|")
        if self._chk_logy.isChecked():
            ax2.set_yscale("log")
        ax2.legend(fontsize=8)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------ output
    def save_to(self, base):
        """Write <base>_scans.txt (every point), <base>.csv (merged, for the fitting app) and
        <base>_genx.dat. Returns the paths."""
        if not self._results or self._merged is None:
            raise ValueError("nothing processed yet")
        base = os.path.splitext(base)[0]
        opt = self.options()
        out = [pr.write_scans(base + "_scans.txt", self._results),
               pr.write_merged(base + ".csv", self._merged, opt, os.path.basename(base)),
               pr.write_genx(base + "_genx.dat", self._merged)]
        self._save_settings()
        self.say(f"Saved {', '.join(os.path.basename(p) for p in out)}")
        return out

    def save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save processed data (base name)", "processed.csv",
                                              "CSV for fitting (*.csv);;All files (*)")
        if path:
            self.guarded(lambda: self.save_to(path))

    # ------------------------------------------------------------------ settings
    WIDGETS = ("_roi", "_monitor", "_transm", "_ctime", "_geometry", "_energy", "_alpha", "_alpha_col", "_beta_col",
               "_a1", "_a2", "_a3", "_chk_pol", "_pol", "_chk_rod", "_chk_area", "_beam_h", "_beam_w", "_sample",
               "_equiv", "_ltol", "_chk_overlap", "_bragg", "_floor")

    def _save_settings(self):
        d = {"spec": self._spec_path.text(), "sums_folder": getattr(self, "_sums_folder", None),
             "sum_paths": self._sum_paths}
        for name in self.WIDGETS:
            w = getattr(self, name)
            d[name] = (w.currentText() if isinstance(w, QComboBox) else w.isChecked() if isinstance(w, QCheckBox)
                       else w.value())
        d["scan_scale"], d["scan_use"] = ({str(k): v for k, v in x.items()} for x in self._table_scales())
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
        self._pending = {}
        for name in self.WIDGETS:
            if name not in d:
                continue
            w = getattr(self, name)
            if isinstance(w, QComboBox):
                self._pending[id(w)] = d[name]
                if w.findText(d[name]) >= 0:
                    w.setCurrentText(d[name])
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(d[name]))
            else:
                w.setValue(float(d[name]))
        self._loading = False
        self._spec_path.setText(d.get("spec") or "")
        self._sums_folder = d.get("sums_folder")
        self._sum_paths = d.get("sum_paths") or []
        if self._sums_folder and os.path.isdir(self._sums_folder):
            self._sums_label.setText(self._sums_folder)
        elif self._sum_paths:
            self._sums_label.setText(f"{len(self._sum_paths)} file(s)")
        if self._spec_path.text() and (self._sum_paths or self._sums_folder):
            QTimer.singleShot(0, self.refresh)

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


def main(argv=None):
    return run_tool(ProcessingWindow)


if __name__ == '__main__':
    main()

"""SXRD HKL File Combiner
======================
Combine multiple HKL-format files into one minimal output:
  - One header line:  H K L I
  - All data rows concatenated, one per line
  - No scan numbers, no image numbers, no trace separators, no comments

Qt window of the data tools (ctr_data_tools.py); the logic is in ctrfit.datatools.
"""
import os

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QGroupBox, QLineEdit, QTextEdit, QMessageBox, QCheckBox,
                             QListWidget, QComboBox)

from ctrfit.app.datatools.common import run_tool
from ctrfit.datatools.hkl import combine_hkl_files, write_hkl_rows


class CombinerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SXRD HKL File Combiner")
        self.resize(700, 600)

        self._files = []

        cw = QWidget()
        self.setCentralWidget(cw)
        lv = QVBoxLayout(cw)

        info = QLabel(
            "Combines multiple HKL files (output of plotter, smoother, or any\n"
            "H K L I file) into one minimal text file with a single H K L I\n"
            "header and one data row per line. No scan markers, no comments."
        )
        info.setStyleSheet("color: #aaa;")
        lv.addWidget(info)

        # File list
        fg = QGroupBox("Input files")
        fl = QVBoxLayout(fg)
        self._list = QListWidget()
        fl.addWidget(self._list)
        bh = QHBoxLayout()
        b_add = QPushButton("Add files...")
        b_add.clicked.connect(self._add_files)
        bh.addWidget(b_add)
        b_clr = QPushButton("Clear")
        b_clr.clicked.connect(self._clear)
        bh.addWidget(b_clr)
        fl.addLayout(bh)
        lv.addWidget(fg)

        # Options
        og = QGroupBox("Options")
        ol = QVBoxLayout(og)
        self._chk_drop_nan = QCheckBox("Drop NaN / non-finite intensity rows")
        self._chk_drop_nan.setChecked(True)
        ol.addWidget(self._chk_drop_nan)
        self._chk_drop_nonpos = QCheckBox("Drop non-positive (I <= 0) rows")
        self._chk_drop_nonpos.setChecked(False)
        ol.addWidget(self._chk_drop_nonpos)

        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort:"))
        self._sort = QComboBox()
        self._sort.addItems(["File order (no sort)", "By L", "By H, K, L"])
        sort_row.addWidget(self._sort)
        ol.addLayout(sort_row)

        self._chk_dedupe = QCheckBox(
            "Remove exact duplicate rows (same H K L I)")
        self._chk_dedupe.setChecked(False)
        ol.addWidget(self._chk_dedupe)
        lv.addWidget(og)

        # Output
        outr = QHBoxLayout()
        outr.addWidget(QLabel("Output:"))
        self._out = QLineEdit()
        self._out.setPlaceholderText("combined.txt")
        outr.addWidget(self._out)
        b_browse = QPushButton("Browse...")
        b_browse.clicked.connect(self._pick_output)
        outr.addWidget(b_browse)
        lv.addLayout(outr)

        b_proc = QPushButton("Combine")
        b_proc.setStyleSheet(
            "QPushButton{background:#2a5a8a;color:white;font-weight:bold;padding:8px;}")
        b_proc.clicked.connect(self._combine)
        lv.addWidget(b_proc)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            "font-family: monospace; background:#1a1a1a; color:#ccc;")
        lv.addWidget(self._log)

    def _log_msg(self, msg):
        self._log.append(msg)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add HKL files", "", "Text (*.txt *.dat);;All (*)")
        for p in paths:
            if p in self._files:
                continue
            self._files.append(p)
            self._list.addItem(os.path.basename(p))

    def _clear(self):
        self._files.clear()
        self._list.clear()

    def _pick_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Output file", "combined.txt", "Text (*.txt *.dat);;All (*)")
        if path:
            self._out.setText(path)

    def _combine(self):
        if not self._files:
            QMessageBox.warning(self, "No files", "Add input files first.")
            return
        out_path = self._out.text().strip()
        if not out_path:
            out_path = os.path.splitext(self._files[0])[0] + "_combined.txt"
            self._out.setText(out_path)
        self._log_msg("\n" + "=" * 50)
        self._log_msg(f"Combining {len(self._files)} file(s) -> {os.path.basename(out_path)}")
        rows, log = combine_hkl_files(self._files, self._chk_drop_nan.isChecked(), self._chk_drop_nonpos.isChecked(),
                                      self._sort.currentText(), self._chk_dedupe.isChecked())
        for line in log:
            self._log_msg(line)
        if not rows:
            self._log_msg("No rows to write.")
            QMessageBox.warning(self, "Empty", "No data rows after filtering.")
            return
        try:
            write_hkl_rows(out_path, rows)
        except Exception as e:
            QMessageBox.critical(self, "Write error", str(e))
            return
        msg = f"Wrote {len(rows)} rows to {os.path.basename(out_path)}"
        self._log_msg(msg)
        QMessageBox.information(self, "Done", msg)


def main(argv=None):
    return run_tool(CombinerWindow)


if __name__ == '__main__':
    main()

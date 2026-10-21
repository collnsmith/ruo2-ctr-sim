"""SXRD HKL File Combiner
======================
Combine multiple HKL-format files into one minimal output:
  - One header line:  H K L I
  - All data rows concatenated, one per line
  - No scan numbers, no image numbers, no trace separators, no comments

Qt window of the data tools (ctr_data_tools.py); the logic is in ctrfit.datatools.
"""
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLineEdit,
                             QTextEdit, QCheckBox, QListWidget, QComboBox, QSplitter)

from ctrfit.app.datatools.common import (StatusMixin, group, hint, mono_font, primary_button, row, run_tool,
                                         sidebar)
from ctrfit.datatools.hkl import SORT_CHOICES, combine_hkl_files, write_hkl_rows


class CombinerWindow(StatusMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SXRD HKL File Combiner")
        self.resize(1000, 650)
        self._files = []

        files, fl = group("Input files", form=False)
        fl.addWidget(hint("HKL files (plotter, appender or smoother output, or any H K L I file). "
                          "Every trace of every file goes into one table."))
        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.ExtendedSelection)
        self._list.setMinimumHeight(160)
        fl.addWidget(self._list, 1)
        b_add, b_rm, b_clr = QPushButton("Add files…"), QPushButton("Remove selected"), QPushButton("Clear")
        b_add.clicked.connect(self._add_files)
        b_rm.clicked.connect(self._remove_selected)
        b_clr.clicked.connect(self._clear)
        fl.addWidget(row(b_add, b_rm, b_clr, 1, stretch_first=False))

        opts, ol = group("Rows")
        self._chk_drop_nan = QCheckBox("Drop NaN / non-finite I")
        self._chk_drop_nan.setChecked(True)
        self._chk_drop_nonpos = QCheckBox("Drop I ≤ 0")
        self._chk_dedupe = QCheckBox("Remove exact duplicates (same H K L I)")
        self._sort = QComboBox()
        self._sort.addItems(list(SORT_CHOICES))
        ol.addRow(self._chk_drop_nan)
        ol.addRow(self._chk_drop_nonpos)
        ol.addRow(self._chk_dedupe)
        ol.addRow("Sort", self._sort)

        out, outl = group("Output")
        self._out = QLineEdit()
        self._out.setPlaceholderText("next to the first input: <name>_combined.txt")
        b_browse = QPushButton("Browse…")
        b_browse.clicked.connect(self._pick_output)
        outl.addRow("File", row(self._out, b_browse))
        outl.addRow(hint("One 'H K L I' header line, one tab-separated row per point; no scan markers or comments."))

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(sidebar(files, opts, out), 1)
        ll.addWidget(primary_button("Combine", self._combine, "Ctrl+R", self))

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(mono_font())
        right, rl = group("Log", form=False)
        rl.addWidget(self._log)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([420, 580])
        self.setCentralWidget(split)
        self.say("Add HKL files, then Combine.")

    def _remove_selected(self):
        for item in sorted(self._list.selectedItems(), key=self._list.row, reverse=True):
            r = self._list.row(item)
            self._list.takeItem(r)
            del self._files[r]

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
            self._list.item(self._list.count() - 1).setToolTip(p)

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
            self.say("Add input files first.", error=True)
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
            self.say("No data rows after filtering.", error=True)
            return
        try:
            write_hkl_rows(out_path, rows)
        except Exception as e:
            self.say(f"Could not write {out_path}: {e}", error=True)
            return
        msg = f"Wrote {len(rows)} rows to {os.path.basename(out_path)}"
        self._log_msg(msg)
        self.say(msg)


def main(argv=None):
    return run_tool(CombinerWindow)


if __name__ == '__main__':
    main()

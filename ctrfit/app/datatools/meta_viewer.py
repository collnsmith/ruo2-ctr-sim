"""SXRD HDF5 / Spec Metadata Viewer
=================================
Quick tool for inspecting:
  - HDF5 file structure and metadata (per scan from a directory)
  - Spec file columns per scan

Qt window of the data tools (ctr_data_tools.py); the logic is in ctrfit.datatools.
"""
import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
                             QListWidget, QLineEdit, QSpinBox, QTextEdit, QTreeWidget, QTreeWidgetItem,
                             QTabWidget, QTableWidget, QTableWidgetItem, QSplitter, QComboBox,
                             QListWidgetItem)

from ctrfit.app.datatools.common import StatusMixin, group, hint, mono_font, row, run_tool
from ctrfit.datatools.h5meta import describe_h5, scan_h5_dir_with_masters
from ctrfit.datatools.images import HAS_H5PY, h5py
from ctrfit.datatools.specfile import parse_spec_file


def populate_tree(parent_item, h5_obj, max_depth=10, depth=0):
    if depth > max_depth:
        QTreeWidgetItem(parent_item, ["...", "(depth limit)"])
        return
    for attr_name in h5_obj.attrs:
        try:
            val = h5_obj.attrs[attr_name]
            val_str = str(val)
            if len(val_str) > 100: val_str = val_str[:97] + "..."
        except Exception as e:
            val_str = f"<error: {e}>"
        item = QTreeWidgetItem(parent_item, [f"@{attr_name}", f"attr = {val_str}"])
        item.setForeground(0, QColor(30, 90, 160))
        item.setForeground(1, QColor(30, 90, 160))
    if isinstance(h5_obj, h5py.Group):
        for name in h5_obj:
            try:
                child = h5_obj[name]
                item = QTreeWidgetItem(parent_item, [name, describe_h5(child)])
                if isinstance(child, h5py.Group):
                    item.setForeground(0, QColor(140, 90, 0))
                populate_tree(item, child, max_depth, depth + 1)
            except Exception as e:
                QTreeWidgetItem(parent_item, [name, f"<error: {e}>"])


class MainWindow(StatusMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SXRD Metadata Viewer")
        self.resize(1150, 750)

        self._all_files = []
        self._filtered_files = []
        self._spec_scans = {}

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_h5_tab(), "HDF5 images")
        self._tabs.addTab(self._build_spec_tab(), "SPEC file")
        self.setCentralWidget(self._tabs)
        self.say("Pick an image folder (HDF5) or a SPEC file.")

    def _build_h5_tab(self):
        if not HAS_H5PY:
            tab = QWidget()
            QVBoxLayout(tab).addWidget(hint("h5py is not installed: pip install h5py"))
            return tab
        src, f = group("Folder")
        self._h5_dir = QLineEdit()
        self._h5_dir.setReadOnly(True)
        bb = QPushButton("Browse…")
        bb.clicked.connect(self._h5_browse)
        f.addRow(row(self._h5_dir, bb))
        self._h5_sf = QSpinBox()
        self._h5_sf.setRange(0, 99999)
        self._h5_st = QSpinBox()
        self._h5_st.setRange(0, 99999)
        self._h5_st.setValue(99999)
        for w in (self._h5_sf, self._h5_st):
            w.valueChanged.connect(self._h5_apply_filter)
        f.addRow("Scans", row(self._h5_sf, QLabel("to"), self._h5_st, stretch_first=False))
        self._h5_show = QComboBox()
        self._h5_show.addItems(['All', 'Data only', 'Master only'])
        self._h5_show.currentTextChanged.connect(self._h5_apply_filter)
        f.addRow("Show", self._h5_show)
        self._h5_count = hint("0 files")
        f.addRow(self._h5_count)

        files, fl = group("Files", form=False)
        self._h5_list = QListWidget()
        self._h5_list.currentItemChanged.connect(self._h5_selected)
        fl.addWidget(self._h5_list)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(src)
        ll.addWidget(files, 1)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self._h5_cur_path = QLineEdit()
        self._h5_cur_path.setReadOnly(True)
        self._h5_find = QLineEdit()
        self._h5_find.setPlaceholderText("Find: highlight matching names and values")
        self._h5_find.textChanged.connect(self._h5_apply_find)
        be = QPushButton("Expand all")
        be.clicked.connect(lambda: self._h5_tree.expandAll())
        bc = QPushButton("Collapse all")
        bc.clicked.connect(lambda: self._h5_tree.collapseAll())
        rl.addWidget(row(QLabel("File"), self._h5_cur_path, stretch_first=False))
        rl.addWidget(row(self._h5_find, be, bc))
        self._h5_tree = QTreeWidget()
        self._h5_tree.setHeaderLabels(["Name", "Info"])
        self._h5_tree.setColumnWidth(0, 280)
        self._h5_tree.setAlternatingRowColors(True)
        self._h5_tree.setFont(mono_font())
        rl.addWidget(self._h5_tree, 1)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([380, 800])
        return split

    def _h5_browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select HDF5 directory")
        if d:
            self._h5_dir.setText(d)
            self._h5_reload()

    def _h5_reload(self):
        d = self._h5_dir.text()
        if not os.path.isdir(d):
            return
        # Use our helper that keeps master files
        self._all_files = scan_h5_dir_with_masters(d)
        if self._all_files:
            scans = sorted(set(f['scan'] for f in self._all_files))
            for w, v in ((self._h5_sf, scans[0]), (self._h5_st, scans[-1])):
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)
        self._h5_apply_filter()
        self.say(f"{len(self._all_files)} HDF5 files in {d}")

    def _h5_apply_filter(self):
        s0, s1 = self._h5_sf.value(), self._h5_st.value()
        show = self._h5_show.currentText()
        files = [f for f in self._all_files if s0 <= f['scan'] <= s1]
        if show == 'Data only':
            files = [f for f in files if not f.get('is_master')]
        elif show == 'Master only':
            files = [f for f in files if f.get('is_master')]
        self._filtered_files = files
        n_master = sum(1 for f in files if f.get('is_master'))
        n_data = len(files) - n_master
        self._h5_count.setText(
            f"{len(files)} files ({n_data} data, {n_master} master), "
            f"{len(set(f['scan'] for f in files))} scans")
        self._h5_list.clear()
        for f in files:
            name = os.path.basename(f['path'])
            if f.get('is_master'):
                item = QListWidgetItem(f"[MASTER] S{f['scan']:03d}  {name}")
                item.setForeground(QColor(170, 90, 0))
            else:
                item = QListWidgetItem(f"S{f['scan']:03d} I{f['image']:03d}  {name}")
            self._h5_list.addItem(item)

    def _h5_selected(self, cur, prev):
        if cur is None: return
        idx = self._h5_list.row(cur)
        if not (0 <= idx < len(self._filtered_files)): return
        path = self._filtered_files[idx]['path']
        self._h5_cur_path.setText(path)
        self._h5_tree.clear()
        try:
            with h5py.File(path, 'r') as f:
                root = QTreeWidgetItem(self._h5_tree, ["/", f"root ({len(f)} children)"])
                root.setForeground(0, QColor(140, 90, 0))
                populate_tree(root, f)
                self._h5_tree.expandItem(root)
        except Exception as e:
            QTreeWidgetItem(self._h5_tree, ["ERROR", str(e)])

    def _h5_apply_find(self, text):
        text = text.strip().lower()
        def recurse(item):
            matched = False
            if text:
                if text in item.text(0).lower() or text in item.text(1).lower():
                    matched = True
            child_matched = False
            for i in range(item.childCount()):
                if recurse(item.child(i)):
                    child_matched = True
            if matched:
                item.setBackground(0, QColor(255, 236, 150))
                item.setBackground(1, QColor(255, 236, 150))
            else:
                item.setBackground(0, QColor(0, 0, 0, 0))
                item.setBackground(1, QColor(0, 0, 0, 0))
            if (matched or child_matched) and text:
                item.setExpanded(True)
            return matched or child_matched
        for i in range(self._h5_tree.topLevelItemCount()):
            recurse(self._h5_tree.topLevelItem(i))

    def _build_spec_tab(self):
        src, f = group("SPEC file")
        self._sp_path = QLineEdit()
        self._sp_path.setReadOnly(True)
        bb = QPushButton("Browse…")
        bb.clicked.connect(self._sp_browse)
        f.addRow(row(self._sp_path, bb))
        self._sp_from = QSpinBox()
        self._sp_from.setRange(0, 99999)
        self._sp_to = QSpinBox()
        self._sp_to.setRange(0, 99999)
        self._sp_to.setValue(99999)
        for w in (self._sp_from, self._sp_to):
            w.valueChanged.connect(self._sp_refresh_list)
        f.addRow("Scans", row(self._sp_from, QLabel("to"), self._sp_to, stretch_first=False))
        scans, sl = group("Scans", form=False)
        self._sp_list = QListWidget()
        self._sp_list.currentItemChanged.connect(self._sp_selected)
        sl.addWidget(self._sp_list)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(src)
        ll.addWidget(scans, 1)

        info, il = group("Scan")
        self._sp_cmd = QLineEdit()
        self._sp_cmd.setReadOnly(True)
        self._sp_cols = QLineEdit()
        self._sp_cols.setReadOnly(True)
        self._sp_cols.setFont(mono_font())
        il.addRow("Command", self._sp_cmd)
        il.addRow("Columns", self._sp_cols)
        self._sp_table = QTableWidget()
        self._sp_table.setAlternatingRowColors(True)
        self._sp_summary = QTextEdit()
        self._sp_summary.setReadOnly(True)
        self._sp_summary.setFont(mono_font())
        self._sp_motors = QTextEdit()
        self._sp_motors.setReadOnly(True)
        self._sp_motors.setFont(mono_font())
        views = QTabWidget()
        views.addTab(self._sp_table, "Data (first 200 rows)")
        views.addTab(self._sp_summary, "Column summary")
        views.addTab(self._sp_motors, "Motor positions")
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(info)
        rl.addWidget(views, 1)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([380, 800])
        return split

    def _sp_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select spec file", "", "Spec files (*.spec *.dat *.txt);;All (*)")
        if not path: return
        try:
            self._spec_scans = parse_spec_file(path)
        except Exception as e:
            self.say(f"Could not read {path}: {e}", error=True)
            return
        self._sp_path.setText(path)
        if self._spec_scans:
            nums = sorted(self._spec_scans.keys())
            for w, v in ((self._sp_from, nums[0]), (self._sp_to, nums[-1])):
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)
        self._sp_refresh_list()
        self.say(f"{len(self._spec_scans)} scans in {os.path.basename(path)}")

    def _sp_refresh_list(self):
        self._sp_list.clear()
        s0, s1 = self._sp_from.value(), self._sp_to.value()
        for sn in sorted(self._spec_scans.keys()):
            if s0 <= sn <= s1:
                scan = self._spec_scans[sn]
                cmd = scan.scan_cmd
                if len(cmd) > 50: cmd = cmd[:47] + "..."
                self._sp_list.addItem(f"S{sn:04d}  {cmd}")

    def _sp_selected(self, cur, prev):
        if cur is None: return
        try:
            sn = int(cur.text().split()[0][1:])
        except (ValueError, IndexError):
            return
        scan = self._spec_scans.get(sn)
        if scan is None: return
        self._sp_cmd.setText(scan.scan_cmd)
        self._sp_cols.setText("  ".join(scan.columns))

        n = min(scan.npoints, 200)
        self._sp_table.clear()
        self._sp_table.setColumnCount(len(scan.columns))
        self._sp_table.setRowCount(n)
        self._sp_table.setHorizontalHeaderLabels(scan.columns)
        for c, col_name in enumerate(scan.columns):
            arr = scan.data.get(col_name)
            if arr is None: continue
            for r in range(n):
                self._sp_table.setItem(r, c, QTableWidgetItem(f"{arr[r]:g}"))
        self._sp_table.resizeColumnsToContents()

        lines = [f"Scan {sn}: {scan.npoints} points, {len(scan.columns)} columns", ""]
        lines.append(f"{'column':<16}{'min':>12}{'max':>12}{'mean':>12}{'std':>12}")
        for col_name in scan.columns:
            arr = scan.data.get(col_name)
            if arr is None or len(arr) == 0: continue
            lines.append(f"{col_name:<16}{arr.min():>12.4g}{arr.max():>12.4g}"
                         f"{arr.mean():>12.4g}{arr.std():>12.4g}")
        self._sp_summary.setPlainText("\n".join(lines))
        mp = scan.motor_positions
        self._sp_motors.setPlainText("\n".join(f"{k:<16}{v:>14.6g}" for k, v in mp.items())
                                     or "no #O / #P motor lines in this file")


def main(argv=None):
    return run_tool(MainWindow)


if __name__ == '__main__':
    main()

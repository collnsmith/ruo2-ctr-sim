"""SXRD HDF5 / Spec Metadata Viewer
=================================
Quick tool for inspecting:
  - HDF5 file structure and metadata (per scan from a directory)
  - Spec file columns per scan

Qt window of the data tools (ctr_data_tools.py); the logic is in ctrfit.datatools.
"""
import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QListWidget, QGroupBox, QLineEdit, QSpinBox, QTextEdit,
                             QMessageBox, QTreeWidget, QTreeWidgetItem, QTabWidget, QTableWidget,
                             QTableWidgetItem, QSplitter, QComboBox, QListWidgetItem)

from ctrfit.app.datatools.common import run_tool
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
        item.setForeground(0, QColor(150, 180, 220))
        item.setForeground(1, QColor(150, 180, 220))
    if isinstance(h5_obj, h5py.Group):
        for name in h5_obj:
            try:
                child = h5_obj[name]
                item = QTreeWidgetItem(parent_item, [name, describe_h5(child)])
                if isinstance(child, h5py.Group):
                    item.setForeground(0, QColor(220, 220, 120))
                populate_tree(item, child, max_depth, depth + 1)
            except Exception as e:
                QTreeWidgetItem(parent_item, [name, f"<error: {e}>"])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SXRD Metadata Viewer")
        self.resize(1150, 750)

        self._all_files = []
        self._filtered_files = []
        self._spec_scans = {}

        cw = QWidget(); self.setCentralWidget(cw)
        lv = QVBoxLayout(cw)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_h5_tab(), "HDF5 Metadata")
        self._tabs.addTab(self._build_spec_tab(), "Spec File")
        lv.addWidget(self._tabs)

    def _build_h5_tab(self):
        tab = QWidget()
        lv = QVBoxLayout(tab)

        if not HAS_H5PY:
            lv.addWidget(QLabel(
                "<b style='color:#ff8080'>h5py not installed.</b><br>"
                "Install: pip install h5py"))
            return tab

        top = QGroupBox("Directory and scan range")
        tl = QHBoxLayout(top)
        tl.addWidget(QLabel("Dir:"))
        self._h5_dir = QLineEdit(); self._h5_dir.setReadOnly(True)
        tl.addWidget(self._h5_dir)
        bb = QPushButton("Browse...")
        bb.clicked.connect(self._h5_browse)
        tl.addWidget(bb)
        tl.addWidget(QLabel("Scans:"))
        self._h5_sf = QSpinBox(); self._h5_sf.setRange(0, 99999)
        tl.addWidget(self._h5_sf)
        tl.addWidget(QLabel("-"))
        self._h5_st = QSpinBox(); self._h5_st.setRange(0, 99999); self._h5_st.setValue(99999)
        tl.addWidget(self._h5_st)
        tl.addWidget(QLabel("Show:"))
        self._h5_show = QComboBox()
        self._h5_show.addItems(['All', 'Data only', 'Master only'])
        self._h5_show.currentTextChanged.connect(self._h5_apply_filter)
        tl.addWidget(self._h5_show)
        bf = QPushButton("Apply")
        bf.clicked.connect(self._h5_apply_filter)
        tl.addWidget(bf)
        lv.addWidget(top)

        self._h5_count = QLabel("0 files")
        lv.addWidget(self._h5_count)

        split = QSplitter(Qt.Horizontal)
        self._h5_list = QListWidget()
        self._h5_list.currentItemChanged.connect(self._h5_selected)
        split.addWidget(self._h5_list)

        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("File:"))
        self._h5_cur_path = QLineEdit(); self._h5_cur_path.setReadOnly(True)
        path_row.addWidget(self._h5_cur_path)
        rl.addLayout(path_row)

        self._h5_tree = QTreeWidget()
        self._h5_tree.setHeaderLabels(["Name", "Info"])
        self._h5_tree.setColumnWidth(0, 260)
        self._h5_tree.setAlternatingRowColors(True)
        self._h5_tree.setFont(QFont("Consolas", 9))
        rl.addWidget(self._h5_tree)

        fr = QHBoxLayout()
        fr.addWidget(QLabel("Find:"))
        self._h5_find = QLineEdit()
        self._h5_find.setPlaceholderText("highlight matching nodes...")
        self._h5_find.textChanged.connect(self._h5_apply_find)
        fr.addWidget(self._h5_find)
        be = QPushButton("Expand all"); be.clicked.connect(lambda: self._h5_tree.expandAll())
        fr.addWidget(be)
        bc = QPushButton("Collapse all"); bc.clicked.connect(lambda: self._h5_tree.collapseAll())
        fr.addWidget(bc)
        rl.addLayout(fr)

        split.addWidget(right)
        split.setSizes([300, 800])
        lv.addWidget(split, stretch=1)
        return tab

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
            self._h5_sf.setValue(scans[0])
            self._h5_st.setValue(scans[-1])
        self._h5_apply_filter()

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
                item.setForeground(QColor(255, 200, 100))
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
                root.setForeground(0, QColor(255, 220, 100))
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
                item.setBackground(0, QColor(80, 110, 50))
                item.setBackground(1, QColor(80, 110, 50))
            else:
                item.setBackground(0, QColor(0, 0, 0, 0))
                item.setBackground(1, QColor(0, 0, 0, 0))
            if (matched or child_matched) and text:
                item.setExpanded(True)
            return matched or child_matched
        for i in range(self._h5_tree.topLevelItemCount()):
            recurse(self._h5_tree.topLevelItem(i))

    def _build_spec_tab(self):
        tab = QWidget()
        lv = QVBoxLayout(tab)

        top = QGroupBox("Spec file")
        tl = QHBoxLayout(top)
        tl.addWidget(QLabel("File:"))
        self._sp_path = QLineEdit(); self._sp_path.setReadOnly(True)
        tl.addWidget(self._sp_path)
        bb = QPushButton("Browse...")
        bb.clicked.connect(self._sp_browse)
        tl.addWidget(bb)
        lv.addWidget(top)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Scans:"))
        fr = QHBoxLayout()
        fr.addWidget(QLabel("From:"))
        self._sp_from = QSpinBox(); self._sp_from.setRange(0, 99999)
        fr.addWidget(self._sp_from)
        fr.addWidget(QLabel("To:"))
        self._sp_to = QSpinBox(); self._sp_to.setRange(0, 99999); self._sp_to.setValue(99999)
        fr.addWidget(self._sp_to)
        bfilt = QPushButton("Filter")
        bfilt.clicked.connect(self._sp_refresh_list)
        fr.addWidget(bfilt)
        ll.addLayout(fr)

        self._sp_list = QListWidget()
        self._sp_list.currentItemChanged.connect(self._sp_selected)
        ll.addWidget(self._sp_list)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        self._sp_cmd = QLineEdit(); self._sp_cmd.setReadOnly(True)
        rl.addWidget(QLabel("Command:"))
        rl.addWidget(self._sp_cmd)
        rl.addWidget(QLabel("Columns:"))
        self._sp_cols = QLineEdit(); self._sp_cols.setReadOnly(True)
        self._sp_cols.setStyleSheet("font-family: Consolas, monospace;")
        rl.addWidget(self._sp_cols)

        rl.addWidget(QLabel("Data (first 200 rows):"))
        self._sp_table = QTableWidget()
        self._sp_table.setAlternatingRowColors(True)
        rl.addWidget(self._sp_table)

        rl.addWidget(QLabel("Column summary:"))
        self._sp_summary = QTextEdit(); self._sp_summary.setReadOnly(True)
        self._sp_summary.setMaximumHeight(160)
        self._sp_summary.setStyleSheet("font-family: Consolas, monospace;")
        rl.addWidget(self._sp_summary)

        split.addWidget(right)
        split.setSizes([260, 860])
        lv.addWidget(split, stretch=1)
        return tab

    def _sp_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select spec file", "", "Spec files (*.spec *.dat *.txt);;All (*)")
        if not path: return
        try:
            self._spec_scans = parse_spec_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed: {e}")
            return
        self._sp_path.setText(path)
        if self._spec_scans:
            nums = sorted(self._spec_scans.keys())
            self._sp_from.setValue(nums[0])
            self._sp_to.setValue(nums[-1])
        self._sp_refresh_list()

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


def main(argv=None):
    return run_tool(MainWindow)


if __name__ == '__main__':
    main()

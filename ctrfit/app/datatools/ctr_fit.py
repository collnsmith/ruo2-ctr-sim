"""SXRD CTR Fitting Tool
Fits crystal truncation rod models to integrated intensity data.
Works with exported peak fit results from sxrd_plotter.py.

Models:
  1. Substrate only (kinematic CTR)
  2. Substrate + epitaxial film with roughness
  3. Substrate + film + surface adlayer
  4. Substrate + film + liquid interface
  5. Multi-layer / graded interface

Launch standalone or from sxrd_plotter via the CTR Fitting button.

Qt window of the data tools (ctr_data_tools.py); the logic is in ctrfit.datatools.
"""
import configparser
import json
import os

import numpy as np
from PyQt5.QtCore import Qt, QEvent, QObject
from PyQt5.QtWidgets import (QFormLayout, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
                             QSpinBox, QDoubleSpinBox, QLineEdit, QFileDialog, QCheckBox, QSplitter,
                             QListWidget, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView)
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from scipy.optimize import least_squares

from ctrfit.app.datatools.common import (StatusMixin, group, hint, mono_font, primary_button, row, run_tool,
                                         sidebar)
from ctrfit.datatools.ctrmodels import CTR_MODELS, ScanGroupManager
from ctrfit.datatools.paths import settings_file


class _ScrollBlockFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            if isinstance(obj, (QSpinBox, QDoubleSpinBox, QComboBox)):
                if not obj.hasFocus():
                    event.ignore(); return True
        return False


class CTRFitWindow(StatusMixin, QMainWindow):
    def __init__(self, data_source=None):
        """data_source: optional dict of {scan_number: {'L': array, 'I': array}}
        from the plotter's peak fit results."""
        super().__init__()
        self.setWindowTitle("SXRD CTR Fitting")
        self.setMinimumSize(1200, 750)

        self._group_mgr = ScanGroupManager()
        self._imported_data = data_source or {}
        self._all_scan_data = {}
        self._group_data = {}  # group_name -> {'L': array, 'I': array}
        self._current_group = None
        self._fit_results = {}
        self._param_spins = []
        self._param_fixed = []

        self._setup_ui()
        self._install_scroll_blocker()
        self._load_settings()

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    def _save_settings(self):
        c = configparser.ConfigParser()
        c['window'] = {
            'geometry': self.saveGeometry().toBase64().data().decode(),
            'hsplitter': self._hsplitter.saveState().toBase64().data().decode(),
        }
        try:
            with open(settings_file("sxrd_ctr_settings.ini"), 'w') as f: c.write(f)
        except Exception: pass

    def _load_settings(self):
        if not os.path.isfile(settings_file("sxrd_ctr_settings.ini")): return
        c = configparser.ConfigParser()
        try: c.read(settings_file("sxrd_ctr_settings.ini"))
        except: return
        from PyQt5.QtCore import QByteArray
        def get(s, k, d=''):
            try: return c.get(s, k)
            except: return d
        geo = get('window', 'geometry')
        if geo: self.restoreGeometry(QByteArray.fromBase64(geo.encode()))
        hs = get('window', 'hsplitter')
        if hs: self._hsplitter.restoreState(QByteArray.fromBase64(hs.encode()))

    def _setup_ui(self):
        self._hsplitter = QSplitter(Qt.Horizontal)

        dg, dl = group("Data", form=False)
        b_pf, b_2c, b_pl = QPushButton("Peak fits…"), QPushButton("L vs I…"), QPushButton("Plotter export…")
        b_pf.setToolTip("Peak fit results (.txt) from the plotter's peak fitting (scan, area); L from the SPEC data")
        b_2c.setToolTip("A two-column text file: L and I")
        b_pl.setToolTip("A multi-trace export from the plotter; each trace becomes a group")
        b_pf.clicked.connect(self._import_from_file)
        b_2c.clicked.connect(self._import_2col)
        b_pl.clicked.connect(self._import_plotter_export)
        dl.addWidget(row(b_pf, b_2c, b_pl, stretch_first=False))
        self._data_info = hint("No data loaded")
        dl.addWidget(self._data_info)
        self._l_col = QComboBox()
        self._l_col.addItems(["L", "H", "K"])
        self._l_col.setEditable(True)
        self._l_col.setToolTip("Column of the SPEC data that gives the rod coordinate of each peak-fit scan")
        lf = QFormLayout()
        lf.addRow("Rod coordinate column", self._l_col)
        dl.addLayout(lf)

        gg, gl = group("Scan groups (one per rod)")
        self._group_list = QListWidget()
        self._group_list.setMaximumHeight(120)
        self._group_list.currentRowChanged.connect(self._on_group_selected)
        gl.addRow(self._group_list)
        self._group_name_edit = QLineEdit()
        self._group_name_edit.setPlaceholderText("new group name, e.g. 0 0 L")
        btn_ag, btn_rg = QPushButton("Add"), QPushButton("Remove")
        btn_ag.clicked.connect(self._add_group)
        btn_rg.clicked.connect(self._remove_group)
        gl.addRow(row(self._group_name_edit, btn_ag, btn_rg))
        self._scans_edit = QLineEdit()
        self._scans_edit.setPlaceholderText("100,101,102,103")
        self._scans_edit.editingFinished.connect(self._update_group_scans)
        gl.addRow("Scans", self._scans_edit)
        self._hk_edit = QLineEdit()
        self._hk_edit.setPlaceholderText("0 0")
        self._hk_edit.editingFinished.connect(self._update_group_hk)
        btn_build = QPushButton("Build rod from group")
        btn_build.clicked.connect(self._build_ctr_from_group)
        gl.addRow("H K", row(self._hk_edit, btn_build))
        btn_sg, btn_lg = QPushButton("Save groups…"), QPushButton("Load groups…")
        btn_sg.clicked.connect(self._save_groups)
        btn_lg.clicked.connect(self._load_groups)
        gl.addRow(row(btn_sg, btn_lg, 1, stretch_first=False))

        mg, ml = group("Model", form=False)
        self._model_combo = QComboBox()
        self._model_combo.addItems(list(CTR_MODELS.keys()))
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        ml.addWidget(self._model_combo)
        self._model_desc = hint("")
        ml.addWidget(self._model_desc)
        self._param_table = QTableWidget()
        self._param_table.setColumnCount(4)
        self._param_table.setHorizontalHeaderLabels(["Parameter", "Value", "Fix", "Bounds"])
        hh = self._param_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        self._param_table.setColumnWidth(1, 115)
        self._param_table.setColumnWidth(2, 30)
        self._param_table.verticalHeader().setVisible(False)
        self._param_table.setMinimumHeight(220)
        self._param_layout = ml
        ml.addWidget(self._param_table)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(sidebar(dg, gg, mg), 1)
        btn_sim = QPushButton("Simulate (no fit)")
        btn_sim.clicked.connect(self._simulate)
        lv.addWidget(row(primary_button("Fit current group", self._run_fit, "Ctrl+R", self), btn_sim))
        btn_exp_data, btn_exp_fit, btn_exp_plot = (QPushButton("Export data…"), QPushButton("Export fit…"),
                                                   QPushButton("Save plot…"))
        btn_exp_data.setToolTip("The current group's L and I")
        btn_exp_fit.setToolTip("The fitted parameters of every group")
        btn_exp_data.clicked.connect(self._export_group_data)
        btn_exp_fit.clicked.connect(self._export_fit_params)
        btn_exp_plot.clicked.connect(self._save_plot)
        lv.addWidget(row(btn_exp_data, btn_exp_fit, btn_exp_plot, stretch_first=False))
        self._hsplitter.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self._fig = Figure(figsize=(8, 6), dpi=100, layout="constrained")
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        rv.addWidget(NavigationToolbar(self._canvas, self))
        rv.addWidget(self._canvas, stretch=1)
        self._fit_info = QLabel("—")
        self._fit_info.setWordWrap(True)
        self._fit_info.setFont(mono_font())
        self._fit_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rv.addWidget(self._fit_info)
        self._hsplitter.addWidget(right)
        self._hsplitter.setStretchFactor(0, 0)
        self._hsplitter.setStretchFactor(1, 1)
        self._hsplitter.setSizes([420, 820])
        self.setCentralWidget(self._hsplitter)
        self._on_model_changed()
        self.say("Import data (peak fits, L vs I or a plotter export), choose a model, then Fit.")

    def _install_scroll_blocker(self):
        self._scroll_filter = _ScrollBlockFilter(self)
        for w in self.findChildren(QSpinBox):
            w.setFocusPolicy(Qt.StrongFocus); w.installEventFilter(self._scroll_filter)
        for w in self.findChildren(QDoubleSpinBox):
            w.setFocusPolicy(Qt.StrongFocus); w.installEventFilter(self._scroll_filter)
        for w in self.findChildren(QComboBox):
            w.setFocusPolicy(Qt.StrongFocus); w.installEventFilter(self._scroll_filter)

    # ---- Data import ----
    def _import_from_file(self):
        """Import peak fit results (scan, area, fit_area, num_area, model, params).

        Parses the full table including the 'params' column (amp, cen, sigma, ...)
        so per-scan fit info is preserved and visible.

        If a companion CTR fit JSON exists next to the file, offers to also load
        groups and CTR fit results from it.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Import peak fit results", "", "Text (*.txt);;All (*)")
        if not path: return

        n_loaded = 0
        models_seen = {}
        try:
            with open(path, 'r') as f:
                header = None
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if header is None:
                        header = [h.strip() for h in parts]
                        continue
                    if len(parts) < 2:
                        continue
                    try:
                        sn = int(parts[0])
                    except ValueError:
                        continue

                    entry = {}
                    # Parse each column by header name
                    for i, h in enumerate(header):
                        if i >= len(parts):
                            break
                        val = parts[i].strip()
                        if h == 'scan':
                            continue
                        if h == 'params':
                            # Parse 'amp=0.165,cen=-121.56,sigma=0.016,...'
                            for kv in val.split(','):
                                kv = kv.strip()
                                if '=' not in kv:
                                    continue
                                k, v = kv.split('=', 1)
                                try:
                                    entry[k.strip()] = float(v.strip())
                                except ValueError:
                                    entry[k.strip()] = v.strip()
                        elif h == 'model':
                            entry['model'] = val
                            models_seen[val] = models_seen.get(val, 0) + 1
                        else:
                            try:
                                entry[h] = float(val)
                            except ValueError:
                                entry[h] = val
                    self._all_scan_data[sn] = entry
                    n_loaded += 1

            # Summary message
            lines = [f"Loaded {n_loaded} scans from {os.path.basename(path)}"]
            for m, c in models_seen.items():
                lines.append(f"  {c}x {m}")
            # Show available per-scan fit keys for one example
            if self._all_scan_data:
                first = next(iter(self._all_scan_data.values()))
                keys = [k for k in first if k not in ('model',)]
                lines.append(f"  Per-scan keys: {', '.join(keys)}")
            self._data_info.setText("\n".join(lines))

            # Check for companion CTR fit JSON
            json_candidates = [
                os.path.splitext(path)[0] + '_ctrfit.json',
                os.path.splitext(path)[0] + '.json',
                os.path.join(os.path.dirname(path), 'ctr_fit.json'),
            ]
            found_json = next((j for j in json_candidates if os.path.exists(j)), None)
            if found_json:
                reply = QMessageBox.question(
                    self, "Companion file found",
                    f"Found CTR fit state:\n{found_json}\n\nLoad groups and fit results too?",
                    QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self._import_ctrfit_json(found_json)

        except Exception as e:
            self.say(str(e), error=True)

    def _import_ctrfit_json(self, path):
        """Load groups and fit results from a CTR fitter JSON export."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.say(f"Could not read JSON: {e}", error=True)
            return

        # Groups
        groups = data.get('groups', {})
        for name, ginfo in groups.items():
            self._group_mgr.groups[name] = {
                'scans': ginfo.get('scans', []),
                'hk': ginfo.get('hk', ''),
                'notes': ginfo.get('notes', ''),
            }

        # Fit results
        fits = data.get('fits', {})
        for name, finfo in fits.items():
            self._fit_results[name] = {
                'params': list(finfo.get('params', [])),
                'model': finfo.get('model', ''),
                'chi2': float(finfo.get('chi2', 0.0)),
            }

        # Group data if present
        gdata = data.get('group_data', {})
        for name, d in gdata.items():
            try:
                self._group_data[name] = {
                    'L': np.array(d['L'], dtype=float),
                    'I': np.array(d['I'], dtype=float),
                }
            except (KeyError, TypeError):
                pass

        self._refresh_group_list()
        # Select first group if any
        if groups:
            first = next(iter(groups.keys()))
            self._current_group = first
            # Apply saved params into parameter editor if fit exists
            if first in self._fit_results:
                fr = self._fit_results[first]
                model_name = fr.get('model', '')
                if model_name:
                    idx = self._model_combo.findText(model_name)
                    if idx >= 0:
                        self._model_combo.setCurrentIndex(idx)
                self._set_param_values(fr['params'])
            self._update_plot()

        self.say(f"Loaded {len(groups)} groups and {len(fits)} fit results")

    def _import_2col(self):
        """Import simple L vs I text file."""
        path, _ = QFileDialog.getOpenFileName(self, "Import L vs I", "", "Text (*.txt *.dat);;All (*)")
        if not path: return
        try:
            data = np.loadtxt(path, comments='#')
            name = os.path.splitext(os.path.basename(path))[0]
            self._group_mgr.add_group(name, scans=[])
            self._group_data[name] = {'L': data[:, 0], 'I': data[:, 1]}
            self._refresh_group_list()
            self._data_info.setText(f"Loaded {len(data)} points as '{name}'")
            self._update_plot()
        except Exception as e:
            self.say(str(e), error=True)

    def _import_plotter_export(self):
        """Import multi-trace export from the plotter (plot_data.txt).
        Each trace block becomes a group or is added to an existing group."""
        path, _ = QFileDialog.getOpenFileName(self, "Import plotter export", "", "Text (*.txt);;All (*)")
        if not path: return
        try:
            traces = []
            current_label = None; current_x = []; current_y = []
            current_meta = {}
            with open(path, 'r') as f:
                for line in f:
                    line = line.rstrip('\n')
                    if line.startswith('# Trace:'):
                        # Save previous trace
                        if current_label and current_x:
                            traces.append({'label': current_label, 'x': np.array(current_x),
                                          'y': np.array(current_y), **current_meta})
                        # Parse: # Trace: S100_ROI1 (scan=100, y=[bg] ROI1)
                        rest = line[len('# Trace:'):].strip()
                        current_label = rest.split('(')[0].strip()
                        current_meta = {}
                        if 'scan=' in rest:
                            try:
                                sn = int(rest.split('scan=')[1].split(',')[0].split(')')[0])
                                current_meta['scan'] = sn
                            except: pass
                        current_x = []; current_y = []
                        continue
                    if line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            x_val = float(parts[0]); y_val = float(parts[1])
                            current_x.append(x_val); current_y.append(y_val)
                        except ValueError:
                            # Header line for a trace block — skip
                            if current_label is None:
                                current_label = parts[1] if len(parts) > 1 else 'trace'
                            continue
                # Save last trace
                if current_label and current_x:
                    traces.append({'label': current_label, 'x': np.array(current_x),
                                  'y': np.array(current_y), **current_meta})

            if not traces:
                self.say("No traces found in file", error=True); return

            # Create a group for each trace
            for t in traces:
                name = t['label']
                scans = [t['scan']] if 'scan' in t else []
                self._group_mgr.add_group(name, scans=scans)
                self._group_data[name] = {'L': t['x'], 'I': t['y']}

            self._refresh_group_list()
            self._data_info.setText(f"Imported {len(traces)} traces from {os.path.basename(path)}")
            self._update_plot()
        except Exception as e:
            self.say(f"Import failed:\n{e}", error=True)

    # ---- Scan groups ----
    def _add_group(self):
        name = self._group_name_edit.text().strip()
        if not name: name = f"Group {len(self._group_mgr.groups) + 1}"
        self._group_mgr.add_group(name)
        self._refresh_group_list()

    def _remove_group(self):
        row = self._group_list.currentRow()
        if row >= 0:
            name = list(self._group_mgr.groups.keys())[row]
            self._group_mgr.remove_group(name)
            self._refresh_group_list()

    def _refresh_group_list(self):
        self._group_list.clear()
        for name, g in self._group_mgr.groups.items():
            n_scans = len(g['scans'])
            self._group_list.addItem(f"{name} ({n_scans} scans, HK={g['hk']})")

    def _on_group_selected(self, row):
        if row < 0: return
        names = list(self._group_mgr.groups.keys())
        if row >= len(names): return
        name = names[row]
        g = self._group_mgr.groups[name]
        self._current_group = name
        self._scans_edit.setText(",".join(str(s) for s in g['scans']))
        self._hk_edit.setText(g['hk'])
        # Load fit results if available
        if name in self._fit_results:
            fr = self._fit_results[name]
            idx = self._model_combo.findText(fr['model'])
            if idx >= 0: self._model_combo.setCurrentIndex(idx)
            self._set_param_values(fr['params'])

    def _update_group_scans(self):
        if self._current_group is None: return
        text = self._scans_edit.text().strip()
        scans = []
        for part in text.replace(' ', ',').split(','):
            part = part.strip()
            if not part: continue
            if '-' in part:
                try:
                    a, b = part.split('-')
                    scans.extend(range(int(a), int(b) + 1))
                except: pass
            else:
                try: scans.append(int(part))
                except: pass
        self._group_mgr.groups[self._current_group]['scans'] = sorted(set(scans))
        self._refresh_group_list()

    def _update_group_hk(self):
        if self._current_group is None: return
        self._group_mgr.groups[self._current_group]['hk'] = self._hk_edit.text().strip()
        self._refresh_group_list()

    def _get_group_data(self, name=None):
        """Get L, I arrays for a group."""
        if name is None: name = self._current_group
        if name is None: return None, None
        gd = self._group_data.get(name)
        if gd is None: return None, None
        return gd['L'], gd['I']

    def _build_ctr_from_group(self):
        """Build L vs I arrays from scan group using peak fit areas."""
        if self._current_group is None:
            self.say("Select a group first", error=True); return
        g = self._group_mgr.groups[self._current_group]
        scans = g['scans']
        if not scans:
            self.say("No scans in group", error=True); return

        l_col = self._l_col.currentText()
        L_vals = []; I_vals = []

        for sn in scans:
            sd = self._all_scan_data.get(sn, self._imported_data.get(sn, {}))
            if not sd: continue
            L_val = sd.get(l_col, sd.get('L', None))
            if L_val is None: continue
            if isinstance(L_val, np.ndarray):
                L_val = float(L_val.mean())
            I_val = sd.get('area', sd.get('I', None))
            if I_val is None: continue
            L_vals.append(float(L_val)); I_vals.append(float(I_val))

        if not L_vals:
            self.say(f"No data found. Check that scans have '{l_col}' and 'area' columns.\n"
                f"Available scans: {sorted(self._all_scan_data.keys())[:20]}", error=True); return

        L_arr = np.array(L_vals); I_arr = np.array(I_vals)
        sort_idx = np.argsort(L_arr)
        self._group_data[self._current_group] = {'L': L_arr[sort_idx], 'I': I_arr[sort_idx]}

        self._data_info.setText(
            f"Group '{self._current_group}': {len(L_vals)} points, "
            f"L=[{L_arr.min():.3f}, {L_arr.max():.3f}]")
        self._update_plot()

    def _save_groups(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save groups", "ctr_groups.json", "JSON (*.json)")
        if path:
            self._group_mgr.save(path)
            self.say(f"Saved groups to {path}")

    def _load_groups(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load groups", "", "JSON (*.json)")
        if path:
            self._group_mgr.load(path)
            self._refresh_group_list()
            self.say(f"Loaded groups from {path}")

    # ---- Model parameters ----
    def _on_model_changed(self):
        name = self._model_combo.currentText()
        model = CTR_MODELS.get(name)
        if model is None: return
        self._model_desc.setText(model['desc'])
        self._build_param_table(model)

    def _build_param_table(self, model):
        params = model['params']
        defaults = model['defaults']
        lo = model['bounds_lo']
        hi = model['bounds_hi']
        self._param_table.setRowCount(len(params))
        self._param_spins = []
        self._param_fixed = []
        for i, (p, d, l, h) in enumerate(zip(params, defaults, lo, hi)):
            # Name
            item = QTableWidgetItem(p)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._param_table.setItem(i, 0, item)
            # Value
            sb = QDoubleSpinBox()
            sb.setRange(-1e12, 1e12); sb.setDecimals(5)
            sb.setSingleStep(max(0.001, d * 0.1)); sb.setValue(d)
            sb.editingFinished.connect(self._update_plot)
            self._param_table.setCellWidget(i, 1, sb)
            self._param_spins.append(sb)
            # Fix checkbox
            chk = QCheckBox()
            self._param_table.setCellWidget(i, 2, chk)
            self._param_fixed.append(chk)
            # Bounds display
            item_b = QTableWidgetItem(f"[{l}, {h}]")
            item_b.setFlags(item_b.flags() & ~Qt.ItemIsEditable)
            self._param_table.setItem(i, 3, item_b)

    def _get_param_values(self):
        return [sb.value() for sb in self._param_spins]

    def _set_param_values(self, values):
        for i, v in enumerate(values):
            if i < len(self._param_spins):
                self._param_spins[i].blockSignals(True)
                self._param_spins[i].setValue(v)
                self._param_spins[i].blockSignals(False)

    def _get_fixed_mask(self):
        return [chk.isChecked() for chk in self._param_fixed]

    # ---- Fitting ----
    def _run_fit(self):
        L, I_obs = self._get_group_data()
        if L is None or I_obs is None:
            self.say("Build CTR data from a group first", error=True); return

        model_name = self._model_combo.currentText()
        model = CTR_MODELS[model_name]
        func = model['func']
        p0 = self._get_param_values()
        fixed = self._get_fixed_mask()
        lo = list(model['bounds_lo'])
        hi = list(model['bounds_hi'])

        free_idx = [i for i, f in enumerate(fixed) if not f]
        if not free_idx:
            self.say("All parameters are fixed", error=True); return

        p_full = list(p0)

        def residuals(p_free):
            for i, idx in enumerate(free_idx):
                p_full[idx] = p_free[i]
            I_calc = func(L, p_full)
            I_calc_safe = np.maximum(I_calc, 1e-20)
            I_obs_safe = np.maximum(I_obs, 1e-20)
            return np.log10(I_calc_safe) - np.log10(I_obs_safe)

        p_free0 = [p0[i] for i in free_idx]
        lo_free = [lo[i] for i in free_idx]
        hi_free = [hi[i] for i in free_idx]

        try:
            result = least_squares(residuals, p_free0, bounds=(lo_free, hi_free),
                                    max_nfev=5000, method='trf')
            for i, idx in enumerate(free_idx):
                p_full[idx] = result.x[i]

            chi2 = np.sum(result.fun**2) / max(1, len(L) - len(free_idx))
            self._set_param_values(p_full)

            if self._current_group:
                self._fit_results[self._current_group] = {
                    'params': list(p_full), 'model': model_name, 'chi2': chi2}

            param_str = ", ".join(f"{n}={v:.4g}" for n, v in zip(model['params'], p_full))
            self._fit_info.setText(f"χ²_red = {chi2:.4g}\n{param_str}")
            self._update_plot()

        except Exception as e:
            self._fit_info.setText(f"Fit failed: {e}")

    def _simulate(self):
        self._update_plot()

    # ---- Plotting ----
    PLOT_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
                   '#9467bd', '#8c564b', '#17becf', '#e377c2']

    def _update_plot(self):
        self._ax.clear()
        self._ax.set_facecolor('white')
        self._ax.tick_params(colors='#222')
        for sp in self._ax.spines.values(): sp.set_color('#999')

        has_data = False
        all_L = []

        # Plot ALL groups that have data
        for i, (gname, gd) in enumerate(self._group_data.items()):
            L, I = gd['L'], gd['I']
            color = self.PLOT_COLORS[i % len(self.PLOT_COLORS)]
            is_current = (gname == self._current_group)
            alpha = 1.0 if is_current else 0.5
            ms = 5 if is_current else 3
            self._ax.semilogy(L, I, 'o', color=color, markersize=ms, alpha=alpha,
                             label=gname)
            all_L.extend(L)
            has_data = True

            # Overlay fit curve if available
            if gname in self._fit_results:
                fr = self._fit_results[gname]
                model = CTR_MODELS.get(fr['model'])
                if model:
                    L_fine = np.linspace(L.min() - 0.3, L.max() + 0.3, 500) + 1e-6
                    try:
                        I_calc = model['func'](L_fine, fr['params'])
                        I_calc = np.maximum(I_calc, 1e-20)
                        self._ax.semilogy(L_fine, I_calc, '-', color=color,
                                         linewidth=1.5, alpha=alpha)
                    except Exception:
                        pass

        # Also show current model curve for the selected group (live from spinboxes)
        if self._current_group and self._current_group in self._group_data:
            L_cur, I_cur = self._group_data[self._current_group]['L'], self._group_data[self._current_group]['I']
            model_name = self._model_combo.currentText()
            model = CTR_MODELS.get(model_name)
            if model:
                L_fine = np.linspace(L_cur.min() - 0.3, L_cur.max() + 0.3, 500) + 1e-6
                p = self._get_param_values()
                try:
                    I_calc = model['func'](L_fine, p)
                    I_calc = np.maximum(I_calc, 1e-20)
                    # Find the color index for current group
                    names = list(self._group_data.keys())
                    ci = names.index(self._current_group) if self._current_group in names else 0
                    color = self.PLOT_COLORS[ci % len(self.PLOT_COLORS)]
                    self._ax.semilogy(L_fine, I_calc, '--', color=color,
                                     linewidth=1.0, alpha=0.7, label=f'{self._current_group} (model)')
                except Exception:
                    pass

        title = "CTR Fitting"
        if self._current_group:
            title += f" — {self._current_group}"
        self._ax.set_title(title, color='#222', fontsize=12)
        self._ax.set_xlabel("L (r.l.u.)", color='#222', fontsize=11)
        self._ax.set_ylabel("Intensity", color='#222', fontsize=11)
        self._ax.grid(True, alpha=0.2, color='#666', linestyle=':')
        if has_data:
            self._ax.legend(facecolor='white', edgecolor='#999', labelcolor='#222', fontsize=8)
        self._canvas.draw()

    # ---- Export ----
    def _export_group_data(self):
        """Export all groups as tab-separated text."""
        if not self._group_data:
            self.say("No group data", error=True); return
        path, _ = QFileDialog.getSaveFileName(self, "Export group data", "ctr_data.txt", "Text (*.txt)")
        if not path: return
        with open(path, 'w') as f:
            for gname, gd in self._group_data.items():
                f.write(f"# Group: {gname}\n")
                hk = self._group_mgr.groups.get(gname, {}).get('hk', '?')
                f.write(f"# HK: {hk}\n")
                f.write("L\tI\n")
                for l, i in zip(gd['L'], gd['I']):
                    f.write(f"{l:.6g}\t{i:.6g}\n")
                f.write("\n")
        self.say(f"Exported {len(self._group_data)} groups to {path}")

    def _export_fit_params(self):
        if not self._fit_results:
            self.say("No fit results", error=True); return
        path, _ = QFileDialog.getSaveFileName(self, "Export fit parameters", "ctr_fit_params.json", "JSON (*.json)")
        if not path: return
        # Convert numpy types for JSON
        export = {}
        for name, fr in self._fit_results.items():
            export[name] = {
                'model': fr['model'],
                'chi2': float(fr['chi2']),
                'params': {p: float(v) for p, v in
                           zip(CTR_MODELS[fr['model']]['params'], fr['params'])}
            }
        with open(path, 'w') as f:
            json.dump(export, f, indent=2)
        self.say(f"Exported to {path}")

    def _save_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", "ctr_fit.png",
                                               "PNG (*.png);;SVG (*.svg)")
        if path:
            self._fig.savefig(path, dpi=200, facecolor='white', bbox_inches='tight')
            self.say(f"Saved to {path}")


def main(argv=None):
    return run_tool(CTRFitWindow)


if __name__ == '__main__':
    main()

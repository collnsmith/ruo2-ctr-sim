"""Beamline helper window: Bragg peak indexer, UB refinement (reflex), angle calculator, SPEC macro maker, calculators.

Run:  python ctr_beamline.py   (or python -m ctrfit beamline)

The computation is in ctrfit.beamline (testable without Qt); this file only holds the window.
"""
import json
import math
import sys
import traceback
from pathlib import Path

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:  # pragma: no cover
    from PySide6 import QtCore, QtGui, QtWidgets

import numpy as np

from ..beamline import macros as mac
from ..beamline import xtools
from ..beamline.indexing import ALLOWED, index_peaks
from ..beamline.psic import (DEFAULT_LIMITS, LATTICES, MODES, MOTORS, Lattice, Psic, angles_for_hkl, as_angles,
                             energy_to_wavelength, mode_with_alpha, parse_angle_lines, ub_from_two_reflections)
from ..beamline.refine import REFLEX_HELP, ReflectionList, parse_reflection_lines, refine_ub
from ..core.settings import DEFAULTS, parse_points, parse_rods

PEAK_HELP = ("One Bragg peak per line, angles in SPEC psic order:  del eta chi phi nu mu  [H K L]\n"
             "or as name=value pairs (del= eta= chi= phi= nu= mu=). The optional [H K L] tag fixes what that peak\n"
             "is called (tag two peaks to stop the indexing changing). Text after # is a label.")


def _mono():
    return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)


def _dspin(lo, hi, val, dec=3, step=0.1, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


class BeamlineWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ctrfit beamline helper (SPEC psic)")
        self.geo = Psic()
        self.UB = None
        self.index_result = None
        self.reflex = ReflectionList()
        self.refine_result = None

        # ---------------- shared: energy, lattice, motor signs
        self.energy = _dspin(3.0, 40.0, DEFAULTS["energy_kev"], 4, 0.1, " keV")
        self.lam_label = QtWidgets.QLabel()
        self.lattice = QtWidgets.QComboBox()
        for key, fn in LATTICES.items():
            self.lattice.addItem(fn().name, key)
        self.lattice.addItem("custom (a b c alpha beta gamma)", "custom")
        self.custom = QtWidgets.QLineEdit("2.9587 6.4965 6.4965 90 90 90")
        self.rule = QtWidgets.QComboBox()
        self.rule.addItems(["tio2_surface", "all"])
        self.rule.setToolTip("Which reflections are Bragg peaks: tio2_surface uses the rutile structure factor "
                             "(K + L even and the rutile extinctions), all allows every integer HKL")
        self.signs = QtWidgets.QLineEdit("")
        self.signs.setPlaceholderText("e.g. eta=-1 nu=-1 (if a motor turns the other way)")
        top = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("Energy"), self.energy, self.lam_label, QtWidgets.QLabel("Lattice"), self.lattice,
                  self.custom, QtWidgets.QLabel("Bragg rule"), self.rule, QtWidgets.QLabel("Motor signs"),
                  self.signs):
            top.addWidget(w)
        top.addStretch(1)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._indexer_tab(), "Bragg peak indexer")
        self.tabs.addTab(self._reflex_tab(), "UB refinement (reflex)")
        self.tabs.addTab(self._angles_tab(), "Angles (HKL <-> motors)")
        self.tabs.addTab(self._macro_tab(), "Macro maker")
        self.tabs.addTab(self._calc_tab(), "Calculators")
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.addLayout(top)
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.status = QtWidgets.QLabel("Ready")
        self.statusBar().addWidget(self.status, 1)
        self.energy.valueChanged.connect(self._energy_changed)
        self.lattice.currentIndexChanged.connect(lambda *_: self.custom.setEnabled(self.lattice.currentData() == "custom"))
        self._energy_changed()
        self.custom.setEnabled(False)
        self.resize(1300, 820)

    # ------------------------------------------------------------------ shared helpers
    @property
    def wavelength(self):
        return energy_to_wavelength(self.energy.value())

    def _energy_changed(self, *_):
        self.lam_label.setText(f"λ = {self.wavelength:.5f} Å")
        if hasattr(self, "calc_out"):
            self.update_calc()

    def current_lattice(self):
        key = self.lattice.currentData()
        if key != "custom":
            return LATTICES[key]()
        vals = [float(x) for x in self.custom.text().replace(",", " ").split()]
        if len(vals) != 6:
            raise ValueError("custom lattice: give a b c alpha beta gamma")
        return Lattice(*vals)

    def current_geo(self):
        signs = {}
        for item in self.signs.text().replace(",", " ").split():
            name, _, v = item.partition("=")
            if name not in MOTORS or v.strip() not in ("1", "-1", "+1"):
                raise ValueError(f"motor signs: '{item}' should look like eta=-1")
            signs[name] = int(v)
        self.geo = Psic(signs)
        return self.geo

    def say(self, text, error=False):
        self.status.setText(text.splitlines()[0][:200])
        if error:
            QtWidgets.QMessageBox.warning(self, "Beamline helper", text)

    def _run(self, fn):
        try:
            return fn()
        except (ValueError, KeyError, np.linalg.LinAlgError) as ex:
            self.say(str(ex), error=True)
        except Exception:  # keep the window alive
            self.say(traceback.format_exc(), error=True)
        return None

    @staticmethod
    def _vec(text, what, n=3):
        vals = [float(x) for x in text.replace(",", " ").split()]
        if len(vals) != n:
            raise ValueError(f"{what}: give {n} numbers")
        return np.array(vals)

    # ------------------------------------------------------------------ indexer
    def _indexer_tab(self):
        w = QtWidgets.QWidget()
        self.peaks_in = QtWidgets.QPlainTextEdit()
        self.peaks_in.setFont(_mono())
        self.peaks_in.setPlaceholderText(PEAK_HELP)
        self.normal_in = QtWidgets.QLineEdit("")
        self.normal_in.setPlaceholderText("optional, e.g. 0 0 1 (normal along the phi axis)")
        self.normal_in.setToolTip("Surface normal in the phi frame. Bragg peaks alone cannot tell K from L in the "
                                  "TiO2(110) surface cell; the normal picks the indexing with L along it.")
        self.tol_q = _dspin(0.001, 0.1, 0.01, 3, 0.002)
        self.tol_ang = _dspin(0.05, 10.0, 1.0, 2, 0.1, " °")
        b_index = QtWidgets.QPushButton("Guess HKL")
        b_index.setMinimumHeight(30)
        b_use = QtWidgets.QPushButton("Use this UB for the angle calculator")
        b_save = QtWidgets.QPushButton("Save UB…")
        self.index_out = QtWidgets.QPlainTextEdit(readOnly=True)
        self.index_out.setFont(_mono())
        form = QtWidgets.QFormLayout()
        form.addRow("Surface normal (phi frame)", self.normal_in)
        form.addRow("|Q| tolerance (relative)", self.tol_q)
        form.addRow("Angle tolerance", self.tol_ang)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("<b>Bragg peaks found</b> (angles as read from SPEC)"))
        left.addWidget(self.peaks_in, 1)
        left.addLayout(form)
        left.addWidget(b_index)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("<b>Best guess</b>"))
        right.addWidget(self.index_out, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(b_use)
        row.addWidget(b_save)
        right.addLayout(row)
        lay = QtWidgets.QHBoxLayout(w)
        lay.addLayout(left, 2)
        lay.addLayout(right, 3)
        b_index.clicked.connect(self.run_index)
        b_use.clicked.connect(self.use_index_ub)
        b_save.clicked.connect(self.save_ub)
        return w

    def run_index(self):
        def go():
            peaks = parse_angle_lines(self.peaks_in.toPlainText())
            if not peaks:
                raise ValueError("enter at least one peak")
            normal = self._vec(self.normal_in.text(), "surface normal") if self.normal_in.text().strip() else None
            res = index_peaks(peaks, self.wavelength, self.current_lattice(), ALLOWED[self.rule.currentText()],
                              tol_q=self.tol_q.value(), tol_angle=self.tol_ang.value(), normal_phi=normal,
                              geo=self.current_geo())
            labels = [p.get("label", "") for p in peaks]
            text = res.summary
            if any(labels) and res.UB is not None:
                text += "\n\nLabels: " + "; ".join(f"{i + 1}: {lab}" for i, lab in enumerate(labels) if lab)
            self.index_result = res
            self.index_out.setPlainText(text)
            self.say(f"Indexed {res.n_indexed} of {len(peaks)} peaks" if res.UB is not None else
                     "One peak: candidates by |Q|")
        self._run(go)

    def use_index_ub(self):
        if self.index_result is None or self.index_result.UB is None:
            self.say("Index at least two peaks first.", error=True)
            return
        self.set_ub(self.index_result.UB, "from the indexer")
        self.tabs.setCurrentWidget(self.angles_page)

    def set_ub(self, UB, source):
        self.UB = np.array(UB, float)
        self.ub_text.blockSignals(True)            # keep the full-precision UB, not the 6 decimals shown
        self.ub_text.setPlainText("\n".join(" ".join(f"{x:.6f}" for x in row) for row in self.UB))
        self.ub_text.blockSignals(False)
        self.ub_source.setText(f"UB {source}")
        n = self.UB @ np.array([0.0, 0.0, 1.0])
        self.normal_ang.setText(" ".join(f"{x:.4f}" for x in n / np.linalg.norm(n)))

    def save_ub(self):
        if self.index_result is None or self.index_result.UB is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save UB", "ub.json", "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(dict(UB=self.index_result.UB.tolist(), energy_kev=self.energy.value(),
                                                  lattice=self.current_lattice().to_dict()), indent=1))
            self.say(f"Saved {Path(path).name}")

    # ------------------------------------------------------------------ UB refinement (reflex)
    COLS = ["use", "H", "K", "L", *MOTORS, "E (keV)", "label", "dangle °", "d|Q|/|Q|"]

    def _reflex_tab(self):
        w = QtWidgets.QWidget()
        self.rx_table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.rx_table.setHorizontalHeaderLabels(self.COLS)
        self.rx_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.rx_table.verticalHeader().setDefaultSectionSize(22)
        self.rx_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.rx_table.horizontalHeader().setStretchLastSection(True)
        self.rx_hkl = QtWidgets.QLineEdit()
        self.rx_hkl.setPlaceholderText("H K L (blank: guess from the current UB)")
        self.rx_ang = QtWidgets.QLineEdit()
        self.rx_ang.setPlaceholderText("del eta chi phi nu mu (as read from SPEC wh)")
        self.rx_label = QtWidgets.QLineEdit()
        self.rx_label.setPlaceholderText("label (optional)")
        b_add = QtWidgets.QPushButton("Add reflection")
        b_add.setMinimumHeight(28)
        self.rx_bulk = QtWidgets.QPlainTextEdit()
        self.rx_bulk.setFont(_mono())
        self.rx_bulk.setPlaceholderText(REFLEX_HELP + "\nEnergy defaults to the energy at the top.")
        self.rx_bulk.setMaximumHeight(90)
        b_bulk = QtWidgets.QPushButton("Add these lines")
        self.rx_free = QtWidgets.QComboBox()
        for key, text in (("orientation", "orientation only (U)"), ("scale", "U + common lattice scale"),
                          ("abc", "U + a, b, c"), ("all", "U + all six lattice parameters"),
                          ("ub", "free UB (9 elements)")):
            self.rx_free.addItem(text, key)
        self.rx_offsets = QtWidgets.QLineEdit("")
        self.rx_offsets.setPlaceholderText("motor zero offsets to refine, e.g. del eta")
        self.rx_auto = QtWidgets.QCheckBox("Refine after each change")
        self.rx_auto.setChecked(True)
        b_refine = QtWidgets.QPushButton("Refine UB")
        b_refine.setMinimumHeight(30)
        b_use = QtWidgets.QPushButton("Use refined UB for the angle calculator")
        b_import = QtWidgets.QPushButton("Add the indexed peaks")
        b_remove = QtWidgets.QPushButton("Remove selected")
        b_clear = QtWidgets.QPushButton("Clear")
        b_load = QtWidgets.QPushButton("Load…")
        b_save = QtWidgets.QPushButton("Save…")
        b_lat = QtWidgets.QPushButton("Set the custom lattice to the refined one")
        self.rx_out = QtWidgets.QPlainTextEdit(readOnly=True)
        self.rx_out.setFont(_mono())

        add = QtWidgets.QHBoxLayout()
        for x, st in ((self.rx_hkl, 2), (self.rx_ang, 4), (self.rx_label, 2), (b_add, 0)):
            add.addWidget(x, st)
        bulk = QtWidgets.QHBoxLayout()
        bulk.addWidget(self.rx_bulk, 1)
        bulk.addWidget(b_bulk)
        opts = QtWidgets.QHBoxLayout()
        for x in (QtWidgets.QLabel("Refine"), self.rx_free, QtWidgets.QLabel("Offsets"), self.rx_offsets,
                  self.rx_auto, b_refine):
            opts.addWidget(x)
        rows = QtWidgets.QHBoxLayout()
        for x in (b_import, b_remove, b_clear, b_load, b_save):
            rows.addWidget(x)
        rows.addStretch(1)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("<b>Reflections</b> (edit cells in place; untick 'use' to leave one out)"))
        left.addWidget(self.rx_table, 1)
        left.addLayout(add)
        left.addLayout(bulk)
        left.addLayout(rows)
        left.addLayout(opts)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("<b>Refined orientation</b>"))
        right.addWidget(self.rx_out, 1)
        right.addWidget(b_use)
        right.addWidget(b_lat)
        lay = QtWidgets.QHBoxLayout(w)
        lay.addLayout(left, 3)
        lay.addLayout(right, 2)
        b_add.clicked.connect(self.add_reflection)
        self.rx_ang.returnPressed.connect(self.add_reflection)
        b_bulk.clicked.connect(self.add_reflection_lines)
        b_refine.clicked.connect(self.refine)
        b_use.clicked.connect(self.use_refined_ub)
        b_lat.clicked.connect(self.use_refined_lattice)
        b_import.clicked.connect(self.import_indexed)
        b_remove.clicked.connect(self.remove_reflections)
        b_clear.clicked.connect(self.clear_reflections)
        b_load.clicked.connect(lambda: self.load_reflections())
        b_save.clicked.connect(lambda: self.save_reflections())
        self.rx_table.itemChanged.connect(self._reflex_edited)
        self.rx_free.currentIndexChanged.connect(lambda *_: self._auto_refine())
        return w

    def _guess_ub(self):
        if self.refine_result is not None:
            return self.refine_result.UB
        return self.UB

    def _fill_reflex_table(self):
        t = self.rx_table
        t.blockSignals(True)
        t.setRowCount(len(self.reflex))
        fits = {f.index: f for f in self.refine_result.fits} if self.refine_result is not None else {}
        for i, r in enumerate(self.reflex):
            use = QtWidgets.QTableWidgetItem()
            use.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            use.setCheckState(QtCore.Qt.Checked if r.use else QtCore.Qt.Unchecked)
            t.setItem(i, 0, use)
            vals = [f"{x:g}" for x in r.hkl] + [f"{r.angles[m]:.4f}" for m in MOTORS] + [f"{r.energy_kev:.4f}",
                                                                                         r.label]
            f = fits.get(i)
            vals += [f"{f.dangle:.4f}", f"{f.dq_rel:.1e}"] if f else ["", ""]
            for j, v in enumerate(vals, start=1):
                it = QtWidgets.QTableWidgetItem(v)
                if j >= len(self.COLS) - 2:
                    it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                    if f and f.outlier:
                        it.setForeground(QtGui.QColor("#c0392b"))
                t.setItem(i, j, it)
        t.blockSignals(False)

    def _reflex_edited(self, item):
        i, j = item.row(), item.column()
        r = self.reflex[i]
        try:
            if j == 0:
                r.use = item.checkState() == QtCore.Qt.Checked
            elif j <= 3:
                h = list(r.hkl)
                h[j - 1] = float(item.text())
                r.hkl = tuple(h)
            elif j <= 9:
                r.angles[MOTORS[j - 4]] = float(item.text())
            elif j == 10:
                r.energy_kev = float(item.text())
            elif j == 11:
                r.label = item.text()
                return
        except ValueError:
            self.say(f"row {i + 1}: '{item.text()}' is not a number", error=True)
            self._fill_reflex_table()
            return
        self._auto_refine()

    def _auto_refine(self):
        if self.rx_auto.isChecked() and len(self.reflex.used) >= 2:
            self.refine(quiet=True)
        else:
            self.refine_result = None
            self._fill_reflex_table()

    def add_reflection(self):
        def go():
            hkl = self._vec(self.rx_hkl.text(), "HKL") if self.rx_hkl.text().strip() else None
            ang = self._vec(self.rx_ang.text(), "angles", 6)
            r = self.reflex.add(hkl, ang, self.energy.value(), self.rx_label.text().strip(), UB=self._guess_ub(),
                                geo=self.current_geo())
            self.rx_hkl.clear()
            self.rx_ang.clear()
            self.rx_label.clear()
            self.say(f"Added ({' '.join(f'{x:g}' for x in r.hkl)})" + (" (HKL guessed from the UB)" if hkl is None
                                                                        else ""))
            self._auto_refine()
        self._run(go)

    def add_reflection_lines(self):
        def go():
            rows = parse_reflection_lines(self.rx_bulk.toPlainText(), self.energy.value())
            if not rows:
                raise ValueError("no reflections in the text")
            for row in rows:              # each guessed HKL uses the UB refined from the rows before it
                self.reflex.add(row["hkl"], row["angles"], row["energy_kev"], row["label"], UB=self._guess_ub(),
                                geo=self.current_geo())
                if len(self.reflex.used) >= 2:
                    self.refine(quiet=True)
            self.rx_bulk.clear()
            self._auto_refine()
            self.say(f"Added {len(rows)} reflection(s)")
        self._run(go)

    def import_indexed(self):
        if self.index_result is None or self.index_result.UB is None:
            self.say("Index Bragg peaks first (indexer tab).", error=True)
            return
        n = 0
        for p in self.index_result.peaks:
            if p.indexed:
                self.reflex.add(p.hkl, p.angles, self.energy.value(), p.angles.get("label", "") or "indexer")
                n += 1
        self.say(f"Added {n} indexed peak(s)")
        self._auto_refine()

    def remove_reflections(self):
        rows = sorted({ix.row() for ix in self.rx_table.selectedIndexes()}, reverse=True)
        for i in rows:
            self.reflex.remove(i)
        self.refine_result = None
        self._auto_refine()

    def clear_reflections(self):
        self.reflex.clear()
        self.refine_result = None
        self.rx_out.clear()
        self._fill_reflex_table()

    def refine(self, quiet=False):
        def go():
            offsets = self.rx_offsets.text().replace(",", " ").split()
            res = refine_ub(self.reflex, self.current_lattice(), self.rx_free.currentData(), offsets,
                            geo=self.current_geo())
            self.refine_result = res
            self.rx_out.setPlainText(res.summary)
            self._fill_reflex_table()
            self.say(f"UB refined on {res.n_used} reflections: rms angle {res.rms_angle:.4f}°"
                     + (f", {len(res.warnings)} warning(s)" if res.warnings else ""))
        if quiet:
            try:
                go()
            except (ValueError, np.linalg.LinAlgError) as ex:   # e.g. not enough reflections for the parameters
                self.refine_result = None
                self.rx_out.setPlainText(f"Not refined: {ex}")
                self._fill_reflex_table()
        else:
            self._run(go)

    def use_refined_ub(self):
        if self.refine_result is None:
            self.say("Refine the UB first.", error=True)
            return
        self.set_ub(self.refine_result.UB, f"refined on {self.refine_result.n_used} reflections")
        self.tabs.setCurrentWidget(self.angles_page)

    def use_refined_lattice(self):
        if self.refine_result is None:
            self.say("Refine the UB first.", error=True)
            return
        lat = self.refine_result.lattice
        self.lattice.setCurrentIndex(self.lattice.findData("custom"))
        self.custom.setText(" ".join(f"{getattr(lat, n):.5f}" for n in ("a", "b", "c", "alpha", "beta", "gamma")))
        self.say("Custom lattice set to the refined one")

    def save_reflections(self, path=None):
        path = path or QtWidgets.QFileDialog.getSaveFileName(self, "Save reflections", "reflex.json",
                                                             "JSON (*.json);;Text (*.txt)")[0]
        if not path:
            return
        if str(path).endswith(".txt"):
            Path(path).write_text(self.reflex.text() + "\n", encoding="utf-8")
        else:
            d = self.reflex.to_dict()
            d.update(energy_kev=self.energy.value(), lattice=self.current_lattice().to_dict())
            if self.refine_result is not None:
                d["refined"] = self.refine_result.to_dict()
            Path(path).write_text(json.dumps(d, indent=1), encoding="utf-8")
        self.say(f"Saved {Path(path).name}")

    def load_reflections(self, path=None):
        path = path or QtWidgets.QFileDialog.getOpenFileName(self, "Load reflections", "",
                                                             "Reflections (*.json *.txt);;All files (*)")[0]
        if not path:
            return

        def go():
            self.reflex = ReflectionList.load(path)
            self.refine_result = None
            self.say(f"Loaded {len(self.reflex)} reflections from {Path(path).name}")
            self._auto_refine()
        self._run(go)

    # ------------------------------------------------------------------ angles
    def _angles_tab(self):
        w = self.angles_page = QtWidgets.QWidget()
        self.ub_text = QtWidgets.QPlainTextEdit()
        self.ub_text.setFont(_mono())
        self.ub_text.setMaximumHeight(80)
        self.ub_text.setPlaceholderText("UB matrix (3 rows of 3), from the indexer or two reflections")
        self.ub_source = QtWidgets.QLabel("no UB yet")
        self.or0_hkl, self.or1_hkl = QtWidgets.QLineEdit("0 0 2"), QtWidgets.QLineEdit("1 1 1")
        self.or0_ang, self.or1_ang = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for e in (self.or0_ang, self.or1_ang):
            e.setPlaceholderText("del eta chi phi nu mu")
        b_or = QtWidgets.QPushButton("UB from these two reflections")
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(list(MODES))
        self.alpha = _dspin(0.0, 30.0, 0.5, 3, 0.05, " °")
        self.normal_ang = QtWidgets.QLineEdit("0 0 1")
        self.normal_ang.setToolTip("Surface normal in the phi frame (needed for alpha / beta modes); "
                                   "set from UB (0 0 1) when a UB is loaded")
        self.fixed_ang = QtWidgets.QLineEdit("")
        self.fixed_ang.setPlaceholderText("other fixed motors, e.g. phi=0 (defaults of the mode otherwise)")
        self.hkl_in = QtWidgets.QLineEdit("0 1 1.5")
        b_calc = QtWidgets.QPushButton("Angles for HKL")
        self.where_in = QtWidgets.QLineEdit()
        self.where_in.setPlaceholderText("del eta chi phi nu mu  ->  HKL, alpha, beta")
        b_where = QtWidgets.QPushButton("HKL from angles")
        self.ang_out = QtWidgets.QPlainTextEdit(readOnly=True)
        self.ang_out.setFont(_mono())

        g = QtWidgets.QGridLayout()
        g.addWidget(QtWidgets.QLabel("<b>Orientation</b>"), 0, 0, 1, 4)
        g.addWidget(self.ub_text, 1, 0, 1, 3)
        g.addWidget(self.ub_source, 1, 3)
        g.addWidget(QtWidgets.QLabel("or0 HKL"), 2, 0)
        g.addWidget(self.or0_hkl, 2, 1)
        g.addWidget(QtWidgets.QLabel("angles"), 2, 2)
        g.addWidget(self.or0_ang, 2, 3)
        g.addWidget(QtWidgets.QLabel("or1 HKL"), 3, 0)
        g.addWidget(self.or1_hkl, 3, 1)
        g.addWidget(QtWidgets.QLabel("angles"), 3, 2)
        g.addWidget(self.or1_ang, 3, 3)
        g.addWidget(b_or, 4, 3)
        g.addWidget(QtWidgets.QLabel("<b>Mode</b>"), 5, 0, 1, 4)
        g.addWidget(QtWidgets.QLabel("Mode"), 6, 0)
        g.addWidget(self.mode, 6, 1)
        g.addWidget(QtWidgets.QLabel("Incidence angle alpha"), 6, 2)
        g.addWidget(self.alpha, 6, 3)
        g.addWidget(QtWidgets.QLabel("Surface normal (phi frame)"), 7, 0)
        g.addWidget(self.normal_ang, 7, 1)
        g.addWidget(QtWidgets.QLabel("Fixed motors"), 7, 2)
        g.addWidget(self.fixed_ang, 7, 3)
        g.addWidget(QtWidgets.QLabel("HKL"), 8, 0)
        g.addWidget(self.hkl_in, 8, 1)
        g.addWidget(b_calc, 8, 2)
        g.addWidget(QtWidgets.QLabel("Angles"), 9, 0)
        g.addWidget(self.where_in, 9, 1, 1, 2)
        g.addWidget(b_where, 9, 3)
        lay = QtWidgets.QVBoxLayout(w)
        lay.addLayout(g)
        lay.addWidget(self.ang_out, 1)
        b_or.clicked.connect(self.ub_from_or)
        b_calc.clicked.connect(self.calc_angles)
        b_where.clicked.connect(self.calc_where)
        self.ub_text.textChanged.connect(self._ub_from_text)
        return w

    def _ub_from_text(self):
        try:
            vals = [float(x) for x in self.ub_text.toPlainText().replace(",", " ").split()]
            if len(vals) == 9:
                self.UB = np.array(vals).reshape(3, 3)
        except ValueError:
            pass

    def ub_from_or(self):
        def go():
            UB = ub_from_two_reflections(self.current_lattice(), self._vec(self.or0_hkl.text(), "or0 HKL"),
                                         self._vec(self.or0_ang.text(), "or0 angles", 6),
                                         self._vec(self.or1_hkl.text(), "or1 HKL"),
                                         self._vec(self.or1_ang.text(), "or1 angles", 6), self.wavelength,
                                         self.current_geo())
            self.set_ub(UB, "from or0 / or1")
            self.say("UB from two reflections")
        self._run(go)

    def _need_ub(self):
        if self.UB is None:
            raise ValueError("no UB: index Bragg peaks or enter two reflections first")
        return self.UB

    def calc_angles(self):
        def go():
            UB = self._need_ub()
            hkl = self._vec(self.hkl_in.text(), "HKL")
            mode, fixed = mode_with_alpha(self.mode.currentText(), self.alpha.value())
            for item in self.fixed_ang.text().replace(",", " ").split():
                name, _, v = item.partition("=")
                if name not in MOTORS:
                    raise ValueError(f"fixed motors: unknown motor {name}")
                fixed[name] = float(v)
            normal = self._vec(self.normal_ang.text(), "surface normal")
            sols = angles_for_hkl(hkl, UB, self.wavelength, mode, fixed=fixed, normal_phi=normal,
                                  geo=self.current_geo(), limits=DEFAULT_LIMITS)
            lines = [f"({' '.join(f'{x:g}' for x in hkl)}) in mode '{self.mode.currentText()}', "
                     f"|Q| = {np.linalg.norm(UB @ hkl):.4f} 1/Å"]
            if not sols:
                lines.append("no solution within the motor limits (del, nu between -5 and 120°)")
            hdr = " ".join(f"{m:>9}" for m in MOTORS) + f" {'2theta':>9} {'alpha':>8} {'beta':>8}"
            lines.append(hdr)
            for s in sols[:6]:
                lines.append(" ".join(f"{s[m]:>9.4f}" for m in MOTORS) + f" {s['two_theta']:>9.4f} "
                             f"{s.get('alpha', float('nan')):>8.3f} {s.get('beta', float('nan')):>8.3f}")
            if sols:
                best = sols[0]
                lines.append("\nSPEC:  umv " + " ".join(f"{m} {best[m]:.4f}" for m in MOTORS))
            self.ang_out.setPlainText("\n".join(lines))
            self.say(f"{len(sols)} solution(s)")
        self._run(go)

    def calc_where(self):
        def go():
            UB = self._need_ub()
            a = as_angles(self._vec(self.where_in.text(), "angles", 6))
            g = self.current_geo()
            h = g.hkl(a, UB, self.wavelength)
            normal = self._vec(self.normal_ang.text(), "surface normal")
            al, be = g.surface_angles(a, normal, self.wavelength)
            self.ang_out.setPlainText(f"H K L = {h[0]:.4f} {h[1]:.4f} {h[2]:.4f}\n2theta = {g.two_theta(a):.4f}°, "
                                      f"alpha = {al:.4f}°, beta = {be:.4f}°")
        self._run(go)

    # ------------------------------------------------------------------ macro maker
    def _macro_tab(self):
        w = QtWidgets.QWidget()
        self.mac_kind = QtWidgets.QComboBox()
        self.mac_kind.addItems(["rod scans", "fixed points", "potential series (fixed points)",
                                "potential series (rods)", "energy scan (fixed HKL)"])
        self.mac_rods = QtWidgets.QLineEdit("0 1; 1 0; 0 0; 1 1")
        self.mac_lmin, self.mac_lmax = _dspin(0.0, 20.0, 0.3, 3, 0.1), _dspin(0.1, 20.0, 4.0, 3, 0.5)
        self.mac_step = _dspin(0.001, 1.0, 0.02, 4, 0.005)
        self.mac_fine = _dspin(0.0, 0.5, 0.0, 4, 0.002)
        self.mac_fine.setToolTip("Step within 0.3 L of Bragg peaks (0: same as the step)")
        self.mac_excl = _dspin(0.0, 0.5, 0.05, 3, 0.01)
        self.mac_t = _dspin(0.1, 600.0, 1.0, 2, 0.5, " s")
        self.mac_scale = QtWidgets.QCheckBox("Count times from the model (t ∝ 1/I, 1 to 60 s)")
        self.mac_points = QtWidgets.QLineEdit(DEFAULTS["study_points"])
        self.mac_pot = QtWidgets.QLineEdit("0.4, 0.8, 1.0, 1.2")
        self.mac_wait = _dspin(0.0, 3600.0, 60.0, 0, 10.0, " s")
        self.mac_energies = QtWidgets.QLineEdit("21.9, 22.0, 22.1, 22.12, 22.14, 22.2, 22.3")
        self.mac_hkl = QtWidgets.QLineEdit("0 1 1.5")
        self.cmd_move = QtWidgets.QLineEdit(mac.MacroSettings.move)
        self.cmd_count = QtWidgets.QLineEdit(mac.MacroSettings.count)
        self.cmd_rock = QtWidgets.QLineEdit("")
        self.cmd_rock.setPlaceholderText("optional rocking scan, e.g. dscan phi -{hw:g} {hw:g} {n} {t:g}")
        self.cmd_pot = QtWidgets.QLineEdit(mac.MacroSettings.set_potential)
        self.cmd_energy = QtWidgets.QLineEdit(mac.MacroSettings.set_energy)
        self.cmd_before = QtWidgets.QLineEdit("")
        self.cmd_before.setPlaceholderText("before each rod, e.g. fon")
        self.mac_def = QtWidgets.QLineEdit("")
        self.mac_def.setPlaceholderText("optional: wrap in def <name>")
        b_make = QtWidgets.QPushButton("Make macro")
        b_save = QtWidgets.QPushButton("Save .mac…")
        self.mac_out = QtWidgets.QPlainTextEdit()
        self.mac_out.setFont(_mono())
        form = QtWidgets.QFormLayout()
        for lab, wid in (("Macro", self.mac_kind), ("Rods (H K)", self.mac_rods), ("L min", self.mac_lmin),
                         ("L max", self.mac_lmax), ("L step", self.mac_step), ("Fine step near Bragg", self.mac_fine),
                         ("Skip near Bragg (±L)", self.mac_excl), ("Count time", self.mac_t), ("", self.mac_scale),
                         ("Points (label: H K L; …)", self.mac_points), ("Potentials (V)", self.mac_pot),
                         ("Wait after setting", self.mac_wait), ("Energies (keV)", self.mac_energies),
                         ("HKL (energy scan)", self.mac_hkl), ("Move command", self.cmd_move),
                         ("Count command", self.cmd_count), ("Rocking scan", self.cmd_rock),
                         ("Set potential", self.cmd_pot), ("Set energy", self.cmd_energy),
                         ("Before each rod", self.cmd_before), ("Macro name", self.mac_def)):
            form.addRow(lab, wid)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.addLayout(form)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(b_make)
        row.addWidget(b_save)
        ll.addLayout(row)
        ll.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(460)
        lay = QtWidgets.QHBoxLayout(w)
        lay.addWidget(scroll)
        lay.addWidget(self.mac_out, 1)
        b_make.clicked.connect(self.make_macro)
        b_save.clicked.connect(self.save_macro)
        return w

    def macro_settings(self):
        return mac.MacroSettings(move=self.cmd_move.text(), count=self.cmd_count.text(), rock=self.cmd_rock.text(),
                                 set_potential=self.cmd_pot.text(), set_energy=self.cmd_energy.text(),
                                 before_rod=self.cmd_before.text())

    def make_macro(self):
        def floats(text, what):
            vals = [float(x) for x in text.replace(",", " ").replace(";", " ").split()]
            if not vals:
                raise ValueError(f"{what}: enter at least one number")
            return vals

        def go():
            s = self.macro_settings()
            kind = self.mac_kind.currentText()
            t = self.mac_t.value()

            def rods():
                from ..model.model import Model
                model = Model.from_settings(dict(DEFAULTS, energy_kev=self.energy.value()))
                return mac.rod_macro(parse_rods(self.mac_rods.text()), s, self.mac_lmin.value(), self.mac_lmax.value(),
                                     self.mac_step.value(), t, model, self.mac_excl.value(),
                                     self.mac_fine.value() or None, scale_times=self.mac_scale.isChecked())

            def points():
                return mac.points_macro(parse_points(self.mac_points.text()), s, t)
            if kind == "rod scans":
                m = rods()
            elif kind == "fixed points":
                m = points()
            elif kind.startswith("potential series"):
                inner = points() if "fixed" in kind else rods()
                m = mac.potential_macro(floats(self.mac_pot.text(), "potentials"), inner, s, self.mac_wait.value())
            else:
                m = mac.energy_macro(self._vec(self.mac_hkl.text(), "HKL"), floats(self.mac_energies.text(), "energies"),
                                     s, t)
            text = m.text(kind)
            if self.mac_def.text().strip():
                text = mac.wrap_def(self.mac_def.text().strip(), text)
            self.mac_out.setPlainText(text)
            self.say(f"{kind}: {m.n_points} points, about {mac.duration_text(m.seconds)}")
        self._run(go)

    def save_macro(self, path=None):
        if not self.mac_out.toPlainText().strip():
            self.make_macro()
        path = path or QtWidgets.QFileDialog.getSaveFileName(self, "Save macro", "ctr_scan.mac",
                                                             "SPEC macro (*.mac);;All files (*)")[0]
        if path:
            mac.write_macro(path, self.mac_out.toPlainText())
            self.say(f"Saved {Path(path).name}")

    # ------------------------------------------------------------------ calculators
    def _calc_tab(self):
        w = QtWidgets.QWidget()
        self.calc_hkl = QtWidgets.QLineEdit("0 0 2")
        self.calc_mat = QtWidgets.QComboBox()
        self.calc_mat.addItems(list(xtools.MATERIALS))
        self.calc_alpha = _dspin(0.0, 10.0, 0.5, 3, 0.05, " °")
        self.calc_beam = _dspin(0.001, 5.0, 0.02, 3, 0.005, " mm")
        self.calc_sample = _dspin(0.1, 100.0, 5.0, 2, 0.5, " mm")
        self.calc_out = QtWidgets.QPlainTextEdit(readOnly=True)
        self.calc_out.setFont(_mono())
        form = QtWidgets.QFormLayout()
        form.addRow("Reflection HKL (current lattice)", self.calc_hkl)
        form.addRow("Material", self.calc_mat)
        form.addRow("Incidence angle", self.calc_alpha)
        form.addRow("Beam height (vertical size)", self.calc_beam)
        form.addRow("Sample length along the beam", self.calc_sample)
        lay = QtWidgets.QHBoxLayout(w)
        box = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(box)
        bl.addLayout(form)
        bl.addStretch(1)
        lay.addWidget(box)
        lay.addWidget(self.calc_out, 1)
        for wid in (self.calc_alpha, self.calc_beam, self.calc_sample):
            wid.valueChanged.connect(self.update_calc)
        self.calc_hkl.textChanged.connect(self.update_calc)
        self.calc_mat.currentIndexChanged.connect(self.update_calc)
        return w

    def update_calc(self, *_):
        try:
            e = self.energy.value()
            lat = self.current_lattice()
            hkl = self._vec(self.calc_hkl.text(), "HKL")
            formula, rho = xtools.MATERIALS[self.calc_mat.currentText()]
            delta, beta = xtools.refraction(formula, rho, e)
            ac = xtools.critical_angle(formula, rho, e)
            a = self.calc_alpha.value()
            fp, frac = xtools.footprint(self.calc_beam.value(), a, self.calc_sample.value())
            lines = [f"Energy {e:.4f} keV, wavelength {self.wavelength:.5f} Å, k = {2 * math.pi / self.wavelength:.4f} 1/Å",
                     "",
                     f"({' '.join(f'{x:g}' for x in hkl)}) in {lat.name}: |Q| = {lat.q(hkl)[0]:.4f} 1/Å, "
                     f"d = {lat.d(hkl)[0]:.4f} Å, 2theta = {xtools.two_theta(lat, hkl, e):.3f}°",
                     "",
                     f"{self.calc_mat.currentText()} ({formula}, {rho:g} g/cm³): delta = {delta:.3e}, beta = {beta:.3e}",
                     f"  critical angle {ac:.4f}°; at alpha = {a:g}° (alpha / alpha_c = {a / ac:.2f}) the 1/e "
                     f"penetration depth is {xtools.penetration_depth(a, formula, rho, e):.4g} Å",
                     "",
                     f"Footprint of a {self.calc_beam.value():g} mm beam at {a:g}°: {fp:.2f} mm; "
                     f"{100 * frac:.0f} % of the beam lands on a {self.calc_sample.value():g} mm sample",
                     f"  the whole beam lands on the sample above alpha = "
                     f"{xtools.alpha_for_footprint(self.calc_beam.value(), self.calc_sample.value()):.3f}°",
                     "",
                     "Note: the fitting model takes structure factors; footprint, Lorentz and polarization",
                     "corrections belong to the data reduction (see docs/ruo2_tio2_ctr_scan_plan.md)."]
            self.calc_out.setPlainText("\n".join(lines))
        except (ValueError, KeyError) as ex:
            self.calc_out.setPlainText(str(ex))


def main(argv=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv if argv is None else argv)
    app.setStyle("Fusion")
    win = BeamlineWindow()
    win.show()
    return app.exec() if hasattr(app, "exec") else app.exec_()


if __name__ == "__main__":
    sys.exit(main())

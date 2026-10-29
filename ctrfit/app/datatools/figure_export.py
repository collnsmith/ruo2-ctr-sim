"""Export figure window of the plotter: a preview at the true proportions of the saved image and every
option for it (size, DPI, format, fonts, colours, title and labels, limits and scales, ticks, grid,
legend, and each trace: shown, label, colour, line, marker, axis). The working plot is not changed;
the logic is in ctrfit.datatools.figure.
"""
import os

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QApplication, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog,
                             QHeaderView, QLineEdit, QMainWindow, QPushButton, QScrollArea, QSpinBox,
                             QSplitter, QTableWidget, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ctrfit.app.datatools.common import StatusMixin, group, hint, primary_button, row, sidebar
from ctrfit.datatools.figure import (CM, FONTS, FORMATS, LEGEND_LOCS, LINESTYLES, MARKERS, FigureSpec,
                                     FigureStyle, render, save)
from ctrfit.datatools.paths import settings_file

STYLE_FILE = "plot_export_style.json"
PRESETS = {
    "Paper, one column (8.5 cm)": dict(width=8.5 / CM, height=6.5 / CM, dpi=600, title_size=9, label_size=9,
                                      tick_size=8, legend_size=7, spine_width=0.8),
    "Paper, two columns (17 cm)": dict(width=17 / CM, height=9 / CM, dpi=600, title_size=10, label_size=10,
                                      tick_size=9, legend_size=8, spine_width=0.8),
    "Slide (16:9)": dict(width=10.0, height=5.6, dpi=200, title_size=18, label_size=16, tick_size=14,
                         legend_size=13, spine_width=1.5),
    "Poster": dict(width=12.0, height=8.0, dpi=300, title_size=24, label_size=22, tick_size=18, legend_size=18,
                   spine_width=2.0),
    "Dark background": dict(fig_bg="#1e1e1e", axes_bg="#1e1e1e", text_color="#eeeeee"),
}


def _spin(lo, hi, val, dec=2, step=None, suffix=""):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setValue(val)
    if step:
        w.setSingleStep(step)
    if suffix:
        w.setSuffix(suffix)
    return w


class ColorButton(QPushButton):
    """A swatch that opens a colour picker."""

    def __init__(self, color, on_change):
        super().__init__()
        self.setFixedWidth(46)
        self._on_change = on_change
        self.set_color(color)
        self.clicked.connect(self._pick)

    def set_color(self, color):
        self.color = QColor(color).name()
        self.setStyleSheet(f"background: {self.color}; border: 1px solid #888; min-height: 18px;")

    def _pick(self):
        c = QColorDialog.getColor(QColor(self.color), self, "Colour")
        if c.isValid():
            self.set_color(c.name())
            self._on_change()


class FigureExportWindow(StatusMixin, QMainWindow):
    """Preview and options for the final image of a FigureSpec (a snapshot of the working plot)."""

    def __init__(self, spec, get_spec=None, parent=None, default_name="plot"):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("Export figure")
        self.resize(1400, 900)
        self.spec = spec
        self._get_spec = get_spec            # to refresh the snapshot from the working plot
        self._default_name = default_name
        self._loading = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._render)
        self.style = FigureStyle.from_spec(spec, base=self._last_style())
        self._build()
        self._load_style_into_widgets()
        self.say("Adjust the figure; the preview has the proportions of the saved image. Save with Ctrl+S.")

    # ------------------------------------------------------------------ persistence
    @staticmethod
    def _last_style():
        try:
            with open(settings_file(STYLE_FILE)) as f:
                return FigureStyle.from_json(f.read())
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def _remember_style(self):
        try:
            with open(settings_file(STYLE_FILE), "w") as f:
                f.write(self.style.to_json())
        except OSError:
            pass

    # ------------------------------------------------------------------ layout
    def _build(self):
        ch = self._changed
        # size and output
        sz, sf = group("Size and output")
        self._units = QComboBox()
        self._units.addItems(["in", "cm"])
        self._units.currentIndexChanged.connect(self._units_changed)
        self._w = _spin(0.5, 250, 6.0, 2, 0.1)
        self._h = _spin(0.5, 250, 4.5, 2, 0.1)
        for w in (self._w, self._h):
            w.valueChanged.connect(ch)
        sf.addRow("Width, height", row(self._w, self._h, self._units, stretch_first=False))
        self._preset = QComboBox()
        self._preset.addItem("Preset…")
        self._preset.addItems(list(PRESETS))
        self._preset.activated.connect(self._apply_preset)
        sf.addRow("Preset", self._preset)
        self._dpi = QSpinBox()
        self._dpi.setRange(50, 2400)
        self._dpi.setSingleStep(50)
        self._dpi.valueChanged.connect(ch)
        self._fmt = QComboBox()
        self._fmt.addItems([f.upper() for f in FORMATS])
        sf.addRow("DPI, format", row(self._dpi, self._fmt, stretch_first=False))
        self._transparent = QCheckBox("Transparent background")
        self._tight = QCheckBox("Tight layout")
        for w in (self._transparent, self._tight):
            w.toggled.connect(ch)
        sf.addRow(row(self._tight, self._transparent, 1, stretch_first=False))
        self._px = hint("")
        sf.addRow(self._px)

        # text
        tx, tf = group("Title and labels")
        self._title, self._xlabel, self._ylabel, self._y2label = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        for w in (self._title, self._xlabel, self._ylabel, self._y2label):
            w.textChanged.connect(ch)
        self._y2label.setPlaceholderText("only with a second y axis")
        tf.addRow("Title", self._title)
        tf.addRow("X label", self._xlabel)
        tf.addRow("Y label", self._ylabel)
        tf.addRow("Right Y label", self._y2label)
        self._font = QComboBox()
        self._font.setEditable(True)
        self._font.addItems(list(FONTS))
        self._font.currentTextChanged.connect(ch)
        tf.addRow("Font", self._font)
        self._fs = {}
        for key, label in (("title_size", "title"), ("label_size", "labels"), ("tick_size", "ticks"),
                           ("legend_size", "legend")):
            w = _spin(4, 72, 10, 1, 0.5, " pt")
            w.setPrefix(label + " ")
            w.valueChanged.connect(ch)
            self._fs[key] = w
        tf.addRow("Sizes", row(self._fs["title_size"], self._fs["label_size"], stretch_first=False))
        tf.addRow("", row(self._fs["tick_size"], self._fs["legend_size"], stretch_first=False))

        # colours
        co, cf = group("Colours")
        self._c_fig = ColorButton("#ffffff", ch)
        self._c_axes = ColorButton("#ffffff", ch)
        self._c_text = ColorButton("#000000", ch)
        cf.addRow("Figure, axes, text", row(self._c_fig, self._c_axes, self._c_text, 1, stretch_first=False))

        # axes
        ax, af = group("Axes")
        self._lims = {}
        for key in ("xmin", "xmax", "ymin", "ymax", "y2min", "y2max"):
            e = QLineEdit()
            e.setPlaceholderText("auto")
            e.textChanged.connect(ch)
            self._lims[key] = e
        af.addRow("X from, to", row(self._lims["xmin"], self._lims["xmax"]))
        af.addRow("Y from, to", row(self._lims["ymin"], self._lims["ymax"]))
        af.addRow("Right Y from, to", row(self._lims["y2min"], self._lims["y2max"]))
        self._xlog, self._ylog = QCheckBox("Log X"), QCheckBox("Log Y")
        for w in (self._xlog, self._ylog):
            w.toggled.connect(ch)
        b_fromplot = QPushButton("Limits from plot")
        b_fromplot.setToolTip("Use the axis ranges of the working plot (e.g. after zooming there)")
        b_fromplot.clicked.connect(self._limits_from_plot)
        af.addRow(row(self._xlog, self._ylog, b_fromplot, stretch_first=False))

        # ticks, grid, legend
        tg, gf = group("Ticks, grid, legend")
        self._ticks_in, self._ticks_all, self._minor = (QCheckBox("Ticks inside"), QCheckBox("All sides"),
                                                       QCheckBox("Minor ticks"))
        for w in (self._ticks_in, self._ticks_all, self._minor):
            w.toggled.connect(ch)
        gf.addRow(row(self._ticks_in, self._ticks_all, self._minor, stretch_first=False))
        self._spine = _spin(0.1, 6, 1.0, 1, 0.1, " pt")
        self._spine.valueChanged.connect(ch)
        gf.addRow("Frame width", self._spine)
        self._grid = QCheckBox("Grid")
        self._grid_style = QComboBox()
        self._grid_style.addItems(["-", "--", "-.", ":"])
        self._grid_alpha = _spin(0, 1, 0.5, 2, 0.05)
        self._grid_alpha.setPrefix("alpha ")
        for w in (self._grid, ):
            w.toggled.connect(ch)
        self._grid_style.currentIndexChanged.connect(ch)
        self._grid_alpha.valueChanged.connect(ch)
        gf.addRow(row(self._grid, self._grid_style, self._grid_alpha, stretch_first=False))
        self._legend, self._legend_frame = QCheckBox("Legend"), QCheckBox("Frame")
        self._legend_loc = QComboBox()
        self._legend_loc.addItems(list(LEGEND_LOCS))
        for w in (self._legend, self._legend_frame):
            w.toggled.connect(ch)
        self._legend_loc.currentIndexChanged.connect(ch)
        gf.addRow(row(self._legend, self._legend_loc, self._legend_frame, stretch_first=False))

        b_refresh = QPushButton("Refresh from plot")
        b_refresh.setToolTip("Take the working plot's data again (e.g. after changing scans); keeps these settings")
        b_refresh.clicked.connect(self.refresh_from_plot)
        b_reset = QPushButton("Reset traces")
        b_reset.setToolTip("Trace colours, lines and markers as in the working plot")
        b_reset.clicked.connect(self._reset_traces)
        b_copy = QPushButton("Copy image")
        b_copy.setToolTip("Copy the figure (at the export size) to the clipboard")
        b_copy.clicked.connect(self.copy_to_clipboard)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(sidebar(sz, tx, co, ax, tg, width=400), 1)
        lv.addWidget(row(b_refresh, b_reset, b_copy, stretch_first=False))
        lv.addWidget(primary_button("Save figure…", self.save_dialog, "Ctrl+S", self))

        # preview and traces
        self._fig = Figure()
        self._canvas = FigureCanvas(self._fig)
        self._scroll = QScrollArea()
        self._scroll.setAlignment(Qt.AlignCenter)
        self._scroll.setWidget(self._canvas)
        self._scroll.setStyleSheet("QScrollArea { background: #9a9a9a; }")
        self._zoom = hint("")
        prev = QWidget()
        pv = QVBoxLayout(prev)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(self._scroll, 1)
        pv.addWidget(self._zoom)
        tr, trl = group("Traces", form=False)
        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(["Show", "Label", "Colour", "Line", "Width", "Marker", "Size", "Axis"])
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        trl.addWidget(self._table)
        right = QSplitter(Qt.Vertical)
        right.addWidget(prev)
        right.addWidget(tr)
        right.setSizes([640, 240])
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([420, 980])
        self.setCentralWidget(split)

    # ------------------------------------------------------------------ widgets <-> style
    def _inch(self, v):
        return v / CM if self._units.currentText() == "cm" else v

    def _units_changed(self):
        k = CM if self._units.currentText() == "cm" else 1.0
        self._loading = True
        self._w.setValue(self.style.width * k)
        self._h.setValue(self.style.height * k)
        self._loading = False
        self._changed()

    def _load_style_into_widgets(self):
        st = self.style
        self._loading = True
        k = CM if self._units.currentText() == "cm" else 1.0
        self._w.setValue(st.width * k)
        self._h.setValue(st.height * k)
        self._dpi.setValue(int(st.dpi))
        self._transparent.setChecked(st.transparent)
        self._tight.setChecked(st.tight)
        for w, v in ((self._title, st.title), (self._xlabel, st.xlabel), (self._ylabel, st.ylabel),
                     (self._y2label, st.y2label)):
            w.setText(v)
        self._font.setCurrentText(st.font)
        for key, w in self._fs.items():
            w.setValue(getattr(st, key))
        self._c_fig.set_color(st.fig_bg)
        self._c_axes.set_color(st.axes_bg)
        self._c_text.set_color(st.text_color)
        for key, e in self._lims.items():
            e.setText(getattr(st, key))
        self._xlog.setChecked(st.xlog)
        self._ylog.setChecked(st.ylog)
        self._ticks_in.setChecked(st.ticks_in)
        self._ticks_all.setChecked(st.ticks_all_sides)
        self._minor.setChecked(st.minor_ticks)
        self._spine.setValue(st.spine_width)
        self._grid.setChecked(st.grid)
        self._grid_style.setCurrentText(st.grid_style)
        self._grid_alpha.setValue(st.grid_alpha)
        self._legend.setChecked(st.legend)
        self._legend_frame.setChecked(st.legend_frame)
        self._legend_loc.setCurrentText(st.legend_loc)
        self._y2label.setEnabled(self.spec.has_right)
        for key in ("y2min", "y2max"):
            self._lims[key].setEnabled(self.spec.has_right)
        self._fill_table()
        self._loading = False
        self._render()

    def _fill_table(self):
        self._table.setRowCount(len(self.style.traces))
        self._rows = []
        for i, ts in enumerate(self.style.traces):
            show = QCheckBox()
            show.setChecked(ts.visible)
            label = QLineEdit(ts.label)
            color = ColorButton(ts.color, self._changed)
            line = QComboBox()
            line.addItems(list(LINESTYLES))
            line.setCurrentText(ts.linestyle)
            width = _spin(0, 20, ts.linewidth, 1, 0.25)
            marker = QComboBox()
            marker.addItems(list(MARKERS))
            marker.setCurrentText(ts.marker)
            size = _spin(0, 40, ts.markersize, 1, 0.5)
            axis = QComboBox()
            axis.addItems(["left", "right"])
            axis.setCurrentText(ts.axis)
            show.toggled.connect(self._changed)
            label.textChanged.connect(self._changed)
            for w in (line, marker, axis):
                w.currentIndexChanged.connect(self._changed)
            for w in (width, size):
                w.valueChanged.connect(self._changed)
            for c, w in enumerate((show, label, color, line, width, marker, size, axis)):
                self._table.setCellWidget(i, c, w)
            self._rows.append((show, label, color, line, width, marker, size, axis))
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

    def _read_widgets(self):
        st = self.style
        k = CM if self._units.currentText() == "cm" else 1.0
        for attr, w in (("width", self._w), ("height", self._h)):     # keep the exact size unless edited
            if abs(w.value() - round(getattr(st, attr) * k, w.decimals())) > 1e-9:
                setattr(st, attr, self._inch(w.value()))
        st.dpi = self._dpi.value()
        st.transparent = self._transparent.isChecked()
        st.tight = self._tight.isChecked()
        st.title, st.xlabel, st.ylabel, st.y2label = (self._title.text(), self._xlabel.text(), self._ylabel.text(),
                                                      self._y2label.text())
        st.font = self._font.currentText()
        for key, w in self._fs.items():
            setattr(st, key, w.value())
        st.fig_bg, st.axes_bg, st.text_color = self._c_fig.color, self._c_axes.color, self._c_text.color
        for key, e in self._lims.items():
            setattr(st, key, e.text().strip())
        st.xlog, st.ylog = self._xlog.isChecked(), self._ylog.isChecked()
        st.ticks_in, st.ticks_all_sides = self._ticks_in.isChecked(), self._ticks_all.isChecked()
        st.minor_ticks = self._minor.isChecked()
        st.spine_width = self._spine.value()
        st.grid, st.grid_style, st.grid_alpha = (self._grid.isChecked(), self._grid_style.currentText(),
                                                 self._grid_alpha.value())
        st.legend, st.legend_frame = self._legend.isChecked(), self._legend_frame.isChecked()
        st.legend_loc = self._legend_loc.currentText()
        for ts, (show, label, color, line, width, marker, size, axis) in zip(st.traces, self._rows):
            ts.visible, ts.label, ts.color = show.isChecked(), label.text(), color.color
            ts.linestyle, ts.linewidth = line.currentText(), width.value()
            ts.marker, ts.markersize, ts.axis = marker.currentText(), size.value(), axis.currentText()

    # ------------------------------------------------------------------ actions
    def _changed(self, *_):
        if not self._loading:
            self._timer.start()

    def _apply_preset(self, index):
        name = self._preset.itemText(index)
        if name in PRESETS:
            self._read_widgets()
            for k, v in PRESETS[name].items():
                setattr(self.style, k, v)
            self._load_style_into_widgets()
            self.say(f"Preset: {name}")
        self._preset.setCurrentIndex(0)

    def _limits_from_plot(self):
        sp = self.spec
        self._loading = True
        for key, val in (("xmin", sp.xlim[0]), ("xmax", sp.xlim[1]), ("ymin", sp.ylim[0]), ("ymax", sp.ylim[1])):
            self._lims[key].setText(f"{val:.6g}")
        if sp.y2lim:
            self._lims["y2min"].setText(f"{sp.y2lim[0]:.6g}")
            self._lims["y2max"].setText(f"{sp.y2lim[1]:.6g}")
        self._loading = False
        self._changed()

    def refresh_from_plot(self):
        if self._get_spec is None:
            return
        try:
            spec = self._get_spec()
        except ValueError as e:
            self.say(str(e), error=True)
            return
        self._read_widgets()
        self.spec = spec
        self.style = FigureStyle.from_spec(spec, base=self.style, keep_text=True, keep_traces=True)
        self._load_style_into_widgets()
        self.say(f"{len(spec.traces)} traces from the working plot")

    def _reset_traces(self):
        self._read_widgets()
        fresh = FigureStyle.from_spec(self.spec)
        self.style.traces = fresh.traces
        self._load_style_into_widgets()

    def _render(self):
        """Draw the preview at the export size, scaled to fit (the same proportions as the file)."""
        self._read_widgets()
        st = self.style
        vw = max(self._scroll.viewport().width() - 20, 100)
        vh = max(self._scroll.viewport().height() - 20, 100)
        screen = self.logicalDpiX() or 96
        pdpi = max(20.0, min(vw / st.width, vh / st.height, 2.0 * screen))
        self._fig.set_dpi(pdpi)
        self._fig.set_size_inches(st.width, st.height)
        try:
            render(self._fig, self.spec, st)
        except Exception as e:  # a bad limit or font: show it, keep the window
            self._fig.clear()
            self._fig.text(0.5, 0.5, f"cannot draw: {e}", ha="center", va="center", color="#b00020")
        self._canvas.setFixedSize(int(st.width * pdpi), int(st.height * pdpi))
        self._canvas.draw_idle()
        self._zoom.setText(f"Preview at {100 * pdpi / screen:.0f} % of the printed size")
        self._px.setText(f"{st.width:.2f} × {st.height:.2f} in = {st.width * CM:.1f} × {st.height * CM:.1f} cm; "
                         f"{int(st.width * st.dpi)} × {int(st.height * st.dpi)} px at {st.dpi} DPI")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._changed()

    def save_to(self, path):
        self._read_widgets()
        fmt = os.path.splitext(path)[1].lstrip(".").lower() or self._fmt.currentText().lower()
        if not os.path.splitext(path)[1]:
            path = f"{path}.{fmt}"
        save(self.spec, self.style, path, fmt)
        self._remember_style()
        self.say(f"Saved {path} ({int(self.style.width * self.style.dpi)} × "
                 f"{int(self.style.height * self.style.dpi)} px)")
        return path

    def save_dialog(self):
        fmt = self._fmt.currentText().lower()
        filters = ";;".join(f"{f.upper()} (*.{f})" for f in [fmt] + [x for x in FORMATS if x != fmt])
        path, _ = QFileDialog.getSaveFileName(self, "Save figure", f"{self._default_name}.{fmt}", filters)
        if path:
            try:
                self.save_to(path)
            except (ValueError, OSError) as e:
                self.say(f"Could not save: {e}", error=True)

    def copy_to_clipboard(self):
        import io
        from PyQt5.QtGui import QImage
        self._read_widgets()
        buf = io.BytesIO()
        fig = Figure(figsize=(self.style.width, self.style.height), dpi=self.style.dpi)
        render(fig, self.spec, self.style)
        fig.savefig(buf, format="png", dpi=self.style.dpi, facecolor=fig.get_facecolor())
        QApplication.clipboard().setImage(QImage.fromData(buf.getvalue(), "PNG"))
        self._remember_style()
        self.say("Figure copied to the clipboard")

    def closeEvent(self, event):
        self._read_widgets()
        self._remember_style()
        super().closeEvent(event)


def spec_from_canvas(fig):
    """FigureSpec of the working plot (raises ValueError if nothing is plotted)."""
    return FigureSpec.from_axes(fig.axes)


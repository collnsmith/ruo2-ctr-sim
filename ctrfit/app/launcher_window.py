"""Launcher: one dark-themed window with a card for every app in the repository.

Apps are found by their '# @app ...' tag line (see ctrfit/apps.py), so a new script appears after
adding the tag and pressing Rescan. Windows ('gui') start detached and keep running when the
launcher closes; scripts run inside the launcher and print to its console.

Run:  python ctr_launcher.py   (or python -m ctrfit launcher)
"""
import sys
import time
from pathlib import Path

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    Signal = QtCore.pyqtSignal
except ImportError:  # pragma: no cover
    from PySide6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.Signal

from ..apps import discover

ACCENT = {"Simulate": "#4fa3ff", "Fit": "#f5a524", "Beamline": "#3ddc97"}
GLYPH = {"Simulate": "◆", "Fit": "▲", "Beamline": "●"}
OTHER = "#c084fc"

STYLE = """
QWidget { background: #0f1218; color: #e6e9ef; font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial;
          font-size: 13px; }
QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }
QLabel#title { font-size: 24px; font-weight: 700; color: #ffffff; }
QLabel#subtitle { color: #8b95a7; font-size: 12px; }
QLabel#group { font-size: 12px; font-weight: 700; letter-spacing: 2px; padding: 14px 4px 4px 4px; }
QLineEdit#search { background: #171c25; border: 1px solid #262d3a; border-radius: 10px; padding: 7px 12px;
                   color: #e6e9ef; selection-background-color: #4fa3ff; }
QLineEdit#search:focus { border: 1px solid #4fa3ff; }
QPushButton#ghost { background: #171c25; border: 1px solid #262d3a; border-radius: 10px; padding: 7px 14px;
                    color: #c7cdd8; }
QPushButton#ghost:hover { border-color: #4fa3ff; color: #ffffff; }
QFrame#card { background: #171c25; border: 1px solid #232a36; border-radius: 16px; }
QFrame#card:hover { background: #1c2230; }
QFrame#card[disabled="true"] { background: #13161d; }
QLabel#cardtitle { font-size: 15px; font-weight: 650; color: #ffffff; background: transparent; }
QLabel#carddesc { color: #9aa4b2; background: transparent; }
QLabel#chip { border-radius: 8px; padding: 2px 8px; font-size: 11px; background: #232a36; color: #aab3c2; }
QLabel#path { color: #5f6878; font-size: 11px; background: transparent; }
QPlainTextEdit#console { background: #0a0c10; border: 1px solid #1f2530; border-radius: 12px; color: #b9c2d0;
                         font-family: "JetBrains Mono", "Consolas", "DejaVu Sans Mono", monospace; font-size: 12px;
                         padding: 6px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #2a3140; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { background: #1c2230; color: #e6e9ef; border: 1px solid #2c3444; padding: 4px; }
"""


def tint(color, alpha=0.14):
    """rgba() of a #rrggbb colour (Qt style sheets read 8-digit hex as #aarrggbb)."""
    c = QtGui.QColor(color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha})"


class AppCard(QtWidgets.QFrame):
    clicked = Signal(object)

    def __init__(self, app, root):
        super().__init__()
        self.app = app
        self.setObjectName("card")
        color = ACCENT.get(app.group, OTHER)
        self.missing = app.missing
        self.setProperty("disabled", bool(self.missing))
        self.setCursor(QtCore.Qt.ArrowCursor if self.missing else QtCore.Qt.PointingHandCursor)
        self.setMinimumHeight(118)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        icon = QtWidgets.QLabel(GLYPH.get(app.group, "■"))
        icon.setAlignment(QtCore.Qt.AlignCenter)
        icon.setFixedSize(44, 44)
        icon.setStyleSheet(f"background: {tint(color)}; color: {color}; border-radius: 12px; font-size: 20px;")
        title = QtWidgets.QLabel(app.title)
        title.setObjectName("cardtitle")
        desc = QtWidgets.QLabel(app.desc or " ")
        desc.setObjectName("carddesc")
        desc.setWordWrap(True)
        self.kind = QtWidgets.QLabel("window" if app.kind == "gui" else "script")
        self.kind.setObjectName("chip")
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("chip")
        self.status.hide()
        try:
            rel = app.path.relative_to(root)
        except ValueError:
            rel = app.path
        path = QtWidgets.QLabel(str(rel))
        path.setObjectName("path")

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(icon, 0, QtCore.Qt.AlignTop)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(3)
        text.addWidget(title)
        text.addWidget(desc)
        top.addLayout(text, 1)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.kind)
        bottom.addWidget(self.status)
        bottom.addStretch(1)
        bottom.addWidget(path)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.addLayout(top)
        lay.addStretch(1)
        lay.addLayout(bottom)
        if self.missing:
            self.setToolTip(f"Needs {', '.join(self.missing)}: pip install {' '.join(self.missing)}")
            self.set_status("needs " + ", ".join(self.missing), "#f87171")
            for w in (title, desc):
                w.setStyleSheet("color: #6b7280; background: transparent;")
        else:
            self.setToolTip(f"Start {app.title} ({rel})")
        self._glow = QtWidgets.QGraphicsDropShadowEffect(self, blurRadius=0, offset=QtCore.QPointF(0, 0),
                                                        color=QtGui.QColor(color))
        self.setGraphicsEffect(self._glow)

    def set_status(self, text, color="#3ddc97"):
        self.status.setText(text)
        self.status.setStyleSheet(f"background: {tint(color)}; color: {color}; border-radius: 8px; padding: 2px 8px; "
                                  "font-size: 11px;")
        self.status.setVisible(bool(text))

    def enterEvent(self, ev):
        if not self.missing:
            self._glow.setBlurRadius(28)
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._glow.setBlurRadius(0)
        super().leaveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and not self.missing and self.rect().contains(ev.pos()):
            self.clicked.emit(self.app)
        super().mouseReleaseEvent(ev)


class LauncherWindow(QtWidgets.QWidget):
    def __init__(self, root=None, columns=3):
        super().__init__()
        self.root = Path(root) if root else Path(__file__).resolve().parents[2]
        self.columns = columns
        self.cards = []
        self.procs = {}                       # script path -> QProcess
        self.setWindowTitle("RuO2 / TiO2(110) toolkit")
        self.setStyleSheet(STYLE)

        title = QtWidgets.QLabel("RuO2 / TiO2(110) toolkit")
        title.setObjectName("title")
        self.subtitle = QtWidgets.QLabel()
        self.subtitle.setObjectName("subtitle")
        self.search = QtWidgets.QLineEdit()
        self.search.setObjectName("search")
        self.search.setPlaceholderText("Filter apps…")
        self.search.setFixedWidth(260)
        rescan = QtWidgets.QPushButton("Rescan")
        rescan.setObjectName("ghost")
        self.toggle_console = QtWidgets.QPushButton("Console")
        self.toggle_console.setObjectName("ghost")
        self.toggle_console.setCheckable(True)

        head = QtWidgets.QHBoxLayout()
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(title)
        titles.addWidget(self.subtitle)
        head.addLayout(titles, 1)
        head.addWidget(self.search)
        head.addWidget(rescan)
        head.addWidget(self.toggle_console)

        self.body = QtWidgets.QWidget()
        self.grid = QtWidgets.QVBoxLayout(self.body)
        self.grid.setContentsMargins(0, 0, 6, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.body)

        self.console = QtWidgets.QPlainTextEdit(readOnly=True)
        self.console.setObjectName("console")
        self.console.setMaximumBlockCount(5000)
        self.console.setMinimumHeight(120)
        self.console.setMaximumHeight(200)
        self.console.hide()

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(26, 22, 26, 20)
        lay.setSpacing(12)
        lay.addLayout(head)
        lay.addWidget(scroll, 1)
        lay.addWidget(self.console)

        self.search.textChanged.connect(self._filter)
        rescan.clicked.connect(self.scan)
        self.toggle_console.toggled.connect(self.console.setVisible)
        shortcut = getattr(QtWidgets, "QShortcut", None) or QtGui.QShortcut
        shortcut(QtGui.QKeySequence("Ctrl+F"), self, activated=self.search.setFocus)
        shortcut(QtGui.QKeySequence("F5"), self, activated=self.scan)
        self.resize(1080, 760)
        self.scan()

    # ------------------------------------------------------------------ cards
    def scan(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                _clear(item.layout())
        self.cards = []
        apps = discover(self.root)
        groups = []
        for a in apps:
            if a.group not in groups:
                groups.append(a.group)
        for g in groups:
            color = ACCENT.get(g, OTHER)
            head = QtWidgets.QLabel(g.upper())
            head.setObjectName("group")
            head.setStyleSheet(f"color: {color};")
            self.grid.addWidget(head)
            grid = QtWidgets.QGridLayout()
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(14)
            members = [a for a in apps if a.group == g]
            for i, a in enumerate(members):
                card = AppCard(a, self.root)
                card.clicked.connect(self.launch)
                if a.path in self.procs:
                    card.set_status("running")
                self.cards.append(card)
                grid.addWidget(card, i // self.columns, i % self.columns)
            for c in range(self.columns):
                grid.setColumnStretch(c, 1)
            holder = QtWidgets.QWidget()
            holder.setLayout(grid)
            self.grid.addWidget(holder)
        self.grid.addStretch(1)
        n_missing = sum(1 for c in self.cards if c.missing)
        self.subtitle.setText(f"{len(self.cards)} apps in {self.root}" +
                              (f"  ·  {n_missing} need packages that are not installed" if n_missing else "") +
                              "  ·  tag a script with  # @app title: …  to add it")
        self._filter(self.search.text())
        return apps

    def _filter(self, text):
        t = text.strip().lower()
        for c in self.cards:
            hit = not t or t in f"{c.app.title} {c.app.desc} {c.app.group} {c.app.path.name}".lower()
            c.setVisible(hit)

    # ------------------------------------------------------------------ running
    def log(self, text):
        self.console.appendPlainText(text.rstrip("\n"))

    def card_for(self, app):
        return next((c for c in self.cards if c.app.path == app.path), None)

    def launch(self, app):
        stamp = time.strftime("%H:%M:%S")
        if app.kind == "gui":
            ok, pid = QtCore.QProcess.startDetached(sys.executable, [str(app.path)], str(app.path.parent))
            self.log(f"[{stamp}] started {app.title} ({app.path.name})" + (f", pid {pid}" if ok else " FAILED"))
            card = self.card_for(app)
            if card:
                card.set_status("started" if ok else "failed", "#3ddc97" if ok else "#f87171")
                timer = QtCore.QTimer(card, singleShot=True, interval=4000)   # dies with the card on rescan
                timer.timeout.connect(lambda c=card: c.set_status(""))
                timer.start()
            return ok
        if app.path in self.procs:
            self.log(f"[{stamp}] {app.title} is already running")
            return False
        self.toggle_console.setChecked(True)
        proc = QtCore.QProcess(self)
        proc.setProgram(sys.executable)
        proc.setArguments(["-u", str(app.path)])
        proc.setWorkingDirectory(str(app.path.parent))
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(lambda p=proc: self.log(bytes(p.readAllStandardOutput()).decode(
            "utf-8", "replace")))
        proc.finished.connect(lambda code, _s=None, a=app: self._finished(a, code))
        self.procs[app.path] = proc
        self.log(f"[{stamp}] ▶ {app.title} ({app.path.name})")
        card = self.card_for(app)
        if card:
            card.set_status("running")
        proc.start()
        return True

    def _finished(self, app, code):
        self.procs.pop(app.path, None)
        self.log(f"[{time.strftime('%H:%M:%S')}] ■ {app.title} finished (exit code {code})")
        card = self.card_for(app)
        if card:
            card.set_status("done" if code == 0 else f"exit {code}", "#3ddc97" if code == 0 else "#f87171")

    def closeEvent(self, ev):
        for p in list(self.procs.values()):
            p.kill()
            p.waitForFinished(2000)
        super().closeEvent(ev)


def _clear(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            _clear(item.layout())


def main(argv=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv if argv is None else argv)
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, col in ((QtGui.QPalette.Window, "#0f1218"), (QtGui.QPalette.Base, "#171c25"),
                      (QtGui.QPalette.Text, "#e6e9ef"), (QtGui.QPalette.WindowText, "#e6e9ef"),
                      (QtGui.QPalette.Button, "#171c25"), (QtGui.QPalette.ButtonText, "#e6e9ef"),
                      (QtGui.QPalette.Highlight, "#4fa3ff")):
        pal.setColor(role, QtGui.QColor(col))
    app.setPalette(pal)
    win = LauncherWindow()
    win.show()
    return app.exec() if hasattr(app, "exec") else app.exec_()


if __name__ == "__main__":
    sys.exit(main())

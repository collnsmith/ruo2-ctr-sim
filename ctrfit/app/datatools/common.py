"""Shared Qt pieces for the data tools, in the style of the other ctrfit windows: the system theme,
a sidebar of titled groups on the left, one blue primary button per tool, messages in the status
bar (errors also in a dialog), and running one tool on its own."""
import sys
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontDatabase, QKeySequence
from PyQt5.QtWidgets import (QApplication, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QScrollArea, QShortcut, QSizePolicy, QVBoxLayout, QWidget)

PRIMARY_STYLE = ("QPushButton { background: #1f77b4; color: white; font-weight: 600; border-radius: 4px; "
                 "padding: 6px 12px; } QPushButton:disabled { background: #9bbbd6; }")
HINT_STYLE = "color: #666;"
IMAGE_BG = "background: #1a1a1a; border: 1px solid #444;"     # detector images stay on a dark ground


def mono_font():
    return QFontDatabase.systemFont(QFontDatabase.FixedFont)


def primary_button(text, slot=None, shortcut=None, owner=None):
    """The tool's main action (blue). shortcut works while `owner` (the tool window) is on screen."""
    b = QPushButton(text + (f"  ({shortcut})" if shortcut else ""))
    b.setStyleSheet(PRIMARY_STYLE)
    b.setMinimumHeight(32)
    if slot is not None:
        b.clicked.connect(lambda *_: slot())
    if shortcut and owner is not None and slot is not None:
        sc = QShortcut(QKeySequence(shortcut), owner)
        sc.setContext(Qt.WidgetWithChildrenShortcut)          # tabs share one top-level window
        sc.activated.connect(slot)
    return b


def hint(text):
    """A muted, word-wrapped help line."""
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(HINT_STYLE)
    return lab


def group(title, form=True):
    """(QGroupBox, its layout): a form layout by default, else a vertical box."""
    box = QGroupBox(title)
    lay = QFormLayout(box) if form else QVBoxLayout(box)
    if form:
        lay.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    return box, lay


def row(*widgets, stretch_first=True):
    """Widgets side by side in one QWidget (e.g. a path field and its Browse button)."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for i, x in enumerate(widgets):
        if isinstance(x, int):
            lay.addStretch(x)
        else:
            lay.addWidget(x, 1 if (stretch_first and i == 0) else 0)
    return w


def sidebar(*items, width=380):
    """A vertical scroll area of groups/widgets, as the left column of a tool."""
    inner = QWidget()
    lay = QVBoxLayout(inner)
    lay.setContentsMargins(4, 4, 4, 4)
    for x in items:
        if x is None:
            continue
        lay.addWidget(x)
    lay.addStretch(1)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(inner)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    bar = scroll.style().pixelMetric(scroll.style().PM_ScrollBarExtent)
    scroll.setMinimumWidth(max(width, inner.minimumSizeHint().width() + bar + 6))   # never clip the groups
    scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    return scroll


class StatusMixin:
    """say(): one line in the status bar; errors also open a warning with the full text."""
    _status_label = None

    def say(self, text, error=False):
        text = str(text)
        owner = getattr(self, "_mw", None)
        if not hasattr(self, "statusBar") and hasattr(owner, "say"):
            owner.say(text, error)         # a dialog of a tool reports in the tool's status bar
            return
        bar = self.statusBar() if hasattr(self, "statusBar") else None
        if bar is not None:
            if self._status_label is None:
                self._status_label = QLabel()
                bar.addWidget(self._status_label, 1)
            self._status_label.setStyleSheet("color: #b00020;" if error else "")
            self._status_label.setText(text.splitlines()[0][:220] if text else "")
        if error:
            QMessageBox.warning(self, self.windowTitle() or "Data tools", text)
        elif bar is None:
            QMessageBox.information(self, self.windowTitle() or "Data tools", text)

    def guarded(self, fn):
        """Run fn, report exceptions with say() instead of raising (keeps the window alive)."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            return fn()
        except (ValueError, KeyError, RuntimeError, OSError) as ex:
            self.say(str(ex), error=True)
        except Exception:
            self.say(traceback.format_exc(), error=True)
        finally:
            QApplication.restoreOverrideCursor()
        return None


def run_tool(cls, **kw):
    """Run one tool window on its own (python -m ctrfit.app.datatools.<tool>)."""
    app = QApplication.instance() or QApplication(sys.argv)
    win = cls(**kw)
    win.show()
    return app.exec_()

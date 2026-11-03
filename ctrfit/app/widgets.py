"""Small Qt widgets and theme-aware colours shared by the ctrfit windows."""
try:  # same Qt choice as the windows
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:  # pragma: no cover
    from PySide6 import QtCore, QtGui, QtWidgets


# ---------------------------------------------------------------------- colours that follow the theme
# A fixed pale background (e.g. #fff0c2) under the theme's text is unreadable in a dark theme (white
# text on near white). These colours are worked out from the current palette each time they are
# used, so they follow light and dark themes; in a light theme they equal the old fixed colours.
FLAG = "#ffc107"        # amber: a parameter named in a warning, a search match
LINK = "#3b6fd1"        # blue: a linked parameter
TODO = "#ff9800"        # orange: a field still to fill
BAD = "#ff4444"         # red: a field that is wrong
DARK_BOOST = 1.6        # a dark base needs more of the accent to show


def palette():
    app = QtWidgets.QApplication.instance()
    return app.palette() if app is not None else QtGui.QPalette()


def is_dark(pal=None):
    return (pal or palette()).color(QtGui.QPalette.Base).lightness() < 128


def tint(accent, amount, pal=None, min_contrast=4.5):
    """A cell background: `accent` mixed into the palette's base colour (amount 0..1, more in a dark
    theme), and less of it when needed so that the palette's own text keeps a contrast of at least
    min_contrast on it."""
    pal = pal or palette()
    base, a, txt = pal.color(QtGui.QPalette.Base), QtGui.QColor(accent), pal.color(QtGui.QPalette.Text)
    t = min(1.0, amount * (DARK_BOOST if is_dark(pal) else 1.0))
    while True:
        mix = lambda x, y: int(round(x + (y - x) * t))
        c = QtGui.QColor(mix(base.red(), a.red()), mix(base.green(), a.green()), mix(base.blue(), a.blue()))
        if t < 0.02 or contrast(c, txt) >= min_contrast:
            return c
        t *= 0.85


def text_color(pal=None):
    return (pal or palette()).color(QtGui.QPalette.Text)


def muted_text(pal=None):
    """Grey text for secondary entries (the theme's disabled text)."""
    return (pal or palette()).color(QtGui.QPalette.Disabled, QtGui.QPalette.Text)


def accent_text(light, dark, pal=None):
    """A coloured text: `light` on a light base, `dark` (a lighter shade) on a dark one."""
    return QtGui.QColor(dark if is_dark(pal) else light)


def alert_text(pal=None):
    return accent_text("#b00020", "#ff7b7b", pal)


def contrast(c1, c2):
    """WCAG contrast ratio of two colours (1 to 21; 4.5 is the usual minimum for text)."""
    def lum(c):
        ch = [v / 255.0 for v in (c.red(), c.green(), c.blue())]
        ch = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in ch]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
    a, b = sorted((lum(QtGui.QColor(c1)), lum(QtGui.QColor(c2))), reverse=True)
    return (a + 0.05) / (b + 0.05)


class ElidedLabel(QtWidgets.QLabel):
    """A one-line label that never asks for more width than it is given: text that does not fit ends in
    '…' and the whole text is the tooltip. A message that changes while a fit runs (progress, the
    list of free parameters) therefore cannot resize, and so move, the window. text() is the whole
    text."""

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.setMinimumWidth(20)
        self.setText(text)

    def setText(self, text):
        self._full = str(text or "")
        self.setToolTip(self._full if len(self._full) > 60 or "\n" in self._full else "")
        self._elide()

    def text(self):
        return self._full

    def _elide(self):
        line = self._full.splitlines()[0] if self._full else ""
        width = max(self.width() - 2, 10)
        super().setText(self.fontMetrics().elidedText(line, QtCore.Qt.ElideRight, width))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def minimumSizeHint(self):
        return QtCore.QSize(20, super().minimumSizeHint().height())

    def sizeHint(self):
        return QtCore.QSize(min(super().sizeHint().width(), 200), super().sizeHint().height())

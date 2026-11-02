"""Small Qt widgets shared by the ctrfit windows."""
try:  # same Qt choice as the windows
    from PyQt5 import QtCore, QtWidgets
except ImportError:  # pragma: no cover
    from PySide6 import QtCore, QtWidgets


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

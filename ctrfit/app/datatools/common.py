"""Shared Qt helpers for the data tools: the dark Fusion palette they were designed for and running
one tool on its own."""
import sys

from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

DARK = [(QPalette.Window, (53, 53, 53)), (QPalette.WindowText, (220, 220, 220)), (QPalette.Base, (35, 35, 35)),
        (QPalette.AlternateBase, (53, 53, 53)), (QPalette.ToolTipBase, (25, 25, 25)),
        (QPalette.ToolTipText, (220, 220, 220)), (QPalette.Text, (220, 220, 220)), (QPalette.Button, (53, 53, 53)),
        (QPalette.ButtonText, (220, 220, 220)), (QPalette.Link, (42, 130, 218)), (QPalette.Highlight, (42, 130, 218)),
        (QPalette.HighlightedText, (0, 0, 0))]


def dark_palette(app):
    app.setStyle("Fusion")
    pal = app.palette()
    for role, rgb in DARK:
        pal.setColor(role, QColor(*rgb))
    app.setPalette(pal)


def run_tool(cls, **kw):
    """Run one tool window on its own (python -m ctrfit.app.datatools.<tool>)."""
    app = QApplication.instance() or QApplication(sys.argv)
    dark_palette(app)
    win = cls(**kw)
    win.show()
    return app.exec_()

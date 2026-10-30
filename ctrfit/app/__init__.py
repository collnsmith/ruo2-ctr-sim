"""Qt application (the only place in ctrfit that imports Qt). Fitting window: fit_window.FitWindow."""
import os

# Qt 5 on Wayland logs "Wayland does not support QWindow::requestActivate()" every time a window,
# dialog or popup asks for focus. Harmless (the compositor decides focus), but it floods the console:
# hide that category's warnings unless the user set their own logging rules.
QUIET_RULES = "qt.qpa.wayland.warning=false"


def quiet_qt_logging():
    rules = os.environ.get("QT_LOGGING_RULES", "")
    if "qt.qpa.wayland" not in rules:
        os.environ["QT_LOGGING_RULES"] = f"{rules};{QUIET_RULES}" if rules else QUIET_RULES


quiet_qt_logging()

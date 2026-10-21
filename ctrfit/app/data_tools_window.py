"""SXRD data tools in one window, one tab per tool, in the order of the processing chain:

    Metadata -> Background subtraction (with the background ROI finder) -> Plotter (peak fits)
    -> CTR fit (quick models) -> HKL appender -> Smoother -> Combiner

Each tab is the tool's own window embedded as a widget; the logic is in ctrfit.datatools. Settings
of every tool are saved when this window closes (folder: ctrfit.datatools.paths.settings_dir()).
"""
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget


# (key, tab title, module, window class, tooltip)
TOOLS = [
    ("meta", "Metadata", "meta_viewer", "MainWindow", "HDF5 structure and metadata per scan; SPEC columns per scan"),
    ("bgsub", "Background subtraction", "bgsub", "MainWindow",
     "Eiger .h5 / .tif images: signal and background ROIs, per-image overrides, batch ROI sums"),
    ("bg_finder", "Background ROI finder", "bg_finder", "BgFinderWindow",
     "Per-scan pixel statistics, avoid bright spots, place background ROIs near the signal"),
    ("plotter", "Plotter", "plotter", "MainWindow",
     "SPEC scans with ROI sums, figure options, peak fits (rocking curves), smoothing, exports"),
    ("ctr_fit", "CTR fit", "ctr_fit", "CTRFitWindow", "Quick kinematic CTR models on integrated intensities"),
    ("appender", "HKL appender", "appender", "MainWindow",
     "Add H, K, L from the SPEC scan to plotter exports and peak-fit tables; GenX export"),
    ("smoother", "Smoother", "smoother", "HKLSmoother", "Smooth HKL files with Bragg exclusion zones"),
    ("combiner", "Combiner", "combiner", "CombinerWindow", "Combine HKL files into one H K L I table"),
]


class DataToolsWindow(QMainWindow):
    def __init__(self, tools=None):
        super().__init__()
        import importlib
        self.setWindowTitle("SXRD data tools")
        self.resize(1500, 950)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)
        self.tools = {}
        for key, title, mod, cls, tip in TOOLS:
            if tools is not None and key not in tools:
                continue
            w = getattr(importlib.import_module(f"ctrfit.app.datatools.{mod}"), cls)()
            w.setWindowFlags(Qt.Widget)          # a QMainWindow embedded as a tab page
            w.host = self
            self.tools[key] = w
            i = self.tabs.addTab(w, title)
            self.tabs.setTabToolTip(i, tip)

    def show_tool(self, key):
        """Bring one tool's tab to the front; returns its window."""
        w = self.tools[key]
        self.tabs.setCurrentWidget(w)
        return w

    def save_settings(self):
        for w in self.tools.values():
            save = getattr(w, "_save_settings", None)
            if save is not None:
                try:
                    save()
                except Exception as e:  # one tool's settings must not block closing
                    print(f"could not save settings of {type(w).__name__}: {e}", file=sys.stderr)

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)


def main(argv=None):
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    win = DataToolsWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

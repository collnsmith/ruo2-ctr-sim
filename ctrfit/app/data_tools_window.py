"""SXRD data tools in one window, one tab per tool, in the order of the processing chain:

    Metadata -> Background subtraction (with the background ROI finder) -> Plotter (peak fits)
    -> Scan library (film and CV / CA per scan, comparisons) -> Processing -> CTR fit (quick models)
    -> HKL appender -> Combiner

(Smoothing is in the plotter: "Smoothing" and "CTR smoothing" act on the plotted traces.)

Each tab is the tool's own window embedded as a widget; the logic is in ctrfit.datatools. Settings
of every tool are saved when this window closes (folder: ctrfit.datatools.paths.settings_dir()).
The tabs can be dragged into another order; the order and the open tab are kept (TAB_FILE there),
and a right click on the tab bar puts them back in the order above.
"""
import json
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QTabWidget


# (key, tab title, module, window class, tooltip)
TOOLS = [
    ("meta", "Metadata", "meta_viewer", "MainWindow", "HDF5 structure and metadata per scan; SPEC columns per scan"),
    ("bgsub", "Background subtraction", "bgsub", "MainWindow",
     "Eiger .h5 / .tif images: signal and background ROIs, per-image overrides, batch ROI sums"),
    ("bg_finder", "Background ROI finder", "bg_finder", "BgFinderWindow",
     "Per-scan pixel statistics, avoid bright spots, place background ROIs near the signal"),
    ("plotter", "Plotter", "plotter", "MainWindow",
     "SPEC scans with ROI sums, figure options, peak fits (rocking curves), smoothing, exports"),
    ("library", "Scan library", "library", "LibraryWindow",
     "Sort ROI sums scans by film and CV / CA (a library text file; data read only) and compare groups: "
     "CVs at one H K L, CAs at one potential on one rod, ..."),
    ("processing", "Processing", "processing", "ProcessingWindow",
     "ROI sums + SPEC: normalise, correct (polarisation, rod interception, area), |F| with errors, merge and scale"),
    ("ctr_fit", "CTR fit", "ctr_fit", "CTRFitWindow", "Quick kinematic CTR models on integrated intensities"),
    ("appender", "HKL appender", "appender", "MainWindow",
     "Add H, K, L from the SPEC scan to plotter exports and peak-fit tables; GenX export"),
    ("combiner", "Combiner", "combiner", "CombinerWindow", "Combine HKL files into one H K L I table"),
]


TAB_FILE = "sxrd_data_tools_tabs.json"       # {"order": [keys], "current": key}


class DataToolsWindow(QMainWindow):
    def __init__(self, tools=None):
        super().__init__()
        import importlib
        self.setWindowTitle("SXRD data tools")
        self.resize(1500, 950)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)                 # drag a tab to move it
        bar = self.tabs.tabBar()
        bar.setContextMenuPolicy(Qt.CustomContextMenu)
        bar.customContextMenuRequested.connect(self._tab_menu)
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
            self.tabs.setTabToolTip(i, tip + "\n(drag to move the tab; right click: reset the order)")
        self._restore_tabs()

    # ------------------------------------------------------------------ tab order
    def tab_order(self):
        """The tool keys in the order the tabs are shown."""
        keys = {id(w): k for k, w in self.tools.items()}
        return [keys[id(self.tabs.widget(i))] for i in range(self.tabs.count())]

    def set_tab_order(self, keys):
        """Put the tabs in this order (unknown keys skipped, tools not named keep their order after)."""
        keys = [k for k in dict.fromkeys(keys) if k in self.tools]
        keys += [k for k in self.tab_order() if k not in keys]
        for target, key in enumerate(keys):
            cur = self.tabs.indexOf(self.tools[key])
            if cur != target:
                self.tabs.tabBar().moveTab(cur, target)

    def reset_tab_order(self):
        self.set_tab_order([t[0] for t in TOOLS])

    def _tab_menu(self, pos):
        menu = QMenu(self)
        menu.addAction("Reset tab order", self.reset_tab_order)
        menu.exec_(self.tabs.tabBar().mapToGlobal(pos))

    def _restore_tabs(self):
        from ctrfit.datatools.paths import settings_file
        try:
            with open(settings_file(TAB_FILE)) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return
        if isinstance(d.get("order"), list):
            self.set_tab_order([str(k) for k in d["order"]])
        if d.get("current") in self.tools:
            self.tabs.setCurrentWidget(self.tools[d["current"]])

    def _save_tabs(self):
        from ctrfit.datatools.paths import settings_file
        cur = self.tabs.currentWidget()
        current = next((k for k, w in self.tools.items() if w is cur), None)
        try:
            with open(settings_file(TAB_FILE), "w") as f:
                json.dump({"order": self.tab_order(), "current": current}, f, indent=1)
        except OSError as e:
            print(f"could not save the tab order: {e}", file=sys.stderr)

    def show_tool(self, key):
        """Bring one tool's tab to the front; returns its window."""
        w = self.tools[key]
        self.tabs.setCurrentWidget(w)
        return w

    def save_settings(self):
        self._save_tabs()
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
    from .datatools.common import exec_and_delete, install_crash_handler
    win = DataToolsWindow()
    install_crash_handler(win)
    win.show()
    return exec_and_delete(app, win)


if __name__ == "__main__":
    sys.exit(main())

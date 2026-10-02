# @app title: Fitting | group: Fit | order: 10 | kind: gui | needs: PyQt5 | desc: Data, parameter table, guided workflow, live fits, uncertainties and Ask Claude
"""Fitting window for the RuO2/TiO2(110) CTR model (ctrfit). The simulator stays in ctr_gui.py.

Run:  python ctr_fit_gui.py [data.csv] [model.json] [project.ctrproj.json]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ctrfit.app.fit_window import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

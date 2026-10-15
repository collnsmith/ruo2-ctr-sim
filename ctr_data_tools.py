# @app title: SXRD data tools | group: Data | order: 10 | kind: gui | needs: PyQt5, h5py | desc: Metadata viewer, Eiger background subtraction and ROI finder, SPEC plotter with peak fits, quick CTR fit, HKL appender, smoother and combiner in one tabbed window
"""SXRD data processing tools in one tabbed window. Run:  python ctr_data_tools.py
(one tool alone: python -m ctrfit.app.datatools.<tool>, e.g. bgsub, plotter)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ctrfit.app.data_tools_window import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

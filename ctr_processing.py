# @app title: CTR processing | group: Data | order: 20 | kind: gui | needs: PyQt5 | desc: Background-subtracted ROI sums + SPEC file: normalise, correct (polarisation, rod interception, illuminated area), structure factors with errors, merge equivalents and scale; output for the fitting app and GenX
"""CTR processing (steps after background subtraction). Run:  python ctr_processing.py
(also a tab of python ctr_data_tools.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ctrfit.app.datatools.common import run_tool  # noqa: E402
from ctrfit.app.datatools.processing import ProcessingWindow  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_tool(ProcessingWindow))

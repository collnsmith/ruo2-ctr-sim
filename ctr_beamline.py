# @app title: Beamline helper | group: Beamline | order: 10 | kind: gui | needs: PyQt5 | desc: Bragg peak indexer (SPEC psic), HKL to angles, SPEC macro maker, calculators
"""Beamline helper for SXRD runs (SPEC psic): Bragg peak indexer, angle calculator, macro maker,
calculators. Run:  python ctr_beamline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ctrfit.app.beamline_window import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

# @app title: In-situ CV analysis | group: Electrochemistry | order: 10 | kind: gui | needs: PyQt5 | desc: Stationary SPEC scan (loopscan / unmoving phi scan) vs potential from EC-Lab .mpr/.mpt or a hand-defined CV; sync, background, sigmoid and CTR-model fits
"""In-situ CV analysis: intensity of a stationary SPEC scan against the potential of the CV.
Run:  python ctr_echem.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ctrfit.app.echem_window import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

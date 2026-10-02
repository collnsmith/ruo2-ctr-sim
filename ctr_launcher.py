"""Launcher: a card for every app in this folder (found by its '# @app' tag line).

Run:  python ctr_launcher.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ctrfit.app.launcher_window import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

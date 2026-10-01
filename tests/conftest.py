"""Shared test setup.

xraydb is blocked for every in-process test, so the "auto" anomalous terms always come from the
built-in table and the pinned numbers do not depend on what is installed.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))
sys.modules["xraydb"] = None          # import xraydb -> ImportError
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

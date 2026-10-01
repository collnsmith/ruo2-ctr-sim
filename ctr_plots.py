"""Compatibility name for ctrfit.project.plots (plotting). New code should import ctrfit.project.plots.

The module object is the same, so attributes set here (or patched in tests) act on ctrfit.project.plots.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # find ctrfit when run from another folder
import ctrfit.project.plots as _mod  # noqa: E402

sys.modules[__name__] = _mod

"""Compatibility name for ctrfit.core.ctrmodel (physics). New code should import ctrfit.core.ctrmodel.

The module object is the same, so attributes set here (or patched in tests) act on ctrfit.core.ctrmodel.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # find ctrfit when run from another folder
import ctrfit.core.ctrmodel as _mod  # noqa: E402

sys.modules[__name__] = _mod

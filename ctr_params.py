"""Compatibility name for ctrfit.core.settings (settings schema, units and parsers). New code should import ctrfit.core.settings.

The module object is the same, so attributes set here (or patched in tests) act on ctrfit.core.settings.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # find ctrfit when run from another folder
import ctrfit.core.settings as _mod  # noqa: E402

sys.modules[__name__] = _mod

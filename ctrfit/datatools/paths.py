"""Where the data tools keep their settings (window state, last folders, label map, plot configs).

The old stand-alone tools wrote their .ini files next to the scripts. Now they live in one folder:
$CTRFIT_SETTINGS_DIR if set, else ~/.ctrfit/data_tools. Copy old .ini files there to keep them.
"""
import os
from pathlib import Path


def settings_dir():
    """The settings folder (created on first use)."""
    d = Path(os.environ.get("CTRFIT_SETTINGS_DIR") or Path.home() / ".ctrfit" / "data_tools")
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def settings_file(name):
    """Path of one settings file in the settings folder."""
    return os.path.join(settings_dir(), name)

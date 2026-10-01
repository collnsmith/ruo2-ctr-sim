"""Package layout: ctrfit imports without Qt, the compatibility names point at ctrfit."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_qt_imports_in_ctrfit():
    bad = []
    for f in (ROOT / "ctrfit").rglob("*.py"):
        if "app" in f.relative_to(ROOT / "ctrfit").parts:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*(from|import)\s+(PyQt5|PyQt6|PySide2|PySide6|qtpy)\b", line):
                bad.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not bad, "\n".join(bad)


def test_ctrfit_import_does_not_load_qt():
    code = ("import sys, ctrfit, ctrfit.core, ctrfit.model, ctrfit.data, ctrfit.fit, ctrfit.project, "
            "ctrfit.project.plots; "
            "print([m for m in sys.modules if m.split('.')[0] in ('PyQt5', 'PySide6', 'PyQt6', 'PySide2')])")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"


def test_compat_modules_are_the_package_modules():
    import ctr_engine
    import ctr_params
    import ctr_plots
    import ctrfit.core.ctrmodel
    import ctrfit.core.settings
    import ctrfit.project.plots
    assert ctr_engine is ctrfit.core.ctrmodel
    assert ctr_params is ctrfit.core.settings
    assert ctr_plots is ctrfit.project.plots

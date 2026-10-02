"""App discovery for the launcher (no Qt).

A file shows up in the launcher when one of its first 40 lines is a tag comment:

    # @app title: Fitting | group: Fit | order: 10 | kind: gui | needs: PyQt5, numba | desc: What it does

title (required), group (section in the launcher), order (within the group), kind ('gui': runs in its
own window; 'script': output goes to the launcher console), needs (importable packages; missing ones
grey the button out), desc (one line). Add the tag to a new script and it appears on the next scan.
"""
import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path

TAG = re.compile(r"^#\s*@app\b(.*)$")
KINDS = ("gui", "script")
GROUP_ORDER = ("Simulate", "Fit", "Beamline")
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "build", "dist", "tests", "node_modules", ".venv", "venv"}


@dataclass
class App:
    path: Path
    title: str
    group: str = "Other"
    order: int = 100
    kind: str = "gui"
    needs: list = field(default_factory=list)
    desc: str = ""

    @property
    def missing(self):
        """Packages listed in `needs` that cannot be imported here."""
        out = []
        for name in self.needs:
            try:
                if importlib.util.find_spec(name) is None:
                    out.append(name)
            except (ImportError, ValueError):
                out.append(name)
        return out


def parse_tag(text):
    """Fields of one '@app' tag line (the part after '@app'), as a dict."""
    out = {}
    for part in text.split("|"):
        key, sep, value = part.partition(":")
        if sep:
            out[key.strip().lower()] = value.strip()
    return out


def read_app(path, max_lines=40):
    """App for a file with an @app tag, else None."""
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= max_lines:
                    break
                m = TAG.match(line.rstrip())        # tag at column 0 only (not indented examples)
                if m:
                    f = parse_tag(m.group(1))
                    if not f.get("title"):
                        return None
                    try:
                        order = int(f.get("order", 100))
                    except ValueError:
                        order = 100
                    kind = f.get("kind", "gui").lower()
                    return App(path=path, title=f["title"], group=f.get("group", "Other"), order=order,
                               kind=kind if kind in KINDS else "gui",
                               needs=[n.strip() for n in f.get("needs", "").replace(";", ",").split(",") if n.strip()],
                               desc=f.get("desc", ""))
    except (OSError, UnicodeDecodeError):
        return None
    return None


def discover(root, exclude=()):
    """All tagged apps under `root` (recursive, skipping caches and tests), sorted by group, then order."""
    root = Path(root)
    apps = []
    for p in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts[:-1]) or p.name in exclude:
            continue
        a = read_app(p)
        if a is not None:
            apps.append(a)
    def group_key(g):
        return (GROUP_ORDER.index(g), "") if g in GROUP_ORDER else (len(GROUP_ORDER), g)
    return sorted(apps, key=lambda a: (group_key(a.group), a.order, a.title))

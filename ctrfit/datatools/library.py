"""Scan library: background-subtracted scans sorted by film and electrochemical condition, for
comparisons (no Qt).

A library is a text file that points at the data (SPEC files and the ROI sums files of the background
subtraction) and says, per scan, which film it was measured on and under what condition:

    CV   the scan ran during a cyclic voltammogram; optional: the SPEC point where the CV starts
         (cv_start), which puts t = 0 there on the 'time from CV start' axis
    CA   chronoamperometry: the potential was held at one value (per scan, in the library's unit)
    OCV  at open circuit (no potential applied)

The data files are only ever opened for reading; the library file is the only file written, and
write_library refuses to write over one of its own data files.

Library file (tab separated; '#' lines are comments; a section starts with a '# [name]' line):

    # [settings]   key <tab> value: roi (default ROI), monitor, transmission, count_time (SPEC
                   columns for the normalisation, blank: not used), unit (of the CA potentials),
                   sys_error (% of I added in quadrature to the counting error, e.g. 5)
    # [sources]    id <tab> what <tab> value [<tab> size <tab> mtime]; what is label, spec or sums
                   (one row per ROI sums file); paths relative to the library file's folder; size
                   and mtime of each file when it was read (a changed file is reported on opening).
                   A source with no spec row is a placeholder (an assignment file written before the
                   data were processed): the first SPEC file added to the library fills it.
    # [scans]      source scan film mode potential cv_start roi use | kind H K L L_min L_max points
                   command | notes; the '# source ...' header line names the columns (files of an
                   older layout still read). roi blank: the default ROI; the columns from 'kind' to
                   'command' describe the data and are worked out again from the data on opening
                   (kept from the file for a scan the data do not have yet)

Kinds, from the SPEC scan at the images that have ROI sums: rod (L moves, H and K fixed), stationary
(H K L fixed: a time scan during a CV or a CA hold; H K L from the columns or from '#Q'), other
(H or K moves: rocking or in-plane scans), no SPEC (the SPEC file does not have the scan), missing
(the ROI sums do not have the scan, yet or any more; its entry and assignments are kept).

apply_assignments() copies film, measurement, potential, CV start, ROI, use and notes from another
library (e.g. an assignment file) onto the scans of one source, by scan number.

Two ROI sums files of one SPEC file can both have a scan (the background subtraction run twice):
each keeps its own ROIs, and a ROI name that an earlier file already has for that scan becomes
'<name> (<file name>)'.

Comparisons: select() picks the scans of one kind by measurement, film, potential, rod and L;
traces() gives their intensity (the scan's ROI, divided by monitor / transmission / count time as
set) against L (rod scans) or the time from the scan start, the SPEC point or a SPEC column
(stationary scans); group_label() names the group a trace is coloured by.
"""
import copy
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from ..echem.external import parse_roi_sums
from .images import read_roi_positions, roi_position
from .specfile import parse_spec_file

FORMAT = "# CTR scan library (ctrfit data tools, format 1)"
MODES = ("", "CV", "CA", "OCV")           # "" = not assigned yet
HOLDS = ("CA", "OCV")                     # held for the whole scan: compared by potential
OCV = "OCV"                               # the 'potential' of an OCV scan in select() and the lists
KINDS = ("rod", "stationary", "other", "no SPEC", "missing")
TOL = 0.01                                # H, K, L spans at or below this: the index is fixed
E_TOL = 5e-4                              # potentials closer than this are the same (V)
STATIONARY_COMMANDS = ("loopscan", "timescan", "tscan", "loop", "ctscan")
TIME_COLUMNS = ("epoch", "time", "elapsed", "elapsed_time", "time_s", "t")
X_TIME, X_POINT, X_L = "time from scan start (s)", "SPEC point", "L"
X_CV = "time from CV start (s)"           # t = 0 at the entry's cv_start point
NORMS = ("as measured", "÷ max", "÷ first point")
COLOUR_BY = ("film", "potential", "film and potential", "scan")
SETTINGS = ("roi", "monitor", "transmission", "count_time", "unit", "sys_error")
DEFAULT_SETTINGS = {"roi": "", "monitor": "", "transmission": "", "count_time": "", "unit": "V vs RHE",
                    "sys_error": "0"}
SCAN_COLUMNS = ("source", "scan", "film", "mode", "potential", "cv_start", "roi", "use", "kind", "H", "K", "L",
                "L_min", "L_max", "points", "command", "notes")
SCAN_COLUMNS_1 = ("source", "scan", "film", "mode", "potential", "roi", "use", "kind", "H", "K", "L", "L_min",
                  "L_max", "points", "command", "notes")      # the first layout (no header needed to read it)
KEEP = object()                           # assign(): leave this field as it is


@dataclass
class Source:
    id: str                               # S1, S2, ...
    spec: str                             # absolute paths
    sums: list = field(default_factory=list)
    label: str = ""
    stamps: dict = field(default_factory=dict)   # path -> (size, mtime) when last read


@dataclass
class Entry:
    source: str
    scan: int
    film: str = ""
    mode: str = ""                        # "", "CV", "CA" or "OCV"
    potential: float = None               # CA: the held potential (library unit)
    roi: str = ""                         # "": the library's default ROI
    use: bool = True
    notes: str = ""
    # what the data say (worked out on loading)
    kind: str = ""
    H: float = math.nan
    K: float = math.nan
    L: float = math.nan
    L_min: float = math.nan
    L_max: float = math.nan
    points: int = 0
    command: str = ""
    rois: list = field(default_factory=list)     # the ROI names this scan has
    cv_start: int = None                  # CV: SPEC point (0-based) where the CV starts

    @property
    def key(self):
        return (self.source, self.scan)

    @property
    def rod(self):
        """(h, k) rounded to integers, or None if unknown."""
        if not (np.isfinite(self.H) and np.isfinite(self.K)):
            return None
        return (int(round(self.H)), int(round(self.K)))

    @property
    def assigned(self):
        return bool(self.film) and self.mode in MODES[1:]

    def condition(self, unit=""):
        """'CV', 'OCV', 'CA 1.2 V vs RHE', 'CA (no potential)' or 'no condition'."""
        if self.mode in ("CV", "OCV"):
            return self.mode
        if self.mode == "CA":
            return f"CA {format_potential(self.potential, unit)}" if self.potential is not None else "CA (no potential)"
        return "no condition"


@dataclass
class ScanData:
    spec: object = None                   # SpecScan, or None if the SPEC file does not have the scan
    rois: dict = field(default_factory=dict)     # name -> (images, I, sigma or None, path of the sums file)


@dataclass
class Trace:
    entry: Entry
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray                     # None if the ROI has no '_err' column
    roi: str
    xlabel: str
    group: str = ""
    notes: list = field(default_factory=list)


def format_potential(E, unit=""):
    if E is None or not np.isfinite(E):
        return "?"
    return f"{float(E):.4g}" + (f" {unit}" if unit else "")


def natural_key(text):
    """Sort '2 nm' before '10 nm'."""
    return [(0, float(p), "") if re.fullmatch(r"\d+(\.\d+)?", p) else (1, 0.0, p.lower())
            for p in re.split(r"(\d+(?:\.\d+)?)", str(text)) if p != ""]


def _clean(text):
    """One field of the tab-separated file: no tabs or line breaks."""
    return re.sub(r"[\t\r\n]+", " ", str(text or "")).strip()


def _stamp(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (int(st.st_size), round(float(st.st_mtime), 3))


def _column(scan, name):
    """A SPEC column by name (case-insensitive), as float, or None."""
    if scan is None or not name:
        return None
    for c in scan.columns:
        if c == name or c.lower() == name.lower():
            return scan.data[c].astype(float)
    return None


# ------------------------------------------------------------------ reading the data (read only)
def read_sums(paths):
    """{scan: {roi name: (images, I, sigma or None, path)}} of ROI sums files, in order. A ROI name an
    earlier file already gave a scan becomes '<name> (<file stem>)'."""
    out = {}
    for path in paths:
        with open(path, "r", errors="replace") as f:
            sums = parse_roi_sums(f.read())
        if sums is None:
            raise ValueError(f"{os.path.basename(path)} is not a ROI sums file (header: scan, image, ROIs)")
        stem = os.path.splitext(os.path.basename(path))[0]
        for sn, cols in sums.items():
            have = out.setdefault(sn, {})
            img = cols["image"].astype(int)
            for c in cols:
                if c.lower() in ("scan", "image") or c.endswith("_err"):
                    continue
                name = c if c not in have else f"{c} ({stem})"
                err = cols.get(f"{c}_err")
                have[name] = (img, cols[c].astype(float), None if err is None else err.astype(float), path)
    return out


def describe(scan, images=None):
    """(kind, H, K, L, L_min, L_max) of a SPEC scan at the given images (default: every point)."""
    if scan is None:
        return ("no SPEC",) + (math.nan,) * 5
    n = scan.npoints
    idx = np.arange(n) if images is None else np.asarray([i for i in images if 0 <= i < n], int)
    hkl = [_column(scan, c) for c in "HKL"]
    if all(x is not None for x in hkl) and idx.size:
        H, K, L = (x[idx] for x in hkl)
        span = [float(np.nanmax(x) - np.nanmin(x)) for x in (H, K, L)]
        med = [float(np.nanmedian(x)) for x in (H, K, L)]
        if max(span) <= TOL:
            kind = "stationary"
        elif span[0] <= TOL and span[1] <= TOL:
            kind = "rod"
        else:
            kind = "other"
        return (kind, med[0], med[1], med[2], float(np.nanmin(L)), float(np.nanmax(L)))
    words = (scan.scan_cmd or "").split()
    kind = "stationary" if words and words[0] in STATIONARY_COMMANDS else "other"
    h, k, l = scan.hkl if scan.hkl else (math.nan,) * 3
    return (kind, h, k, l, l, l)


class Library:
    def __init__(self):
        self.sources = []
        self.entries = []
        self.settings = dict(DEFAULT_SETTINGS)
        self.path = None
        self.notes = []                   # messages of the last load
        self._data = {}                   # (source, scan) -> ScanData
        self._spec = {}                   # source id -> {scan: SpecScan}
        self._positions = {}              # sums path -> ROI positions section

    # -------------------------------------------------------------- sources
    def source(self, sid):
        return next((s for s in self.sources if s.id == sid), None)

    def _next_id(self):
        used = {s.id for s in self.sources}
        i = 1
        while f"S{i}" in used:
            i += 1
        return f"S{i}"

    def add_source(self, spec, sums, label=""):
        """Add a SPEC file and its ROI sums files (or more ROI sums to a SPEC file already in the
        library); reads them and adds every scan with ROI sums. Returns the Source."""
        spec = os.path.abspath(str(spec))
        sums = [os.path.abspath(str(p)) for p in sums]
        if not os.path.isfile(spec):
            raise ValueError(f"no SPEC file {spec}")
        if not sums:
            raise ValueError("pick at least one ROI sums file")
        for p in sums:
            if not os.path.isfile(p):
                raise ValueError(f"no ROI sums file {p}")
        read_sums(sums)                   # every file is a ROI sums file, before anything is changed
        src = next((s for s in self.sources if s.spec and os.path.normcase(s.spec) == os.path.normcase(spec)),
                   None)
        if src is None:
            src = next((s for s in self.sources if not s.spec), None)        # a placeholder: fill it
            if src is not None:
                src.spec = spec
                src.label = label or os.path.splitext(os.path.basename(spec))[0]
        if src is None:
            src = Source(self._next_id(), spec, [], label or os.path.splitext(os.path.basename(spec))[0])
            self.sources.append(src)
        elif label:
            src.label = label
        for p in sums:
            if all(os.path.normcase(p) != os.path.normcase(q) for q in src.sums):
                src.sums.append(p)
        self.load([src.id])
        return src

    def remove_source(self, sid):
        """Take a source and its scans out of the library (the files stay as they are)."""
        self.sources = [s for s in self.sources if s.id != sid]
        self.entries = [e for e in self.entries if e.source != sid]       # the read data stay cached (undo)

    def changed_files(self):
        """Data files whose size or time differ from when the library last read them."""
        out = []
        for s in self.sources:
            for p in [s.spec] + list(s.sums):
                old = s.stamps.get(p)
                if old is not None and _stamp(p) != tuple(old):
                    out.append(p)
        return out

    def load(self, ids=None):
        """Read the data files of the sources (all, or the ids given): a scan with ROI sums that has
        no entry yet gets one; entries keep their assignments. Returns notes (also in self.notes)."""
        notes = []
        changed = set(self.changed_files())
        for s in self.sources:
            if ids is not None and s.id not in ids:
                continue
            if not s.spec:
                notes.append(f"{s.id} {s.label}: no SPEC file yet (Add SPEC + ROI sums fills it)")
                continue
            for p in [s.spec] + list(s.sums):
                if p in changed:
                    notes.append(f"{os.path.basename(p)} changed since it was last read")
            missing = [p for p in [s.spec] + list(s.sums) if not os.path.isfile(p)]
            if missing:
                notes.append(f"{s.id}: file(s) not found: {', '.join(missing)}")
            spec = parse_spec_file(s.spec) if os.path.isfile(s.spec) else {}
            sums = read_sums([p for p in s.sums if os.path.isfile(p)])
            for p in s.sums:
                if os.path.isfile(p):
                    try:
                        self._positions[p] = read_roi_positions(p)
                    except OSError:
                        self._positions[p] = {}
            s.stamps = {p: _stamp(p) for p in [s.spec] + list(s.sums) if _stamp(p) is not None}
            self._spec[s.id] = spec
            have = {e.scan: e for e in self.entries if e.source == s.id}
            for sn in sorted(set(sums) | set(have)):
                e = have.get(sn)
                if e is None:
                    e = Entry(s.id, sn)
                    self.entries.append(e)
                rois = sums.get(sn, {})
                scan = spec.get(sn)
                self._data[e.key] = ScanData(scan, rois)
                images = sorted({int(i) for v in rois.values() for i in v[0]})
                if sn not in sums:
                    if scan is not None:                 # where the scan is, from the SPEC file
                        e.H, e.K, e.L, e.L_min, e.L_max = describe(scan)[1:]
                    e.kind = "missing"                   # else H K L as the library file had them
                    notes.append(f"{s.id} scan {sn}: no ROI sums (yet, or any more; its assignments are kept)")
                else:
                    e.kind, e.H, e.K, e.L, e.L_min, e.L_max = describe(scan, images)
                    if scan is None:
                        notes.append(f"{s.id} scan {sn}: not in the SPEC file {os.path.basename(s.spec)}")
                e.points = len(images)
                e.command = scan.scan_cmd if scan is not None else e.command
                e.rois = list(rois)
                if e.roi and e.roi not in e.rois and sn in sums:
                    notes.append(f"{s.id} scan {sn}: ROI '{e.roi}' not in its ROI sums any more")
        order = {s.id: i for i, s in enumerate(self.sources)}
        self.entries.sort(key=lambda e: (order.get(e.source, 99), e.scan))
        self.notes = notes
        return notes

    # -------------------------------------------------------------- entries
    def entry(self, source, scan):
        return next((e for e in self.entries if e.source == source and e.scan == int(scan)), None)

    def data(self, entry):
        return self._data.get(entry.key, ScanData())

    def spec_columns(self):
        out = []
        for scans in self._spec.values():
            for sc in scans.values():
                out += [c for c in sc.columns if c not in out]
        return out

    def roi_names(self):
        out = []
        for e in self.entries:
            out += [r for r in e.rois if r not in out]
        return out

    def films(self):
        return sorted({e.film for e in self.entries if e.film}, key=natural_key)

    def potentials(self, entries=None):
        """The CA potentials (sorted, values within E_TOL counted once)."""
        vals = sorted(e.potential for e in (self.entries if entries is None else entries)
                      if e.mode == "CA" and e.potential is not None)
        out = []
        for v in vals:
            if not out or abs(v - out[-1]) > E_TOL:
                out.append(v)
        return out

    def holds(self, entries=None):
        """What scans were held at, for the potential list: 'OCV' (if any) then the CA potentials."""
        es = self.entries if entries is None else entries
        return ([OCV] if any(e.mode == OCV for e in es) else []) + self.potentials(es)

    def effective_roi(self, entry):
        """The ROI a scan is shown with: its own pick, else the default, else its first ROI with an
        error column, else its first ROI (None if it has none)."""
        rois = self.data(entry).rois
        for pick in (entry.roi, self.settings.get("roi")):
            if pick and pick in rois:
                return pick
        with_err = [r for r, v in rois.items() if v[2] is not None]
        return (with_err or list(rois) or [None])[0]

    def roi_description(self, entry, name):
        """'signal ROI 40 x 30 px at (512, 300)' from the ROI positions section, or ''."""
        d = self.data(entry).rois.get(name)
        if d is None:
            return ""
        base = re.sub(r" \(.*\)$", "", name)
        pos = self._positions.get(d[3], {})
        p = roi_position(pos, entry.scan, "all", base)
        if p is None:
            moves = any(k[0] == entry.scan and k[2] == base for k in pos)
            where = "moves from image to image" if moves else "no position recorded"
            return f"{name}: {where} ({os.path.basename(d[3])})"
        kind, x, y, w, h = p
        return f"{name}: {kind} ROI {w} x {h} px at ({x}, {y}) in {os.path.basename(d[3])}"

    def assign(self, entries, film=KEEP, mode=KEEP, potential=KEEP, roi=KEEP, use=KEEP, notes=KEEP,
               cv_start=KEEP):
        """Set fields of several entries. mode CA keeps a potential that is given (or the entry's);
        another mode clears it; a potential alone makes the scan a CA. cv_start (SPEC point) is kept
        by CV only; a start point alone makes the scan a CV."""
        if mode is not KEEP and mode not in MODES:
            raise ValueError(f"measurement must be one of CV, CA, OCV or blank, not '{mode}'")
        if cv_start is not KEEP and cv_start is not None and int(cv_start) < 0:
            raise ValueError("the CV start point is a SPEC point number, 0 or more")
        for e in entries:
            if film is not KEEP:
                e.film = _clean(film)
            if mode is not KEEP:
                e.mode = mode
                if mode != "CA":
                    e.potential = None
                if mode != "CV":
                    e.cv_start = None
            if potential is not KEEP:
                e.potential = None if potential is None else float(potential)
                if potential is not None and mode is KEEP:
                    e.mode = "CA"
                    e.cv_start = None
            if cv_start is not KEEP:
                e.cv_start = None if cv_start is None else int(cv_start)
                if cv_start is not None and mode is KEEP:
                    e.mode = "CV"
                    e.potential = None
            if roi is not KEEP:
                e.roi = _clean(roi)
            if use is not KEEP:
                e.use = bool(use)
            if notes is not KEEP:
                e.notes = _clean(notes)

    def snapshot(self):
        """State for undo: sources, entries and settings."""
        return copy.deepcopy((self.sources, self.entries, self.settings))

    def restore(self, state):
        self.sources, self.entries, self.settings = copy.deepcopy(state)


# ------------------------------------------------------------------ file
def _rel(path, base):
    try:
        return os.path.relpath(path, base)
    except ValueError:                    # another drive (Windows)
        return path


def write_library(lib, path=None, remember=True):
    """Write the library file (the data files are never touched). Returns the path. remember=False
    leaves lib.path as it is (a draft copy)."""
    path = os.path.abspath(str(path or lib.path or ""))
    if not path or os.path.isdir(path):
        raise ValueError("pick a file name for the library")
    for s in lib.sources:
        for p in [s.spec] + list(s.sums):
            if os.path.normcase(os.path.abspath(p)) == os.path.normcase(path):
                raise ValueError(f"{os.path.basename(path)} is one of the library's data files; the data are "
                                 "read only, save the library under another name")
    base = os.path.dirname(path)
    lines = [FORMAT,
             "# The data files are only read, never changed. Paths are relative to this file's folder.",
             "# Tab separated. mode: CV, CA, OCV or blank; potential: the CA hold, in the unit below; cv_start:",
             "# the SPEC point where a CV starts; roi blank: the default ROI; use: 1 or 0. The columns kind to",
             "# command describe the data and are worked out again from the data when the library is opened.",
             "", "# [settings]"]
    lines += [f"{k}\t{_clean(lib.settings.get(k, ''))}" for k in SETTINGS]
    lines += ["", "# [sources]", "# id\twhat\tvalue\tsize\tmtime"]
    for s in lib.sources:
        lines.append(f"{s.id}\tlabel\t{_clean(s.label)}")
        for what, p in ([("spec", s.spec)] if s.spec else []) + [("sums", q) for q in s.sums]:
            st = s.stamps.get(p) or _stamp(p)
            extra = f"\t{st[0]}\t{st[1]:.3f}" if st else ""
            lines.append(f"{s.id}\t{what}\t{_rel(p, base)}{extra}")
    lines += ["", "# [scans]", "# " + "\t".join(SCAN_COLUMNS)]
    fmt = lambda v, d=4: "" if v is None or not np.isfinite(v) else f"{v:.{d}g}"
    for e in lib.entries:
        vals = (e.source, str(e.scan), _clean(e.film), e.mode, "" if e.potential is None else f"{e.potential:.6g}",
                "" if e.cv_start is None else str(e.cv_start), _clean(e.roi), "1" if e.use else "0", e.kind, fmt(e.H), fmt(e.K), fmt(e.L), fmt(e.L_min),
                fmt(e.L_max), str(e.points), _clean(e.command), _clean(e.notes))
        lines.append("\t".join(vals))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)                 # a crash while writing never leaves half a library
    if remember:
        lib.path = path
    return path


def read_library(path, load=True):
    """Library from its file; load=True also reads the data files (lib.notes says what changed)."""
    path = os.path.abspath(str(path))
    base = os.path.dirname(path)
    with open(path, "r", errors="replace") as f:
        text = f.read()
    if not text.startswith(FORMAT[:22]):
        raise ValueError(f"{os.path.basename(path)} is not a scan library file")
    lib = Library()
    lib.path = path
    section = None
    columns = SCAN_COLUMNS_1
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        m = re.match(r"#\s*\[(\w+)\]", line)
        if m:
            section = m.group(1)
            continue
        head = line.lstrip("#").strip().split("\t")
        if section == "scans" and line.startswith("#") and head[:2] == ["source", "scan"]:
            columns = tuple(h.strip() for h in head)             # the file names its columns
            continue
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if section == "settings" and parts[0] in SETTINGS:
            lib.settings[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
        elif section == "sources" and len(parts) >= 3:
            sid, what, value = parts[0], parts[1], parts[2]
            src = lib.source(sid)
            if src is None:
                src = Source(sid, "")
                lib.sources.append(src)
            if what == "label":
                src.label = value
            elif what in ("spec", "sums") and value.strip():
                p = os.path.normpath(value if os.path.isabs(value) else os.path.join(base, value))
                if what == "spec":
                    src.spec = p
                else:
                    src.sums.append(p)
                if len(parts) >= 5:
                    try:
                        src.stamps[p] = (int(parts[3]), float(parts[4]))
                    except ValueError:
                        pass
        elif section == "scans" and len(parts) >= 7:
            row = dict.fromkeys(SCAN_COLUMNS, "")
            row.update(zip(columns, parts))
            try:
                scan = int(row["scan"])
            except ValueError:
                continue
            mode = row["mode"].strip().upper()
            pot = row["potential"].strip()
            try:
                E = float(pot) if pot else None
            except ValueError:
                raise ValueError(f"scan {row['source']} {scan}: potential '{pot}' is not a number")
            e = Entry(row["source"], scan, row["film"].strip(), mode if mode in MODES else "", E,
                      row["roi"].strip(), row["use"].strip() not in ("0", "no", "false", "False"),
                      row["notes"].strip())
            if E is not None and e.mode == "":
                e.mode = "CA"
            start = row["cv_start"].strip()
            try:
                e.cv_start = int(float(start)) if start and e.mode in ("", "CV") else None
            except ValueError:
                raise ValueError(f"scan {row['source']} {scan}: CV start '{start}' is not a point number")
            if e.cv_start is not None:
                e.mode = "CV"
            e.command = row["command"].strip()
            e.kind = row["kind"].strip()                          # until the data say otherwise
            for name in ("H", "K", "L", "L_min", "L_max"):
                try:
                    setattr(e, name, float(row[name]) if row[name].strip() else math.nan)
                except ValueError:
                    pass
            if e.kind == "rod" and not np.isfinite(e.L) and np.isfinite(e.L_min) and np.isfinite(e.L_max):
                e.L = (e.L_min + e.L_max) / 2
            lib.entries.append(e)
    if load:
        lib.load()
    return lib


FIELDS = ("film", "mode", "potential", "cv_start", "roi", "use", "notes")


def apply_assignments(lib, other, sid, from_source=None):
    """Copy the assignments (film, measurement, potential, CV start, ROI, use, notes) of another
    library's scans onto the scans of source `sid` of lib, by scan number. from_source: the other
    library's source to take them from (None: every source). A scan the data do not have yet gets an
    entry (kind missing), so the assignment is there once it is processed. Returns (matched, added)."""
    if lib.source(sid) is None:
        raise ValueError(f"no source {sid} in the library")
    if isinstance(other, (str, os.PathLike)):
        other = read_library(other, load=False)
    matched = added = 0
    for o in other.entries:
        if from_source is not None and o.source != from_source:
            continue
        e = lib.entry(sid, o.scan)
        if e is None:
            e = Entry(sid, o.scan, kind="missing", H=o.H, K=o.K, L=o.L, L_min=o.L_min, L_max=o.L_max,
                      command=o.command)
            lib.entries.append(e)
            added += 1
        else:
            matched += 1
        for name in FIELDS:
            setattr(e, name, copy.copy(getattr(o, name)))
    order = {s.id: i for i, s in enumerate(lib.sources)}
    lib.entries.sort(key=lambda e: (order.get(e.source, 99), e.scan))
    return matched, added


# ------------------------------------------------------------------ comparisons
def select(lib, kind, modes=None, films=None, potential=None, rod=None, L=None, tol=TOL, used_only=True):
    """Entries of one kind ('rod' or 'stationary'). modes: a set of 'CV', 'CA', 'OCV', '' (None: any);
    films: a set of film names, '' for no film (None: any); potential: a number (CA scans held at it)
    or 'OCV' (OCV scans); CV and unassigned scans pass; rod: (h, k); L: stationary scans within tol."""
    out = []
    for e in lib.entries:
        if e.kind != kind or (used_only and not e.use):
            continue
        if modes is not None and e.mode not in modes:
            continue
        if films is not None and e.film not in films:
            continue
        if potential is not None and e.mode in HOLDS:
            if potential == OCV:
                if e.mode != OCV:
                    continue
            elif e.mode != "CA" or e.potential is None or abs(e.potential - float(potential)) > E_TOL:
                continue
        if rod is not None and e.rod != tuple(rod):
            continue
        if L is not None and not (np.isfinite(e.L) and abs(e.L - L) <= tol):
            continue
        out.append(e)
    return out


def rods(entries):
    return sorted({e.rod for e in entries if e.rod is not None})


def positions(entries, tol=TOL):
    """[(h, k, L)] of stationary entries: L values of one rod within tol are one position."""
    by = {}
    for e in entries:
        if e.rod is not None and np.isfinite(e.L):
            by.setdefault(e.rod, []).append(e.L)
    out = []
    for (h, k), Ls in sorted(by.items()):
        Ls.sort()
        groups = [[Ls[0]]]
        for v in Ls[1:]:
            (groups[-1].append(v) if v - groups[-1][0] <= tol else groups.append([v]))
        out += [(h, k, float(np.mean(g))) for g in groups]
    return out


def group_label(lib, e, colour_by):
    unit = lib.settings.get("unit", "")
    film = e.film or "(no film)"
    if colour_by == "film":
        return film
    if colour_by == "potential":
        return e.condition(unit)
    if colour_by == "film and potential":
        return f"{film}, {e.condition(unit)}"
    src = lib.source(e.source)
    many = len(lib.sources) > 1
    return (f"{src.label if src and src.label else e.source} " if many else "") + f"#{e.scan}"


def group_key(lib, e, colour_by):
    """Sort key of a trace's group: films in natural order, CV before CA, potentials by value."""
    cond = {"CV": (0, 0.0), OCV: (1, -math.inf)}.get(e.mode) or \
        ((1, e.potential if e.potential is not None else math.inf) if e.mode == "CA" else (2, 0.0))
    film = natural_key(e.film) if e.film else [(2, 0.0, "")]
    if colour_by == "potential":
        return (cond, film)
    if colour_by == "scan":
        return ([s.id for s in lib.sources].index(e.source) if lib.source(e.source) else 99, e.scan)
    return (film, cond)


def time_axis(scan, count_time=None):
    """(seconds from the scan's first point at every point, how) from a time column (Epoch, Time,
    ...), else the summed count times, else None."""
    for name in TIME_COLUMNS:
        t = _column(scan, name)
        if t is not None and np.isfinite(t).any():
            return t - t[np.isfinite(t)][0], name
    ct = _column(scan, count_time)
    if ct is not None:
        return np.concatenate([[0.0], np.cumsum(ct)[:-1]]), f"summed {count_time}"
    return None, None


def trace(lib, e, x=X_TIME, mon_ref=None):
    """Trace of one entry: its ROI against L (rod scans) or x (stationary: X_TIME, X_CV, X_POINT or a
    SPEC column), divided by the normalisation columns of the library's settings; sigma is the
    counting error with the systematic error (sys_error, % of I) added in quadrature."""
    d = lib.data(e)
    roi = lib.effective_roi(e)
    if roi is None:
        raise ValueError(f"{e.source} scan {e.scan}: no ROI sums")
    img, I, sig, _ = d.rois[roi]
    notes = []
    scan = d.spec
    n = scan.npoints if scan is not None else int(img.max()) + 1
    ok = (img >= 0) & (img < n)
    if not ok.all():
        notes.append(f"{int((~ok).sum())} image(s) beyond the scan's {n} points left out")
    img, I = img[ok], I[ok]
    sig = sig[ok] if sig is not None else None
    f = np.ones(I.size)
    st = lib.settings
    for key in ("monitor", "transmission", "count_time"):
        name = st.get(key)
        if not name:
            continue
        col = _column(scan, name)
        if col is None:
            notes.append(f"no column '{name}': not divided by it")
            continue
        v = col[img]
        bad = ~(v > 0)
        if bad.any():
            notes.append(f"{int(bad.sum())} point(s) with {name} <= 0 left out")
            v = np.where(bad, np.nan, v)
        if key == "monitor":
            f = f * (mon_ref if mon_ref else float(np.nanmean(v))) / v
        else:
            f = f / v
    y = I * f
    s = sig * f if sig is not None else None
    try:
        pct = float(st.get("sys_error") or 0)
    except ValueError:
        pct = 0.0
    if pct > 0:
        floor = np.abs(y) * pct / 100
        s = np.sqrt(s ** 2 + floor ** 2) if s is not None else floor
    if e.kind == "rod":
        Lc = _column(scan, "L")
        if Lc is None:
            raise ValueError(f"{e.source} scan {e.scan}: no L column in the SPEC file")
        xs, xlabel = Lc[img], "L (r.l.u.)"
    elif x == X_POINT or scan is None:
        xs, xlabel = img.astype(float), "SPEC point"
    elif x in (X_TIME, X_CV):
        t, how = time_axis(scan, st.get("count_time"))
        if t is None:
            xs, xlabel = img.astype(float), "SPEC point"
            notes.append("no time column: plotted against the SPEC point")
        elif x == X_CV:
            xs, xlabel = t[img], X_CV
            if e.cv_start is not None and 0 <= e.cv_start < t.size and np.isfinite(t[e.cv_start]):
                xs = xs - t[e.cv_start]
            else:
                notes.append("no CV start point: t = 0 at the scan start")
        else:
            xs, xlabel = t[img], "time from scan start (s)"
    else:
        col = _column(scan, x)
        if col is None:
            raise ValueError(f"{e.source} scan {e.scan}: no SPEC column '{x}'")
        xs, xlabel = col[img], x
    order = np.argsort(xs, kind="stable") if e.kind == "rod" else np.arange(xs.size)
    keep = np.isfinite(y[order]) & np.isfinite(xs[order])
    order = order[keep]
    return Trace(e, xs[order], y[order], s[order] if s is not None else None, roi, xlabel, notes=notes)


def monitor_reference(lib, entries):
    """Mean monitor over the entries (so normalised intensities stay in counts), or None."""
    name = lib.settings.get("monitor")
    if not name:
        return None
    vals = []
    for e in entries:
        col = _column(lib.data(e).spec, name)
        if col is not None and np.any(col > 0):
            vals.append(float(np.mean(col[col > 0])))
    return float(np.mean(vals)) if vals else None


def scaled(t, norm):
    """(y, sigma) of a trace divided by its max or its first point (NORMS)."""
    if norm == NORMS[0] or t.y.size == 0:
        return t.y, t.sigma
    ref = float(np.nanmax(np.abs(t.y))) if norm == NORMS[1] else float(t.y[0])
    if not np.isfinite(ref) or ref == 0:
        return t.y, t.sigma
    return t.y / ref, (t.sigma / abs(ref) if t.sigma is not None else None)


def traces(lib, entries, x=X_TIME, colour_by="film"):
    """Traces of the entries in group order (one shared monitor reference), and notes."""
    ref = monitor_reference(lib, entries)
    out, notes = [], []
    for e in sorted(entries, key=lambda e: (group_key(lib, e, colour_by), e.source, e.scan)):
        try:
            t = trace(lib, e, x, ref)
        except ValueError as ex:
            notes.append(str(ex))
            continue
        t.group = group_label(lib, e, colour_by)
        out.append(t)
        notes += [f"{e.source} scan {e.scan}: {n}" for n in t.notes]
    return out, notes


def write_traces(path, trs, title=""):
    """The plotted traces as text: one '# Trace: <source>_<scan>' block each (the format the in-situ CV
    tool and the plotter's exports use), with x, I and its error. Returns the path."""
    lines = [f"# Scan library traces{': ' + title if title else ''}"]
    for t in trs:
        e = t.entry
        lines += ["", f"# Trace: {e.source}_{e.scan}  {t.group}; film {e.film or '-'}; {e.condition()}; ROI {t.roi}",
                  "\t".join([t.xlabel.split(" (")[0].replace(" ", "_"), "I", "I_err"])]
        sig = t.sigma if t.sigma is not None else np.full(t.y.size, np.nan)
        lines += [f"{a:.8g}\t{b:.8g}\t{c:.6g}" for a, b, c in zip(t.x, t.y, sig)]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path

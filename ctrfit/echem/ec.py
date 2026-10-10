"""Potentiostat files: EC-Lab .mpr (binary, read with the optional `galvani` package), EC-Lab .mpt
text exports, and plain CSV / whitespace tables with a time and a potential column.

Every reader returns an ECData: time (s from the start of the file), potential (V), current
(mA, if present), the EC-Lab cycle number (if present) and the local start time of the file.
"""
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .spec import to_seconds

TIME_NAMES = ("time/s", "time", "t/s", "time (s)", "elapsed time (s)")
E_NAMES = ("Ewe/V", "<Ewe>/V", "Ewe-Ece/V", "<Ewe/V>", "E/V", "Ewe", "potential/V", "E (V)", "potential (V)",
           "control/V")
I_NAMES = ("I/mA", "<I>/mA", "control/mA", "I (mA)", "current/mA")
CYCLE_NAMES = ("cycle number", "cycle", "Cycle")


@dataclass
class ECData:
    time: np.ndarray                 # s from the file start
    potential: np.ndarray            # V
    current: np.ndarray = None       # mA
    cycle: np.ndarray = None         # EC-Lab cycle number
    start: datetime = None           # local start time of the acquisition
    columns: dict = field(default_factory=dict)
    source: str = ""
    notes: list = field(default_factory=list)

    @property
    def start_seconds(self):
        return to_seconds(self.start) if self.start is not None else None

    def absolute_time(self):
        if self.start is None:
            raise ValueError("this potentiostat file has no start time; use the manual or event alignment")
        return self.start_seconds + self.time


def _pick(names, candidates, what, required=True):
    low = {n.lower(): n for n in names}
    for c in candidates:
        if c in names:
            return c
        if c.lower() in low:
            return low[c.lower()]
    if required:
        raise KeyError(f"no {what} column (looked for {', '.join(candidates)}; found {', '.join(names)})")
    return None


def _from_columns(cols, start, source, time_col=None, e_col=None):
    names = list(cols)
    tc = time_col or _pick(names, TIME_NAMES, "time")
    ec = e_col or _pick(names, E_NAMES, "potential")
    ic = _pick(names, I_NAMES, "current", required=False)
    cc = _pick(names, CYCLE_NAMES, "cycle", required=False)
    t = np.asarray(cols[tc], float)
    order = np.argsort(t, kind="stable")
    get = lambda c: None if c is None else np.asarray(cols[c], float)[order]  # noqa: E731
    d = ECData(t[order], get(ec), get(ic), get(cc), start, {k: np.asarray(v)[order] for k, v in cols.items()}, source)
    if ec == "control/V":
        d.notes.append("no measured potential column: using the control potential")
    return d


# ----------------------------------------------------------------------------------------------
# .mpr (binary) through galvani
# ----------------------------------------------------------------------------------------------
def read_mpr(path):
    try:
        from galvani import BioLogic
    except ImportError:
        raise RuntimeError("reading .mpr files needs the 'galvani' package (pip install galvani; if the build "
                           "fails, pip install --use-pep517 galvani), or export the file from EC-Lab as text "
                           "(.mpt) and load that") from None
    mpr = BioLogic.MPRfile(str(path))
    data = mpr.data
    cols = {name: np.asarray(data[name]) for name in data.dtype.names}
    start = getattr(mpr, "timestamp", None)
    out = _from_columns(cols, start, f"{Path(path).name} (EC-Lab .mpr)")
    if start is None:
        sd = getattr(mpr, "startdate", None)
        out.notes.append("the .mpr has no log module, so no start time of day" +
                         (f" (date {sd})" if sd else "") + ": use the manual or event alignment, or enter it")
    return out


# ----------------------------------------------------------------------------------------------
# .mpt (EC-Lab text export) and plain tables
# ----------------------------------------------------------------------------------------------
DATE_FORMATS = ("%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S")


def parse_ec_date(text, dayfirst=False):
    text = text.strip()
    fmts = DATE_FORMATS if not dayfirst else tuple(f for f in DATE_FORMATS if f.startswith("%d")) + DATE_FORMATS
    for fmt in fmts:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"cannot read the EC-Lab date '{text}' (e.g. 03/07/2024 14:23:11.123)")


def _numbers(lines, ncol):
    rows = []
    for ln in lines:
        parts = ln.rstrip("\n").split("\t") if "\t" in ln else ln.split()
        if len(parts) < ncol:
            continue
        try:
            rows.append([float(p.replace(",", ".")) for p in parts[:ncol]])
        except ValueError:
            continue
    return np.array(rows, float).reshape(-1, ncol)


def read_mpt(path=None, text=None, dayfirst=False):
    text = text if text is not None else Path(path).read_text(encoding="latin-1")
    lines = text.splitlines()
    n_header = None
    start = None
    for ln in lines[:200]:
        m = re.match(r"\s*Nb header lines\s*:\s*(\d+)", ln)
        if m:
            n_header = int(m.group(1))
        m = re.match(r"\s*Acquisition started on\s*:\s*(.+)$", ln)
        if m:
            start = parse_ec_date(m.group(1), dayfirst)
    if n_header is None:                       # plain table: first line with a time and potential name
        return read_table(text=text, source=Path(path).name if path else "table")
    header = lines[n_header - 1].rstrip("\n").split("\t")
    header = [h.strip() for h in header if h.strip()] if "\t" in lines[n_header - 1] else lines[n_header - 1].split()
    arr = _numbers(lines[n_header:], len(header))
    cols = {h: arr[:, i] for i, h in enumerate(header)}
    out = _from_columns(cols, start, f"{Path(path).name if path else 'text'} (EC-Lab .mpt)")
    if start is None:
        out.notes.append("no 'Acquisition started on' line: use the manual or event alignment, or enter the start")
    return out


def read_table(path=None, text=None, source="table", time_col=None, e_col=None, start=None):
    """CSV or whitespace table with a header line naming a time and a potential column."""
    text = text if text is not None else Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError("empty potentiostat file")
    delim = "," if lines[0].count(",") >= 1 and "\t" not in lines[0] else ("\t" if "\t" in lines[0] else None)
    header = [h.strip() for h in (lines[0].split(delim) if delim else lines[0].split())]
    body = "\n".join(lines[1:])
    arr = np.genfromtxt(io.StringIO(body), delimiter=delim, dtype=float)
    arr = arr.reshape(-1, len(header))
    cols = {h: arr[:, i] for i, h in enumerate(header)}
    return _from_columns(cols, start, source if path is None else Path(path).name, time_col, e_col)


def read_ec(path, dayfirst=False):
    p = Path(path)
    if p.suffix.lower() == ".mpr":
        return read_mpr(p)
    if p.suffix.lower() in (".mpt", ".txt"):
        return read_mpt(p, dayfirst=dayfirst)
    return read_table(p)


def write_mpt(path, ec, start=None):
    """A minimal EC-Lab-style .mpt (used for synthetic examples and tests)."""
    start = start or ec.start
    names = ["time/s", "Ewe/V", "I/mA", "cycle number"]
    head = ["EC-Lab ASCII FILE", "Nb header lines : 5", "",
            f"Acquisition started on : {start.strftime('%m/%d/%Y %H:%M:%S.%f')[:-3]}", "\t".join(names)]
    cur = ec.current if ec.current is not None else np.zeros_like(ec.time)
    cyc = ec.cycle if ec.cycle is not None else np.zeros_like(ec.time)
    rows = [f"{t:.4f}\t{e:.6f}\t{i:.6e}\t{c:g}" for t, e, i, c in zip(ec.time, ec.potential, cur, cyc)]
    Path(path).write_text("\n".join(head + rows) + "\n", encoding="latin-1")
    return path


# ----------------------------------------------------------------------------------------------
# a CV defined by hand (no potentiostat file)
# ----------------------------------------------------------------------------------------------
def cv_knots(E_hold, t_hold, E_upper, E_lower, rate, n_cycles, first="up", E_end=None):
    """Corner points (time s, potential V, cycle) of a hold followed by n_cycles of
    E_hold -> first vertex -> other vertex -> E_hold (EC-Lab style), ending at E_end (default E_hold)."""
    if rate <= 0:
        raise ValueError("scan rate must be positive")
    if E_upper <= E_lower:
        raise ValueError("upper vertex must be above the lower vertex")
    if n_cycles < 1:
        raise ValueError("at least one cycle")
    if t_hold < 0:
        raise ValueError("hold time cannot be negative")
    if first not in ("up", "down"):
        raise ValueError("first sweep must be 'up' or 'down'")
    v1, v2 = (E_upper, E_lower) if first == "up" else (E_lower, E_upper)
    t, E, c = [0.0], [E_hold], [0]
    if t_hold > 0:
        t.append(t_hold)
        E.append(E_hold)
        c.append(0)
    for k in range(1, n_cycles + 1):
        for target in (v1, v2, E_hold if k < n_cycles or E_end is None else E_end):
            t.append(t[-1] + abs(target - E[-1]) / rate)
            E.append(target)
            c.append(k)
    return np.array(t), np.array(E), np.array(c)


def make_cv(E_hold, t_hold, E_upper, E_lower, rate, n_cycles, first="up", E_end=None, start=None, dt=None):
    """ECData for a hand-defined CV: hold, then cycles (see cv_knots). rate in V/s; sampled every dt
    seconds (default: 1 mV steps, at most 0.5 s). start: local start time of the hold (datetime)."""
    kt, kE, kc = cv_knots(E_hold, t_hold, E_upper, E_lower, rate, n_cycles, first, E_end)
    dt = dt or min(0.5, 1e-3 / rate)
    t = np.unique(np.concatenate([np.arange(0, kt[-1], dt), kt]))
    E = np.interp(t, kt, kE)
    cyc = kc[np.clip(np.searchsorted(kt, t, side="left"), 0, kc.size - 1)]
    out = ECData(t, E, None, cyc.astype(float), start, {}, "CV defined by hand")
    out.notes.append(f"hand-defined CV: hold {E_hold:g} V for {t_hold:g} s, {n_cycles} cycle(s) {E_lower:g} to "
                     f"{E_upper:g} V at {1e3 * rate:g} mV/s, first sweep {first}")
    return out

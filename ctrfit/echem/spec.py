"""SPEC data files: scans, columns, motor positions and the times of every point.

Times are kept as seconds on a naive local clock (datetime treated as UTC, so no time zone is ever
applied): the file header pairs #E (epoch) with #D (its local date), so

    epoch columns   t = #D(file) + Epoch            (Epoch counts from #E)
    elapsed columns t = #D(scan) + Time             (loopscan / timescan 'Time' from the scan start)

both on the same clock as the potentiostat's local start time. #D has one-second resolution; the
sync step (sync.py) corrects any remaining offset.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

EPOCH0 = datetime(1970, 1, 1)
STATIONARY_COMMANDS = ("loopscan", "timescan", "tscan", "loop", "ctscan")
MOTOR_SCANS = ("ascan", "dscan", "lup", "a2scan", "d2scan", "a3scan", "d3scan", "mesh", "dmesh")


def to_seconds(dt):
    """Naive local datetime -> seconds (treated as UTC so no time zone is applied)."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return (dt - EPOCH0).total_seconds()


def from_seconds(s):
    return EPOCH0 + timedelta(seconds=float(s))


def parse_spec_date(text):
    """SPEC #D date, e.g. 'Thu Mar 07 14:23:11 2024' (also without the weekday)."""
    text = " ".join(text.split())
    for fmt in ("%a %b %d %H:%M:%S %Y", "%b %d %H:%M:%S %Y", "%Y-%m-%d %H:%M:%S", "%a %b %d %H:%M:%S.%f %Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"cannot read the SPEC date '{text}'")


@dataclass
class Scan:
    number: int
    command: str
    labels: list
    data: np.ndarray                    # (npoints, ncolumns)
    date: datetime = None               # #D of the scan
    file_date: datetime = None          # #D of the file header (local date of #E)
    file_epoch: float = None            # #E
    motors: dict = field(default_factory=dict)
    hkl: tuple = None                   # #Q
    comments: list = field(default_factory=list)
    count_time: float = None            # from the command or #T

    @property
    def kind(self):
        words = self.command.split()
        return words[0] if words else ""

    @property
    def npts(self):
        return self.data.shape[0]

    @property
    def stationary(self):
        """True for loopscans / timescans and motor scans whose start equals the end (no motor moves)."""
        w = self.command.split()
        if not w:
            return False
        if w[0] in STATIONARY_COMMANDS:
            return True
        if w[0] in MOTOR_SCANS:
            nums = [x for x in w[1:] if _isnum(x)]
            names = [x for x in w[1:] if not _isnum(x)]
            n_mot = max(len(names), 1)
            try:
                pairs = [(float(nums[2 * i]), float(nums[2 * i + 1])) for i in range(n_mot)]
            except IndexError:
                return False
            return all(abs(a - b) < 1e-9 for a, b in pairs)
        return False

    def column(self, name):
        try:
            return self.data[:, self.labels.index(name)]
        except ValueError:
            raise KeyError(f"scan {self.number} has no column '{name}' (columns: {', '.join(self.labels)})") from None

    def has(self, name):
        return name in self.labels

    def default_time_column(self):
        """(column, kind): 'epoch' (Epoch since #E) for motor scans, 'elapsed' (Time since the scan start)
        for loopscans; whichever exists."""
        prefer = (("Time", "elapsed"), ("Epoch", "epoch")) if self.kind in STATIONARY_COMMANDS else \
            (("Epoch", "epoch"), ("Time", "elapsed"))
        for name, kind in prefer:
            if self.has(name):
                return name, kind
        for name in self.labels:
            if name.lower() in ("epoch", "time", "seconds_elapsed", "elapsed"):
                return name, "epoch" if name.lower() == "epoch" else "elapsed"
        raise KeyError(f"scan {self.number}: no Epoch or Time column")

    def default_counter(self):
        for name in ("Detector", "detector", "det", "imroi1", "roi1", "Det", "apd", "pil_roi1"):
            if self.has(name):
                return name
        skip = {"Time", "Epoch", "Seconds", "Monitor", "H", "K", "L"} | set(self.motors)
        rest = [n for n in self.labels if n not in skip]
        return rest[-1] if rest else self.labels[-1]

    def default_transmission(self):
        for name in ("transm", "transmission", "trans", "Transm"):
            if self.has(name):
                return name
        return None

    def default_monitor(self):
        for name in ("Monitor", "monitor", "mon", "i0", "I0", "ic1"):
            if self.has(name):
                return name
        return None

    def times(self, column=None, kind=None, mark="end"):
        """Seconds (naive local clock, see the module docstring) of every point.

        kind: 'epoch' (seconds since #E), 'elapsed' (seconds since the scan's #D), 'unix' (absolute
        Unix time, converted with the time-zone offset of the file's #D/#E pair). Without the file
        header (scans cut out of a file) epoch times count from the scan's #D: the first point ends
        one count time after it (the #D is then good to the scan's start-up time, ~1 s).
        mark: when the time stamp is taken: 'end' of the count (SPEC default), 'start' or 'middle';
        the point is put at the middle of its counting window."""
        if column is None:
            column, kind = self.default_time_column()
        t = self.column(column).astype(float)
        kind = kind or ("epoch" if column.lower() == "epoch" else "elapsed")
        if kind == "epoch":
            if self.file_date is None:                 # scans cut out of the file: no #E / #D header
                if self.date is None:
                    raise ValueError("epoch times need the file header (#E, #D) or the scan's #D")
                ct0 = float(self.point_count_times()[0])
                t = to_seconds(self.date) + (t - t[0]) + ct0
            else:
                t = to_seconds(self.file_date) + t
        elif kind == "elapsed":
            if self.date is None:
                raise ValueError(f"scan {self.number} has no #D date")
            t = to_seconds(self.date) + t
        elif kind == "unix":
            if self.file_epoch is None or self.file_date is None:
                raise ValueError("unix times need #E and #D in the file header")
            t = t + (to_seconds(self.file_date) - self.file_epoch)
        else:
            raise ValueError(f"time kind must be epoch, elapsed or unix, not {kind!r}")
        ct = self.point_count_times()
        shift = {"end": -0.5, "start": 0.5, "middle": 0.0}[mark]
        return t + shift * ct

    def point_count_times(self):
        if self.has("Seconds"):
            return self.column("Seconds").astype(float)
        return np.full(self.npts, self.count_time or 0.0)


def _isnum(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def _count_time(command):
    w = command.split()
    nums = [float(x) for x in w[1:] if _isnum(x)]
    if not w or not nums:
        return None
    if w[0] in STATIONARY_COMMANDS:          # loopscan npts count_time [sleep]
        return nums[1] if len(nums) > 1 else None
    return nums[-1]                          # ascan mot start end intervals time


class SpecFile:
    """All scans of a SPEC data file (header #E, #D, #O motor names; per scan #S, #D, #P, #Q, #L, data)."""

    def __init__(self, path=None, text=None):
        self.path = Path(path) if path else None
        if text is None:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        self.scans = []
        self.file_epoch = None
        self.file_date = None
        self.motor_names = []
        self._parse(text)

    def _parse(self, text):
        cur = None
        motor_names, file_date, file_epoch = [], None, None
        rows = []

        def finish():
            if cur is None:
                return
            ncol = len(cur["labels"])
            good = [r for r in rows if len(r) == ncol]
            data = np.array(good, float).reshape(-1, ncol) if good else np.zeros((0, ncol))
            motors = {}
            for name, val in zip(motor_names, cur["P"]):
                motors[name] = val
            scan = Scan(cur["n"], cur["cmd"], cur["labels"], data, cur.get("D"), file_date, file_epoch, motors,
                        cur.get("Q"), cur["C"], _count_time(cur["cmd"]))
            if scan.count_time is None and cur.get("T") is not None:
                scan.count_time = cur["T"]
            self.scans.append(scan)

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#S "):
                finish()
                parts = line[3:].split(None, 1)
                cur = dict(n=int(parts[0]), cmd=parts[1].strip() if len(parts) > 1 else "", labels=[], P=[], C=[])
                rows = []
            elif line.startswith("#E ") and cur is None:
                file_epoch = float(line[3:].split()[0])
            elif line.startswith("#D "):
                d = parse_spec_date(line[3:])
                if cur is None:
                    file_date = d
                else:
                    cur["D"] = d
            elif re.match(r"#O\d+ ", line) and cur is None:
                motor_names += _split_names(line.split(None, 1)[1] if " " in line else "")
            elif cur is None:
                continue
            elif re.match(r"#P\d+ ", line):
                cur["P"] += [float(x) for x in line.split()[1:]]
            elif line.startswith("#Q"):
                vals = [float(x) for x in line[2:].split()[:3] if _isnum(x)]
                if len(vals) == 3:
                    cur["Q"] = tuple(vals)
            elif line.startswith("#T "):
                try:
                    cur["T"] = float(line[3:].split()[0])
                except ValueError:
                    pass
            elif line.startswith("#L "):
                cur["labels"] = _split_names(line[3:])
            elif line.startswith("#C"):
                cur["C"].append(line[2:].strip())
            elif line.startswith("#") or line.startswith("@"):
                continue
            else:
                try:
                    rows.append([float(x) for x in line.split()])
                except ValueError:
                    continue
        finish()
        self.file_epoch, self.file_date, self.motor_names = file_epoch, file_date, motor_names

    def scan(self, number):
        """The last scan with this number (SPEC files can repeat numbers after a restart)."""
        hits = [s for s in self.scans if s.number == number]
        if not hits:
            raise KeyError(f"no scan #{number} in the file")
        return hits[-1]

    def stationary_scans(self):
        return [s for s in self.scans if s.stationary and s.npts > 0]


def _split_names(text):
    """SPEC separates column and motor names by two or more spaces (names may contain one space)."""
    parts = [p.strip() for p in re.split(r"\s{2,}", text.strip()) if p.strip()]
    if len(parts) <= 1 and " " in text.strip():
        parts = text.split()
    return parts


def poisson_sigma(counts):
    """sqrt(N) for counts (at least 1, so zero-count points keep a finite weight)."""
    return np.sqrt(np.maximum(np.asarray(counts, float), 1.0))


__all__ = ["SpecFile", "Scan", "parse_spec_date", "to_seconds", "from_seconds", "poisson_sigma"]

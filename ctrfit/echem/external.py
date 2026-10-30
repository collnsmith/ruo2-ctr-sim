"""Intensity of a stationary scan from a file instead of a SPEC counter (no Qt).

Two kinds of file are recognised:

* ROI sums from the background subtraction tool: a header with 'scan' and 'image' (tab or space
  separated, optionally starting with '#'), then one row per image: scan, image, one column per ROI.
  The image number is the SPEC point number of the scan.
* Trace files from the plotter: 'Export data', 'Export t, E, I' and the smoothing's 'Export smoothed
  data' (blocks that start with '# Trace: <label>', then a header line and numeric rows), or a plain
  two-column x, y table.

A ROI sums column `<name>_err` (the background subtraction's counting error, or the processing
tool's propagated error) is the sigma of `<name>`: `read_intensity_sigma` returns it with the
intensity (None without one, then the caller falls back to sqrt(N)).

`read_intensity(path, scan_number, n_points, elapsed, column)` returns the intensity at every point of
the scan (NaN-free: images missing from a ROI file are interpolated from their neighbours) and a note.
For a trace whose length is not the scan's, the trace is interpolated on its x as seconds from the
first point (the plotter's 't = 0 at each scan's start'), with `elapsed` the scan's point times.
"""
import re

import numpy as np


def _rows(lines):
    """Numeric rows of lines (rows that do not parse are skipped)."""
    out = []
    for line in lines:
        parts = line.split()
        try:
            out.append([float(x) for x in parts])
        except ValueError:
            continue
    return out


def parse_roi_sums(text):
    """{scan: {column: array}} of a ROI sums file, or None if the text is not one."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    head = lines[0].lstrip("#").split()
    if len(head) < 3 or head[0].lower() != "scan" or head[1].lower() != "image":
        return None
    data = {}
    for row in _rows(lines[1:]):
        if len(row) != len(head):
            continue
        d = data.setdefault(int(row[0]), {h: [] for h in head})
        for h, v in zip(head, row):
            d[h].append(v)
    return {sn: {h: np.array(v) for h, v in cols.items()} for sn, cols in data.items()}


def parse_traces(text):
    """[{'label', 'columns': [names], 'data': (n, m) array}] of a trace file (one block per
    '# Trace:'), or of a plain x, y table (one block)."""
    blocks, cur = [], None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"#\s*Trace:\s*(\S+)", s)
        if m:
            cur = {"label": m.group(1), "columns": None, "rows": []}
            blocks.append(cur)
            continue
        if s.startswith("#"):
            continue
        if cur is None:
            cur = {"label": "data", "columns": None, "rows": []}
            blocks.append(cur)
        try:
            cur["rows"].append([float(x) for x in s.split()])
        except ValueError:
            if cur["columns"] is None and not cur["rows"]:
                cur["columns"] = s.split("\t") if "\t" in s else s.split()
    out = []
    for b in blocks:
        rows = [r for r in b["rows"] if b["rows"] and len(r) == len(b["rows"][0])]
        if not rows:
            continue
        data = np.array(rows, float)
        cols = b["columns"] if b["columns"] and len(b["columns"]) == data.shape[1] else \
            [f"col{i + 1}" for i in range(data.shape[1])]
        out.append({"label": b["label"], "columns": cols, "data": data})
    return out


def _y_index(columns, label):
    """The intensity column of a trace: the one named like the trace (or '..._smoothed'), else the
    last; 't_s E I' exports: the third."""
    low = [c.lower() for c in columns]
    for i, c in enumerate(low):
        if c == label.lower() or c.endswith("_smoothed") or c == label.lower() + "_smoothed":
            return i
    if len(columns) >= 3 and low[0].startswith("t") and low[1].startswith("e_"):
        return 2
    if low[:3] == ["h", "k", "l"] and len(columns) >= 4:
        return 3
    return len(columns) - 1


def choices(path):
    """('roi', [ROI names]) or ('trace', [trace labels]) of a file, for the window's column list."""
    text = open(path, errors="replace").read()
    sums = parse_roi_sums(text)
    if sums is not None:
        names = []
        for cols in sums.values():
            names += [c for c in cols if c.lower() not in ("scan", "image") and not c.endswith("_err")
                      and c not in names]
        return "roi", names
    traces = parse_traces(text)
    if not traces:
        raise ValueError(f"{path}: neither a ROI sums file nor a trace file")
    return "trace", [t["label"] for t in traces]


def read_intensity(path, scan_number, n_points, elapsed=None, column=None):
    """(intensity at each of the scan's n_points points, note). column: the ROI name (ROI sums) or
    the trace label (trace file); None: the first ROI / the trace of this scan ('S<number>...') or
    the first trace."""
    y, _, note = read_intensity_sigma(path, scan_number, n_points, elapsed, column)
    return y, note


def read_intensity_sigma(path, scan_number, n_points, elapsed=None, column=None):
    """(intensity, sigma or None, note) at each of the scan's n_points points; sigma from the
    '<column>_err' column of a ROI sums file."""
    text = open(path, errors="replace").read()
    sums = parse_roi_sums(text)
    if sums is not None:
        if scan_number not in sums:
            raise ValueError(f"scan {scan_number} is not in the ROI sums file (scans {sorted(sums)})")
        cols = sums[scan_number]
        names = [c for c in cols if c.lower() not in ("scan", "image") and not c.endswith("_err")]
        name = column or names[0]
        if name not in cols:
            raise ValueError(f"no ROI '{name}' in the ROI sums file ({', '.join(names)})")
        img = cols["image"].astype(int)
        ok = (img >= 0) & (img < n_points)
        if not ok.any():
            raise ValueError(f"no image of scan {scan_number} matches its {n_points} points")
        order = np.argsort(img[ok])
        idx, val = img[ok][order], cols[name][ok][order]
        y = np.interp(np.arange(n_points), idx, val)
        sig = None
        if f"{name}_err" in cols:
            sig = np.interp(np.arange(n_points), idx, cols[f"{name}_err"][ok][order])
        missing = n_points - np.unique(idx).size
        note = f"intensity: ROI '{name}' of {path.rsplit('/', 1)[-1]}"
        note += f", sigma from '{name}_err'" if sig is not None else ", sigma = sqrt(N)"
        if missing:
            note += f" ({missing} of {n_points} points missing, interpolated)"
        return y, sig, note
    traces = parse_traces(text)
    if not traces:
        raise ValueError(f"{path}: neither a ROI sums file nor a trace file")
    if column:
        pick = [t for t in traces if t["label"] == column]
        if not pick:
            raise ValueError(f"no trace '{column}' in the file ({', '.join(t['label'] for t in traces)})")
        tr = pick[0]
    else:
        mine = [t for t in traces if re.match(rf"S{scan_number}(\D|$)", t["label"])]
        tr = (mine or traces)[0]
    y = tr["data"][:, _y_index(tr["columns"], tr["label"])]
    note = f"intensity: trace '{tr['label']}' of {path.rsplit('/', 1)[-1]}"
    if y.size == n_points:
        return y.astype(float), None, note
    if elapsed is None:
        raise ValueError(f"trace '{tr['label']}' has {y.size} points, the scan {n_points}")
    x = tr["data"][:, 0]
    t = np.asarray(elapsed, float) - float(elapsed[0])
    order = np.argsort(x)
    return np.interp(t, x[order], y[order]), None, note + f" ({y.size} points interpolated on x as seconds from the first point)"

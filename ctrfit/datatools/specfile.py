"""SPEC file reader shared by the data tools (plotter, HKL appender, metadata viewer).

parse_spec_file(path) -> {scan number: SpecScan}. Columns come from '#L' (split on two or more
spaces, the SPEC convention), motor positions from '#O' names and '#P' values. Rows with the wrong
number of values or text in them are skipped. A scan number that appears twice keeps the last one.
"""
import re
from collections import OrderedDict

import numpy as np


class SpecScan:
    def __init__(self, scan_number, scan_cmd, columns, data, motor_positions=None):
        self.scan_number = scan_number
        self.scan_cmd = scan_cmd
        self.columns = columns
        self.data = data
        self.motor_positions = motor_positions or {}
        self.npoints = len(next(iter(data.values()))) if data else 0

    def __repr__(self):
        return f"Scan {self.scan_number}: {self.scan_cmd} ({self.npoints} pts)"


def _split_names(text):
    return [c.strip() for c in re.split(r"  +", text) if c.strip()]


def parse_spec_file(filepath):
    scans = {}
    motor_names = {}                  # '#O<n>' index -> names (file header)
    current_scan = None
    current_columns, current_data, current_cmd, motor_positions = [], [], "", {}

    def flush():
        if current_scan is not None and current_columns and current_data:
            arr = np.array(current_data, dtype=np.float64)
            d = OrderedDict()
            for i, col in enumerate(current_columns):
                if i < arr.shape[1]:
                    d[col] = arr[:, i]
            scans[current_scan] = SpecScan(current_scan, current_cmd, current_columns, d, dict(motor_positions))

    with open(filepath, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("#S "):
                flush()
                parts = line.split()
                current_scan = int(parts[1])
                current_cmd = " ".join(parts[2:])
                current_columns, current_data, motor_positions = [], [], {}
            elif re.match(r"#O\d+ ", line):
                n = int(line[2:line.index(" ")])
                motor_names[n] = _split_names(line[line.index(" ") + 1:])
            elif re.match(r"#P\d+ ", line):
                n = int(line[2:line.index(" ")])
                vals = line[line.index(" ") + 1:].split()
                for name, v in zip(motor_names.get(n, []), vals):
                    try:
                        motor_positions[name] = float(v)
                    except ValueError:
                        pass
            elif line.startswith("#L "):
                current_columns = _split_names(line[3:])
            elif line.startswith("#") or line.strip() == "":
                continue
            elif current_columns:
                vals = line.split()
                if len(vals) == len(current_columns):
                    try:
                        current_data.append([float(v) for v in vals])
                    except ValueError:
                        pass
    flush()
    return scans

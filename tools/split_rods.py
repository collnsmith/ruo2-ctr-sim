"""Split fitting CSV files into one file per film, potential and rod: all the scans of one rod at one
potential together (repeats too), with the metadata the Fitting app reads (film, thickness,
potential, condition, energy, error floor).

    python tools/split_rods.py fit_4.1_nm_CA_0_V.csv fit_7_nm_*.csv          # next to the first file, in split_by_rod/
    python tools/split_rods.py *.csv --out my_folder
    python tools/split_rods.py                                               # pick the files in a dialog

Files of the same film and potential (e.g. a repeat exported separately) are combined per rod. A
file's rows keep their order. Input files are never written over. Not part of the launcher.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctrfit.data.dataset import Dataset  # noqa: E402
from ctrfit.data.io import write_csv  # noqa: E402


def _safe(text):
    return re.sub(r"[^\w.+-]+", "_", str(text)).strip("_")


def _condition(ds, path):
    if ds.meta.get("condition"):
        return ds.meta["condition"]
    if ds.meta.get("potential_V") is not None:
        return f"{ds.meta['potential_V']:g} V"
    return Path(path).stem


def split(paths, out_dir=None):
    """Write one CSV per (film, potential, rod) of the given files; returns [(path, points, notes)]."""
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("no files given")
    groups = {}
    for p in paths:
        d = Dataset.load(p)
        key0 = (d.meta.get("film", ""), _condition(d, p))
        for lab, idx in d.rods().items():
            groups.setdefault(key0 + (lab,), []).append((p, d, idx))
    out = Path(out_dir) if out_dir else Path(paths[0]).resolve().parent / "split_by_rod"
    out.mkdir(parents=True, exist_ok=True)
    inputs = {os.path.normcase(str(Path(p).resolve())) for p in paths}
    written = []
    for (film, cond, lab), parts in groups.items():
        subs = [d.subset(idx) for _, d, idx in parts]
        cat = lambda a: np.concatenate([getattr(s, a) for s in subs])
        first = parts[0][1]
        notes = []
        for k in ("energy_kev", "thickness_nm"):
            vals = {d.meta.get(k) for _, d, _ in parts}
            if len(vals) > 1:
                notes.append(f"{k} differs between the files ({', '.join(map(str, vals))}): the first is kept")
        if len({d.sys_floor for _, d, _ in parts}) > 1:
            notes.append("the error floors differ between the files: the first is kept")
        has_scans = any(s.has_scans for s in subs)
        scans = list(dict.fromkeys(np.concatenate([s.point_scans() for s in subs]).tolist())) if has_scans else []
        files = ", ".join(Path(p).name for p, _, _ in parts)
        meta = {k: v for k, v in first.meta.items() if k != "name"}
        meta["notes"] = (f"scans {', '.join(scans)}; " if scans else "") + f"rod {lab} of {files}" + (
            f"; the file listed: {first.meta['notes']}" if not scans and first.meta.get("notes") else "")
        ds = Dataset(cat("H"), cat("K"), cat("L"), cat("F"), cat("sigma"), cat("rod"),
                     name=f"{film} {cond} ({lab})".strip(), sys_floor=first.sys_floor,
                     scan=cat("scan") if has_scans else None, **meta)
        path = out / (_safe(f"{film}_{cond}_rod_{lab}") + ".csv")
        if os.path.normcase(str(path.resolve())) in inputs:
            raise ValueError(f"{path} is one of the input files: choose another output folder")
        write_csv(ds, path)
        written.append((path, len(ds), notes))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description="Split fitting CSV files into one file per film, potential and rod")
    ap.add_argument("files", nargs="*", help="fitting CSV files (from the scan library's Export for fitting)")
    ap.add_argument("--out", default=None, help="output folder (default: split_by_rod next to the first file)")
    a = ap.parse_args(argv)
    files = a.files
    if not files:                                  # no arguments: pick the files in a dialog
        try:
            from PyQt5 import QtWidgets
        except ImportError:
            ap.error("give the CSV files (PyQt5 is not installed for the file dialog)")
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(None, "Fitting CSV files to split by rod", "",
                                                          "CSV (*.csv);;All files (*)")
        if not files:
            return 1
    written = split(files, a.out)
    for path, n, notes in written:
        print(f"{path.name}: {n} points" + "".join(f"\n   note: {x}" for x in notes))
    print(f"{len(written)} files in {written[0][0].parent}")
    if not a.files:
        QtWidgets.QMessageBox.information(None, "Split by rod", f"{len(written)} files written to\n"
                                          f"{written[0][0].parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

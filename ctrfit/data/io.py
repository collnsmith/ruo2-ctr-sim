"""Import and export of structure-factor data.

CSV:   a header row with at least H, K, L, F, sigma (any order, case-insensitive; 'sigF', 'dF',
       'err' also accepted for sigma). With intensities=True the columns are I and sigma (or sigI).
       An optional 'rod' column labels the points. Lines starting with # are comments;
       '# key: value' comment lines set metadata (name, energy_kev, potential_V, thickness_nm,
       sys_floor).
.dat:  whitespace columns H K L F sigma (GenX / ANA-ROD style), # comments, no header.

intensities_to_F converts integrated intensities to amplitudes with propagated errors.
"""
import csv

import numpy as np

from .dataset import Dataset

_SIGMA_NAMES = ("sigma", "sigf", "df", "err", "error", "sigma_f", "sigi", "di", "sigma_i")
_FLOAT_META = ("energy_kev", "potential_V", "thickness_nm", "sys_floor")


def intensities_to_F(I, sigma_I):
    """F = sqrt(I) and sigma_F = sigma_I / (2 sqrt(I)).

    The linear propagation diverges for weak points, so for I < sigma_I / 4 (including I <= 0) the
    amplitude error is set to sqrt(sigma_I), the value the linear rule reaches at I = sigma_I / 4,
    and F = sqrt(max(I, 0)).
    """
    I = np.asarray(I, float)
    sI = np.asarray(sigma_I, float)
    F = np.sqrt(np.clip(I, 0.0, None))
    strong = I >= sI / 4
    sF = np.where(strong, sI / (2 * np.sqrt(np.where(strong, I, 1.0))), np.sqrt(sI))
    return F, sF


def _meta_from_comments(lines):
    meta = {}
    for ln in lines:
        body = ln.lstrip("#").strip()
        if ":" in body:
            k, v = (s.strip() for s in body.split(":", 1))
            if k in _FLOAT_META:
                try:
                    meta[k] = float(v)
                except ValueError:
                    pass
            elif k in ("name", "notes"):
                meta[k] = v
    return meta


def _finish(cols, meta, intensities, name, sys_floor):
    H, K, L, Y, sY = (np.asarray(cols[c], float) for c in ("H", "K", "L", "Y", "sY"))
    if intensities:
        Y, sY = intensities_to_F(Y, sY)
    meta = dict(meta)
    floor = meta.pop("sys_floor", 0.0) if sys_floor is None else sys_floor
    nm = name or meta.pop("name", "data")
    meta.pop("name", None)
    return Dataset(H, K, L, Y, sY, cols.get("rod"), name=nm, sys_floor=floor, **meta)


def read_csv(path, intensities=False, name=None, sys_floor=None):
    with open(path, encoding="utf-8", newline="") as fh:
        raw = fh.read().splitlines()
    comments = [ln for ln in raw if ln.lstrip().startswith("#")]
    body = [ln for ln in raw if ln.strip() and not ln.lstrip().startswith("#")]
    rows = list(csv.reader(body))
    header = [h.strip() for h in rows[0]]
    low = [h.lower() for h in header]

    def col(*names):
        for n in names:
            if n.lower() in low:
                return low.index(n.lower())
        raise ValueError(f"{path}: no column named {' or '.join(names)} in header {header}")

    idx = dict(H=col("H"), K=col("K"), L=col("L"),
               Y=col("I", "intensity") if intensities else col("F", "Fobs", "amplitude"),
               sY=col(*_SIGMA_NAMES))
    data = rows[1:]
    cols = {k: [float(r[i]) for r in data] for k, i in idx.items()}
    if "rod" in low:
        cols["rod"] = [r[low.index("rod")].strip() for r in data]
    return _finish(cols, _meta_from_comments(comments), intensities, name, sys_floor)


def read_dat(path, intensities=False, name=None, sys_floor=None):
    with open(path, encoding="utf-8") as fh:
        raw = fh.read().splitlines()
    comments = [ln for ln in raw if ln.lstrip().startswith("#")]
    rows = [ln.split() for ln in raw if ln.strip() and not ln.lstrip().startswith("#")]
    try:
        arr = np.array([[float(x) for x in r[:5]] for r in rows])
    except ValueError:
        raise ValueError(f"{path}: expected numeric columns H K L F sigma") from None
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"{path}: expected 5 columns H K L F sigma")
    cols = dict(H=arr[:, 0], K=arr[:, 1], L=arr[:, 2], Y=arr[:, 3], sY=arr[:, 4])
    return _finish(cols, _meta_from_comments(comments), intensities, name, sys_floor)


def _meta_lines(ds):
    items = dict(name=ds.name, sys_floor=ds.sys_floor, **ds.meta)
    return [f"# {k}: {v!r}" if isinstance(v, float) else f"# {k}: {v}" for k, v in items.items()]


def write_csv(ds, path):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        for ln in _meta_lines(ds):
            fh.write(ln + "\n")
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["H", "K", "L", "F", "sigma", "rod"])
        for row in zip(ds.H, ds.K, ds.L, ds.F, ds.sigma, ds.rod):
            w.writerow([int(row[0]), int(row[1]), repr(float(row[2])), repr(float(row[3])), repr(float(row[4])),
                        row[5]])


def write_dat(ds, path):
    with open(path, "w", encoding="utf-8") as fh:
        for ln in _meta_lines(ds):
            fh.write(ln + "\n")
        fh.write("# H K L F sigma\n")
        for h, k, l, f, s in zip(ds.H, ds.K, ds.L, ds.F, ds.sigma):
            fh.write(f"{int(h)} {int(k)} {float(l)!r} {float(f)!r} {float(s)!r}\n")


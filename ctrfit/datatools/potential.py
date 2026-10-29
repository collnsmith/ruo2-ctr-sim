"""Potentiostat data next to SPEC scans in the plotter (no Qt).

Time axis: the plotter's X column (e.g. Epoch, s) minus t0. `time_zero` gives the t0 that puts the
start of one scan at 0. The potentiostat record (an EC-Lab .mpr / .mpt / table, or a CV defined by
hand) has its own clock from its first sample; `offset` is where that first sample lies on the
plotter's time axis (s), so on the axis the potential is E(t - offset).

    potential_at(ec, t, offset)        E at the plotted times (NaN outside the record)
    trace_between(ec, offset, lo, hi)  the potential curve over [lo, hi] for a second axis
"""
import numpy as np

from ..echem.ec import make_cv, read_ec

MODES = ("Potential on a second axis (vs time)", "Intensity vs potential")
FIRST_SWEEPS = ("up", "down")


def time_zero(x):
    """First finite value of a scan's time column (the t0 that makes the scan start at 0)."""
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        raise ValueError("this scan has no finite values in the X column")
    return float(finite[0])


def load_potential(path):
    """ECData from an EC-Lab .mpr (needs galvani), .mpt / .txt export or a CSV / table file."""
    return read_ec(path)


def hand_cv(E_start, hold, E_upper, E_lower, rate_mV_s, cycles, first="up"):
    """A CV defined by hand: hold at E_start (V) for `hold` s, then `cycles` cycles between E_lower and
    E_upper (V) at rate_mV_s (mV/s), the first sweep going `first` ('up' or 'down')."""
    if rate_mV_s <= 0:
        raise ValueError("the sweep rate must be positive")
    if E_upper <= E_lower:
        raise ValueError("the upper potential must be above the lower one")
    if first not in FIRST_SWEEPS:
        raise ValueError(f"first sweep must be one of {FIRST_SWEEPS}")
    return make_cv(E_start, hold, E_upper, E_lower, rate_mV_s / 1000.0, int(cycles), first)


def potential_at(ec, t, offset=0.0):
    """Potential (V) at the plotted times t (s); NaN before the first or after the last sample."""
    t = np.asarray(t, dtype=float)
    return np.interp(t - offset, ec.time, ec.potential, left=np.nan, right=np.nan)


def trace_between(ec, offset, lo, hi):
    """(t, E) of the record on the plotted time axis, cut to [lo, hi] with the end points
    interpolated, for drawing E(t) on a second axis over the X range of the data."""
    t = ec.time + offset
    keep = (t > lo) & (t < hi)
    ends_t = np.array([lo, hi], dtype=float)
    ends_E = np.interp(ends_t - offset, ec.time, ec.potential, left=np.nan, right=np.nan)
    tt = np.concatenate([[ends_t[0]], t[keep], [ends_t[1]]])
    EE = np.concatenate([[ends_E[0]], ec.potential[keep], [ends_E[1]]])
    ok = np.isfinite(EE)
    return tt[ok], EE[ok]

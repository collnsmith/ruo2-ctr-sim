"""Put the SPEC points on the potentiostat time axis and give each point its potential.

Three ways to line up the clocks (EC time = SPEC time + offset):

    "absolute"  both files' own clocks (SPEC #D/#E, EC-Lab 'Acquisition started on'); the offset (s)
                corrects a difference between the two computers' clocks (0 if they are synchronised).
    "manual"    times from each file's own start (the scan's #D, the first EC sample); the offset is
                minus the delay of the EC start after the scan start (EC started 60 s later: -60).
    "events"    the offset that puts the intensity change after the relaxation period on the start of
                the potential sweep (both detected; either can be typed in). Useful when the clocks are
                unknown, but biased if the intensity only starts to change some time into the CV.

A point's potential is the mean potential over its counting window, so a fast sweep is not
sampled at one instant.
"""
from dataclasses import dataclass, field

import numpy as np

MODES = ("absolute", "manual", "events")


# ----------------------------------------------------------------------------------------------
# events
# ----------------------------------------------------------------------------------------------
def detect_sweep_start(t, E, rate_frac=0.3, persist=5):
    """Time when the potential starts to sweep: |dE/dt| first stays above rate_frac of the typical
    sweep rate for `persist` points. Returns (time, typical rate in V/s)."""
    t, E = np.asarray(t, float), np.asarray(E, float)
    if t.size < 3:
        raise ValueError("too few potentiostat points")
    dEdt = np.gradient(E, t)
    k = max(1, min(25, t.size // 200))
    if k > 1:
        dEdt = np.convolve(dEdt, np.ones(k) / k, mode="same")
    mag = np.abs(dEdt)
    moving = mag[mag > 1e-6]
    if moving.size == 0:
        raise ValueError("the potential never changes")
    rate = float(np.percentile(moving, 75))
    above = mag > rate_frac * rate
    run = np.convolve(above.astype(int), np.ones(persist, int), mode="valid") == persist
    idx = np.flatnonzero(run)
    if idx.size == 0:
        raise ValueError("no potential sweep found")
    return float(t[idx[0]]), rate


def detect_intensity_onset(t, I, base_frac=0.2, n_sigma=5.0, persist=5, base_end=None):
    """Time when the intensity leaves its relaxation-period behaviour: a straight line is fitted to
    the first `base_frac` of the points (or those before base_end) and the onset is the first point
    that starts a run of `persist` points more than n_sigma scatter away from it."""
    t, I = np.asarray(t, float), np.asarray(I, float)
    n = t.size
    if n < 10:
        raise ValueError("too few SPEC points to find the onset")
    base = t <= base_end if base_end is not None else np.arange(n) < max(5, int(base_frac * n))
    if base.sum() < 3:
        raise ValueError("the relaxation window holds fewer than 3 points")
    p = np.polyfit(t[base] - t[0], I[base], 1)
    model = np.polyval(p, t - t[0])
    resid = I[base] - model[base]
    sig = max(float(1.4826 * np.median(np.abs(resid - np.median(resid)))), 1e-12 * max(abs(I).max(), 1))
    off = np.abs(I - model) > n_sigma * sig
    off[base] = False                          # the relaxation window itself is never the onset
    run = np.convolve(off.astype(int), np.ones(persist, int), mode="valid") == persist
    first = np.flatnonzero(run)
    if first.size == 0:
        raise ValueError("no intensity change found after the relaxation period")
    return float(t[first[0]])


# ----------------------------------------------------------------------------------------------
# alignment
# ----------------------------------------------------------------------------------------------
@dataclass
class Alignment:
    mode: str
    offset: float                    # EC time = SPEC time + offset (s), on the mode's time axes
    spec_t: np.ndarray               # SPEC point times on the mode's axis
    ec_t: np.ndarray                 # EC times on the mode's axis
    onset_spec: float = None         # detected / given intensity onset (mode's SPEC axis)
    sweep_start_ec: float = None     # detected / given sweep start (mode's EC axis)
    sweep_rate: float = None
    notes: list = field(default_factory=list)

    @property
    def t_ec_of_points(self):
        return self.spec_t + self.offset


def align(spec_abs, ec, mode="absolute", offset=0.0, intensity=None, onset=None, sweep_start=None,
          fine_shift=0.0, spec_zero=None, axes=None):
    """spec_abs: SPEC point times (seconds on the local clock, from Scan.times); ec: ECData.
    onset / sweep_start (events mode, optional) override the detected values; they are on the
    mode's axes (absolute clocks if both files have them, else from each file's start).
    spec_zero: SPEC start for the relative axis (the scan's #D; default the first point).
    axes: 'absolute' or 'relative' to force the time axes (default: from the mode)."""
    if mode not in MODES:
        raise ValueError(f"alignment mode must be one of {', '.join(MODES)}")
    spec_abs = np.asarray(spec_abs, float)
    notes = []
    use_abs = axes == "absolute" if axes else (mode == "absolute" or (mode == "events" and ec.start is not None))
    if use_abs:
        s_t, e_t = spec_abs, ec.absolute_time()
    else:
        zero = spec_abs[0] if spec_zero is None else float(spec_zero)
        s_t, e_t = spec_abs - zero, np.asarray(ec.time, float) - float(ec.time[0])
        if mode == "events":
            notes.append("the potentiostat file has no start time: events aligned on times from each file's start")
    al = Alignment(mode, float(offset), s_t, e_t, notes=notes)
    if mode == "events":
        if intensity is None and onset is None:
            raise ValueError("events alignment needs the intensity (or a given onset time)")
        al.onset_spec = float(onset) if onset is not None else detect_intensity_onset(s_t, intensity)
        if sweep_start is not None:
            al.sweep_start_ec, al.sweep_rate = float(sweep_start), None
        else:
            al.sweep_start_ec, al.sweep_rate = detect_sweep_start(e_t, ec.potential)
        al.offset = al.sweep_start_ec - al.onset_spec + float(fine_shift)
        notes.append("events: the intensity onset is put on the sweep start; if the intensity only changes some "
                     "time into the CV this shifts the potential axis, so check with the absolute clocks")
    lo, hi = e_t[0], e_t[-1]
    t_pts = al.t_ec_of_points
    out = int(np.sum((t_pts < lo) | (t_pts > hi)))
    if out:
        notes.append(f"{out} of {t_pts.size} SPEC points fall outside the potentiostat record (no potential)")
    if not np.any((t_pts >= lo) & (t_pts <= hi)):
        notes.append("no SPEC point overlaps the potentiostat record: check the files, clocks and offset")
    return al


def potential_at_points(al, ec, count_times=None, n_sub=9):
    """Mean potential (and current) over each point's counting window; NaN outside the EC record."""
    t = al.t_ec_of_points
    ct = np.zeros_like(t) if count_times is None else np.asarray(count_times, float)
    sub = np.linspace(-0.5, 0.5, n_sub)
    tt = t[:, None] + ct[:, None] * sub[None, :]
    inside = (tt >= al.ec_t[0]) & (tt <= al.ec_t[-1])
    E = np.interp(tt, al.ec_t, ec.potential)
    E = np.where(inside.all(axis=1), E.mean(axis=1), np.nan)
    I = None
    if ec.current is not None:
        I = np.interp(tt, al.ec_t, ec.current).mean(axis=1)
        I = np.where(inside.all(axis=1), I, np.nan)
    return E, I


# ----------------------------------------------------------------------------------------------
# cycles and sweep direction
# ----------------------------------------------------------------------------------------------
def sweep_branches(ec_t, E, t_points, hold_frac=0.2):
    """Direction at each point: +1 anodic (rising potential), -1 cathodic, 0 hold; and the cycle
    number (a cycle starts at each change from cathodic to anodic, first sweep is cycle 1).
    Points outside the EC record get direction 0 and cycle 0."""
    ec_t, E = np.asarray(ec_t, float), np.asarray(E, float)
    dEdt = np.gradient(E, ec_t)
    k = max(1, min(25, ec_t.size // 200))
    if k > 1:
        dEdt = np.convolve(dEdt, np.ones(k) / k, mode="same")
    moving = np.abs(dEdt)[np.abs(dEdt) > 1e-6]
    rate = float(np.percentile(moving, 75)) if moving.size else 0.0
    sgn = np.where(np.abs(dEdt) > hold_frac * rate, np.sign(dEdt), 0).astype(int)
    # cycle counter on the EC samples: count cathodic -> anodic changes, ignoring holds
    cyc = np.zeros(ec_t.size, int)
    c, last, started = 0, 0, False
    for i, s in enumerate(sgn):
        if s != 0:
            if not started:
                c, started = 1, True
            elif s == 1 and last == -1:
                c += 1
            last = s
        cyc[i] = c if started else 0
    t_points = np.asarray(t_points, float)
    inside = (t_points >= ec_t[0]) & (t_points <= ec_t[-1])
    j = np.clip(np.searchsorted(ec_t, t_points), 0, ec_t.size - 1)
    branch = np.where(inside, sgn[j], 0)
    cycle = np.where(inside, cyc[j], 0)
    return branch, cycle, rate


__all__ = ["MODES", "Alignment", "align", "detect_sweep_start", "detect_intensity_onset", "potential_at_points",
           "sweep_branches"]

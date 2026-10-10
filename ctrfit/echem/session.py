"""The whole chain for one stationary scan and one potentiostat file (no Qt):

    SPEC scan -> intensity (counter / monitor, per second) and point times
    EC file   -> potential vs time
    sync      -> potential, current, sweep direction and cycle of every point
    background (relaxation period) -> corrected intensity
    cycles / bins -> I(V) per sweep direction
    sigmoid fits per direction, hysteresis
    CTR model at the HKL: simulated I(V) for a composition path, path fit, coverage from intensity
"""
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.settings import DEFAULTS, parse_comp, read_ini
from . import analysis as an
from . import predict as pr
from . import sync as sy
from .ec import read_ec
from .spec import SpecFile, poisson_sigma


@dataclass
class Options:
    scan: int = None
    counter: str = None
    monitor: str = None              # None: no monitor normalisation
    transmission: str = None         # attenuator transmission column (intensity / transmission), None: none
    per_second: bool = True
    time_column: str = None
    time_kind: str = None            # epoch | elapsed | unix (None: from the column)
    time_mark: str = "end"           # the SPEC time stamp marks the end / start / middle of the count
    sync_mode: str = "absolute"
    offset: float = 0.0              # EC time = SPEC time + offset (s)
    onset: float = None              # events mode: intensity onset (s from the first point), None = detect
    onset_point: int = None          # events mode: the CV starts at this SPEC point (0-based), overrides onset
    sweep_start: float = None        # events mode: sweep start (s from the EC file start), None = detect
    fine_shift: float = 0.0          # events mode: added to the event offset (s)
    ec_start: str = None             # override the EC file's start time ('YYYY-mm-dd HH:MM:SS.fff')
    cv_manual: bool = False          # define the CV by hand instead of loading a potentiostat file
    cv_hold_V: float = 0.5           # potential during the relaxation period
    cv_hold_s: float = 300.0         # relaxation (hold) time before the first sweep
    cv_upper: float = 1.4            # V
    cv_lower: float = 0.4            # V
    cv_rate: float = 10.0            # mV/s
    cv_cycles: int = 3               # 0: as many as fit until the end point
    cv_start_point: int = None       # the first sweep starts at this SPEC point (0-based); overrides cv_delay
    cv_end_point: int = None         # the CV was stopped at this SPEC point (0-based)
    cv_start_marks: str = "sweep"    # cv_start_point marks the start of the first 'sweep' or of the 'hold' 
    cv_first: str = "up"             # first sweep towards the upper ('up') or lower vertex
    cv_delay: float = 0.0            # the hold starts this many s after the scan start (#D)
    background: str = "linear"
    correction: str = "divide"
    relax_end: float = None          # end of the relaxation window (s from the first point), None = onset
    cycles: str = "all"
    bin_width: float = 0.01          # V
    average: bool = True
    n_transitions: int = 1
    slope: bool = False
    hkl: tuple = None                # None: from the scan's #Q
    states: str = "H2O -> OH -> O"
    transitions: str = "0.8, 0.03; 1.1, 0.04"
    theta: float = 1.0
    cathodic_shift: float = 0.0
    reference: str = "H2O=1"         # state during the relaxation period (sets the scale)
    fit_shift: bool = True
    fit_theta: bool = False
    sim_settings: str = None         # simulator settings .ini (None: defaults)

    def to_dict(self):
        d = dict(self.__dict__)
        d["hkl"] = list(self.hkl) if self.hkl is not None else None
        return d

    @classmethod
    def from_dict(cls, d):
        o = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if o.hkl is not None:
            o.hkl = tuple(o.hkl)
        return o


@dataclass
class Result:
    t: np.ndarray = None             # s from the first SPEC point
    t_ec: np.ndarray = None          # EC time of each point
    I_raw: np.ndarray = None
    I: np.ndarray = None             # normalised intensity
    sigma: np.ndarray = None
    I_corr: np.ndarray = None        # after the background
    sigma_corr: np.ndarray = None
    V: np.ndarray = None
    current: np.ndarray = None
    branch: np.ndarray = None
    cycle: np.ndarray = None
    alignment: object = None
    background: object = None
    binned: list = field(default_factory=list)
    fits: dict = field(default_factory=dict)          # branch -> SigmoidFit
    hysteresis: list = field(default_factory=list)
    period_I: float = None           # repeat time of the intensity during the CV (s)
    period_E: float = None           # repeat time of the potential (s)
    point_model: object = None
    sim: dict = field(default_factory=dict)           # V grid, I per branch, fractions
    path_fit: object = None
    coverage: object = None
    scale: float = None
    notes: list = field(default_factory=list)


class Session:
    def __init__(self, options=None):
        self.opt = options or Options()
        self.spec = None
        self.ec = None
        self.res = Result()
        self._pm_key = None

    # ------------------------------------------------------------------ loading
    def load_spec(self, path=None, text=None):
        self.spec = SpecFile(path, text)
        if not self.spec.scans:
            raise ValueError("no scans in the SPEC file")
        if self.opt.scan is None or self.opt.scan not in [s.number for s in self.spec.scans]:
            st = self.spec.stationary_scans()
            self.opt.scan = (st or self.spec.scans)[-1].number
        return self.spec

    def load_ec(self, path):
        self.ec = read_ec(path)
        return self.ec

    @property
    def scan(self):
        if self.spec is None:
            raise ValueError("load a SPEC file first")
        return self.spec.scan(self.opt.scan)

    def hkl(self):
        if self.opt.hkl is not None:
            h = tuple(self.opt.hkl)
        elif self.scan.hkl is not None:
            h = self.scan.hkl
        elif all(self.scan.has(c) for c in "HKL"):
            h = tuple(float(np.mean(self.scan.column(c))) for c in "HKL")
        else:
            raise ValueError("no HKL in the scan (#Q or H K L columns): enter it")
        H, K = round(h[0]), round(h[1])                # the model is evaluated on the rod
        if max(abs(h[0] - H), abs(h[1] - K)) > 0.1:
            raise ValueError(f"HKL ({h[0]:.3f} {h[1]:.3f} {h[2]:.3f}) is not on a rod: enter it")
        return (float(H), float(K), float(h[2]))

    # ------------------------------------------------------------------ steps
    def intensity(self):
        s, o = self.scan, self.opt
        counter = o.counter or s.default_counter()
        raw = s.column(counter).astype(float)
        sig = poisson_sigma(raw)
        f = np.ones_like(raw)
        if o.transmission:
            tr = s.column(o.transmission).astype(float)
            if np.any(tr <= 0):
                raise ValueError(f"transmission '{o.transmission}' has zero or negative values")
            f = f / tr
        if o.monitor:
            mon = s.column(o.monitor).astype(float)
            if np.any(mon <= 0):
                raise ValueError(f"monitor '{o.monitor}' has zero or negative values")
            f = f * np.mean(mon) / mon
        if o.per_second:
            ct = s.point_count_times()
            if np.all(ct > 0):
                f = f / ct
        return raw, raw * f, sig * f

    def events_offset(self):
        """The offset that lines up the intensity onset with the sweep start, on the time axes of the
        current sync mode (to type into 'absolute' or 'manual'); the session itself is not changed."""
        mode = self.opt.sync_mode
        keep = self.res
        self.opt.sync_mode = "events"
        try:
            r = self.run(upto="sync", axes="absolute" if mode == "absolute" else "relative")
            return r.alignment.offset - self.opt.fine_shift
        finally:
            self.opt.sync_mode = mode
            self.res = keep

    def loop_offset(self, lo=-120.0, hi=120.0, step=2.0):
        """Time shift (s, added to the offset; to the fine shift in events mode) that makes the anodic
        and cathodic I(V) curves agree best (mean |I_anodic - I_cathodic| / I over their common
        potential range, binned). Returns (shift, mismatch at the shift, mismatch at 0, scan table).
        A real hysteresis (slow kinetics) is closed by this too: use it to test a timing error, not
        to remove hysteresis."""
        o = self.opt
        field_ = "fine_shift" if o.sync_mode == "events" else "offset"
        base = getattr(o, field_)
        keep = self.res
        avg = o.average

        def mismatch(shift):
            setattr(o, field_, base + shift)
            r = self.run(upto="fit")
            b = {x.branch: x for x in r.binned}
            if 1 not in b or -1 not in b:
                return np.nan
            v0 = max(b[1].V.min(), b[-1].V.min())
            v1 = min(b[1].V.max(), b[-1].V.max())
            if v1 - v0 < 3 * o.bin_width:
                return np.nan
            v = np.linspace(v0, v1, 60)
            ia, ic = np.interp(v, b[1].V, b[1].I), np.interp(v, b[-1].V, b[-1].I)
            return float(np.mean(np.abs(ia - ic)) / np.mean(np.abs(ia) + np.abs(ic)) * 2)
        try:
            o.average = True
            shifts = np.arange(lo, hi + step / 2, step)
            m = np.array([mismatch(x) for x in shifts])
            if not np.any(np.isfinite(m)):
                raise ValueError("no shift gives overlapping anodic and cathodic sweeps")
            i = int(np.nanargmin(m))
            best = float(shifts[i])
            fine = np.arange(best - step, best + step + 1e-9, step / 8)
            mf = np.array([mismatch(x) for x in fine])
            j = int(np.nanargmin(mf))
            best, m_best = float(fine[j]), float(mf[j])
            m0 = mismatch(0.0)
        finally:
            setattr(o, field_, base)
            o.average = avg
            self.res = keep
        return best, m_best, m0, np.column_stack([shifts, m])

    def run(self, upto="all", axes=None):
        """Recompute from the current options. upto: 'sync', 'background', 'fit' or 'all'."""
        o, r = self.opt, Result()
        if self.spec is None:
            raise ValueError("load a SPEC file first")
        s = self.scan
        r.I_raw, r.I, r.sigma = self.intensity()
        col, kind = (o.time_column, o.time_kind) if o.time_column else s.default_time_column()
        t_abs = s.times(col, kind, o.time_mark)
        r.t = t_abs - t_abs[0]
        if o.cv_manual:
            from .spec import to_seconds as _ts
            self.ec = self.manual_cv(_ts(s.date) if s.date is not None else t_abs[0], t_abs)
        if self.ec is None:
            raise ValueError("load the potentiostat file, or define the CV by hand")
        if o.ec_start and not o.cv_manual:
            from .ec import parse_ec_date
            from datetime import datetime
            try:
                self.ec.start = datetime.fromisoformat(o.ec_start.strip())
            except ValueError:
                self.ec.start = parse_ec_date(o.ec_start)
        onset = (t_abs[0] + o.onset if o.sync_mode == "events" and self.ec.start is not None else o.onset) \
            if o.onset is not None else None
        sweep = (self.ec.start_seconds + o.sweep_start if o.sync_mode == "events" and self.ec.start is not None
                 else o.sweep_start) if o.sweep_start is not None else None
        from .spec import to_seconds
        zero = to_seconds(s.date) if s.date is not None else None
        if o.onset_point is not None and o.sync_mode == "events":
            onset_rel = float(t_abs[self._point(o.onset_point)] - t_abs[0])
            onset = onset_rel + (t_abs[0] if self.ec.start is not None else 0.0)
        if o.sync_mode == "events" and onset is not None and self.ec.start is None and zero is not None:
            onset = onset + t_abs[0] - zero            # given from the first point; axis is from #D
        if axes == "relative" and o.sync_mode == "events":         # typed event times are always from the starts
            given = (float(t_abs[self._point(o.onset_point)] - t_abs[0]) if o.onset_point is not None else o.onset)
            onset = None if given is None else given + t_abs[0] - (zero if zero is not None else t_abs[0])
            sweep = o.sweep_start
        r.alignment = sy.align(t_abs, self.ec, o.sync_mode, o.offset, r.I, onset, sweep, o.fine_shift, zero, axes)
        r.notes += r.alignment.notes + self.ec.notes
        r.t_ec = r.alignment.t_ec_of_points
        r.V, r.current = sy.potential_at_points(r.alignment, self.ec, s.point_count_times())
        r.branch, r.cycle, rate = sy.sweep_branches(r.alignment.ec_t, self.ec.potential, r.t_ec)
        no_v = ~np.isfinite(r.V)                     # counting window not inside the record: no potential
        r.branch[no_v], r.cycle[no_v] = 0, 0
        if self.ec.cycle is not None and np.nanmax(self.ec.cycle) > 0:      # the potentiostat's own cycle numbers
            j = np.clip(np.searchsorted(r.alignment.ec_t, r.t_ec), 0, self.ec.cycle.size - 1)
            sweeping = np.isfinite(r.V) & (r.branch != 0)
            cyc = np.round(self.ec.cycle[j]).astype(int)
            zero = sweeping & (cyc == 0)
            if zero.sum() > 0.05 * max(sweeping.sum(), 1):                  # numbering from 0: shift to 1
                cyc = cyc + 1
            elif zero.any():                                                 # a point at the sweep start
                cyc = np.maximum(cyc, 1)
            r.cycle = np.where(sweeping, cyc, 0)
        self.res = r
        if upto == "sync":
            return r
        # background on the relaxation period
        t_end = o.relax_end
        if t_end is None:
            t_end = self.default_relax_end()
        r.background = an.fit_background(r.t, r.I, t_end, o.background, o.correction, r.sigma)
        r.notes += r.background.notes
        r.I_corr, r.sigma_corr = r.background.correct(r.t, r.I, r.sigma)
        sweeping = r.branch != 0
        if sweeping.sum() > 20:
            et, eE = r.alignment.ec_t, self.ec.potential
            moving = np.flatnonzero(np.abs(np.gradient(eE, et)) > 1e-6)
            if moving.size > 20:
                sel = slice(moving[0], moving[-1] + 1)
                r.period_E = an.dominant_period(et[sel], eE[sel])
            if r.period_E:
                r.period_I = an.dominant_period(r.t[sweeping], r.I_corr[sweeping], 0.5 * r.period_E, 1.6 * r.period_E)
        if upto == "background":
            return r
        # I(V)
        cycles = an.parse_cycles(o.cycles, r.cycle)
        if o.average:
            r.binned = an.bin_by_potential(r.V, r.I_corr, r.sigma_corr, r.branch, r.cycle, cycles, o.bin_width)
        else:
            use = np.isin(r.cycle, cycles) & np.isfinite(r.V)
            r.binned = [an.Binned(b, r.V[use & (r.branch == b)], r.I_corr[use & (r.branch == b)],
                                  r.sigma_corr[use & (r.branch == b)], np.ones(int(np.sum(use & (r.branch == b)))))
                        for b in (1, -1) if np.any(use & (r.branch == b))]
        for b in r.binned:
            if b.V.size > 1 + int(o.slope) + 3 * o.n_transitions:
                r.fits[b.branch] = an.fit_sigmoids(b.V, b.I, b.sigma, o.n_transitions, b.branch, o.slope)
                r.notes += [f"{'anodic' if b.branch > 0 else 'cathodic'}: {n}" for n in r.fits[b.branch].notes]
        r.hysteresis = an.hysteresis(r.fits.get(1), r.fits.get(-1))
        if upto == "fit":
            return r
        self.simulate()
        return r

    def _point(self, i):
        n = self.scan.npts
        if not 0 <= int(i) < n:
            raise ValueError(f"point #{i} is not in the scan (points 0 to {n - 1})")
        return int(i)

    def manual_cv(self, t0_abs, t_points=None):
        """ECData for the hand-defined CV on the SPEC clock, so every sync mode works ('absolute' with
        offset 0). The hold starts cv_delay s after the scan start (#D, or the first point), or, with
        cv_start_point, the first sweep starts at that point's time (the middle of its count). With
        cv_end_point the CV stops at that point's time."""
        from .ec import make_cv
        from .spec import from_seconds
        o = self.opt
        sweep0 = t0_abs + o.cv_delay + o.cv_hold_s
        if o.cv_start_point is not None:
            if o.cv_start_marks not in ("sweep", "hold"):
                raise ValueError("cv_start_marks must be 'sweep' or 'hold'")
            sweep0 = float(t_points[self._point(o.cv_start_point)])
            if o.cv_start_marks == "hold":
                sweep0 += o.cv_hold_s
        stop = None
        if o.cv_end_point is not None:
            stop = float(t_points[self._point(o.cv_end_point)]) - sweep0
            if stop <= 0:
                raise ValueError("the CV end point comes before its start point")
        ec = make_cv(o.cv_hold_V, o.cv_hold_s, o.cv_upper, o.cv_lower, o.cv_rate / 1e3, int(o.cv_cycles),
                     o.cv_first, start=from_seconds(sweep0 - o.cv_hold_s), stop=stop)
        if o.cv_start_point is not None:
            ec.notes.append(f"the {'first sweep' if o.cv_start_marks == 'sweep' else 'hold'} starts at point "
                            f"#{o.cv_start_point} (the middle of its count)"
                            + (f"; stopped at point #{o.cv_end_point}" if o.cv_end_point is not None else ""))
        return ec

    def default_relax_end(self):
        """End of the relaxation window: a little before the intensity onset, or before the first
        point with a sweeping potential."""
        r = self.res
        try:
            onset = sy.detect_intensity_onset(r.t, r.I)
        except ValueError:
            onset = None
        moving = np.flatnonzero(r.branch != 0)
        sweep = r.t[moving[0]] if moving.size else None
        cands = [x for x in (onset, sweep) if x is not None]
        if not cands:
            return float(r.t[max(5, r.t.size // 5)])
        end = min(cands)
        dt = np.median(np.diff(r.t)) if r.t.size > 1 else 1.0
        return float(max(end - 2 * dt, r.t[min(4, r.t.size - 1)]))

    # ------------------------------------------------------------------ CTR model
    def sim_params(self):
        p = dict(DEFAULTS)
        if self.opt.sim_settings:
            vals, _ = read_ini(self.opt.sim_settings)
            p.update(vals)
        return p

    def point_model(self):
        key = (self.hkl(), json.dumps(self.sim_params(), sort_keys=True, default=str))
        if self._pm_key != key:
            self._pm = pr.PointModel(self.sim_params(), self.hkl())
            self._pm_key = key
        return self._pm

    def simulate(self, V=None):
        """Predicted I(V) for the composition path in the options, scaled to the data with the
        reference state (relaxation period) when data are loaded."""
        o, r = self.opt, self.res
        pm = self.point_model()
        r.point_model = pm
        states = pr.parse_states(o.states)
        E0, w = pr.parse_transitions(o.transitions, len(states) - 1)
        if V is None:
            if r.V is not None and np.any(np.isfinite(r.V)):
                V = np.linspace(np.nanmin(r.V), np.nanmax(r.V), 400)
            else:
                V = np.linspace(min(E0) - 0.3, max(E0) + 0.3, 400)
        ref = parse_comp(o.reference, "reference state")
        I_ref_model = pm.of_comp(ref)
        sim = dict(V=V, states=states, E0=E0, widths=w, ref_model=I_ref_model,
                   anodic=pr.simulate_path(pm, V, states, E0, w, o.theta, np.ones_like(V), o.cathodic_shift),
                   cathodic=pr.simulate_path(pm, V, states, E0, w, o.theta, -np.ones_like(V), o.cathodic_shift),
                   fractions=pr.path_fractions(V, states, E0, w, o.theta),
                   states_I={st: float(pm.intensity(pr.state_vector(st, o.theta))[0]) for st in states})
        r.sim = sim
        r.scale = None
        if r.I_corr is not None and r.background is not None:
            relax = r.t <= r.background.t_end
            if relax.any():
                r.scale = pr.scale_from_reference(pm, float(np.mean(r.I_corr[relax])), ref)
        return sim

    def fit_path(self):
        o, r = self.opt, self.res
        if r.I_corr is None:
            raise ValueError("run the analysis first")
        pm = self.point_model()
        states = pr.parse_states(o.states)
        E0, w = pr.parse_transitions(o.transitions, len(states) - 1)
        cycles = an.parse_cycles(o.cycles, r.cycle)
        use = np.isin(r.cycle, cycles) & np.isfinite(r.V) & (r.branch != 0)
        r.path_fit = pr.fit_path(pm, r.V[use], r.I_corr[use], r.sigma_corr[use], states, E0, w, o.theta,
                                 r.branch[use], o.fit_shift, o.fit_theta, r.scale)
        return r.path_fit

    def coverage(self):
        o, r = self.opt, self.res
        if r.I_corr is None:
            raise ValueError("run the analysis first")
        if r.scale is None:
            self.simulate()
        if r.scale is None:
            raise ValueError("no relaxation period to set the scale")
        pm = self.point_model()
        states = pr.parse_states(o.states)
        ref = parse_comp(o.reference, "reference state")
        start = 0.0
        ref_v = pr.comp_vector(ref)
        for k, st in enumerate(states):
            if np.allclose(pr.state_vector(st, o.theta), ref_v):
                start = float(k)
        r.coverage = pr.coverage_from_intensity(pm, r.I_corr / r.scale, states, o.theta, start,
                                                sigma=r.sigma_corr / r.scale)
        return r.coverage

    # ------------------------------------------------------------------ output
    def summary(self):
        o, r = self.opt, self.res
        lines = []
        if self.spec is not None:
            s = self.scan
            lines.append(f"Scan #{s.number}: {s.command}  ({s.npts} points" + (", stationary)" if s.stationary else
                                                                                 ", NOT stationary)"))
        if self.ec is not None:
            lines.append(f"Potentiostat: {self.ec.source}, {self.ec.time.size} samples, start "
                         f"{self.ec.start or 'unknown'}")
        if r.alignment is not None:
            a = r.alignment
            lines.append(f"Sync: {a.mode}, offset {a.offset:+.2f} s (EC time = SPEC time + offset)")
            if a.onset_spec is not None:
                lines.append(f"  intensity onset at SPEC {a.onset_spec - a.spec_t[0]:.1f} s, sweep start at EC "
                             f"{a.sweep_start_ec - a.ec_t[0]:.1f} s")
        if r.period_E:
            msg = f"CV repeats every {r.period_E:.1f} s"
            if r.period_I:
                d = r.period_I - r.period_E
                msg += f"; the intensity every {r.period_I:.1f} s ({d:+.1f} s"
                msg += ", check the scan rate and vertices)" if abs(d) > 0.03 * r.period_E else ", consistent)"
            lines.append(msg)
        try:
            lines.append(f"HKL: ({' '.join(f'{x:g}' for x in self.hkl())})")
        except ValueError:
            pass
        if r.background is not None and r.background.model != "none":
            b = r.background
            lines.append(f"Background: {b.model} ({b.correction}) fitted up to {b.t_end:.1f} s; parameters "
                         + ", ".join(f"{p:.4g}" for p in b.params) + f"; rms {b.rms:.3g}")
        for br, f in r.fits.items():
            lines.append(f"{'Anodic' if br > 0 else 'Cathodic'} sweep: reduced chi2 {f.red_chi2:.2f}")
            for k, t in enumerate(f.transitions):
                lines.append(f"  E{k + 1} = {t.E0:.4f} ± {t.E0_err:.4f} V, width {1e3 * t.width:.1f} ± "
                             f"{1e3 * t.width_err:.1f} mV (apparent n = {t.n_apparent:.2f}), step {t.step:.4g} ± "
                             f"{t.step_err:.2g}")
        for k, (h, e) in enumerate(r.hysteresis):
            lines.append(f"Hysteresis E{k + 1}: {1e3 * h:.1f} ± {1e3 * e:.1f} mV (anodic - cathodic)")
        if r.sim:
            st = r.sim["states_I"]
            ref = r.sim["ref_model"]
            lines.append("Model at this HKL (relative to the reference state): " +
                         ", ".join(f"{k} {v / ref:.3f}" for k, v in st.items()))
        if r.path_fit is not None:
            p = r.path_fit
            lines.append(f"Path fit ({' -> '.join(p.states)}): reduced chi2 {p.red_chi2:.2f}, scale {p.scale:.4g}")
            for k in range(len(p.E0)):
                lines.append(f"  {p.states[k]} -> {p.states[k + 1]}: E0 {p.E0[k]:.4f} ± {p.E0_err[k]:.4f} V, width "
                             f"{1e3 * p.widths[k]:.1f} ± {1e3 * p.widths_err[k]:.1f} mV")
            if o.fit_shift:
                lines.append(f"  cathodic shift {1e3 * p.cathodic_shift:.1f} ± {1e3 * p.shift_err:.1f} mV")
            if o.fit_theta:
                lines.append(f"  coverage {p.theta:.3f}")
            lines += ["  " + n for n in p.notes]
        if r.coverage is not None:
            lines += ["Coverage: " + n for n in r.coverage.notes]
        lines += ["Note: " + n for n in dict.fromkeys(r.notes)]
        return "\n".join(lines)

    def export_csv(self, path):
        r = self.res
        cols = dict(t_s=r.t, t_ec_s=r.t_ec, V=r.V, current_mA=r.current, branch=r.branch, cycle=r.cycle,
                    I_raw=r.I_raw, I=r.I, sigma=r.sigma, I_corr=r.I_corr, sigma_corr=r.sigma_corr)
        if r.coverage is not None:
            for i, s in enumerate(pr.ADSORBATE_SPECIES):
                cols[f"frac_{s}"] = r.coverage.fractions[:, i]
        cols = {k: v for k, v in cols.items() if v is not None}
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(list(cols))
            for row in zip(*cols.values()):
                w.writerow([f"{x:.6g}" if np.isfinite(x) else "" for x in row])
        return path

    def export_json(self, path):
        r = self.res
        d = dict(options=self.opt.to_dict(), summary=self.summary().splitlines())
        d["fits"] = {("anodic" if b > 0 else "cathodic"): [t.__dict__ for t in f.transitions] for b, f in r.fits.items()}
        if r.path_fit is not None:
            p = r.path_fit
            d["path_fit"] = dict(states=p.states, E0=p.E0.tolist(), E0_err=p.E0_err.tolist(), widths=p.widths.tolist(),
                                 widths_err=p.widths_err.tolist(), scale=p.scale, cathodic_shift=p.cathodic_shift,
                                 theta=p.theta, red_chi2=p.red_chi2)
        Path(path).write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")
        return path


__all__ = ["Options", "Result", "Session"]

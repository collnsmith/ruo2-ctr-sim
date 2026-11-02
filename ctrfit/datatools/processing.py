"""CTR processing after background subtraction (no Qt): steps 2 to 5 for one image per rod point.

    ROI sums files (scan, image, <ROI>, <ROI>_err) + SPEC file
      2. normalise     I_n = I / (monitor / monitor_ref) / transmission / count time
      3. correct       I_c = I_n / (P * C_rod * C_area)
      4. amplitudes    F = sqrt(I_c), sigma_F from sigma_I (ctrfit.data.io.intensities_to_F)
      5. output        every kept point as measured (nothing is averaged or merged: overlapping
                       scans of a rod and symmetry equivalents stay separate points, each with its
                       scan and rod); points near Bragg peaks left out; a systematic floor for the
                       fit; optional (off by default): scale later scans of a rod to the first on
                       their overlap

Groups: every scan has a group label (a potential, a film, ...; blank: one group). Overlap scaling
stays inside a group, and each group is its own dataset (one file per group).

Stationary scans (H K L fixed, e.g. during a CV) get steps 2 to 4 point by point and only the hand
scale: they are not overlap-scaled, and the Bragg window does not apply. They are written
as their own file (write_stationary, the ROI sums format the in-situ CV tool reads) with errors.

The image number of a ROI sums row is the SPEC point of its scan; H, K, L, monitor, transmission
and count time come from that point. A time scan (loopscan, timescan) usually has no H K L columns:
its H K L is the '#Q' position of the scan. A scan that cannot be processed (no H K L at all, a ROI
or column it lacks) is left out with a note; the others go on (an error only if every scan fails).

Corrections (stationary geometry, an area detector that sees the whole footprint and the whole rod
profile; the angles come from H K L, the energy and the incidence angle, or from SPEC columns):

* polarisation P = p (1 - (e_h . k_out)^2) + (1 - p) (1 - (e_v . k_out)^2), with p the horizontal
  fraction of the beam polarisation (synchrotron: about 0.98), e_h / e_v the horizontal / vertical
  directions perpendicular to the beam. Horizontal surface (normal vertical) or vertical surface
  (normal horizontal) sets how the surface frame sits in the lab.
* rod interception C_rod = 1 / sin(beta_out): the detector integrates the rod where it crosses the
  Ewald sphere; a rod along the surface normal crosses it over an area proportional to
  1 / (n . k_out) (C. M. Schlepuetz et al., Acta Cryst. A61, 418 (2005), J. Appl. Cryst. 44, 73
  (2011)).
* illuminated area C_area = w * min(h / sin(alpha_in), sample length) (beam width w and height h,
  sample length along the beam): more illuminated surface, more intensity.

Only ratios matter (the rods are scaled in the fit), so constant factors are left out.
"""
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from ..data.dataset import Dataset
from ..data.io import intensities_to_F, write_csv
from ..echem.external import parse_roi_sums
from .specfile import parse_spec_file

TIO2_SURFACE = (2.9587, 6.4964, 6.4964)        # a1 = c, a2 = a3 = sqrt(2) a of TiO2 (A)
GEOMETRIES = ("horizontal surface", "vertical surface")


@dataclass
class Options:
    roi: str = None                      # intensity column of the ROI sums (None: the first with an _err)
    monitor: str = None                  # SPEC columns; None: not used
    transmission: str = None
    count_time: str = None
    energy_kev: float = 16.0
    lattice: tuple = TIO2_SURFACE        # surface cell a1, a2, a3 (A), right angles
    geometry: str = GEOMETRIES[0]
    alpha_deg: float = 0.5               # incidence angle (fixed), unless alpha_column
    alpha_column: str = None
    beta_column: str = None              # exit angle from a SPEC column (else from L)
    polarization: bool = True
    pol_fraction: float = 0.98           # horizontal fraction of the beam polarisation
    rod_interception: bool = True
    area: bool = True
    beam_h_mm: float = 0.05              # beam height (in the plane of incidence)
    beam_w_mm: float = 0.5
    sample_mm: float = 5.0               # sample length along the beam
    l_tol: float = 0.01                  # H K L moving less than this: a stationary scan; overlap range padding
    match_overlaps: bool = False         # scale later scans of a rod to the first on their overlap
    exclude_bragg: float = 0.0           # leave out |L - integer| < this (0: keep all)
    floor_pct: float = 5.0               # systematic error floor for the fit (% of F)
    scan_scale: dict = field(default_factory=dict)   # scan -> extra scale factor (by hand)
    scan_use: dict = field(default_factory=dict)     # scan -> False to leave a scan out
    scan_roi: dict = field(default_factory=dict)     # scan -> its intensity column (else roi)
    scan_group: dict = field(default_factory=dict)   # scan -> group label (one dataset each)


@dataclass
class ScanResult:
    scan: int
    image: np.ndarray
    H: np.ndarray
    K: np.ndarray
    L: np.ndarray
    I_raw: np.ndarray
    s_raw: np.ndarray
    I_norm: np.ndarray
    s_norm: np.ndarray
    I_corr: np.ndarray
    s_corr: np.ndarray
    F: np.ndarray
    sF: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    P: np.ndarray
    C_rod: np.ndarray
    C_area: np.ndarray
    keep: np.ndarray                     # False: left out (Bragg window, bad angles, I not finite)
    scale: float = 1.0                   # overlap and hand scale applied to I_corr / F
    notes: list = field(default_factory=list)
    stationary: bool = False             # H K L fixed over the scan (e.g. during a CV)
    roi: str = "I"                       # the ROI sums column used

    @property
    def rod(self):
        return (int(round(float(np.nanmedian(self.H)))), int(round(float(np.nanmedian(self.K)))))


@dataclass
class Points:
    """Every kept point of the rod scans, as measured (nothing averaged), in group, rod, L order."""
    H: np.ndarray
    K: np.ndarray
    L: np.ndarray
    I: np.ndarray
    sI: np.ndarray
    F: np.ndarray
    sF: np.ndarray
    rod: list                            # 'h k' of each point's scan
    group: list                          # group label of each point
    scan: np.ndarray                     # the scan each point comes from

    def groups(self):
        return sorted(set(self.group or [""]))

    def select(self, group):
        """The points of one group (a Points of its own)."""
        g = np.array(self.group or [""] * self.L.size)
        k = g == group
        pick = lambda a: np.asarray(a)[k]
        return Points(pick(self.H), pick(self.K), pick(self.L), pick(self.I), pick(self.sI), pick(self.F),
                      pick(self.sF), [str(x) for x in pick(self.rod)], [str(x) for x in g[k]], pick(self.scan))


# ------------------------------------------------------------------ input
def load_sums(paths):
    """{scan: {column: array}} from one or several ROI sums files (later files add scans / columns)."""
    out = {}
    for p in paths:
        with open(p, errors="replace") as f:
            sums = parse_roi_sums(f.read())
        if sums is None:
            raise ValueError(f"{os.path.basename(p)} is not a ROI sums file (header: scan, image, ROIs)")
        for sn, cols in sums.items():
            out.setdefault(sn, {}).update(cols)
    return out


def sum_files(folder):
    """The ROI sums .txt files of a folder (files that are not ROI sums are skipped)."""
    out = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if name.lower().endswith(".txt") and os.path.isfile(path):
            try:
                with open(path, errors="replace") as f:
                    head = f.readline()
            except OSError:
                continue
            if head.lstrip("#").split()[:2] == ["scan", "image"]:
                out.append(path)
    return out


def roi_columns(sums):
    """(intensity columns, those with an _err column) of loaded ROI sums."""
    names, with_err = [], []
    for cols in sums.values():
        for c in cols:
            if c in ("scan", "image") or c.endswith("_err") or c in names:
                continue
            names.append(c)
            if f"{c}_err" in cols:
                with_err.append(c)
    return names, with_err


def scan_hkl(scan):
    """H, K, L at every point of a SPEC scan: its H / K / L columns, else the scan's '#Q' position
    (time scans such as loopscans have no H K L columns)."""
    q = getattr(scan, "hkl", None)
    out = []
    for i, c in enumerate("HKL"):
        col = next((scan.data[x] for x in scan.columns if x == c or x.lower() == c.lower()), None)
        if col is not None:
            out.append(np.asarray(col, float))
        elif q is not None:
            out.append(np.full(scan.npoints, float(q[i])))
        else:
            raise ValueError(f"scan {scan.scan_number}: no {c} column and no #Q line (H K L of the scan) in the "
                             "SPEC file")
    return out


def _col(scan, name):
    if not name:
        return None
    for c in scan.columns:
        if c == name or c.lower() == name.lower():
            return scan.data[c].astype(float)
    raise ValueError(f"scan {scan.scan_number} has no column '{name}'")


# ------------------------------------------------------------------ geometry and corrections
def surface_angles(H, K, L, energy_kev, lattice, alpha_deg):
    """(alpha, beta, cos psi) from H K L at a fixed incidence angle: sin(beta) = Q_perp / k - sin(alpha);
    psi is the in-plane angle between the incident and exit directions."""
    a1, a2, a3 = lattice
    k = 2 * math.pi / (12.398419843320026 / energy_kev)
    q_par = 2 * math.pi * np.sqrt((np.asarray(H, float) / a1) ** 2 + (np.asarray(K, float) / a2) ** 2)
    q_perp = 2 * math.pi * np.asarray(L, float) / a3
    alpha = np.radians(np.broadcast_to(np.asarray(alpha_deg, float), np.shape(q_perp))).copy()
    spec = q_par < 1e-9                     # specular rod: only alpha = beta reaches it
    alpha[spec] = np.arcsin(np.clip(q_perp[spec] / (2 * k), -1, 1))
    sin_b = q_perp / k - np.sin(alpha)
    beta = np.arcsin(np.clip(sin_b, -1, 1))
    beta = np.where(np.abs(sin_b) <= 1, beta, np.nan)
    ca, cb = np.cos(alpha), np.cos(beta)
    cos_psi = (ca ** 2 + cb ** 2 - (q_par / k) ** 2) / (2 * ca * cb)
    cos_psi = np.where(np.abs(cos_psi) <= 1 + 1e-9, np.clip(cos_psi, -1, 1), np.nan)
    cos_psi[spec] = 1.0
    return np.degrees(alpha), np.degrees(beta), cos_psi


def corrections(alpha_deg, beta_deg, cos_psi, opt):
    """(P, C_rod, C_area) for each point (1 where a correction is switched off)."""
    a, b = np.radians(alpha_deg), np.radians(beta_deg)
    n = np.size(a)
    ones = np.ones(n)
    # surface frame: z = normal, x = in-plane direction of the incident beam
    kin = np.stack([np.cos(a), np.zeros(n), -np.sin(a)], axis=-1)
    sin_psi = np.sqrt(np.clip(1 - cos_psi ** 2, 0, None))
    kout = np.stack([np.cos(b) * cos_psi, np.cos(b) * sin_psi, np.sin(b)], axis=-1)
    if opt.polarization:
        y = np.stack([np.zeros(n), ones, np.zeros(n)], axis=-1)          # perpendicular to beam and normal
        nz = np.stack([np.zeros(n), np.zeros(n), ones], axis=-1)
        in_plane = nz - np.sum(nz * kin, axis=-1, keepdims=True) * kin      # in the plane of incidence
        in_plane /= np.linalg.norm(in_plane, axis=-1, keepdims=True)
        if opt.geometry == GEOMETRIES[0]:      # horizontal surface: the normal is vertical
            e_h, e_v = y, in_plane
        else:                                  # vertical surface: the normal is horizontal
            e_h, e_v = in_plane, y
        p = float(opt.pol_fraction)
        P = p * (1 - np.sum(e_h * kout, axis=-1) ** 2) + (1 - p) * (1 - np.sum(e_v * kout, axis=-1) ** 2)
    else:
        P = ones
    C_rod = 1.0 / np.sin(b) if opt.rod_interception else ones
    if opt.area:
        foot = np.minimum(opt.beam_h_mm / np.sin(a), opt.sample_mm)
        C_area = opt.beam_w_mm * foot / (opt.beam_w_mm * opt.sample_mm)        # relative to the full sample
    else:
        C_area = ones
    return np.asarray(P, float), np.asarray(C_rod, float), np.asarray(C_area, float)


# ------------------------------------------------------------------ steps 2 to 4 for one scan
def process_scan(number, cols, scan, opt, monitor_ref=None):
    """ScanResult of one scan: ROI sums columns `cols`, SPEC scan `scan`."""
    names, with_err = roi_columns({number: cols})
    default = with_err[0] if with_err else (names[0] if names else None)
    notes = []
    roi = opt.scan_roi.get(number)
    if roi:
        if roi not in cols:
            raise ValueError(f"scan {number}: no ROI column '{roi}' in the ROI sums ({', '.join(names)})")
    elif opt.roi and opt.roi in cols:
        roi = opt.roi
    else:
        roi = default
        if roi is None:
            raise ValueError(f"scan {number}: no ROI column in the ROI sums")
        if opt.roi:
            notes.append(f"no ROI '{opt.roi}' in this scan: '{roi}' used (pick one in the ROI column)")
    img = cols["image"].astype(int)
    ok = (img >= 0) & (img < scan.npoints)
    if not ok.all():
        notes.append(f"{int((~ok).sum())} image(s) beyond the scan's {scan.npoints} points left out")
    img = img[ok]
    I = cols[roi][ok].astype(float)
    if f"{roi}_err" in cols:
        s = cols[f"{roi}_err"][ok].astype(float)
    else:
        s = np.sqrt(np.clip(np.abs(I), 1, None))
        notes.append(f"no '{roi}_err' column: sigma = sqrt(|I|), a lower bound (run the background "
                     "subtraction again to get proper errors)")
    H, K, L = (x[img] for x in scan_hkl(scan))
    # 2. normalise
    f = np.ones_like(I)
    if opt.monitor:
        mon = _col(scan, opt.monitor)[img]
        if np.any(mon <= 0):
            raise ValueError(f"scan {number}: monitor '{opt.monitor}' has zero or negative values")
        ref = monitor_ref if monitor_ref else float(np.mean(mon))
        f = f * ref / mon
    if opt.transmission:
        tr = _col(scan, opt.transmission)[img]
        if np.any(tr <= 0):
            raise ValueError(f"scan {number}: transmission '{opt.transmission}' has zero or negative values")
        f = f / tr
    if opt.count_time:
        ct = _col(scan, opt.count_time)[img]
        if np.any(ct <= 0):
            raise ValueError(f"scan {number}: count time '{opt.count_time}' has zero or negative values")
        f = f / ct
    I_n, s_n = I * f, s * f
    # 3. correct
    alpha = _col(scan, opt.alpha_column)[img] if opt.alpha_column else np.full(I.size, float(opt.alpha_deg))
    a_deg, b_deg, cos_psi = surface_angles(H, K, L, opt.energy_kev, opt.lattice, alpha)
    if opt.beta_column:
        b_deg = _col(scan, opt.beta_column)[img]
        a, b = np.radians(a_deg), np.radians(b_deg)
        a1, a2, _ = opt.lattice
        k = 2 * math.pi / (12.398419843320026 / opt.energy_kev)
        q_par = 2 * math.pi * np.sqrt((H / a1) ** 2 + (K / a2) ** 2)
        cos_psi = np.clip((np.cos(a) ** 2 + np.cos(b) ** 2 - (q_par / k) ** 2) / (2 * np.cos(a) * np.cos(b)), -1, 1)
    P, C_rod, C_area = corrections(a_deg, b_deg, cos_psi, opt)
    C = P * C_rod * C_area
    good = np.isfinite(C) & (C > 0) & np.isfinite(I_n) & np.isfinite(b_deg) & (b_deg > 0)
    if (~good).any():
        notes.append(f"{int((~good).sum())} point(s) with no valid geometry (exit angle <= 0 or beyond the "
                     "Ewald sphere) left out")
    with np.errstate(invalid="ignore", divide="ignore"):
        I_c, s_c = np.where(good, I_n / C, np.nan), np.where(good, s_n / C, np.nan)
    keep = good.copy()
    span = max(float(np.ptp(x)) if x.size else 0.0 for x in (H, K, L))
    stationary = I.size > 1 and span <= opt.l_tol
    if opt.exclude_bragg > 0 and not stationary:
        near = np.abs(L - np.round(L)) < opt.exclude_bragg
        keep &= ~near
    F, sF = intensities_to_F(np.nan_to_num(I_c), np.nan_to_num(s_c, nan=1.0))
    return ScanResult(number, img, H, K, L, I, s, I_n, s_n, I_c, s_c, F, sF, a_deg, b_deg, P, C_rod, C_area,
                      keep, 1.0, notes, stationary, roi)


# ------------------------------------------------------------------ step 5
def _overlap_ratio(ref, other, tol):
    """Median ratio ref / other of I_corr over L where both were measured (None if no overlap)."""
    a, b = ref, other
    ka, kb = a.keep & (a.I_corr > 0), b.keep & (b.I_corr > 0)
    if ka.sum() < 2 or kb.sum() < 2:
        return None
    La, Ia = a.L[ka] * 1.0, a.I_corr[ka] * a.scale
    Lb, Ib = b.L[kb], b.I_corr[kb]
    lo, hi = max(La.min(), Lb.min()) - tol, min(La.max(), Lb.max()) + tol
    inside = (Lb >= lo) & (Lb <= hi)
    if inside.sum() < 2:
        return None
    order = np.argsort(La)
    Ia_at = np.interp(Lb[inside], La[order], Ia[order])
    r = Ia_at / Ib[inside]
    r = r[np.isfinite(r) & (r > 0)]
    return float(np.median(r)) if r.size else None


def scale_scans(results, opt):
    """Apply the hand scales, then (match_overlaps) scale every further scan of a rod to the first
    one on their common L range. Returns notes."""
    notes = []
    for r in results:
        r.scale = float(opt.scan_scale.get(r.scan, 1.0))
    if not opt.match_overlaps:
        return notes
    by_rod = {}
    for r in results:
        if not r.stationary:
            by_rod.setdefault(_rod_key(r.rod, opt, r.scan), []).append(r)
    for (group, *rod), rs in by_rod.items():
        rod = tuple(rod)
        rs.sort(key=lambda r: r.scan)
        for r in rs[1:]:
            ratios = [x for x in (_overlap_ratio(q, r, opt.l_tol) for q in rs if q is not r and q.scan < r.scan)
                      if x]
            if ratios:
                r.scale *= float(np.median(ratios))
                notes.append(f"scan {r.scan} scaled by {r.scale:.4g} to match the overlap with rod {rod}"
                             + (f" in group '{group}'" if group else ""))
    return notes


def safe_name(text):
    """A group label as part of a file name."""
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", text).strip("_") or "group"


def scan_group(opt, scan):
    return str(opt.scan_group.get(scan, "") or "").strip()


def _rod_key(hk, opt, scan=None):
    """(group, h, k): scans of different groups never share a key."""
    return (scan_group(opt, scan),) + tuple(hk)


def points(results, opt):
    """Every kept point of the used rod scans, as measured (no averaging of overlaps or symmetry
    equivalents), with the scan's scale; None if there is none."""
    rows = []
    for r in results:
        if r.stationary or not opt.scan_use.get(r.scan, True):
            continue
        h, k = r.rod
        g = scan_group(opt, r.scan)
        for L, I, s, kp in zip(r.L, r.I_corr * r.scale, r.s_corr * r.scale, r.keep):
            if kp and np.isfinite(I) and np.isfinite(s) and s > 0:
                rows.append((g, h, k, float(L), r.scan, float(I), float(s)))
    if not rows:
        return None
    rows.sort()
    g, H, K, L, scan, I, sI = (np.array(x) for x in zip(*rows))
    F, sF = intensities_to_F(I, sI)
    return Points(H.astype(int), K.astype(int), L, I, sI, F, sF, [f"{h} {k}" for h, k in zip(H, K)],
                  [str(x) for x in g], scan.astype(int))


# ------------------------------------------------------------------ the whole chain
def process(sums, spec_scans, opt):
    """(results per scan, Points, notes) for ROI sums {scan: columns} and SPEC scans {number: SpecScan}."""
    notes, results = [], []
    missing = sorted(set(sums) - set(spec_scans))
    if missing:
        notes.append(f"scan(s) {missing} not in the SPEC file: left out")
    ref = None
    if opt.monitor:                                   # one monitor reference for every scan
        vals = []
        for sn in sums:
            if sn in spec_scans:
                try:
                    vals.append(np.mean(_col(spec_scans[sn], opt.monitor)))
                except ValueError:
                    pass
        ref = float(np.mean(vals)) if vals else None
    failed = []
    for sn in sorted(set(sums) & set(spec_scans)):
        try:
            r = process_scan(sn, sums[sn], spec_scans[sn], opt, ref)
        except ValueError as ex:                      # one scan's problem does not stop the others
            msg = str(ex)
            failed.append(msg)
            notes.append(f"scan {sn} left out: {msg[len(f'scan {sn}: '):] if msg.startswith(f'scan {sn}: ') else msg}")
            continue
        results.append(r)
        notes += [f"scan {sn}: {x}" for x in r.notes]
    if failed and not results:
        raise ValueError(failed[0] + (f" (and {len(failed) - 1} more scan(s) failed)" if len(failed) > 1 else ""))
    notes += scale_scans(results, opt)
    return results, points(results, opt) if results else None, notes


def run_files(sum_paths, spec_path, opt):
    return process(load_sums(sum_paths), parse_spec_file(spec_path), opt)


# ------------------------------------------------------------------ output
def write_scans(path, results, opt=None):
    """Every point of every scan: raw, normalised and corrected intensities, corrections, F."""
    cols = ("scan", "group", "image", "H", "K", "L", "I_raw", "sigma_raw", "I_norm", "sigma_norm", "P", "C_rod", "C_area",
            "alpha", "beta", "scale", "I_corr", "sigma_corr", "F", "sigma_F", "used")
    with open(path, "w") as f:
        f.write("# CTR processing: I_norm = I / monitor / transmission / count time, I_corr = I_norm / "
                "(P C_rod C_area) * scale, F = sqrt(I_corr)\n")
        f.write("\t".join(cols) + "\n")
        for r in results:
            for i in range(r.L.size):
                vals = (r.scan, (scan_group(opt, r.scan) if opt else "") or "-", r.image[i], r.H[i], r.K[i], r.L[i], r.I_raw[i], r.s_raw[i], r.I_norm[i], r.s_norm[i],
                        r.P[i], r.C_rod[i], r.C_area[i], r.alpha[i], r.beta[i], r.scale, r.I_corr[i] * r.scale,
                        r.s_corr[i] * r.scale, r.F[i] * math.sqrt(r.scale), r.sF[i] * math.sqrt(r.scale),
                        int(bool(r.keep[i])))
                f.write("\t".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in vals) + "\n")
    return path


def stationary(results, opt=None):
    """The stationary scans that are used."""
    use = opt.scan_use if opt is not None else {}
    return [r for r in results if r.stationary and use.get(r.scan, True)]


def write_stationary(path, results, roi=None):
    """Stationary scans in the ROI sums format (scan, image, columns), which the in-situ CV tool reads
    as an intensity file: image = SPEC point, then the corrected intensity and its error (pick it
    there with monitor, transmission and count time switched off: they are already applied), the
    normalised one, F and the corrections."""
    roi = roi or (results[0].roi if results else "I")
    c, n = f"{roi}_corr", f"{roi}_norm"
    cols = ("scan", "image", c, f"{c}_err", n, f"{n}_err", "F", "F_err", "H", "K", "L", "P", "C_rod", "C_area")
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in results:
            for i in range(r.L.size):
                if not np.isfinite(r.I_corr[i]):
                    continue
                vals = (r.scan, int(r.image[i]), r.I_corr[i] * r.scale, r.s_corr[i] * r.scale, r.I_norm[i],
                        r.s_norm[i], r.F[i] * math.sqrt(r.scale), r.sF[i] * math.sqrt(r.scale), r.H[i], r.K[i],
                        r.L[i], r.P[i], r.C_rod[i], r.C_area[i])
                f.write("\t".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in vals) + "\n")
    return path


def points_dataset(pts, opt, name="processed"):
    return Dataset(pts.H, pts.K, pts.L, pts.F, pts.sF, pts.rod, name=name,
                   sys_floor=opt.floor_pct / 100.0, energy_kev=opt.energy_kev)


def write_points(path, pts, opt, name="processed"):
    """CSV for the fitting app (H, K, L, F, sigma, rod; the systematic floor as '# sys_floor'): every
    point as measured."""
    write_csv(points_dataset(pts, opt, name), path)
    return path


def write_genx(path, pts):
    """GenX sls_sxrd style: H K L I sigma_I."""
    with open(path, "w") as f:
        f.write("# H K L I I_err\n")
        for h, k, l, i, s in zip(pts.H, pts.K, pts.L, pts.I, pts.sI):
            f.write(f"{h:d}\t{k:d}\t{l:.6g}\t{i:.6g}\t{s:.6g}\n")
    return path

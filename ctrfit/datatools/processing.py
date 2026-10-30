"""CTR processing after background subtraction (no Qt): steps 2 to 5 for one image per rod point.

    ROI sums files (scan, image, <ROI>, <ROI>_err) + SPEC file
      2. normalise     I_n = I / (monitor / monitor_ref) / transmission / count time
      3. correct       I_c = I_n / (P * C_rod * C_area)
      4. amplitudes    F = sqrt(I_c), sigma_F from sigma_I (ctrfit.data.io.intensities_to_F)
      5. merge, scale  overlapping scans of a rod scaled together, symmetry equivalents averaged
                       (R_sym), points near Bragg peaks left out; a systematic floor for the fit

Stationary scans (H K L fixed, e.g. during a CV) get steps 2 to 4 point by point and only the hand
scale: they are not merged or overlap-scaled, and the Bragg window does not apply. They are written
as their own file (write_stationary, the ROI sums format the in-situ CV tool reads) with errors.

The image number of a ROI sums row is the SPEC point of its scan; H, K, L, monitor, transmission
and count time come from that point.

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
from dataclasses import dataclass, field

import numpy as np

from ..data.dataset import Dataset
from ..data.io import intensities_to_F, write_csv
from ..echem.external import parse_roi_sums
from .specfile import parse_spec_file

TIO2_SURFACE = (2.9587, 6.4964, 6.4964)        # a1 = c, a2 = a3 = sqrt(2) a of TiO2 (A)
GEOMETRIES = ("horizontal surface", "vertical surface")
EQUIVALENTS = ("none", "±H ±K (p2mm)")


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
    equivalents: str = EQUIVALENTS[0]
    l_tol: float = 0.01                  # points of equivalent rods within this L are merged
    match_overlaps: bool = True          # scale scans of the same rod on their overlap
    exclude_bragg: float = 0.0           # leave out |L - integer| < this (0: keep all)
    floor_pct: float = 5.0               # systematic error floor for the fit (% of F)
    scan_scale: dict = field(default_factory=dict)   # scan -> extra scale factor (by hand)
    scan_use: dict = field(default_factory=dict)     # scan -> False to leave a scan out


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
class Merged:
    H: np.ndarray
    K: np.ndarray
    L: np.ndarray
    I: np.ndarray
    sI: np.ndarray
    F: np.ndarray
    sF: np.ndarray
    n: np.ndarray                        # points merged into each
    rod: list
    r_sym: float                         # sum |I - <I>| / sum I over multiply measured points (nan: none)
    n_multiple: int


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
    roi = opt.roi or (with_err[0] if with_err else (names[0] if names else None))
    if roi not in cols:
        raise ValueError(f"scan {number}: no ROI column '{roi}' in the ROI sums")
    notes = []
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
    hkl = [_col(scan, c) for c in "HKL"]
    H, K, L = (x[img] for x in hkl)
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
            by_rod.setdefault(_rod_key(r.rod, opt), []).append(r)
    for rod, rs in by_rod.items():
        rs.sort(key=lambda r: r.scan)
        for r in rs[1:]:
            ratios = [x for x in (_overlap_ratio(q, r, opt.l_tol) for q in rs if q is not r and q.scan < r.scan)
                      if x]
            if ratios:
                r.scale *= float(np.median(ratios))
                notes.append(f"scan {r.scan} scaled by {r.scale:.4g} to match the overlap with rod {rod}")
    return notes


def _rod_key(hk, opt):
    h, k = hk
    return (abs(h), abs(k)) if opt.equivalents == EQUIVALENTS[1] else (h, k)


def merge(results, opt):
    """Average the kept points of equivalent rods (and of overlapping scans) that lie within l_tol in
    L, weighted by 1/sigma^2. Returns Merged with R_sym over the points measured more than once."""
    rows = {}
    for r in results:
        if r.stationary or not opt.scan_use.get(r.scan, True):
            continue
        key = _rod_key(r.rod, opt)
        for L, I, s, k in zip(r.L, r.I_corr * r.scale, r.s_corr * r.scale, r.keep):
            if k and np.isfinite(I) and np.isfinite(s) and s > 0:
                rows.setdefault(key, []).append((L, I, s))
    if not rows:
        return None
    H, K, L, I, sI, n, rod_lab = [], [], [], [], [], [], []
    num = den = 0.0
    n_mult = 0
    for key in sorted(rows):
        pts = sorted(rows[key])
        groups, cur = [], [pts[0]]
        for p in pts[1:]:
            if p[0] - cur[0][0] <= opt.l_tol:
                cur.append(p)
            else:
                groups.append(cur)
                cur = [p]
        groups.append(cur)
        for g in groups:
            Ls, Is, ss = (np.array(x) for x in zip(*g))
            w = 1 / ss ** 2
            Im = float(np.sum(w * Is) / np.sum(w))
            H.append(key[0]); K.append(key[1]); L.append(float(np.sum(w * Ls) / np.sum(w)))
            I.append(Im); sI.append(float(1 / np.sqrt(np.sum(w)))); n.append(len(g))
            rod_lab.append(f"{key[0]} {key[1]}")
            if len(g) > 1:
                num += float(np.sum(np.abs(Is - Im)))
                den += float(np.sum(np.abs(Is)))
                n_mult += 1
    I, sI = np.array(I), np.array(sI)
    F, sF = intensities_to_F(I, sI)
    return Merged(np.array(H), np.array(K), np.array(L), I, sI, F, sF, np.array(n), rod_lab,
                  num / den if den > 0 else float("nan"), n_mult)


# ------------------------------------------------------------------ the whole chain
def process(sums, spec_scans, opt):
    """(results per scan, Merged, notes) for ROI sums {scan: columns} and SPEC scans {number: SpecScan}."""
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
    for sn in sorted(set(sums) & set(spec_scans)):
        r = process_scan(sn, sums[sn], spec_scans[sn], opt, ref)
        results.append(r)
        notes += [f"scan {sn}: {x}" for x in r.notes]
    notes += scale_scans(results, opt)
    merged = merge(results, opt) if results else None
    return results, merged, notes


def run_files(sum_paths, spec_path, opt):
    return process(load_sums(sum_paths), parse_spec_file(spec_path), opt)


# ------------------------------------------------------------------ output
def write_scans(path, results):
    """Every point of every scan: raw, normalised and corrected intensities, corrections, F."""
    cols = ("scan", "image", "H", "K", "L", "I_raw", "sigma_raw", "I_norm", "sigma_norm", "P", "C_rod", "C_area",
            "alpha", "beta", "scale", "I_corr", "sigma_corr", "F", "sigma_F", "used")
    with open(path, "w") as f:
        f.write("# CTR processing: I_norm = I / monitor / transmission / count time, I_corr = I_norm / "
                "(P C_rod C_area) * scale, F = sqrt(I_corr)\n")
        f.write("\t".join(cols) + "\n")
        for r in results:
            for i in range(r.L.size):
                vals = (r.scan, r.image[i], r.H[i], r.K[i], r.L[i], r.I_raw[i], r.s_raw[i], r.I_norm[i], r.s_norm[i],
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


def merged_dataset(merged, opt, name="processed"):
    return Dataset(merged.H, merged.K, merged.L, merged.F, merged.sF, merged.rod, name=name,
                   sys_floor=opt.floor_pct / 100.0, energy_kev=opt.energy_kev)


def write_merged(path, merged, opt, name="processed"):
    """CSV for the fitting app (H, K, L, F, sigma, rod; the systematic floor as '# sys_floor')."""
    write_csv(merged_dataset(merged, opt, name), path)
    return path


def write_genx(path, merged):
    """GenX sls_sxrd style: H K L I sigma_I."""
    with open(path, "w") as f:
        f.write("# H K L I I_err\n")
        for h, k, l, i, s in zip(merged.H, merged.K, merged.L, merged.I, merged.sI):
            f.write(f"{h:d}\t{k:d}\t{l:.6g}\t{i:.6g}\t{s:.6g}\n")
    return path

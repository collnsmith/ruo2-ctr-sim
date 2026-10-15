"""Area-detector images and ROIs for background subtraction (from sxrd_bgsub, no Qt).

Files are <prefix>_<scan:3>_<image:3>_data_<n>.h5 (Eiger, needs h5py) or .tif (needs tifffile).
ROI / ROIConfig hold signal and background boxes, per-image overrides and offsets; BG_METHODS
maps a name to subtract(img, signal_rois, bg_rois, mask) -> (corrected image, per-ROI sums).
"""
import os
import re
from collections import OrderedDict

import numpy as np


try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import tifffile
    HAS_TIFF = True
except ImportError:
    HAS_TIFF = False

FILE_PATTERN = re.compile(
    r'^(.+?)_(\d{3})_(\d{3})_(data_\d+|master)\.(h5|tif|tiff)$'
)


def parse_filename(path):
    name = os.path.basename(path)
    m = FILE_PATTERN.match(name)
    if not m:
        return None
    return {
        'prefix': m.group(1),
        'scan': int(m.group(2)),
        'image': int(m.group(3)),
        'suffix': m.group(4),
        'ext': m.group(5),
        'path': path,
    }


def scan_directory(dirpath, ext_filter=None):
    entries = []
    for f in os.listdir(dirpath):
        full = os.path.join(dirpath, f)
        if not os.path.isfile(full):
            continue
        info = parse_filename(full)
        if info is None:
            continue
        if 'master' in info['suffix']:
            continue
        if ext_filter and info['ext'] not in ext_filter:
            continue
        entries.append(info)
    entries.sort(key=lambda x: (x['scan'], x['image']))
    return entries


def load_image(filepath, ext):
    if ext in ('h5',):
        if not HAS_H5PY:
            raise ImportError("h5py required for HDF5 files")
        with h5py.File(filepath, 'r') as f:
            for path in ['entry/data/data', 'entry/instrument/detector/data', 'data']:
                if path in f:
                    ds = f[path]
                    if ds.ndim == 3:
                        return np.array(ds[0], dtype=np.float64)
                    elif ds.ndim == 2:
                        return np.array(ds, dtype=np.float64)
            raise KeyError(f"No image dataset found in {filepath}")
    elif ext in ('tif', 'tiff'):
        if not HAS_TIFF:
            raise ImportError("tifffile required for TIF files")
        img = tifffile.imread(filepath)
        if img.ndim == 3:
            img = img[0]
        return img.astype(np.float64)
    else:
        raise ValueError(f"Unknown extension: {ext}")


class ROI:
    def __init__(self, x=0, y=0, w=50, h=50, roi_type='signal', name='ROI'):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.roi_type = roi_type
        self.name = name
        # Constant-offset following: this ROI rigidly follows another ROI.
        # follow_id is the *list index* of the master ROI (resolved at apply time
        # against the current ROI list). offset_dx/dy is THIS ROI's center
        # position relative to the master's center. None = independent.
        self.follow_id = None
        self.offset_dx = 0
        self.offset_dy = 0

    def to_tuple(self):
        return (self.x, self.y, self.w, self.h)

    def to_dict(self):
        return {'x': self.x, 'y': self.y, 'w': self.w, 'h': self.h,
                'type': self.roi_type, 'name': self.name,
                'follow_id': self.follow_id,
                'offset_dx': self.offset_dx, 'offset_dy': self.offset_dy}

    @staticmethod
    def from_dict(d):
        r = ROI(d['x'], d['y'], d['w'], d['h'], d.get('type', 'signal'), d.get('name', 'ROI'))
        r.follow_id = d.get('follow_id', None)
        r.offset_dx = d.get('offset_dx', 0)
        r.offset_dy = d.get('offset_dy', 0)
        return r

    def extract(self, img):
        h_img, w_img = img.shape[:2]
        x1 = max(0, self.x)
        y1 = max(0, self.y)
        x2 = min(w_img, self.x + self.w)
        y2 = min(h_img, self.y + self.h)
        if x2 <= x1 or y2 <= y1:
            return np.array([])
        return img[y1:y2, x1:x2]

    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)

    def set_center(self, cx, cy):
        self.x = int(cx) - self.w // 2
        self.y = int(cy) - self.h // 2

    def area(self):
        return self.w * self.h

    def copy(self):
        r = ROI(self.x, self.y, self.w, self.h, self.roi_type, self.name)
        r.follow_id = self.follow_id
        r.offset_dx = self.offset_dx
        r.offset_dy = self.offset_dy
        return r

    def __eq__(self, other):
        if not isinstance(other, ROI):
            return False
        return (self.x == other.x and self.y == other.y and
                self.w == other.w and self.h == other.h and
                self.roi_type == other.roi_type and self.name == other.name)

    def __hash__(self):
        return hash((self.x, self.y, self.w, self.h, self.roi_type, self.name))


class ROIConfig:
    def __init__(self):
        self.default_rois = []
        self.overrides = {}

    def get_rois(self, scan, image):
        key = (scan, image)
        if key in self.overrides:
            return [r.copy() for r in self.overrides[key]]
        return [r.copy() for r in self.default_rois]

    def has_override(self, scan, image):
        return (scan, image) in self.overrides

    def set_default(self, rois):
        self.default_rois = [r.copy() for r in rois]

    def set_override(self, scan, image, rois):
        key = (scan, image)
        default_copy = [r.copy() for r in self.default_rois]
        if len(rois) == len(default_copy) and all(a == b for a, b in zip(rois, default_copy)):
            self.overrides.pop(key, None)
        else:
            self.overrides[key] = [r.copy() for r in rois]

    def clear_override(self, scan, image):
        self.overrides.pop((scan, image), None)

    def save(self, filepath):
        roi_defs = OrderedDict()
        counter = [0]

        def get_id(roi):
            key = (roi.x, roi.y, roi.w, roi.h, roi.roi_type, roi.name,
                   roi.follow_id, roi.offset_dx, roi.offset_dy)
            if key not in roi_defs:
                roi_defs[key] = counter[0]
                counter[0] += 1
            return roi_defs[key]

        default_ids = [get_id(r) for r in self.default_rois]
        override_ids = {}
        for (scan, img), rois in self.overrides.items():
            override_ids[(scan, img)] = [get_id(r) for r in rois]

        with open(filepath, 'w') as f:
            f.write("# SXRD ROI Configuration\n")
            f.write("# ROI_DEF: id, x, y, w, h, type, name, follow_id, offset_dx, offset_dy\n")
            for (x, y, w, h, rtype, name, follow_id, off_dx, off_dy), rid in roi_defs.items():
                follow_str = "-" if follow_id is None else str(follow_id)
                f.write(f"ROI_DEF, {rid}, {x}, {y}, {w}, {h}, {rtype}, {name}, "
                        f"{follow_str}, {off_dx}, {off_dy}\n")
            f.write(f"DEFAULT, {', '.join(str(i) for i in default_ids)}\n")
            for (scan, img), ids in sorted(override_ids.items()):
                f.write(f"OVERRIDE, {scan:03d}, {img:03d}, {', '.join(str(i) for i in ids)}\n")

    def load(self, filepath):
        roi_defs = {}
        self.default_rois = []
        self.overrides = {}

        # Detect format from header (or fall back to length-based detection):
        #   New: id, x, y, w, h, type, name, follow_id, offset_dx, offset_dy   (11 cols incl ROI_DEF)
        #   Old: id, x, y, w, h, type, name, tracking, track_method, track_margin (11 cols incl ROI_DEF)
        # Both have 11 fields. Distinguish by content of field [9]:
        #   - integer or "-"  -> new format (follow_id)
        #   - non-numeric tracking method name -> old format

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip() for p in line.split(',')]
                if parts[0] == 'ROI_DEF':
                    rid = int(parts[1])
                    r = ROI(
                        x=int(parts[2]), y=int(parts[3]),
                        w=int(parts[4]), h=int(parts[5]),
                        roi_type=parts[6],
                        name=parts[7] if len(parts) > 7 else f'ROI{rid}'
                    )
                    if len(parts) > 8:
                        # Decide: new (follow_id) vs old (tracking flag) format
                        f8 = parts[8]
                        # Old format had tracking method name like "Cross-correlation"
                        # in parts[9]. New format has integer or "-" in parts[8].
                        is_new = False
                        if f8 == '-':
                            is_new = True
                        else:
                            try:
                                int(f8)
                                # Could be new (follow_id) or old (tracking 0/1).
                                # Old had method name (string) in parts[9].
                                if len(parts) > 9:
                                    try:
                                        int(parts[9])  # offset_dx is int
                                        is_new = True
                                    except ValueError:
                                        is_new = False  # old format
                                else:
                                    is_new = True
                            except ValueError:
                                is_new = False
                        if is_new:
                            r.follow_id = None if f8 == '-' else int(f8)
                            if len(parts) > 9:
                                try: r.offset_dx = int(parts[9])
                                except ValueError: r.offset_dx = 0
                            if len(parts) > 10:
                                try: r.offset_dy = int(parts[10])
                                except ValueError: r.offset_dy = 0
                        # else: old tracking fields - silently dropped
                    roi_defs[rid] = r
                elif parts[0] == 'DEFAULT':
                    ids = [int(p) for p in parts[1:] if p]
                    self.default_rois = [roi_defs[i].copy() for i in ids]
                elif parts[0] == 'OVERRIDE':
                    scan = int(parts[1])
                    img = int(parts[2])
                    ids = [int(p) for p in parts[3:] if p]
                    self.overrides[(scan, img)] = [roi_defs[i].copy() for i in ids]


def subtract_bg_average(img, signal_rois, bg_rois, mask=None):
    result = img.copy()
    if mask is not None:
        result[mask] = 0
    bg_pixels = []
    for roi in bg_rois:
        region = roi.extract(result)
        if region.size > 0:
            if mask is not None:
                m = roi.extract(mask)
                if m.size == region.size:
                    bg_pixels.append(region[~m].ravel())
                else:
                    bg_pixels.append(region.ravel())
            else:
                bg_pixels.append(region.ravel())
    if not bg_pixels:
        return result
    all_bg = np.concatenate(bg_pixels)
    if all_bg.size == 0:
        return result
    result -= np.mean(all_bg)
    return result


def subtract_bg_median(img, signal_rois, bg_rois, mask=None):
    result = img.copy()
    if mask is not None:
        result[mask] = 0
    bg_pixels = []
    for roi in bg_rois:
        region = roi.extract(result)
        if region.size > 0:
            if mask is not None:
                m = roi.extract(mask)
                if m.size == region.size:
                    bg_pixels.append(region[~m].ravel())
                else:
                    bg_pixels.append(region.ravel())
            else:
                bg_pixels.append(region.ravel())
    if not bg_pixels:
        return result
    all_bg = np.concatenate(bg_pixels)
    if all_bg.size == 0:
        return result
    result -= np.median(all_bg)
    return result


def subtract_bg_polyfit(img, signal_rois, bg_rois, mask=None, order=2):
    result = img.copy()
    if mask is not None:
        result[mask] = 0
    coords_y, coords_x, values = [], [], []
    for roi in bg_rois:
        y1 = max(0, roi.y)
        x1 = max(0, roi.x)
        y2 = min(img.shape[0], roi.y + roi.h)
        x2 = min(img.shape[1], roi.x + roi.w)
        if y2 <= y1 or x2 <= x1:
            continue
        yy, xx = np.mgrid[y1:y2, x1:x2]
        region = img[y1:y2, x1:x2]
        if mask is not None:
            m = mask[y1:y2, x1:x2]
            coords_y.append(yy[~m].ravel())
            coords_x.append(xx[~m].ravel())
            values.append(region[~m].ravel())
        else:
            coords_y.append(yy.ravel())
            coords_x.append(xx.ravel())
            values.append(region.ravel())
    if not values:
        return result
    cy = np.concatenate(coords_y)
    cx = np.concatenate(coords_x)
    vals = np.concatenate(values)
    if len(vals) < (order + 1) ** 2:
        return subtract_bg_average(img, signal_rois, bg_rois, mask)
    cols = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            cols.append((cy ** i) * (cx ** j))
    A = np.column_stack(cols)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
    except np.linalg.LinAlgError:
        return subtract_bg_average(img, signal_rois, bg_rois, mask)
    yy_full, xx_full = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    bg_surface = np.zeros_like(img)
    idx = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            bg_surface += coeffs[idx] * (yy_full ** i) * (xx_full ** j)
            idx += 1
    result = img - bg_surface
    if mask is not None:
        result[mask] = 0
    return result


BG_METHODS = {
    'Average': subtract_bg_average,
    'Median': subtract_bg_median,
    'Polynomial': subtract_bg_polyfit,
}


def apply_offsets(rois):
    """For every ROI with follow_id set, reposition it so that:
        follower.center = master.center + (offset_dx, offset_dy)

    follow_id is the index into the rois list. If the index is out of
    range or refers to the ROI itself, the follower is left in place.
    Master ROIs (those without follow_id) are not touched - they should
    have been moved by the user already (manually or through the canvas).
    """
    n = len(rois)
    for i, roi in enumerate(rois):
        if roi.follow_id is None:
            continue
        mid = roi.follow_id
        if not (0 <= mid < n) or mid == i:
            continue
        master = rois[mid]
        mcx, mcy = master.center()
        new_cx = mcx + roi.offset_dx
        new_cy = mcy + roi.offset_dy
        roi.set_center(new_cx, new_cy)


def lock_offset(roi, master):
    """Capture the current relative position of roi vs master and store
    it as roi.offset_dx/dy. Sets roi.follow_id only if caller does so."""
    fcx, fcy = roi.center()
    mcx, mcy = master.center()
    roi.offset_dx = fcx - mcx
    roi.offset_dy = fcy - mcy

"""Area-detector images and ROIs for background subtraction (from sxrd_bgsub, no Qt).

Files are <prefix>_<scan:3>_<image:3>_data_<n>.h5 (Eiger, needs h5py) or .tif (needs tifffile).
ROI / ROIConfig hold signal and background boxes, per-image overrides and offsets; BG_METHODS
maps a name to subtract(img, signal_rois, bg_rois, mask) -> (corrected image, per-ROI sums).
"""
import functools
import os
import threading
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

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
    # surface sum_ij c_ij y^i x^j = Y C X^T (separable: no full-size coordinate grids)
    C = np.zeros((order + 1, order + 1))
    idx = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            C[i, j] = coeffs[idx]
            idx += 1
    Y = np.arange(img.shape[0], dtype=float)[:, None] ** np.arange(order + 1)
    X = np.arange(img.shape[1], dtype=float)[:, None] ** np.arange(order + 1)
    result = img - (Y @ C) @ X.T
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



def apply_rois(config, keys, rois, keep_masters=False, set_default=False):
    """Give every image in `keys` ((scan, image) pairs) a copy of `rois`. By default the copy is exact:
    the images get these positions. With keep_masters, each image keeps the position of its own master
    ROIs (those that follow no other ROI, e.g. a per-image tracked signal) and only the followers are
    placed again from their offsets. Other images are not touched. set_default also makes `rois` the
    defaults (for images without their own ROIs). Returns the number of images changed."""
    n = 0
    for key in keys:
        new = [r.copy() for r in rois]
        if keep_masters:
            existing = config.overrides.get(key)
            if existing is not None:
                for i, r in enumerate(new):
                    if r.follow_id is None and i < len(existing):
                        r.x, r.y = existing[i].x, existing[i].y
            apply_offsets(new)
        config.overrides[key] = new
        n += 1
    if set_default:
        config.set_default(rois)
    return n

# ============================================================
# DISPLAY (shared by the image view and the batch PNGs)
# ============================================================

@functools.lru_cache(maxsize=None)
def _lut(name):
    x = np.linspace(0, 1, 256)
    if name == 'hot':
        rgb = (np.clip(x * 3, 0, 1), np.clip(x * 3 - 1, 0, 1), np.clip(x * 3 - 2, 0, 1))
    elif name == 'viridis':      # simplified viridis approximation
        rgb = (np.clip(0.267004 + x * (0.282327 + x * (-1.964 + x * 1.596)), 0, 1),
               np.clip(0.004874 + x * (1.015 + x * (-0.474 + x * 0.153)), 0, 1),
               np.clip(0.329415 + x * (1.406 + x * (-3.192 + x * 1.820)), 0, 1))
    elif name == 'inferno':
        rgb = (np.clip(x * 2.8 - 0.15, 0, 1), np.clip(x * 2.0 - 0.6, 0, 1), np.clip(0.45 + x * 1.2 - x * x * 2.8, 0, 1))
    elif name == 'coolwarm':
        rgb = (np.clip(0.23 + x * 1.5, 0, 1), np.clip(0.3 + 2.0 * (0.5 - np.abs(x - 0.5)), 0, 1), np.clip(1.0 - x * 1.5, 0, 1))
    elif name == 'jet':
        rgb = (np.clip(1.5 - np.abs(x - 0.75) * 4, 0, 1), np.clip(1.5 - np.abs(x - 0.5) * 4, 0, 1),
               np.clip(1.5 - np.abs(x - 0.25) * 4, 0, 1))
    else:                        # gray
        rgb = (x, x, x)
    lut = np.stack([(c * 255).astype(np.uint8) for c in rgb], axis=1)
    lut.setflags(write=False)
    return lut


def colormap_lut(name):
    """256 x 3 uint8 lookup table of a colormap (gray, hot, viridis, inferno, coolwarm, jet)."""
    return _lut(name)


AUTO_LIMIT_SAMPLES = 200_000


def display_limits(img, vmin=None, vmax=None, log=True, exact=False):
    """Colour limits in display units: given limits (counts) go through the log like the image; missing
    ones are the 1 % and 99.5 % percentiles of the positive pixels. Unless exact, large images use an
    even subsample for the percentiles (the same limits to a fraction of a percent, several times faster)."""
    if vmin is not None:
        vmin = np.log10(vmin + 1) if log and vmin > 0 else vmin
    if vmax is not None:
        vmax = np.log10(vmax + 1) if log and vmax > 0 else vmax
    if vmin is None or vmax is None:
        flat = img.ravel()
        if not exact and flat.size > AUTO_LIMIT_SAMPLES:
            flat = flat[::flat.size // AUTO_LIMIT_SAMPLES + 1]
        pos = flat[flat > 0]
        if pos.size == 0:
            pos = flat
        if pos.size:
            lo, hi = np.percentile(pos, [1, 99.5])
        else:
            lo, hi = 0.0, 1.0
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    return float(vmin), float(vmax)


def display_rgb(img, log=True, vmin=None, vmax=None, cmap='gray', exact=False):
    """(H, W, 3) uint8 image as the viewer shows it: log10(I + 1) of the positive part (if log), scaled
    between the limits (see display_limits), through the colormap."""
    data = np.asarray(img, dtype=np.float64 if exact else np.float32)     # exact: the batch PNGs, as before
    if log:
        data = np.log10(np.fmax(data, 0) + 1)          # <= 0 and NaN -> 0
    lo, hi = display_limits(data, vmin, vmax, log, exact)
    if exact:
        idx = np.clip((data - lo) / (hi - lo + 1e-10) * 255, 0, 255).astype(np.uint8)
    else:
        idx = np.clip((data - lo) * np.float32(255.0 / (hi - lo + 1e-10)), 0, 255).astype(np.uint8)
    return colormap_lut(cmap)[idx]


# ============================================================
# IMAGE CACHE (slow or busy disks: keep recent images, read ahead)
# ============================================================

class ImageCache:
    """Recently used images in memory, and read-ahead in background threads, so stepping through a
    scan does not wait on the disk (network drives, a transfer running on the same disk). Entries
    are keyed by path, size and modification time, so a file that is still being written is read
    again once it changes. Images are returned read-only; copy before changing one."""

    def __init__(self, maxsize=24, workers=2):
        self.maxsize = maxsize
        self._items = OrderedDict()            # key -> array
        self._pending = {}                     # key -> Future
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="image-read")

    @staticmethod
    def _key(path):
        try:
            st = os.stat(path)
            return (str(path), st.st_size, st.st_mtime_ns)
        except OSError:
            return (str(path), None, None)

    @staticmethod
    def _read(path, ext):
        img = load_image(path, ext)
        img.setflags(write=False)
        return img

    def _store(self, key, img):
        with self._lock:
            self._items[key] = img
            self._items.move_to_end(key)
            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)

    def get(self, path, ext):
        """The image (from memory, a read already under way, or the disk)."""
        key = self._key(path)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            fut = self._pending.pop(key, None)
        img = fut.result() if fut is not None else self._read(path, ext)
        self._store(key, img)
        return img

    def prefetch(self, files):
        """Start reading these ({'path', 'ext'} dicts) in the background; errors are ignored here and
        raised again by get()."""
        for f in files:
            key = self._key(f['path'])
            with self._lock:
                if key in self._items or key in self._pending:
                    continue
                self._pending[key] = self._pool.submit(self._read, f['path'], f['ext'])
            with self._lock:
                while len(self._pending) > 8:                         # drop the oldest unclaimed reads
                    self._pending.pop(next(iter(self._pending))).cancel()

    def clear(self):
        with self._lock:
            self._items.clear()
            for fut in self._pending.values():
                fut.cancel()
            self._pending.clear()


# ============================================================
# ONE IMAGE OF THE BATCH
# ============================================================

def roi_sigma(img, roi, bg_rois, mask=None, method=None):
    """Counting error of a background-subtracted signal ROI sum, from the raw counts (the Eiger counts
    photons: Poisson per pixel, no read-out noise): sigma^2 = S + (n_s / n_b)^2 * B * f, with S the
    raw counts and n_s the pixels of the signal ROI, B the raw counts and n_b the pixels of the
    background ROIs, and f = pi/2 for a median background (1 for the average; used as an estimate for
    the polynomial surface). Masked pixels are left out."""
    def pixels(r):
        region = r.extract(img)
        if region.size == 0:
            return np.array([])
        if mask is not None:
            m = r.extract(mask)
            if m.size == region.size:
                return region[~m]
        return region.ravel()
    s = pixels(roi)
    S, n_s = float(np.sum(np.clip(s, 0, None))), s.size
    bg = [pixels(r) for r in bg_rois]
    bg = np.concatenate(bg) if bg else np.array([])
    if bg.size == 0 or n_s == 0:
        return float(np.sqrt(max(S, 1.0)))
    f = np.pi / 2 if method is subtract_bg_median else 1.0
    B, n_b = float(np.sum(np.clip(bg, 0, None))), bg.size
    return float(np.sqrt(max(S + (n_s / n_b) ** 2 * B * f, 1.0)))


ROI_SECTION = "# [ROI positions]"


def roi_positions_section(entries):
    """Lines of the ROI positions section of a batch sums file, from [(scan, image, rois)] as used.
    Every line starts with '#', so readers of the sums table skip it. A ROI that stays put over a
    scan gets one row with image 'all'; one that moves (per-image overrides, tracking, following)
    gets a row for every image."""
    by = {}
    for scan, image, rois in entries:
        for r in rois:
            by.setdefault((int(scan), r.name), []).append((int(image), (r.roi_type, r.x, r.y, r.w, r.h)))
    lines = [ROI_SECTION + " x, y: top-left pixel; w, h: size in pixels; image 'all': the same in "
             "every image of the scan",
             "# scan\timage\troi\ttype\tx\ty\tw\th"]
    for (scan, name), rows in sorted(by.items()):
        rows.sort()
        if len({p for _, p in rows}) == 1:
            rows = [("all", rows[0][1])]
        for image, (kind, x, y, w, h) in rows:
            im = image if image == "all" else f"{image:03d}"
            lines.append(f"# {scan:03d}\t{im}\t{name}\t{kind}\t{x}\t{y}\t{w}\t{h}")
    return lines


def read_roi_positions(path):
    """{(scan, image or 'all', roi name): (type, x, y, w, h)} from the ROI positions section of a sums
    file ({} if it has none)."""
    out, inside = {}, False
    with open(path, errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith(ROI_SECTION):
                inside = True
                continue
            if not inside or not s.startswith("#"):
                continue
            parts = s.lstrip("#").strip().split("\t")
            if len(parts) != 8 or parts[0] == "scan":
                continue
            try:
                key = (int(parts[0]), parts[1] if parts[1] == "all" else int(parts[1]), parts[2])
                out[key] = (parts[3],) + tuple(int(v) for v in parts[4:])
            except ValueError:
                continue
    return out


def roi_position(positions, scan, image, name):
    """(type, x, y, w, h) of a ROI at one image from read_roi_positions, or None."""
    return positions.get((scan, image, name)) or positions.get((scan, "all", name))


def process_image(img, rois, method, mask=None, clip_negative=False):
    """Background-subtracted image and ROI sums {name: sum} as the batch writes them: masked pixels
    are 0 and left out of the sums. Every signal ROI also gets '<name>_err', its counting error
    (roi_sigma)."""
    sig = [r for r in rois if r.roi_type == 'signal']
    bg = [r for r in rois if r.roi_type == 'background']
    sub = method(img, sig, bg, mask) if sig and bg else np.array(img, dtype=float, copy=True)
    if mask is not None:
        sub[mask] = 0
    if clip_negative:
        sub = np.clip(sub, 0, None)
    sums = {}
    for r in rois:
        region = r.extract(sub)
        if region.size > 0:
            if mask is not None:
                m = r.extract(mask)
                sums[r.name] = float(np.sum(region[~m])) if m.size == region.size else float(np.sum(region))
            else:
                sums[r.name] = float(np.sum(region))
        else:
            sums[r.name] = 0.0
    for r in sig:
        sums[f"{r.name}_err"] = roi_sigma(img, r, bg, mask, method)
    return sub, sums


# ============================================================
# UNDO / REDO OF ROI AND MASK EDITS
# ============================================================

class EditState:
    """A snapshot of what the user edits: the default ROIs, the per-image overrides, the ROIs of the
    image on screen and the mask (copies)."""

    def __init__(self, config, current_rois, mask):
        self.defaults = [r.copy() for r in config.default_rois]
        self.overrides = {k: [r.copy() for r in v] for k, v in config.overrides.items()}
        self.current = [r.copy() for r in current_rois]
        self.mask = None if mask is None else mask.copy()
        self.signature = (tuple(tuple(r.to_dict().items()) for r in self.defaults),
                          tuple((k, tuple(tuple(r.to_dict().items()) for r in v)) for k, v in sorted(self.overrides.items())),
                          tuple(tuple(r.to_dict().items()) for r in self.current),
                          None if mask is None else (mask.shape, hash(np.packbits(mask).tobytes())))

    def apply_to(self, config):
        """Put the ROIs back into the config; returns (ROIs of the image on screen, mask)."""
        config.default_rois = [r.copy() for r in self.defaults]
        config.overrides = {k: [r.copy() for r in v] for k, v in self.overrides.items()}
        return [r.copy() for r in self.current], (None if self.mask is None else self.mask.copy())


class EditHistory:
    """Undo and redo stacks of EditState snapshots. checkpoint() before every change (repeats of the
    same state are kept once); undo(current) / redo(current) return the state to restore or None."""

    def __init__(self, limit=100):
        self.limit = limit
        self.undo_stack, self.redo_stack = [], []

    def checkpoint(self, state):
        if self.undo_stack and self.undo_stack[-1].signature == state.signature:
            return
        self.undo_stack.append(state)
        del self.undo_stack[:-self.limit]
        self.redo_stack.clear()

    def _step(self, src, dst, current):
        while src and src[-1].signature == current.signature:      # nothing changed since that checkpoint
            src.pop()
        if not src:
            return None
        dst.append(current)
        return src.pop()

    def undo(self, current):
        return self._step(self.undo_stack, self.redo_stack, current)

    def redo(self, current):
        return self._step(self.redo_stack, self.undo_stack, current)

    @property
    def can_undo(self):
        return bool(self.undo_stack)

    @property
    def can_redo(self):
        return bool(self.redo_stack)

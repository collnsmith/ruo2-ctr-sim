"""Background ROI finder statistics (from sxrd_bg_finder, no Qt): per-pixel statistics over a
scan's images (on the GPU with CuPy when available), a mask of persistent bright spots, and
background boxes placed near a clicked signal centre, with a fallback search."""
import sys

import numpy as np


try:
    import cupy as cp
    HAS_CUPY = True
    try:
        cp.cuda.Device(0).compute_capability
        GPU_NAME = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    except Exception:
        HAS_CUPY = False
        GPU_NAME = None
except ImportError:
    HAS_CUPY = False
    GPU_NAME = None

def _gpu_free_bytes():
    if not HAS_CUPY: return None
    try:
        free, _ = cp.cuda.runtime.memGetInfo()
        return free
    except Exception:
        return None


def compute_stats(images, use_gpu=True, percentile=20):
    """Compute per-pixel statistics for a stack of images (N, H, W)."""
    if len(images) == 0:
        return None

    arr_cpu = np.stack(images, axis=0).astype(np.float32)
    n, h, w = arr_cpu.shape

    backend = 'numpy'
    if use_gpu and HAS_CUPY:
        # Check memory
        need = arr_cpu.nbytes * 4  # stack + intermediates
        free = _gpu_free_bytes() or 0
        if need < free * 0.7:
            backend = 'cupy'

    if backend == 'cupy':
        try:
            arr = cp.asarray(arr_cpu)
            median = cp.asnumpy(cp.median(arr, axis=0))
            pct_low = cp.asnumpy(cp.percentile(arr, percentile, axis=0))
            pct_high = cp.asnumpy(cp.percentile(arr, 100 - percentile, axis=0))
            med_gpu = cp.median(arr, axis=0)
            mad = cp.asnumpy(cp.median(cp.abs(arr - med_gpu[None, :, :]), axis=0))
            mx = cp.asnumpy(arr.max(axis=0))
            del arr, med_gpu
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            print(f"GPU failed: {e}, fallback CPU", file=sys.stderr)
            backend = 'numpy'

    if backend == 'numpy':
        median = np.median(arr_cpu, axis=0)
        pct_low = np.percentile(arr_cpu, percentile, axis=0)
        pct_high = np.percentile(arr_cpu, 100 - percentile, axis=0)
        mad = np.median(np.abs(arr_cpu - median[None, :, :]), axis=0)
        mx = arr_cpu.max(axis=0)

    return {
        'median': median,
        'percentile_low': pct_low,
        'percentile_high': pct_high,
        'mad': mad,
        'max': mx,
        'n': n,
        'shape': (h, w),
        'backend': backend,
    }


def make_exclusion_mask(stats, hot_cap_percentile=90):
    """Build a boolean mask of pixels to exclude from BG ROI placement.

    Excludes 'stationary relative bright spots' - pixels with:
        - low MAD (stable across scan) AND
        - high median (persistently bright)
    These are powder rings, hot pixels, parasitic peaks.

    Returns exclude_mask (True = exclude).
    """
    mad = stats['mad']
    med = stats['median']

    # Threshold: pixels stable in time
    mad_thresh = float(np.percentile(mad, 50))  # median MAD
    stable = mad <= mad_thresh

    # Threshold: pixels with high intensity
    # Use a high percentile so only bright ones are flagged
    intensity_cap = float(np.percentile(med, hot_cap_percentile))
    bright = med >= intensity_cap

    # Exclude = stable AND bright (powder rings, hot pixels)
    exclude = stable & bright

    # Also exclude pixels that are EVER very bright even if unstable
    # (saturated detector artifacts) - use max > very high percentile
    abs_cap = float(np.percentile(stats['max'], 99))
    exclude |= (stats['max'] >= abs_cap)

    return exclude


def find_bg_rois_near_point(stats, signal_xy, exclude_mask,
                            search_radius=150, signal_exclude_radius=30,
                            box_size=30, n_boxes=4, min_fill=0.7):
    """Place background ROIs in a ring around the signal center.

    Uses octant-based placement: tries to put one box in each of 8 octants
    around the signal, at distances in [signal_exclude_radius, search_radius].

    Parameters:
        stats: from compute_stats
        signal_xy: (x, y) pixel coords of signal center
        exclude_mask: bool mask (True = exclude this pixel)
        search_radius: max distance from signal to place BG
        signal_exclude_radius: min distance from signal
        box_size: side length of each BG ROI (pixels)
        n_boxes: how many boxes to try to place (max 8 octants)
        min_fill: minimum fraction of box pixels that must be non-excluded

    Returns list of (x, y, w, h) bounding boxes.
    """
    sx, sy = signal_xy
    h_img, w_img = stats['shape']
    med = stats['median']

    # Candidate positions: uniformly spaced around the signal in a ring
    # Try multiple radii within the allowed range
    radii = np.linspace(signal_exclude_radius + box_size // 2,
                        search_radius, 4)

    placed = []
    # For each octant, find the best box (lowest median intensity, good fill)
    n_angles = max(n_boxes, 8)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    # Rank candidate boxes by how clean they are
    candidates = []
    for angle in angles:
        for radius in radii:
            cx = int(sx + radius * np.cos(angle))
            cy = int(sy + radius * np.sin(angle))
            x = cx - box_size // 2
            y = cy - box_size // 2
            # Bounds check
            if x < 0 or y < 0 or x + box_size > w_img or y + box_size > h_img:
                continue
            # Check exclusion fill
            patch_excl = exclude_mask[y:y + box_size, x:x + box_size]
            fill = 1.0 - patch_excl.mean()
            if fill < min_fill:
                continue
            # Score: prefer lower median (darker = more likely true background)
            patch_med = med[y:y + box_size, x:x + box_size]
            # Only count non-excluded pixels for score
            valid = ~patch_excl
            if not valid.any(): continue
            score = float(patch_med[valid].mean())
            candidates.append((score, angle, x, y))

    # Sort by score (lowest first = darkest = best background)
    candidates.sort(key=lambda c: c[0])

    # Enforce octant diversity: pick at most one per octant until we have n_boxes
    used_octants = set()
    for score, angle, x, y in candidates:
        octant = int((angle / (2 * np.pi)) * 8) % 8
        if octant in used_octants:
            continue
        used_octants.add(octant)
        placed.append((x, y, box_size, box_size))
        if len(placed) >= n_boxes:
            break

    # If we didn't fill n_boxes due to octant constraint, fill from best remaining
    if len(placed) < n_boxes:
        already = set(tuple(p) for p in placed)
        for score, angle, x, y in candidates:
            box = (x, y, box_size, box_size)
            if box in already:
                continue
            placed.append(box)
            already.add(box)
            if len(placed) >= n_boxes:
                break

    return placed


def fallback_find_rois(stats, exclude_mask, min_area=400, max_rois=5):
    """Fallback: find background regions without a signal click.
    Looks for large contiguous low-MAD low-intensity regions.
    """
    mad = stats['mad']
    med = stats['median']
    mad_t = float(np.median(mad)) * 1.5
    int_t = float(np.percentile(med, 30))
    candidate = (mad <= mad_t) & (med <= int_t) & ~exclude_mask
    try:
        from scipy.ndimage import label, find_objects
        labeled, _ = label(candidate)
    except ImportError:
        ys, xs = np.where(candidate)
        if len(xs) == 0: return []
        return [(int(xs.min()), int(ys.min()),
                 int(xs.max() - xs.min() + 1),
                 int(ys.max() - ys.min() + 1))]
    boxes = []
    for i, sl in enumerate(find_objects(labeled)):
        if sl is None: continue
        ys, xs = sl
        h_box = ys.stop - ys.start
        w_box = xs.stop - xs.start
        comp_pixels = (labeled[sl] == (i + 1)).sum()
        if comp_pixels < min_area: continue
        if comp_pixels / (h_box * w_box) < 0.5: continue
        boxes.append((int(xs.start), int(ys.start), int(w_box), int(h_box)))
    boxes.sort(key=lambda b: -b[2] * b[3])
    return boxes[:max_rois]

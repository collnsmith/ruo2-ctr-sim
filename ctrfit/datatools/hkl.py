"""HKL files (from sxrd_hkl_appender, sxrd_hkl_smoother and sxrd_hkl_combiner, no Qt).

Plotter exports (x, y per trace) get H, K, L interpolated from the SPEC scan; peak-fit tables get
HKL per scan; HKL files (H K L I, several traces marked by "# Trace:") are read, smoothed with
Bragg exclusion zones, combined into one H K L I table, or written as GenX files."""
import os
import re

import numpy as np


def parse_exported_file(filepath):
    """Parse plotter-exported text file.
    Returns: list of trace dicts {scan, y_col, x_label, y_label, x, y, raw_lines_before}
             plus header_lines (the # lines before the first trace)
    """
    traces = []
    header_lines = []
    current = None
    seen_first_trace = False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Match trace headers like "# Trace: S479_ROI2 (scan=479, y=[bg] ROI2)"
    trace_re = re.compile(r'^#\s*Trace:\s*\S+\s*\(scan=(\d+),\s*y=(.+)\)')

    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\r\n')
        m = trace_re.match(line)
        if m:
            if current:
                traces.append(current)
            scan_num = int(m.group(1))
            y_col_meta = m.group(2).strip().rstrip(')')
            current = {
                'scan': scan_num,
                'y_col_meta': y_col_meta,
                'header_line': line,
                'x_label': None,
                'y_label': None,
                'x': [],
                'y': [],
            }
            seen_first_trace = True
            # next non-empty non-comment line should be the column header
            i += 1
            while i < len(lines):
                hl = lines[i].rstrip('\r\n')
                if not hl.strip():
                    i += 1; continue
                if hl.startswith('#'):
                    i += 1; continue
                # column header
                parts = hl.split('\t')
                if len(parts) >= 2:
                    current['x_label'] = parts[0].strip()
                    current['y_label'] = parts[1].strip()
                i += 1
                break
            continue

        if not seen_first_trace and line.startswith('#'):
            header_lines.append(line)

        if current and line.strip() and not line.startswith('#'):
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    current['x'].append(float(parts[0]))
                    current['y'].append(float(parts[1]))
                except ValueError:
                    pass
        i += 1

    if current:
        traces.append(current)

    # Convert lists to arrays
    for t in traces:
        t['x'] = np.array(t['x'])
        t['y'] = np.array(t['y'])

    return header_lines, traces


def get_hkl_columns(spec_scan):
    """Find the H, K, L columns in a spec scan (case-insensitive).
    Returns dict {'H': array, 'K': array, 'L': array} or None if any missing.
    """
    result = {}
    for target in ['H', 'K', 'L']:
        found = None
        for col in spec_scan.columns:
            if col.strip().upper() == target:
                found = col
                break
        if found is None:
            return None
        result[target] = spec_scan.data[found]
    return result


def interpolate_hkl(x_data, x_label, hkl_arrays, spec_scan):
    """Given x values from the data file and the spec scan, compute H, K, L
    at each x point.

    If x_label matches one of H/K/L (case-insensitive), use x_data directly
    for that axis and interpolate the other two against the matching spec column.

    Otherwise interpolate H, K, L against the first spec column (assumed to
    be the scanned motor) - this is a fallback and may be inaccurate.
    """
    x_label_upper = x_label.strip().upper()

    # Find which spec column to interpolate AGAINST
    interp_axis = None
    if x_label_upper in ('H', 'K', 'L'):
        interp_axis = x_label_upper
    else:
        # Try to find the x_label in the spec columns directly
        for col in spec_scan.columns:
            if col.strip().lower() == x_label.strip().lower():
                # Use this column as the interpolation axis
                spec_x = spec_scan.data[col]
                H = np.interp(x_data, spec_x, hkl_arrays['H'])
                K = np.interp(x_data, spec_x, hkl_arrays['K'])
                L = np.interp(x_data, spec_x, hkl_arrays['L'])
                return H, K, L
        # Fall back to L axis
        interp_axis = 'L'

    spec_x = hkl_arrays[interp_axis]

    # Sort spec_x for np.interp (requires monotonic)
    sort_idx = np.argsort(spec_x)
    spec_x_sorted = spec_x[sort_idx]

    out = {}
    for axis in ['H', 'K', 'L']:
        if axis == interp_axis:
            out[axis] = x_data.copy()
        else:
            spec_y_sorted = hkl_arrays[axis][sort_idx]
            out[axis] = np.interp(x_data, spec_x_sorted, spec_y_sorted)

    return out['H'], out['K'], out['L']


def snap_constant(arr, tol=0.05, snap_step=1.0):
    """If the spread of arr is less than tol, snap all values to the nearest
    multiple of snap_step. Returns (snapped_array, was_snapped, original_value, snapped_value).

    snap_step examples:
        1.0  -> nearest integer
        0.5  -> nearest half-integer
        0.0  -> use the mean (no rounding)
    """
    if len(arr) == 0:
        return arr, False, None, None
    spread = arr.max() - arr.min()
    if spread > tol:
        return arr, False, None, None
    mean_val = float(arr.mean())
    if snap_step > 0:
        snapped = round(mean_val / snap_step) * snap_step
    else:
        snapped = mean_val
    return np.full_like(arr, snapped), True, mean_val, snapped


def write_output(output_path, header_lines, traces, spec_scans,
                 snap_settings=None, log_fn=None):
    """Write the augmented file with H K L columns.

    snap_settings: dict like {
        'H': {'enabled': True, 'tol': 0.05, 'step': 1.0},
        'K': {'enabled': True, 'tol': 0.05, 'step': 1.0},
        'L': {'enabled': False, 'tol': 0.05, 'step': 1.0},
    }
    """
    n_ok = 0
    n_skip = 0

    if snap_settings is None:
        snap_settings = {}

    with open(output_path, 'w') as f:
        # Original header
        for hl in header_lines:
            f.write(hl + '\n')
        f.write('# Augmented with H K L from spec file\n\n')

        for t in traces:
            sn = t['scan']
            x_label = t['x_label'] or 'X'
            y_label = t['y_label'] or 'Y'
            x_data = t['x']
            y_data = t['y']

            if sn not in spec_scans:
                if log_fn:
                    log_fn(f"  WARNING: Scan {sn} not in spec file - skipping HKL")
                # Write trace as-is
                f.write(t['header_line'] + '\n')
                f.write(f"{x_label}\t{y_label}\n")
                for xi, yi in zip(x_data, y_data):
                    f.write(f"{xi:.6g}\t{yi:.6g}\n")
                f.write('\n')
                n_skip += 1
                continue

            spec_scan = spec_scans[sn]
            hkl_arrays = get_hkl_columns(spec_scan)
            if hkl_arrays is None:
                if log_fn:
                    log_fn(f"  WARNING: Scan {sn} missing H/K/L columns - skipping")
                f.write(t['header_line'] + '\n')
                f.write(f"{x_label}\t{y_label}\n")
                for xi, yi in zip(x_data, y_data):
                    f.write(f"{xi:.6g}\t{yi:.6g}\n")
                f.write('\n')
                n_skip += 1
                continue

            H, K, L = interpolate_hkl(x_data, x_label, hkl_arrays, spec_scan)

            # Apply snapping to nearly-constant columns
            snap_log = []
            for axis_name, axis_arr in [('H', H), ('K', K), ('L', L)]:
                cfg = snap_settings.get(axis_name, {})
                if cfg.get('enabled', False):
                    new_arr, was_snapped, orig, snapped = snap_constant(
                        axis_arr,
                        tol=cfg.get('tol', 0.05),
                        snap_step=cfg.get('step', 1.0))
                    if was_snapped:
                        snap_log.append(f"{axis_name}:{orig:.4f}->{snapped:g}")
                        if axis_name == 'H': H = new_arr
                        elif axis_name == 'K': K = new_arr
                        else: L = new_arr

            # Determine output format
            x_label_upper = x_label.strip().upper()
            x_is_hkl = x_label_upper in ('H', 'K', 'L')

            f.write(t['header_line'] + '\n')
            if x_is_hkl:
                # H K L y_col
                f.write(f"H\tK\tL\t{y_label}\n")
                for h, k, l, y in zip(H, K, L, y_data):
                    f.write(f"{h:.6g}\t{k:.6g}\t{l:.6g}\t{y:.6g}\n")
            else:
                # H K L y_col x_orig
                f.write(f"H\tK\tL\t{y_label}\t{x_label}\n")
                for h, k, l, y, x in zip(H, K, L, y_data, x_data):
                    f.write(f"{h:.6g}\t{k:.6g}\t{l:.6g}\t{y:.6g}\t{x:.6g}\n")
            f.write('\n')
            n_ok += 1
            if log_fn:
                snap_str = f" snapped:[{', '.join(snap_log)}]" if snap_log else ""
                log_fn(f"  Scan {sn}: OK ({len(x_data)} pts, "
                       f"H~{H.mean():.3f}, K~{K.mean():.3f}, "
                       f"L=[{L.min():.2f},{L.max():.2f}]){snap_str}")

    return n_ok, n_skip


def parse_peakfit_file(filepath):
    """Parse a peak fit file with tab-separated columns.
    First row is the header. Returns (header_list, list_of_dicts).
    Typical columns: scan, area, fit_area, num_area, model, params
    """
    rows = []
    header = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split('\t')
            if header is None:
                header = [p.strip() for p in parts]
                continue
            if len(parts) < len(header):
                continue
            row = {}
            for h, v in zip(header, parts):
                row[h] = v.strip()
            rows.append(row)
    return header, rows


def process_peakfit_file(input_path, output_path, spec_scans, area_col,
                         snap_settings, log_fn=None):
    """Read a peak fit file, match scan numbers to spec scans, and write
    H K L <area> <scan> rows.

    Returns number of output rows written.
    """
    header, rows = parse_peakfit_file(input_path)
    if not header:
        if log_fn: log_fn("  ERROR: empty or malformed file")
        return 0

    if 'scan' not in header:
        if log_fn: log_fn(f"  ERROR: no 'scan' column (found: {header})")
        return 0
    if area_col not in header:
        if log_fn: log_fn(f"  ERROR: no '{area_col}' column (found: {header})")
        return 0

    # Build output rows: (H, K, L, area, scan)
    out_rows = []
    skipped = 0
    for r in rows:
        try:
            sn = int(r['scan'])
            area_val = float(r[area_col])
        except (ValueError, KeyError):
            skipped += 1
            continue

        if sn not in spec_scans:
            if log_fn: log_fn(f"  WARNING: scan {sn} not in spec file")
            skipped += 1
            continue

        spec_scan = spec_scans[sn]
        hkl = get_hkl_columns(spec_scan)
        if hkl is None:
            if log_fn: log_fn(f"  WARNING: scan {sn} missing H/K/L columns")
            skipped += 1
            continue

        # Use the mean HKL across the scan
        H = float(np.mean(hkl['H']))
        K = float(np.mean(hkl['K']))
        L = float(np.mean(hkl['L']))
        out_rows.append({'H': H, 'K': K, 'L': L, 'area': area_val, 'scan': sn})

    if not out_rows:
        return 0

    # Apply snap settings per axis across the whole output
    def apply_snap(axis_key):
        cfg = snap_settings.get(axis_key, {})
        if not cfg.get('enabled', False):
            return
        arr = np.array([r[axis_key] for r in out_rows])
        new_arr, was_snapped, orig, snapped = snap_constant(
            arr, tol=cfg.get('tol', 0.05), snap_step=cfg.get('step', 1.0))
        if was_snapped:
            for r in out_rows:
                r[axis_key] = float(snapped)
            if log_fn:
                log_fn(f"  Snapped {axis_key}: {orig:.4f} -> {snapped:g}")

    apply_snap('H')
    apply_snap('K')
    apply_snap('L')

    # Write output
    with open(output_path, 'w') as f:
        f.write(f"# Peak fit data converted to HKL format\n")
        f.write(f"# Source: {os.path.basename(input_path)}\n")
        f.write(f"# Area column: {area_col}\n")
        f.write(f"H\tK\tL\tI\tscan\n")
        # Sort by L for nicer output
        for r in sorted(out_rows, key=lambda x: x['L']):
            f.write(f"{r['H']:.6g}\t{r['K']:.6g}\t{r['L']:.6g}\t"
                    f"{r['area']:.6g}\t{r['scan']}\n")

    if log_fn and skipped:
        log_fn(f"  Skipped {skipped} rows (missing spec or HKL)")
    return len(out_rows)


def parse_hkl_file(filepath):
    """Parse an HKL-augmented file (output of the Plotter Data tab).

    Format expected: tab-separated with header row containing H, K, L, then
    one or more intensity columns. May contain multiple "# Trace:" blocks,
    each with its own header row. Comment lines start with '#'.

    Returns list of dicts: {'label': str, 'H': arr, 'K': arr, 'L': arr,
                            'I': arr, 'extra_x': arr or None, 'extra_x_name': str}
    """
    traces = []
    current = None
    current_header = None
    h_idx = k_idx = l_idx = i_idx = None
    extra_x_idx = None
    extra_x_name = None

    trace_re = re.compile(r'^#\s*Trace:\s*(\S+)')

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\r\n')
            stripped = line.strip()

            # New trace marker
            m = trace_re.match(stripped) if stripped.startswith('#') else None
            if m:
                if current and len(current['L']) > 0:
                    traces.append(current)
                current = {'label': m.group(1), 'H': [], 'K': [], 'L': [],
                           'I': [], 'extra_x': [], 'extra_x_name': None}
                current_header = None
                continue

            # Skip blank or other comment lines
            if not stripped or stripped.startswith('#'):
                continue

            parts = stripped.split('\t')
            if len(parts) < 2:
                parts = stripped.split()

            # Header row?
            if current_header is None:
                upper = [p.strip().upper() for p in parts]
                if 'H' in upper and 'K' in upper and 'L' in upper:
                    current_header = parts
                    h_idx = upper.index('H')
                    k_idx = upper.index('K')
                    l_idx = upper.index('L')
                    # Intensity column = first non-HKL column
                    i_idx = None
                    extra_x_idx = None
                    for j in range(len(parts)):
                        if j in (h_idx, k_idx, l_idx):
                            continue
                        if i_idx is None:
                            i_idx = j
                        else:
                            extra_x_idx = j
                            extra_x_name = parts[j]
                    # Initialize a trace if none yet
                    if current is None:
                        label = os.path.splitext(os.path.basename(filepath))[0]
                        current = {'label': label, 'H': [], 'K': [], 'L': [],
                                   'I': [], 'extra_x': [], 'extra_x_name': extra_x_name}
                    else:
                        current['extra_x_name'] = extra_x_name
                    continue
                else:
                    # No HKL header found - treat as a 4-col HKL file with no header
                    h_idx, k_idx, l_idx, i_idx = 0, 1, 2, 3
                    current_header = ['H', 'K', 'L', 'I']
                    if current is None:
                        label = os.path.splitext(os.path.basename(filepath))[0]
                        current = {'label': label, 'H': [], 'K': [], 'L': [],
                                   'I': [], 'extra_x': [], 'extra_x_name': None}
                    # Fall through to data parsing

            # Data row
            if current is None:
                continue
            try:
                h = float(parts[h_idx])
                k = float(parts[k_idx])
                l = float(parts[l_idx])
                i = float(parts[i_idx])
            except (ValueError, IndexError):
                continue
            current['H'].append(h)
            current['K'].append(k)
            current['L'].append(l)
            current['I'].append(i)
            if extra_x_idx is not None:
                try:
                    current['extra_x'].append(float(parts[extra_x_idx]))
                except (ValueError, IndexError):
                    current['extra_x'].append(None)

    if current and len(current['L']) > 0:
        traces.append(current)

    # Convert to arrays
    for t in traces:
        t['H'] = np.array(t['H'])
        t['K'] = np.array(t['K'])
        t['L'] = np.array(t['L'])
        t['I'] = np.array(t['I'])

    return traces


def write_genx_file(out_path, traces, hk_overrides=None,
                    error_mode='sqrt', error_floor=1.0,
                    add_lb_dl=False, dl_value=1.0,
                    drop_negative=True, log_fn=None):
    """Write traces to a single GenX sls_sxrd-compatible file.

    Format (5 columns): H K L I Ierr
    Optional 7 columns: H K L I Ierr LB dL  (for beta roughness model)

    Parameters:
        traces: list from parse_hkl_file (or merged across multiple files)
        hk_overrides: dict {trace_label: (H, K)} to force HK values per trace
                      (useful when the original H/K were wrong/offset)
        error_mode: 'sqrt' (sqrt of intensity, counting stats),
                    'percent_5' (5% of intensity),
                    'constant' (use error_floor for everything)
        error_floor: minimum error value
        add_lb_dl: include LB and dL columns
        dl_value: dL value (Bragg spacing)
        drop_negative: drop rows where I <= 0 (can't fit in log space)

    Returns number of rows written.
    """
    n_total = 0
    n_dropped = 0
    rows_to_write = []

    for t in traces:
        label = t['label']
        H_arr = t['H'].copy()
        K_arr = t['K'].copy()

        if hk_overrides and label in hk_overrides:
            h_o, k_o = hk_overrides[label]
            if h_o is not None: H_arr[:] = h_o
            if k_o is not None: K_arr[:] = k_o

        for j in range(len(t['L'])):
            I = t['I'][j]
            if drop_negative and I <= 0:
                n_dropped += 1
                continue
            if error_mode == 'sqrt':
                Ierr = max(np.sqrt(abs(I)), error_floor)
            elif error_mode == 'percent_5':
                Ierr = max(abs(I) * 0.05, error_floor)
            else:
                Ierr = error_floor
            row = (float(H_arr[j]), float(K_arr[j]), float(t['L'][j]),
                   float(I), float(Ierr))
            rows_to_write.append(row)

    # Sort by H, K, then L (so each rod is contiguous)
    rows_to_write.sort(key=lambda r: (round(r[0], 3), round(r[1], 3), r[2]))

    with open(out_path, 'w') as f:
        f.write("# GenX sls_sxrd format\n")
        if add_lb_dl:
            f.write(f"# Columns: H K L I Ierr LB dL  (dL={dl_value})\n")
        else:
            f.write("# Columns: H K L I Ierr\n")
        for (H, K, L, I, Ierr) in rows_to_write:
            if add_lb_dl:
                LB = round(L / dl_value) * dl_value
                f.write(f"{H:.6g}\t{K:.6g}\t{L:.6g}\t{I:.6g}\t{Ierr:.6g}"
                        f"\t{LB:.6g}\t{dl_value:.6g}\n")
            else:
                f.write(f"{H:.6g}\t{K:.6g}\t{L:.6g}\t{I:.6g}\t{Ierr:.6g}\n")
        n_total = len(rows_to_write)

    if log_fn:
        log_fn(f"  Wrote {n_total} rows to {os.path.basename(out_path)}")
        if n_dropped:
            log_fn(f"  Dropped {n_dropped} non-positive intensity rows")
    return n_total


def smooth_segment(y, x, method, window, poly_order):
    """Apply smoothing to a single contiguous segment.
    Returns smoothed y (same length as input)."""
    n = len(y)
    win = min(window, n)
    if win % 2 == 0:
        win = max(win - 1, 3)
    if win < 3:
        return y.copy()

    if method == "Savitzky-Golay":
        from scipy.signal import savgol_filter
        order = min(poly_order, win - 1)
        return savgol_filter(y, win, order)
    elif method == "Moving average":
        kernel = np.ones(win) / win
        padded = np.pad(y, win // 2, mode='edge')
        return np.convolve(padded, kernel, mode='valid')[:n]
    elif method == "Gaussian kernel":
        from scipy.ndimage import gaussian_filter1d
        sigma = win / 4
        return gaussian_filter1d(y, sigma)
    elif method == "Lowess":
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            frac = min(win / n, 0.9)
            return lowess(y, x, frac=frac, return_sorted=False)
        except ImportError:
            from scipy.signal import savgol_filter
            return savgol_filter(y, min(win, n if n % 2 == 1 else n - 1),
                                 min(poly_order, win - 1))
    elif method == "Spline":
        from scipy.interpolate import UnivariateSpline
        try:
            s_factor = len(y) * np.var(y) * (1.0 / max(poly_order, 1))
            spl = UnivariateSpline(x, y, s=s_factor, k=min(poly_order, 5))
            return spl(x)
        except Exception:
            return y.copy()
    return y.copy()


def smooth_trace(x, y, method, window, poly_order, exclusions):
    """Smooth y(x) honoring exclusion zones.

    exclusions: list of (x_min, x_max). Points inside any zone are skipped
    by smoothing and returned as NaN. Smoothing is applied independently to
    each contiguous included segment.

    Returns array of same length as y, with NaN where excluded.
    """
    if len(x) == 0:
        return np.array([])

    mask = np.ones(len(x), dtype=bool)
    for xmin, xmax in exclusions:
        mask &= ~((x >= xmin) & (x <= xmax))

    smoothed = np.full(len(y), np.nan)

    inc_indices = np.where(mask)[0]
    if len(inc_indices) < 4:
        # Too few included points - just copy original at those positions
        smoothed[mask] = y[mask]
        return smoothed

    # Find contiguous segments
    segments = []
    seg_start = 0
    for i in range(1, len(inc_indices)):
        if inc_indices[i] - inc_indices[i - 1] > 1:
            segments.append(inc_indices[seg_start:i])
            seg_start = i
    segments.append(inc_indices[seg_start:])

    win = window if window % 2 == 1 else window + 1

    for seg_idx in segments:
        if len(seg_idx) < 3:
            continue
        x_seg = x[seg_idx]
        y_seg = y[seg_idx]
        y_smooth = smooth_segment(y_seg, x_seg, method, win, poly_order)
        smoothed[seg_idx] = y_smooth

    return smoothed


# ============================================================
# COMBINE
# ============================================================

SORT_CHOICES = ("File order (no sort)", "By L", "By H, K, L")


def combine_hkl_files(paths, drop_nan=True, drop_nonpos=False, sort="File order (no sort)", dedupe=False):
    """Rows (H, K, L, I) of every trace in every HKL file, filtered, sorted and deduplicated.
    Returns (rows, log lines)."""
    rows, log = [], []
    n_nan = n_nonpos = 0
    for p in paths:
        try:
            traces = parse_hkl_file(p)
        except Exception as e:
            log.append(f"  ERROR parsing {os.path.basename(p)}: {e}")
            continue
        if not traces:
            log.append(f"  {os.path.basename(p)}: no traces")
            continue
        n_file = 0
        for t in traces:
            for h, k, l, i in zip(t["H"], t["K"], t["L"], t["I"]):
                iv = float(i)
                if drop_nan and not np.isfinite(iv):
                    n_nan += 1
                    continue
                if drop_nonpos and iv <= 0:
                    n_nonpos += 1
                    continue
                rows.append((float(h), float(k), float(l), iv))
                n_file += 1
        log.append(f"  {os.path.basename(p)}: {len(traces)} trace(s), {n_file} rows")
    if sort == "By L":
        rows.sort(key=lambda r: r[2])
    elif sort == "By H, K, L":
        rows.sort(key=lambda r: (round(r[0], 4), round(r[1], 4), r[2]))
    if dedupe:
        seen, unique = set(), []
        for r in rows:
            key = (round(r[0], 6), round(r[1], 6), round(r[2], 6), r[3])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        if len(unique) < len(rows):
            log.append(f"  Deduped: removed {len(rows) - len(unique)} duplicate row(s)")
        rows = unique
    if n_nan:
        log.append(f"  Dropped {n_nan} NaN/inf row(s)")
    if n_nonpos:
        log.append(f"  Dropped {n_nonpos} non-positive row(s)")
    return rows, log


def write_hkl_rows(path, rows):
    """One 'H K L I' header line and one tab-separated row per point."""
    with open(path, "w") as f:
        f.write("H\tK\tL\tI\n")
        for h, k, l, i in rows:
            f.write(f"{h:.6g}\t{k:.6g}\t{l:.6g}\t{i:.6g}\n")
    return path

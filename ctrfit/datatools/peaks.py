"""Peak fitting and ROI-sum files for the plotter (from sxrd_plotter, no Qt): bgsub sum files,
the axis label map, Gaussian / Lorentzian / pseudo-Voigt with a linear background, fit_peak,
integrated and numerical areas."""
import os

import numpy as np
from scipy.optimize import curve_fit

from ctrfit.datatools.paths import settings_file


_trapezoid = getattr(np, "trapezoid", None) or np.trapz      # NumPy 2 renamed trapz


def parse_bgsub_file(filepath):
    """Parse a single bgsub tab-separated output file."""
    scan_data = {}
    header = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if header is None: header = parts; continue
            if len(parts) != len(header): continue
            try:
                sn = int(parts[0])
            except ValueError:
                continue
            if sn not in scan_data: scan_data[sn] = {h: [] for h in header}
            for h, v in zip(header, parts):
                try: scan_data[sn][h].append(float(v))
                except ValueError: scan_data[sn][h].append(0.0)
    result = {}
    for sn, cols in scan_data.items():
        result[sn] = {k: np.array(v) for k, v in cols.items()}
    return result, header


def parse_bgsub_dir(dirpath):
    """Parse all bgsub output files in a directory.
    Returns merged dict of scan_number -> {col: array} and list of all headers."""
    import glob
    all_data = {}
    all_headers = set()
    files_loaded = 0
    for fpath in sorted(glob.glob(os.path.join(dirpath, '*.txt'))):
        try:
            data, header = parse_bgsub_file(fpath)
            if header and 'scan' in header and 'image' in header:
                for sn, cols in data.items():
                    if sn not in all_data:
                        all_data[sn] = cols
                    else:
                        # Merge: append if same columns, skip if conflict
                        for k, v in cols.items():
                            if k not in all_data[sn]:
                                all_data[sn][k] = v
                if header:
                    all_headers.update(header)
                files_loaded += 1
        except Exception:
            continue
    return all_data, sorted(all_headers), files_loaded


def load_label_map():
    m = {}
    path = settings_file("axis_label_map.txt")
    if os.path.isfile(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    m[k.strip()] = v.strip()
    return m


def save_label_map(m):
    with open(settings_file("axis_label_map.txt"), "w") as f:
        f.write("# Axis label map: original = display\n")
        for k, v in sorted(m.items()):
            f.write(f"{k} = {v}\n")


def display_label(name, lm):
    return lm.get(name, name)


def gaussian(x, amp, cen, sigma, bg_const, bg_slope):
    return amp * np.exp(-0.5 * ((x - cen) / sigma)**2) + bg_const + bg_slope * x


def lorentzian(x, amp, cen, gamma, bg_const, bg_slope):
    return amp * gamma**2 / ((x - cen)**2 + gamma**2) + bg_const + bg_slope * x


def pseudo_voigt(x, amp, cen, sigma, eta, bg_const, bg_slope):
    g = np.exp(-0.5 * ((x - cen) / sigma)**2)
    l = sigma**2 / ((x - cen)**2 + sigma**2)
    return amp * (eta * l + (1 - eta) * g) + bg_const + bg_slope * x


FIT_MODELS = {
    'Gaussian + linear bg': {
        'func': gaussian,
        'params': ['amp', 'cen', 'sigma', 'bg_const', 'bg_slope'],
        'p0': lambda x, y: [y.max() - np.median(y), x[np.argmax(y)],
                             (x.max()-x.min())/10, np.median(y), 0],
        'bounds': lambda x, y: ([0, x.min(), 0, -np.inf, -np.inf],
                                 [np.inf, x.max(), x.max()-x.min(), np.inf, np.inf]),
        'area': lambda p: p[0] * p[2] * np.sqrt(2 * np.pi),
    },
    'Lorentzian + linear bg': {
        'func': lorentzian,
        'params': ['amp', 'cen', 'gamma', 'bg_const', 'bg_slope'],
        'p0': lambda x, y: [y.max() - np.median(y), x[np.argmax(y)],
                             (x.max()-x.min())/10, np.median(y), 0],
        'bounds': lambda x, y: ([0, x.min(), 0, -np.inf, -np.inf],
                                 [np.inf, x.max(), x.max()-x.min(), np.inf, np.inf]),
        'area': lambda p: p[0] * np.pi * p[2],
    },
    'Pseudo-Voigt + linear bg': {
        'func': pseudo_voigt,
        'params': ['amp', 'cen', 'sigma', 'eta', 'bg_const', 'bg_slope'],
        'p0': lambda x, y: [y.max() - np.median(y), x[np.argmax(y)],
                             (x.max()-x.min())/10, 0.5, np.median(y), 0],
        'bounds': lambda x, y: ([0, x.min(), 0, 0, -np.inf, -np.inf],
                                 [np.inf, x.max(), x.max()-x.min(), 1, np.inf, np.inf]),
        'area': None,  # numerical integration
    },
    'Gaussian (no bg)': {
        'func': lambda x, a, c, s: a * np.exp(-0.5*((x-c)/s)**2),
        'params': ['amp', 'cen', 'sigma'],
        'p0': lambda x, y: [y.max(), x[np.argmax(y)], (x.max()-x.min())/10],
        'bounds': lambda x, y: ([0, x.min(), 0], [np.inf, x.max(), x.max()-x.min()]),
        'area': lambda p: p[0] * p[2] * np.sqrt(2 * np.pi),
    },
}


def fit_peak(x, y, model_name, center_hint=None, fit_window=None):
    """Fit a peak model to data.
    center_hint: override initial center guess
    fit_window: (half-width) restrict fit to center_hint ± fit_window
    Returns (popt, pcov, model_info, x_used, y_used) or None on failure."""
    model = FIT_MODELS.get(model_name)
    if model is None:
        return None

    # Apply fit window
    x_fit, y_fit = x, y
    if fit_window is not None and center_hint is not None and fit_window > 0:
        mask = (x >= center_hint - fit_window) & (x <= center_hint + fit_window)
        if mask.sum() < 4:  # need at least a few points
            mask = np.ones(len(x), dtype=bool)
        x_fit = x[mask]
        y_fit = y[mask]

    try:
        p0 = model['p0'](x_fit, y_fit)
        bounds = model['bounds'](x_fit, y_fit)

        if center_hint is not None:
            p0[1] = center_hint

            # Smarter initial guesses from local data around hint
            near_mask = np.abs(x_fit - center_hint) < (x_fit.max() - x_fit.min()) / 4
            if near_mask.sum() > 2:
                local_y = y_fit[near_mask]
                # Amplitude from local peak minus edge background
                edge_bg = (y_fit[:3].mean() + y_fit[-3:].mean()) / 2 if len(y_fit) > 6 else np.median(y_fit)
                p0[0] = max(local_y.max() - edge_bg, 1.0)  # amp

                # Sigma from local FWHM estimate
                above_half = local_y > (local_y.max() + edge_bg) / 2
                if above_half.sum() > 1:
                    local_x = x_fit[near_mask]
                    fwhm_est = np.ptp(local_x[above_half])
                    if fwhm_est > 0:
                        p0[2] = fwhm_est / 2.35  # sigma ≈ FWHM/2.35

                # Background from edges
                if len(p0) >= 5:
                    p0[-2] = edge_bg  # bg_const
                    if len(x_fit) > 4:
                        p0[-1] = (y_fit[-3:].mean() - y_fit[:3].mean()) / (x_fit[-1] - x_fit[0])  # bg_slope

        popt, pcov = curve_fit(model['func'], x_fit, y_fit, p0=p0, bounds=bounds, maxfev=5000)
        return popt, pcov, model, x_fit, y_fit
    except Exception:
        return None


def integrate_peak(x, y, popt, model):
    """Compute integrated intensity (area) from fit."""
    if model['area'] is not None:
        return model['area'](popt)
    x_fine = np.linspace(x.min(), x.max(), 1000)
    y_fit = model['func'](x_fine, *popt)
    if len(popt) >= 5:
        bg = popt[-2] + popt[-1] * x_fine
    else:
        bg = 0
    return _trapezoid(y_fit - bg, x_fine)


def numerical_area(x, y, center=None, window=None):
    """Model-free integrated area by trapezoidal rule after subtracting
    a linear background estimated from the edges of the region."""
    if center is not None and window is not None and window > 0:
        mask = (x >= center - window) & (x <= center + window)
        if mask.sum() < 3:
            mask = np.ones(len(x), dtype=bool)
        x = x[mask]; y = y[mask]

    if len(x) < 3:
        return 0.0

    # Linear background from first/last 3 points (or fewer)
    n_edge = min(3, len(x) // 3)
    if n_edge < 1:
        n_edge = 1
    x_lo, y_lo = x[:n_edge].mean(), y[:n_edge].mean()
    x_hi, y_hi = x[-n_edge:].mean(), y[-n_edge:].mean()
    if abs(x_hi - x_lo) > 0:
        slope = (y_hi - y_lo) / (x_hi - x_lo)
        intercept = y_lo - slope * x_lo
    else:
        slope = 0; intercept = np.median(y)
    bg = slope * x + intercept
    y_sub = y - bg

    try:
        return float(np.trapezoid(np.maximum(y_sub, 0), x))
    except AttributeError:
        return float(_trapezoid(np.maximum(y_sub, 0), x))

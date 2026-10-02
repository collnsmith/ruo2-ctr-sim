"""Covariance, correlations and the warnings every fit report carries."""
import numpy as np

CORR_WARN = 0.9
BOUND_FRAC = 0.01
REL_ERR_WARN = 1.0
MAX_CORR_LINES = 5


def covariance_from_jacobian(J, red_chi2):
    """cov = red_chi2 * (J^T J)^-1 (pseudo-inverse). Returns (cov, condition number)."""
    J = np.asarray(J, float)
    if J.size == 0:
        return np.zeros((0, 0)), 1.0
    _, sv, Vt = np.linalg.svd(J, full_matrices=False)
    cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
    keep = sv > sv[0] * 1e-12
    inv = (Vt[keep].T / sv[keep] ** 2) @ Vt[keep]
    scale = red_chi2 if np.isfinite(red_chi2) and red_chi2 > 0 else 1.0
    return scale * inv, cond


def correlation(cov):
    d = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = cov / np.outer(d, d)
    c[~np.isfinite(c)] = 0.0
    np.fill_diagonal(c, 1.0)
    return c


def fit_warnings(params, names, values, errors, corr, cond=1.0):
    """Plain-language warnings: strong correlations, values at bounds, undetermined parameters."""
    out = []
    if not np.isfinite(cond) or cond > 1e10:
        out.append("The fit is ill-conditioned (some parameter combinations are not determined by the data); "
                   "errors are unreliable. Fix or link parameters.")
    n = len(names)
    pairs = sorted(((abs(corr[i, j]), i, j) for i in range(n) for j in range(i + 1, n)
                    if abs(corr[i, j]) > CORR_WARN), reverse=True)
    for _, i, j in pairs[:MAX_CORR_LINES]:
        out.append(f"{names[i]} and {names[j]} are strongly correlated (r = {corr[i, j]:+.2f}): the data "
                   "cannot separate them. Fix one, link them, or add data that distinguishes them.")
    if len(pairs) > MAX_CORR_LINES:
        out.append(f"... and {len(pairs) - MAX_CORR_LINES} more parameter pairs with |r| > {CORR_WARN:g} "
                   "(see the correlation matrix).")
    for nm in names:
        p = params[nm]
        if p.at_bound(BOUND_FRAC):
            side = "lower" if values[nm] - p.min <= p.max - values[nm] else "upper"
            out.append(f"{nm} = {values[nm]:.4g} is within 1% of its {side} bound "
                       f"({p.min if side == 'lower' else p.max:g}); its error is not reliable.")
    for nm in names:
        v, e = values[nm], errors.get(nm, np.nan)
        if v != 0 and np.isfinite(e) and abs(e) > REL_ERR_WARN * abs(v):
            out.append(f"{nm} = {v:.4g} +- {e:.2g}: the relative error exceeds 100%, so the data do not "
                       "determine it.")
        elif not np.isfinite(e):
            out.append(f"{nm}: no error estimate (not determined by the data).")
    return out

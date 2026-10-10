"""Intensity vs potential: relaxation-period background, cycle averaging and sigmoidal transitions.

Background: a model of intensity vs time fitted to the relaxation period (constant, linear or
exponential decay) is removed from the whole trace, by division (beam / sample decay, the usual
case) or subtraction. Extrapolating a fitted decay far beyond the relaxation window is a guess;
the result says how far it was extrapolated.

Transitions: per sweep direction,

    I(V) = a + b (V - Vmid) + sum_k dI_k / (1 + exp(-(V - E_k) / w_k))

with E_k, w_k and dI_k free (b optional). For a Langmuir-type transition of n electrons
w = RT / (nF) (25.7 mV / n at 25 C); the apparent n is reported. Hysteresis is E(anodic) - E(cathodic)
for transitions matched in order of potential.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

RT_F = 8.314462618 * 298.15 / 96485.33212      # V at 25 C


# ----------------------------------------------------------------------------------------------
# background
# ----------------------------------------------------------------------------------------------
BACKGROUNDS = ("none", "constant", "linear", "exponential")


@dataclass
class Background:
    model: str
    params: tuple
    t0: float
    t_end: float                      # end of the relaxation window used for the fit
    correction: str                   # "divide" or "subtract"
    rms: float = float("nan")
    notes: list = field(default_factory=list)

    def __call__(self, t):
        x = np.asarray(t, float) - self.t0
        if self.model in ("none",):
            return np.ones_like(x) if self.correction == "divide" else np.zeros_like(x)
        if self.model == "constant":
            return np.full_like(x, self.params[0])
        if self.model == "linear":
            return self.params[0] + self.params[1] * x
        a, b, tau = self.params
        return a + b * np.exp(-x / tau)

    def correct(self, t, I, sigma=None):
        """Corrected intensity (and sigma), normalised to the background at the end of the relaxation
        window when dividing, so the numbers stay in counts."""
        if self.model == "none":
            return np.asarray(I, float), sigma
        bg = self(t)
        if self.correction == "divide":
            ref = float(self(np.array([self.t_end]))[0])
            f = ref / bg
            return np.asarray(I) * f, None if sigma is None else np.asarray(sigma) * f
        return np.asarray(I) - bg + float(self(np.array([self.t_end]))[0]), sigma


def fit_background(t, I, t_end, model="linear", correction="divide", sigma=None):
    """Fit `model` to the points with t <= t_end (the relaxation period)."""
    if model not in BACKGROUNDS:
        raise ValueError(f"background model must be one of {', '.join(BACKGROUNDS)}")
    if correction not in ("divide", "subtract"):
        raise ValueError("correction must be 'divide' or 'subtract'")
    t, I = np.asarray(t, float), np.asarray(I, float)
    sel = (t <= t_end) & np.isfinite(I)
    t0 = float(t[0])
    if model == "none":
        return Background("none", (), t0, float(t_end), correction)
    need = {"constant": 1, "linear": 2, "exponential": 4}[model]
    if sel.sum() < need + 1:
        raise ValueError(f"the relaxation window holds {int(sel.sum())} points; the {model} background needs "
                         f"at least {need + 1}")
    x, y = t[sel] - t0, I[sel]
    w = 1 / np.asarray(sigma, float)[sel] if sigma is not None else np.ones_like(y)
    if model == "constant":
        p = (float(np.average(y, weights=w ** 2)),)
    elif model == "linear":
        p = tuple(np.polyfit(x, y, 1, w=w)[::-1])
    else:
        span = max(x[-1] - x[0], 1e-9)
        a0, b0 = float(y[-max(3, y.size // 10):].mean()), float(y[:max(3, y.size // 10)].mean() - y[-1])

        def res(q):
            return (q[0] + q[1] * np.exp(-x / q[2]) - y) * w
        r = least_squares(res, [a0, b0, span / 3], bounds=([-np.inf, -np.inf, span / 1e3], [np.inf, np.inf, 1e3 * span]))
        p = tuple(r.x)
    bg = Background(model, p, t0, float(t_end), correction)
    bg.rms = float(np.sqrt(np.mean((bg(t[sel]) - y) ** 2)))
    span = t[-1] - t_end
    if model in ("linear", "exponential") and span > 2 * (t_end - t0):
        bg.notes.append(f"the {model} background is extrapolated {span:.0f} s beyond a {t_end - t0:.0f} s "
                        "relaxation window; treat slow trends in the CV part with care")
    if model == "exponential" and p[2] > 3 * (t_end - t0):
        bg.notes.append("the fitted decay time is longer than the relaxation window: it is poorly determined")
    return bg


# ----------------------------------------------------------------------------------------------
# cycles and averaging
# ----------------------------------------------------------------------------------------------
def parse_cycles(text, available):
    """'all', '2', '2-4', '1,3,5-6' -> sorted list of cycle numbers present in `available`."""
    avail = sorted({int(c) for c in available if c > 0})
    text = (text or "all").strip().lower()
    if text in ("", "all", "*"):
        return avail
    want = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            want |= set(range(int(a), int(b) + 1))
        else:
            want.add(int(part))
    out = [c for c in avail if c in want]
    if not out:
        raise ValueError(f"cycles '{text}': none of them is in the data (cycles {avail})")
    return out


@dataclass
class Binned:
    branch: int
    V: np.ndarray
    I: np.ndarray
    sigma: np.ndarray                 # standard error of the mean (or propagated counting error)
    n: np.ndarray                     # points per bin


def bin_by_potential(V, I, sigma, branch, cycle, cycles, width=0.01, which=(1, -1)):
    """Average points in potential bins per sweep direction over the chosen cycles. The error is the
    larger of the standard error of the mean and the propagated counting error."""
    V, I = np.asarray(V, float), np.asarray(I, float)
    sigma = np.asarray(sigma, float) if sigma is not None else np.zeros_like(I)
    if width <= 0:
        raise ValueError("bin width must be positive")
    use = np.isin(cycle, list(cycles)) & np.isfinite(V) & np.isfinite(I)
    out = []
    for b in which:
        sel = use & (branch == b)
        if not sel.any():
            continue
        lo = math.floor(V[sel].min() / width) * width
        idx = np.floor((V[sel] - lo) / width).astype(int)
        Vs, Is, ss = V[sel], I[sel], sigma[sel]
        nb = idx.max() + 1
        n = np.bincount(idx, minlength=nb)
        keep = n > 0
        mV = np.bincount(idx, Vs, nb)[keep] / n[keep]
        mI = np.bincount(idx, Is, nb)[keep] / n[keep]
        var = np.bincount(idx, Is ** 2, nb)[keep] / n[keep] - mI ** 2
        nn = n[keep]
        sem = np.sqrt(np.clip(var, 0, None) / np.maximum(nn - 1, 1))
        prop = np.sqrt(np.bincount(idx, ss ** 2, nb)[keep]) / nn
        err = np.maximum(np.where(nn > 1, sem, prop), prop)
        out.append(Binned(b, mV, mI, err, nn))
    return out


# ----------------------------------------------------------------------------------------------
# sigmoidal transitions
# ----------------------------------------------------------------------------------------------
def sigmoid(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


@dataclass
class Transition:
    E0: float
    E0_err: float
    width: float
    width_err: float
    step: float
    step_err: float

    @property
    def n_apparent(self):
        return RT_F / self.width if self.width > 0 else float("nan")


@dataclass
class SigmoidFit:
    branch: int
    transitions: list
    base: float
    slope: float
    Vmid: float
    red_chi2: float
    V: np.ndarray
    I: np.ndarray
    sigma: np.ndarray
    notes: list = field(default_factory=list)

    def model(self, V, parts=False):
        V = np.asarray(V, float)
        comps = [tr.step * sigmoid((V - tr.E0) / tr.width) for tr in self.transitions]
        base = self.base + self.slope * (V - self.Vmid)
        return (base, comps) if parts else base + sum(comps, np.zeros_like(V))


def _guess_steps(V, I, n):
    order = np.argsort(V)
    v, y = V[order], I[order]
    k = max(3, v.size // 15)
    ys = np.convolve(y, np.ones(k) / k, mode="same")
    d = np.gradient(ys, v)
    d[:k // 2 + 1], d[-(k // 2 + 1):] = 0, 0
    picks = []
    mag = np.abs(d)
    span = v[-1] - v[0]
    for _ in range(n):
        i = int(np.argmax(mag))
        picks.append(v[i])
        mag[np.abs(v - v[i]) < 0.12 * span] = 0
    picks = sorted(picks)
    steps = []
    edges = [v[0]] + [0.5 * (a + b) for a, b in zip(picks[:-1], picks[1:])] + [v[-1]]
    for i, e in enumerate(picks):
        lo = y[(v >= edges[i]) & (v < e)]
        hi = y[(v > e) & (v <= edges[i + 1])]
        steps.append(float((hi.mean() if hi.size else y[-1]) - (lo.mean() if lo.size else y[0])))
    return picks, steps


def fit_sigmoids(V, I, sigma=None, n=1, branch=0, slope=False, E0=None, width0=0.03):
    """Least-squares fit of n sigmoidal steps (see the module docstring). E0: optional initial guesses."""
    V, I = np.asarray(V, float), np.asarray(I, float)
    ok = np.isfinite(V) & np.isfinite(I)
    V, I = V[ok], I[ok]
    s = np.asarray(sigma, float)[ok] if sigma is not None else np.full(V.size, max(np.std(I), 1e-12) * 0.05)
    s = np.where(s > 0, s, np.median(s[s > 0]) if np.any(s > 0) else 1.0)
    npar = 1 + int(slope) + 3 * n
    if V.size <= npar:
        raise ValueError(f"{V.size} points cannot fit {n} transition(s) ({npar} parameters)")
    Vmid = 0.5 * (V.min() + V.max())
    span = V.max() - V.min()
    guess_E, guess_s = _guess_steps(V, I, n)
    if E0 is not None:
        guess_E = sorted(float(e) for e in E0)[:n] + guess_E[len(E0):]
    lo_I = float(I[np.argsort(V)][:max(2, V.size // 20)].mean())
    p0 = [lo_I] + ([0.0] if slope else [])
    lb = [-np.inf] + ([-np.inf] if slope else [])
    ub = [np.inf] + ([np.inf] if slope else [])
    for e, st in zip(guess_E, guess_s):
        p0 += [e, min(max(width0, 2e-3), 0.4 * span), st or (I.max() - I.min()) / n]
        lb += [V.min() - 0.1 * span, 1e-3, -np.inf]
        ub += [V.max() + 0.1 * span, max(0.5 * span, 2e-3), np.inf]
    p0 = np.clip(p0, np.array(lb) + 1e-12, np.array(ub) - 1e-12)

    def unpack(p):
        a, i = p[0], 1
        b = 0.0
        if slope:
            b, i = p[1], 2
        trs = [(p[i + 3 * k], p[i + 3 * k + 1], p[i + 3 * k + 2]) for k in range(n)]
        return a, b, trs

    def model(p, v):
        a, b, trs = unpack(p)
        return a + b * (v - Vmid) + sum((st * sigmoid((v - e) / w) for e, w, st in trs), np.zeros_like(v))

    res = least_squares(lambda p: (model(p, V) - I) / s, p0, bounds=(lb, ub), x_scale="jac", max_nfev=5000)
    dof = max(V.size - npar, 1)
    red = float(np.sum(res.fun ** 2) / dof)
    try:
        cov = np.linalg.pinv(res.jac.T @ res.jac) * max(red, 1.0)
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(npar, np.nan)
    a, b, trs = unpack(res.x)
    i0 = 1 + int(slope)
    trans = [Transition(e, err[i0 + 3 * k], w, err[i0 + 3 * k + 1], st, err[i0 + 3 * k + 2])
             for k, (e, w, st) in enumerate(trs)]
    order = np.argsort([t.E0 for t in trans])
    trans = [trans[i] for i in order]
    fit = SigmoidFit(branch, trans, a, b, Vmid, red, V, I, s)
    for t in trans:
        if t.width <= 1.05e-3 or t.width >= 0.49 * span:
            fit.notes.append(f"transition at {t.E0:.3f} V: width at its limit ({1e3 * t.width:.1f} mV)")
        if not (V.min() <= t.E0 <= V.max()):
            fit.notes.append(f"transition at {t.E0:.3f} V lies outside the measured potential range")
    return fit


def hysteresis(anodic, cathodic):
    """[(E_anodic - E_cathodic, error)] for transitions matched in order of potential."""
    if anodic is None or cathodic is None:
        return []
    return [(a.E0 - c.E0, math.hypot(a.E0_err, c.E0_err))
            for a, c in zip(anodic.transitions, cathodic.transitions)]


__all__ = ["BACKGROUNDS", "Background", "fit_background", "parse_cycles", "Binned", "bin_by_potential",
           "sigmoid", "Transition", "SigmoidFit", "fit_sigmoids", "hysteresis", "RT_F"]

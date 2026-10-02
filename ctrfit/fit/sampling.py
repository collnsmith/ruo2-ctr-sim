"""Uncertainty beyond the covariance matrix, and model comparison.

bootstrap(fit, n)       resample data points with replacement, refit by least squares from the best
                        fit; the spread of the refitted values is the bootstrap error.
mcmc(fit, ...)          emcee ensemble sampler of exp(-chi2 / 2) with flat priors inside the bounds.
information_criteria    AIC, AICc and BIC from chi2 (Gaussian errors, known sigma).
compare_models          table of several fit results ranked by AIC / BIC, with Akaike weights.

All random draws use fixed seeds.
"""
import math

import numpy as np
from scipy.optimize import least_squares


class SampleResult:
    """Samples of the free parameters with summary statistics."""

    def __init__(self, names, samples, method, **info):
        self.names = list(names)
        self.samples = np.asarray(samples, float)
        self.method = method
        self.info = info

    @property
    def mean(self):
        return dict(zip(self.names, self.samples.mean(0)))

    @property
    def std(self):
        return dict(zip(self.names, self.samples.std(0, ddof=1)))

    def percentiles(self, q=(15.865, 50.0, 84.135)):
        p = np.percentile(self.samples, q, axis=0)
        return {n: tuple(p[:, i]) for i, n in enumerate(self.names)}

    def interval(self, name, level=0.6827):
        """Central credible / confidence interval of one parameter."""
        a = 50 * (1 - level)
        return tuple(np.percentile(self.samples[:, self.names.index(name)], [a, 100 - a]))

    @property
    def correlation(self):
        return np.corrcoef(self.samples, rowvar=False) if len(self.names) > 1 else np.ones((1, 1))

    @property
    def summary(self):
        pc = self.percentiles()
        lines = [f"{self.method}: {self.samples.shape[0]} samples"
                 + "".join(f", {k} {v:.3g}" for k, v in self.info.items() if isinstance(v, (int, float)))]
        lines.append(f"{'parameter':<22} {'mean':>11} {'std':>10} {'16%':>11} {'50%':>11} {'84%':>11}")
        for n in self.names:
            lo, med, hi = pc[n]
            lines.append(f"{n:<22} {self.mean[n]:>11.5g} {self.std[n]:>10.3g} {lo:>11.5g} {med:>11.5g} {hi:>11.5g}")
        return "\n".join(lines)

    def to_dict(self):
        return dict(method=self.method, names=self.names, mean=self.mean, std=self.std,
                    percentiles={k: list(v) for k, v in self.percentiles().items()},
                    info={k: v for k, v in self.info.items() if isinstance(v, (int, float, str))})


def _weighted_residuals(fit, names, weights):
    sw = np.sqrt(weights)
    n_pen = len(fit._penalty())

    def resid(x):
        try:
            r = fit.residuals(x, names)
        except (ValueError, KeyError, ZeroDivisionError):
            return np.full(sw.size + n_pen, 1e10)
        r = r.copy()
        r[:sw.size] *= sw
        return r
    return resid


def bootstrap(fit, n=100, seed=0, max_nfev=200):
    """Resample points with replacement (per dataset) and refit from the current best values."""
    names = fit.names
    x_best = fit.model.params.get_vector(names)
    lo, hi = fit.model.params.bounds(names)
    sizes = [len(ds) for ds in fit.datasets]
    rng = np.random.default_rng(seed)
    out, n_fail = [], 0
    for _ in range(n):
        w = np.concatenate([np.bincount(rng.integers(0, m, m), minlength=m) for m in sizes]).astype(float)
        res = least_squares(_weighted_residuals(fit, names, w), x_best, bounds=(lo, hi), method="trf",
                            x_scale="jac", max_nfev=max_nfev)
        if res.success:
            out.append(res.x)
        else:
            n_fail += 1
    fit.set_x(x_best, names)
    return SampleResult(names, out, "bootstrap", n=n, failed=n_fail, seed=seed)


def mcmc(fit, nwalkers=None, nsteps=600, burn=200, thin=1, seed=0, init_scale=0.5, errors=None,
         temper_by_red_chi2=False):
    """Sample exp(-chi2/2) with flat priors inside the parameter bounds (needs emcee).

    Walkers start in a Gaussian ball around the current values with init_scale times the given
    errors (default: from the covariance at the current point). With temper_by_red_chi2=True the
    log-likelihood is divided by the reduced chi2 of the starting point, which widens the posterior
    the same way the covariance errors are scaled (use it when the model does not fit to within
    the error bars)."""
    try:
        import emcee
    except ImportError as ex:
        raise ImportError("MCMC needs emcee: pip install emcee") from ex
    names = fit.names
    ndim = len(names)
    x0 = fit.model.params.get_vector(names)
    lo, hi = fit.model.params.bounds(names)
    res = fit.report()
    if errors is None:
        errors = res.errors
    temper = res.fom["red_chi2"] if temper_by_red_chi2 and np.isfinite(res.fom["red_chi2"]) else 1.0
    err = np.array([errors[n] if np.isfinite(errors[n]) and errors[n] > 0 else 1e-3 * (h - l)
                    for n, l, h in zip(names, lo, hi)])
    nwalkers = nwalkers or max(2 * ndim + 2, 16)
    rng = np.random.default_rng(seed)
    p0 = x0 + init_scale * err * rng.standard_normal((nwalkers, ndim))
    p0 = np.clip(p0, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))

    def log_prob(x):
        if np.any(x < lo) or np.any(x > hi):
            return -np.inf
        try:
            r = fit.residuals(x, names)
        except (ValueError, KeyError, ZeroDivisionError):
            return -np.inf
        return -0.5 * float(r @ r) / temper

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(p0, nsteps, progress=False)
    chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
    try:
        tau = float(np.max(sampler.get_autocorr_time(discard=burn, tol=0)))
    except Exception:          # chain too short for an estimate
        tau = float("nan")
    fit.set_x(x0, names)
    return SampleResult(names, chain, "mcmc", walkers=nwalkers, steps=nsteps, burn=burn,
                        acceptance=float(np.mean(sampler.acceptance_fraction)), autocorr_steps=tau, seed=seed,
                        temper=temper)


def information_criteria(chi2, n_points, n_params):
    """AIC = chi2 + 2k, AICc = AIC + 2k(k+1)/(N-k-1), BIC = chi2 + k ln N (up to a shared constant)."""
    k, N = n_params, n_points
    aic = chi2 + 2 * k
    aicc = aic + (2 * k * (k + 1) / (N - k - 1) if N - k - 1 > 0 else math.inf)
    return dict(aic=aic, aicc=aicc, bic=chi2 + k * math.log(N))


def compare_models(results, labels=None):
    """Rank fit results of the same data by AIC and BIC. Returns (rows, text)."""
    labels = labels or [f"model {i + 1}" for i in range(len(results))]
    ics = [information_criteria(r.fom["chi2"], r.fom["N"], r.fom["p"]) for r in results]
    best_aic = min(ic["aic"] for ic in ics)
    best_bic = min(ic["bic"] for ic in ics)
    w = np.array([math.exp(-0.5 * (ic["aic"] - best_aic)) for ic in ics])
    w /= w.sum()
    rows = [dict(label=lab, p=r.fom["p"], chi2=r.fom["chi2"], red_chi2=r.fom["red_chi2"], **ic,
                 d_aic=ic["aic"] - best_aic, d_bic=ic["bic"] - best_bic, akaike_weight=float(wi))
            for lab, r, ic, wi in zip(labels, results, ics, w)]
    rows.sort(key=lambda d: d["bic"])
    text = [f"{'model':<24} {'p':>3} {'chi2':>10} {'dAIC':>8} {'dBIC':>8} {'weight':>7}"]
    text += [f"{d['label']:<24} {d['p']:>3} {d['chi2']:>10.2f} {d['d_aic']:>8.2f} {d['d_bic']:>8.2f} "
             f"{d['akaike_weight']:>7.3f}" for d in rows]
    text.append("dBIC > 6: strong evidence against that model; < 2: not distinguishable.")
    return rows, "\n".join(text)


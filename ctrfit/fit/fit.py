"""Fitting API.

    fit = Fit(model, datasets)                    # free parameters: model.params[...].fit
    fit.run("de", maxiter=40, popsize=12, seed=1) # global search (bounded, fixed seed, no polish)
    fit.refine()                                  # least squares (trf, bounded)
    res = fit.report()                            # values, errors, correlations, FOMs, warnings
    print(res.summary); res.save("result.json")

One scale per dataset: with several datasets each gets its own scoped scale ("<scope>.scale").
Free per-rod scales ("rodscale_H_K") get a penalty residual (value - 1) / rod_scale_penalty.
"""
import json
import time

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from . import fom as F_
from .sampling import information_criteria
from .uncertainty import covariance_from_jacobian, correlation, fit_warnings

BAD = 1e300


class FitCancelled(Exception):
    """Raised inside the objective when fit.cancel() was called (e.g. a Stop button)."""


class Fit:
    def __init__(self, model, datasets, fom="chi2", rod_scale_penalty=0.05):
        self.model = model
        self.datasets = list(datasets) if isinstance(datasets, (list, tuple)) else [datasets]
        if fom not in F_.FOMS:
            raise ValueError(f"fom must be one of {F_.FOMS}")
        self.fom_name = fom
        self.rod_scale_penalty = rod_scale_penalty
        if len(self.datasets) > 1:
            for i, ds in enumerate(self.datasets):
                ds.scope = ds.scope or f"ds{i + 1}"
                name = f"{ds.scope}.scale"
                if name not in model.params:
                    g = model.params["scale"]
                    model.params.add(name=name, value=g.value, min=g.min, max=g.max, fit=g.fit, unit=g.unit,
                                     group="scale", description=f"Scale of dataset {ds.name}")
            model.params["scale"].fit = False
        self.evaluators = [model.evaluator(ds.H, ds.K, ds.L, scope=ds.scope) for ds in self.datasets]
        self.history = []
        self.stages = []
        self._jac = None
        self._n_eval = 0
        self.cancelled = False
        self.progress = None          # optional callable(stage, step, fom), e.g. for a live plot
        self.on_best = None           # optional callable(stage, step, fom, values): new best values, once per
                                      # DE generation and whenever least squares improves (live model curves)

    def cancel(self):
        """Stop the running optimizer at its next evaluation; the best values so far are kept."""
        self.cancelled = True

    def _notify(self, stage, step, fom):
        if self.progress is not None:
            self.progress(stage, step, fom)

    def _notify_best(self, stage, step, fom, x, names):
        """Pass the values of parameter vector x (all parameters, links evaluated) to on_best."""
        if self.on_best is not None:
            self.on_best(stage, step, fom, dict(self.model.params.set_vector(x, names)))

    # ------------------------------------------------------------------ evaluation
    @property
    def names(self):
        return self.model.params.free_names

    @property
    def n_data(self):
        return sum(len(ds) for ds in self.datasets)

    def set_x(self, x, names=None):
        self.model.params.set_vector(x, names or self.names)

    def model_F(self, x=None, names=None):
        if x is not None:
            self.set_x(x, names)
        self._n_eval += 1
        return [ev(self.model.params.view(ds.scope)) for ev, ds in zip(self.evaluators, self.datasets)]

    def _penalty(self):
        if not self.rod_scale_penalty:
            return np.zeros(0)
        v = self.model.params.values()
        return np.array([(v[n] - 1.0) / self.rod_scale_penalty for n in self.names
                         if n.rpartition(".")[2].startswith("rodscale_")])

    def residuals(self, x=None, names=None):
        """chi residuals of all datasets plus rod-scale penalties."""
        Fc = self.model_F(x, names)
        r = [F_.chi_residuals(ds.F, f, ds.sigma_eff) for ds, f in zip(self.datasets, Fc)]
        return np.concatenate(r + [self._penalty()])

    def fom(self, x=None, name=None, names=None):
        name = name or self.fom_name
        if name == "chi2":
            return float(np.sum(self.residuals(x, names) ** 2))
        Fc = np.concatenate(self.model_F(x, names))
        Fo = np.concatenate([ds.F for ds in self.datasets])
        pen = float(np.sum(self._penalty() ** 2))
        return (F_.log_fom(Fo, Fc) if name == "log" else F_.r1(Fo, Fc)) + 1e-4 * pen

    def _safe(self, fn):
        def wrapped(x):
            try:
                out = fn(x)
            except (ValueError, KeyError, FloatingPointError, ZeroDivisionError):
                return None
            return out
        return wrapped

    # ------------------------------------------------------------------ optimizers
    def run(self, method="de", **kw):
        method = method.lower()
        if method in ("de", "differential_evolution"):
            return self.run_de(**kw)
        if method in ("lsq", "least_squares", "refine"):
            return self.refine(**kw)
        raise ValueError(f"unknown method {method!r} (use 'de' or 'lsq')")

    def run_de(self, maxiter=50, popsize=12, seed=1, tol=1e-6, mutation=(0.5, 1.0), recombination=0.7,
               init="latinhypercube", fom=None, include_start=True):
        names = self.names
        lo, hi = self.model.params.bounds(names)
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            bad = [n for n, a, b in zip(names, lo, hi) if not (np.isfinite(a) and np.isfinite(b))]
            raise ValueError(f"differential evolution needs finite bounds; set min/max for {bad}")
        fom = fom or self.fom_name
        x0 = np.clip(self.model.params.get_vector(names), lo, hi)
        best = dict(f=np.inf, x=x0.copy())
        safe = self._safe(lambda x: self.fom(x, fom, names))

        def objective(x):
            if self.cancelled:
                raise FitCancelled()
            f = safe(x)
            f = BAD if f is None or not np.isfinite(f) else f
            if f < best["f"]:
                best["f"], best["x"] = f, np.array(x, float)
            return f

        gen = [0]

        def callback(xk, convergence=None):
            gen[0] += 1
            self.history.append(dict(stage="de", step=gen[0], fom=float(best["f"]), fom_name=fom))
            self._notify("de", gen[0], float(best["f"]))
            self._notify_best("de", gen[0], float(best["f"]), best["x"], names)

        t0 = time.perf_counter()
        n0 = self._n_eval
        try:
            res = differential_evolution(objective, list(zip(lo, hi)), maxiter=maxiter, popsize=popsize, seed=seed,
                                         tol=tol, mutation=mutation, recombination=recombination, init=init,
                                         polish=False, x0=x0 if include_start else None, updating="immediate",
                                         callback=callback)
            x, f_best, msg, nit = (best["x"] if best["f"] <= res.fun else res.x), min(best["f"], res.fun), \
                str(res.message), int(res.nit)
        except FitCancelled:
            x, f_best, msg, nit = best["x"], best["f"], "stopped by user", gen[0]
        self.set_x(x, names)
        self.stages.append(dict(method="de", nfev=int(self._n_eval - n0), nit=nit, fom=float(f_best),
                                fom_name=fom, seconds=time.perf_counter() - t0, message=msg))
        self._jac = None
        return self.stages[-1]

    def refine(self, max_nfev=None, ftol=1e-10, xtol=1e-10, gtol=1e-10):
        names = self.names
        lo, hi = self.model.params.bounds(names)
        span = np.where(np.isfinite(hi - lo), hi - lo, 1.0)
        x0 = np.clip(self.model.params.get_vector(names), lo + 1e-9 * span, hi - 1e-9 * span)
        n_res = len(self.residuals(x0, names))
        safe = self._safe(lambda x: self.residuals(x, names))
        step = [0]
        best = dict(f=np.inf, x=x0.copy())

        def resid(x):
            if self.cancelled:
                raise FitCancelled()
            r = safe(x)
            r = np.full(n_res, 1e10) if r is None or not np.all(np.isfinite(r)) else r
            step[0] += 1
            f = float(np.sum(r ** 2))
            improved = f < best["f"]
            if improved:
                best["f"], best["x"] = f, np.array(x, float)
            self.history.append(dict(stage="lsq", step=step[0], fom=f, fom_name="chi2"))
            self._notify("lsq", step[0], f)
            if improved:
                self._notify_best("lsq", step[0], f, x, names)
            return r

        t0 = time.perf_counter()
        try:
            res = least_squares(resid, x0, bounds=(lo, hi), method="trf", x_scale="jac", ftol=ftol, xtol=xtol,
                                gtol=gtol, max_nfev=max_nfev)
        except FitCancelled:
            self.set_x(best["x"], names)
            self._jac = None
            self.stages.append(dict(method="lsq", nfev=step[0], fom=float(best["f"]), fom_name="chi2",
                                    seconds=time.perf_counter() - t0, message="stopped by user", status=-2))
            return self.stages[-1]
        self.set_x(res.x, names)
        self._jac = (names, res.x.copy(), np.array(res.jac, float))
        self.stages.append(dict(method="lsq", nfev=int(res.nfev), fom=float(2 * res.cost), fom_name="chi2",
                                seconds=time.perf_counter() - t0, message=str(res.message), status=int(res.status)))
        return self.stages[-1]

    def profile(self, name, grid, refine=True, best=True, polish=3):
        """chi2 profile: fix `name` at each grid value and refine the other free parameters.

        Shows separate minima (thickness fringes give minima about one trilayer apart, traded
        against eps_perp). With best=True the parameters end at the best point found: the `polish`
        lowest grid points are refined again with `name` free too (a grid point next to the true
        minimum can score worse than one in a wrong basin when the minimum is narrow), and the
        lowest chi2 wins. Returns the grid as a list of dicts value, chi2, values.
        """
        params = self.model.params
        p = params[name]
        was_fit, start = p.fit, params.values()
        settable = [n for n in params.names if not params[n].linked]

        def restore(vals):
            params.update({n: vals[n] for n in settable})

        def chi2_now():
            return float(np.sum(self.residuals() ** 2))

        others = [n for n in self.names if n != name]
        p.fit = False
        out = []
        for g in grid:
            if self.cancelled:
                break
            restore(start)
            p.value = float(g)
            if refine and others:
                self.refine()
            out.append(dict(value=float(g), chi2=chi2_now(), values=params.values()))
        p.fit = was_fit
        if not (best and out):
            restore(start)
            self._jac = None
            return out
        candidates = [dict(chi2=d["chi2"], values=d["values"]) for d in out]
        if was_fit and polish:
            for d in sorted(out, key=lambda d: d["chi2"])[:polish]:
                restore(d["values"])
                self.refine()
                candidates.append(dict(chi2=chi2_now(), values=params.values()))
        restore(min(candidates, key=lambda d: d["chi2"])["values"])
        self._jac = None
        return out

    # ------------------------------------------------------------------ uncertainty and report
    def jacobian(self, names=None, rel_step=1e-6):
        """d(residuals)/d(free parameters) by central differences at the current values."""
        names = names or self.names
        x = self.model.params.get_vector(names)
        lo, hi = self.model.params.bounds(names)
        span = np.where(np.isfinite(hi - lo), hi - lo, np.maximum(np.abs(x), 1.0))
        cols = []
        for i in range(len(x)):
            h = rel_step * span[i]
            xp, xm = x.copy(), x.copy()
            xp[i] = min(x[i] + h, hi[i])
            xm[i] = max(x[i] - h, lo[i])
            cols.append((self.residuals(xp, names) - self.residuals(xm, names)) / (xp[i] - xm[i]))
        self.set_x(x, names)
        return np.array(cols).T

    def _current_jacobian(self, names):
        if self._jac is not None:
            jn, jx, J = self._jac
            if jn == names and np.array_equal(jx, self.model.params.get_vector(names)):
                return J
        return self.jacobian(names)

    def report(self):
        names = self.names
        x = self.model.params.get_vector(names)
        Fc = self.model_F(x, names)
        Fo = np.concatenate([ds.F for ds in self.datasets])
        so = np.concatenate([ds.sigma_eff for ds in self.datasets])
        Fcat = np.concatenate(Fc)
        foms = F_.all_foms(Fo, Fcat, so, len(names))
        foms.update(information_criteria(foms["chi2"], foms["N"], foms["p"]))
        per_ds = {ds.name: F_.all_foms(ds.F, f, ds.sigma_eff, len(names)) for ds, f in zip(self.datasets, Fc)}
        cov, cond = covariance_from_jacobian(self._current_jacobian(names), foms["red_chi2"]) if names else \
            (np.zeros((0, 0)), 1.0)
        values = self.model.params.values()
        errors = {n: float(np.sqrt(max(cov[i, i], 0))) for i, n in enumerate(names)}
        errors.update(self._linked_errors(names, x, cov))
        corr = correlation(cov) if names else np.zeros((0, 0))
        warn = fit_warnings(self.model.params, names, values, errors, corr, cond)
        return FitResult(names=names, values=values, errors=errors, correlation=corr, cov=cov, fom=foms,
                         fom_per_dataset=per_ds, warnings=warn, history=list(self.history), stages=list(self.stages),
                         model=self.model.to_dict(), datasets=[ds.name for ds in self.datasets], condition=cond,
                         dataset_meta=[dict(name=ds.name, scope=ds.scope, **ds.meta) for ds in self.datasets])

    def _linked_errors(self, names, x, cov):
        linked = [p.name for p in self.model.params if p.linked]
        if not linked or not names:
            return {}
        lo, hi = self.model.params.bounds(names)
        G = np.zeros((len(linked), len(names)))
        for i in range(len(names)):
            h = 1e-6 * (hi[i] - lo[i] if np.isfinite(hi[i] - lo[i]) else max(abs(x[i]), 1.0))
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            vp = self.model.params.set_vector(xp, names)
            vm = self.model.params.set_vector(xm, names)
            G[:, i] = [(vp[n] - vm[n]) / (2 * h) for n in linked]
        self.model.params.set_vector(x, names)
        var = np.einsum("ij,jk,ik->i", G, cov, G)
        return {n: float(np.sqrt(max(v, 0))) for n, v in zip(linked, var)}


class FitResult:
    def __init__(self, names, values, errors, correlation, cov, fom, fom_per_dataset, warnings, history, stages,
                 model, datasets, condition=1.0, dataset_meta=None):
        self.names, self.values, self.errors = list(names), dict(values), dict(errors)
        self.dataset_meta = list(dataset_meta or [dict(name=n, scope="") for n in datasets])
        self.correlation, self.cov = np.asarray(correlation), np.asarray(cov)
        self.fom, self.fom_per_dataset = fom, fom_per_dataset
        self.warnings, self.history, self.stages = list(warnings), list(history), list(stages)
        self.model, self.datasets, self.condition = model, list(datasets), float(condition)

    def error(self, name):
        return self.errors.get(name, float("nan"))

    @property
    def summary(self):
        f = self.fom
        lines = [f"Fit of {', '.join(self.datasets)}: {f['N']} points, {f['p']} free parameters",
                 f"chi2 = {f['chi2']:.4g}, reduced chi2 = {f['red_chi2']:.4g}, log FOM = {f['log']:.4g}, "
                 f"R1 = {f['R1']:.4g}, AIC = {f['aic']:.4g}, BIC = {f['bic']:.4g}"]
        for st in self.stages:
            lines.append(f"  {st['method']}: {st['nfev']} evaluations, {st['seconds']:.1f} s, "
                         f"{st['fom_name']} = {st['fom']:.6g}")
        lines.append("")
        lines.append(f"{'parameter':<22} {'value':>12} {'error':>10}")
        for n in self.names:
            lines.append(f"{n:<22} {self.values[n]:>12.6g} {self.errors.get(n, float('nan')):>10.3g}")
        linked = [n for n in self.errors if n not in self.names]
        for n in linked:
            lines.append(f"{n:<22} {self.values[n]:>12.6g} {self.errors[n]:>10.3g}  (linked)")
        if len(self.names) > 1:
            lines.append("")
            lines.append("correlations above 0.5:")
            pairs = [(abs(self.correlation[i, j]), self.names[i], self.names[j], self.correlation[i, j])
                     for i in range(len(self.names)) for j in range(i + 1, len(self.names))]
            strong = sorted([p for p in pairs if p[0] > 0.5], reverse=True)
            lines += [f"  {a:<20} {b:<20} {r:+.3f}" for _, a, b, r in strong] or ["  none"]
        lines.append("")
        lines += ["Warnings:"] + [f"  - {w}" for w in self.warnings] if self.warnings else ["No warnings."]
        return "\n".join(lines)

    def to_dict(self):
        return dict(names=self.names, values=self.values, errors=self.errors,
                    correlation=self.correlation.tolist(), cov=self.cov.tolist(), fom=self.fom,
                    fom_per_dataset=self.fom_per_dataset, warnings=self.warnings, history=self.history,
                    stages=self.stages, model=self.model, datasets=self.datasets, condition=self.condition,
                    dataset_meta=self.dataset_meta,
                    summary=self.summary)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1, default=float)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        d.pop("summary", None)
        return cls(**d)

"""Model evaluations on several cores, in separate processes (threads do not help: the evaluation
holds the interpreter lock, and numpy's own threads already compete for the cores).

Used by Fit when fit.cores > 1:
- differential evolution evaluates each generation's trials in parallel (scipy's `workers`, with
  'deferred' updating: reproducible for a seed, but not the same trajectory as one core);
- least squares computes the Jacobian columns in parallel (forward differences with scipy's step).
Each worker process holds its own copy of the fit (model, datasets), sent once per optimizer run,
and sets only the varied parameters per evaluation. Workers run numpy single-threaded so N processes
use N cores. The pool is started once (about a second per worker) and kept for later fits.

    fit = Fit(model, datasets, cores=4)        # or fit.cores = 4
    fit.run("de"); fit.refine()
"""
import atexit
import multiprocessing as mp
import os
import pickle
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout

import numpy as np

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")
MIRRORED = ("xraydb",)          # a module blocked in this process (sys.modules[name] = None) is blocked in workers too

_pool = None
_pool_n = 0


def available_cores():
    return os.cpu_count() or 1


def default_cores():
    """Half the logical CPUs (about the physical cores), at least 1."""
    return max(1, available_cores() // 2)


def _ping(delay):
    time.sleep(delay)
    return os.getpid()


def get_pool(n):
    """The worker pool of n processes (started now if needed; a pool of another size is replaced)."""
    global _pool, _pool_n
    if _pool is not None and _pool_n == n:
        return _pool
    shutdown()
    saved = {k: os.environ.get(k) for k in THREAD_VARS}
    os.environ.update({k: "1" for k in THREAD_VARS})          # inherited by the workers at their start
    try:
        pool = ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"))
        list(pool.map(_ping, [0.3] * n))                        # start every worker now, while the variables hold
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _pool, _pool_n = pool, n
    return pool


def shutdown():
    """Stop the worker processes (the window does this when it closes; also at exit)."""
    global _pool, _pool_n
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
    _pool, _pool_n = None, 0


atexit.register(shutdown)


class Cancelled(Exception):
    pass


class ParallelFit:
    """The evaluations of one optimizer run of `fit` on n worker processes."""

    def __init__(self, fit, n):
        self.n = int(n)
        self.pool = get_pool(self.n)
        self.sid = uuid.uuid4().hex
        blocked = [m for m in MIRRORED if m in sys.modules and sys.modules[m] is None]
        self.state = pickle.dumps(dict(model=fit.model.to_dict(), datasets=[d.to_dict() for d in fit.datasets],
                                       fom=fit.fom_name, penalty=fit.rod_scale_penalty, blocked=blocked))

    def map(self, kind, X, extra, cancelled=lambda: False):
        """[result per row of X] in order; kind 'de' gives (fom, scales) or None, 'resid' a residual vector."""
        X = np.atleast_2d(np.asarray(X, float))
        if not len(X):
            return []
        chunks = [c for c in np.array_split(X, min(len(X), 2 * self.n)) if len(c)]
        futs = [self.pool.submit(_work, self.sid, self.state, kind, c, extra) for c in chunks]
        out = []
        try:
            for f in futs:
                while True:
                    if cancelled():
                        raise Cancelled()
                    try:
                        out += f.result(timeout=0.1)
                        break
                    except FutureTimeout:
                        continue
        except BaseException:
            for f in futs:
                f.cancel()
            raise
        return out


# ---------------------------------------------------------------------- in the worker processes
_FITS = {}


def _fit_for(sid, state):
    fit = _FITS.get(sid)
    if fit is None:
        d = pickle.loads(state)
        for m in d["blocked"]:
            sys.modules[m] = None
        from ..data.dataset import Dataset
        from ..model.model import Model
        from .fit import Fit
        fit = Fit(Model.from_dict(d["model"]), [Dataset.from_dict(x) for x in d["datasets"]], fom=d["fom"],
                  rod_scale_penalty=d["penalty"])
        _FITS.clear()                       # one fit at a time
        _FITS[sid] = fit
    return fit


def _work(sid, state, kind, X, extra):
    fit = _fit_for(sid, state)
    out = []
    if kind == "de":
        names, fom, groups = extra
        for x in X:
            try:
                f, sc = fit._solved_fom(x, names, fom, groups)
                out.append((float(f), sc) if np.isfinite(f) else None)
            except (ValueError, KeyError, FloatingPointError, ZeroDivisionError):
                out.append(None)
    elif kind == "resid":
        names = extra
        for x in X:
            try:
                r = fit.residuals(x, names)
                out.append(r if np.all(np.isfinite(r)) else None)
            except (ValueError, KeyError, FloatingPointError, ZeroDivisionError):
                out.append(None)
    else:
        raise ValueError(f"unknown work {kind!r}")
    return out

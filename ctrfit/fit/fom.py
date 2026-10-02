"""Figures of merit for |F| data. F: measured, Fc: model, s: effective sigma.

chi2   sum(((F - Fc) / s)^2); reduced: chi2 / (N - p)
log    GenX-style logarithmic figure of merit: sum(|log10 F - log10 Fc|) / (N - 1)
R1     sum(|F - Fc|) / sum(|F|)
"""
import numpy as np

TINY = 1e-30


def chi_residuals(F, Fc, s):
    return (np.asarray(F) - np.asarray(Fc)) / np.asarray(s)


def chi2(F, Fc, s):
    return float(np.sum(chi_residuals(F, Fc, s) ** 2))


def reduced_chi2(F, Fc, s, n_par):
    dof = np.size(F) - n_par
    return chi2(F, Fc, s) / dof if dof > 0 else float("nan")


def log_residuals(F, Fc):
    return np.log10(np.maximum(np.asarray(F, float), TINY)) - np.log10(np.maximum(np.asarray(Fc, float), TINY))


def log_fom(F, Fc):
    n = np.size(F)
    return float(np.sum(np.abs(log_residuals(F, Fc))) / max(n - 1, 1))


def r1(F, Fc):
    F = np.asarray(F, float)
    return float(np.sum(np.abs(F - np.asarray(Fc))) / np.sum(np.abs(F)))


FOMS = ("chi2", "log", "R1")


def all_foms(F, Fc, s, n_par):
    return dict(chi2=chi2(F, Fc, s), red_chi2=reduced_chi2(F, Fc, s, n_par), log=log_fom(F, Fc), R1=r1(F, Fc),
                N=int(np.size(F)), p=int(n_par))

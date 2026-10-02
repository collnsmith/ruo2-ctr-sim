"""Synthetic data from a model, with reproducible noise.

F_obs = F_true * (1 + e_rod) + N(0, sigma) + N(0, sys_floor * F_true)

sigma = noise_rel * F_true (relative counting noise, reported as the error bar), plus an optional
absolute floor sigma_abs; e_rod ~ N(0, rod_scale_err) is one scale error per rod. The systematic
floor is not part of the reported sigma but is stored as dataset.sys_floor, so sigma_eff includes it.
"""
import numpy as np

from .dataset import Dataset


def rod_points(rods, L_min=0.3, L_max=5.0, step=0.05, avoid_bragg=0.08):
    """(H, K, L) arrays for whole rods, skipping points closer than avoid_bragg to integer L where
    TiO2 Bragg peaks sit (they also mask the film peaks on real data)."""
    L = np.arange(L_min, L_max + 1e-9, step)
    if avoid_bragg > 0:
        L = L[np.abs(L - np.round(L)) >= avoid_bragg]
    H = np.concatenate([np.full(L.size, h) for h, _ in rods])
    K = np.concatenate([np.full(L.size, k) for _, k in rods])
    return H, K, np.tile(L, len(rods))


def make_dataset(model, rods=None, points=None, noise_rel=0.03, sigma_abs=0.0, sys_floor=0.0,
                 rod_scale_err=0.0, seed=0, name="synthetic", values=None, L_min=0.3, L_max=5.0, step=0.05,
                 avoid_bragg=0.08, **meta):
    """Simulate a dataset from `model` at its current parameter values (or `values`).
    Points: `points` = (H, K, L), or whole `rods` sampled by rod_points(L_min, L_max, step,
    avoid_bragg). Extra keywords (energy_kev, potential_V, thickness_nm) are stored as metadata."""
    if points is None:
        points = rod_points(rods, L_min, L_max, step, avoid_bragg)
    H, K, L = points
    ev = model.evaluator(H, K, L)
    F_true = ev(values)
    rng = np.random.default_rng(seed)
    labels = np.array([f"{h} {k}" for h, k in zip(H, K)])
    uniq = list(dict.fromkeys(labels.tolist()))
    e_rod = dict(zip(uniq, rng.normal(0.0, rod_scale_err, len(uniq)) if rod_scale_err > 0 else np.zeros(len(uniq))))
    sigma = np.sqrt((noise_rel * F_true) ** 2 + sigma_abs ** 2)
    F = (F_true * (1 + np.array([e_rod[lab] for lab in labels])) + rng.normal(0.0, 1.0, F_true.size) * sigma
         + rng.normal(0.0, 1.0, F_true.size) * sys_floor * F_true)
    meta.setdefault("notes", f"synthetic, seed {seed}, noise {noise_rel:g}, floor {sys_floor:g}, "
                             f"rod scale {rod_scale_err:g}")
    ds = Dataset(H, K, L, np.abs(F), sigma, labels, name=name, sys_floor=sys_floor, **meta)
    ds.truth = dict(F=F_true, rod_scale={k: 1 + v for k, v in e_rod.items()})
    return ds

"""Fast evaluator: equals the reference engine (rtol 1e-9), caches correctly, and is fast."""
import time
from pathlib import Path

import numpy as np
import pytest

from ctr_params import DEFAULTS
from ctrfit.model.model import Model

REF = np.load(Path(__file__).resolve().parent / "reference.npz")
RODS = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 5)]


def stack(rods, L):
    H = np.concatenate([np.full(L.size, h) for h, _ in rods])
    K = np.concatenate([np.full(L.size, k) for _, k in rods])
    return H, K, np.tile(L, len(rods))


@pytest.mark.parametrize("case, over, key", [
    ("default", {}, "default_I"), ("relaxed", dict(thickness_nm=10.0, relax_001=0.5), "relaxed_I"),
    ("domains", dict(mix_mode="domains"), "default_domains"),
])
def test_fast_equals_reference(case, over, key):
    model = Model.from_settings(dict(DEFAULTS, **over))
    ev = model.evaluator(*stack(RODS, REF["L"]))
    ref = np.concatenate([REF[f"{key}_{H}{K}"] for H, K in RODS])
    np.testing.assert_allclose(ev.intensity(), ref, rtol=1e-9, atol=0)
    np.testing.assert_allclose(ev(), np.sqrt(ref), rtol=1e-9, atol=0)     # scale 1


def _random_values(model, rng, names):
    v = model.params.view()
    for n in names:
        p = model.params[n]
        lo, hi = max(p.min, -1e3), min(p.max, 1e3)
        v[n] = float(rng.uniform(lo + 0.1 * (hi - lo), hi - 0.6 * (hi - lo)) if n.endswith("_tl")
                     else rng.uniform(lo, hi))
    v["thickness_mean_tl"] = float(rng.uniform(4, 14))
    v["coherent_tl"] = float(rng.integers(2, 10))
    v["roughness_tl"] = float(rng.uniform(0, 1.5))
    v["thickness_spread_tl"] = float(rng.uniform(0, 2))
    return v


@pytest.mark.parametrize("seed, over", [
    (0, {}), (1, dict(mix_mode="domains")), (2, dict(relaxed_rod_mode="coherent_only")),
    (3, dict(include_H=False, water_on=False)), (4, dict(eps_perp_mode="elastic", energy_kev=13.0)),
])
def test_fast_equals_reference_random_parameters(seed, over):
    rng = np.random.default_rng(seed)
    model = Model.from_settings(dict(DEFAULTS, **over), water_layer=True)
    rods = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (0, 3)]
    L = np.sort(rng.uniform(0.3, 4.0, 40))
    ev = model.evaluator(*stack(rods, L))
    for _ in range(3):
        v = _random_values(model, rng, [n for n in model.params.names if n != "scale"])
        v["relax_001"] = float(rng.choice([0.0, rng.uniform(0.1, 1.0)]))
        ref = np.concatenate([model.intensity(H, K, L, v) for H, K in rods])
        np.testing.assert_allclose(ev.intensity(v), ref, rtol=1e-9, atol=0)


def test_caching_follows_parameter_changes():
    model = Model.from_settings(DEFAULTS)
    rods = [(0, 0), (0, 1), (1, 1)]
    L = np.linspace(0.3, 4.0, 50)
    ev = model.evaluator(*stack(rods, L))
    steps = [dict(z_OH=2.3), dict(theta=0.7), dict(thickness_mean_tl=18.4), dict(eps_perp=2.4), dict(b_sub_O=0.8),
             dict(z_OH=2.0), dict(el_z0=3.0), dict(relax_001=0.4, coherent_tl=10), dict(x_OH=0.2)]
    v = model.params.view()
    for change in steps:
        v.update(change)
        ref = np.concatenate([model.intensity(H, K, L, v) for H, K in rods])
        np.testing.assert_allclose(ev.intensity(dict(v)), ref, rtol=1e-9, atol=0)
    st = ev.fast.stats
    assert st["sub:hit"] > 0 and st["sub:miss"] == 2           # substrate rebuilt only for b_sub_O


def test_surface_only_reuses_substrate_and_buried_film():
    model = Model.from_settings(DEFAULTS)
    ev = model.evaluator(*stack([(0, 1), (1, 0)], np.linspace(0.3, 4, 100)))
    v = model.params.view()
    for z in np.linspace(1.8, 2.4, 6):
        v["z_OH"] = z
        ev(v)
    st = ev.fast.stats
    for name in ("sub", "Ab", "UEc", "layers"):
        assert st[f"{name}:miss"] == 1, name
    assert st["Sb0:miss"] == 6


def test_rod_scales_and_scale():
    model = Model.from_settings(DEFAULTS)
    model.add_rod_scales([(0, 1)])
    L = np.linspace(0.5, 3, 20)
    ev = model.evaluator(*stack([(0, 1), (1, 0)], L))
    base = ev()
    model.params["scale"].value = 2.0
    model.params["rodscale_0_1"].value = 1.5
    F = ev()
    np.testing.assert_allclose(F[:20], 3.0 * base[:20])
    np.testing.assert_allclose(F[20:], 2.0 * base[20:])
    np.testing.assert_allclose(F[:20], model.F(0, 1, L), rtol=1e-12)


def test_rejects_fractional_rod_indices():
    with pytest.raises(ValueError):
        Model.from_settings(DEFAULTS).evaluator([0.5], [1], [1.0])


def test_benchmark_surface_only_2000_points(record_property, capsys):
    """About 2000 points on 8 rods, surface parameters free: fast evaluator vs legacy rod by rod."""
    model = Model.from_settings(DEFAULTS)
    rods = [(0, 1), (1, 0), (0, 3), (0, 5), (0, 0), (1, 1), (0, 2), (2, 0)]
    L = np.linspace(0.31, 5.0, 250)
    ev = model.evaluator(*stack(rods, L))
    ev()                                                        # compile and fill the caches
    rng = np.random.default_rng(0)
    names = ["z_OH", "x_OH", "theta", "B_OH", "dz_Obr"]

    def perturbed():
        v = model.params.view()
        for n in names:
            p = model.params[n]
            v[n] = float(np.clip(v[n] + 0.02 * rng.standard_normal(), p.min, p.max))
        return v

    n_fast = 30
    t_fast = min(_timed(lambda: [ev(perturbed()) for _ in range(n_fast)]) for _ in range(3)) / n_fast
    v = perturbed()
    t_legacy = min(_timed(lambda: [model.F(H, K, L, v) for H, K in rods]) for _ in range(2))
    speedup = t_legacy / t_fast
    record_property("fast_ms", 1e3 * t_fast)
    record_property("legacy_ms", 1e3 * t_legacy)
    with capsys.disabled():
        print(f"\n[benchmark] {ev.n} points, surface-only: fast {1e3 * t_fast:.2f} ms/eval, "
              f"legacy rod by rod {1e3 * t_legacy:.1f} ms, speedup {speedup:.0f}x")
    assert speedup >= 10


def _timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t

"""3D viewer: scene building, CPU renderer vs CUDA kernel (simulator), and float32-only PTX."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SKIP = 77


def _helper(mode, **env):
    r = subprocess.run([sys.executable, str(HERE / "viewer_check.py"), mode], capture_output=True, text=True,
                       timeout=300, env=dict(os.environ, **env))
    if r.returncode == SKIP:
        pytest.skip(r.stdout.strip()[-300:] or "package missing")
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_scene_builds():
    pytest.importorskip("numba")
    pytest.importorskip("PyQt5")
    import ctr_viewer3d as v
    from ctr_params import DEFAULTS
    sc = v.build_scene(dict(DEFAULTS), nx=4, ny=2, n_sub=2, seed=1, show_H=True, show_water=True)
    n = len(sc["el"])
    assert sc["xyz"].shape == (n, 3) and len(sc["role"]) == n and len(sc["label"]) == n
    assert set(sc["role"]) <= set(v.COLORS) and set(sc["el"]) <= set(v.RADIUS)
    m = sc["model"]
    assert sc["top_k"].shape == (4, 2) and sc["top_k"].min() >= 1 and sc["top_k"].max() <= m.n_film + 10
    assert {"Ru", "Ti", "O"} <= set(sc["el"])
    rad = np.array([v.RADIUS[e] for e in sc["el"]], np.float32)
    gp, gd, start, items = v.build_grid(sc["xyz"], rad)
    assert start[-1] == len(items) and np.all(np.diff(start) >= 0)


def test_cpu_matches_cuda_simulator():
    pytest.importorskip("numba")
    res = _helper("sim", NUMBA_ENABLE_CUDASIM="1")
    assert res["n_hit"] > 0 and res["n_hit"] < res["n_px"]    # atoms and background both in view
    assert res["ids_equal"]
    assert res["max_diff"] <= 2


def test_cuda_kernel_ptx_is_float32_only():
    pytest.importorskip("numba")
    res = _helper("ptx")
    assert res["target_sm75"]
    assert res["n_f64"] == 0, "\n".join(res["f64"])

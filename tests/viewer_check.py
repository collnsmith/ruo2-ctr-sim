"""Helper run in a subprocess by test_viewer.py (Numba's CUDA mode is chosen at import time).

  python viewer_check.py sim   CPU renderer vs CUDA kernel in the simulator (NUMBA_ENABLE_CUDASIM=1)
  python viewer_check.py ptx   compile the kernel to PTX for sm_75 and list f64 instructions
Exit code 77 means a needed package is missing (the test skips).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SKIP = 77


def tiny_scene():
    import numpy as np
    import ctr_viewer3d as v
    from ctr_params import DEFAULTS
    sc = v.build_scene(dict(DEFAULTS), nx=3, ny=2, n_sub=1, seed=3)
    rad = np.array([v.RADIUS[e] for e in sc["el"]], np.float32)
    col = np.array([v.COLORS[r] for r in sc["role"]], np.float32)
    grid = v.build_grid(sc["xyz"], rad)
    cam = v.OrbitCamera()
    cam.frame_scene(sc["xyz"], float(sc["surface_z"].max()))
    return v, sc, rad, col, grid, cam


def run_sim():
    import numpy as np
    try:
        import ctr_viewer3d as v
    except ImportError as ex:
        print(f"missing: {ex}")
        return SKIP
    if v.cuda is None or not v.HAVE_CUDA:
        print("numba CUDA simulator not available")
        return SKIP
    v, sc, rad, col, grid, cam = tiny_scene()
    W, H = 12, 8
    light = np.array([-0.45, -0.55, 0.70])
    light = (light / np.linalg.norm(light)).astype(np.float32)
    camv = cam.vectors(W / H).astype(np.float32)
    args_scene = [np.ascontiguousarray(a) for a in (sc["xyz"], rad, col, *grid)]
    flags = np.array([1, 1], np.int32)
    res = {}
    for name in ("cpu", "sim"):
        accum = np.zeros((H, W, 3), np.float32)
        out = np.zeros((H, W, 3), np.uint8)
        ids = np.full((H, W), -1, np.int32)
        for frame in (0, 1):
            args = (W, H, camv, light, *args_scene, frame, flags, np.float32(3.5), accum, out, ids)
            if name == "cpu":
                v.render_cpu(*args)
            else:
                kern = v.make_cuda_kernel()[0]
                kern[(1, 1), (W, H)](*args)
        res[name] = (out.copy(), ids.copy())
    (o_cpu, i_cpu), (o_sim, i_sim) = res["cpu"], res["sim"]
    print(json.dumps(dict(ids_equal=bool(np.array_equal(i_cpu, i_sim)),
                          n_hit=int((i_cpu >= 0).sum()), n_px=W * H,
                          max_diff=int(np.abs(o_cpu.astype(int) - o_sim.astype(int)).max()))))
    return 0


def types_ns(**kw):
    from types import SimpleNamespace
    return SimpleNamespace(**kw)


class _FakeDevice:
    compute_capability = (7, 5)
    name = b"fake sm_75"
    id = 0


def run_ptx():
    try:
        from numba import cuda, types
        import numba.cuda.dispatcher as disp_mod
    except Exception as ex:          # numba-cuda missing or its runtime libraries not found
        print(f"missing: {ex}")
        return SKIP
    import numba.cuda.codegen as codegen_mod
    import ctr_viewer3d as v
    # no GPU here: both places that ask the driver for the compute capability get a fake sm_75 device
    disp_mod.get_current_device = lambda: _FakeDevice()
    codegen_mod.devices = types_ns(get_context=lambda *a, **k: types_ns(device=_FakeDevice()))
    disp_mod._Kernel.bind = lambda self: None      # loading the module needs the driver; PTX does not
    kern = v.make_cuda_kernel()[0]
    f32, i32, u8 = types.float32, types.int32, types.uint8
    sig = (types.int64, types.int64, f32[::1], f32[::1], f32[:, ::1], f32[::1], f32[:, ::1], f32[::1],
           i32[::1], i32[::1], i32[::1], types.int64, i32[::1], f32, f32[:, :, ::1], u8[:, :, ::1], i32[:, ::1])
    try:
        kern.compile(sig)
        ptx = kern.inspect_asm(sig)
    except Exception as ex:
        if "NVVM" in str(ex) or "libnvvm" in str(ex).lower():
            print(f"missing: {ex}")
            return SKIP
        raise
    lines = [ln.strip() for ln in ptx.splitlines()
             if re.search(r"\.f64\b", ln) and not ln.strip().startswith("//")]
    print(json.dumps(dict(target_sm75=".target sm_75" in ptx, f64=lines[:20], n_f64=len(lines))))
    return 0


if __name__ == "__main__":
    sys.exit({"sim": run_sim, "ptx": run_ptx}[sys.argv[1]]())

# @app title: 3D film viewer | group: Simulate | order: 20 | kind: gui | needs: PyQt5, numba | desc: Ray-traced real-space model with roughness islands, CUS species and the X-ray beams
"""3D viewer for the RuO2(110)/TiO2(110) film model, ray traced on the GPU with Numba CUDA.

Builds one random real-space version of the model in ctr_engine.py: the film thickness, the
roughness islands (from the model's layer occupancies), the strain spacings, and the CUS species
drawn from the surface composition. It renders that with shadows, ambient occlusion and
progressive anti-aliasing.

Run:   python ctr_viewer3d.py
Needs: numpy, scipy, numba (with CUDA), PyQt5. Without a CUDA GPU it falls back to the CPU (slower).

Mouse: left drag = orbit, right or middle drag = pan, wheel = zoom, click an atom = identify it.
Keys:  R reset view, Space turntable, S screenshot, A ambient occlusion, H hydrogens, W water,
       B X-ray beams (incident beam and exit beam to the selected H K and L).
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from ctr_engine import ADSORBATE_H, CTRModel          # noqa: E402
from ctrfit.core.geometry import beam_geometry        # noqa: E402
from ctr_params import DEFAULTS, parse_comp, read_ini  # noqa: E402

try:  # same Qt choice as the main GUI
    from PyQt5 import QtCore, QtGui, QtWidgets
    QT_API = "PyQt5"
except ImportError:  # pragma: no cover
    from PySide6 import QtCore, QtGui, QtWidgets
    QT_API = "PySide6"

from numba import float32, njit, prange, uint8  # noqa: E402

try:
    from numba import cuda
    HAVE_CUDA = bool(cuda.is_available())
except Exception:  # numba without CUDA support
    cuda, HAVE_CUDA = None, False

# ==============================================================================================
# scene: one random real-space version of the model
# ==============================================================================================
RADIUS = {"Ru": 0.85, "Ti": 0.80, "O": 0.66, "H": 0.33}           # display radii (Å) at size 1
COLORS = {                                                         # linear RGB
    "Ru": (0.12, 0.42, 0.72),
    "Ti": (0.66, 0.68, 0.72),
    "O": (0.80, 0.14, 0.13),        # lattice O
    "O_OH": (1.00, 0.52, 0.06),     # O of OH on a CUS Ru
    "O_H2O": (0.05, 0.72, 0.80),    # O of H2O on a CUS Ru
    "O_O": (0.62, 0.36, 0.85),      # oxo O on a CUS Ru
    "H": (0.93, 0.93, 0.93),
    "O_w": (0.55, 0.72, 0.95),      # electrolyte water
}
LEGEND = [("Ru", "Ru"), ("Ti", "Ti (substrate)"), ("O", "lattice O"), ("O_OH", "OH on CUS"),
          ("O_H2O", "H2O on CUS"), ("O_O", "O (oxo) on CUS"), ("H", "H"), ("O_w", "electrolyte water")]


def _trilayer(m, metal, parity, z0, d, u, relax=None, skip=()):
    at = m._new()
    m.add_trilayer(at, metal, parity, z0, d, u, 1.0, 0.0, 0.0, "t", skip=skip, relax=relax)
    at = m._done(at)
    return at["el"], at["f1"], at["f2"], at["z"], [t.split(":")[1] for t in at["tag"]]


def build_scene(params, nx=8, ny=4, n_sub=4, island_A=10.0, seed=1, show_H=True, show_water=False):
    """Return dict with centers (N,3), element/role keys, labels, and the model."""
    m = CTRModel(params)
    comp = parse_comp(params["comp_default"], "surface state")
    rng = np.random.default_rng(seed)
    A1, A2 = m.A1, m.A2
    atoms = []  # (x, y, z, element, role, label)

    def add(el, role, x, y, z, label):
        atoms.append((x, y, z, el, role, label))

    # top trilayer index per column: correlated random field mapped onto the model's distribution
    ks, zl, dl, is_top, occ, exposed = m.film_layers(m.n_film)
    prob = exposed / exposed.sum()
    field = rng.normal(size=(nx, ny))
    kx = np.fft.fftfreq(nx)[:, None] / A1
    ky = np.fft.fftfreq(ny)[None, :] / A2
    field = np.fft.ifft2(np.fft.fft2(field) * np.exp(-2 * (np.pi * island_A) ** 2 * (kx ** 2 + ky ** 2))).real
    order = np.argsort(field.ravel())
    cum = np.cumsum(prob)
    top_k = np.empty(nx * ny, int)
    for rank, idx in enumerate(order):
        top_k[idx] = ks[min(np.searchsorted(cum, (rank + 0.5) / (nx * ny)), len(ks) - 1)]
    top_k = top_k.reshape(nx, ny)

    u_f, u_s = params["ruo2_u"], params["tio2_u"]
    bulk = {k: _trilayer(m, "Ru", k % 2, zl[i], dl[i], u_f) for i, k in enumerate(ks)}
    expo = {k: _trilayer(m, "Ru", k % 2, zl[i], dl[i], u_f, relax=m.relax_top, skip=("M_cus",))
            for i, k in enumerate(ks)}
    zpl = dict(zip(ks, zl))
    subs = [_trilayer(m, "Ti", j % 2, -j * m.D_SUB, m.D_SUB, u_s) for j in range(n_sub)]
    species = [s for s in ("OH", "H2O", "O") if comp.get(s, 0) > 0]
    weights = [comp[s] for s in species] + [max(0.0, 1 - sum(comp.values()))]
    surface_z = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            ox, oy = i * A1, j * A2
            for js, (el, f1, f2, z, names) in enumerate(subs):
                for e, a, b, c, nm in zip(el, f1, f2, z, names):
                    add(e, e, ox + a * A1, oy + b * A2, c, f"TiO2 substrate, trilayer {js + 1} from the top, {nm}")
            kt = top_k[i, j]
            for k in ks[ks <= kt]:
                el, f1, f2, z, names = expo[k] if k == kt else bulk[k]
                where = "relaxed part" if is_top[k - 1] else "commensurate part"
                for e, a, b, c, nm in zip(el, f1, f2, z, names):
                    add(e, e, ox + a * A1, oy + b * A2, c, f"RuO2 trilayer {k} ({where}), {nm}")
            # CUS site of the exposed trilayer
            z0, s = zpl[kt], 0.5 * (kt % 2)
            cx, cy = ox, oy + ((0.5 + s) % 1.0) * A2
            sp = rng.choice(species + ["empty"], p=np.array(weights) / sum(weights))
            surface_z[i, j] = z0
            if sp == "empty":
                add("Ru", "Ru", cx, cy, z0 + m.relax_top["M_cus_vacant"], f"CUS Ru (empty site), trilayer {kt}")
                continue
            ad = m.adsorbates[sp]
            zM = z0 + ad["dz_Mcus"]
            zO = zM + ad["z_O"]
            add("Ru", "Ru", cx, cy, zM, f"CUS Ru under {sp}, trilayer {kt}")
            add("O", f"O_{sp}", cx, cy, zO, f"{sp} oxygen on CUS Ru, {ad['z_O']:.2f} Å above it")
            if show_H:
                phi = rng.uniform(0, 2 * np.pi)   # random azimuth (display only)
                for dx, dy, dz in ADSORBATE_H[sp]:
                    hx, hy = dx * np.cos(phi) - dy * np.sin(phi), dx * np.sin(phi) + dy * np.cos(phi)
                    add("H", "H", cx + hx, cy + hy, zO + dz, f"H of {sp} on CUS Ru")
            for xl in m.extra_layers:
                if rng.random() < xl["occ"]:
                    site = xl["site"]
                    f1x, f2x = {"cus": (0.0, 0.5), "br": (0.0, 0.0)}[site] if isinstance(site, str) else site
                    add(xl["el"], "O_w" if xl["el"] == "O" else xl["el"], ox + f1x * A1,
                        oy + ((f2x + s) % 1.0) * A2, z0 + xl["z"], "extra ordered layer")

    if show_water and m.electrolyte["on"]:
        _add_water(atoms, m, rng, nx, ny, surface_z, show_H)

    xyz = np.array([[a[0], a[1], a[2]] for a in atoms], np.float32)
    return dict(xyz=xyz, el=[a[3] for a in atoms], role=[a[4] for a in atoms],
                label=[a[5] for a in atoms], model=m, top_k=top_k,
                extent=(nx * A1, ny * A2), surface_z=surface_z)


def _add_water(atoms, m, rng, nx, ny, surface_z, show_H):
    """Random water molecules at bulk density above the local surface (display only)."""
    A1, A2 = m.A1, m.A2
    spacing = (1.0 / (m.electrolyte["rho"] / 10.0)) ** (1 / 3)     # 10 electrons per molecule
    occupied = cKDTree(np.array([[a[0], a[1], a[2]] for a in atoms]))
    zlo = surface_z.min() + m.electrolyte["z0"] - 0.5
    gx, gy, gz = (np.arange(0, nx * A1, spacing), np.arange(0, ny * A2, spacing),
                  np.arange(zlo, surface_z.max() + m.electrolyte["z0"] + 9.0, spacing))
    for x in gx:
        for y in gy:
            col_z = surface_z[min(int(x / A1), nx - 1), min(int(y / A2), ny - 1)] + m.electrolyte["z0"]
            for z in gz:
                p = np.array([x, y, z]) + rng.uniform(-0.08, 0.08, 3) * spacing
                if p[2] < col_z or occupied.query(p)[0] < 2.6:
                    continue
                atoms.append((p[0], p[1], p[2], "O", "O_w", "electrolyte water (random, display only)"))
                if show_H:
                    a, b = rng.normal(size=3), rng.normal(size=3)
                    a /= np.linalg.norm(a)
                    b -= a * b.dot(a)
                    b /= np.linalg.norm(b)
                    for sgn in (1, -1):
                        h = p + 0.957 * (np.cos(0.9119) * a + sgn * np.sin(0.9119) * b)
                        atoms.append((h[0], h[1], h[2], "H", "H", "electrolyte water H"))


def build_grid(xyz, rad, cell=1.6):
    """Uniform grid for ray traversal: each sphere is listed in every cell its box touches."""
    pad = float(rad.max()) + 0.05
    gmin = xyz.min(0) - pad
    gmax = xyz.max(0) + pad
    dims = np.maximum(1, np.ceil((gmax - gmin) / cell)).astype(np.int32)
    lo = np.clip(np.floor((xyz - rad[:, None] - gmin) / cell).astype(int), 0, dims - 1)
    hi = np.clip(np.floor((xyz + rad[:, None] - gmin) / cell).astype(int), 0, dims - 1)
    cells, items = [], []
    for s in range(len(xyz)):
        ix, iy, iz = np.meshgrid(np.arange(lo[s, 0], hi[s, 0] + 1), np.arange(lo[s, 1], hi[s, 1] + 1),
                                 np.arange(lo[s, 2], hi[s, 2] + 1), indexing="ij")
        c = ((ix * dims[1] + iy) * dims[2] + iz).ravel()
        cells.append(c)
        items.append(np.full(c.size, s))
    cells, items = np.concatenate(cells), np.concatenate(items)
    order = np.argsort(cells, kind="stable")
    cells, items = cells[order], items[order].astype(np.int32)
    start = np.searchsorted(cells, np.arange(int(np.prod(dims)) + 1)).astype(np.int32)
    gp = np.array([gmin[0], gmin[1], gmin[2], cell], np.float32)
    return gp, dims.astype(np.int32), start, items


# ==============================================================================================
# ray tracer, written once and compiled for CUDA and for the CPU
# ==============================================================================================
def make_shader(dev):
    INF = np.float32(1e30)
    EPS = np.float32(2e-3)
    Z = np.float32(0.0)
    ONE = np.float32(1.0)
    TWO = np.float32(2.0)
    TAU = np.float32(2 * np.pi)
    INV32 = np.float32(2.3283064e-10)

    @dev
    def hash32(x):
        x = (x ^ 61) ^ (x >> 16)
        x = (x * 9) & 0xFFFFFFFF
        x = x ^ (x >> 4)
        x = (x * 0x27D4EB2D) & 0xFFFFFFFF
        return x ^ (x >> 15)

    @dev
    def hit_sphere(cx, cy, cz, r, ox, oy, oz, dx, dy, dz):
        lx, ly, lz = cx - ox, cy - oy, cz - oz
        b = lx * dx + ly * dy + lz * dz
        disc = b * b - (lx * lx + ly * ly + lz * lz - r * r)
        if disc < Z:
            return INF
        s = math.sqrt(disc)
        t = b - s
        if t > EPS:
            return t
        t = b + s
        if t > EPS:
            return t
        return INF

    @dev
    def axis_setup(o, d, gmin, cell, n, t):
        """Cell index, step, distance to next boundary, and boundary spacing along one axis."""
        p = o + d * t
        i = int((p - gmin) / cell)
        i = max(0, min(n - 1, i))
        if d > Z:
            return i, 1, (gmin + float32(i + 1) * cell - o) / d, cell / d
        if d < Z:
            return i, -1, (gmin + float32(i) * cell - o) / d, -cell / d
        return i, 0, INF, INF

    @dev
    def slab(o, d, lo, hi, t0, t1):
        if d == Z:
            if o < lo or o > hi:
                return ONE, Z           # empty interval
            return t0, t1
        a = (lo - o) / d
        b = (hi - o) / d
        if a > b:
            a, b = b, a
        return max(t0, a), min(t1, b)

    @dev
    def traverse(ox, oy, oz, dx, dy, dz, tmax, any_hit, cen, rad, gp, gd, cstart, citems):
        nx, ny, nz = gd[0], gd[1], gd[2]
        cell = gp[3]
        t0, t1 = slab(ox, dx, gp[0], gp[0] + float32(nx) * cell, Z, tmax)
        t0, t1 = slab(oy, dy, gp[1], gp[1] + float32(ny) * cell, t0, t1)
        t0, t1 = slab(oz, dz, gp[2], gp[2] + float32(nz) * cell, t0, t1)
        if t0 > t1:
            return INF, -1
        ix, sx, tnx, tdx = axis_setup(ox, dx, gp[0], cell, nx, t0)
        iy, sy, tny, tdy = axis_setup(oy, dy, gp[1], cell, ny, t0)
        iz, sz, tnz, tdz = axis_setup(oz, dz, gp[2], cell, nz, t0)
        best_t = tmax
        best = -1
        while True:
            c = (ix * ny + iy) * nz + iz
            for q in range(cstart[c], cstart[c + 1]):
                s = citems[q]
                th = hit_sphere(cen[s, 0], cen[s, 1], cen[s, 2], rad[s], ox, oy, oz, dx, dy, dz)
                if th < best_t:
                    best_t = th
                    best = s
                    if any_hit:
                        return best_t, best
            tn = min(tnx, min(tny, tnz))
            if best_t <= tn or tn > t1:
                break
            if tnx <= tny and tnx <= tnz:
                ix += sx
                tnx += tdx
                if ix < 0 or ix >= nx:
                    break
            elif tny <= tnz:
                iy += sy
                tny += tdy
                if iy < 0 or iy >= ny:
                    break
            else:
                iz += sz
                tnz += tdz
                if iz < 0 or iz >= nz:
                    break
        return best_t, best

    @dev
    def shade(px, py, W, H, cam, light, cen, rad, col, gp, gd, cstart, citems, frame, flags, ao_r):
        seed = hash32((py * 7919 + px) * 104729 + frame * 15485863 + 1)
        jx = np.float32(0.5)
        jy = np.float32(0.5)
        if frame > 0:                       # sub-pixel jitter = progressive anti-aliasing
            seed = hash32(seed)
            jx = float32(seed) * INV32
            seed = hash32(seed)
            jy = float32(seed) * INV32
        sx = (TWO * (float32(px) + jx) / float32(W) - ONE) * cam[12] * cam[13]
        sy = (ONE - TWO * (float32(py) + jy) / float32(H)) * cam[12]
        dx = cam[3] + sx * cam[6] + sy * cam[9]
        dy = cam[4] + sx * cam[7] + sy * cam[10]
        dz = cam[5] + sx * cam[8] + sy * cam[11]
        inv = ONE / math.sqrt(dx * dx + dy * dy + dz * dz)
        dx *= inv
        dy *= inv
        dz *= inv
        t, s = traverse(cam[0], cam[1], cam[2], dx, dy, dz, INF, 0, cen, rad, gp, gd, cstart, citems)
        if s < 0:                           # background: soft vertical gradient
            g = np.float32(0.5) * (sy / cam[12] + ONE)
            return (np.float32(0.010) + np.float32(0.030) * g, np.float32(0.013) + np.float32(0.040) * g,
                    np.float32(0.020) + np.float32(0.060) * g, -1)
        hx = cam[0] + dx * t
        hy = cam[1] + dy * t
        hz = cam[2] + dz * t
        r = rad[s]
        nxv = (hx - cen[s, 0]) / r
        nyv = (hy - cen[s, 1]) / r
        nzv = (hz - cen[s, 2]) / r
        ox = hx + nxv * EPS
        oy = hy + nyv * EPS
        oz = hz + nzv * EPS
        lx, ly, lz = light[0], light[1], light[2]
        ndl = max(Z, nxv * lx + nyv * ly + nzv * lz)
        lit = ONE
        if flags[0] != 0 and ndl > Z:
            _, blk = traverse(ox, oy, oz, lx, ly, lz, INF, 1, cen, rad, gp, gd, cstart, citems)
            if blk >= 0:
                lit = Z
        hxv, hyv, hzv = lx - dx, ly - dy, lz - dz
        hn = ONE / math.sqrt(hxv * hxv + hyv * hyv + hzv * hzv)
        spec = math.pow(max(Z, (nxv * hxv + nyv * hyv + nzv * hzv) * hn), np.float32(48.0)) * np.float32(0.35) * lit
        ao = ONE
        if flags[1] != 0:                   # one cosine-weighted occlusion sample per frame
            seed = hash32(seed + 17)
            u1 = float32(seed) * INV32
            seed = hash32(seed)
            u2 = float32(seed) * INV32
            rr = math.sqrt(u1)
            a = TAU * u2
            lxs, lys, lzs = rr * math.cos(a), rr * math.sin(a), math.sqrt(max(Z, ONE - u1))
            if abs(nxv) < np.float32(0.9):
                tx, ty, tz = Z, -nzv, nyv
            else:
                tx, ty, tz = nzv, Z, -nxv
            tn = ONE / math.sqrt(tx * tx + ty * ty + tz * tz)
            tx *= tn
            ty *= tn
            tz *= tn
            bx, by, bz = nyv * tz - nzv * ty, nzv * tx - nxv * tz, nxv * ty - nyv * tx
            ddx = tx * lxs + bx * lys + nxv * lzs
            ddy = ty * lxs + by * lys + nyv * lzs
            ddz = tz * lxs + bz * lys + nzv * lzs
            _, occ = traverse(ox, oy, oz, ddx, ddy, ddz, ao_r, 1, cen, rad, gp, gd, cstart, citems)
            if occ >= 0:
                ao = Z
        fill = max(Z, -(nxv * lx + nyv * ly) * np.float32(0.6) + nzv * np.float32(0.2))
        k = np.float32(0.32) * ao + np.float32(0.78) * ndl * lit + np.float32(0.12) * fill * ao
        return col[s, 0] * k + spec, col[s, 1] * k + spec, col[s, 2] * k + spec, s

    return shade


def _tonemap(v):
    v = max(np.float32(0.0), min(np.float32(1.0), v))
    return v ** np.float32(1.0 / 2.2) * np.float32(255.0) + np.float32(0.5)


_shade_cpu = make_shader(lambda f: njit(fastmath=True)(f))
_tonemap_cpu = njit(fastmath=True)(_tonemap)


@njit(parallel=True, fastmath=True)
def render_cpu(W, H, cam, light, cen, rad, col, gp, gd, cstart, citems, frame, flags, ao_r, accum, out, ids):
    n = np.float32(frame + 1)
    for y in prange(H):
        for x in range(W):
            r, g, b, s = _shade_cpu(x, y, W, H, cam, light, cen, rad, col, gp, gd, cstart, citems, frame,
                                    flags, ao_r)
            if frame == 0:
                accum[y, x, 0], accum[y, x, 1], accum[y, x, 2] = r, g, b
                ids[y, x] = s
            else:
                accum[y, x, 0] += r
                accum[y, x, 1] += g
                accum[y, x, 2] += b
            out[y, x, 0] = uint8(_tonemap_cpu(accum[y, x, 0] / n))
            out[y, x, 1] = uint8(_tonemap_cpu(accum[y, x, 1] / n))
            out[y, x, 2] = uint8(_tonemap_cpu(accum[y, x, 2] / n))


def make_cuda_kernel():
    shade = make_shader(lambda f: cuda.jit(device=True, fastmath=True)(f))
    tonemap = cuda.jit(device=True, fastmath=True)(_tonemap)

    def kernel(W, H, cam, light, cen, rad, col, gp, gd, cstart, citems, frame, flags, ao_r, accum, out, ids):
        x, y = cuda.grid(2)
        if x >= W or y >= H:
            return
        r, g, b, s = shade(x, y, W, H, cam, light, cen, rad, col, gp, gd, cstart, citems, frame, flags, ao_r)
        if frame == 0:
            accum[y, x, 0] = r
            accum[y, x, 1] = g
            accum[y, x, 2] = b
            ids[y, x] = s
        else:
            accum[y, x, 0] += r
            accum[y, x, 1] += g
            accum[y, x, 2] += b
        n = float32(frame + 1)
        out[y, x, 0] = uint8(tonemap(accum[y, x, 0] / n))
        out[y, x, 1] = uint8(tonemap(accum[y, x, 1] / n))
        out[y, x, 2] = uint8(tonemap(accum[y, x, 2] / n))

    return cuda.jit(fastmath=True)(kernel), kernel


class Renderer:
    """Holds scene buffers on the GPU (or CPU) and renders frames into an RGB array."""

    def __init__(self, prefer_cuda=True):
        self.gpu = prefer_cuda and HAVE_CUDA
        self.kernel = make_cuda_kernel()[0] if self.gpu else None
        self.W = self.H = 0
        self.scene_dev = None

    @property
    def name(self):
        if not self.gpu:
            return "CPU (CUDA not available)"
        dev = cuda.get_current_device()
        n = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
        return f"CUDA: {n}"

    def set_scene(self, cen, rad, col, gp, gd, cstart, citems):
        arrs = [np.ascontiguousarray(a) for a in (cen, rad, col, gp, gd, cstart, citems)]
        self.scene_dev = [cuda.to_device(a) for a in arrs] if self.gpu else arrs

    def resize(self, W, H):
        self.W, self.H = W, H
        self.out = np.zeros((H, W, 3), np.uint8)
        if self.gpu:
            self.d_accum = cuda.device_array((H, W, 3), np.float32)
            self.d_out = cuda.device_array((H, W, 3), np.uint8)
            self.d_ids = cuda.device_array((H, W), np.int32)
        else:
            self.d_accum = np.zeros((H, W, 3), np.float32)
            self.d_out = self.out
            self.d_ids = np.full((H, W), -1, np.int32)

    def render(self, cam, light, frame, flags, ao_r):
        cam, light, flags = (np.asarray(cam, np.float32), np.asarray(light, np.float32),
                             np.asarray(flags, np.int32))
        args = (self.W, self.H, cam, light, *self.scene_dev, frame, flags, np.float32(ao_r),
                self.d_accum, self.d_out, self.d_ids)
        if self.gpu:
            tpb = (16, 8)
            blocks = ((self.W + tpb[0] - 1) // tpb[0], (self.H + tpb[1] - 1) // tpb[1])
            self.kernel[blocks, tpb](*args)
            self.d_out.copy_to_host(self.out)
        else:
            render_cpu(*args)
        return self.out

    def pick(self, x, y):
        if not (0 <= x < self.W and 0 <= y < self.H):
            return -1
        if self.gpu:
            return int(self.d_ids[y:y + 1, x:x + 1].copy_to_host()[0, 0])
        return int(self.d_ids[y, x])


# ==============================================================================================
# camera
# ==============================================================================================
class OrbitCamera:
    def __init__(self):
        self.target = np.zeros(3)
        self.yaw, self.pitch, self.dist, self.fov = -0.65, 0.42, 60.0, 35.0

    def frame_scene(self, xyz, top_z):
        """Look down at the surface region, with the film's side faces in view."""
        lo, hi = xyz.min(0), xyz.max(0)
        lateral = float(max(hi[0] - lo[0], hi[1] - lo[1]))
        self.target = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, top_z - 0.25 * lateral])
        self.dist = 3.3 * lateral
        self.yaw, self.pitch = -0.75, 0.55

    def vectors(self, aspect):
        cp = math.cos(self.pitch)
        offset = np.array([cp * math.cos(self.yaw), cp * math.sin(self.yaw), math.sin(self.pitch)])
        origin = self.target + self.dist * offset
        fwd = -offset
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return np.concatenate([origin, fwd, right, up, [math.tan(math.radians(self.fov) / 2), aspect]])

    def pan(self, dx, dy, height_px):
        cam = self.vectors(1.0)
        scale = 2 * self.dist * cam[12] / max(height_px, 1)
        self.target += (-dx * cam[6:9] + dy * cam[9:12]) * scale


# ==============================================================================================
# X-ray beams (drawn as an overlay on the rendered image)
# ==============================================================================================
BEAM_IN = (255, 214, 64)      # incident beam
BEAM_OUT = (80, 220, 255)     # exit (diffracted) beam


def project(cam, pts, W, H):
    """Pixel coordinates of 3D points for camera vector `cam` (same convention as the shader).
    Returns (N, 2) array and a mask of points in front of the camera."""
    pts = np.atleast_2d(np.asarray(pts, float))
    o, f, r, u = cam[0:3], cam[3:6], cam[6:9], cam[9:12]
    tan, aspect = cam[12], cam[13]
    d = pts - o
    depth = d @ f
    front = depth > 1e-6
    depth = np.where(front, depth, 1e-6)
    sx = (d @ r) / depth / (tan * aspect)
    sy = (d @ u) / depth / tan
    return np.column_stack([(sx + 1) / 2 * W, (1 - sy) / 2 * H]), front


def visible(cam, pts, centers, radii, margin=0.05):
    """True for points not hidden behind any sphere as seen from the camera."""
    o = cam[0:3].astype(float)
    d = np.asarray(pts, float) - o
    dist = np.linalg.norm(d, axis=1)
    u = d / dist[:, None]
    oc = centers.astype(float) - o
    b = u @ oc.T                                       # (points, spheres)
    c = np.einsum("ij,ij->i", oc, oc) - radii.astype(float) ** 2
    disc = b ** 2 - c[None, :]
    t = b - np.sqrt(np.clip(disc, 0, None))
    hidden = (disc > 0) & (t > 1e-3) & (t < dist[:, None] - margin)
    return ~hidden.any(axis=1)


def beam_lines(scene, geo, length=None):
    """3D end points of the incident and exit beams through the centre of the visible surface:
    (spot, start of the incident beam, end of the exit beam)."""
    ex, ey = scene["extent"]
    spot = np.array([ex / 2, ey / 2, float(scene["surface_z"].max())])
    length = length or 0.9 * max(ex, ey)
    kin = geo["k_in"] / np.linalg.norm(geo["k_in"])
    kout = geo["k_out"] / np.linalg.norm(geo["k_out"])
    return spot, spot - kin * length, spot + kout * length


# ==============================================================================================
# window
# ==============================================================================================
class View(QtWidgets.QWidget):
    picked = QtCore.pyqtSignal(int, int) if QT_API == "PyQt5" else QtCore.Signal(int, int)

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.image = None
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumSize(320, 240)
        self._press = None
        self._last = None

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(8, 11, 16))
        if self.image is not None:
            p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            p.drawImage(self.rect(), self.image)
        self.owner.draw_beams(p, self.width(), self.height())
        p.end()

    def mousePressEvent(self, ev):
        self._press = self._last = ev.pos()

    def mouseMoveEvent(self, ev):
        if self._last is None:
            return
        d = ev.pos() - self._last
        self._last = ev.pos()
        cam = self.owner.camera
        if ev.buttons() & QtCore.Qt.LeftButton:
            cam.yaw -= d.x() * 0.008
            cam.pitch = float(np.clip(cam.pitch + d.y() * 0.008, -1.5, 1.5))
        else:
            cam.pan(d.x(), d.y(), self.height())
        self.owner.dirty = True

    def mouseReleaseEvent(self, ev):
        if self._press is not None and (ev.pos() - self._press).manhattanLength() < 4:
            self.picked.emit(ev.pos().x(), ev.pos().y())
        self._press = self._last = None

    def wheelEvent(self, ev):
        self.owner.camera.dist *= 0.9 ** (ev.angleDelta().y() / 120)
        self.owner.dirty = True

    def keyPressEvent(self, ev):
        self.owner.key(ev.key())


class ViewerWindow(QtWidgets.QMainWindow):
    MAX_SAMPLES = 96

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RuO2 / TiO2(110) film, 3D")
        self.camera = OrbitCamera()
        self.dirty, self.frame, self.spin = True, 0, False
        self.scene = None
        self.seed = 1
        self.renderer = Renderer()
        self.view = View(self)
        self.view.picked.connect(self.pick)

        # ---------------- controls
        self.source = QtWidgets.QComboBox()
        self.source.setToolTip("Model settings to build the film from")
        self._fill_sources()
        self.thick = self._dspin(0.3, 30, 0.1, 2, " nm", "Film thickness, rounded to whole trilayers")
        self.rough = self._dspin(0.0, 3, 0.02, 3, " nm", "Top roughness (rms)")
        self.comp = QtWidgets.QLineEdit()
        self.comp.setToolTip("CUS sites, e.g. OH=0.5, H2O=0.5 (rest empty)")
        self.island = self._dspin(1, 60, 1, 1, " Å", "Lateral size of roughness islands (display only)")
        self.nx = self._ispin(1, 40, "Unit cells along [001]")
        self.ny = self._ispin(1, 20, "Unit cells along [1-10]")
        self.nsub = self._ispin(0, 20, "TiO2 trilayers shown under the film")
        self.size = self._dspin(0.3, 1.8, 0.05, 2, "", "Atom size")
        self.chk_H = QtWidgets.QCheckBox("Hydrogens")
        self.chk_w = QtWidgets.QCheckBox("Electrolyte water")
        self.chk_sh = QtWidgets.QCheckBox("Shadows")
        self.chk_ao = QtWidgets.QCheckBox("Ambient occlusion")
        self.chk_beam = QtWidgets.QCheckBox("X-ray beams (B)")
        self.chk_beam.setToolTip("Incident beam and the exit beam to the selected reflection (H K L)")
        self.beam_h = QtWidgets.QSpinBox()
        self.beam_k = QtWidgets.QSpinBox()
        for w, tip in ((self.beam_h, "H of the rod (along [001])"), (self.beam_k, "K of the rod (along [1-10])")):
            w.setRange(-10, 10)
            w.setToolTip(tip)
        self.beam_l = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.beam_l.setRange(0, 600)
        self.beam_l.setToolTip("L along the rod (r.l.u.)")
        self.beam_l_label = QtWidgets.QLabel()
        self.beam_l_label.setMinimumWidth(42)
        self.beam_alpha = self._dspin(0.02, 20.0, 0.05, 2, " °", "Incidence angle onto the surface (H, K not both 0); "
                                      "on the specular rod it follows from L")
        self.beam_info = QtWidgets.QLabel()
        self.beam_info.setWordWrap(True)
        self.beam = None
        self.scale = QtWidgets.QComboBox()
        self.scale.addItems(["100 %", "75 %", "50 %", "35 %"])
        self.scale.setToolTip("Render resolution relative to the window")
        b_build = QtWidgets.QPushButton("Rebuild")
        b_shuffle = QtWidgets.QPushButton("New random surface")
        b_view = QtWidgets.QPushButton("Reset view")
        b_shot = QtWidgets.QPushButton("Save image")
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.pick_label = QtWidgets.QLabel("Click an atom to identify it.")
        self.pick_label.setWordWrap(True)

        self.thick.setValue(DEFAULTS["thickness_nm"])
        self.rough.setValue(DEFAULTS["roughness_nm"])
        self.comp.setText(DEFAULTS["comp_default"])
        self.island.setValue(10.0)
        self.nx.setValue(14)
        self.ny.setValue(6)
        self.nsub.setValue(4)
        self.size.setValue(1.0)
        for c in (self.chk_H, self.chk_sh, self.chk_ao, self.chk_beam):
            c.setChecked(True)
        self.beam_h.setValue(0)
        self.beam_k.setValue(1)
        self.beam_l.setValue(150)
        self.beam_alpha.setValue(0.5)
        self.scale.setCurrentIndex(0 if self.renderer.gpu else 2)

        form = QtWidgets.QFormLayout()
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        form.addRow("Settings from", self.source)
        form.addRow("Thickness", self.thick)
        form.addRow("Roughness", self.rough)
        form.addRow("Surface", self.comp)
        form.addRow("Island size", self.island)
        form.addRow("Cells along [001]", self.nx)
        form.addRow("Cells along [1-10]", self.ny)
        form.addRow("Substrate trilayers", self.nsub)
        form.addRow("Atom size", self.size)
        form.addRow("Resolution", self.scale)
        side = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(side)
        sl.addLayout(form)
        for w in (self.chk_H, self.chk_w, self.chk_sh, self.chk_ao):
            sl.addWidget(w)
        beam_box = QtWidgets.QGroupBox("X-ray beam")
        bl = QtWidgets.QFormLayout(beam_box)
        bl.addRow(self.chk_beam)
        hk = QtWidgets.QHBoxLayout()
        hk.addWidget(QtWidgets.QLabel("H"))
        hk.addWidget(self.beam_h)
        hk.addWidget(QtWidgets.QLabel("K"))
        hk.addWidget(self.beam_k)
        bl.addRow("Rod", hk)
        lrow = QtWidgets.QHBoxLayout()
        lrow.addWidget(self.beam_l, 1)
        lrow.addWidget(self.beam_l_label)
        bl.addRow("L", lrow)
        bl.addRow("Incidence", self.beam_alpha)
        bl.addRow(self.beam_info)
        sl.addWidget(beam_box)
        row = QtWidgets.QGridLayout()
        row.addWidget(b_build, 0, 0)
        row.addWidget(b_shuffle, 0, 1)
        row.addWidget(b_view, 1, 0)
        row.addWidget(b_shot, 1, 1)
        sl.addLayout(row)
        sl.addWidget(self._legend())
        sl.addWidget(self.pick_label)
        sl.addStretch(1)
        sl.addWidget(self.info)
        side.setMinimumWidth(250)
        side.setMaximumWidth(340)

        split = QtWidgets.QSplitter()
        split.addWidget(side)
        split.addWidget(self.view)
        split.setStretchFactor(1, 1)
        split.setSizes([300, 1100])
        self.setCentralWidget(split)
        self.resize(1400, 860)

        b_build.clicked.connect(self.rebuild)
        b_shuffle.clicked.connect(self.reshuffle)
        b_view.clicked.connect(self.reset_view)
        b_shot.clicked.connect(self.screenshot)
        self.source.activated.connect(lambda *_: self.load_source())
        self.size.valueChanged.connect(lambda *_: self.upload())
        for c in (self.chk_sh, self.chk_ao):
            c.toggled.connect(self._touch)
        self.scale.currentIndexChanged.connect(self._touch)
        for c in (self.chk_H, self.chk_w):
            c.toggled.connect(lambda *_: self.rebuild())
        self.chk_beam.toggled.connect(lambda *_: self.update_beam())
        for w in (self.beam_h, self.beam_k, self.beam_alpha):
            w.valueChanged.connect(lambda *_: self.update_beam())
        self.beam_l.valueChanged.connect(lambda *_: self.update_beam())

        self.timer = QtCore.QTimer(self, interval=15)
        self.timer.timeout.connect(self.tick)
        self._t_last = time.perf_counter()
        self._fps = 0.0
        self.load_source(build=False)
        QtCore.QTimer.singleShot(50, self.first_build)

    # ---------------- helpers
    def _dspin(self, lo, hi, step, dec, suffix, tip):
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(dec)
        w.setSuffix(suffix)
        w.setToolTip(tip)
        w.setKeyboardTracking(False)
        return w

    def _ispin(self, lo, hi, tip):
        w = QtWidgets.QSpinBox()
        w.setRange(lo, hi)
        w.setToolTip(tip)
        w.setKeyboardTracking(False)
        return w

    def _legend(self):
        box = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(0, 8, 0, 8)
        for i, (key, text) in enumerate(LEGEND):
            sw = QtWidgets.QLabel()
            r, g, b = (int(255 * c ** (1 / 2.2)) for c in COLORS[key])
            sw.setFixedSize(14, 14)
            sw.setStyleSheet(f"background: rgb({r},{g},{b}); border-radius: 7px;")
            grid.addWidget(sw, i // 2, 2 * (i % 2))
            grid.addWidget(QtWidgets.QLabel(text), i // 2, 2 * (i % 2) + 1)
        return box

    def _fill_sources(self):
        self.source.clear()
        self.source.addItem("Defaults", None)
        if (APP_DIR / "settings.ini").exists():
            self.source.addItem("Main app (settings.ini)", APP_DIR / "settings.ini")
            self.source.setCurrentIndex(1)
        for p in sorted((APP_DIR / "presets").glob("*.ini")):
            self.source.addItem(f"Preset: {p.stem}", p)

    def _touch(self, *_):
        self.dirty = True

    def params(self):
        path = self.source.currentData()
        base = dict(DEFAULTS) if path is None else read_ini(path)[0]
        base.update(thickness_nm=self.thick.value(), roughness_nm=self.rough.value(),
                    comp_default=self.comp.text().strip())
        return base

    # ---------------- scene
    def load_source(self, build=True):
        path = self.source.currentData()
        vals = dict(DEFAULTS) if path is None else read_ini(path)[0]
        self.thick.setValue(vals["thickness_nm"])
        self.rough.setValue(vals["roughness_nm"])
        self.comp.setText(vals["comp_default"])
        if build:
            self.rebuild()

    def first_build(self):
        if self.renderer.gpu:
            self.info.setText("Compiling the CUDA kernel (first run only takes a few seconds)…")
        else:
            self.info.setText("Compiling the CPU renderer…")
        QtWidgets.QApplication.processEvents()
        self.rebuild(reframe=True)
        self.timer.start()

    def reshuffle(self):
        self.seed += 1
        self.rebuild()

    def rebuild(self, reframe=False):
        try:
            sc = build_scene(self.params(), self.nx.value(), self.ny.value(), self.nsub.value(),
                             self.island.value(), self.seed, self.chk_H.isChecked(), self.chk_w.isChecked())
        except (ValueError, KeyError) as ex:
            QtWidgets.QMessageBox.warning(self, "Check the settings", str(ex))
            return
        self.scene = sc
        if reframe:
            self.camera.frame_scene(sc["xyz"], float(sc["surface_z"].max()))
        self.upload()

    def upload(self):
        sc = self.scene
        if sc is None:
            return
        rad = np.array([RADIUS[e] for e in sc["el"]], np.float32) * np.float32(self.size.value())
        col = np.array([COLORS[r] for r in sc["role"]], np.float32)
        grid = build_grid(sc["xyz"], rad)
        self._rad = rad
        self.renderer.set_scene(sc["xyz"], rad, col, *grid)
        m = sc["model"]
        tk = sc["top_k"]
        self.info_base = (f"{self.renderer.name}\n{len(rad)} atoms, film {m.n_film} trilayers "
                          f"({m.n_film * m.D_FILM / 10:.2f} nm), surface trilayers {tk.min()} to {tk.max()}"
                          + ("; relaxed top" if m.relaxed else ""))
        self.dirty = True
        self.update_beam()

    # ---------------- X-ray beams
    def beam_hkl(self):
        return self.beam_h.value(), self.beam_k.value(), self.beam_l.value() / 100.0

    def update_beam(self):
        H, K, L = self.beam_hkl()
        self.beam_l_label.setText(f"{L:.2f}")
        self.beam = None
        if not self.chk_beam.isChecked() or self.scene is None:
            self.beam_info.setText("")
            self.view.update()
            return
        m = self.scene["model"]
        g = beam_geometry(m, H, K, L, self.beam_alpha.value())
        self.beam_alpha.setEnabled(not (H == 0 and K == 0))
        if g["k_in"] is None or not g["ok"]:
            self.beam_info.setText(f"({H} {K} {L:.2f}): {g['reason']}")
        else:
            self.beam = g
            self.beam_info.setText(
                f"({H} {K} {L:.2f}) at {12.398419843 / g['lam']:.2f} keV: 2θ {g['two_theta']:.2f}°, "
                f"incidence {g['alpha']:.2f}°, exit {g['beta']:.2f}°, sample azimuth {g['phi']:.1f}°, "
                f"|q| {np.linalg.norm(g['q']):.3f} 1/Å")
        self.view.update()

    def draw_beams(self, p, W, H):
        """Overlay of the incident beam, the exit beam and the reflection label (QPainter p)."""
        if self.beam is None or self.scene is None:
            return
        cam = self.camera.vectors(W / max(H, 1))
        spot, start, end = beam_lines(self.scene, self.beam)
        xy, front = project(cam, np.array([start, spot, end]), W, H)
        if not front.all():
            return
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        a, b, c = (QtCore.QPointF(*map(float, v)) for v in xy)
        n = 60
        for (p0, p1), col in (((start, spot), BEAM_IN), ((spot, end), BEAM_OUT)):
            t = np.linspace(0, 1, n + 1)[:, None]
            pts3 = p0 + t * (p1 - p0)
            pix, _ = project(cam, pts3, W, H)
            vis = visible(cam, 0.5 * (pts3[1:] + pts3[:-1]), self.scene["xyz"], self._rad)
            for shown in (False, True):
                if shown:
                    pens = [QtGui.QPen(QtGui.QColor(*col, 70), 9, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap),
                            QtGui.QPen(QtGui.QColor(*col), 2.5, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)]
                else:                                   # behind atoms: dashed and dim
                    pens = [QtGui.QPen(QtGui.QColor(*col, 150), 1.5, QtCore.Qt.DashLine)]
                for pen in pens:
                    p.setPen(pen)
                    for i in np.flatnonzero(vis == shown):
                        p.drawLine(QtCore.QPointF(*pix[i]), QtCore.QPointF(*pix[i + 1]))
            q0, q1 = QtCore.QPointF(*pix[-2]), QtCore.QPointF(*pix[-1])
            self._arrow_head(p, q0, q1, QtGui.QColor(*col))
        p.setBrush(QtGui.QColor(255, 255, 255))
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(b, 4, 4)
        Hh, Kk, Ll = self.beam_hkl()
        p.setPen(QtGui.QColor(*BEAM_OUT))
        f = p.font()
        f.setPointSizeF(f.pointSizeF() * 1.2)
        f.setBold(True)
        p.setFont(f)
        p.drawText(c + QtCore.QPointF(8, -6), f"({Hh} {Kk} {Ll:.2f})")
        p.setPen(QtGui.QColor(*BEAM_IN))
        p.drawText(a + QtCore.QPointF(8, -6), "incident")

    @staticmethod
    def _arrow_head(p, p0, p1, col, size=13.0):
        d = p1 - p0
        n = math.hypot(d.x(), d.y())
        if n < 1:
            return
        ux, uy = d.x() / n, d.y() / n
        tip = p1
        left = QtCore.QPointF(tip.x() - size * ux + 0.5 * size * uy, tip.y() - size * uy - 0.5 * size * ux)
        right = QtCore.QPointF(tip.x() - size * ux - 0.5 * size * uy, tip.y() - size * uy + 0.5 * size * ux)
        p.setBrush(col)
        p.setPen(QtCore.Qt.NoPen)
        p.drawPolygon(QtGui.QPolygonF([tip, left, right]))

    def reset_view(self):
        if self.scene is not None:
            self.camera.frame_scene(self.scene["xyz"], float(self.scene["surface_z"].max()))
            self.dirty = True

    # ---------------- rendering loop
    def tick(self):
        if self.scene is None:
            return
        if self.spin:
            self.camera.yaw += 0.006
            self.dirty = True
        scale = [1.0, 0.75, 0.5, 0.35][self.scale.currentIndex()]
        W = max(16, int(self.view.width() * scale))
        H = max(16, int(self.view.height() * scale))
        if (W, H) != (self.renderer.W, self.renderer.H):
            self.renderer.resize(W, H)
            self.dirty = True
        if self.dirty:
            self.frame, self.dirty = 0, False
        elif self.frame >= self.MAX_SAMPLES:
            return
        else:
            self.frame += 1
        light = np.array([-0.45, -0.55, 0.70])
        light /= np.linalg.norm(light)
        flags = [int(self.chk_sh.isChecked()), int(self.chk_ao.isChecked())]
        out = self.renderer.render(self.camera.vectors(W / H), light, self.frame, flags, 3.5)
        self.view.image = QtGui.QImage(out.data, W, H, 3 * W, QtGui.QImage.Format_RGB888).copy()
        self.view.update()
        now = time.perf_counter()
        self._fps = 0.9 * self._fps + 0.1 / max(now - self._t_last, 1e-6)
        self._t_last = now
        self.info.setText(f"{self.info_base}\n{W} x {H} px, {self._fps:.0f} frames/s, "
                          f"sample {self.frame + 1}/{self.MAX_SAMPLES + 1}")

    def pick(self, x, y):
        scale = self.renderer.W / max(self.view.width(), 1)
        s = self.renderer.pick(int(x * scale), int(y * scale))
        if s < 0 or self.scene is None:
            self.pick_label.setText("Click an atom to identify it.")
            return
        p = self.scene["xyz"][s]
        self.pick_label.setText(f"{self.scene['el'][s]}: {self.scene['label'][s]}\n"
                                f"x {p[0]:.2f}, y {p[1]:.2f}, z {p[2]:.2f} Å (z from the top TiO2 Ti plane)")

    def key(self, k):
        Qt = QtCore.Qt
        if k == Qt.Key_R:
            self.reset_view()
        elif k == Qt.Key_Space:
            self.spin = not self.spin
        elif k == Qt.Key_S:
            self.screenshot()
        elif k == Qt.Key_A:
            self.chk_ao.toggle()
        elif k == Qt.Key_H:
            self.chk_H.toggle()
        elif k == Qt.Key_W:
            self.chk_w.toggle()
        elif k == Qt.Key_B:
            self.chk_beam.toggle()

    def screenshot(self):
        if self.view.image is None:
            return
        folder = APP_DIR / "screenshots"
        folder.mkdir(exist_ok=True)
        path = folder / time.strftime("film_%Y%m%d_%H%M%S.png")
        (self.view.grab() if self.beam is not None else self.view.image).save(str(path))
        self.pick_label.setText(f"Saved screenshots/{path.name}")


def main():
    if QT_API == "PyQt5":
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ViewerWindow()
    win.show()
    sys.exit(app.exec() if hasattr(app, "exec") else app.exec_())


if __name__ == "__main__":
    main()

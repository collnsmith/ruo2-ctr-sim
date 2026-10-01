"""Vectorized structure factors for many (H, K, L) points of many rods in one call.

Same physics as CTRModel.intensity, rearranged for speed. The film is a sum over trilayers k,

    F_film(n) = sum_k [on_k(n) A(q) + e_k(n) S(q)] P_k exp(i qz z_k)

with A the structure factor of a full buried trilayer, S that of an exposed trilayer (with its CUS
species), on_k / e_k the buried / exposed occupancy of trilayer k for film thickness n, P_k the
in-plane parity phase (-1)^(K k) and z_k the trilayer heights. A and S are built with the same
CTRModel methods (add_trilayer, add_exposed_layer) as the reference engine. A buried and a relaxed
(top) region have their own A and S. Every intermediate is cached by the exact content it depends
on (atom lists, heights, occupancies), so only what changed is recomputed.
"""
import numpy as np

from .electrolyte import electrolyte_amplitude

OCC_MIN, PART_MIN = 1e-6, 1e-9          # same cut-offs as CTRModel.build_film_regions


class Points:
    """(H, K, L) points; H and K must be whole numbers (rod indices)."""

    def __init__(self, H, K, L):
        L = np.atleast_1d(np.asarray(L, float))
        H = np.broadcast_to(np.asarray(H), L.shape)
        K = np.broadcast_to(np.asarray(K), L.shape)
        if np.any(H != np.round(H)) or np.any(K != np.round(K)):
            raise ValueError("H and K must be whole numbers")
        self.H, self.K, self.L = H.astype(int), K.astype(int), L
        self.n = L.size
        self.h0 = self.H == 0
        self.spec = self.h0 & (self.K == 0)
        self.parity_odd = np.where(self.K % 2 == 0, 1.0, -1.0)      # P for odd trilayers


def _key(*parts):
    out = []
    for p in parts:
        if isinstance(p, np.ndarray):
            out.append((p.dtype.str, p.shape, p.tobytes()))
        elif isinstance(p, dict):
            out.append(tuple(sorted((k, _key(v)) for k, v in p.items())))
        elif isinstance(p, (list, tuple)):
            out.append(tuple(_key(v) for v in p))
        else:
            out.append(p)
    return tuple(out)


def _atoms_key(at):
    return _key(at["el"], at["f1"], at["f2"], at["z"], at["occ"], at["B"])


class FastCTR:
    """Intensities of a CTRModel (or FilmCTRModel) at fixed points, with caching between calls."""

    def __init__(self, H, K, L):
        self.pts = Points(H, K, L)
        self._cache = {}
        self.stats = {}

    def _cached(self, name, key, fn):
        hit = self._cache.get(name)
        if hit is not None and hit[0] == key:
            self.stats[name + ":hit"] = self.stats.get(name + ":hit", 0) + 1
            return hit[1]
        self.stats[name + ":miss"] = self.stats.get(name + ":miss", 0) + 1
        val = fn()
        self._cache[name] = (key, val)
        return val

    # ------------------------------------------------------------------ building blocks
    def _q(self, m, a1):
        pts = self.pts
        qx = 2 * np.pi * pts.H / a1
        qy, qz = 2 * np.pi * pts.K / m.A2, 2 * np.pi * pts.L / m.C3
        return qz, np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)

    def _qf(self, m, region):
        a1 = m.A1 if region == "b" else m.a1_top
        key = _key(a1, m.A1, m.A2, m.C3, m.f_anom)

        def build():
            qz, q = self._q(m, a1)
            return qz, q, {el: m.f_atom(el, q) for el in ("Ru", "Ti", "O", "H")}
        return self._cached("q" + region, key, build)

    def sf(self, at, region, m):
        """Structure factor of an atom list (positions relative to the trilayer) at every point."""
        qz, q, fq = self._qf(m, region)
        pts = self.pts
        F = np.zeros(pts.n, complex)
        if at["el"].size == 0:
            return F
        for el in np.unique(at["el"]):
            s = at["el"] == el
            ph = np.exp(2j * np.pi * (np.outer(pts.H, at["f1"][s]) + np.outer(pts.K, at["f2"][s]))
                        - np.outer(q ** 2, at["B"][s]) / (16 * np.pi ** 2) + 1j * np.outer(qz, at["z"][s]))
            F += fq[el] * (ph @ at["occ"][s])
        return F

    def _sf_cached(self, name, at, region, m):
        return self._cached(name, _key(region, _atoms_key(at)), lambda: self.sf(at, region, m))

    def substrate(self, m):
        def build():
            return self.sf(m.SUB_CELL, "b", m) / (1 - np.exp(-2j * np.pi * self.pts.L - m.alpha_abs))
        return self._cached("sub", _key(_atoms_key(m.SUB_CELL), m.alpha_abs, m.A1, m.A2, m.C3, m.f_anom), build)

    def _unit_bulk(self, m, top):
        at = m._new()
        d = m.D_TOP if top else m.D_FILM
        m.add_trilayer(at, "Ru", 0, 0.0, d, m.p["ruo2_u"], 1.0, m.B_FILM["M"], m.B_FILM["O"], "u")
        at = m._done(at)
        if top:
            at["B"] = at["B"] + m.b_top_extra
        return at

    def _unit_exposed(self, m, top, comp):
        at = m._new()
        d = m.D_TOP if top else m.D_FILM
        m.add_exposed_layer(at, 0, 0.0, d, 1.0, comp, m.check_comp(comp), "u")
        at = m._done(at)
        if top:
            at["B"] = at["B"] + m.b_top_extra
        return at

    @staticmethod
    def surface_states(m, comp, mode):
        """[(composition, weight)]: one coherent mixture, or one pure state per domain."""
        mode = mode or m.mix_mode
        if mode not in ("coherent", "domains"):
            raise ValueError(f"mixing mode must be 'coherent' or 'domains', got {mode!r}")
        theta = m.check_comp(comp)
        if mode == "coherent":
            return [(comp, 1.0)]
        states = [({sp: 1.0}, frac) for sp, frac in comp.items() if frac > 0]
        if 1 - theta > 1e-9:
            states.append(({}, 1 - theta))
        return states

    def _layers(self, m, weights):
        """Per geometry: heights, region mask, and occupancy coefficients for every thickness n."""
        groups = {}
        for n, w in weights:
            relaxed = bool(m.relax_001 > 0 and n > m.n_coherent)
            groups.setdefault(relaxed, []).append((n, w))
        out = []
        for relaxed, nw in groups.items():
            n_max = max(n for n, _ in nw)
            ks, z, _, top, _, _ = m.film_layers(n_max)
            K = ks.size
            on_b, ex_b, on_t, ex_t, e_raw = (np.zeros((K, len(nw))) for _ in range(5))
            for j, (n, _) in enumerate(nw):
                ks_n, z_n, _, top_n, occ, e = m.film_layers(n)
                kk = ks_n.size
                if not (np.array_equal(z_n, z[:kk]) and np.array_equal(top_n, top[:kk])):
                    raise RuntimeError("trilayer heights differ between thicknesses of one geometry")
                keep = occ >= OCC_MIN
                on = np.where(keep & (occ - e > PART_MIN), occ - e, 0.0)
                ex = np.where(keep & (e > PART_MIN), e, 0.0)
                on_b[:kk, j], ex_b[:kk, j] = np.where(top_n, 0.0, on), np.where(top_n, 0.0, ex)
                on_t[:kk, j], ex_t[:kk, j] = np.where(top_n, on, 0.0), np.where(top_n, ex, 0.0)
                e_raw[:kk, j] = e
            out.append(dict(relaxed=relaxed, nw=nw, ks=ks, z=z, top=top,
                            C=np.hstack([on_b, ex_b, on_t, ex_t, e_raw]),
                            has_top=[bool(np.any(on_t[:, j] > 0) or np.any(ex_t[:, j] > 0))
                                     for j in range(len(nw))]))
        return out

    def _phase_sums(self, m, lay):
        """U = E @ C with E[p, k] = P_k exp(i qz_p z_k): (points, 5 * thicknesses)."""
        qz, _, _ = self._qf(m, "b")
        name = "E" + ("r" if lay["relaxed"] else "c")

        def build_E():
            par = np.where(lay["ks"] % 2 == 1, self.pts.parity_odd[:, None], 1.0)
            return np.exp(1j * np.outer(qz, lay["z"])) * par
        E = self._cached(name, _key(lay["z"], lay["ks"], qz), build_E)
        return self._cached("U" + name, _key(lay["z"], lay["C"], qz), lambda: E @ lay["C"])

    # ------------------------------------------------------------------ intensity
    def intensity(self, m, comp, mode=None):
        """Same as m.intensity(H, K, L, comp, mode), for all points at once."""
        pts = self.pts
        states = self.surface_states(m, comp, mode)
        Fs = self.substrate(m)
        A = {"b": self._sf_cached("Ab", self._unit_bulk(m, False), "b", m)}
        weights = m.film_n_weights()
        layers = self._cached("layers", _key([(int(n), float(w)) for n, w in weights], m.sigma_rough, m.n_coherent,
                                             m.relax_001, m.Z_INT, m.D_FILM, m.D_TOP),
                              lambda: self._layers(m, weights))
        if any(any(lay["has_top"]) for lay in layers):
            A["t"] = self._sf_cached("At", self._unit_bulk(m, True), "t", m)
        S = []
        for i, (c, _) in enumerate(states):
            s = {"b": self._sf_cached(f"Sb{i}", self._unit_exposed(m, False, c), "b", m)}
            if "t" in A:
                s["t"] = self._sf_cached(f"St{i}", self._unit_exposed(m, True, c), "t", m)
            S.append(s)
        E = m.electrolyte
        el_on = E["on"] and pts.spec.any()
        if el_on:
            qz, _, _ = self._qf(m, "b")
            amp = self._cached("el", _key(qz, E["rho"], E["z0"], E["sigma"], m.A1, m.A2),
                               lambda: np.where(pts.spec, electrolyte_amplitude(qz, E["rho"], E["sigma"], m.A1 * m.A2)
                                                * np.exp(1j * qz * E["z0"]), 0.0))
        integrated = m.relaxed_rod_mode == "integrated"
        I = np.zeros(pts.n)
        for lay in layers:
            U = self._phase_sums(m, lay)
            nn = len(lay["nw"])
            Uon_b, Uex_b, Uon_t, Uex_t, Ue = (U[:, i * nn:(i + 1) * nn] for i in range(5))
            for j, (_, wn) in enumerate(lay["nw"]):
                Fel = amp * Ue[:, j] if el_on else 0.0
                for (_, ws), s in zip(states, S):
                    Fb = A["b"] * Uon_b[:, j] + s["b"] * Uex_b[:, j] + Fel
                    if lay["has_top"][j]:
                        Ft = A["t"] * Uon_t[:, j] + s["t"] * Uex_t[:, j]
                        Ib = np.abs(Fs + Fb) ** 2
                        inc = Ib + np.abs(Ft) ** 2 if integrated else Ib
                        Ij = np.where(pts.h0, np.abs(Fs + Fb + Ft) ** 2, inc)
                    else:
                        Ij = np.abs(Fs + Fb) ** 2
                    I += wn * ws * Ij
        return I

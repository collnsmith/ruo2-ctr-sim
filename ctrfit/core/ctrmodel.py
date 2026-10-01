"""Kinematic CTR engine for RuO2(110) films on TiO2(110) with CUS adsorbates.

This is the reference engine (moved here from ctr_engine.py, physics unchanged). Conventions:
  H || [001] (a1 = c_TiO2), K || [1-10] (a2 = sqrt2 a_TiO2), L || [110] (a3 = sqrt2 a_TiO2).
The in-plane film lattice is locked to TiO2; a relaxed upper part (along [001] only) can be added.
"""
import numpy as np
from scipy.special import erfc
from scipy.signal import find_peaks

from .electrolyte import electrolyte_F
from .lattice import c11_prime, film_spacing, trilayers
from .settings import DEFAULTS, parse_comp, parse_extra_layers, parse_floats, parse_points, parse_rods
from .sf import CROMER_MANN, F_ANOM_TABLE, HC, f_atom, resolve_anomalous, sf_atoms  # noqa: F401
from .structures import (ADSORBATE_H, add_atom, add_trilayer, finish_atoms, new_atoms,  # noqa: F401
                         oh_h2o)


class CTRModel:
    """All derived geometry plus structure-factor calculations for one parameter set."""

    def __init__(self, params):
        p = dict(DEFAULTS)
        p.update(params)
        self.p = p
        self.notes = []

        # beam
        self.lam = HC / p["energy_kev"]
        self.q_max = 2 * (2 * np.pi / self.lam) * np.sin(np.radians(p["two_theta_max"] / 2))
        self.L_min, self.L_max, self.L_step = p["L_min"], p["L_max"], p["L_step"]
        if self.L_max <= self.L_min:
            raise ValueError("L max must be larger than L min")

        # lattices
        self.A1 = p["tio2_c"]
        self.A2 = p["tio2_a"] * np.sqrt(2)
        self.D_SUB = p["tio2_a"] / np.sqrt(2)
        self.C3 = 2 * self.D_SUB
        self.strain_001 = p["strain_001_pct"] / 100
        self.strain_1m10 = p["strain_1m10_pct"] / 100
        self.a1_film = p["ruo2_c"] * (1 + self.strain_001)
        self.a2_film = p["ruo2_a"] * np.sqrt(2) * (1 + self.strain_1m10)
        self.D_FILM, self.eps_perp = film_spacing(p)
        self.Z_INT = 0.5 * (self.D_SUB + self.D_FILM) if p["interface_A"] <= 0 else p["interface_A"]
        self.mis1, self.mis2 = self.a1_film / self.A1 - 1, self.a2_film / self.A2 - 1
        if max(abs(self.mis1), abs(self.mis2)) > p["coherent_tol_pct"] / 100:
            self.notes.append("In-plane mismatch exceeds the pseudomorphic tolerance. A coherent sum of "
                              "film and substrate is not valid for these strains.")

        # thickness in trilayers
        self.n_film = trilayers(p["thickness_nm"], self.D_FILM, minimum=1)
        self.n_coherent = trilayers(p["coherent_nm"], self.D_FILM, minimum=0)
        self.film_n_spread = 10 * p["spread_nm"] / self.D_FILM
        self.sigma_rough = 10 * p["roughness_nm"] / self.D_FILM

        # relaxation of the [001] strain above n_coherent
        self.relax_001 = p["relax_001"]
        c11p = c11_prime(p)
        self.eps_001_top = self.strain_001 * (1 - self.relax_001)
        self.a1_top = p["ruo2_c"] * (1 + self.eps_001_top)
        self.eps_perp_top = self.eps_perp - p["C13"] / c11p * (self.eps_001_top - self.strain_001)
        self.D_TOP = p["ruo2_a"] / np.sqrt(2) * (1 + self.eps_perp_top)
        self.b_top_extra = p["b_top_extra"]
        self.relaxed_rod_mode = p["relaxed_rod_mode"]
        self.relaxed = self.relax_001 > 0 and self.n_film > self.n_coherent

        # Debye-Waller, adsorbates, relaxations, electrolyte
        self.B_SUB = dict(M=p["b_sub_M"], O=p["b_sub_O"])
        self.B_FILM = dict(M=p["b_film_M"], O=p["b_film_O"])
        self.B_SURF = dict(M=p["b_surf_M"], O=p["b_surf_O"])
        self.adsorbates = {
            "H2O": dict(z_O=p["h2o_z"], B_O=p["h2o_B"], dz_Mcus=p["h2o_dz"], H=ADSORBATE_H["H2O"]),
            "OH": dict(z_O=p["oh_z"], B_O=p["oh_B"], dz_Mcus=p["oh_dz"], H=ADSORBATE_H["OH"]),
            "O": dict(z_O=p["o_z"], B_O=p["o_B"], dz_Mcus=p["o_dz"], H=ADSORBATE_H["O"]),
        }
        self.include_H, self.b_H_extra = p["include_H"], p["b_H_extra"]
        self.relax_top = dict(M_6f=p["relax_M6f"], O_br=p["relax_Obr"], O_ip=p["relax_Oip"],
                              M_cus_vacant=p["relax_Mcus_vac"])
        self.extra_layers = parse_extra_layers(p["extra_layers"])
        self.electrolyte = dict(on=p["water_on"], rho=p["water_rho"], z0=p["water_z0"],
                                sigma=p["water_sigma"])
        self.mix_mode = p["mix_mode"]
        self.alpha_abs = p["alpha_abs"]
        self.f_anom, self.anom_src = self._resolve_anomalous()

        self.SUB_CELL = self.build_sub_cell()
        self.FILM_CELL = self.build_film_cell(self.D_FILM)
        self.FILM_CELL_TOP = self.build_film_cell(self.D_TOP)

    # ------------------------------------------------------------------ form factors
    def _resolve_anomalous(self):
        return resolve_anomalous(self.p, self.notes)

    def f_atom(self, el, q):
        return f_atom(el, q, self.f_anom)

    # ------------------------------------------------------------------ structure
    _new = staticmethod(new_atoms)
    _add = staticmethod(add_atom)
    _done = staticmethod(finish_atoms)
    add_trilayer = staticmethod(add_trilayer)

    def build_sub_cell(self):
        at = self._new()
        for p_, z0 in ((0, 0.0), (1, -self.D_SUB)):
            self.add_trilayer(at, "Ti", p_, z0, self.D_SUB, self.p["tio2_u"], 1.0,
                              self.B_SUB["M"], self.B_SUB["O"], f"sub{p_}")
        return self._done(at)

    def build_film_cell(self, d):
        at = self._new()
        for p_, z0 in ((1, 0.0), (0, d)):
            self.add_trilayer(at, "Ru", p_, z0, d, self.p["ruo2_u"], 1.0,
                              self.B_FILM["M"], self.B_FILM["O"], f"fc{p_}")
        return self._done(at)

    def layer_occupancy(self, k, n_film):
        k = np.asarray(k, float)
        if self.sigma_rough <= 0:
            return (k <= n_film).astype(float)
        return 0.5 * erfc((k - n_film - 0.5) / (np.sqrt(2) * self.sigma_rough))

    def film_n_weights(self):
        if self.film_n_spread <= 0:
            return [(self.n_film, 1.0)]
        ns = np.arange(max(1, int(np.floor(self.n_film - 2.5 * self.film_n_spread))),
                       int(np.ceil(self.n_film + 2.5 * self.film_n_spread)) + 1)
        w = np.exp(-0.5 * ((ns - self.n_film) / self.film_n_spread) ** 2)
        return list(zip(ns, w / w.sum()))

    def check_comp(self, comp):
        unknown = set(comp) - set(self.adsorbates)
        if unknown:
            raise KeyError(f"species {sorted(unknown)} not defined")
        if any(v < -1e-12 for v in comp.values()):
            raise ValueError(f"negative fraction in composition {comp}")
        theta = sum(comp.values())
        if theta > 1 + 1e-9:
            raise ValueError(f"CUS coverage {theta:.3f} > 1")
        return theta

    def film_layers(self, n_film):
        ks = np.arange(1, n_film + int(np.ceil(5 * max(self.sigma_rough, 0))) + 2)
        top = (ks > self.n_coherent) & (self.relax_001 > 0) & (n_film > self.n_coherent)
        d = np.where(top, self.D_TOP, self.D_FILM)
        z = self.Z_INT + np.concatenate([[0.0], np.cumsum(0.5 * (d[:-1] + d[1:]))])
        occ = self.layer_occupancy(ks, n_film)
        exposed = np.clip(occ - np.append(occ[1:], 0.0), 0.0, None)
        return ks, z, d, top, occ, exposed

    def build_film_regions(self, comp, n_film=None):
        n_film = self.n_film if n_film is None else n_film
        theta = self.check_comp(comp)
        regions = (self._new(), self._new())
        u = self.p["ruo2_u"]
        BF = self.B_FILM
        for k, z0, d_k, is_top, o, e in zip(*self.film_layers(n_film)):
            if o < 1e-6:
                continue
            at = regions[int(is_top)]
            on = o - e
            p_ = k % 2
            if on > 1e-9:
                self.add_trilayer(at, "Ru", p_, z0, d_k, u, on, BF["M"], BF["O"], f"L{k}")
            if e <= 1e-9:
                continue
            self.add_exposed_layer(at, p_, z0, d_k, e, comp, theta, f"L{k}s")
        bottom, top = self._done(regions[0]), self._done(regions[1])
        if top["B"].size:
            top["B"] = top["B"] + self.b_top_extra
        return bottom, top

    def add_exposed_layer(self, at, p_, z0, d_k, e, comp, theta, tag):
        """Atoms of an exposed trilayer with occupancy e: lattice without M_cus, top relaxations,
        CUS species of the composition, empty CUS Ru, and the extra ordered layers."""
        u = self.p["ruo2_u"]
        BF, BS = self.B_FILM, self.B_SURF
        s = 0.5 * p_
        self.add_trilayer(at, "Ru", p_, z0, d_k, u, e, BS["M"], BS["O"], tag,
                          skip=("M_cus",), relax=self.relax_top, B_Olo=BF["O"])
        f2c = (0.5 + s) % 1.0
        for sp, frac in comp.items():
            if frac <= 0:
                continue
            ad = self.adsorbates[sp]
            zM = z0 + ad["dz_Mcus"]
            zO = zM + ad["z_O"]
            self._add(at, "Ru", 0.0, f2c, zM, e * frac, BS["M"], f"{tag}:M_cus[{sp}]")
            self._add(at, "O", 0.0, f2c, zO, e * frac, ad["B_O"], f"{tag}:{sp}_O")
            if self.include_H:
                for dx, dy, dz in ad["H"]:
                    self._add(at, "H", dx / self.A1, (f2c + dy / self.A2) % 1.0, zO + dz, e * frac,
                              ad["B_O"] + self.b_H_extra, f"{tag}:{sp}_H")
        if 1 - theta > 1e-9:
            self._add(at, "Ru", 0.0, f2c, z0 + self.relax_top["M_cus_vacant"], e * (1 - theta),
                      BS["M"], f"{tag}:M_cus[vac]")
        for xl in self.extra_layers:
            site = xl["site"]
            f1x, f2x = {"cus": (0.0, 0.5), "br": (0.0, 0.0)}[site] if isinstance(site, str) else site
            self._add(at, xl["el"], f1x, (f2x + s) % 1.0, z0 + xl["z"], e * xl["occ"], xl["B"],
                      f"{tag}:extra")

    def build_film(self, comp, n_film=None):
        bottom, top = self.build_film_regions(comp, n_film)
        return {k: np.concatenate([bottom[k], top[k]]) for k in bottom}

    # ------------------------------------------------------------------ structure factors
    def qvec(self, H, K, L, a1=None):
        L = np.atleast_1d(np.asarray(L, float))
        qx = 2 * np.pi * H / (self.A1 if a1 is None else a1)
        qy, qz = 2 * np.pi * K / self.A2, 2 * np.pi * L / self.C3
        return qz, np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)

    def sf_atoms(self, at, H, K, L, a1=None):
        qz, q = self.qvec(H, K, L, a1)
        return sf_atoms(at, H, K, qz, q, self.f_atom)

    def F_sub(self, H, K, L):
        L = np.atleast_1d(np.asarray(L, float))
        return self.sf_atoms(self.SUB_CELL, H, K, L) / (1 - np.exp(-2j * np.pi * L - self.alpha_abs))

    def exposed_planes(self, n_film=None):
        _, z, _, _, _, e = self.film_layers(self.n_film if n_film is None else n_film)
        return z, e

    def F_electrolyte(self, H, K, L, n_film=None):
        qz, _ = self.qvec(H, K, L)
        E = self.electrolyte
        if not E["on"] or H != 0 or K != 0:
            return np.zeros(qz.shape, complex)
        z_pl, e = self.exposed_planes(n_film)
        return electrolyte_F(qz, z_pl, e, E["rho"], E["z0"], E["sigma"], self.A1 * self.A2)

    def film_parts(self, H, K, L, comp, n_film=None):
        bottom, top = self.build_film_regions(comp, n_film)
        Fb = self.sf_atoms(bottom, H, K, L) + self.F_electrolyte(H, K, L, n_film)
        Ft = self.sf_atoms(top, H, K, L, a1=self.a1_top) if top["el"].size else np.zeros_like(Fb)
        return Fb, Ft

    def _combine(self, Fs, Fb, Ft, H):
        if H == 0 or not np.any(Ft):
            return np.abs(Fs + Fb + Ft) ** 2
        I = np.abs(Fs + Fb) ** 2
        return I + np.abs(Ft) ** 2 if self.relaxed_rod_mode == "integrated" else I

    def intensity(self, H, K, L, comp, mode=None):
        mode = mode or self.mix_mode
        if mode not in ("coherent", "domains"):
            raise ValueError(f"mixing mode must be 'coherent' or 'domains', got {mode!r}")
        theta = self.check_comp(comp)
        Fs = self.F_sub(H, K, L)
        I = np.zeros_like(Fs.real)
        for n, wn in self.film_n_weights():
            if mode == "coherent":
                I += wn * self._combine(Fs, *self.film_parts(H, K, L, comp, n), H)
                continue
            for sp, frac in comp.items():
                if frac > 0:
                    I += wn * frac * self._combine(Fs, *self.film_parts(H, K, L, {sp: 1.0}, n), H)
            if 1 - theta > 1e-9:
                I += wn * (1 - theta) * self._combine(Fs, *self.film_parts(H, K, L, {}, n), H)
        return I

    def film_intensity(self, H, K, L, comp):
        return sum(wn * self._combine(0.0, *self.film_parts(H, K, L, comp, n), H)
                   for n, wn in self.film_n_weights())

    def bragg_L(self, H, K, Lmax):
        Ls = np.arange(1, int(np.floor(Lmax)) + 1, dtype=float)
        sub = Ls[np.abs(self.sf_atoms(self.SUB_CELL, H, K, Ls)) > 1e-4] if Ls.size else Ls
        film = []
        cells = [(self.FILM_CELL, self.D_FILM)]
        if self.relaxed:
            cells.append((self.FILM_CELL_TOP, self.D_TOP))
        for cell, d in cells:
            step = self.C3 / (2 * d)
            Lf = np.arange(1, int(np.floor(Lmax / step)) + 1) * step
            if Lf.size:
                film.append(Lf[np.abs(self.sf_atoms(cell, H, K, Lf)) > 1e-4])
        film = np.sort(np.concatenate(film)) if film else np.array([])
        return sub, film

    def L_range(self, H, K):
        qpar = 2 * np.pi * np.hypot(H / self.A1, K / self.A2)
        if qpar >= self.q_max:
            return None
        L_hi = min(self.L_max, np.sqrt(self.q_max ** 2 - qpar ** 2) * self.C3 / (2 * np.pi))
        if L_hi - self.L_min < 0.1:
            return None
        return np.arange(self.L_min + self.L_step / 2, L_hi, self.L_step)

    def bragg_mask(self, H, K, L, excl):
        sub, film = self.bragg_L(H, K, L.max() + 1)
        m = np.ones(L.shape, bool)
        for Lb in np.concatenate([sub, film]):
            m &= np.abs(L - Lb) >= excl
        return m

    # ------------------------------------------------------------------ high-level computations
    def rod_parts(self, H, K, comp):
        L = self.L_range(H, K)
        if L is None:
            return None
        sub, film = self.bragg_L(H, K, L.max())
        return dict(H=H, K=K, L=L, sub=np.abs(self.F_sub(H, K, L)) ** 2,
                    film=self.film_intensity(H, K, L, comp), total=self.intensity(H, K, L, comp),
                    bragg_sub=sub, bragg_film=film,
                    water=(H, K) == (0, 0) and self.electrolyte["on"])

    def compare(self, H, K, comps):
        L = self.L_range(H, K)
        if L is None:
            return None
        sub, film = self.bragg_L(H, K, L.max())
        return dict(H=H, K=K, L=L, I={lab: self.intensity(H, K, L, c) for lab, c in comps.items()},
                    bragg_sub=sub, bragg_film=film)

    def sensitivity_scan(self, comp_a, comp_b, h_max, k_max, metric="snr", excl=0.15,
                         rel_min=0.10, i_bg=0.0, progress=None):
        res, peaks = {}, []
        todo = [(H, K) for H in range(h_max + 1) for K in range(k_max + 1)]
        for i, (H, K) in enumerate(todo):
            if progress:
                progress(i, len(todo), f"scanning ({H} {K} L)")
            L = self.L_range(H, K)
            if L is None or L.size < 20:
                res[(H, K)] = None
                continue
            IA, IB = self.intensity(H, K, L, comp_a), self.intensity(H, K, L, comp_b)
            m = self.bragg_mask(H, K, L, excl)
            Im, dI = 0.5 * (IA + IB), IA - IB
            rel = np.where(m, np.abs(dI) / Im, 0.0)
            snr = np.where(m & (rel >= rel_min), np.abs(dI) / np.sqrt(Im + i_bg), 0.0)
            sub, film = self.bragg_L(H, K, L.max())
            res[(H, K)] = dict(L=L, IA=IA, IB=IB, rel=rel, snr=snr, mask=m,
                               n_bragg=int(np.sum(sub >= self.L_min)), Lmax=L.max(),
                               bragg_sub=sub, bragg_film=film)
            met = rel if metric == "rel" else snr
            pk, _ = find_peaks(np.concatenate([[0], met, [0]]), distance=max(1, int(0.2 / self.L_step)))
            mp = np.concatenate([[False], m, [False]])
            for j in pk - 1:
                if met[j] > 0:
                    edge = not (mp[j] and mp[j + 2])
                    peaks.append(dict(H=H, K=K, L=L[j], rel=rel[j], snr=snr[j], I=Im[j],
                                      A_gt_B=bool(dI[j] > 0), edge=edge))
        snr_max = max((r["snr"].max() for r in res.values() if r is not None), default=1.0) or 1.0
        for r in res.values():
            if r is not None:
                r["snr"] = r["snr"] / snr_max
                r["rel_max"], r["snr_max"] = r["rel"].max(), r["snr"].max()
                r["L_rel"], r["L_snr"] = r["L"][r["rel"].argmax()], r["L"][r["snr"].argmax()]
        for pk_ in peaks:
            pk_["snr"] /= snr_max
        peaks.sort(key=lambda d: d[metric], reverse=True)
        if progress:
            progress(len(todo), len(todo), "scan done")
        return res, peaks

    def bragg_counts(self, h_lim, k_lim):
        out = {}
        for H in range(h_lim + 1):
            for K in range(k_lim + 1):
                L = self.L_range(H, K)
                out[(H, K)] = None if L is None else int(np.sum(self.bragg_L(H, K, L.max())[0] >= self.L_min))
        return out

    def summary(self):
        p = self.p
        t_nm = self.n_film * self.D_FILM / 10
        lines = [
            f"Wavelength {self.lam:.4f} Å, q_max {self.q_max:.2f} 1/Å",
            f"TiO2 surface cell: a1 = {self.A1:.4f} Å [001], a2 = {self.A2:.4f} Å [1-10], "
            f"d(110) = {self.D_SUB:.4f} Å",
            f"RuO2 strained in-plane: a1 = {self.a1_film:.4f} Å, a2 = {self.a2_film:.4f} Å",
            f"  residual mismatch to TiO2: [001] {100 * self.mis1:+.2f} %, [1-10] {100 * self.mis2:+.2f} %",
            f"  out-of-plane strain {100 * self.eps_perp:+.2f} % ({p['eps_perp_mode']}), "
            f"d(110) = {self.D_FILM:.4f} Å",
            f"Film: {p['thickness_nm']:g} nm entered -> {self.n_film} trilayers = {t_nm:.2f} nm",
            f"  thickness spread {self.film_n_spread:.2f} trilayers, roughness {self.sigma_rough:.2f} trilayers",
            f"  RuO2 00L Bragg peaks at L = {', '.join(f'{n * self.C3 / (2 * self.D_FILM):.4f}' for n in (1, 2, 3))}",
        ]
        if self.relaxed:
            lines += [
                f"Relaxation: bottom {self.n_coherent} trilayers ({self.n_coherent * self.D_FILM / 10:.2f} nm) "
                f"commensurate, top {self.n_film - self.n_coherent} trilayers relaxed by {self.relax_001:g}",
                f"  top part: [001] strain {100 * self.eps_001_top:+.2f} %, out-of-plane "
                f"{100 * self.eps_perp_top:+.2f} %, d(110) = {self.D_TOP:.4f} Å",
                f"  its H rods sit at H x {self.A1 / self.a1_top:.4f}; its 00L peaks at L = "
                f"{', '.join(f'{n * self.C3 / (2 * self.D_TOP):.4f}' for n in (1, 2, 3))}",
            ]
        elif self.relax_001 > 0:
            lines.append(f"Relaxation: film ({self.n_film} trilayers) is not thicker than the commensurate "
                         f"part ({self.n_coherent} trilayers), so it is treated as commensurate")
        ru = self.f_anom.get("Ru", (0.0, 0.0))
        lines.append(f"Anomalous terms: {self.anom_src}; Ru f' = {ru[0]:+.3f}, f'' = {ru[1]:.3f}")
        if self.electrolyte["on"]:
            lines.append(f"Bulk electrolyte on 00L: {self.electrolyte['rho']:g} e/Å³ from "
                         f"{self.electrolyte['z0']:g} Å above the top metal plane")
        return "\n".join(lines + [f"Note: {n}" for n in self.notes])


def relaxation_study(params, points, thick_nm, relax_vals, comp_a, comp_b, progress=None):
    """Relative A/B change at fixed points versus film thickness and [001] relaxation."""
    out = {}
    total, done = len(thick_nm) * len(relax_vals), 0
    for R in relax_vals:
        for t in thick_nm:
            if progress:
                progress(done, total, f"study: {t:g} nm, relaxation {R:g}")
            m = CTRModel(dict(params, thickness_nm=t, relax_001=R))
            for lab, (H, K, Lv) in points.items():
                a = m.intensity(H, K, [Lv], comp_a)[0]
                b = m.intensity(H, K, [Lv], comp_b)[0]
                out[(lab, R, t)] = 2 * (a - b) / (a + b)
            done += 1
    if progress:
        progress(total, total, "study done")
    return out


def parse_inputs(p):
    """Parse every text field once so errors show up before any computation starts."""
    return dict(
        comp_default=parse_comp(p["comp_default"], "surface state for plots"),
        x_oh_list=parse_floats(p["x_oh_list"], "OH fractions"),
        rods=parse_rods(p["rods"]),
        sens_a=parse_comp(p["sens_a"], "state A"),
        sens_b=parse_comp(p["sens_b"], "state B"),
        study_thick=parse_floats(p["study_thick_nm"], "study thicknesses"),
        study_relax=parse_floats(p["study_relax"], "study relaxation values"),
        study_points=parse_points(p["study_points"]),
        extra_layers=parse_extra_layers(p["extra_layers"]),
    )

"""Physics checks that used to be done by hand."""
import itertools

import numpy as np
import pytest
from scipy.integrate import simpson
from scipy.special import erfc

import ctr_engine
from ctr_engine import CTRModel
from ctr_params import DEFAULTS


def rutile_conventional_F(m, h, k, l):
    """Conventional rutile cell (a, a, c), Ti at 0 and 1/2, O at +-(u,u,0), +-(1/2+u,1/2-u,1/2)."""
    a, c, u = m.p["tio2_a"], m.p["tio2_c"], m.p["tio2_u"]
    q = 2 * np.pi * np.sqrt((h / a) ** 2 + (k / a) ** 2 + (l / c) ** 2)
    ti = [(0, 0, 0), (0.5, 0.5, 0.5)]
    ox = [(u, u, 0), (-u, -u, 0), (0.5 + u, 0.5 - u, 0.5), (0.5 - u, 0.5 + u, 0.5)]
    F = 0j
    for el, sites in (("Ti", ti), ("O", ox)):
        dw = np.exp(-m.B_SUB["M" if el == "Ti" else "O"] * q ** 2 / (16 * np.pi ** 2))
        f = m.f_atom(el, q)
        F += f * dw * sum(np.exp(2j * np.pi * (h * x + k * y + l * z)) for x, y, z in sites)
    return F


def test_bulk_cell_equals_twice_rutile():
    """Surface cell (H || c, K || [1-10], L || [110]) holds two conventional cells.
    h = (K + L)/2, k = (L - K)/2, l = H; reflections with K + L odd are forbidden."""
    m = CTRModel(DEFAULTS)
    n_checked = 0
    for H, K, L in itertools.product(range(-2, 3), range(-3, 4), range(0, 7)):
        Fs = m.sf_atoms(m.SUB_CELL, H, K, [float(L)])[0]
        if (K + L) % 2:
            assert abs(Fs) < 1e-9, (H, K, L)
            continue
        Fc = rutile_conventional_F(m, (K + L) // 2, (L - K) // 2, H)
        assert abs(Fs) == pytest.approx(2 * abs(Fc), rel=1e-10, abs=1e-9), (H, K, L)
        n_checked += 1
    assert n_checked > 100


def _film_as_substrate(n_trilayers):
    p = dict(DEFAULTS)
    d = p["tio2_a"] / np.sqrt(2)
    p.update(ruo2_a=p["tio2_a"], ruo2_c=p["tio2_c"], ruo2_u=p["tio2_u"],
             strain_001_pct=0.0, strain_1m10_pct=0.0, eps_perp_mode="manual", eps_perp_pct=0.0,
             interface_A=0.0, thickness_nm=n_trilayers * d / 10, spread_nm=0.0, roughness_nm=0.0,
             relax_001=0.0, water_on=False, anom_mode="off", extra_layers="", alpha_abs=1e-8,
             b_film_M=p["b_sub_M"], b_film_O=p["b_sub_O"], b_surf_M=p["b_sub_M"], b_surf_O=p["b_sub_O"],
             relax_M6f=0.0, relax_Obr=0.0, relax_Oip=0.0, relax_Mcus_vac=0.0)
    return p


@pytest.mark.parametrize("n_trilayers", [5, 6])
@pytest.mark.parametrize("HK", [(0, 0), (0, 1), (1, 0), (1, 1), (0, 3)])
def test_film_with_substrate_lattice_continues_substrate(monkeypatch, n_trilayers, HK):
    # the film metal is Ru in the engine; let it scatter as Ti so the film is just more substrate
    monkeypatch.setitem(ctr_engine.CROMER_MANN, "Ru", ctr_engine.CROMER_MANN["Ti"])
    m = CTRModel(_film_as_substrate(n_trilayers))
    assert m.n_film == n_trilayers and m.Z_INT == pytest.approx(m.D_SUB)
    H, K = HK
    L = np.arange(0.3, 6.0, 0.01)
    L = L[np.abs(L - np.round(L)) > 0.05]          # away from Bragg peaks
    I_total = m.intensity(H, K, L, {})
    I_sub = np.abs(m.F_sub(H, K, L)) ** 2
    assert np.max(np.abs(I_total / I_sub - 1)) < 1e-5


def test_electrolyte_matches_numerical_integral():
    p = dict(DEFAULTS, roughness_nm=0.3, water_sigma=0.8, water_z0=3.1)
    m = CTRModel(p)
    E = m.electrolyte
    z_pl, e = m.exposed_planes()
    zs = z_pl + E["z0"]
    keep = e > 1e-12
    zs, e = zs[keep], e[keep]
    assert len(e) > 2
    L = np.array([0.37, 1.13, 2.71, 3.9, 5.55])
    qz = 2 * np.pi * L / m.C3
    a, b = zs.min() - 12 * E["sigma"], zs.max() + 12 * E["sigma"]
    z = np.linspace(a, b, 200001)
    rho = E["rho"] * m.A1 * m.A2 * sum(ek * 0.5 * erfc((zk - z) / (np.sqrt(2) * E["sigma"]))
                                       for zk, ek in zip(zs, e))
    rho_inf = E["rho"] * m.A1 * m.A2 * e.sum()
    num = simpson(rho[None, :] * np.exp(1j * qz[:, None] * z[None, :]), x=z, axis=1)
    num += rho_inf * 1j / qz * np.exp(1j * qz * b)      # constant tail beyond b
    eng = m.F_electrolyte(0, 0, L)
    np.testing.assert_allclose(eng, num, rtol=1e-6)
    assert not np.any(m.F_electrolyte(0, 1, L)) and not np.any(m.F_electrolyte(1, 0, L))

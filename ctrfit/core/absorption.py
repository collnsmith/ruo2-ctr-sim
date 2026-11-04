"""Absorption of the beams in the electrolyte film and the window above the surface (in situ cell).

An X-ray film cell: the beam crosses a thin electrolyte layer (thickness t_liquid) and the window film
(t_window, Kapton) on the way in, at the incidence angle alpha, and on the way out, at the exit angle
beta. Both are flat layers parallel to the surface, so the paths are t / sin(alpha) and t / sin(beta),
and the measured |F| is multiplied by

    T = exp(-(mu_liquid t_liquid + mu_window t_window) (1 / sin alpha + 1 / sin beta) / 2)

(the square root of the intensity transmission). At a fixed incidence angle the alpha part is the same
for every point (the scale takes it); the beta part grows towards low L, where the exit angle is small.
Linear attenuation coefficients (1/cm) of 0.1 M H2SO4 (density 1.005) and Kapton (C22H10N2O5, 1.42
g/cm3) from xraydb, tabulated every keV from 5 to 40 keV (log-log interpolation), so the numbers do not
depend on the machine. Refraction near the critical angle is not included: leave out points with an
exit angle below about twice the critical angle.
"""
import math

import numpy as np

E_KEV = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40)
MU_ELECTROLYTE = (43.817, 25.381, 15.977, 10.702, 7.526, 5.5038, 4.158, 3.2292, 2.5685, 2.0862, 1.7262, 1.4522, 1.2401, 1.0734, 0.94042, 0.83314, 0.74562, 0.67347, 0.61347, 0.56314, 0.52059, 0.48437, 0.45333, 0.42656, 0.40334, 0.38308, 0.36531, 0.34965, 0.33579, 0.32347, 0.31248, 0.30262, 0.29375, 0.28574, 0.27848, 0.27188)
MU_KAPTON = (36.28, 20.868, 13.081, 8.7477, 6.1558, 4.5156, 3.4295, 2.683, 2.1536, 1.768, 1.4806, 1.2621, 1.0931, 0.96023, 0.85438, 0.76897, 0.69925, 0.64175, 0.59388, 0.55368, 0.51964, 0.49061, 0.46568, 0.44412, 0.42537, 0.40897, 0.39454, 0.38178, 0.37044, 0.36032, 0.35125, 0.34308, 0.3357, 0.329, 0.3229, 0.31732)
HC = 12.398419843320026          # keV Å
BETA_MIN_DEG = 0.01              # exit angles below this (unreachable points) count as this


def mu(table, energy_kev):
    """Linear attenuation coefficient (1/cm) at energy_kev from one of the tables."""
    e = float(np.clip(energy_kev, E_KEV[0], E_KEV[-1]))
    return float(np.exp(np.interp(np.log(e), np.log(E_KEV), np.log(table))))


def surface_angles(H, K, L, energy_kev, lattice, alpha_deg):
    """(alpha, beta, cos psi) in degrees from H K L at a fixed incidence angle alpha:
    sin(beta) = Q_perp / k - sin(alpha); on the specular rod alpha = beta. lattice = (a1, a2, a3) in Å.
    psi is the in-plane angle between the incident and exit directions (nan where unreachable)."""
    a1, a2, a3 = lattice
    k = 2 * math.pi / (HC / energy_kev)
    q_par = 2 * math.pi * np.sqrt((np.asarray(H, float) / a1) ** 2 + (np.asarray(K, float) / a2) ** 2)
    q_perp = 2 * math.pi * np.asarray(L, float) / a3
    alpha = np.radians(np.broadcast_to(np.asarray(alpha_deg, float), np.shape(q_perp))).copy()
    spec = q_par < 1e-9                     # specular rod: only alpha = beta reaches it
    alpha[spec] = np.arcsin(np.clip(q_perp[spec] / (2 * k), -1, 1))
    sin_b = q_perp / k - np.sin(alpha)
    beta = np.arcsin(np.clip(sin_b, -1, 1))
    beta = np.where(np.abs(sin_b) <= 1, beta, np.nan)
    ca, cb = np.cos(alpha), np.cos(beta)
    cos_psi = (ca ** 2 + cb ** 2 - (q_par / k) ** 2) / (2 * ca * cb)
    cos_psi = np.where(np.abs(cos_psi) <= 1 + 1e-9, np.clip(cos_psi, -1, 1), np.nan)
    cos_psi[spec] = 1.0
    return np.degrees(alpha), np.degrees(beta), cos_psi


def path_factor(H, K, L, energy_kev, lattice, alpha_deg):
    """1 / sin(alpha) + 1 / sin(beta) per point (beta at least BETA_MIN_DEG)."""
    a, b, _ = surface_angles(H, K, L, energy_kev, lattice, alpha_deg)
    b = np.where(np.isfinite(b), np.maximum(b, BETA_MIN_DEG), BETA_MIN_DEG)
    return 1 / np.sin(np.radians(np.maximum(a, BETA_MIN_DEG))) + 1 / np.sin(np.radians(b))


def transmission_F(path, energy_kev, t_liquid_um, t_window_um):
    """The |F| factor exp(-(mu_l t_l + mu_w t_w) path / 2) for paths from path_factor."""
    if not t_liquid_um and not t_window_um:
        return 1.0
    mt = mu(MU_ELECTROLYTE, energy_kev) * t_liquid_um * 1e-4 + mu(MU_KAPTON, energy_kev) * t_window_um * 1e-4
    return np.exp(-0.5 * mt * np.asarray(path, float))

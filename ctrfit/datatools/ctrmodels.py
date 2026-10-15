"""Quick CTR models for the rod fitter (from sxrd_ctr_fit, no Qt): substrate, film, adlayer,
liquid and three-layer kinematic models with roughness, and scan groups. These are simple
phenomenological rods for a first look; the full RuO2/TiO2 model is in ctrfit.core / ctrfit.fit."""
import json
from collections import OrderedDict

import numpy as np


def ctr_substrate(L, params):
    """Model 1: Semi-infinite substrate CTR with roughness."""
    f_bulk, d_spacing, beta_sub, scale, bg = params
    qz = 2 * np.pi * L / d_spacing
    phase = np.exp(1j * qz * d_spacing)
    F_sub = f_bulk / (1 - beta_sub * phase)
    return scale * np.abs(F_sub)**2 + bg


def ctr_substrate_film(L, params):
    """Model 2: Substrate + epitaxial thin film with roughness."""
    f_bulk, f_film, d_sub, d_film, N_film, beta_sub, beta_film, scale, bg = params
    N_film = max(1, int(round(N_film)))
    qz_sub = 2 * np.pi * L / d_sub
    qz_film = 2 * np.pi * L / d_film
    phase_sub = np.exp(1j * qz_sub * d_sub)
    F_sub = f_bulk / (1 - beta_sub * phase_sub)
    phase_film = np.exp(1j * qz_film * d_film)
    # Finite film sum
    safe = np.abs(phase_film - 1) > 1e-12
    F_film_sum = np.where(safe,
        f_film * (1 - phase_film**N_film) / (1 - phase_film),
        N_film * f_film)
    # Film surface roughness
    F_film = F_film_sum * (1 - beta_film) / (1 - beta_film * phase_film)
    F_total = F_sub + F_film
    return scale * np.abs(F_total)**2 + bg


def ctr_substrate_film_adlayer(L, params):
    """Model 3: Substrate + film + surface adlayer."""
    (f_bulk, f_film, f_ad, d_sub, d_film, d_ad,
     N_film, occ_ad, beta_sub, beta_film, scale, bg) = params
    N_film = max(1, int(round(N_film)))
    qz_sub = 2 * np.pi * L / d_sub
    qz_film = 2 * np.pi * L / d_film
    phase_sub = np.exp(1j * qz_sub * d_sub)
    F_sub = f_bulk / (1 - beta_sub * phase_sub)
    phase_film = np.exp(1j * qz_film * d_film)
    safe = np.abs(phase_film - 1) > 1e-12
    F_film_sum = np.where(safe,
        f_film * (1 - phase_film**N_film) / (1 - phase_film),
        N_film * f_film)
    F_film = F_film_sum * (1 - beta_film) / (1 - beta_film * phase_film)
    F_total = F_sub + F_film
    # Adlayer
    qz = 2 * np.pi * L / d_sub
    z_ad = N_film * d_film + d_ad
    F_ad = occ_ad * f_ad * np.exp(1j * qz * z_ad)
    F_total = F_total + F_ad
    return scale * np.abs(F_total)**2 + bg


def ctr_substrate_film_liquid(L, params):
    """Model 4: Substrate + film + liquid interface."""
    (f_bulk, f_film, f_liq, d_sub, d_film, d_liq,
     N_film, rho_liq, sigma_liq, beta_sub, beta_film, scale, bg) = params
    N_film = max(1, int(round(N_film)))
    qz_sub = 2 * np.pi * L / d_sub
    qz_film = 2 * np.pi * L / d_film
    phase_sub = np.exp(1j * qz_sub * d_sub)
    F_sub = f_bulk / (1 - beta_sub * phase_sub)
    phase_film = np.exp(1j * qz_film * d_film)
    safe = np.abs(phase_film - 1) > 1e-12
    F_film_sum = np.where(safe,
        f_film * (1 - phase_film**N_film) / (1 - phase_film),
        N_film * f_film)
    F_film = F_film_sum * (1 - beta_film) / (1 - beta_film * phase_film)
    F_total = F_sub + F_film
    # Liquid interface
    qz = 2 * np.pi * L / d_sub
    z_liq = N_film * d_film
    dw = np.exp(-0.5 * (qz * sigma_liq)**2)
    F_liq_layer = f_liq * np.exp(1j * qz * (z_liq + d_liq)) * dw
    F_liq_bulk = rho_liq * f_bulk * np.exp(1j * qz * z_liq) * dw / (1j * qz * d_sub + 0.01)
    F_total = F_total + F_liq_layer + F_liq_bulk
    return scale * np.abs(F_total)**2 + bg


def ctr_multilayer(L, params):
    """Model 5: Substrate + N graded layers (variable spacing/occupancy)."""
    # params: f_bulk, d_sub, beta_sub, [f_i, d_i, occ_i]*N_layers, scale, bg
    # We encode N_layers = 3 fixed for the UI, user sets occupancy=0 to disable
    f_bulk = params[0]; d_sub = params[1]; beta_sub = params[2]
    n_layers = 3
    layer_params = params[3:3 + n_layers * 3]  # f, d, occ for each
    scale = params[3 + n_layers * 3]
    bg_val = params[4 + n_layers * 3]

    qz = 2 * np.pi * L / d_sub
    phase_sub = np.exp(1j * qz * d_sub)
    F_sub = f_bulk / (1 - beta_sub * phase_sub)
    F_total = F_sub.copy()

    z_current = 0.0
    for i in range(n_layers):
        f_i = layer_params[i * 3]
        d_i = layer_params[i * 3 + 1]
        occ_i = layer_params[i * 3 + 2]
        if occ_i < 0.001:
            continue
        z_current += d_i
        F_layer = occ_i * f_i * np.exp(1j * qz * z_current)
        F_total = F_total + F_layer

    return scale * np.abs(F_total)**2 + bg_val


CTR_MODELS = OrderedDict([
    ('1: Substrate only', {
        'func': ctr_substrate,
        'params': ['f_bulk', 'd_spacing', 'beta_sub', 'scale', 'bg'],
        'defaults': [1.0, 3.248, 0.1, 1.0, 0.001],
        'bounds_lo': [0.01, 0.5, 0, 1e-6, 0],
        'bounds_hi': [100, 20, 0.99, 1e6, 1e4],
        'desc': 'Semi-infinite substrate with surface roughness',
    }),
    ('2: Substrate + film', {
        'func': ctr_substrate_film,
        'params': ['f_bulk', 'f_film', 'd_sub', 'd_film', 'N_film',
                   'beta_sub', 'beta_film', 'scale', 'bg'],
        'defaults': [1.0, 1.2, 3.248, 3.186, 5, 0.1, 0.2, 1.0, 0.001],
        'bounds_lo': [0.01, 0.01, 0.5, 0.5, 1, 0, 0, 1e-6, 0],
        'bounds_hi': [100, 100, 20, 20, 200, 0.99, 0.99, 1e6, 1e4],
        'desc': 'Epitaxial thin film on substrate (e.g. RuO2/TiO2)',
    }),
    ('3: Substrate + film + adlayer', {
        'func': ctr_substrate_film_adlayer,
        'params': ['f_bulk', 'f_film', 'f_ad', 'd_sub', 'd_film', 'd_ad',
                   'N_film', 'occ_ad', 'beta_sub', 'beta_film', 'scale', 'bg'],
        'defaults': [1.0, 1.2, 0.5, 3.248, 3.186, 2.0, 5, 0.5, 0.1, 0.2, 1.0, 0.001],
        'bounds_lo': [0.01, 0.01, 0, 0.5, 0.5, 0, 1, 0, 0, 0, 1e-6, 0],
        'bounds_hi': [100, 100, 100, 20, 20, 10, 200, 1, 0.99, 0.99, 1e6, 1e4],
        'desc': 'Film + surface reconstruction or modified termination',
    }),
    ('4: Substrate + film + liquid', {
        'func': ctr_substrate_film_liquid,
        'params': ['f_bulk', 'f_film', 'f_liq', 'd_sub', 'd_film', 'd_liq',
                   'N_film', 'rho_liq', 'sigma_liq', 'beta_sub', 'beta_film', 'scale', 'bg'],
        'defaults': [1.0, 1.2, 0.3, 3.248, 3.186, 2.8, 5, 0.33, 1.0, 0.1, 0.2, 1.0, 0.001],
        'bounds_lo': [0.01, 0.01, 0, 0.5, 0.5, 0, 1, 0, 0.01, 0, 0, 1e-6, 0],
        'bounds_hi': [100, 100, 100, 20, 20, 20, 200, 2, 10, 0.99, 0.99, 1e6, 1e4],
        'desc': 'Film + electrolyte interface (in-situ)',
    }),
    ('5: Multi-layer (3 layers)', {
        'func': ctr_multilayer,
        'params': ['f_bulk', 'd_sub', 'beta_sub',
                   'f_L1', 'd_L1', 'occ_L1', 'f_L2', 'd_L2', 'occ_L2',
                   'f_L3', 'd_L3', 'occ_L3', 'scale', 'bg'],
        'defaults': [1.0, 3.248, 0.1, 1.0, 3.2, 1.0, 0.8, 3.2, 0.5, 0.5, 3.2, 0.0, 1.0, 0.001],
        'bounds_lo': [0.01, 0.5, 0, 0, 0.5, 0, 0, 0.5, 0, 0, 0.5, 0, 1e-6, 0],
        'bounds_hi': [100, 20, 0.99, 100, 20, 1, 100, 20, 1, 100, 20, 1, 1e6, 1e4],
        'desc': 'Substrate + up to 3 graded layers (set occ=0 to disable)',
    }),
])


class ScanGroupManager:
    """Manages groups of scans belonging to the same CTR rod."""
    def __init__(self):
        self.groups = OrderedDict()

    def add_group(self, name, scans=None, hk='0 0', notes=''):
        self.groups[name] = {'scans': scans or [], 'hk': hk, 'notes': notes}

    def remove_group(self, name):
        self.groups.pop(name, None)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({n: g for n, g in self.groups.items()}, f, indent=2)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.groups = OrderedDict()
        for name, g in data.items():
            self.groups[name] = {
                'scans': g.get('scans', []),
                'hk': g.get('hk', '0 0'),
                'notes': g.get('notes', '')}

"""Bulk electrolyte above the exposed film terraces (specular rod only)."""
import numpy as np


def electrolyte_amplitude(qz, rho, sigma, area):
    """Fourier transform of one erfc step of density rho (e/Å³) and width sigma at z = 0, per cell."""
    qz_safe = np.where(np.abs(qz) < 1e-9, 1e-9, qz)
    return rho * area * 1j / qz_safe * np.exp(-0.5 * (qz * sigma) ** 2)


def electrolyte_F(qz, z_planes, exposed, rho, z0, sigma, area):
    """Structure factor of an erfc-onset electron density rho (e/Å³) starting z0 above each exposed
    metal plane (weights `exposed`), per surface cell of the given area (Å²)."""
    ph = np.exp(1j * np.outer(qz, z_planes + z0)) @ exposed
    return electrolyte_amplitude(qz, rho, sigma, area) * ph

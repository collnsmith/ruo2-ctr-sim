"""Lattice spacings and strain of the commensurate RuO2 film on TiO2(110)."""
import numpy as np


def c11_prime(p):
    """In-plane stiffness for (110) films: C11' = (C11 + C12)/2 + C66."""
    return 0.5 * (p["C11"] + p["C12"]) + p["C66"]


def film_spacing(p):
    """(110) trilayer spacing of the commensurate film (Angstrom) and eps_perp, from GUI values."""
    s001, s110 = p["strain_001_pct"] / 100, p["strain_1m10_pct"] / 100
    if p["eps_perp_mode"] == "elastic":
        c11p = 0.5 * (p["C11"] + p["C12"]) + p["C66"]
        c12p = 0.5 * (p["C11"] + p["C12"]) - p["C66"]
        eps = -(p["C13"] * s001 + c12p * s110) / c11p
    else:
        eps = p["eps_perp_pct"] / 100
    return p["ruo2_a"] / np.sqrt(2) * (1 + eps), eps


def trilayers(nm, d_A, minimum=0):
    return max(minimum, int(round(10.0 * nm / d_A)))

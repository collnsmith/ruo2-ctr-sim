"""Beamline calculators: energy and wavelength, d spacing and 2 theta, critical angle,
penetration depth, footprint and beam spill-over.

Refraction uses n = 1 - delta + i beta with delta = r_e lambda^2 sum_j n_j (Z_j + f'_j) / (2 pi) and
beta = r_e lambda^2 sum_j n_j f''_j / (2 pi). f' and f'' come from the same built-in table as the
structure factors (Ru, Ti, O, H, 12 to 21 keV) or from xraydb if installed; other elements get
f' = f'' = 0 (fine for delta well above edges, not for beta).
"""
import math
import re

from ..core.sf import F_ANOM_TABLE
from .psic import HC, energy_to_wavelength

R_E = 2.8179403262e-5          # classical electron radius (Å)
N_A = 6.02214076e23
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "S": 16, "Cl": 17, "K": 19, "Ti": 22, "Ru": 44,
     "Pt": 78, "Au": 79, "Ir": 77}
MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Si": 28.0855, "S": 32.06,
        "Cl": 35.45, "K": 39.098, "Ti": 47.867, "Ru": 101.07, "Pt": 195.08, "Au": 196.97, "Ir": 192.22}
MATERIALS = {          # formula, density (g/cm^3)
    "TiO2 (rutile)": ("TiO2", 4.23),
    "RuO2": ("RuO2", 6.97),
    "water": ("H2O", 1.00),
    "Si": ("Si", 2.329),
    "Kapton": ("C22H10N2O5", 1.42),
}


def parse_formula(formula):
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula):
        if el not in Z:
            raise ValueError(f"element {el} not known (add it to ctrfit.beamline.xtools.Z and MASS)")
        out[el] = out.get(el, 0.0) + (float(n) if n else 1.0)
    return out


def anomalous(el, energy_kev):
    """(f', f'') for one element at an energy: xraydb if installed, else the built-in table."""
    try:
        import xraydb
        e = energy_kev * 1e3
        return float(xraydb.f1_chantler(el, e)), float(xraydb.f2_chantler(el, e))
    except ImportError:
        e_tab = min(F_ANOM_TABLE, key=lambda e: abs(e - energy_kev))
        return F_ANOM_TABLE[e_tab].get(el, (0.0, 0.0))


def refraction(formula, density, energy_kev):
    """(delta, beta) of a material (density in g/cm^3)."""
    comp = parse_formula(formula)
    mass = sum(MASS[el] * n for el, n in comp.items())
    n_units = density / mass * N_A * 1e-24            # formula units per Å^3
    lam = energy_to_wavelength(energy_kev)
    pre = R_E * lam ** 2 / (2 * math.pi) * n_units
    d = b = 0.0
    for el, n in comp.items():
        fp, fpp = anomalous(el, energy_kev)
        d += n * (Z[el] + fp)
        b += n * fpp
    return pre * d, pre * b


def critical_angle(formula, density, energy_kev):
    """Critical angle of total external reflection (degrees): sqrt(2 delta)."""
    delta, _ = refraction(formula, density, energy_kev)
    return math.degrees(math.sqrt(2 * delta))


def penetration_depth(alpha_deg, formula, density, energy_kev):
    """1/e intensity penetration depth (Å) at incidence angle alpha (Dosch's formula)."""
    delta, beta = refraction(formula, density, energy_kev)
    a = math.radians(alpha_deg)
    ac2 = 2 * delta
    lam = energy_to_wavelength(energy_kev)
    B = math.sqrt(0.5 * (math.sqrt((a ** 2 - ac2) ** 2 + 4 * beta ** 2) - (a ** 2 - ac2)))
    return lam / (4 * math.pi * B) if B > 0 else math.inf


def footprint(beam_height_mm, alpha_deg, sample_length_mm=None):
    """Footprint length (mm) of a beam of the given height at incidence alpha, and the fraction of
    the beam that hits a sample of the given length (top-hat beam, centred)."""
    a = math.radians(alpha_deg)
    length = beam_height_mm / math.sin(a) if a > 0 else math.inf
    frac = 1.0 if sample_length_mm is None else min(1.0, sample_length_mm * math.sin(a) / beam_height_mm)
    return length, frac


def alpha_for_footprint(beam_height_mm, sample_length_mm):
    """Smallest incidence angle (degrees) at which the whole beam lands on the sample."""
    return math.degrees(math.asin(min(1.0, beam_height_mm / sample_length_mm)))


def two_theta(lattice, hkl, energy_kev):
    """2 theta (degrees) of a reflection; nan if it is beyond the reach of the energy."""
    q = float(lattice.q(hkl)[0])
    s = q * energy_to_wavelength(energy_kev) / (4 * math.pi)
    return math.degrees(2 * math.asin(s)) if s <= 1 else float("nan")


__all__ = ["HC", "MATERIALS", "refraction", "critical_angle", "penetration_depth", "footprint",
           "alpha_for_footprint", "two_theta", "parse_formula", "anomalous"]

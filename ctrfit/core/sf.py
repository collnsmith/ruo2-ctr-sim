"""Atomic form factors, anomalous terms and the atom-list structure factor."""
import numpy as np

HC = 12.398419843  # keV * Angstrom

CROMER_MANN = {  # International Tables Vol. C, neutral atoms: (a1..a4), (b1..b4), c
    "O":  ([3.0485, 2.2868, 1.5463, 0.8670], [13.2771, 5.7011, 0.3239, 32.9089], 0.2508),
    "Ti": ([9.7595, 7.3558, 1.6991, 1.9021], [7.8508, 0.5000, 35.6338, 116.105], 1.2807),
    "Ru": ([19.2674, 12.9182, 4.86337, 1.56756], [0.80852, 8.43467, 24.7997, 94.2928], 5.37874),
    "H":  ([0.489918, 0.262003, 0.196767, 0.049879], [20.6593, 7.74039, 49.5519, 2.20159], 0.001305),
}

F_ANOM_TABLE = {  # Chantler f', f'' (via xraydb), used when xraydb is not installed
    12.0: {"Ru": (-0.481, 1.627), "Ti": (0.367, 0.884), "O": (0.021, 0.014), "H": (-0.001, 0.0)},
    13.0: {"Ru": (-0.612, 1.410), "Ti": (0.352, 0.761), "O": (0.018, 0.011), "H": (-0.001, 0.0)},
    14.0: {"Ru": (-0.744, 1.233), "Ti": (0.333, 0.664), "O": (0.015, 0.010), "H": (-0.001, 0.0)},
    15.0: {"Ru": (-0.881, 1.085), "Ti": (0.313, 0.584), "O": (0.012, 0.008), "H": (-0.001, 0.0)},
    16.0: {"Ru": (-1.027, 0.963), "Ti": (0.293, 0.518), "O": (0.011, 0.007), "H": (-0.001, 0.0)},
    17.0: {"Ru": (-1.187, 0.862), "Ti": (0.274, 0.463), "O": (0.009, 0.006), "H": (-0.001, 0.0)},
    18.0: {"Ru": (-1.373, 0.777), "Ti": (0.256, 0.416), "O": (0.007, 0.006), "H": (-0.001, 0.0)},
    19.0: {"Ru": (-1.602, 0.704), "Ti": (0.240, 0.377), "O": (0.006, 0.005), "H": (-0.001, 0.0)},
    20.0: {"Ru": (-1.919, 0.643), "Ti": (0.225, 0.343), "O": (0.005, 0.004), "H": (-0.001, 0.0)},
    21.0: {"Ru": (-2.452, 0.590), "Ti": (0.211, 0.313), "O": (0.004, 0.004), "H": (-0.001, 0.0)},
}


def resolve_anomalous(p, notes=None):
    """(f', f'') per element and a description of the source, from the GUI settings."""
    if p["anom_mode"] == "off":
        return {}, "off"
    if p["anom_mode"] == "manual":
        return ({"Ru": (p["fp_Ru"], p["fpp_Ru"]), "Ti": (p["fp_Ti"], p["fpp_Ti"]),
                 "O": (p["fp_O"], p["fpp_O"]), "H": (0.0, 0.0)}, "manual")
    try:
        import xraydb
        e = p["energy_kev"] * 1e3
        return ({el: (float(xraydb.f1_chantler(el, e)), float(xraydb.f2_chantler(el, e)))
                 for el in ("Ru", "Ti", "O", "H")}, "Chantler (xraydb)")
    except ImportError:
        e_tab = min(F_ANOM_TABLE, key=lambda e: abs(e - p["energy_kev"]))
        if abs(e_tab - p["energy_kev"]) > 0.5 and notes is not None:
            notes.append(f"No anomalous table entry near {p['energy_kev']} keV (nearest {e_tab} keV). "
                         "Install xraydb or set f', f'' by hand.")
        return F_ANOM_TABLE[e_tab], f"built-in table at {e_tab:g} keV"


def f_atom(el, q, f_anom):
    """Complex scattering factor of a neutral atom at |q| (1/Å)."""
    a, b, c = CROMER_MANN[el]
    s2 = (np.asarray(q) / (4 * np.pi)) ** 2
    f = c + sum(ai * np.exp(-bi * s2) for ai, bi in zip(a, b))
    fp, fpp = f_anom.get(el, (0.0, 0.0))
    return f + fp + 1j * fpp


def sf_atoms(at, H, K, qz, q, f_of):
    """Structure factor of an atom list (dict of arrays el, f1, f2, z, occ, B) at one rod.

    qz, q: arrays of the out-of-plane and total momentum transfer; f_of(el, q) gives form factors.
    """
    F = np.zeros(qz.shape, complex)
    if at["el"].size == 0:
        return F
    ph_ip = np.exp(2j * np.pi * (H * at["f1"] + K * at["f2"]))
    for el in np.unique(at["el"]):
        m = at["el"] == el
        w = at["occ"][m] * ph_ip[m]
        dw = np.exp(-np.outer(q ** 2, at["B"][m]) / (16 * np.pi ** 2))
        ph = np.exp(1j * np.outer(qz, at["z"][m]))
        F += f_of(el, q) * ((dw * ph) @ w)
    return F

"""Atom lists for the rutile (110) trilayer and the CUS adsorbates.

An atom list is a dict of arrays: el, f1 (fraction of a1, along [001]), f2 (fraction of a2, along
[1-10]), z (Å along [110]), occ, B (Å²) and tag.
"""
import numpy as np

ADSORBATE_H = {  # H offsets from the O (dx || [001], dy || [1-10], dz || [110]) in Angstrom; crude
    "H2O": [(+0.76, 0.0, 0.59), (-0.76, 0.0, 0.59)],
    "OH": [(0.0, 0.0, 0.97)],
    "O": [],
}

# Adsorbed (bi)sulfate, monodentate on the CUS Ru: the bonding O (at the species height z_O), the S
# 1.47 Å straight above it, and the three upper O 0.49 Å above the S at 1.39 Å from the S-O axis (an
# ideal tetrahedron), their azimuth unknown: 12 positions every 30° at a quarter occupancy each (the
# ring average of four orientations). (el, dx || [001], dy || [1-10], dz from the bonding O, occupancy)
_SO, _RING_R, _RING_Z = 1.47, 1.47 * np.sin(np.arccos(-1 / 3)), 1.47 * -np.cos(np.arccos(-1 / 3))
ADSORBATE_HEAVY = {
    "SO4": [("S", 0.0, 0.0, _SO, 1.0)] + [("O", _RING_R * np.cos(a), _RING_R * np.sin(a), _SO + _RING_Z, 0.25)
                                         for a in np.radians(np.arange(0, 360, 30))],
}
ADSORBATE_H["SO4"] = []          # bisulfate H: not seen

FIELDS = ("el", "f1", "f2", "z", "occ", "B", "tag")


def oh_h2o(x_oh, theta=1.0):
    return {"OH": theta * x_oh, "H2O": theta * (1.0 - x_oh)}


def new_atoms():
    return dict(el=[], f1=[], f2=[], z=[], occ=[], B=[], tag=[])


def add_atom(at, el, f1, f2, z, occ, B, tag):
    for k, v in zip(FIELDS, (el, f1, f2, z, occ, B, tag)):
        at[k].append(v)


def finish_atoms(at):
    out = {k: np.array(v) for k, v in at.items()}
    for k in ("f1", "f2", "z", "occ", "B"):
        out[k] = out[k].astype(float)
    out["el"] = out["el"].astype(str)
    out["tag"] = out["tag"].astype(str)
    return out


def trilayer_sites(metal, d_cell, u, B_M, B_O, B_Olo=None):
    """The six sites of one (110) trilayer: (name, element, f1, f2, dz, B), parity 0."""
    h_br = (0.5 - u) * 2 * d_cell
    return [("M_6f", metal, 0.5, 0.0, 0.0, B_M),
            ("M_cus", metal, 0.0, 0.5, 0.0, B_M),
            ("O_ip", "O", 0.5, u, 0.0, B_O),
            ("O_ip", "O", 0.5, -u, 0.0, B_O),
            ("O_br", "O", 0.0, 0.0, +h_br, B_O),
            ("O_br_lo", "O", 0.0, 0.0, -h_br, B_O if B_Olo is None else B_Olo)]


def add_trilayer(at, metal, parity, z0, d_cell, u, occ, B_M, B_O, tag, skip=(), relax=None, B_Olo=None):
    s = 0.5 * parity
    r = relax or {}
    for name, el, f1, f2, dz, B in trilayer_sites(metal, d_cell, u, B_M, B_O, B_Olo):
        if name in skip:
            continue
        add_atom(at, el, f1, (f2 + s) % 1.0, z0 + dz + r.get(name, 0.0), occ, B, f"{tag}:{name}")

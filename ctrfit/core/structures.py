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

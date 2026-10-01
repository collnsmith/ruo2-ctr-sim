"""Model templates: turn GUI settings into fit parameters, and parameter values into engine inputs.

Template "rutile110_film": RuO2(110) film on TiO2(110) with OH / H2O / O on the CUS sites.

Composition on the CUS sites (any values inside the bounds are valid):
    O = theta * x_O,  OH = theta * (1 - x_O) * x_OH,  H2O = theta * (1 - x_O) * (1 - x_OH)
"""
from ..core.ctrmodel import CTRModel
from ..core.settings import DEFAULTS, parse_comp, parse_extra_layers

# name, unit, min, max, fit by default, group, description
_SPECS = [
    # ---------------------------------------------------------------- film
    ("thickness_mean_tl", "TL", 1.0, 300.0, True, "film",
     "Mean film thickness in (110) trilayers (continuous; Gaussian weights over whole trilayers)"),
    ("thickness_spread_tl", "TL", 0.0, 10.0, False, "film",
     "rms lateral thickness spread in trilayers (at least 0.3 TL is used, to keep the fit smooth)"),
    ("roughness_tl", "TL", 0.0, 5.0, True, "film", "rms roughness of the film top in trilayers"),
    ("eps_perp", "%", -5.0, 8.0, True, "film", "Out-of-plane strain of the commensurate film vs bulk RuO2"),
    ("interface_A", "Å", 1.5, 5.0, False, "film", "Top TiO2 metal plane to first RuO2 metal plane"),
    ("relax_001", "", 0.0, 1.0, False, "film",
     "[001] relaxation of the film above the commensurate part (0 commensurate, 1 bulk)"),
    ("coherent_tl", "TL", 0.0, 300.0, False, "film",
     "Commensurate bottom part in trilayers (rounded to a whole number; keep fixed)"),
    ("b_top_extra", "Å²", 0.0, 20.0, False, "film", "Extra B factor of the relaxed part"),
    ("b_film_Ru", "Å²", 0.0, 10.0, False, "film", "B factor of buried RuO2 Ru"),
    ("b_film_O", "Å²", 0.0, 10.0, False, "film", "B factor of buried RuO2 O"),
    ("b_sub_Ti", "Å²", 0.0, 10.0, False, "film", "B factor of TiO2 Ti"),
    ("b_sub_O", "Å²", 0.0, 10.0, False, "film", "B factor of TiO2 O"),
    # ---------------------------------------------------------------- surface
    ("b_surf_Ru", "Å²", 0.0, 20.0, False, "surface", "B factor of Ru in the exposed (top) trilayer"),
    ("b_surf_O", "Å²", 0.0, 20.0, False, "surface", "B factor of lattice O in the exposed trilayer"),
    ("dz_Ru6f", "Å", -0.5, 0.5, False, "surface", "Outward shift of top-layer 6-fold Ru"),
    ("dz_Obr", "Å", -0.5, 0.5, False, "surface", "Outward shift of top-layer bridging O"),
    ("dz_Oip", "Å", -0.5, 0.5, False, "surface", "Outward shift of top-layer in-plane O"),
    ("dz_Rucus_vac", "Å", -0.5, 0.5, False, "surface", "Outward shift of an empty CUS Ru"),
    ("dz_Rucus_OH", "Å", -0.5, 0.5, False, "surface", "Outward shift of a CUS Ru under OH"),
    ("dz_Rucus_H2O", "Å", -0.5, 0.5, False, "surface", "Outward shift of a CUS Ru under H2O"),
    ("dz_Rucus_O", "Å", -0.5, 0.5, False, "surface", "Outward shift of a CUS Ru under oxo O"),
    ("theta", "", 0.0, 1.0, True, "surface", "Coverage of the CUS sites (OH + H2O + O)"),
    ("x_OH", "", 0.0, 1.0, True, "surface", "OH share of the OH + H2O species"),
    ("x_O", "", 0.0, 1.0, False, "surface", "Oxo O share of the covered CUS sites"),
    ("z_OH", "Å", 1.0, 4.0, True, "surface", "Height of the OH oxygen above its CUS Ru"),
    ("z_H2O", "Å", 1.0, 4.0, False, "surface", "Height of the H2O oxygen above its CUS Ru"),
    ("z_O", "Å", 1.0, 4.0, False, "surface", "Height of the oxo O above its CUS Ru"),
    ("B_OH", "Å²", 0.0, 30.0, False, "surface", "B factor of the OH oxygen"),
    ("B_H2O", "Å²", 0.0, 30.0, False, "surface", "B factor of the H2O oxygen"),
    ("B_O", "Å²", 0.0, 30.0, False, "surface", "B factor of the oxo O"),
    ("el_z0", "Å", 0.0, 20.0, False, "surface", "Onset of the bulk electrolyte above the top metal plane"),
    ("el_sigma", "Å", 0.05, 10.0, False, "surface", "Width of the electrolyte onset"),
    # ---------------------------------------------------------------- scale
    ("scale", "", 1e-6, 1e6, True, "scale", "Overall scale of |F| (data = scale x model |F|)"),
]

_WATER_SPECS = [
    ("wl_z", "Å", 1.0, 8.0, False, "surface", "Height of the ordered water layer above the top metal plane"),
    ("wl_occ", "", 0.0, 2.0, False, "surface", "Occupancy of the ordered water layer per site"),
    ("wl_B", "Å²", 0.0, 60.0, False, "surface", "B factor of the ordered water layer"),
]

# parameter -> GUI setting with the same value and unit
_DIRECT = {
    "relax_001": "relax_001", "b_top_extra": "b_top_extra",
    "b_film_Ru": "b_film_M", "b_film_O": "b_film_O", "b_surf_Ru": "b_surf_M", "b_surf_O": "b_surf_O",
    "b_sub_Ti": "b_sub_M", "b_sub_O": "b_sub_O",
    "dz_Ru6f": "relax_M6f", "dz_Obr": "relax_Obr", "dz_Oip": "relax_Oip", "dz_Rucus_vac": "relax_Mcus_vac",
    "dz_Rucus_OH": "oh_dz", "dz_Rucus_H2O": "h2o_dz", "dz_Rucus_O": "o_dz",
    "z_OH": "oh_z", "z_H2O": "h2o_z", "z_O": "o_z", "B_OH": "oh_B", "B_H2O": "h2o_B", "B_O": "o_B",
    "el_z0": "water_z0", "el_sigma": "water_sigma", "interface_A": "interface_A",
}

DEFAULT_OPTIONS = dict(water_layer=False, water_layer_site="cus")


def composition(theta, x_OH, x_O=0.0):
    """CUS fractions from coverage and shares."""
    return {"O": theta * x_O, "OH": theta * (1 - x_O) * x_OH, "H2O": theta * (1 - x_O) * (1 - x_OH)}


def shares(comp):
    """(theta, x_OH, x_O) from a composition dict; inverse of composition()."""
    O, OH, H2O = comp.get("O", 0.0), comp.get("OH", 0.0), comp.get("H2O", 0.0)
    theta = O + OH + H2O
    x_O = O / theta if theta > 0 else 0.0
    x_OH = OH / (OH + H2O) if OH + H2O > 0 else 0.5
    return theta, x_OH, x_O


class Rutile110Film:
    name = "rutile110_film"

    def __init__(self, options=None):
        unknown = set(options or {}) - set(DEFAULT_OPTIONS)
        if unknown:
            raise KeyError(f"unknown template options {sorted(unknown)}")
        self.options = dict(DEFAULT_OPTIONS, **(options or {}))

    def specs(self):
        return _SPECS + (_WATER_SPECS if self.options["water_layer"] else [])

    def values_from_settings(self, settings):
        """Parameter values that reproduce the legacy engine for these GUI settings."""
        s = dict(DEFAULTS, **settings)
        m = CTRModel(s)
        v = {p: s[k] for p, k in _DIRECT.items()}
        v.update(thickness_mean_tl=float(m.n_film), thickness_spread_tl=m.film_n_spread,
                 roughness_tl=m.sigma_rough, eps_perp=100 * m.eps_perp, interface_A=m.Z_INT,
                 coherent_tl=float(m.n_coherent), scale=1.0)
        v["theta"], v["x_OH"], v["x_O"] = shares(parse_comp(s["comp_default"]))
        if self.options["water_layer"]:
            layers = [xl for xl in parse_extra_layers(s["extra_layers"]) if xl["el"] == "O"]
            xl = layers[0] if len(layers) == 1 else dict(z=3.4, occ=0.0, B=4.0)
            v.update(wl_z=xl["z"], wl_occ=xl["occ"], wl_B=xl["B"])
        return v

    def fixed_settings(self, settings):
        """The settings that stay fixed (everything the parameters do not replace)."""
        s = dict(DEFAULTS, **settings)
        if self.options["water_layer"]:
            layers = parse_extra_layers(s["extra_layers"])
            if len([xl for xl in layers if xl["el"] == "O"]) == 1:
                s["extra_layers"] = "; ".join(_layer_text(xl) for xl in layers if xl["el"] != "O")
        return s

    def engine_inputs(self, fixed, v):
        """(settings, geometry, comp) for FilmCTRModel from fixed settings and parameter values."""
        s = dict(fixed)
        for p, k in _DIRECT.items():
            s[k] = v[p]
        s["eps_perp_mode"], s["eps_perp_pct"] = "manual", v["eps_perp"]
        geometry = dict(mean_tl=v["thickness_mean_tl"], spread_tl=v["thickness_spread_tl"],
                        rough_tl=v["roughness_tl"], coherent_tl=v["coherent_tl"])
        if self.options["water_layer"]:
            site = self.options["water_layer_site"]
            geometry["extra_layers"] = parse_extra_layers(s["extra_layers"]) + [
                dict(el="O", site=site, z=v["wl_z"], occ=v["wl_occ"], B=v["wl_B"])]
        return s, geometry, composition(v["theta"], v["x_OH"], v["x_O"])


def _layer_text(xl):
    site = xl["site"] if isinstance(xl["site"], str) else f"{xl['site'][0]!r},{xl['site'][1]!r}"
    return f"{xl['el']} {site} {xl['z']!r} {xl['occ']!r} {xl['B']!r}"


TEMPLATES = {Rutile110Film.name: Rutile110Film}


def get_template(name, options=None):
    if name not in TEMPLATES:
        raise KeyError(f"unknown template {name!r}; available: {', '.join(TEMPLATES)}")
    return TEMPLATES[name](options)

"""Parameter schema, text parsers and INI helpers for the RuO2/TiO2 CTR simulator.

All values are stored in the units shown in the GUI (nm, %, Angstrom, keV ...).
The engine converts them to internal units.
"""
import configparser
import re

ADSORBATE_SPECIES = ("H2O", "OH", "O")

# Each group: (title, expanded_by_default, [param specs])
# spec keys: key, label, type (float|int|bool|choice|text), default, and optionally
# min, max, step, decimals, unit, choices, tip
GROUPS = [
    ("Beam and reciprocal space", True, [
        dict(key="energy_kev", label="Photon energy", type="float", default=16.0, min=5.0, max=40.0,
             step=0.5, decimals=2, unit="keV"),
        dict(key="two_theta_max", label="Max 2θ", type="float", default=60.0, min=5.0, max=170.0,
             step=1.0, decimals=1, unit="°", tip="Largest scattering angle reachable through the cell"),
        dict(key="L_min", label="L min", type="float", default=0.3, min=0.0, max=20.0, step=0.1, decimals=3),
        dict(key="L_max", label="L max", type="float", default=6.0, min=0.2, max=20.0, step=0.5, decimals=3),
        dict(key="L_step", label="L step", type="float", default=0.005, min=0.001, max=0.1, step=0.001,
             decimals=4),
    ]),
    ("Film", True, [
        dict(key="thickness_nm", label="Thickness", type="float", default=6.5, min=0.3, max=100.0,
             step=0.1, decimals=2, unit="nm",
             tip="Rounded to the nearest whole number of (110) trilayers"),
        dict(key="spread_nm", label="Thickness spread (rms)", type="float", default=0.32, min=0.0, max=10.0,
             step=0.05, decimals=2, unit="nm",
             tip="Lateral thickness variation, averaged incoherently. Washes out fringe minima"),
        dict(key="roughness_nm", label="Top roughness (rms)", type="float", default=0.16, min=0.0, max=5.0,
             step=0.02, decimals=3, unit="nm"),
        dict(key="interface_A", label="Interface spacing", type="float", default=0.0, min=0.0, max=10.0,
             step=0.01, decimals=3, unit="Å",
             tip="TiO2 top metal plane to first RuO2 metal plane. 0 = mean of the two (110) spacings"),
    ]),
    ("Strain", True, [
        dict(key="strain_001_pct", label="Strain along [001]", type="float", default=-4.7, min=-10.0,
             max=10.0, step=0.1, decimals=2, unit="%", tip="Relative to bulk RuO2. Literature: -4.7 %"),
        dict(key="strain_1m10_pct", label="Strain along [1-10]", type="float", default=2.4, min=-10.0,
             max=10.0, step=0.1, decimals=2, unit="%", tip="Relative to bulk RuO2. Literature: +2.3 %"),
        dict(key="eps_perp_mode", label="Out-of-plane strain from", type="choice", default="manual",
             choices=["manual", "elastic"],
             tip="manual: value below. elastic: from the elastic constants (advanced lattice section)"),
        dict(key="eps_perp_pct", label="Out-of-plane strain [110]", type="float", default=2.0, min=-10.0,
             max=10.0, step=0.1, decimals=2, unit="%",
             tip="Measured +2.0 % for commensurate films (Ruf et al. 2021). Replace with your own value"),
        dict(key="coherent_tol_pct", label="Pseudomorphic tolerance", type="float", default=0.5, min=0.0,
             max=5.0, step=0.1, decimals=2, unit="%"),
    ]),
    ("Relaxation along [001]", True, [
        dict(key="coherent_nm", label="Commensurate thickness", type="float", default=4.0, min=0.0,
             max=100.0, step=0.1, decimals=2, unit="nm",
             tip="Bottom part that stays commensurate. Rounded to whole trilayers"),
        dict(key="relax_001", label="Relaxation above it", type="float", default=0.0, min=0.0, max=1.0,
             step=0.05, decimals=3, tip="0 = commensurate, 1 = bulk RuO2 c. [1-10] strain is kept"),
        dict(key="b_top_extra", label="Extra disorder in relaxed part", type="float", default=0.0,
             min=0.0, max=20.0, step=0.1, decimals=2, unit="Å²"),
        dict(key="relaxed_rod_mode", label="H ≠ 0 rods see relaxed part", type="choice",
             default="integrated", choices=["integrated", "coherent_only"],
             tip="integrated: detector window also catches it (intensities add). "
                 "coherent_only: it falls outside the window"),
    ]),
    ("Surface composition and plots", True, [
        dict(key="comp_default", label="Surface state for plots", type="text", default="OH=0.5, H2O=0.5",
             tip="Species = fraction of CUS sites, e.g. 'OH=0.3, H2O=0.6' (rest vacant). Species: H2O, OH, O"),
        dict(key="x_oh_list", label="OH fractions to compare", type="text", default="0, 0.25, 0.5, 0.75, 1",
             tip="OH/(OH+H2O) values overlaid in the comparison tab"),
        dict(key="theta", label="CUS coverage for comparison", type="float", default=1.0, min=0.0, max=1.0,
             step=0.05, decimals=3),
        dict(key="mix_mode", label="Lateral mixing", type="choice", default="coherent",
             choices=["coherent", "domains"],
             tip="coherent: random mixing on the atomic scale. domains: patches larger than the coherence length"),
        dict(key="rods", label="Rods (H K)", type="text", default="0 0; 0 1; 1 0; 1 1",
             tip="Semicolon-separated H K pairs offered in the rod tabs"),
    ]),
    ("Sensitivity scan", True, [
        dict(key="sens_a", label="State A", type="text", default="OH=1"),
        dict(key="sens_b", label="State B", type="text", default="H2O=1"),
        dict(key="scan_h_max", label="Scan H up to", type="int", default=3, min=0, max=10),
        dict(key="scan_k_max", label="Scan K up to", type="int", default=6, min=0, max=20),
        dict(key="bragg_excl", label="Ignore near Bragg peaks", type="float", default=0.15, min=0.0,
             max=1.0, step=0.01, decimals=3, unit="L"),
        dict(key="rel_min", label="Min relative change", type="float", default=0.10, min=0.0, max=2.0,
             step=0.01, decimals=3, tip="Points below this never count in the SNR ranking"),
        dict(key="metric", label="Rank by", type="choice", default="snr", choices=["snr", "rel"],
             tip="snr: |dI|/sqrt(I + background), counting-limited. rel: |dI|/I"),
        dict(key="i_bg", label="Background", type="float", default=0.0, min=0.0, max=1e6, step=1.0,
             decimals=2, unit="e²"),
        dict(key="top_n", label="Points in ranking", type="int", default=30, min=1, max=500),
        dict(key="physical_aspect", label="True q proportions on map", type="bool", default=True),
    ]),
    ("Thickness and relaxation study", False, [
        dict(key="study_thick_nm", label="Thicknesses (nm)", type="text", default="2, 4, 6, 8, 10"),
        dict(key="study_relax", label="Relaxation values", type="text", default="0, 0.5, 1"),
        dict(key="study_points", label="Points", type="text",
             default="P1: 0 1 1.10; P2: 0 1 2.85; P3: 0 1 3.16; P4: 0 5 2.82; P5: 0 5 3.19; "
                     "P6: 0 3 4.79; (1 0 4.18): 1 0 4.18; (1 2 4.21): 1 2 4.21; C1: 0 0 3.0; C2: 0 2 3.0",
             tip="label: H K L; separated by semicolons"),
    ]),
    ("CUS adsorbates", False, [
        dict(key="h2o_z", label="H2O: Ru-O height", type="float", default=2.20, min=1.0, max=4.0, step=0.01,
             decimals=3, unit="Å", tip="DFT ~2.2 Å. In situ SXRD (Rao 2017) found up to ~2.67 Å at low potential"),
        dict(key="h2o_B", label="H2O: B factor", type="float", default=2.5, min=0.0, max=30.0, step=0.1,
             decimals=2, unit="Å²"),
        dict(key="h2o_dz", label="H2O: Ru_cus shift", type="float", default=0.0, min=-1.0, max=1.0,
             step=0.01, decimals=3, unit="Å"),
        dict(key="oh_z", label="OH: Ru-O height", type="float", default=2.00, min=1.0, max=4.0, step=0.01,
             decimals=3, unit="Å"),
        dict(key="oh_B", label="OH: B factor", type="float", default=1.5, min=0.0, max=30.0, step=0.1,
             decimals=2, unit="Å²"),
        dict(key="oh_dz", label="OH: Ru_cus shift", type="float", default=0.0, min=-1.0, max=1.0,
             step=0.01, decimals=3, unit="Å"),
        dict(key="o_z", label="O: Ru-O height", type="float", default=1.70, min=1.0, max=4.0, step=0.01,
             decimals=3, unit="Å"),
        dict(key="o_B", label="O: B factor", type="float", default=1.0, min=0.0, max=30.0, step=0.1,
             decimals=2, unit="Å²"),
        dict(key="o_dz", label="O: Ru_cus shift", type="float", default=0.0, min=-1.0, max=1.0,
             step=0.01, decimals=3, unit="Å"),
        dict(key="include_H", label="Include H atoms", type="bool", default=True),
        dict(key="b_H_extra", label="H B factor offset", type="float", default=1.0, min=0.0, max=30.0,
             step=0.1, decimals=2, unit="Å²"),
    ]),
    ("Top-layer relaxations", False, [
        dict(key="relax_M6f", label="Ru 6-fold", type="float", default=0.0, min=-1.0, max=1.0, step=0.01,
             decimals=3, unit="Å", tip="+ = outward along [110]"),
        dict(key="relax_Obr", label="Bridging O", type="float", default=0.0, min=-1.0, max=1.0, step=0.01,
             decimals=3, unit="Å"),
        dict(key="relax_Oip", label="In-plane O", type="float", default=0.0, min=-1.0, max=1.0, step=0.01,
             decimals=3, unit="Å"),
        dict(key="relax_Mcus_vac", label="Empty Ru_cus", type="float", default=0.0, min=-1.0, max=1.0,
             step=0.01, decimals=3, unit="Å"),
    ]),
    ("Electrolyte and extra layers", False, [
        dict(key="water_on", label="Bulk electrolyte (00L only)", type="bool", default=True),
        dict(key="water_rho", label="Electron density", type="float", default=0.334, min=0.0, max=2.0,
             step=0.01, decimals=3, unit="e/Å³"),
        dict(key="water_z0", label="Onset above top metal", type="float", default=3.5, min=0.0, max=20.0,
             step=0.1, decimals=2, unit="Å"),
        dict(key="water_sigma", label="Onset width", type="float", default=1.0, min=0.05, max=10.0,
             step=0.1, decimals=2, unit="Å"),
        dict(key="extra_layers", label="Extra ordered layers", type="text", default="",
             tip="element site z occ B; separated by semicolons. site = cus, br or f1,f2. "
                 "Example: O br 3.4 0.5 4.0; O cus 4.6 0.5 6.0"),
    ]),
    ("Debye-Waller factors", False, [
        dict(key="b_sub_M", label="TiO2 Ti", type="float", default=0.35, min=0.0, max=20.0, step=0.05,
             decimals=3, unit="Å²"),
        dict(key="b_sub_O", label="TiO2 O", type="float", default=0.55, min=0.0, max=20.0, step=0.05,
             decimals=3, unit="Å²"),
        dict(key="b_film_M", label="RuO2 Ru (buried)", type="float", default=0.40, min=0.0, max=20.0,
             step=0.05, decimals=3, unit="Å²"),
        dict(key="b_film_O", label="RuO2 O (buried)", type="float", default=0.60, min=0.0, max=20.0,
             step=0.05, decimals=3, unit="Å²"),
        dict(key="b_surf_M", label="Ru (top layer)", type="float", default=0.60, min=0.0, max=20.0,
             step=0.05, decimals=3, unit="Å²"),
        dict(key="b_surf_O", label="O (top layer)", type="float", default=1.00, min=0.0, max=20.0,
             step=0.05, decimals=3, unit="Å²"),
    ]),
    ("Anomalous scattering", False, [
        dict(key="anom_mode", label="f′, f″ source", type="choice", default="auto",
             choices=["auto", "manual", "off"],
             tip="auto: Chantler values (xraydb if installed, else built-in 12-21 keV table)"),
        dict(key="fp_Ru", label="Ru f′", type="float", default=-1.027, min=-30.0, max=30.0, step=0.01,
             decimals=3),
        dict(key="fpp_Ru", label="Ru f″", type="float", default=0.963, min=0.0, max=30.0, step=0.01,
             decimals=3),
        dict(key="fp_Ti", label="Ti f′", type="float", default=0.293, min=-30.0, max=30.0, step=0.01,
             decimals=3),
        dict(key="fpp_Ti", label="Ti f″", type="float", default=0.518, min=0.0, max=30.0, step=0.01,
             decimals=3),
        dict(key="fp_O", label="O f′", type="float", default=0.011, min=-30.0, max=30.0, step=0.001,
             decimals=3),
        dict(key="fpp_O", label="O f″", type="float", default=0.007, min=0.0, max=30.0, step=0.001,
             decimals=3),
    ]),
    ("Lattice and elastic constants", False, [
        dict(key="tio2_a", label="TiO2 a", type="float", default=4.5937, min=3.0, max=6.0, step=0.001,
             decimals=4, unit="Å"),
        dict(key="tio2_c", label="TiO2 c", type="float", default=2.9587, min=2.0, max=4.0, step=0.001,
             decimals=4, unit="Å"),
        dict(key="tio2_u", label="TiO2 u", type="float", default=0.3048, min=0.25, max=0.35, step=0.0001,
             decimals=4),
        dict(key="ruo2_a", label="RuO2 a", type="float", default=4.4919, min=3.0, max=6.0, step=0.001,
             decimals=4, unit="Å"),
        dict(key="ruo2_c", label="RuO2 c", type="float", default=3.1066, min=2.0, max=4.0, step=0.001,
             decimals=4, unit="Å"),
        dict(key="ruo2_u", label="RuO2 u", type="float", default=0.3058, min=0.25, max=0.35, step=0.0001,
             decimals=4),
        dict(key="C11", label="RuO2 C11", type="float", default=299.0, min=1.0, max=2000.0, step=1.0,
             decimals=1, unit="GPa", tip="DFT, Bohnen et al. PRB 75, 092301 (2007)"),
        dict(key="C12", label="RuO2 C12", type="float", default=246.0, min=0.0, max=2000.0, step=1.0,
             decimals=1, unit="GPa"),
        dict(key="C13", label="RuO2 C13", type="float", default=199.0, min=0.0, max=2000.0, step=1.0,
             decimals=1, unit="GPa"),
        dict(key="C66", label="RuO2 C66", type="float", default=227.0, min=0.0, max=2000.0, step=1.0,
             decimals=1, unit="GPa"),
        dict(key="alpha_abs", label="Substrate damping per cell", type="float", default=1e-4, min=1e-8,
             max=0.1, step=1e-4, decimals=6, tip="Numerical only; keeps Bragg peaks finite"),
    ]),
]

SPECS = {s["key"]: s for _, _, specs in GROUPS for s in specs}
DEFAULTS = {k: s["default"] for k, s in SPECS.items()}

# Parameter study: every numeric physics setting can be varied (not the L grid or the scan settings)
_NOT_STUDIED = {"two_theta_max", "L_min", "L_max", "L_step", "scan_h_max", "scan_k_max", "bragg_excl", "rel_min",
                "i_bg", "top_n", "coherent_tol_pct", "alpha_abs"}
STUDY_KEYS = [k for k, s in SPECS.items() if s["type"] == "float" and k not in _NOT_STUDIED]
STUDY_DEFAULT = "roughness_nm, 0.05, 0.4, 5; oh_dz, -0.2, 0.2, 5"


# ----------------------------------------------------------------------------------------------
# value conversion for INI files
# ----------------------------------------------------------------------------------------------
def to_text(key, value):
    t = SPECS[key]["type"]
    if t == "bool":
        return "true" if value else "false"
    if t == "float":
        return repr(float(value))
    return str(value)


def from_text(key, text):
    spec = SPECS[key]
    t = spec["type"]
    if t == "bool":
        return text.strip().lower() in ("1", "true", "yes", "on")
    if t == "int":
        return int(float(text))
    if t == "float":
        return float(text)
    if t == "choice":
        text = text.strip()
        return text if text in spec["choices"] else spec["default"]
    return text


def read_ini(path, section="params"):
    """Return (values dict merged onto defaults, list of problems). Unknown keys are ignored."""
    values, problems = dict(DEFAULTS), []
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    try:
        cp.read(path, encoding="utf-8")
    except configparser.Error as ex:
        return values, [f"could not read {path.name}: {ex}"]
    if cp.has_section(section):
        for key, text in cp.items(section):
            if key in SPECS:
                try:
                    values[key] = from_text(key, text)
                except ValueError:
                    problems.append(f"{key} = {text!r} is not valid, using the default")
    return values, problems


def write_ini(path, values, extra_sections=None):
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp["params"] = {k: to_text(k, values.get(k, DEFAULTS[k])) for k in SPECS}
    for name, items in (extra_sections or {}).items():
        cp[name] = {k: str(v) for k, v in items.items()}
    with open(path, "w", encoding="utf-8") as fh:
        cp.write(fh)


# ----------------------------------------------------------------------------------------------
# text parsers (raise ValueError with a readable message)
# ----------------------------------------------------------------------------------------------
def _split_items(text):
    return [s.strip() for s in re.split(r"[;\n]+", text) if s.strip()]


def parse_comp(text, what="composition"):
    comp = {}
    for item in re.split(r"[;,\n]+", text):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"{what}: '{item}' should look like OH=0.5")
        name, val = (s.strip() for s in item.split("=", 1))
        match = [s for s in ADSORBATE_SPECIES if s.lower() == name.lower()]
        if not match:
            raise ValueError(f"{what}: unknown species '{name}' (use {', '.join(ADSORBATE_SPECIES)})")
        try:
            frac = float(val)
        except ValueError:
            raise ValueError(f"{what}: '{val}' is not a number") from None
        if frac < 0:
            raise ValueError(f"{what}: fractions cannot be negative")
        comp[match[0]] = comp.get(match[0], 0.0) + frac
    if sum(comp.values()) > 1 + 1e-9:
        raise ValueError(f"{what}: fractions add up to {sum(comp.values()):.3f}, more than 1")
    return comp


def comp_label(comp):
    if not comp:
        return "empty CUS"
    return ", ".join(f"{k} {v:g}" for k, v in comp.items())


def parse_rods(text):
    rods = []
    for item in _split_items(text):
        parts = re.split(r"[\s,]+", item)
        try:
            h, k = (int(x) for x in parts)
        except ValueError:
            raise ValueError(f"rods: '{item}' should be two integers, e.g. 0 1") from None
        rods.append((h, k))
    if not rods:
        raise ValueError("rods: enter at least one H K pair")
    return rods


def parse_floats(text, what="list"):
    try:
        vals = [float(x) for x in re.split(r"[\s,;]+", text.strip()) if x]
    except ValueError:
        raise ValueError(f"{what}: use numbers separated by commas") from None
    if not vals:
        raise ValueError(f"{what}: enter at least one number")
    return vals


def parse_points(text):
    pts = {}
    for item in _split_items(text):
        if ":" in item:
            label, rest = (s.strip() for s in item.rsplit(":", 1))
        else:
            label, rest = item, item
        parts = re.split(r"[\s,]+", rest)
        try:
            h, k, l = int(parts[0]), int(parts[1]), float(parts[2])
            if len(parts) != 3:
                raise ValueError
        except (ValueError, IndexError):
            raise ValueError(f"points: '{item}' should look like 'P1: 0 1 1.1'") from None
        pts[label or f"({h} {k} {l:g})"] = (h, k, l)
    if not pts:
        raise ValueError("points: enter at least one point")
    return pts


def study_label(key):
    s = SPECS[key]
    return s["label"] + (f" ({s['unit']})" if s.get("unit") else "")


def parse_study_rows(text):
    """Parameter-study rows from 'key, min, max, steps; key, min, max, steps' -> [(key, lo, hi, n)].
    Values must lie within the setting's limits; steps from 2 to 50."""
    rows = []
    for item in _split_items(text.replace("\n", ";")):
        parts = [x.strip() for x in item.split(",")]
        if len(parts) != 4:
            raise ValueError(f"parameter study: '{item}' should be 'parameter, min, max, steps'")
        key = parts[0]
        if key not in STUDY_KEYS:
            raise ValueError(f"parameter study: '{key}' is not a numeric physics setting")
        try:
            lo, hi, n = float(parts[1]), float(parts[2]), int(parts[3])
        except ValueError:
            raise ValueError(f"parameter study: '{item}': min and max are numbers, steps a whole number") from None
        s = SPECS[key]
        for v in (lo, hi):
            if not s.get("min", -float("inf")) <= v <= s.get("max", float("inf")):
                raise ValueError(f"parameter study: {study_label(key)} = {v:g} is outside "
                                 f"{s.get('min')} to {s.get('max')}")
        if lo == hi:
            raise ValueError(f"parameter study: {study_label(key)}: min and max are equal")
        if not 2 <= n <= 50:
            raise ValueError(f"parameter study: {study_label(key)}: steps must be 2 to 50")
        rows.append((key, lo, hi, n))
    return rows


def format_study_rows(rows):
    return "; ".join(f"{k}, {lo:g}, {hi:g}, {n}" for k, lo, hi, n in rows)


def parse_extra_layers(text):
    layers = []
    for item in _split_items(text):
        parts = item.split()
        if len(parts) != 5:
            raise ValueError(f"extra layers: '{item}' needs 5 fields: element site z occ B")
        el, site, z, occ, b = parts
        if el not in ("O", "H", "Ru", "Ti"):
            raise ValueError(f"extra layers: element '{el}' must be O, H, Ru or Ti")
        if site not in ("cus", "br"):
            try:
                f1, f2 = (float(x) for x in site.split(","))
            except ValueError:
                raise ValueError(f"extra layers: site '{site}' must be cus, br or f1,f2") from None
            site = (f1, f2)
        try:
            layers.append(dict(el=el, site=site, z=float(z), occ=float(occ), B=float(b)))
        except ValueError:
            raise ValueError(f"extra layers: z, occ and B in '{item}' must be numbers") from None
    return layers

# %% [markdown]
# # CTR simulation notebook: RuO2(110) film on TiO2(110) with CUS OH / H2O
#
# Same engine and plots as the GUI (ctr_engine.py, ctr_plots.py, ctr_params.py in this folder).
# There is only one copy of the physics: change it in ctr_engine.py and both the GUI and this
# notebook follow.
#
# Settings use the same names and units as the GUI sidebar (nm, %, Angstrom, keV).
# You can start from the defaults, a preset, or the GUI's last settings (settings.ini).
#
# Conventions: H || [001], K || [1-10], L || [110], TiO2-indexed (TiO2 (110) Bragg peak = (0 0 2)).
# Run cell by cell in VS Code ("Run Cell"), or top to bottom as a script.

# %%
import sys
from pathlib import Path

try:
    APP_DIR = Path(__file__).resolve().parent
except NameError:  # VS Code Interactive Window and Jupyter run from the file's folder
    APP_DIR = Path.cwd()
if not (APP_DIR / "ctr_engine.py").exists():
    raise FileNotFoundError(f"ctr_engine.py not found in {APP_DIR}. Run this file from the ruo2_ctr_gui folder.")
sys.path.insert(0, str(APP_DIR))

import matplotlib.pyplot as plt
import numpy as np

import ctr_plots as plots
from ctr_engine import CTRModel, oh_h2o, parse_inputs, relaxation_study
from ctr_params import DEFAULTS, SPECS, comp_label, read_ini

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})


def new_fig(w, h):
    try:
        return plt.figure(figsize=(w, h), layout="constrained")
    except TypeError:  # matplotlib < 3.5
        return plt.figure(figsize=(w, h), constrained_layout=True)


# %% [markdown]
# ## Settings
# Pick a starting point, then override anything below. Every key and its unit is listed by
# `show_settings()`.

# %%
P = dict(DEFAULTS)
# P, _ = read_ini(APP_DIR / "settings.ini")                              # GUI's last settings
# P, _ = read_ini(APP_DIR / "presets" / "thick_10nm_relaxed_50pct.ini")  # a preset

P.update(                      # overrides; anything not listed keeps its value from above
    # thickness_nm=10.0,       # rounded to whole (110) trilayers
    # coherent_nm=4.0,         # commensurate bottom part
    # relax_001=0.5,           # [001] relaxation above it (0 to 1)
    # eps_perp_pct=2.0,        # measured d(110) expansion; replace with your film's value
    # rods="0 0; 0 1; 0 3; 0 5",
    # sens_a="OH=1", sens_b="H2O=1",
)
RUN_STUDY = True               # thickness and relaxation study (a few seconds)


def show_settings():
    for key, spec in SPECS.items():
        unit = spec.get("unit", "")
        print(f"{key:>18} = {P[key]!r:<28} {unit:<6} {spec['label']}")


INP = parse_inputs(P)
m = CTRModel(P)
print(m.summary())

# %% [markdown]
# ## 1. Structure sanity check

# %%
fig = new_fig(10, 5)
plots.draw_structure(fig, m, INP["comp_default"])
plt.show()

# %% [markdown]
# ## 2. Rods: TiO2 substrate, RuO2 film, total

# %%
for h, k in INP["rods"]:
    fig = new_fig(14, 3.8)
    plots.draw_rod_parts(fig, m.rod_parts(h, k, INP["comp_default"]), comp_label(INP["comp_default"]))
    plt.show()

# %% [markdown]
# ## 3. OH / H2O comparison

# %%
COMPS = {f"x_OH = {x:g}": oh_h2o(x, P["theta"]) for x in INP["x_oh_list"]}
for h, k in INP["rods"]:
    fig = new_fig(9, 6)
    plots.draw_compare(fig, m.compare(h, k, COMPS), P["rel_min"], P["bragg_excl"])
    plt.show()

# %% [markdown]
# ## 4. Sensitivity scan, ranking and H/K map

# %%
LABELS = (comp_label(INP["sens_a"]), comp_label(INP["sens_b"]))
RES, PEAKS = m.sensitivity_scan(INP["sens_a"], INP["sens_b"], P["scan_h_max"], P["scan_k_max"], P["metric"],
                                P["bragg_excl"], P["rel_min"], P["i_bg"])

print(f"A = {LABELS[0]}, B = {LABELS[1]}, mixing {P['mix_mode']}, ranked by '{P['metric']}'")
print(f"{'rank':>4} {'(H K L)':>17} {'|dI|/I':>8} {'SNR':>7} {'I mean (e^2)':>13}  brighter  H+K")
for i, p in enumerate(PEAKS[:P["top_n"]]):
    hkl = f"({p['H']} {p['K']} {p['L']:.2f})" + ("*" if p["edge"] else " ")
    print(f"{i + 1:>4} {hkl:>17} {p['rel']:>8.3f} {p['snr']:>7.3f} {p['I']:>13.3g}  "
          f"{'A' if p['A_gt_B'] else 'B':>8}  {'odd' if (p['H'] + p['K']) % 2 else 'even'}")
print("* = maximum sits next to a Bragg exclusion window or the end of the L range")

HQ, KQ = m.q_max * m.A1 / (2 * np.pi), m.q_max * m.A2 / (2 * np.pi)
COUNTS = m.bragg_counts(max(P["scan_h_max"], int(HQ)), max(P["scan_k_max"], int(KQ)))
fig = new_fig(15, 7)
plots.draw_hk_map(fig, m, RES, COUNTS, P["metric"], P["physical_aspect"], LABELS)
plt.show()

# %% [markdown]
# ## 5. Most sensitive rods, A vs B

# %%
shown = []
for p in PEAKS:
    if (p["H"], p["K"]) in shown:
        continue
    shown.append((p["H"], p["K"]))
    fig = new_fig(9, 5.5)
    plots.draw_ab(fig, RES[(p["H"], p["K"])], p["H"], p["K"], LABELS, p["L"], P["bragg_excl"])
    plt.show()
    if len(shown) == 4:
        break

# %% [markdown]
# ## 6. Thickness and [001] relaxation study

# %%
if RUN_STUDY:
    STUDY = relaxation_study(P, INP["study_points"], INP["study_thick"], INP["study_relax"],
                             INP["sens_a"], INP["sens_b"])
    for R in INP["study_relax"]:
        print(f"\nrelaxation {R:g}")
        print(f"{'point':>14}" + "".join(f"{t:>9g} nm" for t in INP["study_thick"]))
        for lab in INP["study_points"]:
            print(f"{lab:>14}" + "".join(f"{100 * STUDY[(lab, R, t)]:>+11.0f}%" for t in INP["study_thick"]))
    fig = new_fig(14, 4.5)
    plots.draw_study(fig, STUDY, INP["study_points"], INP["study_thick"], INP["study_relax"],
                     m.n_coherent * m.D_FILM / 10)
    plt.show()

# %% [markdown]
# ## 7. Checks worth running
# Change one setting at a time in the Settings cell and re-run sections 4 to 6:
#   * include_H=False                  -> H is nearly invisible to X-rays
#   * h2o_B equal to oh_B              -> isolates the pure height (geometry) sensitivity
#   * h2o_z=2.6                        -> longer Ru-H2O distance, closer to in situ SXRD values
#   * mix_mode="domains"               -> random mixing vs phase-separated domains
#   * relax_001, coherent_nm           -> relaxed top: H = 0 rods keep their sensitivity
#   * roughness_nm, spread_nm          -> how roughness and thickness spread wash out the contrast

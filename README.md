# RuO2 / TiO2(110) CTR simulator

Desktop GUI and notebook for the kinematic crystal truncation rod model of RuO2(110) films on
TiO2(110), with OH / H2O / O on the RuO2 CUS sites.

There is one copy of the physics (`ctrfit/core`, also importable as `ctr_engine`) and one copy of
the plots (`ctrfit/project/plots.py`, also `ctr_plots`).
The GUI (`ctr_gui.py`) and the notebook (`ctr_notebook.py`) both use them, so a change made there
shows up in both. This replaces the old standalone `ruo2_tio2_ctr_sim.py`.

## Run

```
pip install -r requirements.txt
pip install -e .        # optional: installs the ctrfit package
python ctr_launcher.py  # a card for every app; or start one directly, e.g. python ctr_gui.py
```

The launcher (dark theme) finds the apps by a tag line near the top of each file:

```
# @app title: Fitting | group: Fit | order: 10 | kind: gui | needs: PyQt5 | desc: What it does
```

`kind: gui` starts the app in its own window (it keeps running when the launcher closes);
`kind: script` runs it inside the launcher and prints to its console. Apps whose `needs` are not
installed are greyed out with the `pip install` hint. Add the tag to a new script and press Rescan
(F5); Ctrl+F filters.

Optional: `pip install xraydb` for exact anomalous scattering factors at any energy.
Without it a built-in 12 to 21 keV table is used.

## Notebook

Open `ctr_notebook.py` in VS Code and use Run Cell (Jupyter extension), or run it as a script.
Settings use the same names and units as the GUI. The Settings cell can start from a preset or from
the GUI's `settings.ini`, so both tools can share configurations. Run it from this folder.

## 3D viewer (for fun)

`python ctr_viewer3d.py` opens a separate window that builds one random real-space version of the
model (thickness, roughness islands, strain spacings, CUS species drawn from the surface state) and
ray traces it with CUDA through Numba: shadows, ambient occlusion and progressive anti-aliasing.
Without a CUDA GPU it falls back to the CPU at lower resolution.

- Settings can come from the defaults, any preset, or the main app's `settings.ini`.
- Mouse: left drag orbit, right or middle drag pan, wheel zoom, click an atom to identify it.
- Keys: R reset view, Space turntable, S save image (to `screenshots/`), A ambient occlusion,
  H hydrogens, W electrolyte water, B X-ray beams.
- X-ray beam panel: the incident beam (yellow) and the exit beam (cyan) to a reflection: pick H and
  K, move L with the slider, set the incidence angle. The panel shows 2θ, the incidence and exit
  angles and the sample azimuth for the current energy; on the specular rod (0 0 L) the incidence
  angle follows from L. Parts of a beam hidden behind atoms are drawn dashed. The geometry is in
  `ctrfit/core/geometry.py` (elastic scattering, k_out - k_in = q, fixed incidence angle).
- Island shapes, H orientations and the water are for display only; the layer occupancies and
  positions come from `ctr_engine.py`.
- Needs `numba` with CUDA support (newer Numba: `pip install numba numba-cuda`). The first start
  compiles the kernel, which takes a few seconds.

## Layout

- Left: settings, grouped in collapsible sections. Hover a label for help.
  Thickness, commensurate thickness, spread and roughness are entered in nm. The film and
  commensurate thickness are rounded to whole (110) trilayers; the line under each field shows the result.
- Right: Summary, Structure, Rods, OH vs H2O, Sensitivity map, Ranking, Thickness study, Log.
- Run with the button, Ctrl+R or F5. The same button stops a run.
  A yellow note appears when settings changed since the last run.
- Sensitivity map: hover a rod for its values, click it to open its curves in the Ranking tab.
- Each plot has a toolbar for zoom and saving. Tables export to CSV.
- Plots switch between side-by-side and stacked panels as the window changes shape.

## Files (all relative to this folder, so the folder can be moved or copied)

- `settings.ini`: last settings, window layout, and run options. Written after each run and on exit.
  Delete it to start from defaults.
- `presets/*.ini`: saved configurations. Use Save as / Load / Delete in the sidebar.
  Presets are plain text and can be shared or edited.
- `ctrfit/`: the package. `ctrfit/core` physics (form factors, lattice, structures, electrolyte, the
  reference `CTRModel`, the settings list), `ctrfit/project/plots.py` plotting. See "Fitting" below.
- `ctr_engine.py`, `ctr_params.py`, `ctr_plots.py`: compatibility names for `ctrfit.core.ctrmodel`,
  `ctrfit.core.settings` and `ctrfit.project.plots`, so older scripts keep working.
- `ctr_gui.py` window, `ctr_notebook.py` notebook front end, `ctr_viewer3d.py` 3D viewer.
- `tests/`: pytest suite (`python -m pytest`; long tests with `python -m pytest -m slow`).
- `screenshots/`: images saved from the 3D viewer (created when needed).

## Text field formats

- Surface state: `OH=0.5, H2O=0.5` (fractions of CUS sites; the rest is empty). Species: H2O, OH, O.
- Rods: `0 0; 0 1; 1 0`
- Study points: `P1: 0 1 1.10; P2: 0 1 2.85` (label: H K L)
- Extra layers: `O br 3.4 0.5 4.0` (element, site cus / br / f1,f2, height in Å, occupancy, B)

## Fitting (ctrfit, scriptable core)

`ctrfit` fits measured structure factors |F|(H, K, L) with errors to the same model. There is no
fitting GUI yet; use the Python API or the command line.

```python
from ctrfit import Dataset, Fit, Model
from ctrfit.core.settings import DEFAULTS

model = Model.from_settings(DEFAULTS)          # template rutile110_film; parameters from the settings
print(model.params.table())                    # name, value, bounds, unit, fit flag, description
model.params.fit_only(["theta", "x_OH", "z_OH", "scale"])
model.params.link("z_H2O", "z_OH + 0.4")       # links are expressions of other parameters
data = Dataset.load("data.csv")                # header H,K,L,F,sigma (or .dat: H K L F sigma)
data.sys_floor = 0.03                          # systematic error floor, added in quadrature

fit = Fit(model, data)
fit.run("de", maxiter=40, popsize=12, seed=1)  # global search
fit.refine()                                   # least squares, errors from the Jacobian
res = fit.report()
print(res.summary)                             # values, errors, correlations, warnings
res.save("result.json")
```

Fitting window: `python ctr_fit_gui.py [data.csv] [model.json]` (or `python -m ctrfit gui`). Left:
datasets (with the systematic error floor) and the parameter table (tick Fit, edit value, bounds and
links; errors appear after a fit, parameters named in warnings are highlighted). Right: fit settings,
Run / Stop (during a fit the model curve of the shown rod and the Value column follow the best
parameters after every differential-evolution generation and as least squares improves), and the guided workflow (1. film on the even rods with a thickness scan, 2. surface on
the odd rods, 3. everything together). Tabs: data vs model with residuals per rod, live figure of
merit, report with warnings, correlation matrix, per-dataset series for global fits. File menu:
models, data, project files (`*.ctrproj.json`: model, data and results together) and a report folder.

**Ask Claude** (dock on the right of the fitting window, View menu to show it): type a request ("fit
the film first, then tell me whether x_OH is determined") or use a quick button. Claude
(claude-opus-5-5) runs the fit with the same tools the window has (state, set values/bounds/fit
flags, links, per-dataset copies, guided steps, DE, refine, thickness profile, report, per-rod
residuals, AIC/BIC model comparison) on a copy of the model, while the rod plot follows its best
parameters live. When it answers, its result is applied; "Undo Claude's changes" restores the
previous model. Stop with the Run fit button. The panel shows the token use and an approximate cost;
the transcript can be saved. Needs `pip install anthropic` and an API key (`ANTHROPIC_API_KEY`, or
`ant auth login`). Headless: `python -m ctrfit ask model.json data.csv -r "..."`.

Command line (headless; writes result.json, summary.txt and PNG plots per rod):

```
python -m ctrfit fit model.json data.csv --out fit_results
```

- Parameters are in trilayers (TL), Å, Å², % or dimensionless. The film thickness is a continuous
  mean (`thickness_mean_tl`) of a Gaussian distribution over whole trilayers.
- CUS composition: O = theta x_O, OH = theta (1 - x_O) x_OH, H2O = theta (1 - x_O) (1 - x_OH).
- Figures of merit: chi2 (with sigma_eff), a GenX-style log figure of merit, and R1.
- Every report warns about strongly correlated parameters (|r| > 0.9), values at a bound and
  relative errors above 100 %. The OH vs H2O ratio with free heights and B factors is the known
  degenerate case.
- Thickness fringes give separate chi2 minima about one trilayer apart (traded against eps_perp).
  Run `fit.profile("thickness_mean_tl", grid)` after the global search to pick the right one.
- Beyond the covariance errors: `ctrfit.fit.bootstrap(fit, n=100)`, `ctrfit.fit.mcmc(fit)` (needs
  `emcee`), and `ctrfit.fit.compare_models([res_a, res_b], labels)` for AIC / BIC between surface
  models (e.g. OH only vs OH + H2O vs an extra water layer).
- Global fits across potentials or thicknesses: `global_fit(model, datasets, per_dataset=["x_OH",
  "z_OH"])` (from `ctrfit.fit.global_fit`) shares all other free parameters (the film) and gives each
  dataset its own copy of the listed ones and its own scale; `series(result, "x_OH")` returns the
  values against `potential_V`. Command line: `--per-dataset x_OH,z_OH`.
- Worked example with synthetic data: `examples/synthetic_ruo2/` (`run_fit.py`: film on the even
  rods, then the surface on the odd rods, then everything together).

## Beamline helper (SPEC psic)

`python ctr_beamline.py` (or `python -m ctrfit beamline`) opens a helper for the run itself. The
computation is in `ctrfit/beamline` and works from scripts too.

- **Bragg peak indexer.** Enter the motor angles of the Bragg peaks you find (one per line, SPEC psic
  order `del eta chi phi nu mu`, or `del=… eta=…`; text after `#` is a label), the energy and the
  lattice. "Guess HKL" finds the orientation that puts most peaks on allowed reflections, refines it
  and gives each peak's HKL, its misfit and the UB matrix. Allowed reflections come from the rutile
  structure factor (K + L even and the rutile extinctions), so a peak on forbidden integers is
  flagged. With one peak you get the candidates by |Q|. Bragg peaks alone cannot tell K from L in
  the TiO2(110) surface cell (a2 = a3), nor the sign of H and K: all equivalent indexings are listed,
  and the surface normal in the phi frame (e.g. `0 0 1` if it is along the phi axis) picks the one
  with L along it. Command line: `python -m ctrfit index peaks.txt --energy 16 --normal "0 0 1"`.
- **Angles.** UB from the indexer or from two reflections (like `or0` / `or1`). Angles for an HKL in a
  surface mode (horizontal or vertical surface, fixed incidence angle or alpha = beta) or four-circle
  mode, with the SPEC `umv` line; and HKL, alpha and beta from a set of angles.
- **Macro maker.** SPEC command files for rod scans (Bragg peaks skipped, finer steps near them, count
  times from the model so weak anti-Bragg points get more time, optional rocking scans), fixed
  points (e.g. the sensitivity ranking), potential series and energy scans, with a time estimate. The
  commands (`br`, `ct`, the potentiostat and energy macros, shutters) are editable templates, and
  the file can be wrapped in a `def`. Command line: `python -m ctrfit macro --rods "0 1; 1 0"`.
- **Calculators.** Wavelength, |Q|, d and 2theta of a reflection; critical angle and penetration depth
  of TiO2, RuO2, water, Si and Kapton; footprint and the share of the beam that hits the sample.

Geometry: You, J. Appl. Cryst. 32, 614 (1999), as in SPEC psic: beam along +y, x up; mu and nu turn
about the vertical axis, eta, delta and phi about the horizontal one. If a motor at your beamline
turns the other way, set its sign (e.g. `eta=-1`). Check the convention once against a known
reflection before trusting the indexer at a new beamline.

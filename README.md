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
python ctr_gui.py
```

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
Run / Stop, and the guided workflow (1. film on the even rods with a thickness scan, 2. surface on
the odd rods, 3. everything together). Tabs: data vs model with residuals per rod, live figure of
merit, report with warnings, correlation matrix, per-dataset series for global fits. File menu:
models, data, project files (`*.ctrproj.json`: model, data and results together) and a report folder.

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

# CLAUDE.md

## What this is

A kinematic crystal truncation rod (CTR) simulator for RuO2(110) films on TiO2(110), with OH / H2O /
O adsorbates on the RuO2 CUS sites, a bulk electrolyte term, thickness spread, roughness and a
partly relaxed film top. It is growing into a fitting package (`ctrfit/`) by the phases in
`docs/FITTING_PLAN.md`. Read that plan (and its Status section) before starting a phase.

## Conventions

- Surface cell, in TiO2 surface units: H || [001], K || [1-10], L || [110], with
  a1 = c_TiO2, a2 = a3 = sqrt(2) a_TiO2. The TiO2 (110) Bragg peak is (0 0 2).
- The in-plane film lattice is locked to TiO2; z is in Å above the top TiO2 metal plane.
- Settings use GUI units (nm, %, Å, keV, Å²). The engine converts them to internal units.
- In the GUI engine, film and commensurate thickness are whole (110) trilayers. The fitting model
  uses a continuous mean thickness (`thickness_mean_tl`): Gaussian weights over whole trilayer counts,
  spread at least 0.3 TL, smooth window (see `ctrfit/core/film.py`). That is the only physics
  difference between the fitting model and the GUI engine.
- Fit parameters are in trilayers (TL), Å, Å², % (eps_perp) or dimensionless; data are |F| and
  F_data = scale * rodscale * |F_model|.
- Parameter names can be scoped to a dataset (`ds2.oh_z`); links look in their own scope first.

## Layout and rules

```
ctrfit/core      physics: sf.py (form factors, anomalous terms, atom-list structure factor),
                 lattice.py, structures.py (trilayer, adsorbates), electrolyte.py,
                 ctrmodel.py (the reference CTRModel), settings.py (GUI settings schema and parsers),
                 film.py (continuous-mean thickness weights, FilmCTRModel driven by parameters),
                 fast.py (vectorized |F|^2 for many points, cached; must equal CTRModel to 1e-9),
                 geometry.py (k_in, k_out for a reflection; used by the 3D viewer's beams)
ctrfit/model     expr.py (safe link expressions), parameters.py (Parameter, ParameterSet),
                 templates.py (rutile110_film: settings <-> parameters), model.py (Model),
                 evaluator.py (fast vectorized |F|)
ctrfit/data      dataset.py (Dataset, sigma_eff with the systematic floor), io.py (CSV, .dat,
                 I -> F), synthetic.py (make_dataset with fixed seeds)
ctrfit/fit       fom.py (chi2, log, R1), fit.py (Fit: DE, least squares, profile, report;
                 FitResult), uncertainty.py (covariance, correlations, warnings),
                 sampling.py (bootstrap, emcee MCMC, AIC/BIC, compare_models),
                 global_fit.py (shared vs per-dataset parameters, series),
                 workflow.py (guided steps: film on even rods, surface on odd rods, all)
ctrfit/project   plots.py (all plotting, simulator and fits), report.py (writes result files),
                 project.py (Project: model + datasets + results in one JSON file)
ctrfit/beamline  psic.py (SPEC psic geometry, UB, HKL <-> angles), indexing.py (Bragg peak indexer),
                 macros.py (SPEC macro maker), xtools.py (critical angle, footprint, ...)
ctrfit/app       fit_window.py, beamline_window.py (Qt windows; the only Qt code in ctrfit)
ctrfit/cli.py    python -m ctrfit fit model.json data.csv | python -m ctrfit gui
examples/        synthetic_ruo2: generated data, model.json, run_fit.py
ctr_engine.py, ctr_params.py, ctr_plots.py   compatibility names: each *is* the ctrfit module
ctr_gui.py, ctr_notebook.py, ctr_viewer3d.py front ends; ctr_fit_gui.py, ctr_beamline.py launch the ctrfit windows
```

- All physics lives in `ctrfit/core`; diffractometer geometry lives in `ctrfit/beamline`. All plotting
  lives in `ctrfit/project/plots.py`.
- `ctr_gui.py`, `ctr_notebook.py` and `ctr_viewer3d.py` only call the engine and the plots. They
  hold no physics of their own.
- Settings, their units, defaults and the text parsers live in `ctrfit/core/settings.py`
  (imported as `ctr_params` by the front ends).
- No Qt imports anywhere in `ctrfit/` except `ctrfit/app/` (a test enforces this). The window only
  calls ctrfit; fitting logic it needs belongs in `ctrfit/fit` (e.g. workflow.py) so it is testable
  without Qt. Parameter-set changes (per-dataset copies) happen on the GUI thread, never in the worker.
- Never edit `tests/reference.npz`. If a change is meant to alter the physics, regenerate it with
  `python tests/make_reference.py` and explain why in the commit message.
- Fixed seeds everywhere (synthetic data, optimizers, the viewer's random scene).

## Tests

```
pip install -r requirements.txt pytest
pip install -e .               # optional: makes ctrfit importable from anywhere
python -m pytest              # default suite, about 2 min; skips tests marked slow
python -m pytest -m slow      # long tests (direct combined fit, example fit, potential series, GUI end to end, coverage), about 3.5 min
```

- The cloud environment has no GPU and no display. GUI tests run with `QT_QPA_PLATFORM=offscreen`
  (set in `tests/conftest.py`), plots use the Agg backend.
- `tests/conftest.py` blocks `xraydb`, so "auto" anomalous terms always come from the built-in
  table and the pinned numbers are machine independent.
- Viewer tests: the CPU renderer is compared with the CUDA kernel run in Numba's CUDA simulator,
  and the kernel is compiled to PTX for sm_75 (fake device) to check it is float32 only. They skip
  cleanly without `numba` / `numba-cuda`. In the cloud, `numba-cuda` also needs the CUDA runtime
  library: `pip install nvidia-cuda-runtime-cu12`. MCMC tests skip without `emcee`.

What the tests protect:

| Test | Protects |
|---|---|
| `test_regression.py` rods | total, film, substrate and domain-mode intensities of rods 00, 01, 10, 11, 05 (L 0.3 to 6) at defaults and for a 10 nm film with relax_001 = 0.5, to rtol 1e-9 |
| `test_regression.py` ranking, study | top 10 of the sensitivity ranking and every value of the default thickness/relaxation study |
| `test_physics.py` bulk | substrate cell = 2 x conventional rutile structure factor, zero for K + L odd |
| `test_physics.py` continuity | a film with TiO2's lattice (and Ti scattering) continues the substrate rod (< 1e-5) |
| `test_physics.py` electrolyte | the analytic electrolyte term equals a direct numerical integral of the erfc profile |
| `test_apps.py` | the GUI starts offscreen and finishes one run; the notebook runs as a script |
| `test_viewer.py` | the 3D scene builds; CPU and CUDA (simulator) renders agree; the kernel PTX has no f64 |
| `test_package.py` | no Qt in `ctrfit`; `ctr_engine/params/plots` are the `ctrfit` modules |
| `test_parameters.py`, `test_template.py` | safe links, scopes, bounds, JSON; settings -> parameters -> model equals the reference; every parameter changes the model; continuous thickness weights; composition mapping |
| `test_evaluator.py` | fast evaluator equals the reference (rtol 1e-9) incl. random parameters; caching; >= 10x benchmark |
| `test_data.py` | FOMs and I -> F against hand values; CSV / .dat / JSON round trips; CLI |
| `test_fit.py` | synthetic recovery: surface-only, film-only, combined, degenerate case warns, every parameter alone, rod scales, two datasets |
| `test_uncertainty.py` | bootstrap and MCMC agree with the covariance; AIC/BIC choose the right model; coverage (slow) |
| `test_global.py` | per-dataset copies and links; synthetic potential series: trend recovered, shared parameters tighter than one dataset; CLI global fit (slow) |
| `test_workflow.py` | guided steps recover film and surface; per-dataset copies in steps; cancel keeps the best values; project round trip |
| `test_fit_gui.py` | fitting window offscreen: load, table edits (values, bounds, links, fit flags), guided steps end to end, stop, global fit, project save/open, report export |
| `test_geometry.py` | beam geometry (elastic, k_out - k_in = q, specular, unreachable cases); viewer draws the beams; occlusion test |
| `test_beamline.py` | psic identities (|Q|, 2theta, bisecting, motor signs), HKL -> angles -> HKL in every mode, UB from two reflections, extinctions, indexing with and without the normal hint (forbidden and junk peaks), macros, calculators (Si critical angle), CLI, window offscreen |
| `test_example.py` | the example data are reproducible; the example fit recovers the truth (slow) |

## Checklist per PR (docs/FITTING_PLAN.md section 7)

- [ ] Tests pass (Phase 0 regression tests unchanged unless the PR says why)
- [ ] New code has tests, including a synthetic recovery test for any new fit parameter
- [ ] No Qt imports outside `app/`
- [ ] User-visible parameters have units and descriptions
- [ ] `CLAUDE.md` and the plan updated if the architecture changed
- [ ] Checked locally: GUI starts, 3D viewer runs (cloud has no display or GPU)

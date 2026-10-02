# Plan: from CTR simulator to an easy-to-use CTR fitting package

## Status

| Phase | State | Notes |
|---|---|---|
| 0 Safety net | done | `tests/`: regression pins in `tests/reference.npz`, physics checks, app smoke tests |
| 1 Package layout | done | `ctrfit/` with core, model, data, fit, project; `ctr_engine/params/plots.py` are compatibility names; `pyproject.toml` |
| 2 Parameter model | done | `ctrfit/model`: Parameter, ParameterSet (safe AST links, scopes, bounds, JSON), template `rutile110_film`, continuous thickness mean (`ctrfit/core/film.py`) |
| 3 Fast evaluator | done | `ctrfit/core/fast.py` (vectorized, content-keyed caches) + `ctrfit/model/evaluator.py`; about 2 ms per evaluation for 2000 points with surface parameters free (about 80x legacy); numpy only, numba not needed after profiling |
| 4 Data and fitting core | done | `ctrfit/data`, `ctrfit/fit`, `python -m ctrfit fit`, `examples/synthetic_ruo2`; recovery tests for surface-only, film-only, combined, degenerate case, and every parameter alone; `Fit.profile` for the thickness minima |
| 5 Uncertainty | done | `ctrfit/fit/sampling.py`: bootstrap, emcee MCMC, AIC/AICc/BIC, `compare_models`; coverage test over 20 realizations (slow) |
| 6 Fitting GUI | done (cloud-tested offscreen; look and feel still to check locally) | `ctrfit/app/fit_window.py` (`python ctr_fit_gui.py` or `python -m ctrfit gui`): datasets with error floor, parameter table (fit, value, bounds, links, errors, warnings highlighted, group filter), guided steps (`ctrfit/fit/workflow.py`), run/stop with live figure of merit, report, correlations, series, project files (`ctrfit/project/project.py`). The simulator stays in `ctr_gui.py` |
| 7 Global fits | done | `ctrfit/fit/global_fit.py`: per-dataset scoped copies (`ds2.x_OH`), shared film, per-dataset scales, cross-dataset links, `series()` and series plots, CLI `--per-dataset`; synthetic potential series test (trend recovered, shared errors about 2x tighter with 4 datasets) |
| 8, 9 | not started | |

Open issues found while building phases 0 to 5:
- Film thickness has separate chi2 minima about one trilayer apart (traded against eps_perp).
  Differential evolution alone found the right one in only about 3 of 5 seeds on the example data, so
  the film step is DE, then `Fit.profile("thickness_mean_tl", grid)`, then refinement. The
  profile also re-refines its best few grid points with the thickness free (`polish=3`), because
  a coarse grid point next to the narrow true minimum can score worse than one in a wrong basin.
  The guided workflow (Phase 6) should do this automatically.
- Global fits share the fixed settings (energy, lattice, anomalous terms) across datasets; only
  parameters can differ per dataset. Datasets at different energies need per-dataset settings.
- With the thickness spread at its 0.3 TL floor, the mean thickness acts in steps (smooth but
  flat near whole trilayers). Fit the spread, or keep it at a realistic value (default ~1 TL).
- MCMC samples exp(-chi2/2); when the model misfits (reduced chi2 >> 1) use
  `mcmc(..., temper_by_red_chi2=True)` to match the scaled covariance errors.
- Tests block xraydb so pinned numbers do not depend on the machine; real fits use xraydb if installed.
- The cloud image needs `pip install nvidia-cuda-runtime-cu12 emcee` for the CUDA-simulator, PTX and
  MCMC tests to run instead of skipping.

Put this file in the repo as `docs/FITTING_PLAN.md` and point to it from `CLAUDE.md`, so every cloud
session reads it. Work one phase at a time; each phase ends with passing tests and a merged PR.

---

## 1. Decision: evolve the current project, do not start from scratch

| Option | For | Against |
|---|---|---|
| Start from scratch | Clean architecture from day one | Throws away a physics engine that is already verified (bulk structure factors against standard rutile, film/substrate continuity, electrolyte term, literature lattice and strain values). A rewrite re-derives the geometry and conventions and can reintroduce bugs that are now tested away. Costs credits on work that already exists |
| Bolt fitting onto the current code | Fastest start | `ctr_gui.py` is built around one simulation workflow. The engine rebuilds atoms on every call (too slow for fitting) and takes a flat settings dict, with no notion of free parameters, bounds or constraints. Fitting code would end up tangled with GUI code |
| **Evolve (recommended)** | Keep the verified physics; restructure it into a package with a parameter layer, a fast evaluator, a data layer and a fitting core. The current GUI, notebook and viewer keep working throughout | Needs discipline: every step is gated by regression tests against today's numbers |

**Rule for every phase:** the existing regression tests must pass unchanged unless a phase explicitly
says the physics changes. In that case the PR explains why and updates the reference values.

### Consider first: do you need to build this?

- **Fastest route to results for the RuO2 paper:** wrap `ctr_engine.py` as a custom GenX model, or
  fit from the notebook with `scipy` / `lmfit` (Phase 4 below delivers this early anyway).
- **Build the package if** you want a reusable tool: thesis software, other systems (Au(111) in an
  electrochemical cell is the obvious second case), global fits across potentials and thicknesses,
  and uncertainty analysis that GenX does not make easy.

The plan delivers a scriptable fitting API (Phase 4) before any fitting GUI, so publishable fits
are possible early even if the GUI work stalls.

**Out of scope:** reducing area-detector images to integrated intensities. Use the beamline's
tools; this package starts from structure factors (or intensities plus documented corrections).

---

## 2. What "easier than GenX" means here

1. **Models from templates, not code.** Pick "film on substrate + surface adsorbates", choose the
   bulk structures (built-in or CIF), and get a parameter table. Python scripting stays available
   for advanced users.
2. **A parameter table that explains itself.** Each parameter has a name, unit, value, bounds and
   a fit checkbox, plus a plain description ("Ru_cus to OH oxygen height"). Linked parameters are
   written as expressions (`oh_z = h2o_z - 0.2`, or shared across datasets).
3. **A guided fitting workflow** that matches the scan plan: fit film parameters on the even rods,
   then the surface on the odd rods, then global fits across potentials, with sensible defaults.
4. **Uncertainties and correlations by default.** Every fit reports error bars, the correlation
   matrix and warnings for parameters the data cannot pin down (OH vs H2O occupancy against height
   and B is the known case).
5. **Projects, not loose files.** One project file holds data, model, parameters, fit history and
   results, so a fit can be reproduced exactly.
6. **Global fits.** Several datasets (potentials, film thicknesses) share film parameters while
   surface parameters vary per dataset.

---

## 3. Target architecture

```
ctrfit/
  core/        lattice.py, structures.py (templates, CIF import), sf.py (form factors,
               anomalous terms, structure factors), film.py (layers, roughness, relaxation),
               electrolyte.py
  model/       parameters.py (value, bounds, fit flag, units, links), constraints.py,
               model.py (template -> parameterized structure -> fast evaluator)
  data/        dataset.py (H K L F sigma, rod grouping, systematic error floor),
               io.py (CSV, GenX/ANA-ROD style text), synthetic.py
  fit/         fom.py (chi2, log, R1), optimizers.py (differential evolution, least squares,
               MCMC), uncertainty.py, global_fit.py
  project/     project.py (save/load a single project file), report.py (tables, figures)
  app/         Qt application: simulation mode (current tabs) and fitting mode
  cli.py       command-line fitting for batch jobs
tests/         regression, physics checks, synthetic recovery, GUI smoke tests
examples/      RuO2/TiO2 tutorial project with synthetic data
legacy/        current scripts, kept runnable until the new app replaces them
```

Hard rules (put them in `CLAUDE.md`):
- **Pure computation:** `core`, `model`, `data` and `fit` never import Qt.
- **One copy of the physics:** structure-factor physics lives only in `core`.
- **Tested parameters:** every user-visible parameter has a unit, a description and a test that
  changing it changes the model.

---

## 4. Phases

Each phase lists what to build, how to check it, and a starting prompt for a cloud session. Use
**Plan** mode for every phase that touches physics or architecture.

### Phase 0: Safety net (do first)
- **Build:** `CLAUDE.md`, a pytest suite pinning today's results (rod intensities, sensitivity
  ranking, study values), the physics checks already done by hand (bulk structure factor vs
  rutile, continuity, electrolyte integral), and offscreen smoke tests for the GUI and notebook.
- **Done when:** `pytest` passes in the cloud environment and locally.
- **Prompt:** see the "first session" prompt from the setup notes.

### Phase 1: Package restructure (no physics change)
- **Build:** move the engine into `ctrfit/core` and `ctrfit/model`. Keep thin compatibility
  imports so `ctr_gui.py`, `ctr_notebook.py` and `ctr_viewer3d.py` keep working. Settings keep
  their GUI units.
- **Done when:** all Phase 0 tests pass unchanged and the three apps start.
- **Prompt:**
  ```
  Read docs/FITTING_PLAN.md and CLAUDE.md. Do Phase 1: restructure into the ctrfit package
  layout, with no change in behavior. Keep compatibility shims for the existing scripts.
  All existing tests must pass unchanged. Show the test output.
  ```

### Phase 2: Parameter model
- **Build:**
  - a `Parameter` object (name, value, min, max, fit flag, unit, description);
  - a `ParameterSet` with linked expressions and constraints (occupancies on one site sum to
    at most 1, bounds);
  - a template that turns today's settings into parameters.
- **Discrete film thickness:** the trilayer count is an integer, which optimizers handle badly.
  Represent thickness as a weighted distribution over integer trilayer counts (the existing
  thickness spread) with a continuous mean, and add a profile tool that scans integer N.
- **Done when:** round-trip tests pass (settings → parameters → model gives identical
  intensities), and links and constraints are tested.

### Phase 3: Fast evaluator
- **Why:** fitting needs 10^4 to 10^6 model evaluations, and today's `intensity()` rebuilds atom
  lists in Python on every call.
- **Build:**
  - "compile" a model once into arrays;
  - evaluate all data points of all rods in one vectorized call;
  - cache the parts that don't depend on the free parameters (the substrate rod, the buried
    film when only surface parameters are free);
  - add Numba (or JAX, if gradients become useful) only after profiling.
- **Target:** under about 5 ms per evaluation for about 2000 data points with surface-only free
  parameters on an ordinary CPU.
- **Done when:** results equal the reference engine to rtol 1e-9, and a benchmark test records
  the speed.

### Phase 4: Data layer and fitting core (first usable fitting)
- **Data:** import structure factors with errors (CSV, and GenX/ANA-ROD-style text). Group points
  by rod. Add a systematic error floor in quadrature (estimated from symmetry equivalents).
  Generate synthetic data with noise and systematic errors.
- **Figures of merit:** chi2 with sigma, a log figure of merit (GenX-style), and R1.
- **Optimizers:**
  - differential evolution (global search);
  - least squares (local refinement, with covariance-based errors);
  - a fit history log.
- **Scale factors:** one global scale, plus optional per-rod scales with a penalty.
- **API:** `fit = Fit(model, datasets); fit.run("de"); fit.refine(); fit.report()`
- **Done when** these synthetic recovery tests pass:
  - simulate data with known parameters plus noise, fit from perturbed starting values, and
    recover each parameter within its reported 2σ;
  - also test a surface-only fit, a film-only fit and a deliberately degenerate case (OH/H2O
    mixture), which should come back with a warning rather than a confident wrong answer.

### Phase 5: Uncertainty and model comparison
- **Build:**
  - covariance errors;
  - bootstrap over data points;
  - MCMC (e.g. `emcee`) for posteriors;
  - a correlation matrix with warnings for strongly correlated pairs (|r| > 0.9) and for
    parameters whose posterior is close to the prior bounds;
  - AIC/BIC for comparing surface models (single species vs mixture vs extra water layer).
- **Done when:** on synthetic data, MCMC intervals cover the true values at the expected rate
  over repeated noise realizations (a small coverage test).

### Phase 6: Fitting GUI
- **Fitting mode:**
  - load data and choose a template;
  - parameter table (fit checkboxes, bounds, links, units, descriptions);
  - data vs model per rod with residuals;
  - live figure-of-merit trace, run, stop and refine controls;
  - correlation view and an uncertainty report;
  - a project file.
- **Keep the simulation mode** (the current tabs and the map-first view).
- **Guided workflow panel:** step 1 film on even rods, step 2 surface on odd rods, step 3 global
  fit across datasets.
- **Done when:** offscreen GUI tests drive a full synthetic fit end to end, and you have checked
  the look and feel locally.

### Phase 7: Global fits across potentials and thicknesses
- **Build:** several datasets in one project. Each parameter is either shared (film structure,
  B factors) or per dataset (surface species, heights). Add potential-series plots of
  per-dataset parameters with error bars.
- **Done when:** a synthetic potential series with a known OH/H2O trend is recovered, and the
  shared parameters come out tighter than in single-dataset fits.

### Phase 8: Generality and cross-checks
- **Build:**
  - a template system for other film/substrate/surface combinations and CIF import for bulk
    structures;
  - a second built-in template (Au(111), including a reconstructed-surface option later);
  - a cross-check against GenX for one shared model and dataset;
  - a reproduction of a published RuO2(110) fit from its tabulated coordinates.
- **Done when:** both templates pass the physics checks, and the GenX comparison agrees to within
  convention differences that are documented.

### Phase 9: Polish and release
- **Build:** a user guide with a tutorial project, example notebooks, a packaged install
  (`pip install -e .`, optionally a standalone build) and a versioned changelog.

---

## 5. Order, parallel sessions and credits

| Order | Phases | Can run in parallel with |
|---|---|---|
| 1 | 0 | viewer water fix (touches only `ctr_viewer3d.py`) |
| 2 | 1 | nothing (it moves files) |
| 3 | 2, then 3 | data I/O and synthetic data (Phase 4 data part) |
| 4 | 4 | uncertainty prototypes in a separate module (Phase 5) |
| 5 | 6, 7 | Au(111) template (Phase 8) |

- **Best value per credit:** Phases 0 to 5, the restructure, fast evaluator, fitting core and
  tests. They are hard to do by hand and easy to verify with tests. GUI polish (Phase 6 onward) can
  continue on normal plan usage after the credit expires (November 4).
- **Keep sessions small:** one phase, or one sub-step of a large phase, per session, with a clear
  "done when". Large vague sessions burn credit and produce diffs that are hard to review.
- **Avoid conflicts:** never run two sessions that edit the same files at once. Merge, pull, then
  start the next.

---

## 6. Physics and statistics issues to keep in view

- **Identifiability:** a mixture of OH and H2O at two heights looks like one species at an
  intermediate height with extra disorder. The tool should warn and suggest constraints (DFT
  heights, potential-series sharing) instead of reporting a confident ratio.
- **Correlated parameters:** occupancy, B factor and height on the same site, and scale factor
  against roughness. Report correlations with every fit.
- **Error model:** counting errors alone underestimate the real scatter. Use the error floor from
  symmetry equivalents, or the chi2 will mislead.
- **Relaxed films:** H = 0 rods carry the whole film coherently. On H ≠ 0 rods the relaxed part
  is a separate, incoherent contribution with its own width. Fits of films thicker than about 4 nm
  must use the relaxation model, or restrict the surface fit to H = 0 rods.
- **Overfitting:** many surface parameters against few odd-rod points. Use AIC/BIC and the
  guided workflow (film first, surface second) to keep the parameter count honest.
- **Validation before real data:** every new model feature needs a synthetic recovery test
  before it is used on measured data.

---

## 7. Checklist per PR

- [ ] Tests pass (Phase 0 regression tests unchanged unless the PR says why)
- [ ] New code has tests, including a synthetic recovery test for any new fit parameter
- [ ] No Qt imports outside `app/`
- [ ] User-visible parameters have units and descriptions
- [ ] `CLAUDE.md` and this plan updated if the architecture changed
- [ ] Checked locally: GUI starts, 3D viewer runs (cloud has no display or GPU)

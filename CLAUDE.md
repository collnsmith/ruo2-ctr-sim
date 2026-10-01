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
- Film and commensurate thickness are whole (110) trilayers.

## Layout and rules

```
ctrfit/core      physics: sf.py (form factors, anomalous terms, atom-list structure factor),
                 lattice.py, structures.py (trilayer, adsorbates), electrolyte.py,
                 ctrmodel.py (the reference CTRModel), settings.py (GUI settings schema and parsers)
ctrfit/model     parameters, templates and the fast evaluator
ctrfit/data      datasets, import/export, synthetic data
ctrfit/fit       figures of merit, optimizers, uncertainty
ctrfit/project   results, reports and all plotting (plots.py)
ctr_engine.py, ctr_params.py, ctr_plots.py   compatibility names: each *is* the ctrfit module
ctr_gui.py, ctr_notebook.py, ctr_viewer3d.py front ends (unchanged)
```

- All physics lives in `ctrfit/core`. All plotting lives in `ctrfit/project/plots.py`.
- `ctr_gui.py`, `ctr_notebook.py` and `ctr_viewer3d.py` only call the engine and the plots. They
  hold no physics of their own.
- Settings, their units, defaults and the text parsers live in `ctrfit/core/settings.py`
  (imported as `ctr_params` by the front ends).
- No Qt imports anywhere in `ctrfit/` except a future `ctrfit/app/` (a test enforces this).
- Never edit `tests/reference.npz`. If a change is meant to alter the physics, regenerate it with
  `python tests/make_reference.py` and explain why in the commit message.
- Fixed seeds everywhere (synthetic data, optimizers, the viewer's random scene).

## Tests

```
pip install -r requirements.txt pytest
pip install -e .               # optional: makes ctrfit importable from anywhere
python -m pytest              # default suite, about 35 s; skips tests marked slow
python -m pytest -m slow      # long tests (synthetic recovery, coverage)
```

- The cloud environment has no GPU and no display. GUI tests run with `QT_QPA_PLATFORM=offscreen`
  (set in `tests/conftest.py`), plots use the Agg backend.
- `tests/conftest.py` blocks `xraydb`, so "auto" anomalous terms always come from the built-in
  table and the pinned numbers are machine independent.
- Viewer tests: the CPU renderer is compared with the CUDA kernel run in Numba's CUDA simulator,
  and the kernel is compiled to PTX for sm_75 (fake device) to check it is float32 only. They skip
  cleanly without `numba` / `numba-cuda`. In the cloud, `numba-cuda` also needs the CUDA runtime
  library: `pip install nvidia-cuda-runtime-cu12`.

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

## Checklist per PR (docs/FITTING_PLAN.md section 7)

- [ ] Tests pass (Phase 0 regression tests unchanged unless the PR says why)
- [ ] New code has tests, including a synthetic recovery test for any new fit parameter
- [ ] No Qt imports outside `app/`
- [ ] User-visible parameters have units and descriptions
- [ ] `CLAUDE.md` and the plan updated if the architecture changed
- [ ] Checked locally: GUI starts, 3D viewer runs (cloud has no display or GPU)

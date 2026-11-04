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
- Right: Summary, Structure, Rods, OH vs H2O, Sensitivity map, Ranking, Parameter study, Thickness study, Log.
- Run with the button, Ctrl+R or F5. The same button stops a run.
  A yellow note appears when settings changed since the last run.
- Sensitivity map: hover a rod for its values, click it to open its curves in the Ranking tab.
- OH vs H2O: click an L position to predict the stationary CV there (Stationary CV tab): intensity
  change vs time and vs potential (anodic / cathodic) with the CUS fractions, for the path, CV
  program and near-surface change in the sidebar group "Stationary CV prediction" (coverage: "CUS
  coverage for comparison"). Any H K L can also be typed in that tab.
- Parameter study: a table of settings to vary (roughness, Ru_cus shifts, adsorbate heights, B factors,
  strain, electrolyte, ...), each with a min, max and number of steps. "Run parameter study" varies one
  setting at a time, the rest as in the sidebar, on the sidebar's rods and surface state. The ranking
  below orders the settings by the largest change, (max − min) / mean of |F|² over the values, away
  from Bragg peaks, with the rod and L where it happens and the median change. Select a row to see the
  rod curves coloured by value and the change along L; pick another rod above the plot. The rows are
  saved in settings.ini.
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
fit.refine(starts=3, hops=1)                   # least squares from 3 DE points and the neighbouring
                                               # thickness fringes; errors from the Jacobian
res = fit.report()
print(res.summary)                             # values, errors, correlations, warnings
res.save("result.json")
```

Fitting window: `python ctr_fit_gui.py [data.csv] [model.json]` (or `python -m ctrfit gui`). Left:
the fitted datasets (with the systematic error floor), the scans of the loaded files, and the
parameter table (tick Fit, edit value, bounds and links; errors appear after a fit, parameters named
in warnings are highlighted). The scan table lists every scan of every loaded file (scan, rod,
points, L range, film, condition, file, group; the scan library's export and Processing write each
point's scan, a file without a scan column counts one scan per rod). Untick Fit to leave a scan out
of every fit (it is still drawn, hollow and grey, on its rod). Scans with the same Group are fitted
together as one dataset (one scale, one copy of the Per dataset parameters): type a group in the
Group cell, or select rows and use Group…; Own file puts them back. By default a file is one group;
a group can take scans from several files (e.g. repeats of one potential) or split a file. A group
that mixes potentials or films is allowed and noted in the log (one surface structure for all of
it; a mixed film is a film of its own). A group keeps its parameter copies, scale and floor while
its name stays; a new group starts from the copies of the dataset its scans came from. Clicking a
dataset selects its scans; clicking a scan shows its rod, one colour per scan. The project file
keeps the files and the choices. Many rows at once: Fit all / Leave all out and Fit selected / Leave
out selected for the scans; Fit shown / Fix shown (the rows the Show filter lists, e.g. Show
surface, then Fit shown) and Fit selected / Fix selected for the parameters; Space toggles Fit of
the selected rows in either table. The parameter table lists each shared parameter followed by its
copies (`↳ ds1`, ... under `thickness_mean_tl`), collapsed by default: click ▸ / ▾ in the Name cell
(or double-click the name, or press → / ←) to show or hide them, "▾ all" / "▸ all" for every
parameter; a warning on a hidden copy colours its parameter's row. The Dataset column says which dataset a copy belongs to
(and which copy a per-film one follows), "shared" for one value for all, or "per film: 2 films, 3
datasets" / "per dataset: 3" for a parameter that every dataset replaces with its own copy. Such a
parameter is a master row (bold, its Value shows the copies' range): its Fit box fits every copy that
is not linked (one per film for per-film parameters), its Min / Max set all their bounds; values are
edited in the copies' rows. Show "selected dataset" lists only what the dataset picked in the list
uses (its copies and the shared parameters). "Colour by selected parameter" (above the
rod plot, on by default; untick for the plain red curve): the model curve is coloured by how much
|F| changes when the parameter selected in the table moves by ±1σ (its fit error; before a fit, a
small step: 2 % of the value, at most 1 % of its range), red for no change to blue for the largest
change over all rods of the dataset, so a rod the parameter hardly touches stays red; the colour bar
gives the change in %. It shows where the data constrain a parameter (e.g. x_OH on the odd rods,
the thickness around the fringes). A linked parameter, or one that this dataset replaces with its
own copy, leaves the curve red and says why in the status bar. Right: fit settings,
Run / Stop (Solve scales exactly, Refine starts and Fringe hops: see "Less starting guess" below; during a fit the model curve of the shown rod and the Value column follow the best
parameters after every differential-evolution generation and as least squares improves; the rod
plot keeps the data's axes meanwhile and the residuals stay within ±10 σ, so the view does not jump
between trials, and progress messages never resize the window), and the guided workflow (1. film on the even rods with a thickness scan, 2. surface on
the odd rods, 3. everything together). Tabs: data vs model with residuals per rod, live figure of
merit, report with warnings, correlation matrix, per-dataset series for global fits. File menu:
models, data, project files (`*.ctrproj.json`: model, data and results together) and a report folder.

Fitting assistant (command line only; the fitting window has no Claude panel): `python -m ctrfit
ask model.json data.csv -r "fit the film first, then tell me whether x_OH is determined"`. Claude
runs the fit with the fitting tools (state, set values/bounds/fit flags, links, per-dataset copies,
guided steps, DE, refine, thickness profile, report, per-rod residuals, AIC/BIC model comparison)
and writes the result. Needs `pip install anthropic` and an API key (`ANTHROPIC_API_KEY`, or `ant
auth login`).

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
  Run `fit.profile("thickness_mean_tl", grid)` after the global search to pick the right one, or
  `fit.refine(hops=1)` (below).
- In situ cell and surface (all off by default: 0 and not fitted, so the model is unchanged until you use them):
  - `t_liquid` and `t_window` (µm, group instrument): absorption of both beams in the electrolyte
    film (0.1 M H2SO4) and the window (Kapton; 0.0003 inch = 7.6 µm) above the surface. Each point's
    |F| is multiplied by exp(-(mu_l t_l + mu_w t_w)(1/sin alpha + 1/sin beta)/2), with the exit angle
    beta from H K L, the energy and the incidence angle (setting "Incidence angle", default 0.5°, the
    same as Processing's). It lowers low L most (a few % in |F| below L = 0.5 for a few µm at 15-20 keV,
    more for a thicker film); refraction near the critical angle is not included, so leave out points
    with an exit angle below about twice the critical angle (L below about 0.15). Set t_window to your
    window's thickness (fixed) and fit t_liquid, per dataset since the film is set anew by the cell.
  - Model menu, "Add an L offset per scan (rod) in each dataset": `dL_H_K` (r.l.u., ±0.05, fixed at 0
    until ticked) per rod of each dataset, one per scan with one scan per rod: the model is taken at
    L + dL, for a scan whose Bragg peak sits off its nominal L. Fit them only where the substrate peak
    region is in the data (they trade against eps_perp otherwise). "Add a scale per rod in each dataset"
    adds `rodscale_H_K` the same way.
  - Adsorbed (bi)sulfate: `x_SO4` (share of the covered CUS sites), `z_SO4` (height of its bonding O
    above the CUS Ru), `B_SO4`, `dz_Rucus_SO4`; a monodentate SO4 standing on the CUS Ru, its upper three
    O averaged around the S-O axis. `SO4=0.2` in a composition (simulator, settings) too. Test it as a
    model variant against the OH / H2O one by BIC.
- Per dataset (default): `theta, x_OH, x_O, x_SO4, ns_B_O, ns_du, dz_Obr, t_liquid` (what the
  potential, or the cell, changes); per film (default): `thickness_mean_tl, thickness_spread_tl,
  roughness_tl, eps_perp, relax_001, coherent_tl, b_top_extra, interface_A, b_film_Ru, b_film_O` (the
  sample). Only the ticked ones get copies; everything else free is shared, e.g. the species heights
  and B factors (properties of a species, not of the potential), the substrate and the window.
- Less starting guess. The global search only needs bounds that bracket the answer:
  - The dataset scales are solved exactly for every trial of differential evolution (on by default,
    `run_de(solve_scales=False)` / "Solve scales exactly" in the window to search them instead).
    |F| is proportional to the scale, so the best value has a closed form (chi2: weighted least
    squares; log: the median log ratio; R1: a weighted median). A poor starting scale then no longer
    makes every trial structure look bad, and the search has one dimension less per dataset.
  - `fit.refine(starts=3)` runs least squares from the 3 best distinct points the search evaluated
    and keeps the lowest chi2 ("Refine starts" in the window, default 3).
  - `fit.refine(hops=1)` also refines from the film thickness one trilayer up and down (each film's
    thickness in a global fit): the neighbouring fringe minima, which the search often confuses
    ("Fringe hops" in the window, default 1; the guided steps and the command line, `--starts`,
    `--hops`, use them too). On synthetic data over wide bounds (thickness 5 to 40 TL, scale 0.05 to
    20, start at 30 TL and scale 0.2 for a truth of 19.4 TL and 2.7), DE plus a single refinement
    found the right fringe for 2 of 6 seeds; with the scales solved and the hops, 6 of 6.
  - Cores (window, default half the logical CPUs; `Fit(..., cores=4)`; `--cores 4` on the command
    line): differential evolution evaluates each generation's trials, and least squares its Jacobian
    columns, on that many worker processes (numpy single-threaded in each; threads do not help). The
    workers start once, about a second, and are kept for later fits. About 3x faster DE on 4 cores;
    least squares gains less (short refinements). With several cores DE updates once per generation,
    so its path differs from one core's, but it is reproducible for a seed; least squares gives the
    same result on any number of cores.
  - Sample L finely enough for the thickness: with an L step dL, N and N + 2/(2 dL) = N + 1/dL
    trilayers give the same fringes at the measured points (dL = 0.1 cannot tell 19 from 39 TL), so
    keep the thickness bounds narrower than 1/dL or measure on a finer grid.
- Beyond the covariance errors: `ctrfit.fit.bootstrap(fit, n=100)`, `ctrfit.fit.mcmc(fit)` (needs
  `emcee`), and `ctrfit.fit.compare_models([res_a, res_b], labels)` for AIC / BIC between surface
  models (e.g. OH only vs OH + H2O vs an extra water layer).
- Global fits across potentials and films: one dataset per condition (a film at one potential, or
  at OCV), with `potential_V`, `film` and `thickness_nm` in its metadata (the scan library's "Export
  for fitting…" writes them). `global_fit(model, datasets, per_dataset=["theta", "x_OH", "z_OH"],
  per_film=["thickness_mean_tl", "roughness_tl", "eps_perp"])` (from `ctrfit.fit.global_fit`) gives
  each dataset its own copy of what the potential changes (the CUS coverage and species, their
  heights, the near-surface O) and its own scale, each film one value of the per-film parameters
  (shared by all potentials of that film; datasets of the same film name and thickness are one
  film), and shares everything else (substrate, B factors). Each film's thickness starts at its
  datasets' `thickness_nm`, converted to trilayers with the film's spacing. `series(result, "x_OH")`
  returns the values against `potential_V`. Command line: `--per-dataset x_OH,z_OH --per-film
  thickness_mean_tl`. In the window: "Add data…" takes several files at once, the list shows each
  dataset's film and potential, and "Per dataset" / "Per film" set the two lists.
- Worked example with synthetic data: `examples/synthetic_ruo2/` (`run_fit.py`: film on the even
  rods, then the surface on the odd rods, then everything together).

## Near-surface oxygen

The "Near-surface oxygen" settings (simulator sidebar, fitting model, in-situ CV window) change the
lattice oxygen of the top `ns_tl` trilayers of the film (default 3, counted down from the film top,
including the exposed layer): an extra B factor (`ns_B_O`, Å²) and a change of the rutile u
parameter (`ns_du`). It models a change of the oxide below the surface, e.g. proton insertion
(RuO2 + δH⁺ + δe⁻ ⇌ RuO2−δ(OH)δ). On the H + K odd rods the Bragg peaks are oxygen-only reflections
(the two Ru of the cell cancel), so this lowers both flanks of those peaks while the H + K even rods
barely change; a change of the CUS species instead flips sign across a Bragg peak. Both values are
fit parameters (`ns_B_O`, `ns_du`); `ns_tl` is a setting. With both at 0 the model is unchanged.

## In-situ CV analysis

`python ctr_echem.py` (or `python -m ctrfit echem`) analyses a stationary SPEC scan, either a loopscan
or a phi scan that does not move (start = end), recorded while the potentiostat holds a potential
(relaxation) and then runs a CV. The computation is in `ctrfit/echem` and works from scripts too.

- **Files.** The SPEC file (pick the scan; only stationary scans are listed unless you untick it), the
  counter, an optional monitor, per-second normalisation and the time column: `Epoch` (seconds since
  `#E`, phi scans) or `Time` (seconds since the scan's `#D`, loopscans). The HKL comes from `#Q` or is
  typed in (H and K are rounded to the rod). An attenuator transmission column (`transm`) divides
  the intensity, keeping the counting errors right. Scans copied out of a SPEC file without its
  header (no `#E`) are fine: epoch times are then counted from the scan's `#D`. The potential comes
  from an EC-Lab `.mpr` (needs `pip install galvani`), an EC-Lab `.mpt` text export or a CSV, **or a
  CV defined by hand**: hold potential and time, lower and upper vertex, scan rate, number of
  cycles (or "until stopped"), first sweep direction, and either the delay after the scan start or
  the point numbers from your log ("CV start at point # 73", saying whether that point is the start
  of the hold or of the sweep, and "CV end at point # 433").
- **Time sync**, three modes. *Absolute clocks + offset*: both files' own clocks; the offset
  corrects a difference between the two computers (EC time = SPEC time + offset). *Manual offset*:
  times from each file's start; type the delay. *Align by events*: the intensity onset after the
  relaxation is put on the detected start of the sweep (both can be typed in). "Offset from events"
  fills the offset of the first two modes the same way. Events are biased when the intensity only
  starts to change some time into the CV (in the example by about 24 s, i.e. 0.24 V at 10 mV/s), so
  prefer the clocks when they exist. "Shift that closes the loop" finds the time shift that makes
  the anodic and cathodic I(V) agree; it tests a timing error but a real (kinetic) hysteresis is
  closed by it too. Each point gets the mean potential over its counting window, the sweep
  direction and the cycle number, and the summary compares how often the intensity and the
  potential repeat (a mismatch points to a wrong scan rate or vertices).
- **Background.** None, constant, linear or exponential fitted to the relaxation period, removed by
  division (beam or sample decay) or subtraction. A warning says when it is extrapolated far.
- **I vs V.** Points by cycle (filled anodic, open cathodic) and, optionally, cycles averaged in
  potential bins per sweep direction with error bars.
- **Transition fit.** One to four sigmoidal steps per sweep direction: E0, width (with the apparent
  electron number RT/(F·width)), step height, and the hysteresis between the directions.
- **CTR model.** The predicted intensity at the HKL for a composition path vs potential (e.g.
  `H2O -> OH -> O`, one `E0, width` per change, coverage, cathodic shift), from the simulator's
  structure (defaults or a settings `.ini`). The relaxation state sets the intensity scale.
  "Fit path to data" fits the transitions, scale and cathodic shift (optionally the coverage)
  through the CTR model. "Coverage from data" turns each intensity into a composition on the path;
  it flags points that match more than one composition or none. A near-surface oxygen change
  (extra B, Δu at full extent; switching on with the first transition or at its own potential and
  width; affected trilayers) can be added to the prediction, fitted with the path ("fit the
  near-surface O disorder / u change"), and is taken into account in "Coverage from data".
- **Joint fit of several scans.** Analyse each stationary scan (its own file, timing and HKL), press
  "Add current scan" on the Joint fit tab, then "Fit jointly". All scans share the composition
  path (E0 and width per transition, cathodic shift, coverage), the near-surface oxygen change (extra
  B, switching with the first transition or at its own E0 / width) and any structure parameters you
  add (any numeric simulator setting: `oh_z`, `oh_dz`, `h2o_z`, `h2o_dz`, B factors, top-layer
  relaxations, ...); each scan has its own intensity scale. "Profile ns_tl" fits once per
  near-surface depth. Film parameters (thickness, strain, roughness) stay fixed from the settings:
  stationary points cannot determine them, fit them on rods. The set saves to JSON and fits from the
  command line: `python -m ctrfit cv-joint set.json --free ns_dB,oh_z --profile 1,2,3,4`.
- **Output.** The aligned data as CSV, the results as JSON, and the session (files and settings) as JSON.
  Command line: `python -m ctrfit cv scan.spec --ec cv.mpt --offset 7.3 --transitions 2 --fit-path`
  (or `--cv 0.5,300,0.4,1.4,10,3,up,60` instead of `--ec`).
- **Example.** `python examples/insitu_cv/make_data.py` writes a synthetic loopscan, an unmoving phi
  scan and an `.mpt` with a known truth (`truth.txt`); with offset +7.3 s the path fit recovers it.


Intensity: a SPEC counter, or a file (Intensity: "File"): ROI sums from the background subtraction
tool (one row per image, the image number being the SPEC point) or a trace from the plotter
("Export data", "Export t, E, I", the smoothing's "Export smoothed data"); pick the ROI or trace (blank:
the first ROI / this scan's trace). Missing images are interpolated; a trace with another number of
points (e.g. a spline) is interpolated on its x as seconds from the scan's first point. Monitor,
transmission and count time still apply: switch them off for a trace that is already normalised
(e.g. the processing tool's `_stationary.dat`). A `<ROI>_err` column (background subtraction, processing
tool) gives the error bars instead of sqrt(N).
Command line: `--intensity-file sums.txt --intensity-column ROI1`.
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
  with L along it. To fix the choice yourself, tag peaks with the HKL you want them called:
  `del eta chi phi nu mu [H K L]`. Tag as many as you like; two tagged peaks with non-parallel Q fix
  the indexing completely, so adding peaks no longer changes it (and you don't have to redo the
  orientation in SPEC). A tag that cannot be met (wrong |Q|, forbidden, not a symmetry equivalent,
  or in conflict with an earlier tag) is reported with the equivalent used instead and the nearest
  one. Tags win over the surface normal. Command line:
  `python -m ctrfit index peaks.txt --energy 16 --normal "0 0 1"`.
- **UB refinement (reflex).** Like SPEC's reflex file: keep adding reflections as you find them
  (HKL plus the six angles, each at its own energy; leave HKL blank and it is guessed from the
  current UB) and UB is refined by least squares on all of them after every addition. Choose what
  is refined: the orientation only, a common lattice scale, a, b, c, all six lattice parameters, or
  all nine UB elements; motor zero offsets (e.g. `del`) can be refined too. The result gives the
  lattice with standard errors, the angle between measured and calculated Q for every reflection,
  outliers (usually a wrong HKL: untick "use" to leave one out), warnings about undetermined or
  correlated parameters, and a `setlat` line. The list saves to JSON or text, exports as a SPEC
  reflex file ("Export reflex file…": a date line, `_begUB`, `H = ..;  K = ..;  L = ..` and
  `A[del]= ..;` blocks with `_addUB`, then `_fitUB`; only the reflections in use, no energy) and
  loads one back at the energy set at the top, the indexed peaks can be added in one click, and the refined UB goes to the angle calculator. Lattice and offsets need a
  spread of HKL directions and 2theta; with few reflections refine the orientation only. Command
  line: `python -m ctrfit refine reflex.txt --energy 16 --free abc --offsets del`.
- **Angles.** UB from the indexer, the refinement or from two reflections (like `or0` / `or1`). Angles for an HKL in a
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

## SXRD data tools

`python ctr_data_tools.py` (or "SXRD data tools" in the launcher) opens the data processing tools
in one window, one tab per step of the chain (drag a tab to move it; the order and the open tab are
kept, right click on the tabs to reset the order). Every tab is laid out like the other ctrfit windows:
settings in titled groups on the left, the main action as the blue button at the bottom (Ctrl+R or
Ctrl+S where it applies), plots and images on the right, messages in the status bar (errors also
in a dialog):

- **Metadata.** HDF5 structure and metadata per scan of an image folder; SPEC columns per scan.
- **Background subtraction.** Eiger 500K `.h5` (h5py) or `.tif` (tifffile) images named
  `<prefix>_<scan>_<image>_data_<n>`: signal and background ROIs, per-image overrides, ROIs that
  follow a master ROI or track the peak, average / median / polynomial background, a pixel mask,
  batch ROI sums with a counting error per signal ROI (`<ROI>_err`: sigma^2 = S + (n_s/n_b)^2 B,
  times pi/2 for the median background; not sqrt(I)) and, optionally, the subtracted images as PNGs
  ("Also save the subtracted images"). The sums file ends with a "# [ROI positions]" section: scan,
  image, ROI, type, x, y, w, h (top-left pixel and size) of every ROI as used, one row per scan
  ("all") for a ROI that stays put and one per image for a ROI that moves; its lines start with '#',
  so the plotter, the processing tool and the CV tool read the table as before
  (`ctrfit.datatools.images.read_roi_positions` reads the section).
  "Find background ROIs" opens the ROI finder on the same folder and scan range and merges its
  background boxes. Ctrl+Z / Ctrl+Y undo and redo ROI and mask edits (drags, fields, add/remove,
  copies to other images, mask painting, loaded ROI files and masks). Images are kept in memory and
  the next ones read ahead in the background, so a slow or busy disk (network drive, a transfer
  running) does not stall stepping through a scan; while a ROI is dragged or the mask painted the
  preview updates at most every 40 ms. Manual: `docs/data_tools_bgsub_manual.txt`.
- **Background ROI finder.** Per-scan pixel statistics (on the GPU with CuPy if installed), a mask of
  persistent bright spots (powder rings, hot pixels), background boxes around a clicked signal centre.
- **Plotter.** SPEC scans with the ROI sums, figure options, peak fits of rocking curves (Gaussian,
  Lorentzian, pseudo-Voigt with a linear background; fitted or numerical areas), smoothing (of every
  plotted trace at once: "Grab from main plot" takes them all, "Send" puts them all back), per-scan
  Y columns, exports. Several Norm boxes can be ticked: the trace is divided by their product
  (e.g. monitor × transmission). "Refresh files" (F5) reads the SPEC file, the ROI sums folder and the EC-Lab
  file again and keeps the selection. "CTR fitting" switches to the CTR fit tab. "Potential and time": "t = 0 at" the
  start of each scan (every selected scan from 0, e.g. scans with their own CV) or of the first
  scan (one clock for all); the small table sets, per scan, the point that is t = 0 ("t = 0 at
  point #", 0-based, e.g. where that scan's CV started) and a shift in seconds, to line scans up; add the potential from an
  EC-Lab file (.mpr with galvani, .mpt / .txt export, .csv) or a sweep defined by hand (start and
  hold, lower and upper vertex, rate in mV/s, cycles, first direction); "EC start at t =" puts the
  potentiostat's first sample on the time axis to line the two records up, or "At point #N, Align"
  sets it from a SPEC point number (0-based) of the scan selected in the shift table: the first sweep (after any hold,
  found from the potential) or the start of the potentiostat record happens at that point. Show the potential on a
  second y axis against time ("Potential trace" hides or shows that curve), or plot intensity against potential; "Export t, E, I" writes both.
  "Export figure…" opens the final image in its own window, the working view staying as it is:
  a preview with the proportions of the saved file, size (in or cm) and DPI with presets (paper one
  or two columns, slide, poster, dark), format (PNG, SVG, PDF, EPS, TIFF, JPG), title and axis
  labels, fonts and sizes, figure / axes / text colours, limits and log scales, ticks, frame, grid,
  legend (place, frame, outside), and a table of the traces (show, label, colour, line, width,
  marker, size, left or right axis). "Refresh from plot" takes the working plot again and keeps the
  edits; the last style is reused for the next figure; "Copy image" puts it on the clipboard.
  "View options…" (the old figure options) only changes the working view.
- **Scan library.** Sorts the scans for comparisons, and writes the datasets for the Fitting app. "Add processed file…" takes a Processing master
  file (`<name>_master.txt`): every processed scan, normalised and corrected, with no SPEC file or
  ROI sums needed (picking another file of the same save, e.g. `<name>.csv`, which has no scan
  numbers, reads its master file); each scan shows the ROI it was processed with (picked in
  Processing, saved in the master file; read-only here, change it in Processing, save and Refresh),
  and "Quantity" in Compare plots I_corr (normalised and corrected, the default), |F| or I_norm,
  against L or the time from the scan / CV start, and scans of several master files are put on one monitor
  reference (Processing normalises each run to its own mean monitor). "Add SPEC + ROI sums…" takes
  a SPEC file and its ROI sums: a folder (every ROI sums `.txt` file in it, one scan or several per
  file; other `.txt` files are skipped; the folder is listed again on Refresh, so sums written later
  appear) or files picked one by one; several SPEC files can be added (e.g. one per beamtime). A scan
  in several sums files (a re-run) gets the newest file's ROIs under their names, the older ones as
  `<ROI> (<file>)`. Every scan is listed with its kind from the SPEC scan: rod (L moves), stationary (H K L fixed,
  from the columns or `#Q` for time scans), other. Assign to the selected rows: the film, the
  measurement (CV, optionally with the SPEC point where the CV starts; CA at a potential, in the
  library's unit, e.g. V vs RHE; or OCV, open circuit), the ROI, notes, use / skip; or type in the
  table (a film, potential, CV start or note typed with several rows selected sets them all; a
  potential makes the scan a CA, a start point a CV; Ctrl+Z / Ctrl+Y undo and redo). The ROI box
  of each row picks the ROI that scan is shown with (its tooltip gives the ROI's position from the
  sums file); a ROI name that two sums files of one SPEC file share becomes `<name> (<file>)`.
  "Save library" (Ctrl+S) writes the library: a tab-separated text file (settings, sources with
  paths relative to the library, one row per scan: film, mode, potential, ROI, use, then kind,
  H K L, L range, points, SPEC command and notes), readable and editable by hand. The data files are
  only read: the library refuses to be saved over one of them, and a data file that changed since
  the library read it is reported. A library with a file is also saved when the window closes; one
  that has none yet is kept as a draft and comes back the next time. An assignment file (a library
  whose source has no files yet, e.g. written from the beamtime's scan notes) opens like any library:
  "Add SPEC + ROI sums…" fills its source and the assignments land on the scans by number (scans not
  processed yet stay listed as missing until a refresh finds them). "Import assignments…" copies the
  assignments of such a file onto the selected SPEC file of a library you already have. "Compare" plots groups:
  "CVs at one H K L" (stationary scans during CVs at one point, against the time from each CV's
  start point, so the CVs line up, or from the scan start, the SPEC point or any SPEC column such as
  a potential counter) and "CAs on one rod" (rod scans held at one potential or at OCV, against L), or any mix of scans (rod or stationary), measurement,
  rod / point, potential and films, coloured by film, potential, both, rod or scan; scans of one
  group share a colour (a rod measured in two L ranges is one curve). "Equivalents ±H ±K (p2mm)"
  (default) puts symmetry-equivalent rods together, e.g. (0 1) and (0 -1) at the same L, each drawn
  with its own line style (the picked rod solid) so that disagreeing equivalents show. Intensities are the ROI divided by the
  monitor (relative to the mean monitor of the plotted scans), transmission and count time chosen
  under Intensity (not the geometric corrections: those are the same for one rod and L at a given
  energy and incidence angle, so comparisons on one rod are fair; use Processing for |F|); "÷ max"
  or "÷ first point" (I / I0) compare shapes, "Stack ×" spreads groups on a log scale, the error
  band is the ROI's `_err` with the systematic error (% of I, usually 5 to 10) added in quadrature. "Selected rows only" plots the table's selection. Each scan has a film thickness (nm: typed in the
  Thickness column or panel, else the number in the film name, e.g. "7 nm"). "Export for fitting…"
  writes one Fitting-app CSV per film, thickness and condition (CA potential or OCV) from the
  processed rod scans: |F| and sigma_F of the used points (on one monitor reference), each on its
  scan's rod, with `potential_V`, `film`, `thickness_nm`, `condition`, the energy and Processing's
  systematic floor; CV scans are not rods and stay out, raw ROI sums need processing first.
  "Export data…" writes
  the plotted traces (one `# Trace:` block per scan, readable by the in-situ CV tool), "Export
  figure…" opens the styled figure window. Logic in `ctrfit.datatools.library`.
- **Processing** (also "CTR processing" in the launcher, `python ctr_processing.py`). Steps 2 to 5
  after background subtraction, for stationary rods (one image per L point; no rocking scans): pick
  the SPEC file and the ROI sums folder or files; each row's image number is the SPEC point of its
  scan. Time scans (loopscan, timescan) without H K L columns take H K L from the scan's `#Q` line; a
  scan that cannot be processed (no H K L at all, a missing ROI or column) is left out with a note in
  the summary and the status bar, and the other scans go on. 2: normalise by monitor (relative to the mean monitor), transmission and count time.
  3: polarisation (horizontal fraction, horizontal or vertical surface), rod interception 1/sin(beta)
  and illuminated area (beam height and width, sample length); angles from H K L, energy and a fixed
  incidence angle (specular rod: alpha = beta) or from SPEC columns. 4: |F| and sigma_F from the
  propagated `_err` column (without it the counting error is a lower bound, flagged in Notes).
  5: every kept point is written as measured: nothing is merged or averaged, so overlapping scans
  of one rod and symmetry equivalents stay separate points (each with its scan and rod); points
  within a distance of integer L are left out and a systematic floor is written for the fit. A hand
  scale per scan is in the table; "Scale later scans of a rod to the first on their overlap" (off by
  default) multiplies a scan by its median ratio to the earlier scans of its rod where they overlap. The table lists the scans (Use, rod, L range, points, scale, ROI, notes; the ROI column picks
  the signal ROI of each scan when it changes between scans, "(default)" being the one under
  2. Normalise, or the scan's first ROI with errors if that one is not in the scan; picking a ROI
  under 2. Normalise applies it to every scan and resets the column (the status bar names the scans
  that had their own), and a ROI picked in the table after that overrides it for that scan (shown in
  bold); "Process and save…" processes again first, so the files match what is shown; per-scan scale,
  Use and ROI are remembered until another SPEC file is opened); clicking one
  shows before (ROI sum) and after (normalised and corrected, then |F| with the other scans of
  its rod in grey). The Group column labels each scan's condition (potential, film thickness, ...):
  overlap scaling, if on, stays within a group, so scans of one rod at different potentials or on
  different films are never scaled onto each other; each group is saved as its own `<name>_<group>.csv`
  and `_genx.dat` (load them as separate datasets in the Fitting app, e.g. for a global fit). Typing
  a group with several rows selected sets it for all of them. Groups compare on the same absolute
  scale only through the normalisation, so keep the same monitor, transmission and count-time
  columns for every scan.
  "Process and save…" writes `<name>_master.txt`, `<name>.csv` (for the Fitting app) and
  `<name>_genx.dat`. The master file holds every processed scan (rod and stationary) and every
  point: '#' lines say what was processed and how and list the scans (kind, rod, L range, ROI, scale,
  monitor reference, group, SPEC command, notes), then one tab-separated table has H K L, t (s from
  the scan's first point), the raw, normalised and corrected intensities with errors, every
  correction, F and sigma_F and `used` (0: left out, e.g. near a Bragg peak); the scan library opens
  it with "Add processed file…" (pandas: `read_csv(path, sep="\t", comment="#")` reads the table). Stationary scans (H K L fixed, e.g. during a CV) are found
  automatically ("stationary" in the Rod column): steps 2 to 4 point by point, the preview against
  SPEC point, no overlap scaling or Bragg window; they go to `<name>_stationary.dat`, a ROI
  sums style file that the in-situ CV tool opens as its intensity file (pick `<ROI>_corr`, and
  switch off its monitor, transmission and per second, which are already applied). Logic in
  `ctrfit.datatools.processing`.
- **CTR fit.** Quick kinematic rods (substrate, film, adlayer, liquid, three layers) on integrated
  intensities, for a first look; the full RuO2/TiO2 model is the Fitting app.
- **HKL appender.** H, K, L from the SPEC scan added to plotter exports and peak-fit tables
  (constant H and K snapped), and GenX export.
- **Combiner.** Several HKL files into one `H K L I` table (drop NaN or non-positive rows, sort, dedupe).

The logic is in `ctrfit.datatools` (no Qt), the windows in `ctrfit.app.datatools`; one tool alone:
`python -m ctrfit.app.datatools.plotter` (or `bgsub`, `bg_finder`, `library`, `processing`, `ctr_fit`, `appender`,
`combiner`, `meta_viewer`). Settings are saved in `~/.ctrfit/data_tools` (or `$CTRFIT_SETTINGS_DIR`);
copy the old `.ini` files there to keep them.

Errors never close the data tools: an unexpected error is shown and written, with the time, to
`crash.log` in the same folder (hard crashes too, via faulthandler); please send that file with a
bug report. A plot scale that cannot be drawn (logit needs values strictly between 0 and 1, log
needs positive values) goes back to linear with a note in the status bar.

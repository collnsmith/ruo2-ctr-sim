# CTR fits for the CUS OH coverage, 4.1 and 7 nm films (5 Nov 2026)

`summary.pdf` is the one-page result (figures `fig1..3_*.png`, built by `make_summary.py`). Every number
in it comes from the files below.

| File | What it is |
|---|---|
| `both_films_autofit.ctrproj.json` | Both films from start to finish: 26 rod scans (no specular), a fresh model, film per film, surface per film and potential, a scale per scan (`autofit.py`; log and summary `both_films_autofit.*`) |
| `oh_rods_fit.ctrproj.json` | The OH fit: film pinned on the well-behaved rods ((0 ±1), (−1 0), (−2 0)), then x_OH on the 13 (0 ±1) scans with theta = 1 under three bridging-O assumptions (`oh_fit.py`; its model holds the last one, C; every variant's values and x_OH profiles are in `oh_rods_fit_summary.json`) |
| `7nm_uniform_film.ctrproj.json`, `7nm_surface_refit.ctrproj.json` | The 7 nm film alone: a uniform physical film (`fit7.py`), then the surface refit with physical bounds (`fit7e.py`); trimmed to their last result (the uploaded project's 125 earlier fits left out) |
| `oh_sens.py`, `oh_where.py`, `diag_proc.py` | Which scans carry OH information, where along (0 1) and at which CV points OH changes \|F\|, and the processing checks (intensity dependence, steps) |

Open a project in the fitting window (File → Open project). The scripts take a project file as their first
argument and write next to themselves, e.g. `python analysis/2026-11-05_oh_fits/oh_fit.py
analysis/2026-11-05_oh_fits/both_films_autofit.ctrproj.json`. The fits block xraydb (built-in anomalous
terms) like the tests.

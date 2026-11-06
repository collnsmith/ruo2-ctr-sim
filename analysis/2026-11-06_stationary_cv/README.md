# Stationary CVs (3.3, 4.1, 7 nm films), 2026-11-06

Data: `data/ROIs_s<scan>_sums.txt` (ROI sums per image) and `data/Stationary_Scans.txt` (HKL, Start / Stop point,
film, which ROI is Ru and which is Ti). Program: 30 s CA at 0.0 V from Start, then CV 0 to 0.7 V vs Ag/AgCl at
5 mV/s (280 s per cycle). Scan 222 changes to 25 mV/s at about point 257, after its Stop (250), so none of that is used.
No SPEC file: no monitor or filter normalisation, and the time comes from the point number.

## Steps

1. `extract.py`, `extract2.py`: period in points from the data (`ctrfit.echem.analysis.dominant_period`), snapped to whole
   or half cycles after the CA. The timing shift (within 0.15 period) is the one that correlates best with E, which fixes the sign.
   Flat scans: nominally 0.3 points/s. Output `cv_timing.json`.
2. `extract3.py`: Ru, Ti and Ru+Ti, each relative to the CA at 0 V, averaged over the cycles in 14 E bins per sweep.
   For the flat scans, the largest linear response any timing allows (period 65 to 125 points). Output `cv3.json`.
3. `parts.py`: model film(+surface) and substrate intensities at each point, from the OH-fit project
   (`../2026-11-05_oh_fits/oh_rods_fit.ctrproj.json`). The 3.3 nm film uses the 4.1 nm structure at 10.2 TL (approximate).
   Change for x_OH 0 -> 1, coherent and film-spot-only. Output `parts.json`.
4. `cvfit2.py` / `cvfit3.py`: per film, x_OH(E) as a tanh step, optionally with dz_Obr on the same step, plus an offset per scan,
   fitted to the curves. Scan 147 is left out because its timing is not consistent. Outputs `cvfit_*.json`:
   - `sum`: Ru+Ti against the coherent model.
   - `rucoh`: Ru against the coherent model.
   - `ru`: Ru against film+surface alone.
5. `ohfit_c2.py`: the OH fit's variant C with Rao's dz_Obr mapped from RHE to Ag/AgCl (+0.26 V in 0.1 M H2SO4):
   - -0.24 to 0.15 V: 0.35 Å
   - 0.35 V: 0.2 Å
   - 0.7 V: 0 Å
   - OCV: 0.2 Å (measured OCV 0.35 to 0.4 V Ag/AgCl = 0.61 to 0.66 V RHE, the same bin as 0.35 V)

   Output `ohfit_c2_summary.json`.

## Results

| Scan | Film | HKL | Ru | Ti | Ru+Ti | coherent model, x_OH 0->1 |
|---|---|---|---|---|---|---|
| 496, 497 | 4.1 | 0 -1 2.75 | -0.4, +0.3 | +0.4, +0.2 | -0.1, +0.3 (any timing <= 1.3) | -41 |
| 503 | 4.1 | 0 -1 2.84 | -0.9 | -0.6 | -0.8 (<= 1.3) | -34 |
| 498 | 4.1 | 0 -1 3.16 | -0.3 | -0.2 | -0.3 (<= 1.9) | +42 |
| 501 | 4.1 | 0 -1 3.25 | +1.3 | +1.7 | +1.5 (<= 2.8) | +57 |
| 120 | 3.3 | 0 -1 2.75 | -34.3 | +0.3 | -13.1 | -41 |
| 222 | 3.3 | 0 -1 3.16 | +15.5 | +1.2 | +12.7 | +42 |
| 125 | 3.3 | 0 -1 3.25 | -26.9 | -0.7 | -11.4 | +56 (wrong sign, see notes) |
| 311, 344 | 7 | 0 1 2.75 | -34.9, -44.0 | -3.1, -0.9 | -24.6, -38.0 | -40 |
| 314 | 7 | 0 1 3.16 | +32.7 | +1.8 | +18.8 | +58 |
| 316 | 7 | 0 1 3.25 | +18.9 | -0.2 | +11.5 (noisy, timing assumed) | +86 |
| 356 | 7 | 2 0 1.75 | +0.7 | +0.2 | +0.2 | 0 (control) |

Changes are linear 0 -> 0.7 V, in %.

The response is reversible, the anodic and cathodic sweeps nearly overlap, and most of it happens between 0 and 0.4 V.
The Ti ROI is flat in every scan except 344 (noisy, 21% free-sine amplitude, Ru/Ti 6.2: a different ROI layout).

Joint fits (7 nm, x_OH + dz_Obr):

| Fit | red chi2 | E0 (V Ag/AgCl) | w (V) | x_OH | dz_Obr (Å) |
|---|---|---|---|---|---|
| sum, coherent | 4.7 | 0.22 ± 0.04 | 0.13 | 0 -> 1 (each ± 0.5 to 0.9) | -0.06 -> 0.26 |
| Ru, coherent | 11.4 | 0.25 | | | |
| Ru, film only | 50 | | | | |

In the film-only fit, even a full x_OH swing is too small.

The 4.1 nm film allows at most |Δx_OH| = 0.03 (coherent) to 0.10 (film spot only) across 0 to 0.7 V.

C' OH fit:
- red chi2 14.7 (A and B 14.5, C 14.9).
- 4.1 nm: x_OH 0.9 to 1.0 from -0.24 to 0.15 V, 0.80 at 0.35 V, 0.45 at 0.7 V.
- 7 nm: 0.4 to 0.8, then 0.35 at 0.7 V.
- The fall at 0.7 V is the imposed dz_Obr change (0.35 -> 0 Å) being traded against x_OH. It is not a measured change.

## Notes

- The 3.3 nm model is the 4.1 nm structure made thinner. Near the (0 1) Bragg peak at L = 3, the sign of the change
  depends on the fringe phase, so the 125 mismatch says more about that model than about the data.
- The measured Ru/Ti count ratio depends on where the ROIs were drawn: 311 and 344 at the same HKL give 2.1 and 6.2.
  So it cannot test the film/substrate split directly.
- The flat Ti ROI with the large Ru change means the surface does not interfere with the part of the signal in the Ti ROI.
  The CTR fits assume one coherent film + substrate + surface amplitude. A model with an incoherent substrate fraction is the natural test.

## Model-free check of the CTR series (`dratio.py`, `fringes.py`)

Order of measurement: each film's CTR series came first, then its CVs. OCV is assumed to be the first CTR condition.

Double ratio from the CTR files (no scale, no model): D = [I(3.16)/I(2.75)] at E, divided by the same at 0 V, local means ±0.03 L.

| Film | OCV | -0.238 V | -0.1 V | 0.15 V | 0.35 V | 0.7 V | CV 0 -> 0.7 V |
|---|---|---|---|---|---|---|---|
| 4.1 nm | 1.01 ± 0.09 | 0.96 | 0.97 | 1.02 | 1.01 | 1.01 | 1.00 |
| 7 nm | 1.12 ± 0.11 | | 0.95 | 1.22 | 1.24 | 1.21 | 1.73 |

4.1 nm I(3.16)/I(2.84): 0.67 ± 0.06 at OCV, and 0.98 to 1.10 at every controlled potential.

Fringe positions (`fringes.py`): no shift between OCV, 0 V and 0.7 V on any rod, and the even rods are the same at OCV.
So the film itself did not change. The 4.1 nm OCV difference sits on the (0 -1) rod only (rms 11% in ln F, against 6% for 0.7 V),
which makes it a surface change.

Reading:
- **4.1 nm:** the OCV surface differs from every controlled potential. From -0.238 to 0.7 V the rods are the same, and the later CVs are flat.
- **7 nm:** OCV, -0.1 V and 0 V look alike. From 0.15 V up the rods differ (D about 1.2), in the same direction as the CV,
  but the CV's change is about three times larger in ln.

## Robustness (`robust_cv.py`, `robust_ctr.py`, `rodchi.py`, `fig_robust.py`)

What holds:
- **CV amplitude without timing assumptions.** Best sinusoid over periods 60 to 130 points, phase free, with a linear drift term:
  - 4.1 nm: 1.0 to 1.9% peak-to-peak (Ru+Ti) at four HKLs, including a repeat at 2.75.
  - 3.3 nm and 7 nm: 9 to 37% at the same HKLs.
  - 7 nm (2 0 1.75) control: 0.2%.
- **Not insensitive points.** In the model, x_OH 0.3 -> 0.7 changes the 4.1 nm CV points by 14 to 25% for any thickness from 10 to 14.5 TL
  and any dz_Obr from 0 to 0.35 Å.
- **Not saturation or counting statistics.** The 4.1 nm CA counts run from 4.3k to 129k; the low-count 4.1 nm scans (496, 501)
  are at the same level as the responding 7 nm scans (3.7k to 5.4k) and are just as flat.
- **Not beam or cell artefacts.** A potential-locked change in beam, absorption or electrolyte thickness would also show in the Ti ROI
  and in the control rod; both stay flat. The one exception is 344's Ti.
- **No duplicated files** among the CTR conditions.
- **Whole-rod CTR comparison** (one free scale, ±0.08 around the TiO2 Bragg peak left out), as rms of ln(rod / 0 V rod):
  - repeatability, from the even rods and the 7 nm OCV repeat: 0.016 to 0.031
  - 4.1 nm (0 -1): 0.032 to 0.046 at every controlled potential, 0.119 at OCV
  - 7 nm (0 1): 0.028 to 0.034 up to 0.15 V, then 0.056 at 0.35 V and 0.064 at 0.7 V

What does not hold, or is not shown yet:
- **The single double ratio I(3.16)/I(2.75) is fragile.** Over L pairs, windows of 0.02 to 0.05 and a ±0.003 L shift,
  its 10 to 90% range is 0.8 to 1.4 for both films. Use the whole-rod comparison instead.
- **CTR-series differences could be drift in time.** The order of the potentials is not in these files. The CVs are periodic and locked
  to the potential, so they are immune to drift.
- **The potential on the 4.1 nm CVs:** EC-Lab ran without errors (operator), and the cell read a stable rest potential on
  this film. The current traces are to be added when the EC-Lab files are available.
- **One sample per thickness, and the trend is not monotonic** (3.3 and 7 nm respond, 4.1 nm does not). This argues against a simple
  thickness or strain explanation and leaves sample history or preparation open.
- **The 7 nm amplitude varies between repeats:** 311 and 344 at the same HKL give 21% and 37% peak-to-peak.
- **The 7 nm CTR change between 0 and 0.7 V is about a third (in ln) of the CV change at the same L.**

## Work function comparison (Jeong et al., Nat. Commun. 17, 2516 (2026); arXiv 2507.07575)

- Their KPFM work function is non-monotonic in thickness and peaks at about 4 nm, more than 1 eV above relaxed films.
  At that thickness the films change from strained and polar (mm2) to relaxing and nonpolar (4/mmm).
- Our only non-responding film (4.1 nm) is at that peak; 3.3 and 7 nm respond.
- The per-thickness values are not used here: the full text could not be fetched from this session.
- `polar_test.py`: per film, all rods, polar_A free with the shift at the interface (polar_tl 1.5) or through the
  whole film (100 TL), film, surface and scales refit each time:

  | Film | none (red chi2) | interface | film-wide |
  |---|---|---|---|
  | 4.1 nm | 40.58 | 39.49, polar_A -0.18 ± 0.05 Å (eps_perp 1.52 -> 1.29) | 0.00 ± 0.045 Å, no gain |
  | 7 nm | 48.89 | 48.73, -0.026 ± 0.015 Å | +0.02 ± 0.03 Å |

  So there is no sign of a polar distortion through the whole film. A distortion of the reported size (about 0.085 Å)
  is at the 2-sigma limit for 4.1 nm, so it cannot be ruled out either.

Work function values from the group's plot (`jeong_work_function.jpg`): the peak is a single point, 5.40 eV at 3.8 nm.
The points nearest our films:

| Film | WF point | WF (eV) | CV oscillation | Half-change E (V Ag/AgCl) |
|---|---|---|---|---|
| 3.3 nm | 3.2 nm | 4.73 | 10-11% | 0.22-0.30 |
| 4.1 nm | 4.0 nm | 4.86 | 1-2% | none |
| 7 nm | 7.0 nm | 4.44 | 9-37% | 0.17-0.26 |

The response falls as the work function rises; the step potential does not shift. With n = 3, and 3.3 vs 4.1 nm only 0.13 eV apart
(neighbouring KPFM samples differ by up to 0.4 eV), this is a hypothesis.

The KPFM films are different samples from ours (instrument error said to be negligible), so our films sit on the curve by
thickness only. 3.3 nm is 4.73 eV (nearest) or 4.84 eV (interpolated); 4.1 nm is 4.86 or 4.85 eV. Between 3 and 4 nm the
curve moves by up to 1 eV in 0.6 nm, so a thickness-scale offset of 0.3-0.5 nm dominates. KPFM on the same samples would settle it.

## Logbook timeline (`timeline.txt`, `timeline_rods.py`, `ohfit_c3.py`)

The order of measurement is now known, so the CTR comparisons are made in that order (rms of ln ratio):

| Comparison | rms |
|---|---|
| 7 nm, repeats at the same potential (0.7 V; -0.1 V) | 0.039, 0.040 |
| 7 nm, 0.7 vs -0.1 V, down / back up | 0.077 / 0.078 |
| 4.1 nm, every back-to-back pair | 0.031-0.043 |
| 4.1 nm, 0.7 -> -0.238 V | 0.039 |
| 4.1 nm OCV rod (last, fresh spot) vs 0.35 V | 0.110, not used |

- The 4.1 nm CVs were on a fresh spot.
- `ohfit_c3.py` repeats the Rao fit with the logbook's RHE scale (+0.238 V; 0.35 V -> 0.35 Å). The result is the same:
  red chi2 14.77, film difference +0.50 / +0.40 / +0.15 / -0.05 / +0.10 at -0.1 / 0 / 0.15 / 0.35 / 0.7 V.
- `fig_variants.py` and `ctr2cv.py` now use it.

## Second batch: 126, 127, 223, 224, 225 (`process_new.py`, `look_new.png`)

| Scan | Film and reflection | Result |
|---|---|---|
| 126 | 3.3 nm (-2 0 1.75), 5 mV/s | 2.3% peak-to-peak, point noise 2.4%: no response |
| 127 | 3.3 nm (-2 0 1), 25 mV/s | 3.4%, point noise 4% (images 0-78; the rest of the file repeats image 79): no response |
| 223 | 3.3 nm (0 -1 3.16), repeat of 222 | +15.4 +- 0.6% from 0 to 0.7 V over 4 cycles (222: +12.7%), 12.5% peak-to-peak |
| 224 | 3.3 nm (0 -1 3.16), staircase 0-0.7 V | flat within +-2% |
| 225 | 3.3 nm (0 -1 2.75), staircase 0-0.7 V | linear fall to -35% at 0.55 V, then flat |

The staircases disagree with the CVs and with each other. They were recorded after the window broke, with a bubble approaching,
so they are not used (slide outline, backup B7).

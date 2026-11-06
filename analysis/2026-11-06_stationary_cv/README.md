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

## Third batch: 227, 341, 343, 519 (`process_batch3.py`, `look_batch3.png`)

| Scan | Film and reflection | Result |
|---|---|---|
| 227 | 3.3 nm (-1 0 2.32) | 3.5% peak-to-peak, point noise 3.7%: no response (model at most 2%) |
| 519 | 4.1 nm (-2 0 1.75) | 0.8%, point noise 1.1%: no response (images 20-242; the rest of the file repeats image indices) |
| 341 | 7 nm (0 1 2.75) | Ru 42%, Ti 5%, sum 30% peak-to-peak; -27.6 +- 1.3% from 0 to 0.7 V |
| 343 | 7 nm (0 1 2.75) | Ru 42%, Ti 2%, sum 26%; -21.1 +- 1.6% (period taken from 341; the signals collapse after image 143) |

7 nm (0 1 2.75) in time order (311, 341, 343, 344): Ru spot 30, 42, 42, 42%. The sum depends on the ROI split (Ru/Ti 2.1, 2.1,
1.6, 6.2), which explains 344's larger sum change. The response reproduces and does not grow with cycling after the first scan.

## Scan 135: 3.3 nm (1 0 1.75) (`process_batch3.py`, `look_135.png`)

- Full signal (ROI1): 1.1% peak-to-peak, point noise 2.7%. No response (model at most 3%). Images 50-158 were used; after them
  the file repeats image index 159.
- The lower and upper spots (ROI4, ROI5) each wander by 3-5% (near noise), roughly in opposite phase, and cancel in the full
  signal.
- (1 0 1.75) is an odd rod that the model calls OH-blind. Its flat response shows the selection follows the model's OH
  sensitivity, not just odd vs even rods.

## Scan 147 corrected, and the plot data (`extract2.py`, `make_plot_data.py`)

- **147:** the hold starts at the logged Start (image 50), as before. The period now comes from the Ru peaks (150, 221, 293:
  about 72 images), not from the summed signal (162, twice the cycle). The Ru spot drops on the first anodic sweep, then
  stays 20-34% low and recovers only partly (to -12%) at 0.3-0.4 V on the later anodic sweeps, about 80 s after the potential
  returns to 0 V. It is hysteretic and is still left out of the fits. No other scan changed.
- **`plot_data/`:** every plotted quantity as a CSV, sorted by folder (see `plot_data/README.md`). The rebuilt CV curves
  match `cv3.json` to the 6 printed digits.

## Rao's full structure against the data (`rao_full_test.py`)

The OH rods were fitted with each surface structure held fixed, refitting only every rod scale and, where stated, the water height.
Results are chi2 per point, with the difference from the best fit scaled by its chi2 per point:

| Structure | 4.1 nm | 7 nm |
|---|---|---|
| bridging O at 0 Å, OH fraction fitted | 16.1 | 9.7 |
| Rao bridging O, OH fraction fitted | 16.5 (+11) | 9.9 (+6) |
| Rao CUS species and bridging O, water height free | 17.1 (+25), water height drifts to 2.15 Å, an OH-like height | 10.9 (+38) |
| the same, water height fixed at Rao's 2.68 Å | 23.1 (+167) | 12.6 (+93) |

- Rao's full structure fits neither film. It fits 4.1 nm worse.
- The data mildly prefer the bridging O at 0 Å over Rao's protonated height at low potential.
- At 0.7 V (0.94 V RHE) both partial fits use 0 Å and give an OH fraction of 0.35-0.45, consistent with Rao's half water / half OH
  at 1.0 V RHE. Integer rods cannot tell an ordered every-other-site arrangement from a random half coverage.

## EC-Lab data (`data/echem/`, `make_ec_figures.py`, `ec_figures/`, `ec_summary.json`)

- **File matching:** EC files are matched to the X-ray scans by name, with three inferences: 'scan119' is CV 120; on 7 nm,
  'scan342' is CV 343 and 'scan247' is CV 344. The day-4 files are named '7nm' but hold the 4.1 nm session.
- **The potential follows the programme on every film** (Ewe within about 10 mV of the setpoint).
- **Current, 3.3 and 7 nm:** a rising background (6-14 uA/V) with a gap of about 1-2 uA between the sweeps (half-gap 0.4-1.1 uA, i.e.
  80-230 uF at 5 mV/s), and 430-850 uC per anodic sweep in the clean scans. 7 nm 311, 314 and 344 carry large spikes.
- **Current, 4.1 nm:** zero within noise in every file (half-gap at most 0.005 uA, 0.0 uC per cycle, no current at potential steps up to
  0.94 V). Updated below (`ec_leak.py`): the WE lead was on the film, so an open lead is unlikely.
- **Consequence (revised, see the day-4 section below):** with the CE / RE in the cell reservoir, no bubbles, and a good WE
  circuit, the potential most likely did reach the 4.1 nm film and the current record is what failed. The 4.1 nm X-ray
  data are then a film under potential, pending the tests listed there.

### Day 4 current, what the WE lead saw (`ec_leak.py`, `ec_leak.json`)

The user's checks: a disconnected WE reads about 10 V at open circuit on this BioLogic; on day 4 the OCV sat at 0.371-0.40 V and jumped
when the beam hit the film; the contact measures 500 Ohm. So the WE lead was on a wetted film. The recorded current then bounds the
film's connection to the CE / RE:

- **Current channel live:** random noise 0.1-0.25 uA per point (no exact zeros); the same current range code (14) as the other films,
  which leave it only when larger currents flow; Ewe noise as on the other films.
- **Resistive part:** the long holds at 0 and 0.7 V (508-524), which alternate with the OCV readings, draw -0.56 +- 0.38 and
  -0.48 +- 0.28 nA. The difference is 0.1 +- 0.7 nA/V, so more than about 0.7 GOhm. The CVs give a sweep slope of
  5.6 +- 1.3 nA/V; 3.3 and 7 nm give 6,000-14,000 nA/V.
- **Capacitive part:** -0.05 +- 0.05 uF over six CVs; 3.3 and 7 nm give 85-230 uF.
- **What fits all of it:**
  - (a) The film's electrolyte was cut off from the CE / RE compartment (an air gap or bubble in the channel; more than about 1 GOhm).
    The programme then drops across the break, not across the film: Ewe tracks, about nA flows, and the OCV can still be read through
    the break by a 1e12 Ohm electrometer and still react to the beam. No potential at the film.
  - (b) The film was tied to the instrument ground by a second path around the ammeter. Then the potential was applied, and at most
    about 0.1 % of the current reached the ammeter.
  - (a) needs one common cell failure and also explains the flat X-ray data; (b) needs a wiring accident and a film that does not
    switch, unlike 3.3 and 7 nm. Neither is proven.
  - **Update:** the user inspected the cell after removal and found no bubbles. It was set up the night before and used all day.
    That rules out (a) as a bubble; (a) is left only if the CE / RE sat behind another kind of break (a closed valve, a pinched
    tube, a separate compartment). (b) becomes the leading explanation.
  - **Why (b) is plausible:** in grounded mode, if the current is read across a range resistor in the WE line (about 10-100
    kOhm on a uA range), a WE-side short to earth of even a few hundred ohms takes more than 99.9 % of the current. The film
    is then held at the potential through the sense lead and the short, and the ammeter sees about zero. Whether this BioLogic
    reads current that way is not known here.
  - **Only an electronic WE-side short does this.** A leak of electrolyte onto a grounded part does not: the film's own current
    still passes through the WE lead.
  - **Tests:**
    - The EC-Lab channel setting on day 4: grounded or floating. Floating rules out (b).
    - A lab CV with a jumper from the Au wire to the potentiostat ground, to see whether the recorded current collapses while
      Ewe tracks.
    - The resistance from the Au wire / WE clip to the holder and goniometer with the day-4 mount.
  - **Further facts (user):**
    - EC-Lab was set to grounded, for a floating cell.
    - The CE and RE sit in the cell reservoir, not behind tubing or a compartment.
    - The resistance between the leads outside the cell read a couple hundred ohms, checked several times during the run.

    With a continuous electrolyte from the film to the CE / RE, no break is left. Ewe tracking then means the film's potential
    followed the programme, so the film's double layer was charged and that charge went around the ammeter (or the channel
    or cable misread it).
  - **The only way the potential could have missed the film** was an insulating layer on it (grease, residue). The user
    measured every pair of the three electrodes through the cell (WE-CE, WE-RE, CE-RE) and each clip to its electrode's
    wetted area, at a couple hundred ohms. That rules out a blocked film. So the potential was applied, and the open
    question is only where the current went.
  - **The measurement not yet made** is the one the bypass needs: each lead, the Au wire and the film's front edge to the
    instrument ground (black lead / chassis / mains earth) and to the diffractometer, with the cell mounted as on day 4.
  - **A dummy cell** (a known resistor) on the day-4 channel and cable tests the second and third candidates. A WE-to-shield
    fault or a misreading channel would show the same missing current.
  - **How complete the bypass was:** the recorded hold currents at 0 and 0.7 V differ by 0.1 +- 0.5 nA, against about 8 uA
    on 3.3 nm. So less than about 0.02 % of the current was recorded. With a 10-100 kOhm range resistor in the WE line,
    the bypass must be below about 2-20 Ohm: a near-direct metal connection from the WE side to the instrument ground.
    Candidates: the WE clip, Au wire or film edge touching earthed metal (the instrument ground is mains earth in grounded
    mode), or a fault in the WE lead or connector to its shield. The black ground lead was never connected (user), so it is
    not the path. A third candidate is the channel misreading the current while it flows.
  - **Status:** the 4.1 nm null would then be a real film difference (no switch under potential), but it rests on a current
    nobody recorded. To defend it:
    - reproduce the day-4 signature with a jumper from the WE to ground (or find the short, or show a cable / channel fault
      with a dummy cell);
    - show with a lab CV that the 4.1 nm film is electrochemically normal.

## 3.3 nm (0 -1 L) rod at 0.65 V, scans 153-221 (`rod153.py`, `rod153.json`, `fig_rod153.py`, `fig_rod153.png`)

- **Data:** phi rocking scans (`dscan phi -0.1 0.1 60`) with the ROI sums, plus the SPEC file (`data/08_9_3.3nm.spec`). From SPEC:
  L runs 0.30 to 3.70 in steps of 0.05; the logbook's 0.35 start is one step off, which an earlier version of this section
  used. L is constant within each phi scan (phi moves H only). Filters: 170 (transmission 0.19), 206 (part), 207 and 208 (0.036).
  The monitor is steady to 0.5 %. The logbook says the cell window broke during the rod.
- **No L offset:** with the SPEC L, the (0 -1 1) and (0 -1 3) Bragg peaks fall at exactly 1.00 and 3.00. The earlier
  "0.02-0.05 below nominal" came from the wrong start and is withdrawn.
- **Usable range:** the ROI sums are raw (unprocessed) and the signal is inside ROI1 on almost all frames (the
  user's check of the images). Below L about 2.4 the signal is weak and the frame-to-frame profiles are step-shaped:
  the edge of the phi rocking profile, not the spot leaving the ROI (an earlier version of this section said so;
  withdrawn). From 2.40 up each scan's mean over the frames (normalised by transm and monitor) is used as a
  stationary point; points within 0.08 of the Bragg peak are left out.
- **Check against the CVs (same ROI size, counts per 1 s frame):**
  - CV 222 at L 3.168 gives 153k. The rod scans give 181k at 3.15 and 91k at 3.20, so the rod scans measure the same
    intensity as the CV that shares their UB.
  - CV 120 (2.755) and CV 125 (3.257) give 2-3x less than the rod at the same L. Both were measured with the older UB,
    before the orientation was re-anchored on the measured (0 -1 3) peak at 16:22. They were probably off the rod
    maximum, or part of the spot fell outside their own ROI.
- **CV positions (#Q):** 120 at L 2.755 and 125 at 3.257 (older UB), 222 / 223 at 3.168 (the rod's UB), each at K -1.006
  to -1.007 by #Q. The detector arm (nu) sat about 0.04 deg from the calculated position, so this #Q K offset is mostly
  the detector, not the sample.
- **Shape:** the 3.3 nm film puts a strong hump at L 3.10-3.20, then a minimum at 3.30. CV 3.168 sits on the hump;
  CV 3.257 sits on the steep edge just before the minimum (d ln I / dL about -30). A shift of 0.005 in L there is -13 %.
- **Fit** (film from the 4.1 nm fit, surface at the 7 nm 0.7 V state; free thickness, spread, eps_perp, roughness, L
  offset, a slope for the geometry corrections; scale solved): reduced chi2 about 57, thickness on the 5 TL bound, and
  dL +0.047. With wide bounds (3-32 TL, eps -3 to 3 %), local minima from 4 to 29 TL, all with reduced chi2 48-80.
  The rod, as extracted, does not determine the 3.3 nm structure.
- **CV changes at 0.7 V, predicted from the CTR states (best rod fits, three dz_Obr assumptions) vs measured on 3.3 nm:**

  | L | predicted | measured 3.3 nm | measured 7 nm |
  |---|---|---|---|
  | 2.755 | -1 to -8 % | -13 % | -27 % |
  | 3.168 | +4 to +9 % | +11 % | +18 % |
  | 3.257 | +7 to +10 % | -13 % | +6 % |

- **Which structures give the 3.3 nm pattern (-, +, -)?** We scanned a grid of 2,430 film structures (4-30 TL, eps -1 to
  2.5 %, spread 0.5-3 TL, roughness 0.6 / 1.2 TL). The pattern needs every change beyond 2 %:
  - **A (dz_Obr fitted):** 0 structures.
  - **B (dz_Obr = 0):** 3 structures, all at eps 2.5 %.
  - **C' (Rao):** 15 structures, all with a sharp thickness (spread 0.5 TL) at 13-14 or 21-26 TL, so that 3.257 sits
    next to a fringe minimum.

  So the decrease at 3.257 is not generic. It needs a film with sharp fringes placed just so. The minimum seen at L
  3.30 is a hint of that, but this rod cannot show it.
- **A uniform potential-dependent film strain does not fix it:** it moves 3.168 and 3.257 together.
- **Status:** the 3.3 vs 7 nm sign difference at L 3.25 is the only remaining film difference. It rests on one scan
  (125), and whether it is structural (fringes) or a different surface response cannot be decided from this rod. The
  next step is to re-process the images of 153-221 with an ROI that holds the moving spot (the Processing tool,
  with this SPEC file). That gives a proper rod and a real 3.3 nm structure.

## Net anodic current on days 2 and 3, and shorts on the WE side

- **Net charge per CV cycle** (current integrated over whole cycles):
  - 3.3 nm: +970 to +1,260 uC (mean +3.5 to +4.5 uA; about +8 uA at 0.7 V, about 0 at 0 V).
  - 7 nm 311, 316: +7,400 uC (+56 uA at 0.7 V); 314: +35,000 uC (+293 uA at 0.7 V); 344: +14,600 uC.
  - 7 nm 356: +350 uC.
  - 4.1 nm: 0.

  A clean RuO2 film in this window should be close to reversible. Something is oxidised every cycle on days 2 and 3, and
  the CV charges quoted above are dominated by it. Two candidates:
  - Ga-In exposed to the electrolyte. Ga and In dissolve anodically at every potential in the window.
  - Beam radiolysis products (H2, H2O2) oxidised at the film.

  The test: a beam-driven current follows the shutter and the filters; Ga-In corrosion does not. The EC times can be
  synced to the SPEC epochs to check this.
- **Short outside the cell, WE side to the diffractometer:**
  - *Grounded potentiostat (WE at ground):* the film is still under potential, with an error equal to the ground-loop
    voltage between the diffractometer and the potentiostat (mV to tens of mV, plus hum). Part of the current returns
    through the diffractometer and is not recorded; how much depends on the resistance of the short against the WE
    lead and the ammeter input.
  - *Floating potentiostat:* a single short to ground changes nothing.

  This fits all the day-4 facts: Ewe tracks, about zero current, a real OCV that reacts to the beam, and no overload.
  To lose more than 99.9 % of the current, the short must be far lower in resistance than the WE lead path. A good
  clip with a metal short would more likely split the current than remove it.
- **Short inside the cell, Au wire or Ga-In touching the electrolyte:**
  - The exposed metal becomes part of the WE, so the film stays under potential. The error is the film current times
    the 500 Ohm contact, well below 1 mV.
  - The recorded current is the film's plus the metal's. Au adds little below about 0.9 V Ag/AgCl. Ga-In adds a
    steady, irreversible anodic dissolution current that rises with potential, and it pulls the OCV toward its own
    corrosion potential, far below RuO2's.
  - Ga3+ / In3+ go into the electrolyte, and the eutectic is slowly eaten.
  - This cannot explain day 4: zero current and an OCV of +0.37 V. It is one of the two candidates for the net anodic
    current on days 2 and 3, together with their low OCVs (about 0 V).

## Absolute sync: the response at the potential EC-Lab applied (`sync_abs.py`, `sync_abs.json`, `fig_sync_abs.png`)

- **Times:** each X-ray point is timed from the SPEC file (`#D` + the `Epoch` column). Each EC-Lab sequence is timed from
  its file timestamp + `time/s`; the CA and CV files of a sequence share both. The SPEC file covers every day
  (scans 1-544).
- **The clocks agree:**
  - The user's Start marks fall 1-2 s from the EC-Lab sequence start on every day, day 4 included (120: -7 s,
    344: +5 s).
  - The offset that best aligns the X-ray signal with the potential on the responding scans is -12 to +15 s (median
    +4 s). That covers both the clock offset and any lag of the response.
- **Fit, no free timing:** each scan is fitted as I/I0 = quadratic drift + A g(E(t)), with g a step at 0.23 V (the 3.3 / 7 nm
  half-way potential).
  - A is the change from 0 to 0.7 V.
  - Its error is the spread of A when E(t) is shifted by 70-210 s, which keeps the noise's own correlations.
  - Moving the offset by +-10 s changes A by less than its error.
  - A linear-in-E shape gives the same signs and about 1.3x the size.
- **Results (weighted per reflection; 7 nm 2.75 from 311, 341, 343, without 344 at 0.6 deg incidence):**

  | reflection | 3.3 nm | 7 nm | 4.1 nm | 4.1 / 3.3 | 4.1 / 7 |
  |---|---|---|---|---|---|
  | (0 +-1 2.75) | -9.7 +- 2.2 % | -19.9 +- 2.9 % | +0.3 +- 0.3 % (496, 497) | -0.03 +- 0.03 | -0.015 +- 0.016 |
  | (0 +-1 3.16) | +8.5 +- 1.3 % | +14.9 +- 3.2 % | -0.3 +- 0.6 % (498) | -0.03 +- 0.07 | -0.02 +- 0.04 |
  | (0 +-1 3.25) | -8.5 +- 2.3 % | +8.3 +- 2.0 % | +2.6 +- 1.0 % (501) | | +0.32 +- 0.14 |

  Also: 4.1 nm (0 -1 2.84) -0.2 +- 0.4 %, (-2 0 1.75) +0.04 +- 0.12 %; 3.3 nm (0 -1 1.25) -3.3 +- 0.5 %.
- **Reading:**
  - At 2.75 and 3.16 the 4.1 nm response is at most about 4 % and 11 % of the 3.3 nm response, and 2 % and 7 % of the
    7 nm response (2 sigma).
  - The 3.25 value (501, 2.8 sigma) is not a clean step: binned by potential, the low-potential bins sit lower on both
    sweeps, but the signal does not stay up above 0.4 V, unlike the monotonic 316 and 125. Inconclusive.
  - If the potential reached the film (see the day-4 section), 4.1 nm did not switch.
- **Day-4 conditions that differ:**
  - Kapton window.
  - Rest potential 0.37-0.40 V (about 0 V on days 2-3).
  - The logbook turns argon on for day 3 (Eoc 200 -> -20 mV) and says nothing about gas on day 4.

## Clock-synced half-way potentials (`curves_sync.py`, `curves_sync.json`)

- Same normalisation as slide 12b: the first bin is 0 %, the mean of the last three bins is 100 %, sweeps averaged. The
  potential comes from the EC-Lab clock.
- **3.3 nm:** averaged curve 0.25 V; scans 120 0.205, 222 0.204, 223 0.31.
- **7 nm:** averaged curve 0.22 V; scans 311 0.165, 314 0.238, 341 0.206, 344 0.196. 343 is too noisy to place (A -20 +- 14 %).
- The earlier timing gave 0.24 and 0.21 V. The two films still switch at the same potential within the scan spread.

## Is the anodic current driven by the beam? (`beam_current.py`, `beam_current.json`)

- **3.3 nm hold at 0.65 V during the rod 153-221:** +3.7 uA mean at full beam (scan-to-scan sd 1.6); +3.7 uA in the gaps
  between scans. With the beam cut to 3.6 % (scans 207, 208) it is 2.8-3.0 uA, and 3.65 uA at 19 % (170).
- So the current is not produced by the beam at the moment. Radiolysis products built up over hours, or an exposed
  contact, are not excluded.
- **7 nm CTR series:** -2.9 uA at -0.1 V, +0.6 at 0 V, +1.3 at 0.15 V, then 40-85 uA (spiky) at 0.35 and 0.7 V. The gaps
  give the same.

## Slide outline under the stated assumption

`slides_outline.txt` now treats the 4.1 nm film as under potential control on day 4. That is an assumption, stated on the
slides with its support (backup B8). Further changes:
- slide 6 leads with the clock-synced result (`slide06b_cv_response_synced.png`);
- the half-way potentials and the 4.1 nm bounds are updated;
- new backups: B8 (day-4 current record), B9 (currents and the beam), B10 (3.3 nm rod);
- the Q&A covers the assumption, the beam and the 3.25 sign.

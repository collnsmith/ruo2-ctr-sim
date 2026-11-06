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
   - OCV: 0.2 Å (assumed about 0.9 V RHE)

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
The Ti ROI is flat in every scan.

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

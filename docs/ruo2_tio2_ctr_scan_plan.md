# In situ CTR scan plan: RuO2(110) film on TiO2(110)

One film per run. Copy this file for each film/thickness and fill in the blanks.

| Field | Value |
|---|---|
| Film ID | |
| Nominal thickness (nm) / trilayers (1 TL ≈ 0.32 nm) | / |
| Thickness from XRR / lab XRD fringes | |
| Lab XRD film (220) position → ε⊥ | |
| Substrate miscut (deg, direction) | |
| Surface prep / pre-cycling protocol | |
| Electrolyte, pH | |
| Reference / counter electrode | |
| Beamline, date, shift | |
| Energy | 16 keV (λ = 0.7749 Å) |

---

## 0. Tired version

Read only this if it is 3 am.

1. **TiO2 is the boss.** Align on TiO2 peaks only: (0 0 2), (1 1 1), (1 1 3). Never align on a RuO2 peak.
2. **RuO2 rides along.** Same H and K as TiO2. RuO2 peaks sit almost on top of the TiO2 peaks (about 2.005, 4.01, 6.016 on 00L). You will not see a separate RuO2 peak. That is normal.
3. **Is the film relaxed?** Scan H through (2 0 2.03) from H = 1.85 to 2.05. One peak at 2.00 = commensurate. A bump or tail toward 1.90 = the [001] strain has relaxed (expected above ~4 nm). Write it down. A bump in the K scan (toward 2.045) is NOT expected: flag it.
4. **Bright rods first** (they tell you what the film is): 00L, then (1 1 L), (0 2 L), (2 0 L).
5. **Weak rods second** (they tell you OH vs H2O): (0 1 L), (0 3 L), (0 5 L), then (1 0 L). They are 100 to 300 times weaker. Count longer. Background matters.
   - **Films thicker than ~4 nm: trust only rods with H = 0**, i.e. (0 K L). The relaxed top of the film does not line up with TiO2 on H ≠ 0 rods.
6. **One mirror copy:** (0 -1 L). It should match (0 1 L). The mismatch is your real error bar.
7. **Potential series at the magic points** (table in section 8.1): P1 to P6 plus controls C1, C2.
   - P2 and P3 should move in **opposite** directions if OH/H2O changes. Same for P4 and P5. If everything moves together, it is drift or damage, not chemistry.
   - C1 and C2 should **not** move. If they do, the film itself changed.
8. **After every potential step:** wait for the current to settle, then measure.
9. **Do not sit near OER.** RuO2 corrodes. Get in, measure, get out.
10. **End of film:** back to E_ref, rescan 00L from L = 3.4 to 4.6. If the fringes or film peak changed, the film changed. Note it.
11. **Write everything down** (section 14): scan #, potential, current, time, beam spot, attenuator.

If confused: bright even rods = film structure. Weak odd rods = surface chemistry. Controls = sanity.

---

## 1. Goal and logic

**Goal:** determine how the CUS-site adsorbate (H2O / OH, possibly O) on RuO2(110) changes with potential, for one film thickness, with enough film-structure information to fit it.

**Why the rod parity matters:** in every bulk-like (110) trilayer the 6-fold and CUS metal atoms sit at the same height with relative phase (-1)^(H+K).

| Rods | Metal contribution | Brightness | Main use | Model OH vs H2O sensitivity |
|---|---|---|---|---|
| H+K even: (0 0), (1 1), (0 2), (2 0), (2 2) | Ru, Ti add | Bright | Film thickness, strain, roughness, interface, Ru positions | ≤ 5 to 8 % |
| H+K odd: (0 1), (1 0), (1 2), (0 3), (2 1) | Ru, Ti cancel (bulk-like layers) | 100 to 1000x weaker | Surface O, including CUS adsorbate | 20 to 75 % |

**Strategy:** even rods fix the film model, odd rods and fixed-q potential scans resolve the surface. Fitting the surface from odd rods alone leaves too many correlated parameters.

**Thickness and relaxation:** the −4.7 % [001] strain starts relaxing at ~4 nm, while the +2.3 % [1-10] strain persists (cracking at ~17 nm; thickest film here 10 nm). Relaxation along [001] only moves rods with H ≠ 0. Rods with H = 0, (0 K L), have no momentum component along [001], so the whole film, including a relaxed top and its surface, still adds coherently with the substrate there. All potential-series points below are therefore on H = 0 rods, so one point list works for every thickness.

**Caveats to keep in mind while measuring:**
- X-rays barely see H. OH vs H2O is inferred from the CUS Ru-O height (model default 2.0 vs 2.2 Å), occupancy and disorder, not from H directly.
- A random OH/H2O mix looks like one species at an average height with extra disorder. Potential dependence, internal sign checks (P2/P3, P4/P5) and DFT constraints are what make the assignment credible.
- All predicted numbers below come from the simulator (`ruo2_ctr_gui`: GUI or `ctr_notebook.py`) with default parameters (6.5 nm film, ε⊥ = +2.0 % from Ruf et al. 2021). Re-run with measured values before trusting exact L positions.

---

## 2. Conventions and key numbers (16 keV)

**Surface cell** (TiO2-indexed, rectangular, p2mm):

| Index | Direction | Length |
|---|---|---|
| H | [001] | a1 = 2.9587 Å |
| K | [1-10] | a2 = 6.4965 Å |
| L | [110] | a3 = 6.4965 Å (bulk TiO2 (110) = (0 0 2)) |

- Bulk rutile index from surface index: (h k l)_bulk = ((K+L)/2, (L-K)/2, H). Bulk Bragg peaks only where K+L is even, minus rutile extinctions (e.g. (0 1 1) and (1 2 2) are absent).
- p2mm symmetry: (H K L) ≡ (-H K L) ≡ (H -K L) ≡ (-H -K L).
- λ = 0.7749 Å. With 2θ ≤ 60°, q_max = 8.1 Å⁻¹. All rods below reach L = 6.5.
- Critical angle at 16 keV: RuO2 ≈ 0.18°, TiO2 ≈ 0.15°.

**Alignment / reference reflections (TiO2, 16 keV):**

| (H K L) | Bulk rutile | d (Å) | 2θ (°) | Use |
|---|---|---|---|---|
| (0 0 2) | (110) | 3.248 | 13.70 | UB, primary |
| (1 1 1) | (101) | 2.487 | 17.92 | UB, primary |
| (1 1 3) | (211) | 1.687 | 26.55 | UB, primary |
| (0 2 2) | (200) | 2.297 | 19.42 | Check, relaxation along [1-10] |
| (0 0 4) | (220) | 1.624 | 27.60 | Check, film ε⊥ |
| (2 0 2) | (112) | 1.346 | 33.45 | Check, relaxation along [001] |
| (0 0 6) | (330) | 1.083 | 41.94 | Optional, film ε⊥ |

**Bragg peaks along each rod (L ≤ 6.5):** film values use ε⊥ = +2.0 % (film L = TiO2 L × 1.0026). Film and TiO2 peaks overlap within the film peak width, so the film shows up as fringes around the TiO2 peak, not as its own peak.

| Rod | Parity | TiO2 Bragg L | RuO2 Bragg L (pred.) | Median I vs 00L (model) |
|---|---|---|---|---|
| (0 0) | even | 2, 4, 6 | 2.005, 4.010, 6.016 | 1 |
| (1 1) | even | 1, 3, 5 | 1.003, 3.008, 5.013 | 0.58 |
| (0 2) | even | 2, 4, 6 | 2.005, 4.010, 6.016 | 0.63 |
| (2 0) | even | 2, 4, 6 | 2.005, 4.010, 6.016 | 0.50 |
| (2 2) | even | 2, 4, 6 | 2.005, 4.010, 6.016 | 0.39 |
| (0 1) | odd | 3, 5 | 3.008, 5.013 | 1/170 |
| (1 0) | odd | 2, 4, 6 | 2.005, 4.010, 6.016 | 1/140 |
| (1 2) | odd | 4, 6 | 4.010, 6.016 | 1/270 |
| (0 3) | odd | 1, 5 | 1.003, 5.013 | 1/200 |
| (2 1) | odd | 3, 5 | 3.008, 5.013 | 1/1200 |
| (0 5) | odd | 1, 3 | 1.003, 3.008 | 1/650 |

**Laue fringe period and L step vs thickness:** ΔL ≈ 6.4965 / t(Å). Use a step of ≤ ΔL/5.

| Thickness (nm) | Trilayers | Fringe period ΔL | Max L step |
|---|---|---|---|
| 3 | 9 | 0.22 | 0.04 |
| 5 | 16 | 0.13 | 0.026 |
| 6.5 | 20 | 0.10 | 0.020 |
| 10 | 31 | 0.065 | 0.013 |
| 15 | 47 | 0.043 | 0.009 |
| 20 | 62 | 0.032 | 0.006 |

---

## 3. Before the beamtime (per film)

- [ ] Lab XRD 2θ-θ through TiO2 (110)/(220): film peak position → d(110)_film and ε⊥; fringe spacing → thickness.
- [ ] XRR or AFM: thickness, roughness, step structure, miscut.
- [ ] Lab CV in the same electrolyte and cell type: stable window, redox features, OER onset. Pick the potential list (section 8.2).
- [ ] Use an identical surface prep and pre-cycling protocol for every film. Differences here will look like thickness effects.
- [ ] Update the simulator (GUI sidebar or the notebook Settings cell) and save it as a preset for this film:
  - Photon energy 16 keV
  - Thickness (nm) from XRR/XRD
  - Out-of-plane strain from lab XRD (mode "manual")
  - Top roughness and thickness spread (nm) from XRR/AFM
  - CUS Ru-O heights and B factors from DFT or literature
- [ ] Run with the sensitivity scan and the thickness study. Update the P1 to P6 and C1/C2 positions in section 8.1 and the Bragg table in section 2.
- [ ] Prepare scan macros for sections 5 to 8 with L lists that skip or finely sample the Bragg regions.

---

## 4. Cell setup and electrochemistry

- [ ] Fill the cell, check for bubbles on the sample and in the beam path. Bubbles move and change the background.
- [ ] Connect the potentiostat and verify contact **before** the beam hits the sample.
- [ ] Run the in-cell CV (2 to 3 cycles, same range as in the lab). Compare with the lab CV. Save it.
- [ ] Hold at **E_ref** (lowest potential of the stable window, commonly assigned to H2O on the CUS sites in the RuO2(110) literature; treat that assignment as a hypothesis).
- [ ] Minimize the electrolyte path. At 16 keV the attenuation length of water is about 0.7 cm, so a 5 mm path transmits about 50 %.
- [ ] Choose a fixed incidence angle of about 2 to 3 × α_c (roughly 0.4 to 0.5°) unless the cell geometry forces otherwise. This keeps refraction effects small and the footprint constant. Record it; it sets the practical L_min.
- [ ] Check the footprint against the sample size at the chosen angle.

---

## 5. Alignment (TiO2 only)

1. Sample height and surface normal: specular reflection or low-q reflectivity. Record.
2. Find (0 0 2). Center it.
3. Find (1 1 1) and (1 1 3). Build the UB matrix with the TiO2 surface cell (2.9587, 6.4965, 6.4965 Å, 90°, 90°, 90°).
4. Verify on (0 2 2), (0 0 4) and (2 0 2). Record the angular offsets.
5. Miscut: compare the surface normal (step 1) with the (110) lattice normal. Record the magnitude and azimuth.
6. Rod check: short L scan on (1 1 L), L = 0.5 to 0.9. The rod should stay centered in the detector region of interest (ROI). A drifting ROI center means miscut or UB error.

Do **not** add RuO2 peaks to the UB. They are broad and their L offset is real.

---

## 6. Film checks (at E_ref)

### 6.1 Relaxation (do this first, every film)

The fully relaxed RuO2 position in TiO2 units is H × 0.952, K × 1.023, L × 1.023.

| Scan | Range | Step | Pseudomorphic film | Fully relaxed film |
|---|---|---|---|---|
| H scan at (H 0 2.03) | H = 1.85 to 2.05 | 0.002 | on rod at H = 2.000 | peak at H ≈ 1.905 |
| K scan at (0 K 2.03) | K = 1.95 to 2.10 | 0.002 | on rod at K = 2.000 | peak at K ≈ 2.045 |
| Optional H-K mesh at L = 1.02 | H 0.93 to 1.02, K 0.98 to 1.04 | 0.003 | at (1 1) | near (0.952, 1.023) |

- Expected: films ≤ ~4 nm show no relaxation. Thicker films show intensity toward H ≈ 1.905 (partial relaxation gives a shoulder or tail between 1.905 and 2.000). The K scan should stay clean because the [1-10] strain persists; anything there is unexpected.
- Estimate the relaxed fraction and the degree of relaxation (peak position between 2.000 and 1.905) from the H scan. Enter them in the simulator as "Commensurate thickness" (nm, ~4) and "Relaxation above it" (0 = commensurate, 1 = bulk RuO2), then run again with the scan and the study.
- The relaxed part scatters at its own H positions on H ≠ 0 rods and does not interfere with the substrate there. H = 0 rods are unaffected in position. That is why the section 8 points are all on H = 0 rods.

### 6.2 Strain and thickness

- 00L fine scan, L = 3.6 to 4.5, step 0.005 (use attenuators near L = 4).
- A commensurate film has d(110) ≈ 3.240 Å vs 3.248 Å for TiO2 (ε⊥ ≈ +2.0 %), so its peak sits only ~0.01 in L above (0 0 4). It is buried in the TiO2 peak and cannot be peak-picked. Get ε⊥ from the asymmetry of the fringe envelope around L = 2, 4, 6 by fitting with the model (out-of-plane strain setting), or from lab XRD Nelson-Riley analysis.
- If a film peak is clearly resolved away from the TiO2 peak, the film is probably partially relaxed. Cross-check with section 6.1. For a resolved peak at L_f near 4:
  - d(110)_film = 4 × 3.2482 / L_f (Å)
  - ε⊥ = d(110)_film / 3.1763 − 1
- Thickness: t ≈ 6.4965 / ΔL (Å), where ΔL is the fringe period.
- If a laptop is available: update the out-of-plane strain and thickness, re-run the sensitivity scan, and correct the P2/P3 positions (they sit next to the film Bragg peaks).

---

## 7. Rods at E_ref

Measure integrated intensities at every point: an area-detector ROI with background ROIs on both sides (or rocking scans). Use automatic attenuation near Bragg peaks.

### 7.1 Tier 1: even rods (film model)

| Rod | L range | Step | Fine region | Notes |
|---|---|---|---|---|
| (0 0 L) | 0.3 to 6.5 | fringe step (0.02 at 6.5 nm) | 0.005 within ±0.2 of 2, 4, 6 | Most important rod. Low L limited by incidence angle and background |
| (1 1 L) | 0.3 to 6.0 | fringe step | ±0.2 of 1, 3, 5 | |
| (0 2 L) | 0.3 to 6.0 | fringe step | ±0.2 of 2, 4, 6 | |
| (2 0 L) | 0.3 to 6.0 | fringe step | ±0.2 of 2, 4, 6 | |
| (2 2 L) | 0.3 to 6.0 | 2x fringe step | | Optional, if time allows |

### 7.2 Tier 2: odd rods (surface / adsorbate)

| Rod | L range | Step | Fine region | Relative I | Notes |
|---|---|---|---|---|---|
| (0 1 L) | 0.3 to 6.0 | 0.05 | 0.02 within ±0.4 of 3, 5 | 1/170 | Highest priority odd rod (H = 0) |
| (0 3 L) | 0.3 to 6.0 | 0.05 | 0.02 within ±0.4 of 1, 5 | 1/200 | H = 0 |
| (0 5 L) | 0.5 to 5.0 | 0.05 | 0.02 within ±0.4 of 1, 3 | 1/650 (brighter near L = 3) | H = 0; carries P4/P5 |
| (1 0 L) | 0.3 to 6.0 | 0.05 | 0.02 within ±0.4 of 2, 4, 6 | 1/140 | H ≠ 0: reliable only for films ≤ ~4 nm |
| (1 2 L) | 0.5 to 6.0 | 0.05 | 0.02 within ±0.4 of 4, 6 | 1/270 | H ≠ 0: reliable only for films ≤ ~4 nm |
| (2 1 L) | 0.5 to 6.0 | 0.1 | | 1/1200 | Optional; likely background-limited |

- Deep minima on odd rods (model |F|² below about 1 e²) are practically unmeasurable. Cap the time per point rather than chasing them.

### 7.3 Symmetry equivalents

- (0 -1 L) over L = 0.3 to 3.5 and (-1 0 L) over L = 0.3 to 4.5, at the same steps.
- Their agreement with (0 1 L) and (1 0 L) gives the systematic error (typically 5 to 15 % in SXRD). Use it as the error floor in fitting and when judging whether an effect is real.

---

## 8. Potential-dependent measurements

### 8.1 Fixed-q points

Model predictions (defaults: 6.5 nm film, ε⊥ = +2.0 %, 16 keV, bulk water on 00L). The change is 2(I_OH − I_H2O)/(I_OH + I_H2O). I is the model intensity at 50/50 OH/H2O. Update after section 6.2.

| ID | (H K L) | Model I (e²) | Change H2O → OH | Role |
|---|---|---|---|---|
| P1 | (0 1 1.1) | ~20 | −74 % (H2O brighter) | Largest effect, weak |
| P2 | (0 1 2.85) | ~570 | −23 % (H2O brighter) | Pair with P3 |
| P3 | (0 1 3.16) | ~500 | +24 % (OH brighter) | Pair with P2 |
| P4 | (0 5 2.82) | ~230 | −14 % (H2O brighter) | Pair with P5 |
| P5 | (0 5 3.19) | ~175 | +16 % (OH brighter) | Pair with P4 |
| P6 | (0 3 4.79) | ~160 | −17 % | Brighter backup |
| C1 | (0 0 3.0) | ~1210 | ≈ 0 % | Control: film and surface roughness |
| C2 | (0 2 3.0) | ~570 | ≈ 0 % | Control |

All points are on H = 0 rods. In the simulator's thickness study, P1, P2, P4, P5 and P6 keep their size and sign from 2 to 10 nm at any degree of [001] relaxation. P3 stays positive but drops to about +12 % for relaxed 6 to 8 nm films. The previous H ≠ 0 pair, (1 0 4.18) and (1 2 4.21), changes size and even sign once the top relaxes, so it is dropped.
| F1 | (0 0 L) scan, L = 3.9 to 4.2 | bright | - | Film Bragg position vs potential (bulk lattice change) |

Reading the results:
- **P2 vs P3 and P4 vs P5 move oppositely:** consistent with a CUS species change.
- **All points move the same way:** background, beam, drift or damage.
- **C1/C2 move:** the film changed (roughening, dissolution, lattice). Check F1 and 00L fringes.
- **F1 shifts with potential:** a bulk lattice change (e.g. proton insertion) is happening. It will contaminate the "surface" points; model it.
- Bridging-O protonation and Ru_cus relaxation also change with potential. A response at the P points is **consistent** with OH/H2O exchange, not proof of it.

### 8.2 Potential list (fill from your CV)

| Label | E vs RHE (V) | CV feature | Expected CUS state (hypothesis) | Hold time before X-rays |
|---|---|---|---|---|
| E_ref | | lowest stable, double layer | H2O-rich | |
| E1 | | after 1st redox feature | | |
| E2 | | between features | | |
| E3 | | after 2nd redox feature | | |
| E4 | | just below OER onset | OH/O-rich | minimal |
| E_ref (return) | | reversibility check | should match first E_ref | |

- Do not go below the lowest potential you cycled in the lab without changes.
- Keep time at E4 short. RuO2 dissolves at OER potentials (fastest in acid).

### 8.3 Per-potential procedure

1. Step the potential. Wait until the current decays (e.g. below 5 % of the initial transient, or a fixed 2 to 5 min). Log the current.
2. C1.
3. P1 to P6 (count to the targets in section 10).
4. C2.
5. F1.
6. Optional: a short L segment of (0 1 L), 0.6 to 1.6, around P1.

### 8.4 X-ray voltammetry (fixed q, potential sweep)

- Sweep 1 to 2 mV/s from E_ref to E4 and back while counting at one point.
- Priority: P3, P2, P1, C1. Two to three cycles each, averaged.
- Record current simultaneously. Features in I(E) should line up with CV peaks.
- This gives reversibility and hysteresis, which the potential steps cannot.

### 8.5 Full rods at key potentials

- At 2 to 3 potentials (E_ref, one mid potential, E4 or just below): (0 1 L), (1 0 L), (1 2 L), and a fast 00L.
- End with E_ref (return): C1, P1 to P6 and a 00L segment (L 3.4 to 4.6) to check reversibility and film integrity.

---

## 9. Health checks and beam damage

- C1 and C2 after every potential change and roughly every hour.
- 00L segment (L 3.4 to 4.6) at the start, after the potential series, and at the end. Watch the fringe contrast, film peak position and intensity.
- If the cell allows lateral translation: measure P3 on a fresh spot and on the exposed spot at the same potential. A difference means beam damage (radiolysis, local pH change).
- Compare the in-cell CV before and after X-ray exposure.
- Log bubbles, leaks, electrolyte refills and any change in the background level.

---

## 10. Counting statistics

With signal S (net counts) and background B (counts in an ROI of equal size), the relative statistical error is σ ≈ √(S + 2B) / S.

To detect a fractional change Δ between two potentials at 3σ, you need σ ≤ Δ / (3√2). The required net counts per point are:

| Δ | Example point | Net counts (B ≈ 0) | Net counts (B ≈ S) |
|---|---|---|---|
| 0.73 | P1 | ~35 | ~100 |
| 0.25 | P2 to P5 | ~290 | ~860 |
| 0.18 | P6 | ~560 | ~1700 |
| 0.10 | | ~1800 | ~5400 |
| 0.05 | even rods | ~7200 | ~22000 |

- Counting better than the systematic floor (section 7.3) does not help for absolute rods. For fixed-q potential series the floor is lower (same geometry, same spot), usually limited by drift, so repeat C1 often.
- Time scaling: t_odd ≈ t_even × (I_even / I_odd), using the relative intensities in section 2. Cap it and accept larger errors at deep minima.

---

## 11. Data reduction notes

- Integrated intensity from the area detector: signal ROI plus background ROIs on both sides. Keep the raw images.
- Normalize by the incident monitor and attenuator factors.
- Apply the geometric corrections for your diffractometer geometry: polarization, Lorentz, active area (footprint and detector acceptance), rod interception. See Vlieg, J. Appl. Cryst. 30, 532 (1997) and Schlepütz et al., Acta Cryst. A 61, 418 (2005).
- Convert to |F|. Merge symmetry equivalents. Error bar = max(statistical, systematic floor from 7.3).
- Fixed-q potential data: report I(E)/I(E_ref) per point. This cancels most geometric corrections.
- Fit even and odd rods together at each potential. Fix film parameters from E_ref, then let only surface parameters vary with potential, unless F1 shows bulk changes.

---

## 12. Repeating for other thicknesses

| Changes with thickness | Action |
|---|---|
| Fringe period | Adjust the L step (table in section 2) |
| Film Bragg peak width | Thinner = broader, harder to separate from TiO2 |
| [001] relaxation | Onset ~4 nm; [1-10] strain persists; cracking ~17 nm (max film here 10 nm). Run section 6.1 on every film and set the commensurate thickness and relaxation in the simulator |
| Usable rods | ≤ ~4 nm: all rods. > ~4 nm: rely on H = 0 rods (0 K L) for chemistry; H ≠ 0 rods only for relaxation analysis |
| Film peak L | Re-derive P2/P3, which sit next to film Bragg peaks |
| Surface sensitivity | Broad odd-rod features (e.g. P1) are roughly thickness-independent; near-Bragg points are not |

Keep identical across thicknesses:
- Electrolyte, pH, reference electrode, pre-cycling, potential list, hold times.
- Point list IDs (update positions, keep labels).
- Incidence angle and counting targets.

Useful cross-check: a pure surface process gives the same potential response at P1 (away from Bragg peaks) for every thickness. A response that scales with thickness, or F1 shifts, points to a bulk or subsurface process.

---

## 13. Time budget (fill in after the first rod)

| Block | Points | s/point | Estimate (h) | Actual (h) |
|---|---|---|---|---|
| Setup, CV, alignment, UB | - | - | 1.5 to 3 | |
| Relaxation + strain checks (6.1, 6.2) | ~350 | | | |
| Even rods (7.1), 4 rods | ~4 × 350 | | | |
| Odd rods (7.2), 4 rods | ~4 × 130 | | | |
| Equivalents (7.3) | ~150 | | | |
| Potential series (8.3), 6 potentials | ~6 × 10 | | | |
| X-ray voltammetry (8.4), 4 points | - | - | | |
| Full rods at key potentials (8.5) | | | | |
| Health checks (9) | | | | |
| **Total** | | | | |

If time runs short, the priority order is: 6.1 → 00L → (0 1 L) → section 8.1 to 8.3 → (1 0 L) → remaining even rods → 8.4 → remaining odd rods → equivalents.

---

## 14. Log template

| Scan # | Time | Rod / point | L range | Potential (V) | Current (µA) | Spot (x, y) | Attenuator | Counting (s) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

Also log: electrolyte refills, bubbles, beam dumps and refills, cell leaks, any realignment.

---

## 15. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| No separate film peak on 00L | Expected for a commensurate film (peak within ~0.01 of TiO2) | Use the fringes; a resolved film peak is the warning sign (section 6.2) |
| Extra peak near H ≈ 1.90 or K ≈ 2.045 | Partial relaxation | Estimate the fraction; flag; measure the relaxed rods separately if needed |
| Rod ROI drifts along L | Miscut or UB error | Redo UB; check miscut; use rocking scans |
| Odd rods at background level | Electrolyte path too long, bubbles, low flux | Thinner electrolyte layer; prioritize P points over full rods |
| All P points and C1 change together | Drift, beam damage, film change | Check F1 and 00L fringes; fresh spot; recheck alignment |
| E_ref (return) does not match the first E_ref | Irreversible change (dissolution, roughening) | Reduce time at E4; note the history for the fit |
| F1 shifts with potential | Bulk lattice change | Include ε⊥(E) in the model; do not assign it to CUS species |

---

## References (starting points)

- I. K. Robinson, Phys. Rev. B 33, 3830 (1986): CTR theory.
- E. Vlieg, J. Appl. Cryst. 30, 532 (1997): integrated intensities and corrections.
- C. M. Schlepütz et al., Acta Cryst. A 61, 418 (2005): pixel detector data collection.
- R. R. Rao et al., Energy Environ. Sci. 10, 2626 (2017) and Nat. Catal. 3, 516 (2020): in situ/operando SXRD on RuO2(110) surfaces vs potential. Useful for potential ranges and adsorbate geometries; verify details.

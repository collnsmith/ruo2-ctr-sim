# Independent review: conclusions from the recorded data only (2026-11-16)

## Sources

**Used:**
- the SPEC file (scans 1-544);
- the ROI sums;
- SPEC's own detector ROIs (Eiger1-4);
- the EC-Lab files (beamline days 1-4, bench 2026-11-09);
- the processed CTR datasets;
- the work-function / PME table;
- the logbook, as the record kept during the beamtime.

**Left out** (statements made in conversation, not recorded data):
- continuity checks and the absence of bubbles;
- the meter check of polarization;
- the 100 Ohm result;
- the beamline potentiostat model;
- how EC-Lab behaves with a disconnected lead;
- "the potential ran".

## 1. Robust (no model needed)

1. **3.3 and 7 nm: a reversible change of the X-ray intensity locked to the applied potential**, at the reflections
   expected to be surface-sensitive.
   - Timing comes from the instrument clocks. The offset that best aligns the response with the potential is -12 to +15 s,
     independent of any hand-marked start.
   - **Same sign pattern in both films:** L 2.75 falls and L 3.16 rises. A beam, absorption or normalisation artefact
     would move every reflection the same way.
   - **Independent of how the ROIs were drawn:**
     - 3.3 nm: SPEC's own Eiger1-3 show the same signs (z 5-21).
     - 7 nm: Eiger2, the SPEC ROI covering that spot, shows it (z 8-12; 316 not significant).
   - **Controls:** OH-blind control rods stay at their noise (7 nm (2 0 1.75) 0.2 %; 3.3 nm 1.1-3.5 % against noise
     2.4-4.0 %).
   - **Reversible, rate-independent, reproducible:**
     - the anodic and cathodic sweeps overlap;
     - the swing is the same at 5 and 25 mV/s (222);
     - 7 nm (0 1 2.75) gives the same response four times.
2. **Potential control on days 2 and 3 is shown by the EC record:**
   - capacitive loops of 85-230 uF;
   - charge at potential steps, 28.5 +- 0.6 uF/V fast part on 7 nm;
   - charging at every staircase step (224, 225).
3. **The switch potential is the same in both films:** half-way 0.24 +- 0.04 V (3.3 nm) and 0.20 +- 0.02 V (7 nm) vs
   Ag/AgCl, about 0.45-0.48 V RHE. Difference 0.04 +- 0.04 V.
4. **The 7 nm CTRs follow the potential:** 0.7 vs -0.1 V rods differ by rms 0.077 / 0.080 (both directions), twice the
   repeats at one potential (0.039 / 0.040).

## 2. Supported, model-dependent

5. **The CTR model attributes the change to OH on the CUS sites.**
   - The 7 nm OH fraction rises from -0.1 to 0.35 V under both bridging-O assumptions (Rao; 0 Å).
   - The fitted states predict the sign and onset of the CV changes at L 2.75 and 3.16 up to 0.35 V. They do not
     reproduce L 3.25 or the size at 0.7 V.
   - Absolute coverage is not determined: the OH fraction and the bridging-O height trade off.
   - The fit shares three surface parameters with the 4.1 nm data.
6. **Strain:** the 7 nm film is partly relaxed (eps_perp 0.5 % against 1.4 % for 4.1 nm). The 3.3 nm strain is not
   measured: its rod (153-221) does not determine a structure. "3.3 nm fully strained" is an inference from it being
   thinner than the fully strained 4.1 nm film.
7. **Work function and PME** (table, other samples): the PME falls with the work function (slope -0.86 +- 0.23 V/eV). A
   switch that moved like the PME would sit 0.26 V lower on 3.3 nm than on 7 nm. It sits 0.04 +- 0.04 V higher (7.6
   sigma), so the switch does not follow the PME. Two caveats: the samples differ, and the PME reference is assumed
   Ag/AgCl.

## 3. The 4.1 nm film: what the data allow

8. **X-ray: no potential-locked change in any channel.**
   - This covers the ROIs, SPEC's Eiger ROIs, the CTRs at six potentials, and a deep search judged against the beam
     monitors.
   - If controlled, the response is at most 1-12 % of the other films'.
   - The CTR fits place the surface in the high-potential (OH-rich) state at every potential.
9. **EC: no charge passed through the WE lead in any day-4 file.**
   - Capacitance under 0.1 uF.
   - Step charge 0.003 +- 0.026 uF/V.
   - No difference between holds.
   - Ewe followed the programme.
10. **The bench files show that Ewe follows the programme with zero recorded current in two wirings:**
    - a WE also tied to ground (the cell polarized, by circuit physics);
    - a fully disconnected WE.

    The potential record therefore does not show that the film was in the circuit.
11. **The only recorded sign of a connected electrode is the logbook OCV notes:** 0.37-0.40 V, "stable at 0.378 before
    (exposure increases ocv)". The disconnected bench WE started at -0.345 V; its drift is not in the file.
12. **On the recorded data alone, an unpolarized 4.1 nm film is the simpler reading.**
    - One cause explains the missing charge, the flat X-ray, and a surface in the state expected at its rest potential
      (0.37-0.40 V, above the 0.2 V switch).
    - Control would need an unrecorded bypass to ground. The bench reproduces such a bypass, but nothing records it at
      the beamline.
    - Day 4 also differs: Kapton window, a rest potential 0.4 V higher than days 2-3 (-15 / -20 mV), no argon logged,
      no EC problem logged.
    - A weak nA current trend in the day-4 CVs (+1 to +4 nA over 0-0.7 V) is larger than on the disconnected bench run
      (+0.1 nA). It hints at a small real leak, but the instrument differs.
    - **Verdict: control not established.** The 4.1 nm null cannot carry a film-difference claim on these data alone.

## 4. Other findings

13. **3.3 nm vs 7 nm at L 3.25: opposite signs.** Not interpretable:
    - CV 125 used an older orientation matrix, at about half the rod intensity;
    - 3.25 sits on the steepest part of the 3.3 nm rod;
    - the model gives that sign only for a few structures.
14. **3.3 nm, other scans:**
    - 147 (L 1.25) is hysteretic, the only scan with a lag;
    - the staircases 224 / 225 disagree with the CVs. They were recorded after the window broke, with a bubble near
      the surface (logbook).
15. **Steady anodic current on days 2-3:** uA to tens of uA, net anodic every cycle. It is not driven by the beam at the
    moment (unchanged at 3.6 % transmission). Its source is unknown. It does not affect the switch potential (iR under
    1 mV there).
16. **Beam and cell history (logbook):**
    - beam damage on day 1, with a warning against unattenuated time scans;
    - the window broke on days 1 and 2;
    - the 7 nm (2 0) and (0 2) rods left the ROI.

## 5. Bottom line

- **Robust result:** two RuO2(110) films (3.3 and 7 nm) show a reversible, potential-locked, surface-specific structural
  switch at the same potential, about 0.47 V RHE. The CTRs and the model link it to OH on the CUS sites.
- **Insensitivity:** the switch potential does not depend on thickness between 3.3 and 7 nm. It does not follow the
  work-function or PME trend of the reference films. The 3.3 nm strain itself is inferred, not measured.
- **4.1 nm:** the data neither show nor exclude potential control. The recorded data lean to no control. The film is
  best shown as a negative control: no signal from cell, beam or method.
- **Where this differs from the conversation's conclusion:** only on 4.1 nm control. The left-out statements (continuity,
  no bubbles, meter-checked polarization, instrument model) would shift the balance toward control if accepted.

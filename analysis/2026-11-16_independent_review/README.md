# Independent review: conclusions from the recorded data only (2026-11-16)

This review takes a step back from the earlier analysis (`../2026-11-06_stationary_cv`, kept unchanged, including its
first-pass `independent_review.md`) and draws conclusions only from what was recorded. It re-checks that analysis's
numbers and adds four new checks:
- the day-1 stationary CVs;
- the day-1 current, before and after the window was raised;
- the 25 mV/s part of scan 222;
- the step from rest to 0 V on day 4.

It also corrects six earlier statements (section 5).

**Files:**
- `day1_cv.py` → `day1_cv.json`: the day-1 CV scans 91 and 94.
- `checks.py` → `checks.json`: every other number added here.

Both scripts only read the earlier folder (its data, scripts and saved results). Run them from anywhere:

```bash
python analysis/2026-11-16_independent_review/day1_cv.py
```

```bash
python analysis/2026-11-16_independent_review/checks.py
```

## Sources

**Used:**
- the SPEC file (scans 1-544), including SPEC's own detector ROIs (Eiger1-4, Eiger_Max) and beam monitors;
- the ROI sums;
- the EC-Lab files (beamline days 1-4; bench 2026-11-09);
- the processed CTR datasets;
- the work-function / PME table;
- the logbook, as the record kept during the beamtime.

**Not recorded:**
- the counter-electrode potential (Ece) in EC-Lab;
- anything on SPEC's spare analog inputs (keith_v, amptek): 0 in all 516 scans that have them.

So there is no second record of the cell beyond Ewe, I and the logbook.

**Left out** (statements made in conversation, not recorded data):
- continuity and resistance checks;
- the absence of bubbles;
- the meter check of polarization;
- the 100 Ohm result;
- the beamline potentiostat model;
- how EC-Lab behaves with a disconnected lead;
- "the potential ran".

## 1. Robust (no model needed)

1. **3.3 and 7 nm: a reversible change of the X-ray intensity locked to the applied potential**, at the reflections
   expected to be surface-sensitive.
   - **Timing comes from the instrument clocks.**
     - The offset that best aligns each scan with the potential is -12 to +15 s (median +4 s).
     - No hand-marked start is used.
   - **Same sign pattern in both films:** L 2.75 falls and L 3.16 rises. A beam, absorption or normalisation artefact
     would move every reflection the same way.
   - **The strictest test** (600 fake potential waves, beam signals as covariates, `deep41b.py` in the earlier folder):
     - 9 of the 11 scans sit at its floor (p = 0.002);
     - the exceptions are 341 (p = 0.045) and the noisy 343 (p = 0.14).
   - **Independent of how the ROIs were drawn:**
     - 3.3 nm: SPEC's own Eiger1-3 show the same signs (z 5-21).
     - 7 nm: Eiger2 shows the L 2.75 and 3.16 responses (z 8-12).
     - The 7 nm (0 1 3.25) response (+8 %) appears only in the drawn Ru ROI. No SPEC ROI shows it.
   - **A second day (`day1_cv.py`):** day 1, before the cell was rebuilt, has two 3.3 nm loopscans at (0 1 2.73).
     - SPEC's Eiger1 falls by 2.0 +- 0.5 % and 2.9 +- 0.1 % from 0 to 0.7 V (Eiger1-3 at p = 0.002 in the same test).
     - Their own best clock offsets are +1 and +3 s.
     - The monitor is flat.
     - The amplitude is small because SPEC's ROI held mostly other signal.
   - **Controls:** OH-blind control rods stay at their noise:
     - 7 nm (2 0 1.75): 0.2 %;
     - 3.3 nm: 1.1-3.5 %, against noise of 2.4-4.0 %.
   - **Reversible and reproducible:**
     - The anodic and cathodic sweeps overlap; 147 is the exception (item 14).
     - At 25 mV/s the swing is 81-86 % (+- 9-11 %) of that at 5 mV/s, in the same scan (222), depending on the clock
       offset used. Equal within 1.3-2.1 sigma: no strong rate dependence.
     - 7 nm (0 1 2.75) gives the same sign and switch potential four times.
     - Its size varies: 19-21 % in three scans of the summed ROIs, 36 % in 344. 344 was measured at a different
       incidence angle (logbook) with a different ROI split.
2. **Potential control on days 1-3 is shown by the EC record:**
   - day 1, scan 91: a charging current of 16 +- 1 uF and 1.6 uA/V. This is the smallest of days 1-3, recorded before
     the window was raised with more electrolyte;
   - days 2-3: capacitive loops of 85-230 uF;
   - charge at potential steps (28.5 +- 0.6 uF/V fast part on 7 nm);
   - charging at every staircase step (224, 225).
3. **No difference in switch potential between the films is resolved.**
   - Half-way potentials vs Ag/AgCl:
     - 3.3 nm: 0.24 +- 0.04 V (3 scans), 0.48 V RHE;
     - 7 nm: 0.20 +- 0.02 V (4 scans), 0.44 V RHE.
   - The difference is +0.04 +- 0.04 V. With so few scans, its 95 % interval is -0.09 to +0.17 V (Welch).
   - So the potentials are equal to within about 0.15 V. Smaller shifts are not resolved.
   - Day 1 gives 0.20 V (94) and 0.11 V (91, the weaker scan), from SPEC's ROI.
4. **The 7 nm CTRs follow the potential:** rods at 0.7 and -0.1 V differ by rms 0.077 / 0.080 (both directions). That
   is twice the difference between repeats at one potential (0.039 / 0.040).

## 2. Supported, model-dependent

5. **The CTR model attributes the change to OH on the CUS sites.**
   - The 7 nm OH fraction rises from -0.1 to 0.35 V under both bridging-O assumptions (Rao; 0 Å).
   - The fitted states predict the sign and onset of the CV changes at L 2.75 and 3.16 up to 0.35 V. They do not
     reproduce L 3.25 or the size at 0.7 V.
   - Absolute coverage is not determined: the OH fraction and the bridging-O height trade off.
   - The fit shares three surface parameters with the 4.1 nm data.
6. **Strain:**
   - The 7 nm film is partly relaxed: eps_perp 0.5 %, against 1.4 % for 4.1 nm.
   - The 3.3 nm strain is not measured: its rod (153-221) does not determine a structure.
   - "3.3 nm fully strained" is an inference from it being thinner than the fully strained 4.1 nm film.
7. **Work function and PME** (table; other samples):
   - In the table, the PME falls as the work function rises: slope -0.86 +- 0.23 V/eV, r = -0.82, 9 films, residual
     scatter 0.16 V.
   - A switch that moved like that would sit 0.26 +- 0.07 V lower on 3.3 nm than on 7 nm (the table's work functions
     differ by 0.30 eV). It sits 0.04 +- 0.04 V higher.
   - How far off that is depends on the assumptions:
     - 3.8 sigma with the slope's error included;
     - 2.6 and 1.6 sigma if our films' work functions scatter by 0.1 and 0.2 eV, like neighbouring KPFM samples.
   - So a PME-like shift is disfavoured, not excluded.
   - Caveats:
     - the work function of our own films was not measured;
     - the PME is assumed to be vs Ag/AgCl;
     - the falling relation is the opposite sign to the classic rise of a metal's zero-charge potential with its work
       function. So its origin matters before it is used to predict a shift.

## 3. The 4.1 nm film: what the data allow

8. **X-ray: no potential-locked change in any channel.**
   - The search covered the drawn ROIs, SPEC's ROIs and the CTRs at six potentials. It is calibrated against the beam
     monitors (`deep41.py`, `deep41b.py` in the earlier folder).
   - If the film was controlled, its CV response is at most 1-12 % of the other films'.
   - **The largest potential step of the day also gave nothing.** Each sequence started by stepping from rest (0.35-0.40 V)
     to 0 V. A 3.3 / 7 nm-like film loses about 90 % of its switch there.

     | Scan | Expected | Measured |
     |---|---|---|
     | 496 (L 2.75, after holds at 0.35 V) | +9 to +19 % | +1.1 +- 1.6 % |
     | 497 (L 2.75) | +9 to +19 % | -2.3 +- 0.6 % |
     | 498 (L 3.16) | -8 to -14 % | +1.2 +- 0.8 % |

   - The CTR fits place the surface in the high-potential (OH-rich) state at every potential.
9. **EC: no charge passed through the WE lead in any day-4 file.**
   - Capacitance -0.05 +- 0.05 uF.
   - Step charge 0.003 +- 0.026 uF/V.
   - The long holds at 0 and 0.7 V do not differ: 0.1 +- 0.7 nA/V.
   - Ewe followed the programme with the same noise as on day 3: Ewe minus programme sd 8-11 mV, against 8-12 mV.
   - The current range was the same.
   - The files carry the day-3 '7nm' names, so the settings were carried over.
10. **The bench files show that Ewe follows the programme with zero recorded current in two wirings:**
    - a WE also tied to ground (the cell polarized, by circuit physics);
    - a fully disconnected WE.

    So the potential record does not show that the film was in the circuit.
11. **The logbook OCV notes are the only recorded sign that the WE sense lead reached a wetted electrode:** 0.37-0.40 V
    through the day, "stable at 0.378", rising with beam exposure. A floating input would not hold one plausible value for
    hours or follow the beam. This argues against a simply disconnected lead.
12. **Two wiring faults reproduce the day-4 record. Neither is recorded.**
    - **(b) Polarized:** the WE was also tied to earth around the ammeter. The bench reproduces this. The film was then
      polarized, and at most about 0.02 % of a 3.3 nm-like current was recorded.
    - **(a) Not polarized:** a break of about 0.7 GOhm or more sat between the film and the CE / RE, or between the clip
      and the film. This was not bench-tested.
      - The programme then drops across the break, and the film stays at rest.
      - A 1e12 Ohm electrometer still reads a stable, beam-responsive OCV through it.
      - The holds put any resistive path above 0.7 GOhm. A break that large also hides the charging current.
      - **A thin layer alone is not enough.** Day 1 shows what one does. Before the window was raised with more
        electrolyte (logbook), scan 91 still drew a clear charging current (16 +- 1 uF, 1.6 uA/V), and its X-ray switch
        was visible. That is 5-50x below the later scans. Day 4 is more than 100x below 91, so (a) needs a near-complete
        ionic break (a dry gap, a cut-off compartment) or a failed contact.
    - **(a) explains the flat X-ray and the high-state CTRs with nothing else:** the rest potential (0.37-0.40 V) is above
      the 0.2 V switch. **(b) needs, in addition,** a 4.1 nm surface that does not switch anywhere from -0.24 to 0.7 V.
      The films on either side of it in thickness do switch.
    - **The one recorded item that leans the other way is weak.**
      - Within each CV the current rises by about +1 to +4 nA from 0 to 0.7 V (5.6 +- 1.3 nA/V).
      - One resistive break would give the same slope in the CVs and the holds. The holds give 0.1 +- 0.7 nA/V, a
        3.7-sigma tension with (a).
      - A leaked fraction under (b) can give it.
      - It sits at the resolution of an uncharacterised instrument, so it weighs little.
    - **Day 4 also differs:**
      - a rest potential of 0.37-0.40 V, against -15 / -20 mV on days 2-3;
      - no EC problem logged;
      - the window material (Kapton) is named only on day 4. Whether it differs from days 1-3 is not recorded.
    - **Verdict: control not established. The recorded data lean weakly to no control.** The 4.1 nm null cannot carry a
      film-difference claim on these data alone.

## 4. Other findings

13. **3.3 nm vs 7 nm at L 3.25: opposite signs.** Not interpretable:
    - CV 125 used an older orientation matrix, at about half the rod intensity;
    - 3.25 sits on the steepest part of the 3.3 nm rod;
    - the model gives that sign only for a few structures;
    - the 7 nm value is seen only in the drawn ROI (item 1).
14. **3.3 nm, other scans:**
    - 147 (L 1.25) is hysteretic, the only scan with a lag;
    - the staircases 224 / 225 disagree with the CVs. They were recorded after the window broke, with a bubble near the
      surface (logbook).
15. **Steady anodic current on days 1-3:** net anodic every cycle. Mean CV currents range from 2.7 to 17 uA on 3.3 nm, and
    from 1.2 to 102 uA on 7 nm.
    - It is not driven by the beam at that moment: it is unchanged at 3.6 % transmission.
    - On day 1 it appeared only after the window was raised with more electrolyte. Mean current: -0.05 uA in scan 91,
      +27 uA in 94, rising from 7 to 52 uA over the sweep. This fits a newly wetted electrode other than the film, but
      the data cannot say which.
    - Its size, its variability and the low rest potentials (-15 / -20 mV vs Ag/AgCl, about 0.22 V RHE) together suggest
      a parasitic process, for example an exposed contact. Its source is unknown.
    - The cell resistance is not in the record, so its iR shift is not known. At 100 uA, each 100 Ohm shifts the film by
      10 mV, which is small against the +-0.15 V resolution of item 3.
16. **Beam and cell history (logbook):**
    - beam damage on day 1, with a warning against unattenuated time scans;
    - every CV from 222 on was unattenuated except 496, as were the day-1 scans. The responding 3.3 and 7 nm scans were
      among them, so an unattenuated beam does not by itself remove the switch;
    - the window broke on days 1 and 2;
    - the 7 nm (2 0) and (0 2) rods left the ROI.

## 5. Bottom line

- **Robust result:** two RuO2(110) films (3.3 and 7 nm) show a reversible, potential-locked, surface-specific structural
  switch near 0.46 V RHE.
  - The 3.3 nm film shows it on two days, before and after the cell was rebuilt.
  - The CTRs and the model link it to OH on the CUS sites.
- **Insensitivity:**
  - The switch potential does not differ measurably between 3.3 and 7 nm: within about 0.15 V at 95 %.
  - A shift as large as the PME trend of the reference films predicts is disfavoured (1.6-3.8 sigma).
  - The 3.3 nm strain is inferred, not measured.
- **4.1 nm:** the data neither show nor exclude potential control, and lean weakly to no control. The film is best shown
  as a negative control: no signal from cell, beam or method.
- **Corrections to statements made earlier in the conversation or in the earlier folder** (not edited there):

  | Topic | Earlier | Corrected |
  |---|---|---|
  | PME mismatch | 7.6 sigma | 1.6-3.8 sigma (7.6 counted only the switch error) |
  | Switch potentials | "the same" | within about 0.15 V at 95 % |
  | 25 mV/s swing | "the same" | 81-86 % +- 9-11 % (the logbook's 17 % vs 15 % compared scans 222 and 223, not rates) |
  | 7 nm L 3.25 | +8 % response | seen only in the drawn ROI |
  | Kapton window | a day-4 difference | named only on day 4 |
  | Argon | not logged on day 4 | not logged on days 1, 2 and 4 alike, so not a day-4 difference |

  Each earlier wording appears in one or more of: the first-pass review, the slide outlines ("day 4 differed (window,
  rest potential, gas)"), the one-page PDF and the conversation.
- **Where this differs from the conversation's conclusion:** only on 4.1 nm control. The left-out statements would remove
  the break in (a) and make (b) the leading reading, if accepted: continuity and resistance checks, no bubbles,
  meter-checked polarization, the instrument model.

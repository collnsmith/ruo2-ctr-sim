"""Checks behind the independent review (README.md here). Every number the review adds to the earlier analysis
(../2026-11-06_stationary_cv), recomputed from the data or from that folder's saved results. That folder is only read.
Writes checks.json here.

1. SPEC's own detector ROIs against the drawn ROIs, 3.3 / 7 nm CV scans (deep41.json step term; deep41b.json p against
   600 fake potentials with beam covariates).
2. The X-ray change when the potential steps from rest to 0 V at the start of each sequence (deep41.json 'ocv' term),
   against the change a 3.3 / 7 nm-like switch gives for the day-4 rest potential (0.35-0.40 V).
3. Scan 222 at 5 and 25 mV/s: logistic amplitude of each part (sync_abs.py timing), at the median offset and at the
   scan's own offset.
4. Switch potentials (curves_sync.json): difference and its Welch 95 % interval; the PME / work-function comparison with
   the slope error and an assumed scatter of our films' work functions.
5. The EC record per CV file (Ewe minus programme, current range codes, current mean and noise), the columns recorded per
   day (no Ece), and SPEC's spare inputs (keith_v, amptek) over every scan.
6. Day-4 current: long holds at 0 and 0.7 V and the CV slope G (ec_leak.json), and the tension of the CV slope with one
   resistive break (holds and CVs would share its slope).
7. The CV current of single CV files, with the model of ec_leak.py (I = a + G (E - 0.37) + C v sign(dE/dt), block bootstrap):
   day 1 before (91) and after (94) the window was raised with more electrolyte (logbook), against day 2 (120) and day 4
   (496); the first 2 s left out (the start transient)."""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.normpath(os.path.join(HERE, "..", "2026-11-06_stationary_cv"))
src = open(os.path.join(OLD, "sync_abs.py")).read()
__file__ = os.path.join(OLD, "sync_abs.py")
exec(src.split('out = {"offset": {}')[0])                      # SPEC, SCANS, spec_times, ec_trace, xray, shapes, fit, hac
import numpy as np
from scipy.stats import t as student_t
from galvani import BioLogic

J = lambda name: json.load(open(os.path.join(OLD, name)))
out = {}

# 1. SPEC ROIs against the drawn ROIs
A, B = J("deep41.json")["partA"], J("deep41b.json")["xray"]
REF = [120, 125, 147, 222, 223, 311, 314, 316, 341, 343, 344]
t1 = {}
for n in REF:
    row = {}
    for ch in ("Ru+Ti", "ROI1", "ROI2", "Eiger1", "Eiger2", "Eiger3", "Eiger_Max"):
        v = A.get(f"{n}|{ch}")
        if not v:
            continue
        a, e = v["step"]
        b = B.get(f"{n}|{ch}|beam-cov|step")
        row[ch] = dict(A=a, err=e, z=a / e, p_fake=b["p"] if b else None)
    t1[n] = dict(film=A[f"{n}|ROI1"]["film"], hkl=A[f"{n}|ROI1"]["hkl"], channels=row)
out["spec_rois"] = t1
floor = [n for n in REF if t1[n]["channels"]["Ru+Ti"]["p_fake"] < 0.003]
out["fake_null_floor"] = dict(at_floor=floor, others={n: t1[n]["channels"]["Ru+Ti"]["p_fake"] for n in REF if n not in floor})

# 2. rest -> 0 V step at the start of each sequence
S = J("sync_abs.json")["summary"]
s = lambda x: 1 / (1 + np.exp(-(x - 0.23) / 0.057))
g = lambda x: (s(x) - s(0.0)) / (s(0.7) - s(0.0))
full = {L: {f: S[L]["mean"][f][0] for f in ("3.3", "7")} for L in ("2.75", "3.16")}
expect = {L: sorted(-g(Er) * full[L][f] for Er in (0.35, 0.40) for f in ("3.3", "7")) for L in full}
t2 = {}
for n in (496, 497, 498, 501, 503, 519):
    v = A[f"{n}|Ru+Ti"]
    L = v["hkl"].split()[-1]
    t2[n] = dict(hkl=v["hkl"], jump=v["ocv"][0], err=v["ocv"][1],
                 expected=[min(expect[L]), max(expect[L])] if L in expect else None)
out["rest_to_0V"] = dict(g_at_rest={str(Er): float(g(Er)) for Er in (0.35, 0.38, 0.40)}, scans=t2)

# 3. scan 222 at 5 and 25 mV/s
DOFF = J("sync_abs.json")["offset_used"]["d"]
own = J("sync_abs.json")["offset"]["222"]["d"]
film, hkl, ru, ti, folder, pat, a_, b_ = SCANS[222]
y0 = xray(222, ru, ti); ts = spec_times(222)[: y0.size]; y0 = y0[: ts.size]
start, te, Ee = ec_trace(folder, pat)
t3 = {}
for tag, d in (("median offset", DOFF), ("own offset", own)):
    tt = np.array([(x - start).total_seconds() for x in ts]) - d
    E = np.interp(tt, te, Ee, left=np.nan, right=np.nan)
    rate = np.interp(tt, te, np.abs(np.gradient(Ee, te)))
    r = {}
    for name, lo, hi in (("5 mV/s", 0.003, 0.008), ("25 mV/s", 0.015, 0.04)):
        m = np.isfinite(E) & (rate > lo) & (rate < hi)
        yy = y0[m] / np.mean(y0[m][E[m] < 0.05])
        gg = shapes(E[m])["logistic"]
        amp, u = fit(tt[m], yy, gg)
        r[name] = dict(A=float(amp), err=float(hac(tt[m], u, gg)), n=int(m.sum()))
    q = r["25 mV/s"]["A"] / r["5 mV/s"]["A"]
    qe = abs(q) * np.hypot(r["25 mV/s"]["err"] / r["25 mV/s"]["A"], r["5 mV/s"]["err"] / r["5 mV/s"]["A"])
    t3[tag] = dict(offset_s=d, parts=r, ratio=float(q), ratio_err=float(qe))
out["rate_222"] = t3

# 4. switch potentials, PME / work function
c = J("curves_sync.json")["scans"]
v33 = np.array([c[str(n)]["raw"]["halfway"] for n in (120, 222, 223)])
v7 = np.array([c[str(n)]["raw"]["halfway"] for n in (311, 314, 341, 344)])
se33, se7 = v33.std(ddof=1) / np.sqrt(v33.size), v7.std(ddof=1) / np.sqrt(v7.size)
diff, ediff = v33.mean() - v7.mean(), float(np.hypot(se33, se7))
df = ediff ** 4 / (se33 ** 4 / (v33.size - 1) + se7 ** 4 / (v7.size - 1))
q975 = student_t.ppf(0.975, df)
T = np.array([1.7, 2.8, 3.3, 5.0, 6.4, 6.5, 7.4, 13.9, 15.2])
WF = np.array([4.276, 4.362, 4.841, 4.78, 4.573, 4.551, 4.528, 4.915, 4.972])
PME = np.array([0.831, 0.737, 0.441, 0.131, 0.487, 0.481, 0.731, 0.487, 0.081])
p, cov = np.polyfit(WF, PME, 1, cov=True)
dwf = 4.841 - np.interp(7.0, T, WF)
exp_shift, e_slope = p[0] * dwf, np.sqrt(cov[0, 0]) * dwf
sig = {f"{ewf:.2f}": float((diff - exp_shift) / np.sqrt(ediff ** 2 + e_slope ** 2 + (p[0] * ewf) ** 2)) for ewf in (0.0, 0.1, 0.2)}
out["switch"] = dict(halfway_33=v33.tolist(), halfway_7=v7.tolist(), mean_33=float(v33.mean()), se_33=float(se33),
                     mean_7=float(v7.mean()), se_7=float(se7), diff=float(diff), diff_err=ediff, welch_df=float(df),
                     ci95=[float(diff - q975 * ediff), float(diff + q975 * ediff)], rhe_33=float(v33.mean() + 0.238),
                     rhe_7=float(v7.mean() + 0.238))
out["pme"] = dict(slope=float(p[0]), slope_err=float(np.sqrt(cov[0, 0])), r=float(np.corrcoef(WF, PME)[0, 1]),
                  resid_sd=float(np.std(PME - np.polyval(p, WF), ddof=2)), wf_diff=float(dwf), expected_shift=float(exp_shift),
                  expected_err_slope=float(e_slope), sigma_switch_error_only=float((diff - exp_shift) / ediff),
                  sigma_with_slope_and_wf_scatter=sig)

# 5. EC record and the spare SPEC inputs
ec = {}
cols = {}
for day in ("Collin_08092026", "Collin_08102026", "Collin_08112026", "Collin_08122026"):
    names = set()
    for f in sorted(glob.glob(os.path.join(OLD, "data", "echem", day, "*.mpr"))):
        d = BioLogic.MPRfile(f).data
        names |= set(d.dtype.names)
        if not f.endswith("_CV_C01.mpr"):
            continue
        Ew, ctl, I, rg = (np.asarray(d[k], float) for k in ("Ewe/V", "control/V", "<I>/mA", "I Range"))
        ec[os.path.basename(f)] = dict(day=day, ewe_minus_control_mV=float(1e3 * np.mean(Ew - ctl)),
                                       ewe_minus_control_sd_mV=float(1e3 * np.std(Ew - ctl)),
                                       ranges=sorted(int(x) for x in set(rg)), I_mean_nA=float(1e6 * I.mean()),
                                       I_sd_nA=float(1e6 * I.std()))
    cols[day] = sorted(names)
spare = {}
for k in ("keith_v", "amptek_r1", "amptek_r2"):
    vals = [np.max(np.abs(np.asarray(sc.data[k], float))) for sc in SPEC.values() if k in sc.data and len(sc.data[k])]
    spare[k] = dict(scans=len(vals), max_abs=float(max(vals)) if vals else None)
out["ec_files"] = ec
out["ec_columns"] = cols
out["ece_recorded"] = any("Ece" in c for v in cols.values() for c in v)
out["spec_spare_inputs"] = spare

# 6. day-4 current: holds against CV slope
L = J("ec_leak.json")
def wmean(rows):
    w = np.array([1 / r["sem_nA"] ** 2 for r in rows]); x = np.array([r["I_nA"] for r in rows])
    return float(np.sum(w * x) / w.sum()), float(1 / np.sqrt(w.sum()))
long_holds = [h for h in L["holds"] if h["n"] > 1000 and not h["file"].startswith("CA_7nm_scan483")]
h0 = wmean([h for h in long_holds if abs(h["E"]) < 0.01]); h7 = wmean([h for h in long_holds if abs(h["E"] - 0.7) < 0.01])
slope_h = ((h7[0] - h0[0]) / 0.7, float(np.hypot(h0[1], h7[1]) / 0.7))
G = [(v["G_nA_per_V"], v["G_err"]) for v in L["cv"] if v["film"] == "4.1"]
w = np.array([1 / e ** 2 for _, e in G]); Gm = float(np.sum(w * np.array([x for x, _ in G])) / w.sum()); Ge = float(1 / np.sqrt(w.sum()))
C41 = [(v["C_uF"], v["C_err"]) for v in L["cv"] if v["film"] == "4.1"]
wc = np.array([1 / e ** 2 for _, e in C41]); Cm = float(np.sum(wc * np.array([x for x, _ in C41])) / wc.sum()); Ce = float(1 / np.sqrt(wc.sum()))
out["day4_current"] = dict(hold_0V_nA=h0, hold_07V_nA=h7, hold_slope_nA_per_V=slope_h,
                           hold_slope_2sigma_upper=float(slope_h[0] + 2 * slope_h[1]),
                           break_resistance_lower_GOhm=float(1 / (slope_h[0] + 2 * slope_h[1])),
                           cv_slope_G_nA_per_V=(Gm, Ge), cv_capacitance_uF=(Cm, Ce),
                           tension_sigma=float((Gm - slope_h[0]) / np.hypot(Ge, slope_h[1])))
# 7. CV current of single files: day 1 before / after the window was raised, day 2, day 4
EC7 = {"day 1, 91 (before the raise)": "Collin_08092026/0p1M_H2SO4_5mVs_CV_timescan_91_C01.mpr",
       "day 1, 94 (after the raise)": "Collin_08092026/0p1M_H2SO4_5mVs_CV_timescan_RuPeak_scan94_C01.mpr",
       "day 2, 120": "Collin_08102026/0p1M_H2SO4_5mVs_CV_timescan_3p3nm_scan119_02_CV_C01.mpr",
       "day 4, 496": "Collin_08122026/CA_7nm_scan496_02_CV_C01.mpr"}
t7 = {}
for tag, f in EC7.items():
    d = BioLogic.MPRfile(os.path.join(OLD, "data", "echem", f)).data
    tt, ctl, Ew, I = (np.asarray(d[k], float) for k in ("time/s", "control/V", "Ewe/V", "<I>/mA"))
    I = I * 1e6                                                   # nA
    sl = np.sign(np.gradient(ctl, tt)); v = float(np.median(np.abs(np.gradient(ctl, tt)[sl != 0])))
    m = (sl != 0) & (tt > 2.0)
    X = np.c_[np.ones(m.sum()), Ew[m] - 0.37, sl[m] * v]
    coef = np.linalg.lstsq(X, I[m], rcond=None)[0]
    idx = np.array_split(np.arange(m.sum()), 30); rb = np.random.default_rng(0); bs = []
    for _ in range(200):
        pick = np.concatenate([idx[k] for k in rb.integers(0, 30, 30)])
        bs.append(np.linalg.lstsq(X[pick], I[m][pick], rcond=None)[0])
    e = np.std(bs, axis=0)
    t7[tag] = dict(I_mean_nA=float(I[m].mean()), a_nA=float(coef[0]), G_nA_per_V=float(coef[1]), G_err=float(e[1]),
                   C_uF=float(coef[2] / 1e3), C_err=float(e[2] / 1e3))
out["cv_current_single_files"] = t7
json.dump(out, open(os.path.join(HERE, "checks.json"), "w"), indent=1, default=float)

# ------------------------------------------------------------------ report
print("1. SPEC ROIs (step 0 -> 0.7 V, %, z, fake-potential p):")
for n, r in t1.items():
    print(f"   {n} {r['film']:>3} nm ({r['hkl']}): " + "  ".join(
        f"{ch} {100 * v['A']:+.1f} (z {abs(v['z']):.0f}, p {v['p_fake']:.3f})" for ch, v in r["channels"].items()
        if ch in ("Ru+Ti", "Eiger1", "Eiger2", "Eiger3")))
print("   at the fake-null floor:", floor, "others:", out["fake_null_floor"]["others"])
print("2. rest -> 0 V step, 4.1 nm (expected for a 3.3 / 7 nm-like switch, rest 0.35-0.40 V):")
for n, v in t2.items():
    ex = f"expected {100 * v['expected'][0]:+.1f} to {100 * v['expected'][1]:+.1f} %" if v["expected"] else "no reference"
    print(f"   {n} ({v['hkl']}): {100 * v['jump']:+.2f} +- {100 * v['err']:.2f} %   {ex}")
print("3. scan 222:", {k: f"5 mV/s {100 * v['parts']['5 mV/s']['A']:+.2f} +- {100 * v['parts']['5 mV/s']['err']:.2f} %, "
                          f"25 mV/s {100 * v['parts']['25 mV/s']['A']:+.2f} +- {100 * v['parts']['25 mV/s']['err']:.2f} %, "
                          f"ratio {v['ratio']:.2f} +- {v['ratio_err']:.2f} (offset {v['offset_s']:+.0f} s)" for k, v in t3.items()})
sw, pm = out["switch"], out["pme"]
print(f"4. half-way 3.3 nm {sw['mean_33']:.3f} +- {sw['se_33']:.3f} ({sw['rhe_33']:.2f} RHE), 7 nm {sw['mean_7']:.3f} +- {sw['se_7']:.3f} "
      f"({sw['rhe_7']:.2f} RHE); difference {sw['diff']:+.3f} +- {sw['diff_err']:.3f} V, Welch df {sw['welch_df']:.1f}, "
      f"95 % interval {sw['ci95'][0]:+.2f} to {sw['ci95'][1]:+.2f} V")
print(f"   PME vs WF slope {pm['slope']:.3f} +- {pm['slope_err']:.3f} V/eV (r {pm['r']:.2f}, residual sd {pm['resid_sd']:.2f} V); "
      f"WF difference {pm['wf_diff']:.3f} eV; expected shift {pm['expected_shift']:+.3f} +- {pm['expected_err_slope']:.3f} V; "
      f"off by {pm['sigma_switch_error_only']:.1f} sigma (switch error only), "
      + ", ".join(f"{v:.1f} (slope + WF scatter {k} eV)" for k, v in pm["sigma_with_slope_and_wf_scatter"].items()))
print("5. EC CV files (Ewe - programme sd mV, ranges, I mean / sd nA):")
for f, v in ec.items():
    print(f"   {v['day'][-8:]} {f[-40:]:40s} {v['ewe_minus_control_sd_mV']:5.1f}  {v['ranges']}  {v['I_mean_nA']:+10.1f} / {v['I_sd_nA']:8.1f}")
print("   Ece recorded:", out["ece_recorded"], "| SPEC spare inputs:", spare)
dc = out["day4_current"]
print(f"6. day-4 long holds: 0 V {dc['hold_0V_nA'][0]:+.2f} +- {dc['hold_0V_nA'][1]:.2f} nA, 0.7 V {dc['hold_07V_nA'][0]:+.2f} +- "
      f"{dc['hold_07V_nA'][1]:.2f} nA, slope {dc['hold_slope_nA_per_V'][0]:+.2f} +- {dc['hold_slope_nA_per_V'][1]:.2f} nA/V "
      f"(2-sigma upper {dc['hold_slope_2sigma_upper']:.2f}, break > {dc['break_resistance_lower_GOhm']:.2f} GOhm); CV slope G "
      f"{dc['cv_slope_G_nA_per_V'][0]:.1f} +- {dc['cv_slope_G_nA_per_V'][1]:.1f} nA/V; C {dc['cv_capacitance_uF'][0]:+.3f} +- "
      f"{dc['cv_capacitance_uF'][1]:.3f} uF; CV slope vs hold slope {dc['tension_sigma']:.1f} sigma")
print("7. CV current (I = a + G (E - 0.37) + C v sign(dE/dt)):")
for tag, v in t7.items():
    print(f"   {tag:30s} mean {v['I_mean_nA']:+9.1f} nA  G {v['G_nA_per_V']:+10.1f} +- {v['G_err']:7.1f} nA/V  C {v['C_uF']:+8.2f} +- {v['C_err']:6.2f} uF")

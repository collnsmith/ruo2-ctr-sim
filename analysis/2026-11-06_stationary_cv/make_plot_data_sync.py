"""Plot data for the clock-synced CV results and the EC checks (plot_data/11_cv_synced, plot_data/12_ec), from
sync_abs.json, curves_sync.json, beam_current.json and ec_leak.json. Run after those scripts."""
import csv, json, os
here = os.path.dirname(os.path.abspath(__file__))
PD = os.path.join(here, "plot_data")
HKL = {120: "0 -1 2.75", 125: "0 -1 3.25", 147: "0 -1 1.25", 222: "0 -1 3.16", 223: "0 -1 3.16", 311: "0 1 2.75",
       314: "0 1 3.16", 316: "0 1 3.25", 341: "0 1 2.75", 343: "0 1 2.75", 344: "0 1 2.75", 496: "0 -1 2.75",
       497: "0 -1 2.75", 498: "0 -1 3.16", 501: "0 -1 3.25", 503: "0 -1 2.84", 519: "-2 0 1.75"}


def write(rel, header, rows, comments):
    p = os.path.join(PD, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", newline="") as f:
        for c in comments:
            f.write(f"# {c}\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print("  ", rel)


s = json.load(open(os.path.join(here, "sync_abs.json")))
d = s["offset_used"]["d"]
rows = []
for n, v in sorted(s["response"].items(), key=lambda t: int(t[0])):
    h, k, l = HKL[int(n)].split()
    rows.append([n, v["film"], h, k, l, v["n_points"], round(v["cycles"], 2), round(v["logistic"]["A"], 5),
                 round(v["logistic"]["err"], 5), round(v["linear"]["A"], 5), round(v["linear"]["err"], 5)])
write("11_cv_synced/response.csv",
      ["scan", "film_nm", "H", "K", "L", "n_points", "cycles", "A_step", "err_step", "A_linear", "err_linear"], rows,
      ["Change of the X-ray signal (Ru + Ti ROIs) from 0 to 0.7 V, with the potential EC-Lab applied at each point's SPEC time",
       f"(clock offset {d:+.0f} s; the logged Start marks agree within 1-2 s). Fit per scan: I/I0 = quadratic drift + A g(E).",
       "A_step: g a step at 0.23 V (10-90 % over about 0.25 V); A_linear: g = E / 0.7 V. Fractions (0.1 = 10 %).",
       "err: spread of A when E(t) is shifted by 70-210 s (keeps the noise's own correlations). Made by sync_abs.py."])
rows = []
for L, v in s["summary"].items():
    for f, (a, e) in v["mean"].items():
        r33 = v["ratio_41_to"]["3.3"] if f == "4.1" else ("", "")
        r7 = v["ratio_41_to"]["7"] if f == "4.1" else ("", "")
        rows.append([L, f, round(a, 5), round(e, 5), *(round(x, 4) if x != "" else "" for x in (*r33, *r7))])
write("11_cv_synced/summary.csv", ["L", "film_nm", "A_step_mean", "err", "ratio_to_3p3", "ratio_err_3p3", "ratio_to_7", "ratio_err_7"],
      rows, ["Weighted mean of A_step per reflection and film (7 nm 2.75: 311, 341, 343; 344 at 0.6 deg incidence left out).",
             "ratio: the 4.1 nm mean over the 3.3 or 7 nm mean."])
c = json.load(open(os.path.join(here, "curves_sync.json")))
rows = [[n, v["raw"]["halfway"], v["detrended"]["halfway"], v.get("halfway_old_timing")] for n, v in c["scans"].items()]
write("11_cv_synced/halfway.csv", ["scan", "halfway_V_raw", "halfway_V_detrended", "halfway_V_old_timing"],
      [[x if x is not None else "" for x in r] for r in rows],
      ["Half-way potential (V vs Ag/AgCl) of each scan's cycle-averaged curve (sweeps averaged; 0 % = first bin, 100 % = mean of",
       "the last three bins), with the clock-synced potential; raw and with a quadratic drift removed; old timing from cv3.json.",
       "Not meaningful for scans without a response (4.1 nm). Made by curves_sync.py."])
b = json.load(open(os.path.join(here, "beam_current.json")))
for name, tag in (("3.3 nm 0.65 V (rod 153-221)", "beam_3p3nm_hold_065V"), ("7 nm CTR series (323-340)", "beam_7nm_ctr_series"),
                  ("7 nm even rods (346-354)", "beam_7nm_even_rods")):
    v = b[name]
    write(f"12_ec/{tag}_scans.csv", ["scan", "E_V", "I_uA", "transm_mean", "transm_min"],
          [[r["scan"], round(r["E"], 3), round(r["I"], 4), round(r["transm_mean"], 4), round(r["transm_min"], 4)] for r in v["scans"]],
          [f"{name}: mean EC current while each SPEC scan ran, and the scan's transmission (filters). Made by beam_current.py."])
    write(f"12_ec/{tag}_gaps.csv", ["after_scan", "E_V", "I_uA", "seconds"],
          [[g["after"], round(g["E"], 3), round(g["I"], 4), round(g["seconds"], 1)] for g in v["gaps"]],
          [f"{name}: mean EC current in the gaps between SPEC scans."])
e = json.load(open(os.path.join(here, "ec_leak.json")))
write("12_ec/day4_holds.csv", ["file", "E_V", "n", "I_nA", "sem_nA"],
      [[h["file"], h["E"], h["n"], round(h["I_nA"], 3), round(h["sem_nA"], 3)] for h in e["holds"]],
      ["4.1 nm (day 4): mean current of each potential hold (first 10 s left out), block standard error. Made by ec_leak.py."])
write("12_ec/cv_current_fits.csv", ["film_nm", "file", "G_nA_per_V", "G_err", "C_uF", "C_err"],
      [[x["film"], x["file"], round(x["G_nA_per_V"], 3), round(x["G_err"], 3), round(x["C_uF"], 4), round(x["C_err"], 4)] for x in e["cv"]],
      ["Each 5 mV/s CV: I = a + G (E - 0.37) + C v sign(dE/dt). Block-bootstrap errors. Made by ec_leak.py."])

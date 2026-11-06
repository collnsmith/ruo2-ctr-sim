"""Does the beam drive the EC current? The EC current during long holds against the beam on the sample, from the SPEC
file: for each SPEC scan inside a hold, the mean current while the scan runs and its transmission (filters), and the
mean current in the gaps between scans. Holds: 3.3 nm at 0.65 V during the rod 153-221 (filters in at 170, 206-208),
7 nm during the CTR series 323-340 and 346-354. Writes beam_current.json."""
import datetime as dt, glob, json, os, sys, warnings
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..")); sys.modules["xraydb"] = None
warnings.filterwarnings("ignore")
import numpy as np
from galvani import BioLogic
from ctrfit.datatools.specfile import parse_spec_file

S = parse_spec_file(os.path.join(here, "data", "08_9_3.3nm.spec"))
D0 = dt.datetime(2026, 10, 9, 13, 37, 24)
HOLDS = {"3.3 nm 0.65 V (rod 153-221)": "Collin_08102026/0p1M_H2SO4_5mVs_CA_Lscan_3p3nm_scan1_C01.mpr",
         "7 nm CTR series (323-340)": "Collin_08112026/0p1M_H2SO4_5mVs_CA00_Lscan_7nm_scan323_C01.mpr",
         "7 nm even rods (346-354)": "Collin_08112026/CA_7nm_scan345_C01.mpr"}
out = {}
for name, rel in HOLDS.items():
    m = BioLogic.MPRfile(os.path.join(here, "data", "echem", rel)); d = m.data
    t = np.array([(m.timestamp + dt.timedelta(seconds=float(x)) - D0).total_seconds() for x in d["time/s"]])
    I = d["<I>/mA"] * 1e3; E = d["control/V"]
    rows, gaps = [], []
    prev_end = None
    for n in sorted(S):
        s = S[n]
        if "Epoch" not in s.data or not s.npoints:
            continue
        a, b = float(s.data["Epoch"][0]) - 1.0, float(s.data["Epoch"][-1])
        if b < t[0] or a > t[-1]:
            continue
        inside = (t >= a) & (t <= b)
        if inside.sum() < 5:
            continue
        tr = s.data["transm"] if "transm" in s.data else np.ones(s.npoints)
        rows.append(dict(scan=n, E=float(np.median(E[inside])), I=float(np.mean(I[inside])), transm_min=float(np.min(tr)),
                         transm_mean=float(np.mean(tr)), t=float(a)))
        if prev_end is not None:
            g = (t > prev_end + 2) & (t < a - 2)
            if g.sum() >= 5:
                gaps.append(dict(after=rows[-2]["scan"] if len(rows) > 1 else None, E=float(np.median(E[g])), I=float(np.mean(I[g])),
                                 seconds=float(g.sum() * np.median(np.diff(t)))))
        prev_end = b
    out[name] = dict(scans=rows, gaps=gaps)
    print(f"\n{name}: {len(rows)} scans, {len(gaps)} gaps")
    for r in rows:
        flag = "  <-- filters" if r["transm_min"] < 0.99 else ""
        if flag or name.startswith("3.3"):
            pass
    # per potential level: mean current in scans with full beam, with filters, and in gaps
    for lev in sorted(set(round(r["E"], 2) for r in rows)):
        full = [r["I"] for r in rows if round(r["E"], 2) == lev and r["transm_min"] > 0.99]
        filt = [(r["scan"], round(r["transm_mean"], 3), round(r["I"], 2)) for r in rows if round(r["E"], 2) == lev and r["transm_min"] <= 0.99]
        gp = [g["I"] for g in gaps if round(g["E"], 2) == lev]
        print(f"  E {lev:+.2f} V: full beam {np.mean(full):+.2f} uA (scan-to-scan sd {np.std(full):.2f}, n {len(full)});"
              f" gaps {np.mean(gp) if gp else float('nan'):+.2f} uA (n {len(gp)}); filtered scans {filt}")
json.dump(out, open(os.path.join(here, "beam_current.json"), "w"), indent=1)

import json, sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
import numpy as np
from ctrfit.echem.external import parse_roi_sums
from ctrfit.echem.analysis import dominant_period

RAW, INFO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
E1, E2, RATE, CA = 0.0, 0.7, 0.005, 30.0
CYCLE = 2 * (E2 - E1) / RATE                     # 280 s per cycle

def info():
    rows = []
    for line in open(INFO):
        s = line.split("#")[0].split()
        if len(s) >= 10 and s[0].isdigit():
            note = line.split("#", 1)[1] if "#" in line else ""
            ru = "ROI1" if "ROI1 Ru" in note else "ROI2"
            rows.append(dict(scan=int(s[0]), H=int(s[1]), K=int(s[2]), L=float(s[3]), start=int(s[5]), stop=int(s[6]),
                             film=s[9], ru=ru, ti="ROI2" if ru == "ROI1" else "ROI1", note=note.strip()))
    return rows

def load(r):
    t = parse_roi_sums(open(f"{RAW}/ROIs_s{r['scan']}_sums.txt").read())[r["scan"]]
    g = lambda c: np.asarray(t[c], float)
    ru, ti = g(r["ru"]), g(r["ti"])
    err = lambda c, I: g(c + "_err") if c + "_err" in t else np.sqrt(np.maximum(np.abs(I), 1.0))   # no _err: sqrt(I)
    sru, sti = err(r["ru"], ru), err(r["ti"], ti)
    return ru, ti, sru, sti

def potential(n, n0, P):
    ph = ((n - n0) / P) % 1.0
    E = np.where(ph < 0.5, E1 + (E2 - E1) * 2 * ph, E2 - (E2 - E1) * (2 * ph - 1))
    return np.where(n < n0, E1, E), np.where(n < n0, 0, np.where(ph < 0.5, 1, -1))

rows = info()
res = {}
for r in rows:
    ru, ti, sru, sti = load(r)
    y = ru + ti
    n = np.arange(y.size, dtype=float)
    seg = slice(r["start"], r["stop"])
    P = dominant_period(n[seg], y[seg], min_period=40)
    res[r["scan"]] = dict(r, P=P, n=int(y.size))
pps = [v["P"] / CYCLE for v in res.values() if v["P"]]
print("points per second from the periodic scans:", np.round(pps, 3))
for s, v in res.items():
    print(f"  {s} {v['film']:>3s} nm ({v['H']} {v['K']} {v['L']}): period {v['P'] if v['P'] is None else round(v['P'], 1)} points, "
          f"Start->Stop {v['stop'] - v['start']} points")
json.dump({str(k): {kk: vv for kk, vv in v.items()} for k, v in res.items()}, open(OUT, "w"), indent=1, default=float)

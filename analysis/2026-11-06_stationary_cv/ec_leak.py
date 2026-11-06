"""Day-4 (4.1 nm) current: a resistive part (mean current of the CA holds at each potential; slope G in the CVs) and a
capacitive part (C in the CVs: I = a + G (E - 0.37) + C v sign(dE/dt)), against the 3.3 and 7 nm CVs. Writes ec_leak.json."""
import glob, json, os, numpy as np, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
RES = {"holds": [], "cv": []}
warnings.filterwarnings("ignore")
from galvani import BioLogic
def load(f):
    d = BioLogic.MPRfile(f).data
    I = d["<I>/mA"]*1e6 if "<I>/mA" in d.dtype.names else d["I/mA"]*1e6   # nA
    return d["time/s"], d["control/V"], d["Ewe/V"], I
def block_sem(x, nb=20):
    b = np.array_split(x, nb); m = np.array([v.mean() for v in b if len(v)])
    return m.std(ddof=1)/np.sqrt(len(m))
print("CA holds, 4.1 nm: mean current per control level (nA), block SEM")
for f in sorted(glob.glob("data/echem/Collin_08122026/*.mpr")):
    if "_CV_" in f: continue
    t, c, E, I = load(f)
    for lev in np.unique(np.round(c, 2)):
        s = np.round(c, 2) == lev
        s[np.flatnonzero(s)[:20]] = False      # drop the first 10 s after a change
        if s.sum() < 40: continue
        print(f"  {f.split('/')[-1][:32]:32s} {lev:+.2f} V  n={s.sum():5d}  I={I[s].mean():+7.2f} +- {block_sem(I[s]):.2f} nA")
        RES["holds"].append(dict(file=f.split("/")[-1], E=float(lev), n=int(s.sum()), I_nA=float(I[s].mean()), sem_nA=float(block_sem(I[s]))))
print("\nCVs: I = a + G*(E-0.37) + Cv*sign(dE/dt)*v  (v = 5 mV/s)")
for folder, tag in [("Collin_08122026","4.1"),("Collin_08112026","7"),("Collin_08102026","3.3")]:
    for f in sorted(glob.glob(f"data/echem/{folder}/*_CV_C01.mpr")):
        t, c, E, I = load(f)
        sl = np.sign(np.gradient(c, t)); v = np.median(np.abs(np.gradient(c, t)[sl != 0]))
        if not (0.003 < v < 0.007): continue
        m = sl != 0
        X = np.c_[np.ones(m.sum()), E[m]-0.37, sl[m]*v]
        coef, *_ = np.linalg.lstsq(X, I[m], rcond=None)
        r = I[m] - X@coef
        # block bootstrap errors
        nb = 30; idx = np.array_split(np.arange(m.sum()), nb); rng = np.random.default_rng(0); bs = []
        for _ in range(200):
            pick = np.concatenate([idx[k] for k in rng.integers(0, nb, nb)])
            bs.append(np.linalg.lstsq(X[pick], I[m][pick], rcond=None)[0])
        e = np.std(bs, axis=0)
        RES["cv"].append(dict(film=tag, file=f.split("/")[-1], G_nA_per_V=float(coef[1]), G_err=float(e[1]), C_uF=float(coef[2]/1e3), C_err=float(e[2]/1e3)))
        print(f"  {tag:>3} nm {f.split('/')[-1][-30:]:30s} G={coef[1]:8.1f}+-{e[1]:5.1f} nA/V   C={coef[2]/1e3:8.3f}+-{e[2]/1e3:.3f} uF")

json.dump(RES, open("ec_leak.json", "w"), indent=1)

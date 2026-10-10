"""Synthetic in-situ CV experiment with a known truth (fixed seed): a SPEC file with a stationary
scan (loopscan or a phi scan that does not move) and an EC-Lab-style .mpt, on two clocks that
differ by `clock_offset` seconds. The intensity follows the CTR model at the HKL along a
composition path, times a slow exponential decay, with counting noise.
"""
from datetime import datetime, timedelta

import numpy as np

from .ec import ECData, make_cv, write_mpt
from .predict import PointModel, simulate_path
from .spec import to_seconds

TRUTH = dict(hkl=(0, 1, 1.5), states=("H2O", "OH", "O"), E0=(0.8, 1.1), widths=(0.03, 0.04), cathodic_shift=0.03,
             theta=1.0, clock_offset=7.3, cv_delay=60.0, hold=300.0, upper=1.4, lower=0.4, rate=0.01, cycles=3,
             decay=0.05, tau=2500.0, counts=2.0e4, count_time=1.0, dead=0.1)


def make_experiment(kind="loopscan", seed=3, **over):
    """Returns (spec_text, ec ECData with the EC clock start, truth dict). kind: 'loopscan' or 'phi'."""
    tr = dict(TRUTH, **over)
    rng = np.random.default_rng(seed)
    file_date = datetime(2026, 3, 7, 14, 0, 0)
    file_epoch = 1772892000.0
    scan_start = file_date + timedelta(seconds=1800)
    cv0 = scan_start + timedelta(seconds=tr["cv_delay"])             # SPEC clock
    ec = make_cv(0.5, tr["hold"], tr["upper"], tr["lower"], tr["rate"], tr["cycles"], "up",
                 start=cv0 + timedelta(seconds=tr["clock_offset"]))  # EC clock
    total = tr["cv_delay"] + ec.time[-1] + 30
    step = tr["count_time"] + tr["dead"]
    n = int(total / step)
    t_end = (np.arange(n) + 1) * step                                # end of each count, s after the scan start
    t_mid = t_end - tr["count_time"] / 2
    t_ec = t_mid - tr["cv_delay"]
    V = np.interp(t_ec, ec.time, ec.potential, left=0.5, right=ec.potential[-1])
    dE = np.gradient(np.interp(t_ec, ec.time, ec.potential), t_ec)
    branch = np.where(dE > 1e-4, 1, np.where(dE < -1e-4, -1, 0))
    pm = PointModel(hkl=tr["hkl"])
    Imod = simulate_path(pm, V, list(tr["states"]), tr["E0"], tr["widths"], tr["theta"], branch, tr["cathodic_shift"])
    ref = pm.of_comp({tr["states"][0]: tr["theta"]})
    monitor = rng.normal(1.0e5, 1.0e3, n)
    decay = 1 + tr["decay"] * np.exp(-t_mid / tr["tau"])
    lam = tr["counts"] * Imod / ref * decay * monitor / 1e5 * tr["count_time"]
    det = rng.poisson(lam).astype(float)
    epoch = to_seconds(scan_start) - to_seconds(file_date) + t_end
    lines = [f"#F synthetic.spec", f"#E {file_epoch:.0f}", f"#D {file_date.strftime('%a %b %d %H:%M:%S %Y')}",
             "#O0 Two Theta  Theta  Chi  Phi  Nu  Mu", "",
             "#S 1  ascan  phi 10 30 10 1", f"#D {(scan_start - timedelta(seconds=600)).strftime('%a %b %d %H:%M:%S %Y')}",
             "#P0 20 10 90 15 0 0.5", "#N 3", "#L Phi  Epoch  Seconds  Detector"]
    lines += [f"{10 + 2 * i:g} {1000 + i:.3f} 1 {100 + i}" for i in range(11)]
    lines.append("")
    H, K, L = tr["hkl"]
    if kind == "loopscan":
        lines += [f"#S 2  loopscan {n} {tr['count_time']:g} 0", f"#D {scan_start.strftime('%a %b %d %H:%M:%S %Y')}",
                  "#P0 20 10 90 25 0 0.5", f"#Q {H:g} {K:g} {L:g}", "#N 5", "#L Time  Epoch  Seconds  Monitor  Detector"]
        lines += [f"{te:.3f} {ep:.3f} {tr['count_time']:g} {m:.0f} {d:.0f}" for te, ep, m, d in zip(t_end, epoch, monitor, det)]
    else:
        lines += [f"#S 2  ascan  phi 25 25 {n - 1} {tr['count_time']:g}", f"#D {scan_start.strftime('%a %b %d %H:%M:%S %Y')}",
                  "#P0 20 10 90 25 0 0.5", f"#Q {H:g} {K:g} {L:g}", "#N 8", "#L Phi  H  K  L  Epoch  Seconds  Monitor  Detector"]
        lines += [f"25 {H:g} {K:g} {L:g} {ep:.3f} {tr['count_time']:g} {m:.0f} {d:.0f}"
                  for ep, m, d in zip(epoch, monitor, det)]
    tr.update(n=n, scan_start=scan_start, cv0=cv0)
    return "\n".join(lines) + "\n", ec, tr


def write_experiment(folder, kind="loopscan", seed=3, **over):
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    text, ec, tr = make_experiment(kind, seed, **over)
    (folder / "insitu.spec").write_text(text, encoding="utf-8")
    write_mpt(folder / "cv.mpt", ECData(ec.time, ec.potential, np.zeros_like(ec.time), ec.cycle, ec.start))
    return folder / "insitu.spec", folder / "cv.mpt", tr

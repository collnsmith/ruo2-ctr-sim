"""Write tests/reference.npz: today's engine output that the regression tests pin.

Run from the repo root:  python tests/make_reference.py
Only rerun when a change to the physics is intended, and say so in the commit.
xraydb is blocked so the anomalous terms come from the built-in table on every machine.
"""
import sys
from pathlib import Path

sys.modules["xraydb"] = None
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ctr_engine import CTRModel, parse_inputs, relaxation_study  # noqa: E402
from ctr_params import DEFAULTS, parse_comp  # noqa: E402

RODS = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 5)]
CASES = {"default": {}, "relaxed": dict(thickness_nm=10.0, relax_001=0.5)}
OUT = Path(__file__).resolve().parent / "reference.npz"


def rod_L():
    return np.arange(0.3, 6.0 + 1e-9, 0.005)


def compute_rods():
    out = {"L": rod_L()}
    comp = parse_comp(DEFAULTS["comp_default"])
    for case, over in CASES.items():
        m = CTRModel(dict(DEFAULTS, **over))
        for H, K in RODS:
            out[f"{case}_I_{H}{K}"] = m.intensity(H, K, out["L"], comp)
            if case == "default":
                out[f"{case}_film_{H}{K}"] = m.film_intensity(H, K, out["L"], comp)
                out[f"{case}_sub_{H}{K}"] = np.abs(m.F_sub(H, K, out["L"])) ** 2
                out[f"{case}_domains_{H}{K}"] = m.intensity(H, K, out["L"], comp, mode="domains")
    return out


def compute_ranking(n=10):
    p = DEFAULTS
    inp = parse_inputs(p)
    m = CTRModel(p)
    _, peaks = m.sensitivity_scan(inp["sens_a"], inp["sens_b"], p["scan_h_max"], p["scan_k_max"], p["metric"],
                                  p["bragg_excl"], p["rel_min"], p["i_bg"])
    top = peaks[:n]
    return {f"rank_{k}": np.array([pk[k] for pk in top]) for k in ("H", "K", "L", "rel", "snr", "I")}


def compute_study():
    p = DEFAULTS
    inp = parse_inputs(p)
    st = relaxation_study(p, inp["study_points"], inp["study_thick"], inp["study_relax"], inp["sens_a"],
                          inp["sens_b"])
    keys = sorted(st, key=str)
    return {"study_keys": np.array([f"{lab}|{R!r}|{t!r}" for lab, R, t in keys]),
            "study_vals": np.array([st[k] for k in keys])}


def main():
    data = {}
    data.update(compute_rods())
    data.update(compute_ranking())
    data.update(compute_study())
    data["anom_src"] = np.array(CTRModel(DEFAULTS).anom_src)
    np.savez(OUT, **data)
    print(f"wrote {OUT} ({len(data)} arrays, anomalous terms: {data['anom_src']})")


if __name__ == "__main__":
    main()

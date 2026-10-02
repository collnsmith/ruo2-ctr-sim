"""Generate the example: synthetic |F| data for a 6.3 nm RuO2(110) film with 60 % OH on the CUS sites.

    python make_data.py      writes data.csv, model_true.json and model.json (the starting model)

Fixed seed, so the files are reproducible. The anomalous terms come from the built-in table.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.modules["xraydb"] = None                 # built-in anomalous table: same numbers on every machine

from ctrfit import Model, make_dataset        # noqa: E402
from ctrfit.core.settings import DEFAULTS     # noqa: E402

TRUTH = dict(thickness_mean_tl=19.4, roughness_tl=0.6, eps_perp=2.3, theta=0.85, x_OH=0.6, z_OH=2.05,
             z_H2O=2.6, scale=1.0)
RODS = [(0, 0), (1, 1), (0, 2), (2, 0), (0, 1), (1, 0), (0, 3), (0, 5)]
FREE = ["thickness_mean_tl", "roughness_tl", "eps_perp", "theta", "x_OH", "z_OH", "scale"]
START = dict(thickness_mean_tl=18.5, roughness_tl=0.9, eps_perp=1.7, theta=0.95, x_OH=0.35, z_OH=1.85, scale=1.2)
BOUNDS = dict(thickness_mean_tl=(15, 25), roughness_tl=(0.05, 2.0), eps_perp=(0.5, 4.0), z_OH=(1.6, 2.5),
              scale=(0.5, 2.0))


def main(out=HERE):
    out = Path(out)
    true = Model.from_settings(DEFAULTS)
    true.params.update(TRUTH)
    ds = make_dataset(true, rods=RODS, noise_rel=0.03, sys_floor=0.02, seed=2024, step=0.05,
                      name="synthetic_ruo2", energy_kev=DEFAULTS["energy_kev"], thickness_nm=6.3)
    ds.save(out / "data.csv")
    true.save(out / "model_true.json")
    start = Model.from_settings(DEFAULTS)
    start.params.update(dict(z_H2O=TRUTH["z_H2O"]))
    start.params.fit_only(FREE)
    start.params.update(START)
    for n, (lo, hi) in BOUNDS.items():
        start.params[n].min, start.params[n].max = lo, hi
    start.save(out / "model.json")
    print(f"wrote data.csv ({len(ds)} points on {len(RODS)} rods), model_true.json, model.json")


if __name__ == "__main__":
    main()

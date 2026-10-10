# @app title: Make in-situ CV example | group: Electrochemistry | order: 20 | kind: script | desc: Writes a synthetic SPEC file (loopscan and unmoving phi scan) and an EC-Lab .mpt with a known truth
"""Synthetic in-situ CV experiment for trying the In-situ CV window or `python -m ctrfit cv`.

Writes insitu.spec (scan 2: loopscan) and insitu_phi.spec (scan 2: phi scan that does not move),
cv.mpt (EC-Lab text export) and truth.txt. The potentiostat clock runs 7.3 s ahead of SPEC's, so in
'Absolute clocks + offset' the right offset is +7.3 s.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from ctrfit.echem.synthetic import write_experiment  # noqa: E402

if __name__ == "__main__":
    spec, mpt, tr = write_experiment(HERE, "phi")
    spec.rename(HERE / "insitu_phi.spec")
    spec, mpt, tr = write_experiment(HERE, "loopscan")
    keys = ("hkl", "states", "E0", "widths", "cathodic_shift", "clock_offset", "cv_delay", "hold", "lower", "upper",
            "rate", "cycles")
    (HERE / "truth.txt").write_text("\n".join(f"{k} = {tr[k]}" for k in keys) + "\n", encoding="utf-8")
    print(f"wrote {spec.name}, insitu_phi.spec, {mpt.name}, truth.txt in {HERE}")

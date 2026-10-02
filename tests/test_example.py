"""The example in examples/synthetic_ruo2 is reproducible and its fit recovers the truth."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

EX = Path(__file__).resolve().parents[1] / "examples" / "synthetic_ruo2"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"example_{name}", EX / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_example_files_are_reproducible(tmp_path):
    _load("make_data").main(tmp_path)
    for f in ("data.csv", "model.json", "model_true.json"):
        assert (tmp_path / f).read_text() == (EX / f).read_text(), f


@pytest.mark.slow
def test_example_fit_recovers_truth(tmp_path):
    res = _load("run_fit").main(out=tmp_path)
    truth = {p["name"]: p["value"] for p in json.loads((EX / "model_true.json").read_text())["parameters"]}
    for n in res.names:
        assert abs(res.values[n] - truth[n]) <= 2 * res.errors[n], n
    assert (tmp_path / "result.json").exists() and len(list(tmp_path.glob("rod_*.png"))) == 8
    assert np.isfinite(res.fom["red_chi2"])

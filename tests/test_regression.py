"""Regression: the engine must reproduce tests/reference.npz (never edit that file by hand)."""
from pathlib import Path

import numpy as np
import pytest

import make_reference as ref
from ctr_engine import CTRModel
from ctr_params import DEFAULTS

REF = np.load(Path(__file__).resolve().parent / "reference.npz")
RTOL = 1e-9


def test_reference_uses_builtin_table():
    assert str(REF["anom_src"]) == CTRModel(DEFAULTS).anom_src == "built-in table at 16 keV"


@pytest.fixture(scope="module")
def rods():
    return ref.compute_rods()


@pytest.mark.parametrize("key", [k for k in REF.files if k.startswith(("default_", "relaxed_"))])
def test_rod_intensities(rods, key):
    np.testing.assert_array_equal(rods["L"], REF["L"])
    np.testing.assert_allclose(rods[key], REF[key], rtol=RTOL, atol=0)


def test_sensitivity_top10():
    got = ref.compute_ranking()
    for k in ("H", "K"):
        np.testing.assert_array_equal(got[f"rank_{k}"], REF[f"rank_{k}"])
    for k in ("L", "rel", "snr", "I"):
        np.testing.assert_allclose(got[f"rank_{k}"], REF[f"rank_{k}"], rtol=RTOL, atol=0)


def test_thickness_study():
    got = ref.compute_study()
    np.testing.assert_array_equal(got["study_keys"], REF["study_keys"])
    np.testing.assert_allclose(got["study_vals"], REF["study_vals"], rtol=RTOL, atol=1e-15)

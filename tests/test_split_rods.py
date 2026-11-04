"""tools/split_rods.py: fitting CSVs split into one file per film, potential and rod."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ctrfit.data.dataset import Dataset

from test_scans import _file

TOOL = Path(__file__).resolve().parents[1] / "tools" / "split_rods.py"


def _tool():
    spec = importlib.util.spec_from_file_location("split_rods", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_split_by_film_potential_and_rod(tmp_path):
    a = _file("7 nm CA 0.5 V", 0.5, 1, scans=("#10", "#11"))          # rods 0 1 (#10) and 1 0 (#11)
    b = _file("7 nm CA 0.9 V", 0.9, 2, scans=("#20", "#21"))
    rep = _file("7 nm CA 0.5 V repeat", 0.5, 3, scans=("#30", "#31"))   # the same film and potential
    paths = []
    for i, d in enumerate((a, b, rep)):
        d.save(tmp_path / f"in{i}.csv")
        paths.append(tmp_path / f"in{i}.csv")
    written = _tool().split(paths, tmp_path / "out")
    names = sorted(p.name for p, _, _ in written)
    assert names == ["7_nm_CA_0.5_V_rod_0_1.csv", "7_nm_CA_0.5_V_rod_1_0.csv",
                     "7_nm_CA_0.9_V_rod_0_1.csv", "7_nm_CA_0.9_V_rod_1_0.csv"]
    d = Dataset.load(tmp_path / "out" / "7_nm_CA_0.5_V_rod_0_1.csv")
    assert d.rod_labels == ["0 1"] and d.scan_labels == ["#10", "#30"]            # the repeat joined its rod
    assert len(d) == len(a.scans()["#10"]) + len(rep.scans()["#30"])
    assert d.meta["potential_V"] == 0.5 and d.meta["film"] == "7 nm" and d.meta["thickness_nm"] == 7.0
    assert d.name == "7 nm CA 0.5 V (0 1)" and "scans #10, #30" in d.meta["notes"]
    np.testing.assert_allclose(d.F[: len(a.scans()["#10"])], a.F[a.scans()["#10"]])
    assert sum(n for _, n, _ in written) == len(a) + len(b) + len(rep)


def test_never_writes_over_an_input(tmp_path):
    a = _file("7 nm CA 0.5 V", 0.5, 1)
    clash = tmp_path / "7_nm_CA_0.5_V_rod_0_1.csv"          # an input with the name an output would get
    a.select_rods(["0 1"]).save(clash)
    with pytest.raises(ValueError, match="input"):
        _tool().split([clash], tmp_path)
    assert Dataset.load(clash).rod_labels == ["0 1"]


def test_command_line(tmp_path, capsys):
    a = _file("4.1 nm CA 0 V", 0.0, 4, film="4.1 nm")
    a.save(tmp_path / "f.csv")
    assert _tool().main([str(tmp_path / "f.csv")]) == 0
    assert "2 files in" in capsys.readouterr().out and (tmp_path / "split_by_rod").is_dir()

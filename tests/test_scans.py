"""Scans of the loaded files: per-point scan labels, leaving scans out, custom groups (fitted
together), parameters kept in step when the groups change, project round trip, and a synthetic
recovery with repeats grouped by potential."""
import numpy as np
import pytest

from ctrfit import make_dataset
from ctrfit.data.dataset import Dataset
from ctrfit.fit import workflow
from ctrfit.fit.fit import Fit
from ctrfit.fit.scans import ScanSelection, prune_scopes, regroup
from ctrfit.project.project import Project

from test_fit import ODD, truth_model


def _file(name, u, seed, x_OH=0.5, film="7 nm", scans=("#1", "#2"), **kw):
    """A synthetic file: two odd rods, one scan each (labels from `scans`)."""
    d = make_dataset(truth_model(x_OH=x_OH), rods=ODD[:2], noise_rel=0.03, seed=seed, step=0.1, name=name,
                     potential_V=u, film=film, thickness_nm=7.0, condition=f"CA {u:g} V", **kw)
    d.scan = np.where(d.rod == d.rod_labels[0], scans[0], scans[1])
    return d


def test_scan_labels_round_trip(tmp_path):
    d = _file("a", 0.5, 1)
    assert d.scan_labels == ["#1", "#2"] and sum(len(i) for i in d.scans().values()) == len(d)
    for ext in ("csv", "json"):
        d.save(tmp_path / f"a.{ext}")
        back = Dataset.load(tmp_path / f"a.{ext}")
        assert back.scan.tolist() == d.scan.tolist()
    assert "scan" in (tmp_path / "a.csv").read_text().splitlines()[-len(d) - 1]
    sub = d.subset(d.scan == "#2")
    assert sub.scan_labels == ["#2"]
    bare = Dataset(d.H, d.K, d.L, d.F, d.sigma, d.rod)          # no scan column: one scan per rod
    assert not bare.has_scans and bare.scan_labels == [f"rod {r}" for r in d.rod_labels]
    bare.save(tmp_path / "bare.csv")
    assert "scan" not in (tmp_path / "bare.csv").read_text()


def test_default_groups_are_the_files():
    sel = ScanSelection()
    a, b = _file("a", 0.5, 1), _file("a", 0.9, 2)
    sel.add(a)
    sel.add(b)
    assert b.name == "a (2)"                                    # a second file of the same name
    ds, notes = sel.build()
    assert [d.name for d in ds] == ["a", "a (2)"] and not notes
    assert len(ds[0]) == len(a) and ds[0].meta["potential_V"] == 0.5 and ds[0].scan_labels == ["#1", "#2"]
    np.testing.assert_array_equal(ds[1].F, b.F)
    assert len(sel.rows()) == 4 and sel.summary() == "4 of 4 scans fitted, in 2 dataset(s)"


def test_leave_out_and_group():
    sel = ScanSelection()
    sel.add(_file("A", 0.5, 1))
    sel.add(_file("A repeat", 0.5, 2, scans=("#3", "#4")))
    sel.add(_file("B", 0.9, 3, scans=("#5", "#6")))
    sel.set_use([sel.key("A", "#2")], False)
    ds, _ = sel.build()
    assert len(ds[0]) == len(sel.source("A").scans()["#1"])
    out = sel.left_out("A")
    assert out.scan_labels == ["#2"] and sel.left_out("B") is None
    sel.set_group(sel.keys_of("A repeat"), "A")                 # the repeat fitted with A
    ds, notes = sel.build()
    assert [d.name for d in ds] == ["A", "B"] and not notes
    assert ds[0].scan_labels == ["#1", "#3", "#4"] and ds[0].meta["potential_V"] == 0.5
    sel.set_group([sel.key("B", "#5")], "A")                    # another potential with them: noted
    ds, notes = sel.build()
    assert "potential_V" not in ds[0].meta and notes and "one surface structure" in notes[0]
    sel.set_group([sel.key("A repeat", "#3")], "A repeat")      # the file's own name: back to its own group
    assert sel.rows()[2].group == "A repeat" and not sel.rows()[2].custom
    sel.set_group(sel.keys_of("A repeat"), "")
    sel.set_use([sel.key("A", "#2")], True)
    sel.set_group([sel.key("B", "#5")], "")
    assert not sel.choices                                      # only what differs from the default is kept


def test_same_scan_label_in_two_files_stays_apart():
    sel = ScanSelection()
    sel.add(_file("A", 0.5, 1))
    sel.add(_file("B", 0.5, 2))                                 # also '#1' and '#2'
    sel.set_group(sel.keys_of("B"), "A")
    ds, _ = sel.build()
    assert ds[0].scan_labels == ["A: #1", "A: #2", "B: #1", "B: #2"]


def test_mixed_films_are_a_film_of_their_own():
    sel = ScanSelection()
    sel.add(_file("A", 0.5, 1, film="7 nm"))
    sel.add(_file("C", 0.5, 2, film="4 nm"))
    sel.set_group(sel.keys_of("C"), "A")
    ds, notes = sel.build()
    assert ds[0].meta["film"] == "7 nm + 4 nm" and any("films" in n for n in notes)


def test_regroup_keeps_scopes_and_parameter_values():
    model = truth_model()
    sel = ScanSelection()
    for i, (nm, u) in enumerate((("A", 0.5), ("B", 0.9), ("C", 1.1))):
        sel.add(_file(nm, u, 10 + i, scans=(f"#{2 * i}", f"#{2 * i + 1}")))
    ds, _ = regroup(model, sel, [])
    assert [d.scope for d in ds] == ["ds1", "ds2", "ds3"]
    workflow.prepare(model, ds, ["x_OH"], [])
    Fit(model, ds)                                               # the scale copies
    P = model.params
    for s, v in (("ds1", 0.1), ("ds2", 0.2), ("ds3", 0.3)):
        P[f"{s}.x_OH"].value = v
    P["ds1.scale"].value, P["ds1.scale"].fit = 1.3, True
    ds[1].sys_floor = 0.04
    # leaving a scan of B out keeps B's scope, its x_OH and its floor
    sel.set_use([sel.key("B", "#3")], False)
    ds, _ = regroup(model, sel, ds)
    assert [d.scope for d in ds] == ["ds1", "ds2", "ds3"] and P["ds2.x_OH"].value == 0.2 and ds[1].sys_floor == 0.04
    # B's scans into a new group: a new scope that starts from B's copies; B's copies are removed
    sel.set_group(sel.keys_of("B"), "B new")
    ds, _ = regroup(model, sel, ds)
    assert [d.name for d in ds] == ["A", "B new", "C"]
    assert ds[1].scope == "ds2" and P["ds2.x_OH"].value == 0.2       # the freed scope, seeded from B
    # C into A: C's copies go, A keeps its own
    sel.set_group(sel.keys_of("C"), "A")
    ds, _ = regroup(model, sel, ds)
    assert [d.scope for d in ds] == ["ds1", "ds2"] and "ds3.x_OH" not in P and P["ds1.x_OH"].value == 0.1
    # everything in one group: no copies left, the global parameter takes the first one back (and is fitted)
    sel.set_group([r.key for r in sel.rows()], "all")
    ds, _ = regroup(model, sel, ds)
    assert len(ds) == 1 and ds[0].scope == ""
    assert not [n for n in P.names if "." in n]
    assert P["x_OH"].value == 0.1 and P["x_OH"].fit and P["scale"].value == 1.3 and P["scale"].fit
    assert not P.check()


def test_prune_cuts_links_into_removed_copies():
    model = truth_model()
    P = model.params
    for s in ("ds1", "ds2"):
        P.add(name=f"{s}.thickness_mean_tl", value=20.0, min=5, max=40, fit=True, group="film")
    P.link("ds2.thickness_mean_tl", "ds1.thickness_mean_tl + 1")
    prune_scopes(model, ["ds1"])
    assert not P["ds2.thickness_mean_tl"].linked and P["ds2.thickness_mean_tl"].value == 21.0


def test_project_keeps_scan_choices(tmp_path):
    sel = ScanSelection()
    sel.add(_file("A", 0.5, 1))
    sel.add(_file("B", 0.9, 2, scans=("#3", "#4")))
    sel.set_use([sel.key("A", "#2")], False)
    sel.set_group([sel.key("B", "#4")], "B split")
    model = truth_model()
    ds, _ = regroup(model, sel, [])
    ds[2].sys_floor = 0.03
    proj = Project(model, ds, scans=sel)
    proj.save(tmp_path / "p.json")
    back = Project.load(tmp_path / "p.json")
    assert [d.name for d in back.datasets] == ["A", "B", "B split"]
    assert [d.scope for d in back.datasets] == [d.scope for d in ds] and back.datasets[2].sys_floor == 0.03
    assert back.scans.choices == sel.choices and [len(d) for d in back.datasets] == [len(d) for d in ds]
    old = Project(model, [_file("x", 0.5, 3)])                  # a project without scan choices
    assert old.scans.sources[0].name == "x" and old.scans.build()[0][0].name == "x"


def test_grouped_repeats_recover_each_potential():
    """Two potentials, each measured in two files (repeats): grouped by potential, a global fit gives
    one x_OH per potential (the truth), from both files' points."""
    truth = (0.3, 0.7)
    sel = ScanSelection()
    for i, x in enumerate(truth):
        for j in range(2):
            sel.add(_file(f"{0.5 + 0.4 * i:g} V file {j}", 0.5 + 0.4 * i, 500 + 2 * i + j, x_OH=x,
                          scans=(f"#{4 * i + 2 * j}", f"#{4 * i + 2 * j + 1}")))
    for i in range(2):
        u = f"{0.5 + 0.4 * i:g} V"
        sel.set_group(sel.keys_of(f"{u} file 1"), f"{u} file 0")
    model = truth_model()
    ds, _ = regroup(model, sel, [])
    assert len(ds) == 2 and all(len(d.scan_labels) == 4 for d in ds)
    model.params.fit_only(["x_OH", "scale"])
    model.params.update(dict(x_OH=0.5))
    workflow.prepare(model, ds, ["x_OH"], [])
    fit = Fit(model, ds)
    fit.refine()
    res = fit.report()
    np.testing.assert_allclose([res.values["ds1.x_OH"], res.values["ds2.x_OH"]], truth, atol=0.05)
    assert res.fom["N"] == sum(len(sel.source(n)) for n in [s.name for s in sel.sources])


def test_unknown_scan_is_refused():
    sel = ScanSelection([_file("A", 0.5, 1)])
    with pytest.raises(KeyError):
        sel.set_use(["nope\t#1"], False)


def test_shared_parameters_every_dataset_overrides():
    from ctrfit.fit.global_fit import shadowed
    model = truth_model()
    a, b = _file("A", 0.5, 1), _file("B", 0.9, 2, film="4 nm")
    assert shadowed(model.params, [a, b]) == {}                      # no scopes yet
    workflow.prepare(model, [a, b], ["x_OH"], ["thickness_mean_tl"])
    sh = shadowed(model.params, [a, b])
    assert sh["x_OH"] == ["ds1.x_OH", "ds2.x_OH"] and "thickness_mean_tl" in sh and "scale" in sh
    assert "z_OH" not in sh                                          # shared: used by both

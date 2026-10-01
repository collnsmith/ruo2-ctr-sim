"""Parameter model: safe link expressions, links, scopes, bounds, vectors and JSON."""
import math

import numpy as np
import pytest

from ctrfit.model.expr import Expr, ExprError
from ctrfit.model.parameters import Parameter, ParameterSet


def make_set():
    return ParameterSet([
        Parameter("h2o_z", 2.2, 1.0, 4.0, fit=True, unit="Å", group="surface"),
        Parameter("oh_z", 2.0, 1.0, 4.0, unit="Å", group="surface"),
        Parameter("theta", 0.8, 0.0, 1.0, fit=True, group="surface"),
        Parameter("scale", 1.0, 0.0, 10.0, fit=True, group="scale"),
    ])


@pytest.mark.parametrize("text, expected", [
    ("1 + 2 * 3", 7.0), ("-(2 ** 3) % 5", 2.0), ("sqrt(16) + abs(-1)", 5.0),
    ("max(1, 2, 3) - min(4, 5)", -1.0), ("2 * pi", 2 * math.pi), ("exp(log(3.5))", 3.5),
])
def test_expr_arithmetic(text, expected):
    assert Expr(text).evaluate(lambda n: 0.0) == pytest.approx(expected)


@pytest.mark.parametrize("text", [
    "__import__('os')", "open('x')", "a.__class__()", "(lambda: 1)()", "a[0]", "'text'", "a if b else c",
    "a < b", "a and b", "[1, 2]", "f(x=1)", "True", "x.y()", "", "1 +",
])
def test_expr_rejects_unsafe_or_invalid(text):
    with pytest.raises(ExprError):
        Expr(text)


def test_expr_names_and_dotted():
    e = Expr("0.5 * (ds1.theta + ds2.theta) - h2o_z + pi")
    assert e.names == ["ds1.theta", "ds2.theta", "h2o_z"]
    vals = {"ds1.theta": 0.2, "ds2.theta": 0.6, "h2o_z": 1.0}
    assert e.evaluate(vals.__getitem__) == pytest.approx(0.4 - 1.0 + math.pi)


def test_link_follows_and_is_not_free():
    ps = make_set()
    ps.link("oh_z", "h2o_z - 0.2")
    ps["oh_z"].fit = True
    assert "oh_z" not in ps.free_names
    assert ps.values()["oh_z"] == pytest.approx(2.0)
    ps.set_vector([2.5, 0.5, 1.5])               # h2o_z, theta, scale
    assert ps.values()["oh_z"] == pytest.approx(2.3)
    with pytest.raises(ValueError):
        ps.update({"oh_z": 3.0})
    ps.unlink("oh_z")
    assert not ps["oh_z"].linked and ps["oh_z"].value == pytest.approx(2.3)


def test_link_chain_and_cycle():
    ps = make_set()
    ps.add(name="a", value=0.0)
    ps.link("a", "oh_z + 1")
    ps.link("oh_z", "h2o_z * 2")
    assert ps.values()["a"] == pytest.approx(5.4)
    with pytest.raises(ExprError, match="circular"):
        ps.link("h2o_z", "a - 1")
    assert not ps["h2o_z"].linked                 # failed link leaves the parameter as it was
    with pytest.raises(ExprError, match="unknown"):
        ps.link("theta", "nonexistent * 2")


def test_bounds_check():
    ps = make_set()
    assert ps.check() == []
    ps["theta"].value = 1.2
    ps.link("oh_z", "h2o_z * 3")                  # linked value 6.6 > 4
    problems = ps.check()
    assert any("theta" in p for p in problems) and any("linked oh_z" in p for p in problems)
    with pytest.raises(ValueError):
        ps.validate()
    lo, hi = ps.bounds()
    assert list(lo) == [1.0, 0.0, 0.0] and list(hi) == [4.0, 1.0, 10.0]
    with pytest.raises(ValueError):
        Parameter("bad", 1.0, 2.0, 1.0)
    with pytest.raises(ValueError):
        Parameter("bad", 1.0, group="nonsense")


def test_at_bound():
    p = Parameter("x", 0.995, 0.0, 1.0)
    assert p.at_bound()
    p.value = 0.5
    assert not p.at_bound()
    assert not Parameter("y", 1.0).at_bound()     # unbounded


def test_scoped_names_for_global_fits():
    ps = make_set()
    ps.add(name="ds2.oh_z", value=0.0, group="surface")
    ps.add(name="ds2.h2o_z", value=2.6, group="surface")
    ps.add(name="ds3.oh_z", value=0.0, group="surface")
    ps.link("ds2.oh_z", "h2o_z - 0.2")            # resolves to ds2.h2o_z (own scope first)
    ps.link("ds3.oh_z", "h2o_z - 0.2")            # ds3 has no h2o_z: resolves to the global one
    v = ps.values()
    assert v["ds2.oh_z"] == pytest.approx(2.4) and v["ds3.oh_z"] == pytest.approx(2.0)
    view = ps.view("ds2")
    assert view["oh_z"] == pytest.approx(2.4) and view["h2o_z"] == 2.6 and view["theta"] == 0.8
    assert ps.view()["oh_z"] == 2.0 and "ds2.oh_z" not in ps.view("ds2")
    assert ps["ds2.oh_z"].scope == "ds2" and ps["ds2.oh_z"].base == "oh_z"


def test_vectors_and_fit_only():
    ps = make_set()
    ps.fit_only(["theta"])
    assert ps.free_names == ["theta"]
    np.testing.assert_array_equal(ps.get_vector(), [0.8])
    with pytest.raises(KeyError):
        ps.fit_only(["nope"])


def test_json_round_trip(tmp_path):
    ps = make_set()
    ps.link("oh_z", "h2o_z - 0.2")
    ps.add(name="free_unbounded", value=3.0)
    path = tmp_path / "p.json"
    ps.to_json(path)
    back = ParameterSet.from_json(path)
    assert back.names == ps.names
    assert back["oh_z"].expr == "h2o_z - 0.2" and back.values() == ps.values()
    assert back["free_unbounded"].min == -math.inf
    for n in ps.names:
        a, b = ps[n], back[n]
        assert (a.value, a.min, a.max, a.fit, a.unit, a.group) == (b.value, b.min, b.max, b.fit, b.unit, b.group)
    assert "h2o_z" in back.table()

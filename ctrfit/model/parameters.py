"""Fit parameters: value, bounds, fit flag, unit, description, group, and links.

Names may carry a dataset scope, e.g. "ds2.oh_z". A link expression of a scoped parameter looks a
name up in its own scope first ("ds2.h2o_z"), then in the enclosing scopes ("h2o_z"). view("ds2")
returns the values one dataset sees: scoped values replace the global ones of the same base name.
"""
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .expr import Expr, ExprError

GROUPS = ("film", "surface", "scale", "instrument")


@dataclass
class Parameter:
    name: str
    value: float
    min: float = -math.inf
    max: float = math.inf
    fit: bool = False
    unit: str = ""
    description: str = ""
    group: str = "film"
    expr: str = None            # link: value = expression of other parameters
    _expr: Expr = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.value = float(self.value)
        self.min, self.max = float(self.min), float(self.max)
        if self.group not in GROUPS:
            raise ValueError(f"{self.name}: group must be one of {GROUPS}, got {self.group!r}")
        if self.min > self.max:
            raise ValueError(f"{self.name}: min {self.min} > max {self.max}")
        if self.expr is not None:
            self._expr = Expr(self.expr)

    @property
    def linked(self):
        return self.expr is not None

    @property
    def free(self):
        return self.fit and not self.linked

    @property
    def scope(self):
        return self.name.rpartition(".")[0]

    @property
    def base(self):
        return self.name.rpartition(".")[2]

    def at_bound(self, frac=0.01):
        """True if the value sits within frac of the bound range from either bound."""
        if not (math.isfinite(self.min) and math.isfinite(self.max)) or self.max == self.min:
            return False
        tol = frac * (self.max - self.min)
        return self.value - self.min <= tol or self.max - self.value <= tol

    def to_dict(self):
        d = asdict(self)
        d.pop("_expr")
        for k in ("min", "max"):
            if not math.isfinite(d[k]):
                d[k] = None
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d.pop("_expr", None)
        if d.get("min") is None:
            d["min"] = -math.inf
        if d.get("max") is None:
            d["max"] = math.inf
        return cls(**d)


class ParameterSet:
    def __init__(self, params=()):
        self._p = {}
        for p in params:
            self.add(p)

    # ------------------------------------------------------------------ container
    def add(self, p=None, **kw):
        """add(Parameter) or add(name=..., value=..., ...). Returns the parameter."""
        p = Parameter(**kw) if p is None else p
        if p.name in self._p:
            raise KeyError(f"parameter {p.name!r} already exists")
        self._p[p.name] = p
        return p

    def remove(self, name):
        del self._p[name]

    def __getitem__(self, name):
        try:
            return self._p[name]
        except KeyError:
            raise KeyError(f"no parameter {name!r}") from None

    def __contains__(self, name):
        return name in self._p

    def __iter__(self):
        return iter(self._p.values())

    def __len__(self):
        return len(self._p)

    @property
    def names(self):
        return list(self._p)

    def copy(self):
        return ParameterSet.from_dict(self.to_dict())

    # ------------------------------------------------------------------ flags and links
    def set_fit(self, names, fit=True):
        for n in ([names] if isinstance(names, str) else names):
            self[n].fit = fit

    def fit_only(self, names):
        """Free exactly these parameters (links are kept)."""
        names = set([names] if isinstance(names, str) else names)
        unknown = names - set(self._p)
        if unknown:
            raise KeyError(f"unknown parameters {sorted(unknown)}")
        for p in self:
            p.fit = p.name in names

    def link(self, name, expr):
        """Make `name` follow an expression of other parameters, e.g. link('oh_z', 'h2o_z - 0.2')."""
        p = self[name]
        old = (p.expr, p._expr)
        p.expr, p._expr = expr, Expr(expr)
        try:
            self._order()
            for ref in p._expr.names:
                self.resolve(ref, p.scope)
        except Exception:
            p.expr, p._expr = old
            raise

    def unlink(self, name):
        p = self[name]
        p.value = self.values()[name]
        p.expr, p._expr = None, None

    def resolve(self, ref, scope=""):
        """Full name a reference in `scope` points to: 'scope.ref', then outer scopes, then 'ref'."""
        parts = scope.split(".") if scope else []
        for i in range(len(parts), -1, -1):
            cand = ".".join(parts[:i] + [ref])
            if cand in self._p:
                return cand
        raise ExprError(f"link refers to unknown parameter {ref!r}" + (f" (scope {scope!r})" if scope else ""))

    def _order(self):
        """Linked parameters in evaluation order; raises on cycles."""
        deps = {p.name: [self.resolve(r, p.scope) for r in p._expr.names] for p in self if p.linked}
        order, state = [], {}

        def visit(n, chain):
            if state.get(n) == 2:
                return
            if state.get(n) == 1:
                raise ExprError("circular link: " + " -> ".join(chain + [n]))
            state[n] = 1
            for d in deps.get(n, ()):
                visit(d, chain + [n])
            state[n] = 2
            if n in deps:
                order.append(n)

        for n in deps:
            visit(n, [])
        return order

    # ------------------------------------------------------------------ values
    def values(self):
        """All values with links evaluated (dict name -> float). Also stores them on the parameters."""
        vals = {p.name: p.value for p in self}
        for n in self._order():
            p = self._p[n]
            vals[n] = p._expr.evaluate(lambda r, s=p.scope: vals[self.resolve(r, s)])
            p.value = vals[n]
        return vals

    def view(self, scope=""):
        """Base name -> value as seen by one dataset scope (scoped values override global ones)."""
        vals = self.values()
        out = {n: v for n, v in vals.items() if "." not in n}
        if scope:
            pre = scope + "."
            for n, v in vals.items():
                if n.startswith(pre) and "." not in n[len(pre):]:
                    out[n[len(pre):]] = v
        return out

    def update(self, values):
        for n, v in values.items():
            p = self[n]
            if p.linked:
                raise ValueError(f"{n} is linked ({p.expr}); unlink it before setting a value")
            p.value = float(v)

    @property
    def free_names(self):
        return [p.name for p in self if p.free]

    def get_vector(self, names=None):
        names = self.free_names if names is None else names
        return np.array([self[n].value for n in names], float)

    def set_vector(self, x, names=None):
        names = self.free_names if names is None else names
        for n, v in zip(names, np.asarray(x, float)):
            self[n].value = float(v)
        return self.values()

    def bounds(self, names=None):
        names = self.free_names if names is None else names
        return (np.array([self[n].min for n in names]), np.array([self[n].max for n in names]))

    def check(self):
        """List of problems: values outside bounds (free, fixed or linked) and broken links."""
        problems = []
        try:
            vals = self.values()
        except (ExprError, ArithmeticError, ValueError) as ex:
            return [str(ex)]
        for p in self:
            v = vals[p.name]
            if not math.isfinite(v):
                problems.append(f"{p.name} = {v} is not finite")
            elif v < p.min or v > p.max:
                what = "linked " if p.linked else ""
                problems.append(f"{what}{p.name} = {v:.6g} is outside [{p.min:g}, {p.max:g}] {p.unit}".rstrip())
        return problems

    def validate(self):
        problems = self.check()
        if problems:
            raise ValueError("; ".join(problems))

    # ------------------------------------------------------------------ io
    def to_dict(self):
        return {"parameters": [p.to_dict() for p in self]}

    @classmethod
    def from_dict(cls, d):
        ps = cls()
        linked = []
        for item in d["parameters"]:
            item = dict(item)
            expr = item.pop("expr", None)
            ps.add(Parameter.from_dict(item))
            if expr is not None:
                linked.append((item["name"], expr))
        for n, e in linked:
            ps.link(n, e)
        return ps

    def to_json(self, path=None, indent=1):
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_json(cls, path_or_text):
        text = str(path_or_text)
        if not text.lstrip().startswith("{"):
            with open(path_or_text, encoding="utf-8") as fh:
                text = fh.read()
        return cls.from_dict(json.loads(text))

    def table(self, names=None, errors=None):
        """Plain-text parameter table."""
        names = self.names if names is None else names
        vals = self.values()
        rows = [f"{'name':<22} {'value':>12} {'error':>10} {'min':>9} {'max':>9}  {'unit':<5} fit  description"]
        for n in names:
            p = self[n]
            err = "" if not errors or n not in errors else f"{errors[n]:.3g}"
            flag = "link" if p.linked else ("yes" if p.fit else "")
            rows.append(f"{n:<22} {vals[n]:>12.6g} {err:>10} {p.min:>9.4g} {p.max:>9.4g}  {p.unit:<5} {flag:<4} "
                        + (p.description if not p.linked else f"= {p.expr}"))
        return "\n".join(rows)

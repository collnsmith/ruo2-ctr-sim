"""A parameterized model: template + fixed settings + ParameterSet.

    model = Model.from_settings(DEFAULTS)            # template "rutile110_film"
    model.params["z_OH"].value = 2.1
    I = model.intensity(0, 1, L)                     # |F|^2 without scale, reference engine
    F = model.F(0, 1, L)                             # scale * |F|, as compared with data

Data are structure-factor amplitudes: F_data = scale * rodscale(H K) * |F_model|.
"""
import json

import numpy as np

from ..core.film import FilmCTRModel
from ..core.settings import DEFAULTS
from .parameters import Parameter, ParameterSet
from .templates import get_template

ROD_SCALE_BOUNDS = (0.5, 2.0)


def rod_scale_name(H, K, scope=""):
    return (scope + "." if scope else "") + f"rodscale_{H}_{K}"


class Model:
    def __init__(self, template="rutile110_film", settings=None, params=None, options=None):
        self.template = get_template(template, options)
        self.options = self.template.options
        self.settings = self.template.fixed_settings(settings or {})
        if params is None:
            vals = self.template.values_from_settings(settings or {})
            params = ParameterSet(Parameter(name=n, value=vals[n], min=lo, max=hi, fit=fit, unit=u,
                                            description=d, group=g)
                                  for n, u, lo, hi, fit, g, d in self.template.specs())
        self.params = params

    @classmethod
    def from_settings(cls, settings=None, template="rutile110_film", **options):
        return cls(template, settings=settings, options=options)

    # ------------------------------------------------------------------ rod scales
    def add_rod_scales(self, rods, scope="", fit=False):
        """One scale per rod (H, K), off by default; the fit adds a penalty pulling them to 1."""
        for H, K in rods:
            n = rod_scale_name(H, K, scope)
            if n not in self.params:
                self.params.add(name=n, value=1.0, min=ROD_SCALE_BOUNDS[0], max=ROD_SCALE_BOUNDS[1], fit=fit,
                                unit="", group="scale", description=f"Scale of the ({H} {K}) rod relative to the "
                                                                    "dataset scale")

    def rod_scale(self, values, H, K):
        return values.get(f"rodscale_{H}_{K}", 1.0)

    # ------------------------------------------------------------------ evaluation (reference path)
    def engine(self, values=None, scope=""):
        """(FilmCTRModel, composition) for the current or given parameter values."""
        v = self.params.view(scope) if values is None else values
        s, geometry, comp = self.template.engine_inputs(self.settings, v)
        return FilmCTRModel(s, geometry), comp

    def intensity(self, H, K, L, values=None, scope=""):
        """|F|^2 without scale factors, from the reference engine."""
        m, comp = self.engine(values, scope)
        return m.intensity(H, K, np.atleast_1d(np.asarray(L, float)), comp)

    def F(self, H, K, L, values=None, scope=""):
        v = self.params.view(scope) if values is None else values
        return v["scale"] * self.rod_scale(v, H, K) * np.sqrt(self.intensity(H, K, L, v))

    def evaluator(self, H, K, L, free=None, scope=""):
        """Fast vectorized |F| for many points (see ctrfit.model.evaluator)."""
        from .evaluator import Evaluator
        return Evaluator(self, H, K, L, free=free, scope=scope)

    # ------------------------------------------------------------------ io
    def to_dict(self):
        settings = {k: v for k, v in self.settings.items() if k not in DEFAULTS or DEFAULTS[k] != v}
        return dict(template=self.template.name, options=self.options, settings=settings,
                    **self.params.to_dict())

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("template", "rutile110_film"), settings=d.get("settings", {}),
                   params=ParameterSet.from_dict(d), options=d.get("options"))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def copy(self):
        return Model.from_dict(json.loads(json.dumps(self.to_dict())))

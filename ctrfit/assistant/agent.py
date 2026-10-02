"""Claude as a fitting assistant: a tool-use loop over the ctrfit fitting API (no Qt).

    agent = FitAgent(model, datasets, on_event=print)
    agent.ask("Fit the film on the even rods, then the surface, and tell me how sure we are.")
    agent.model        # the agent's copy of the model, at its result

Claude can only call the tools below; each is a thin wrapper over ctrfit (Fit, workflow, report,
compare_models). It works on a copy of the model and datasets, so the caller decides whether to keep
the result. Needs the `anthropic` package and an API key (ANTHROPIC_API_KEY or `ant auth login`).
"""
import json
import math
import time

import numpy as np

from ..data.dataset import Dataset
from ..fit import workflow
from ..fit.fit import Fit
from ..fit.global_fit import make_per_dataset
from ..fit.sampling import compare_models

MODEL_ID = "claude-opus-5-5"
PRICE_PER_MTOK = {"input": 4.0, "output": 20.0, "cache_read": 0.2}   # Claude Opus 5.5, USD
FALLBACK_BETA = "server-side-fallback-2026-07-01"
LIMITS = dict(maxiter=200, popsize=30, profile_points=41, turns=40)

SYSTEM_PROMPT = """You are a careful surface X-ray diffraction analyst. You fit crystal truncation rod (CTR) data
of an RuO2(110) film on TiO2(110), with OH / H2O / O on the RuO2 coordinatively unsaturated (CUS)
sites, by calling the tools you have. The model is kinematic; data are structure-factor amplitudes
|F|(H, K, L) with errors; F_data = scale * rodscale * |F_model|.

Units: H || [001], K || [1-10], L || [110] in TiO2 surface units. Thickness, spread and roughness are
in (110) trilayers (TL); heights in Å; B factors in Å²; eps_perp in %. CUS composition:
O = theta x_O, OH = theta (1 - x_O) x_OH, H2O = theta (1 - x_O) (1 - x_OH).

What is known about this problem:
- Rods with H + K even are dominated by the film (thickness, roughness, strain); rods with H + K odd
  are the most sensitive to the CUS species. The usual order is: film on even rods, then surface on
  odd rods, then everything together (the guided_step tool does each step).
- Film thickness has separate chi2 minima about one trilayer apart, traded against eps_perp; a global
  search alone often lands in the wrong one. Use a thickness profile before trusting a thickness.
- OH vs H2O with free heights and B factors is nearly degenerate: a mixture at two heights looks like
  one species at an intermediate height with more disorder. Do not report a confident OH share unless
  the errors and correlations support it; prefer fixing heights to DFT-like values or linking them.
- Scale correlates with roughness. Reduced chi2 near 1 means the model fits within the errors;
  much larger means a missing ingredient or underestimated errors, not just a bad optimizer start.

How to work:
- Start with get_state. Change only what the user asked for or what the data justify. Free few
  parameters at a time; keep bounds physical.
- After a fit, call report and read the warnings (correlations > 0.9, values at bounds, relative errors
  > 100%). Use rod_residuals to see which rods misfit.
- Keep the budget modest: differential evolution with maxiter <= 60 and popsize <= 15 is usually enough.
- When you compare surface models, use compare_models (AIC / BIC) rather than chi2 alone.
- Finish with a short plain-language summary for the user: what you fitted, the values with errors,
  reduced chi2, the warnings that matter, and what you would do next. Say clearly what the data cannot
  determine. Do not invent numbers: quote only what the tools returned."""


def _nullable(t):
    return {"anyOf": [{"type": t}, {"type": "null"}]}


TOOLS = [
    dict(name="get_state", description="Parameters (value, bounds, unit, fit flag, link, description), datasets "
         "(rods, points, metadata) and the last fit result. Call this first and whenever unsure.",
         input_schema=dict(type="object", properties={}, required=[], additionalProperties=False)),
    dict(name="set_parameters", description="Change parameters: value, bounds and/or fit flag (null leaves a field as "
         "it is). Values must stay inside the bounds.",
         input_schema=dict(type="object", properties={"changes": {"type": "array", "items": dict(
             type="object", properties={"name": {"type": "string"}, "value": _nullable("number"),
                                        "min": _nullable("number"), "max": _nullable("number"),
                                        "fit": _nullable("boolean")},
             required=["name", "value", "min", "max", "fit"], additionalProperties=False)}},
             required=["changes"], additionalProperties=False)),
    dict(name="link_parameter", description="Make a parameter follow an expression of others, e.g. name='z_H2O', "
         "expr='z_OH + 0.4'. An empty expr removes the link.",
         input_schema=dict(type="object", properties={"name": {"type": "string"}, "expr": {"type": "string"}},
                           required=["name", "expr"], additionalProperties=False)),
    dict(name="make_per_dataset", description="With several datasets (e.g. potentials): give each dataset its own copy "
         "of these parameters (ds1.x_OH, ds2.x_OH, ...); everything else free stays shared.",
         input_schema=dict(type="object", properties={"names": {"type": "array", "items": {"type": "string"}}},
                           required=["names"], additionalProperties=False)),
    dict(name="guided_step", description="One step of the standard workflow with its own free parameters: 'film' (even "
         "rods: thickness, roughness, eps_perp, scale; DE + thickness profile + refine), 'surface' (odd rods: theta, "
         "x_OH, z_OH, scale; DE + refine) or 'all' (refine film and surface parameters on all rods).",
         input_schema=dict(type="object", properties={"step": {"type": "string", "enum": ["film", "surface", "all"]},
                                                      "maxiter": {"type": "integer"}, "popsize": {"type": "integer"},
                                                      "seed": {"type": "integer"}},
                           required=["step", "maxiter", "popsize", "seed"], additionalProperties=False)),
    dict(name="run_global_search", description="Differential evolution over the currently free parameters (all need "
         "finite bounds), on all data. fom: chi2, log or R1.",
         input_schema=dict(type="object", properties={"maxiter": {"type": "integer"}, "popsize": {"type": "integer"},
                                                      "seed": {"type": "integer"},
                                                      "fom": {"type": "string", "enum": ["chi2", "log", "R1"]}},
                           required=["maxiter", "popsize", "seed", "fom"], additionalProperties=False)),
    dict(name="refine", description="Least-squares refinement of the free parameters from their current values; gives "
         "errors and correlations.",
         input_schema=dict(type="object", properties={}, required=[], additionalProperties=False)),
    dict(name="profile", description="chi2 profile of one free parameter: fixed at each grid value (start to stop, step) "
         "while the others are refined; keeps the best minimum. Use it for the film thickness.",
         input_schema=dict(type="object", properties={"name": {"type": "string"}, "start": {"type": "number"},
                                                      "stop": {"type": "number"}, "step": {"type": "number"}},
                           required=["name", "start", "stop", "step"], additionalProperties=False)),
    dict(name="report", description="Fit report at the current values: chi2, reduced chi2, log FOM, R1, AIC, BIC, values "
         "with errors, strong correlations and warnings.",
         input_schema=dict(type="object", properties={}, required=[], additionalProperties=False)),
    dict(name="rod_residuals", description="Per dataset and rod: number of points, chi2 per point and mean signed "
         "residual, to see which rods misfit.",
         input_schema=dict(type="object", properties={}, required=[], additionalProperties=False)),
    dict(name="compare_models", description="Compare surface models by AIC / BIC: for each candidate, set its free "
         "parameter list and fixed values, refine on all data, and rank. The model is restored afterwards.",
         input_schema=dict(type="object", properties={"candidates": {"type": "array", "items": dict(
             type="object", properties={"label": {"type": "string"}, "free": {"type": "array", "items": {"type": "string"}},
                                        "fixed": {"type": "array", "items": dict(
                                            type="object", properties={"name": {"type": "string"},
                                                                       "value": {"type": "number"}},
                                            required=["name", "value"], additionalProperties=False)}},
             required=["label", "free", "fixed"], additionalProperties=False)}},
             required=["candidates"], additionalProperties=False)),
]
for _t in TOOLS:
    _t["strict"] = True


class AgentStopped(Exception):
    pass


class FitAgent:
    """Runs Claude over the fitting tools. on_event(kind, data) reports progress:
    'text', 'thinking', 'tool_call', 'tool_result', 'best' (live parameter values), 'usage', 'error', 'done'."""

    def __init__(self, model, datasets, client=None, model_id=MODEL_ID, effort="high", on_event=None,
                 max_turns=LIMITS["turns"], use_fallback=True):
        self.model = model.copy()
        self.datasets = [Dataset.from_dict(ds.to_dict()) for ds in datasets]
        self.client = client
        self.model_id, self.effort = model_id, effort
        self.on_event = on_event or (lambda kind, data: None)
        self.max_turns = max_turns
        self.use_fallback = use_fallback
        self.messages = []
        self.usage = dict(input=0, output=0, cache_read=0)
        self.result = None
        self.stopped = False
        self._fit = None
        self.log = []

    # ------------------------------------------------------------------ control
    def stop(self):
        self.stopped = True
        if self._fit is not None:
            self._fit.cancel()

    @property
    def cost_usd(self):
        u = self.usage
        return (u["input"] * PRICE_PER_MTOK["input"] + u["output"] * PRICE_PER_MTOK["output"]
                + u["cache_read"] * PRICE_PER_MTOK["cache_read"]) / 1e6

    def _emit(self, kind, data):
        self.log.append((kind, data))
        self.on_event(kind, data)

    def _client(self):
        if self.client is None:
            try:
                import anthropic
            except ImportError as ex:
                raise RuntimeError("The fitting assistant needs the anthropic package: pip install anthropic") from ex
            self.client = anthropic.Anthropic()
        return self.client

    # ------------------------------------------------------------------ the loop
    def ask(self, text):
        """Send one request to Claude and run its tool calls until it answers. Returns the final text."""
        self.stopped = False
        self.messages.append({"role": "user", "content": text})
        final = ""
        for _ in range(self.max_turns):
            if self.stopped:
                raise AgentStopped()
            response = self._create()
            self._account(response)
            self.messages.append({"role": "assistant", "content": response.content})
            for block in response.content:
                if block.type == "thinking" and getattr(block, "thinking", ""):
                    self._emit("thinking", block.thinking)
                elif block.type == "text" and block.text.strip():
                    self._emit("text", block.text)
                    final = block.text
            if response.stop_reason == "refusal":
                self._emit("error", "Claude declined this request (refusal); try rephrasing it.")
                break
            if response.stop_reason == "max_tokens":
                self.messages.append({"role": "user", "content": "Your reply was cut off; please continue briefly."})
                continue
            calls = [b for b in response.content if b.type == "tool_use"]
            if not calls:
                break
            results = []
            for call in calls:
                if self.stopped:
                    results.append(dict(type="tool_result", tool_use_id=call.id, is_error=True,
                                        content="Stopped by the user."))
                    continue
                self._emit("tool_call", dict(name=call.name, input=call.input))
                try:
                    out = self.execute(call.name, call.input)
                    err = False
                except (ValueError, KeyError, RuntimeError) as ex:
                    out, err = f"Error: {ex}", True
                self._emit("tool_result", dict(name=call.name, output=out, error=err))
                results.append(dict(type="tool_result", tool_use_id=call.id, content=out, is_error=err))
            self.messages.append({"role": "user", "content": results})
        else:
            self._emit("error", f"Stopped after {self.max_turns} turns without a final answer.")
        self._emit("done", final)
        return final

    def _create(self):
        kw = dict(model=self.model_id, max_tokens=16000, system=SYSTEM_PROMPT, tools=TOOLS, messages=self.messages,
                  thinking={"type": "adaptive", "display": "summarized"}, output_config={"effort": self.effort},
                  cache_control={"type": "ephemeral"})
        client = self._client()
        if not self.use_fallback:
            return client.messages.create(**kw)
        try:
            return client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kw)
        except Exception as ex:          # an account or proxy without server-side fallbacks: plain request
            if "fallback" not in str(ex).lower():
                raise
            self.use_fallback = False
            return client.messages.create(**kw)

    def _account(self, response):
        u = getattr(response, "usage", None)
        if u is None:
            return
        self.usage["input"] += (getattr(u, "input_tokens", 0) or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0)
        self.usage["output"] += getattr(u, "output_tokens", 0) or 0
        self.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        self._emit("usage", dict(self.usage, cost_usd=self.cost_usd))

    # ------------------------------------------------------------------ tools
    def execute(self, name, args):
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            raise ValueError(f"unknown tool {name}")
        return fn(**(args or {}))

    def _new_fit(self, datasets=None, fom="chi2"):
        fit = Fit(self.model, datasets or self.datasets, fom=fom)
        self._hook(fit)
        return fit

    def _hook(self, fit):
        self._fit = fit
        fit.on_best = lambda stage, step, f, values: self._emit("best", dict(stage=stage, step=step, fom=f,
                                                                            values=values))
        if self.stopped:
            fit.cancel()

    def _finish(self, fit, res=None):
        self.result = res or fit.report()
        self._fit = None
        if self.stopped:
            raise AgentStopped()
        return self._brief(self.result)

    @staticmethod
    def _brief(res):
        f = res.fom
        lines = [f"N = {f['N']}, free = {f['p']}, chi2 = {f['chi2']:.4g}, reduced chi2 = {f['red_chi2']:.4g}, "
                 f"R1 = {f['R1']:.4g}, AIC = {f['aic']:.5g}, BIC = {f['bic']:.5g}"]
        lines += [f"{n} = {res.values[n]:.6g} +- {res.errors.get(n, float('nan')):.3g}" for n in res.names]
        if res.warnings:
            lines.append("warnings: " + " | ".join(res.warnings))
        return "\n".join(lines)

    def tool_get_state(self):
        ps = self.model.params
        vals = ps.values()
        params = [dict(name=p.name, value=round(vals[p.name], 6), min=None if not math.isfinite(p.min) else p.min,
                       max=None if not math.isfinite(p.max) else p.max, unit=p.unit, fit=p.fit, link=p.expr,
                       group=p.group, description=p.description) for p in ps]
        data = [dict(name=ds.name, scope=ds.scope, points=len(ds), rods=ds.rod_labels,
                     L_range=[round(float(ds.L.min()), 3), round(float(ds.L.max()), 3)], sys_floor=ds.sys_floor,
                     meta=ds.meta) for ds in self.datasets]
        last = self._brief(self.result) if self.result is not None else "no fit yet"
        return json.dumps(dict(parameters=params, datasets=data, last_fit=last, settings=dict(
            energy_kev=self.model.settings.get("energy_kev"), mix_mode=self.model.settings.get("mix_mode"),
            water_layer=self.model.options.get("water_layer"))), default=str)

    def tool_set_parameters(self, changes):
        done = []
        for c in changes:
            p = self.model.params[c["name"]]
            lo = p.min if c.get("min") is None else float(c["min"])
            hi = p.max if c.get("max") is None else float(c["max"])
            if lo > hi:
                raise ValueError(f"{p.name}: min {lo} > max {hi}")
            v = p.value if c.get("value") is None else float(c["value"])
            if c.get("value") is not None and p.linked:
                raise ValueError(f"{p.name} is linked ({p.expr}); unlink it first")
            if not lo <= v <= hi:
                raise ValueError(f"{p.name}: value {v} outside [{lo}, {hi}]")
            p.min, p.max, p.value = lo, hi, v
            if c.get("fit") is not None:
                p.fit = bool(c["fit"])
            done.append(f"{p.name} = {p.value:g} [{p.min:g}, {p.max:g}] fit={p.fit}")
        problems = self.model.params.check()
        return "\n".join(done + ([f"check: {'; '.join(problems)}"] if problems else []))

    def tool_link_parameter(self, name, expr):
        if expr.strip():
            self.model.params.link(name, expr)
            return f"{name} = {expr} (now {self.model.params.values()[name]:.6g})"
        self.model.params.unlink(name)
        return f"{name} unlinked at {self.model.params[name].value:.6g}"

    def tool_make_per_dataset(self, names):
        if len(self.datasets) < 2:
            raise ValueError("only one dataset: nothing to split")
        make_per_dataset(self.model, self.datasets, names)
        workflow.prepare(self.model, self.datasets)
        return "created " + ", ".join(f"{ds.scope}.{n}" for n in names for ds in self.datasets)

    def _de(self, maxiter, popsize, seed):
        return dict(maxiter=int(np.clip(maxiter, 1, LIMITS["maxiter"])), popsize=int(np.clip(popsize, 4, LIMITS["popsize"])),
                    seed=int(seed))

    def tool_guided_step(self, step, maxiter, popsize, seed):
        de = self._de(maxiter, popsize, seed)
        workflow.prepare(self.model, self.datasets)
        if step == "film":
            fit, res = workflow.step_film(self.model, self.datasets, de=de, fit_hook=self._hook)
        elif step == "surface":
            fit, res = workflow.step_surface(self.model, self.datasets, de=de, fit_hook=self._hook)
        else:
            fit, res = workflow.step_all(self.model, self.datasets, fit_hook=self._hook)
        return f"{step} step done.\n" + self._finish(fit, res)

    def tool_run_global_search(self, maxiter, popsize, seed, fom):
        fit = self._new_fit(fom=fom)
        if not fit.names:
            raise ValueError("no free parameters")
        st = fit.run("de", **self._de(maxiter, popsize, seed))
        return f"DE: {st['nfev']} evaluations, {st['message']}\n" + self._finish(fit)

    def tool_refine(self):
        fit = self._new_fit()
        if not fit.names:
            raise ValueError("no free parameters")
        st = fit.refine()
        return f"least squares: {st['nfev']} evaluations, {st['message']}\n" + self._finish(fit)

    def tool_profile(self, name, start, stop, step):
        if step <= 0 or stop <= start:
            raise ValueError("need start < stop and step > 0")
        n = int(math.floor((stop - start) / step + 1e-9)) + 1
        if n > LIMITS["profile_points"]:
            raise ValueError(f"at most {LIMITS['profile_points']} grid points (asked {n}); use a larger step")
        p = self.model.params[name]
        grid = np.clip(np.arange(start, stop + 1e-9, step), p.min, p.max)
        fit = self._new_fit()
        if name not in fit.names:
            raise ValueError(f"{name} is not free")
        prof = fit.profile(name, grid)
        table = ", ".join(f"{d['value']:g}: {d['chi2']:.1f}" for d in prof)
        fit.refine()
        return f"profile of {name} (value: chi2): {table}\nbest refined:\n" + self._finish(fit)

    def tool_report(self):
        fit = self._new_fit()
        res = fit.report()
        self.result = res
        self._fit = None
        return res.summary

    def tool_rod_residuals(self):
        fit = self._new_fit()
        Fc = fit.model_F()
        self._fit = None
        lines = []
        for ds, f in zip(self.datasets, Fc):
            r = (ds.F - f) / ds.sigma_eff
            for lab, idx in ds.rods().items():
                lines.append(f"{ds.name} ({lab} L): {idx.size} points, chi2/point {np.mean(r[idx] ** 2):.3g}, "
                             f"mean residual {np.mean(r[idx]):+.2f}, max |r| {np.max(np.abs(r[idx])):.2f}")
        return "\n".join(lines)

    def tool_compare_models(self, candidates):
        if not 2 <= len(candidates) <= 6:
            raise ValueError("give 2 to 6 candidates")
        ps = self.model.params
        saved = {p.name: (p.value, p.fit) for p in ps if not p.linked}
        results, labels = [], []
        try:
            for c in candidates:
                for n, (v, f) in saved.items():
                    ps[n].value = v
                ps.fit_only(c["free"])
                for fx in c["fixed"]:
                    ps[fx["name"]].value = float(fx["value"])
                fit = self._new_fit()
                fit.refine()
                results.append(fit.report())
                labels.append(c["label"])
                if self.stopped:
                    raise AgentStopped()
        finally:
            for n, (v, f) in saved.items():
                ps[n].value, ps[n].fit = v, f
            self._fit = None
        _, text = compare_models(results, labels)
        return text + "\n\n" + "\n\n".join(f"[{lab}]\n{self._brief(r)}" for lab, r in zip(labels, results))


def transcript_text(log):
    """Plain-text transcript of an agent log (for saving next to the fit results)."""
    out = [f"# ctrfit assistant transcript, {time.strftime('%Y-%m-%d %H:%M')}"]
    for kind, data in log:
        if kind in ("text", "thinking"):
            out.append(f"[{kind}] {data}")
        elif kind == "tool_call":
            out.append(f"[tool] {data['name']}({json.dumps(data['input'])})")
        elif kind == "tool_result":
            out.append(f"[result{' ERROR' if data['error'] else ''}] {data['output']}")
        elif kind == "error":
            out.append(f"[error] {data}")
    return "\n".join(out) + "\n"

"""Fitting assistant: tool loop and tools, with a scripted stand-in for the Claude API (no network)."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from ctrfit import Model, make_dataset
from ctrfit.assistant.agent import TOOLS, AgentStopped, FitAgent, transcript_text
from test_fit import FILM_TRUTH, ODD, SURF_TRUTH, start_model, truth_model


def text(t):
    return NS(type="text", text=t)


def call(i, tool, **inp):
    return NS(type="tool_use", id=f"tu{i}", name=tool, input=inp)


def response(*blocks, stop=None):
    stop = stop or ("tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn")
    return NS(content=list(blocks), stop_reason=stop,
              usage=NS(input_tokens=1000, output_tokens=200, cache_read_input_tokens=500,
                       cache_creation_input_tokens=0))


class FakeClient:
    """Plays back scripted responses and records every request."""

    def __init__(self, script, fallback_error=False):
        self.script = list(script)
        self.requests = []
        self.fallback_error = fallback_error
        self.beta = NS(messages=NS(create=self._beta_create))
        self.messages = NS(create=self._create)

    def _beta_create(self, **kw):
        if self.fallback_error:
            raise ValueError("fallbacks: not available for this account")
        return self._create(**kw)

    def _create(self, **kw):
        self.requests.append(kw)
        return self.script.pop(0)


def surface_setup():
    data = make_dataset(truth_model(), rods=ODD, noise_rel=0.03, seed=11)
    model = start_model([], dict(z_OH=1.9))
    model.params.update(dict(FILM_TRUTH, theta=SURF_TRUTH["theta"], x_OH=SURF_TRUTH["x_OH"]))
    return model, data


def test_tool_schemas_are_strict():
    def check(schema, where):
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False, where
            assert set(schema.get("required", [])) == set(schema.get("properties", {})), where
            for k, v in schema.get("properties", {}).items():
                check(v, f"{where}.{k}")
        for opt in schema.get("anyOf", []):
            check(opt, where)
        if "items" in schema:
            check(schema["items"], where + "[]")
    names = [t["name"] for t in TOOLS]
    assert len(names) == len(set(names))
    for t in TOOLS:
        assert t["strict"] and t["description"]
        check(t["input_schema"], t["name"])
    assert all(hasattr(FitAgent, f"tool_{n}") for n in names)


def test_agent_loop_fits_and_works_on_a_copy():
    model, data = surface_setup()
    script = [
        response(text("Looking at the state first."), call(1, "get_state")),
        response(call(2, "set_parameters", changes=[
            dict(name="z_OH", value=None, min=1.6, max=2.5, fit=True),
            dict(name="scale", value=None, min=None, max=None, fit=True)])),
        response(call(3, "refine"), call(4, "rod_residuals")),
        response(call(5, "report")),
        response(text("z_OH is 2.05 +- 0.002 and reduced chi2 is near 1.")),
    ]
    client = FakeClient(script)
    events = []
    agent = FitAgent(model, [data], client=client, on_event=lambda k, d: events.append(k))
    final = agent.ask("Fit z_OH.")
    assert final.startswith("z_OH is")
    assert abs(agent.model.params["z_OH"].value - 2.05) < 0.01                  # the copy is fitted
    assert model.params["z_OH"].value == 1.9 and not model.params["z_OH"].fit  # the original is untouched
    assert agent.result is not None and agent.result.fom["red_chi2"] < 1.5
    # requests: model, thinking, effort, strict tools, fallbacks via the beta endpoint
    r0 = client.requests[0]
    assert r0["model"] == "claude-opus-5-5" and r0["thinking"]["type"] == "adaptive"
    assert r0["output_config"]["effort"] == "high" and r0["fallbacks"] == "default"
    assert r0["betas"] == ["server-side-fallback-2026-07-01"] and r0["tools"] is TOOLS
    # conversation: append-only, tool results answer the matching ids in one user message
    msgs = agent.messages
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user", "assistant", "user",
                                        "assistant", "user", "assistant"]
    third = msgs[6]["content"]
    assert [r["tool_use_id"] for r in third] == ["tu3", "tu4"] and not any(r["is_error"] for r in third)
    assert "chi2/point" in third[1]["content"]
    assert {"tool_call", "tool_result", "text", "usage", "done", "best"} <= set(events)
    assert agent.cost_usd == pytest.approx(5 * (1000 * 4 + 200 * 20 + 500 * 0.2) / 1e6)
    assert "[tool] refine" in transcript_text(agent.log)


def test_tool_errors_go_back_to_claude():
    model, data = surface_setup()
    script = [response(call(1, "set_parameters", changes=[dict(name="theta", value=1.5, min=None, max=None,
                                                               fit=None)]),
                       call(2, "profile", name="z_OH", start=1.0, stop=3.0, step=0.01)),
              response(text("ok"))]
    agent = FitAgent(model, [data], client=FakeClient(script))
    agent.ask("try")
    res = agent.messages[2]["content"]
    assert all(r["is_error"] for r in res)
    assert "outside" in res[0]["content"] and "grid points" in res[1]["content"]


def test_fallback_is_dropped_when_unavailable():
    model, data = surface_setup()
    client = FakeClient([response(text("hello"))], fallback_error=True)
    agent = FitAgent(model, [data], client=client)
    agent.ask("hi")
    assert not agent.use_fallback and "fallbacks" not in client.requests[0]


def test_refusal_and_max_turns():
    model, data = surface_setup()
    events = []
    agent = FitAgent(model, [data], client=FakeClient([response(stop="refusal")]),
                     on_event=lambda k, d: events.append((k, d)))
    agent.ask("x")
    assert any(k == "error" and "declined" in d for k, d in events)
    loop = [response(call(i, "get_state")) for i in range(3)]
    agent = FitAgent(model, [data], client=FakeClient(loop), max_turns=3, on_event=lambda k, d: events.append((k, d)))
    agent.ask("x")
    assert any(k == "error" and "turns" in str(d) for k, d in events)


def test_stop_during_a_turn():
    """Stop pressed while a tool runs: the remaining calls are skipped and the loop raises AgentStopped."""
    model, data = surface_setup()
    agent = FitAgent(model, [data], client=FakeClient([response(call(1, "get_state"), call(2, "refine")),
                                                       response(text("never reached"))]))
    agent.on_event = lambda k, d: agent.stop() if k == "tool_result" else None
    with pytest.raises(AgentStopped):
        agent.ask("x")
    res = agent.messages[2]["content"]
    assert not res[0]["is_error"] and res[1]["is_error"] and "Stopped" in res[1]["content"]


def test_tools_directly():
    model, data = surface_setup()
    agent = FitAgent(model, [data], client=None)
    st = agent.tool_get_state()
    assert '"z_OH"' in st and '"rods"' in st
    agent.tool_link_parameter("z_H2O", "z_OH + 0.7")       # 1.9 + 0.7 = 2.6, the true value
    assert agent.model.params["z_H2O"].linked
    agent.tool_link_parameter("z_H2O", "")
    assert not agent.model.params["z_H2O"].linked
    agent.tool_set_parameters([dict(name="z_OH", value=None, min=1.6, max=2.5, fit=True),
                               dict(name="scale", value=None, min=0.5, max=2.0, fit=True)])
    out = agent.tool_run_global_search(maxiter=10, popsize=6, seed=1, fom="chi2")
    assert "reduced chi2" in out
    out = agent.tool_profile("z_OH", 1.9, 2.2, 0.1)
    assert "profile of z_OH" in out and abs(agent.model.params["z_OH"].value - 2.05) < 0.01
    before = agent.model.params.values()
    cmp = agent.tool_compare_models([
        dict(label="OH only", free=["z_OH", "scale"], fixed=[dict(name="x_OH", value=1.0)]),
        dict(label="mixture", free=["z_OH", "scale", "x_OH"], fixed=[])])
    assert "dBIC" in cmp and agent.model.params.values() == pytest.approx(before)   # state restored
    assert agent.model.params.free_names == ["z_OH", "scale"]
    with pytest.raises(ValueError):
        agent.tool_make_per_dataset(["x_OH"])                                       # one dataset only
    with pytest.raises(ValueError):
        agent.execute("rm_rf", {})


def test_guided_step_and_per_dataset_tools():
    data = [make_dataset(truth_model(x_OH=x), rods=[(0, 0), (0, 1), (1, 0)], noise_rel=0.03, seed=60 + i, step=0.1,
                         name=f"d{i}") for i, x in enumerate((0.3, 0.7))]
    model = start_model([], dict(FILM_TRUTH, theta=0.85, z_OH=2.05, x_OH=0.5))
    agent = FitAgent(model, data, client=None)
    assert "ds1.x_OH" in agent.tool_make_per_dataset(["x_OH"])
    out = agent.tool_guided_step("surface", maxiter=8, popsize=6, seed=1)
    assert "surface step done" in out and "ds2.x_OH" in out
    v = agent.model.params.values()
    assert v["ds1.x_OH"] < v["ds2.x_OH"]


def test_ask_claude_panel(monkeypatch):
    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets
    from ctrfit.app.fit_window import FitWindow
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = FitWindow()
    win.show()
    model, data = surface_setup()
    win.project.model = model
    win.datasets.append(data)
    win._refresh_all()
    script = [response(call(1, "set_parameters", changes=[dict(name="z_OH", value=None, min=1.6, max=2.5, fit=True),
                                                          dict(name="scale", value=None, min=None, max=None,
                                                               fit=True)])),
              response(call(2, "run_global_search", maxiter=8, popsize=6, seed=1, fom="chi2")),
              response(call(3, "refine")),
              response(text("Done: z_OH fitted."))]
    panel = win.assistant
    panel.agent = FitAgent(win.model, win.datasets, client=FakeClient(script))
    seen = []
    orig = win._on_best
    monkeypatch.setattr(win, "_on_best", lambda *a: (seen.append(a[0]), orig(*a)))
    panel.ask("Fit z_OH please")
    assert win.worker is not None and win.table.locked
    assert win.wait(120)
    assert "Done: z_OH fitted." in panel.chat.toPlainText() and "run_global_search" in panel.chat.toPlainText()
    assert abs(win.model.params["z_OH"].value - 2.05) < 0.01 and win.result is not None   # applied
    assert "de" in seen                                                                      # live curve updates
    assert "Tokens:" in panel.cost.text() and panel.undo.isEnabled()
    panel.undo_changes()
    assert win.model.params["z_OH"].value == 1.9
    win.close()

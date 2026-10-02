"""'Ask Claude' panel of the fitting window: Claude runs the fit through ctrfit's tools.

Claude works on a copy of the model; the window shows the live model curve while it fits, applies
the result when it answers, and keeps the previous model for Undo. Needs `pip install anthropic`
and an API key (ANTHROPIC_API_KEY, or a profile from `ant auth login`).
"""
import html
import json
import traceback
from pathlib import Path

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    Signal = QtCore.pyqtSignal
except ImportError:  # pragma: no cover
    from PySide6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.Signal

from ..assistant.agent import AgentStopped, FitAgent, transcript_text
from ..data.dataset import Dataset
from ..fit.fit import Fit

QUICK = [
    ("Fit everything", "Fit this data with the guided workflow (film on the even rods, then the surface on the odd "
                       "rods, then everything together). Tell me the result and how reliable each value is."),
    ("Film only", "Fit only the film parameters on the even rods, including a thickness profile, and report."),
    ("OH vs H2O?", "Can these data tell OH from H2O on the CUS sites? Compare an OH-only, an H2O-only and a mixed "
                   "model with AIC/BIC and tell me what the data support."),
    ("Check the fit", "Look at the current fit: which rods misfit, which parameters are poorly determined, and what "
                      "would you change? Do not change anything yet."),
]


class AgentSignals(QtCore.QObject):
    event = Signal(str, object)
    done = Signal(object)
    failed = Signal(str)


class AgentWorker(QtCore.QRunnable):
    def __init__(self, agent, text):
        super().__init__()
        self.agent, self.text = agent, text
        self.signals = AgentSignals()
        agent.on_event = lambda kind, data: self.signals.event.emit(kind, data)

    def stop(self):
        self.agent.stop()

    def run(self):
        try:
            out = self.agent.ask(self.text)
        except AgentStopped:
            self.signals.failed.emit("stopped")
        except Exception as ex:  # API errors, missing package or key: show them, keep the window alive
            self.signals.failed.emit(_explain(ex))
        else:
            self.signals.done.emit(out)


def _explain(ex):
    name = type(ex).__name__
    if name == "AuthenticationError" or "api_key" in str(ex).lower() or "auth" in name.lower():
        return ("Claude could not authenticate. Set ANTHROPIC_API_KEY (or run `ant auth login`) and start the "
                f"window again.\n({name}: {ex})")
    if isinstance(ex, RuntimeError) and "anthropic" in str(ex):
        return str(ex)
    return "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))[-2000:]


class AssistantPanel(QtWidgets.QWidget):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.agent = None
        self.worker = None
        self._undo = None
        self._synced = False
        self.chat = QtWidgets.QTextBrowser()
        self.chat.setOpenExternalLinks(False)
        self.chat.setPlaceholderText("Ask Claude to fit the data. It uses the same tools as the window "
                                     "(guided steps, DE, refine, thickness profile, report, model comparison) "
                                     "and works on a copy of the model.")
        self.request = QtWidgets.QPlainTextEdit()
        self.request.setPlaceholderText("e.g. Fit the film first, then tell me whether x_OH is determined.")
        self.request.setMaximumHeight(80)
        self.effort = QtWidgets.QComboBox()
        self.effort.addItems(["low", "medium", "high", "xhigh"])
        self.effort.setCurrentText("high")
        self.effort.setToolTip("How hard Claude thinks (more effort: slower and more tokens)")
        self.send = QtWidgets.QPushButton("Send to Claude")
        self.send.setMinimumHeight(30)
        self.undo = QtWidgets.QPushButton("Undo Claude's changes")
        self.undo.setEnabled(False)
        self.save = QtWidgets.QPushButton("Save transcript…")
        self.new = QtWidgets.QPushButton("New conversation")
        self.cost = QtWidgets.QLabel("")
        quick = QtWidgets.QHBoxLayout()
        for label, text in QUICK:
            b = QtWidgets.QPushButton(label)
            b.setToolTip(text)
            b.clicked.connect(lambda _=False, t=text: self.ask(t))
            quick.addWidget(b)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Effort"))
        row.addWidget(self.effort)
        row.addWidget(self.send, 1)
        row2 = QtWidgets.QHBoxLayout()
        for w in (self.undo, self.new, self.save):
            row2.addWidget(w)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.chat, 1)
        lay.addLayout(quick)
        lay.addWidget(self.request)
        lay.addLayout(row)
        lay.addLayout(row2)
        lay.addWidget(self.cost)
        self.send.clicked.connect(lambda: self.ask(self.request.toPlainText()))
        self.undo.clicked.connect(self.undo_changes)
        self.save.clicked.connect(lambda: self.save_transcript())
        self.new.clicked.connect(self.new_conversation)

    # ------------------------------------------------------------------ chat display
    def _add(self, who, text, color="#222"):
        body = html.escape(str(text)).replace("\n", "<br>")
        self.chat.append(f'<p style="margin:4px 0"><b style="color:{color}">{who}</b> {body}</p>')
        self.chat.moveCursor(QtGui.QTextCursor.End)

    def _on_event(self, kind, data):
        if kind == "best":
            self.owner._on_best(data["stage"], data["step"], data["fom"], data["values"])
        elif kind == "text":
            self._add("Claude:", data, "#1f77b4")
        elif kind == "thinking":
            self._add("(thinking)", data, "#888")
        elif kind == "tool_call":
            args = json.dumps(data["input"]) if data["input"] else ""
            self._add("→", f"{data['name']}({args[:300]})", "#555")
            self.owner.say(f"Claude: {data['name']}…")
        elif kind == "tool_result":
            out = str(data["output"])
            self._add("←" if not data["error"] else "← error:", out if len(out) < 600 else out[:600] + " …",
                      "#999" if not data["error"] else "#c0392b")
        elif kind == "usage":
            self.cost.setText(f"Tokens: {data['input'] + data['cache_read']:,} in ({data['cache_read']:,} cached), "
                              f"{data['output']:,} out; about ${data['cost_usd']:.2f} so far")
        elif kind == "error":
            self._add("Note:", data, "#c0392b")

    # ------------------------------------------------------------------ running
    def ask(self, text):
        text = text.strip()
        if not text:
            return
        if self.owner.worker is not None:
            self.owner.say("A fit is running; stop it first.")
            return
        if not self.owner.datasets:
            self.owner.warn("No data", "Add a dataset before asking Claude to fit it.")
            return
        if self.agent is None:
            self.agent = FitAgent(self.owner.model, self.owner.datasets, effort=self.effort.currentText())
        else:                                  # keep the conversation, take the window's current state
            self.agent.model = self.owner.model.copy()
            self.agent.datasets = [Dataset.from_dict(ds.to_dict()) for ds in self.owner.datasets]
            self.agent.effort = self.effort.currentText()
            text = "(The model or data may have changed since your last turn; check get_state.)\n" + text
        self._undo = self.owner.model.copy()
        self._add("You:", text.split("\n", 1)[-1] if text.startswith("(The model") else text, "#000")
        self.request.clear()
        w = AgentWorker(self.agent, text)
        w.signals.event.connect(self._on_event)
        w.signals.done.connect(self._done)
        w.signals.failed.connect(self._failed)
        self.worker = w
        self.owner.worker = w                  # the window's Stop button and busy state cover Claude too
        self.owner.history = []
        self.owner._busy(True)
        self._set_busy(True)
        self.owner.say("Claude is working…")
        self.owner.pool.start(w)

    def _set_busy(self, on):
        for w in (self.send, self.new, self.undo):
            w.setEnabled(not on and (w is not self.undo or self._undo is not None))
        self.send.setText("Claude is working… (Stop: Run fit button)" if on else "Send to Claude")

    def _finish(self):
        self.worker = None
        self.owner.worker = None
        self.owner.table.live = {}
        self.owner._busy(False)
        self._set_busy(False)
        self._apply()

    def _apply(self):
        """Take over Claude's model (and dataset scopes); keep the previous model for Undo."""
        a, o = self.agent, self.owner
        o.project.model = a.model.copy()
        for mine, theirs in zip(o.datasets, a.datasets):
            mine.scope = theirs.scope
        if a.result is not None:
            o.result = a.result
            o.fit = Fit(o.model, o.datasets)
            o.project.results.append(a.result.to_dict())
        o._refresh_all()
        if o.result is not None:
            o._show_result()
        self.undo.setEnabled(self._undo is not None)

    def _done(self, final):
        self._finish()
        self.owner.say("Claude finished" + (f" (about ${self.agent.cost_usd:.2f})" if self.agent else ""))

    def _failed(self, msg):
        stopped = msg == "stopped"
        self._add("Note:", "Stopped. The model keeps the best values reached so far (Undo restores the previous "
                           "model)." if stopped else msg, "#c0392b")
        self._finish()

    def undo_changes(self):
        if self._undo is None or self.owner.worker is not None:
            return
        self.owner.project.model = self._undo
        self._undo = None
        self.owner.result = None
        self.owner._refresh_all()
        self.undo.setEnabled(False)
        self._add("Note:", "Restored the model from before Claude's last turn.", "#555")

    def new_conversation(self):
        self.agent = None
        self.chat.clear()
        self.cost.setText("")

    def save_transcript(self, path=None):
        if self.agent is None:
            return
        path = path or QtWidgets.QFileDialog.getSaveFileName(self, "Save transcript", "claude_fit_transcript.txt",
                                                             "Text (*.txt)")[0]
        if path:
            Path(path).write_text(transcript_text(self.agent.log), encoding="utf-8")
            self.owner.say(f"Saved {Path(path).name}")

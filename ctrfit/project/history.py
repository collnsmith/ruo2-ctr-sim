"""Undo / redo over snapshots of a window's state (the fitting window keeps its model, scan choices
and shown result this way, so a fit that made things worse can be taken back).

    h = History()
    h.push(previous_state)              # after each change: the state before it
    state = h.back(current_state)       # Ctrl+Z: the previous state (the current one goes to redo)
    state = h.forward(current_state)    # Ctrl+Y

States are opaque (dicts with a 'label'); a new change after an undo drops the redo list.
"""


class History:
    def __init__(self, limit=50):
        self.limit = int(limit)
        self.undo, self.redo = [], []

    def push(self, state):
        self.undo.append(state)
        del self.undo[:-self.limit]
        self.redo.clear()

    def back(self, current):
        """The state before the current one (None if there is none); the current one can be redone."""
        if not self.undo:
            return None
        self.redo.append(current)
        return self.undo.pop()

    def forward(self, current):
        if not self.redo:
            return None
        self.undo.append(current)
        return self.redo.pop()

    def clear(self):
        self.undo, self.redo = [], []

    @property
    def undo_label(self):
        return self.undo[-1].get("label", "") if self.undo else ""

    @property
    def redo_label(self):
        return self.redo[-1].get("label", "") if self.redo else ""

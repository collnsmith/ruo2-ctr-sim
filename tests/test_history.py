"""Undo / redo history of window states."""
from ctrfit.project.history import History


def test_back_forward_and_a_new_change_drops_redo():
    h = History(limit=3)
    for i in range(5):
        h.push(dict(label=f"s{i}"))
    assert [s["label"] for s in h.undo] == ["s2", "s3", "s4"]          # the limit keeps the newest
    cur = dict(label="now")
    prev = h.back(cur)
    assert prev["label"] == "s4" and h.redo_label == "now"
    assert h.forward(prev)["label"] == "now" and h.undo_label == "s4"
    h.back(cur)
    h.push(dict(label="new"))
    assert not h.redo and h.forward(cur) is None
    h.clear()
    assert h.back(cur) is None and h.undo_label == ""

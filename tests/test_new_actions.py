import json
import types

from app.models import Action, ActionType
from app.safety import SafetyManager
from app.text_editor import TextEditorTool
from app.tools import ToolExecutor


def test_new_actions(monkeypatch, workspace):
    calls = {}
    pg = types.SimpleNamespace(
        moveTo=lambda *a, **k: calls.setdefault("moveTo", []).append((a, k)),
        scroll=lambda v: calls.setdefault("scroll", []).append(v),
        doubleClick=lambda *a, **k: calls.setdefault("doubleClick", []).append((a, k)),
        click=lambda *a, **k: calls.setdefault("click", []).append((a, k)),
        dragTo=lambda *a, **k: calls.setdefault("dragTo", []).append((a, k)),
        hotkey=lambda *a: calls.setdefault("hotkey", []).append(a),
        keyDown=lambda k: calls.setdefault("keyDown", []).append(k),
        keyUp=lambda k: calls.setdefault("keyUp", []).append(k),
        position=lambda: (5, 7),
        write=lambda x: calls.setdefault("write", []).append(x),
    )
    monkeypatch.setitem(__import__("sys").modules, "pyautogui", pg)
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

    t = ToolExecutor(workspace, text_editor=TextEditorTool(workspace))
    assert t.scroll(1, 2, "down", 3).ok
    assert calls["scroll"][-1] == -3
    assert t.key_combo("ctrl+shift+t").ok
    assert calls["hotkey"][-1] == ("ctrl", "shift", "t")
    assert t.wait_action(2).ok
    assert slept[-1] == 2
    assert t.double_click(1, 1).ok and t.right_click(1, 1).ok and t.middle_click(1, 1).ok
    assert t.mouse_move(1, 1).ok and t.left_click_drag(1, 1, 2, 2).ok and t.hold_key("a", 1).ok
    out = t.cursor_position()
    assert json.loads(out.output) == {"x": 5, "y": 7}


def test_safety_key_combo():
    s = SafetyManager()
    dec = s.evaluate(Action(id="1", type=ActionType.key_combo, args={"keys": "ctrl+alt+del"}))
    assert dec.danger.value == "high"

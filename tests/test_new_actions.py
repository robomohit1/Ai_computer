import json
import pytest
import types

from app.models import Action, ActionType
from app.safety import SafetyManager
from app.text_editor import TextEditorTool
from app.tools import ToolExecutor

@pytest.mark.asyncio
async def test_new_actions(monkeypatch, workspace):
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
        write=lambda x, **kw: calls.setdefault("write", []).append(x),
        size=lambda: (1920, 1080)
    )
    monkeypatch.setitem(__import__("sys").modules, "pyautogui", pg)
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

    t = ToolExecutor(workspace, text_editor=TextEditorTool(workspace))
    
    assert (await t.run_action(Action(id="1", type=ActionType.scroll, args={"amount": 3, "x": 1, "y": 2}))).ok
    assert calls["scroll"][-1] == 3
    
    assert (await t.run_action(Action(id="2", type=ActionType.key_combo, args={"keys": "ctrl+shift+t"}))).ok
    assert calls["hotkey"][-1] == ("ctrl", "shift", "t")
    
    assert (await t.run_action(Action(id="3", type=ActionType.wait_action, args={"seconds": 2}))).ok
    assert slept[-1] == 2
    
    assert (await t.run_action(Action(id="4", type=ActionType.double_click, args={"x": 1, "y": 1}))).ok
    assert (await t.run_action(Action(id="5", type=ActionType.right_click, args={"x": 1, "y": 1}))).ok
    assert (await t.run_action(Action(id="6", type=ActionType.middle_click, args={"x": 1, "y": 1}))).ok
    
    assert (await t.run_action(Action(id="7", type=ActionType.mouse_move, args={"x": 1, "y": 1}))).ok
    assert (await t.run_action(Action(id="8", type=ActionType.left_click_drag, args={"x": 2, "y": 2}))).ok
    assert (await t.run_action(Action(id="9", type=ActionType.hold_key, args={"key": "a", "duration": 1}))).ok
    
    out = await t.run_action(Action(id="10", type=ActionType.cursor_position, args={}))
    assert out.data == {"x": 5, "y": 7}

def test_safety_key_combo():
    s = SafetyManager()
    dec = s.evaluate(Action(id="1", type=ActionType.key_combo, args={"keys": "ctrl+alt+del"}))
    assert dec.danger.value == "high"

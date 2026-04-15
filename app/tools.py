from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .models import Action, ActionType, ToolError, ToolResult
from .providers import get_scale_factor


class ToolExecutor:
    def __init__(self, workspace: Path, text_editor=None, plugin_registry=None):
        self.workspace = workspace.resolve()
        self.text_editor = text_editor
        self.plugin_registry = plugin_registry

    def _safe_path(self, value: str) -> Path:
        candidate = (self.workspace / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
        if self.workspace not in candidate.parents and candidate != self.workspace:
            raise ToolError("Path escapes workspace")
        return candidate

    def mouse_click(self, x: int, y: int, button: str = "left", screen_width: int = 1280, screen_height: int = 800) -> ToolResult:
        import pyautogui

        scale = get_scale_factor(screen_width, screen_height)
        real_x = int(x / scale)
        real_y = int(y / scale)
        pyautogui.click(real_x, real_y, button=button)
        return ToolResult(ok=True, output=f"Clicked at ({real_x}, {real_y})")

    def keyboard_type(self, text: str) -> ToolResult:
        import pyautogui

        pyautogui.write(text)
        return ToolResult(ok=True, output="typed")

    def scroll(self, x: int, y: int, direction: str = "down", amount: int = 3) -> ToolResult:
        import pyautogui

        pyautogui.moveTo(x, y)
        delta = -abs(amount) if direction == "down" else abs(amount)
        pyautogui.scroll(delta)
        return ToolResult(ok=True, output=f"Scrolled {direction}")

    def double_click(self, x: int, y: int) -> ToolResult:
        import pyautogui

        pyautogui.doubleClick(x, y)
        return ToolResult(ok=True, output="double click")

    def right_click(self, x: int, y: int) -> ToolResult:
        import pyautogui

        pyautogui.click(x, y, button="right")
        return ToolResult(ok=True, output="right click")

    def middle_click(self, x: int, y: int) -> ToolResult:
        import pyautogui

        pyautogui.click(x, y, button="middle")
        return ToolResult(ok=True, output="middle click")

    def mouse_move(self, x: int, y: int, duration: float = 0.2) -> ToolResult:
        import pyautogui

        pyautogui.moveTo(x, y, duration=duration)
        return ToolResult(ok=True, output="moved")

    def left_click_drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.3) -> ToolResult:
        import pyautogui

        pyautogui.moveTo(start_x, start_y)
        pyautogui.dragTo(end_x, end_y, duration=duration, button="left")
        return ToolResult(ok=True, output="dragged")

    def key_combo(self, keys: str) -> ToolResult:
        import pyautogui

        parts = [k.strip() for k in keys.split("+") if k.strip()]
        pyautogui.hotkey(*parts)
        return ToolResult(ok=True, output="combo")

    def hold_key(self, key: str, duration: float = 1.0) -> ToolResult:
        import pyautogui

        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)
        return ToolResult(ok=True, output="held")

    def wait_action(self, seconds: float = 1.0) -> ToolResult:
        time.sleep(seconds)
        return ToolResult(ok=True, output="waited")

    def cursor_position(self) -> ToolResult:
        import pyautogui

        x, y = pyautogui.position()
        return ToolResult(ok=True, output=json.dumps({"x": x, "y": y}))

    def run_action(self, action: Action, screen_width: int = 1280, screen_height: int = 800) -> ToolResult:
        handlers = {
            ActionType.mouse_click: lambda a: self.mouse_click(a.args["x"], a.args["y"], a.args.get("button", "left"), screen_width, screen_height),
            ActionType.keyboard_type: lambda a: self.keyboard_type(a.args["text"]),
            ActionType.scroll: lambda a: self.scroll(a.args["x"], a.args["y"], a.args.get("direction", "down"), a.args.get("amount", 3)),
            ActionType.double_click: lambda a: self.double_click(a.args["x"], a.args["y"]),
            ActionType.right_click: lambda a: self.right_click(a.args["x"], a.args["y"]),
            ActionType.middle_click: lambda a: self.middle_click(a.args["x"], a.args["y"]),
            ActionType.mouse_move: lambda a: self.mouse_move(a.args["x"], a.args["y"], a.args.get("duration", 0.2)),
            ActionType.left_click_drag: lambda a: self.left_click_drag(a.args["start_x"], a.args["start_y"], a.args["end_x"], a.args["end_y"], a.args.get("duration", 0.3)),
            ActionType.key_combo: lambda a: self.key_combo(a.args["keys"]),
            ActionType.hold_key: lambda a: self.hold_key(a.args["key"], a.args.get("duration", 1.0)),
            ActionType.wait_action: lambda a: self.wait_action(a.args.get("seconds", 1.0)),
            ActionType.cursor_position: lambda a: self.cursor_position(),
            ActionType.text_view: lambda a: self.text_editor.view(a.args["path"], a.args.get("view_range")),
            ActionType.text_create: lambda a: self.text_editor.create(a.args["path"], a.args["file_text"]),
            ActionType.text_str_replace: lambda a: self.text_editor.str_replace(a.args["path"], a.args["old_str"], a.args["new_str"]),
            ActionType.text_insert: lambda a: self.text_editor.insert(a.args["path"], a.args["insert_line"], a.args["new_str"]),
            ActionType.text_undo_edit: lambda a: self.text_editor.undo_edit(a.args["path"]),
        }
        if action.type in handlers:
            return handlers[action.type](action)

        if self.plugin_registry:
            plugin_handlers = self.plugin_registry.handlers()
            if action.type.value in plugin_handlers:
                result = plugin_handlers[action.type.value](**action.args)
                return ToolResult(ok=True, output=str(result))

        if action.type == ActionType.finish:
            return ToolResult(ok=True, output="finished")
        raise ToolError(f"Unsupported action {action.type}")

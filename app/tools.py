from __future__ import annotations
import json
import time
import subprocess
import os
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

    def _scale(self, x: int, y: int, screen_width: int, screen_height: int):
        scale = get_scale_factor(screen_width, screen_height)
        return int(x / scale), int(y / scale)

    def mouse_click(self, x: int, y: int, button: str = "left", sw=1280, sh=800):
        import pyautogui
        rx, ry = self._scale(x, y, sw, sh)
        pyautogui.click(rx, ry, button=button)
        return ToolResult(ok=True, output=f"Clicked {button} at {rx}, {ry}")

    def keyboard_type(self, text: str):
        import pyautogui
        pyautogui.write(text)
        return ToolResult(ok=True, output=f"Typed text")

    def run_command(self, command: str):
        try:
            res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30, cwd=self.workspace)
            return ToolResult(ok=res.returncode==0, output=f"STDOUT: {res.stdout}
STDERR: {res.stderr}")
        except Exception as e:
            return ToolResult(ok=False, output=str(e))

    def read_file(self, path: str):
        p = self._safe_path(path)
        return ToolResult(ok=True, output=p.read_text())

    def write_file(self, path: str, content: str):
        p = self._safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return ToolResult(ok=True, output=f"Wrote to {path}")

    def run_action(self, action: Action, sw=1280, sh=800) -> ToolResult:
        handlers = {
            ActionType.mouse_click: lambda a: self.mouse_click(a.args["x"], a.args["y"], a.args.get("button", "left"), sw, sh),
            ActionType.keyboard_type: lambda a: self.keyboard_type(a.args["text"]),
            ActionType.run_command: lambda a: self.run_command(a.args["command"]),
            ActionType.read_file: lambda a: self.read_file(a.args["path"]),
            ActionType.write_file: lambda a: self.write_file(a.args["path"], a.args["content"]),
        }
        if action.type in handlers: return handlers[action.type](action)
        if self.plugin_registry:
            h = self.plugin_registry.handlers()
            if action.type.value in h:
                return ToolResult(ok=True, output=str(h[action.type.value](**action.args)))
        return ToolResult(ok=True, output="Action completed")

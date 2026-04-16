from __future__ import annotations
import json
import time
import subprocess
import os
import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional
import mss
from PIL import Image
try:
    import pytesseract
except ImportError:
    pytesseract = None

from .models import Action, ActionType, ToolError, ToolResult
from .providers import get_scale_factor

class ToolExecutor:
    def __init__(self, workspace: Path, text_editor=None, plugin_registry=None):
        self.workspace = workspace.resolve()
        self.text_editor = text_editor
        self.plugin_registry = plugin_registry
        import pyautogui
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False # Be careful with this

    def _safe_path(self, value: str) -> Path:
        candidate = (self.workspace / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
        if self.workspace not in candidate.parents and candidate != self.workspace:
            raise ToolError("Path escapes workspace")
        return candidate

    def _scale(self, x: int, y: int, sw: int, sh: int):
        import pyautogui
        screen_w, screen_h = pyautogui.size()
        # Scale from agent's view (sw, sh) to real screen (screen_w, screen_h)
        rx = int(x * screen_w / sw)
        ry = int(y * screen_h / sh)
        return rx, ry

    def mouse_move(self, x: int, y: int, sw=1280, sh=800):
        import pyautogui
        rx, ry = self._scale(x, y, sw, sh)
        pyautogui.moveTo(rx, ry)
        return ToolResult(ok=True, output=f"Moved mouse to {rx}, {ry}")

    def mouse_click(self, x: int, y: int, button: str = "left", clicks=1, sw=1280, sh=800):
        import pyautogui
        rx, ry = self._scale(x, y, sw, sh)
        pyautogui.click(rx, ry, button=button, clicks=clicks)
        return ToolResult(ok=True, output=f"Clicked {button} {clicks} times at {rx}, {ry}")

    def left_click_drag(self, x: int, y: int, sw=1280, sh=800):
        import pyautogui
        rx, ry = self._scale(x, y, sw, sh)
        pyautogui.dragTo(rx, ry, button="left")
        return ToolResult(ok=True, output=f"Dragged to {rx}, {ry}")

    def keyboard_type(self, text: str):
        import pyautogui
        pyautogui.write(text, interval=0.01)
        return ToolResult(ok=True, output="Typed text")

    def key(self, text: str):
        import pyautogui
        keys = text.split(" ")
        for k in keys:
            pyautogui.press(k)
        return ToolResult(ok=True, output=f"Pressed keys: {text}")

    def scroll(self, amount: int):
        import pyautogui
        pyautogui.scroll(amount)
        return ToolResult(ok=True, output=f"Scrolled {amount}")

    def screenshot(self):
        import pyautogui
        screen_w, screen_h = pyautogui.size()
        with mss.mss() as sct:
            monitor = {"left": 0, "top": 0, "width": screen_w, "height": screen_h}
            shot = sct.grab(monitor)
            img = Image.frombytes("RGB", shot.size, shot.rgb)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return ToolResult(ok=True, output=f"Screenshot captured", base64_image=b64)

    def cursor_position(self):
        import pyautogui
        x, y = pyautogui.position()
        return ToolResult(ok=True, output=f"Cursor at {x}, {y}", data={"x": x, "y": y})

    def wait_action(self, seconds: float):
        time.sleep(seconds)
        return ToolResult(ok=True, output=f"Waited {seconds} seconds")

    def ocr_image(self):
        if not pytesseract:
            return ToolResult(ok=False, output="pytesseract not installed")
        import pyautogui
        img = pyautogui.screenshot()
        text = pytesseract.image_to_string(img)
        return ToolResult(ok=True, output=text)

    def run_command(self, command: str):
        try:
            res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60, cwd=self.workspace)
            return ToolResult(ok=res.returncode==0, output=f"STDOUT:
{res.stdout}
STDERR:
{res.stderr}")
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
            ActionType.mouse_move: lambda a: self.mouse_move(a.args["x"], a.args["y"], sw, sh),
            ActionType.mouse_click: lambda a: self.mouse_click(a.args["x"], a.args["y"], a.args.get("button", "left"), 1, sw, sh),
            ActionType.double_click: lambda a: self.mouse_click(a.args["x"], a.args["y"], "left", 2, sw, sh),
            ActionType.right_click: lambda a: self.mouse_click(a.args["x"], a.args["y"], "right", 1, sw, sh),
            ActionType.middle_click: lambda a: self.mouse_click(a.args["x"], a.args["y"], "middle", 1, sw, sh),
            ActionType.left_click_drag: lambda a: self.left_click_drag(a.args["x"], a.args["y"], sw, sh),
            ActionType.keyboard_type: lambda a: self.keyboard_type(a.args["text"]),
            ActionType.key_combo: lambda a: self.key(a.args["text"]),
            ActionType.scroll: lambda a: self.scroll(a.args.get("amount", 0)),
            ActionType.screenshot: lambda a: self.screenshot(),
            ActionType.cursor_position: lambda a: self.cursor_position(),
            ActionType.wait_action: lambda a: self.wait_action(a.args.get("seconds", 1.0)),
            ActionType.ocr_image: lambda a: self.ocr_image(),
            ActionType.run_command: lambda a: self.run_command(a.args["command"]),
            ActionType.read_file: lambda a: self.read_file(a.args["path"]),
            ActionType.write_file: lambda a: self.write_file(a.args["path"], a.args["content"]),
        }
        if action.type in handlers:
            try:
                return handlers[action.type](action)
            except Exception as e:
                return ToolResult(ok=False, output=f"Error executing {action.type}: {str(e)}")
        
        if self.plugin_registry:
            h = self.plugin_registry.handlers()
            if action.type.value in h:
                return ToolResult(ok=True, output=str(h[action.type.value](**action.args)))
        
        return ToolResult(ok=False, output=f"Unknown action type: {action.type}")

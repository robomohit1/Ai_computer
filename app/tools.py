from __future__ import annotations
import asyncio
import json
import time
import subprocess
import os
import base64
import io
import shutil
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
from .text_editor import TextEditorTool


class ToolExecutor:
    def __init__(self, workspace: Path, text_editor=None, plugin_registry=None):
        self.workspace = workspace.resolve()
        self.text_editor = TextEditorTool(workspace)
        self.plugin_registry = plugin_registry
        import pyautogui
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False

    def _safe_path(self, value: str) -> Path:
        """Resolve a path, allowing workspace-relative or user-home-absolute paths."""
        candidate = (self.workspace / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
        # Allow paths inside workspace OR inside user's home directory
        home = Path.home().resolve()
        if (self.workspace in candidate.parents or candidate == self.workspace
                or home in candidate.parents or candidate == home):
            return candidate
        raise ToolError(f"Path escapes allowed directories (workspace or home): {value}")

    def _scale(self, x: int, y: int, sw: int, sh: int):
        import pyautogui
        screen_w, screen_h = pyautogui.size()
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

    def key(self, keys: str):
        import pyautogui
        parts = [p.strip() for p in keys.split("+") if p.strip()]
        pyautogui.hotkey(*parts)
        return ToolResult(ok=True, output=f"Pressed hotkey: {keys}")

    def hold_key(self, key: str, duration: float = 0.5):
        import pyautogui
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)
        return ToolResult(ok=True, output=f"Held {key} for {duration}s")

    def scroll(self, amount: int, x: Optional[int] = None, y: Optional[int] = None, sw=1280, sh=800):
        import pyautogui
        if x is not None and y is not None:
            rx, ry = self._scale(x, y, sw, sh)
            pyautogui.moveTo(rx, ry)
        pyautogui.scroll(amount)
        return ToolResult(ok=True, output=f"Scrolled {amount}")

    def type_with_delay(self, text: str, delay: float = 0.05):
        import pyautogui
        pyautogui.write(text, interval=delay)
        return ToolResult(ok=True, output=f"Typed text with {delay}s delay")

    def find_on_screen(self, image_path: str):
        import pyautogui
        try:
            p = self._safe_path(image_path)
            res = pyautogui.locateOnScreen(str(p))
            if res:
                return ToolResult(ok=True, output=f"Found at {res}")
            return ToolResult(ok=False, output="Not found on screen")
        except Exception as e:
            return ToolResult(ok=False, output=str(e))

    def get_clipboard(self):
        import pyperclip
        text = pyperclip.paste()
        return ToolResult(ok=True, output=text)

    def set_clipboard(self, text: str):
        import pyperclip
        pyperclip.copy(text)
        return ToolResult(ok=True, output="Clipboard updated")

    def notify(self, message: str):
        try:
            from plyer import notification
            notification.notify(title="AI Computer", message=message, timeout=5)
            return ToolResult(ok=True, output="Notification sent")
        except ImportError:
            return ToolResult(ok=False, output="plyer not installed")

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
            return ToolResult(ok=True, output="Screenshot captured", base64_image=b64)

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
            res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120, cwd=self.workspace)
            return ToolResult(ok=res.returncode == 0, output=f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, output="Command timed out after 120 seconds.")
        except Exception as e:
            return ToolResult(ok=False, output=str(e))

    def read_file(self, path: str):
        p = self._safe_path(path)
        return ToolResult(ok=True, output=p.read_text())

    def write_file(self, path: str, content: str):
        # LLMs often over-escape newlines in JSON strings as literal \n
        content = content.replace("\\n", "\n").replace("\\t", "\t")
        p = self._safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return ToolResult(ok=True, output=f"Wrote to {path}")

    def move_file(self, source: str, destination: str):
        src = self._safe_path(source)
        dst = self._safe_path(destination)
        shutil.move(str(src), str(dst))
        return ToolResult(ok=True, output=f"Moved {source} to {destination}")

    def system_info(self):
        """Return OS info, home dir, workspace, and common folder paths."""
        import platform
        home = Path.home()
        info = {
            "os": platform.system(),
            "platform": platform.platform(),
            "home": str(home),
            "workspace": str(self.workspace),
            "downloads": str(home / "Downloads"),
            "desktop": str(home / "Desktop"),
            "documents": str(home / "Documents"),
            "cwd": str(Path.cwd()),
            "user": os.environ.get("USERNAME", os.environ.get("USER", "unknown")),
            "python": "python" if platform.system() == "Windows" else "python3",
        }
        return ToolResult(ok=True, output=json.dumps(info, indent=2))

    def list_directory(self, path: str, max_depth: int = 2):
        """List directory contents. Accepts absolute or workspace-relative paths."""
        p = self._safe_path(path)
        if not p.exists():
            return ToolResult(ok=False, output=f"Path does not exist: {path}")
        if not p.is_dir():
            return ToolResult(ok=False, output=f"Not a directory: {path}")
        entries = []
        root_depth = len(p.parts)
        for item in sorted(p.iterdir()):
            depth = len(item.parts) - root_depth
            if depth > max_depth:
                continue
            prefix = "📁 " if item.is_dir() else "📄 "
            size = ""
            if item.is_file():
                try:
                    sz = item.stat().st_size
                    size = f" ({sz:,} bytes)" if sz < 1_000_000 else f" ({sz/1_000_000:.1f} MB)"
                except OSError:
                    pass
            entries.append(f"{prefix}{item.name}{size}")
        if not entries:
            entries = ["(empty directory)"]
        header = f"Directory: {p}\n{'─' * 40}"
        return ToolResult(ok=True, output=header + "\n" + "\n".join(entries[:100]))

    def api_call(self, method: str, url: str, headers: dict = None, body: dict = None):
        import httpx
        resp = httpx.request(method, url, headers=headers or {}, json=body)
        return ToolResult(ok=True, output=resp.text)

    async def run_action(self, action: Action, sw=1280, sh=800) -> ToolResult:
        handlers = {
            ActionType.mouse_move: lambda a: self.mouse_move(a.args["x"], a.args["y"], sw, sh),
            ActionType.mouse_click: lambda a: self.mouse_click(a.args["x"], a.args["y"], a.args.get("button", "left"), 1, sw, sh),
            ActionType.double_click: lambda a: self.mouse_click(a.args["x"], a.args["y"], "left", 2, sw, sh),
            ActionType.right_click: lambda a: self.mouse_click(a.args["x"], a.args["y"], "right", 1, sw, sh),
            ActionType.middle_click: lambda a: self.mouse_click(a.args["x"], a.args["y"], "middle", 1, sw, sh),
            ActionType.left_click_drag: lambda a: self.left_click_drag(a.args["x"], a.args["y"], sw, sh),
            ActionType.keyboard_type: lambda a: self.keyboard_type(a.args["text"]),
            ActionType.key_combo: lambda a: self.key(a.args["keys"]),
            ActionType.hold_key: lambda a: self.hold_key(a.args["key"], a.args.get("duration", 0.5)),
            ActionType.scroll: lambda a: self.scroll(a.args.get("amount", 0), a.args.get("x"), a.args.get("y"), sw, sh),
            ActionType.type_with_delay: lambda a: self.type_with_delay(a.args["text"], a.args.get("delay", 0.05)),
            ActionType.find_on_screen: lambda a: self.find_on_screen(a.args["image_path"]),
            ActionType.get_clipboard: lambda a: self.get_clipboard(),
            ActionType.set_clipboard: lambda a: self.set_clipboard(a.args["text"]),
            ActionType.notify: lambda a: self.notify(a.args["message"]),
            ActionType.screenshot: lambda a: self.screenshot(),
            ActionType.cursor_position: lambda a: self.cursor_position(),
            ActionType.wait_action: lambda a: self.wait_action(a.args.get("seconds", 1.0)),
            ActionType.ocr_image: lambda a: self.ocr_image(),
            ActionType.run_command: lambda a: self.run_command(a.args["command"]),
            ActionType.read_file: lambda a: self.read_file(a.args["path"]),
            ActionType.write_file: lambda a: self.write_file(a.args["path"], a.args["content"]),
            ActionType.move_file: lambda a: self.move_file(a.args["source"], a.args["destination"]),
            ActionType.api_call: lambda a: self.api_call(
                a.args["method"], a.args["url"], a.args.get("headers"), a.args.get("body")
            ),
            ActionType.text_view: lambda a: self.text_editor.view(a.args["path"], a.args.get("view_range")),
            ActionType.text_create: lambda a: self.text_editor.create(a.args["path"], a.args["file_text"]),
            ActionType.text_str_replace: lambda a: self.text_editor.str_replace(
                a.args["path"], a.args["old_str"], a.args["new_str"]
            ),
            ActionType.text_insert: lambda a: self.text_editor.insert(
                a.args["path"], a.args["insert_line"], a.args["new_str"]
            ),
            ActionType.text_undo_edit: lambda a: self.text_editor.undo_edit(a.args["path"]),
            ActionType.finish: lambda a: ToolResult(ok=True, output=a.args.get("reason", "Task marked complete by agent.")),
            ActionType.system_info: lambda a: self.system_info(),
            ActionType.list_directory: lambda a: self.list_directory(a.args.get("path", "."), a.args.get("max_depth", 2)),
        }
        if action.type in handlers:
            try:
                return handlers[action.type](action)
            except Exception as e:
                return ToolResult(ok=False, output=f"Error executing {action.type}: {str(e)}")

        if self.plugin_registry:
            h = self.plugin_registry.handlers()
            if action.type.value in h:
                try:
                    handler = h[action.type.value]
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(**action.args)
                    else:
                        result = handler(**action.args)
                    return ToolResult(ok=True, output=str(result))
                except Exception as e:
                    return ToolResult(ok=False, output=f"Plugin error: {str(e)}")

        return ToolResult(ok=False, output=f"Unknown action type: {action.type}")

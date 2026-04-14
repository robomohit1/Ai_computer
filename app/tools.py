from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .models import ToolResult
from .plugin_loader import PluginRegistry


class ToolError(RuntimeError):
    pass


class ToolExecutor:
    """Executes low-level OS actions requested by the planner."""

    def __init__(self, workspace: Path, plugins: PluginRegistry):
        self.workspace = workspace.resolve()
        self.plugins = plugins

    def _safe_path(self, value: str) -> Path:
        path = (self.workspace / value).resolve() if not os.path.isabs(value) else Path(value).resolve()
        if self.workspace not in path.parents and path != self.workspace:
            raise ToolError(f"Path '{path}' is outside workspace '{self.workspace}'.")
        return path

    def run_action(self, action_type: str, args: Dict[str, Any]) -> ToolResult:
        handlers = {
            "finish": self.finish,
            "run_command": self.run_command,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "move_file": self.move_file,
            "mouse_click": self.mouse_click,
            "keyboard_type": self.keyboard_type,
            "screenshot": self.screenshot,
            "ocr_image": self.ocr_image,
        }
        if action_type in handlers:
            return handlers[action_type](**args)

        plugin_handler = self.plugins.get_handler(action_type)
        if plugin_handler:
            output = plugin_handler(**args)
            return ToolResult(ok=True, action_type=action_type, summary="Plugin action executed", output=str(output))

        raise ToolError(f"Unknown action type: {action_type}")

    def finish(self, message: str = "Task complete") -> ToolResult:
        return ToolResult(ok=True, action_type="finish", summary=message, output=message)

    def run_command(self, command: str, cwd: str = ".", timeout: int = 90) -> ToolResult:
        full_cwd = self._safe_path(cwd)
        process = subprocess.run(
            command,
            cwd=full_cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = f"$ {command}\nexit={process.returncode}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        if process.returncode != 0:
            raise ToolError(output)
        return ToolResult(ok=True, action_type="run_command", summary="Command executed", output=output)

    def read_file(self, path: str) -> ToolResult:
        target = self._safe_path(path)
        if not target.exists():
            raise ToolError(f"File does not exist: {target}")
        text = target.read_text(encoding="utf-8")
        return ToolResult(ok=True, action_type="read_file", summary=f"Read file {target}", output=text[:8000])

    def write_file(self, path: str, content: str) -> ToolResult:
        target = self._safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(ok=True, action_type="write_file", summary=f"Wrote {len(content)} bytes", metadata={"path": str(target)})

    def move_file(self, src: str, dst: str) -> ToolResult:
        source = self._safe_path(src)
        target = self._safe_path(dst)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        return ToolResult(ok=True, action_type="move_file", summary=f"Moved {source} -> {target}")

    def screenshot(self, path: str = "artifacts/screenshot.png") -> ToolResult:
        target = self._safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            import mss  # type: ignore
        except Exception as exc:
            raise ToolError("screenshot requires `mss` installed and a display server") from exc

        with mss.mss() as sct:
            sct.shot(output=str(target))
        return ToolResult(ok=True, action_type="screenshot", summary="Screenshot captured", metadata={"path": str(target)})

    def ocr_image(self, path: str) -> ToolResult:
        target = self._safe_path(path)
        if not target.exists():
            raise ToolError(f"Image not found for OCR: {target}")
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise ToolError("ocr_image requires pytesseract and pillow installed") from exc
        text = pytesseract.image_to_string(Image.open(target))
        return ToolResult(ok=True, action_type="ocr_image", summary="OCR completed", output=text[:5000])

    def mouse_click(self, x: int, y: int, button: str = "left", move_duration: float = 0.2) -> ToolResult:
        try:
            import pyautogui  # type: ignore
        except Exception as exc:
            raise ToolError("mouse_click requires pyautogui installed and GUI session available") from exc
        pyautogui.moveTo(x=x, y=y, duration=move_duration)
        pyautogui.click(x=x, y=y, button=button)
        return ToolResult(ok=True, action_type="mouse_click", summary=f"Clicked {button} at ({x}, {y})")

    def keyboard_type(self, text: str, interval: float = 0.02) -> ToolResult:
        try:
            import pyautogui  # type: ignore
        except Exception as exc:
            raise ToolError("keyboard_type requires pyautogui installed and GUI session available") from exc
        pyautogui.write(text, interval=interval)
        return ToolResult(
            ok=True,
            action_type="keyboard_type",
            summary=f"Typed {len(text)} chars",
            metadata={"ts": datetime.now(timezone.utc).isoformat()},
        )

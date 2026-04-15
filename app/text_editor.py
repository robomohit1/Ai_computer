from __future__ import annotations

from pathlib import Path

from .models import ToolError, ToolResult


class TextEditorTool:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self._history: dict[str, list[str]] = {}

    def _safe_path(self, value: str) -> Path:
        candidate = (self.workspace / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
        if self.workspace not in candidate.parents and candidate != self.workspace:
            raise ToolError("Path escapes workspace")
        return candidate

    def view(self, path: str, view_range: list[int] | None = None) -> ToolResult:
        p = self._safe_path(path)
        if p.is_dir():
            out = []
            root_depth = len(p.parts)
            for item in sorted(p.rglob("*")):
                if len(item.parts) - root_depth <= 2:
                    out.append(str(item.relative_to(p)))
            return ToolResult(ok=True, output="\n".join(out))
        if not p.exists():
            raise ToolError("File not found")
        lines = p.read_text().splitlines()
        start, end = 1, len(lines)
        if view_range:
            start, end = view_range
        numbered = [f"{i:4d}: {line}" for i, line in enumerate(lines, start=1) if start <= i <= end]
        return ToolResult(ok=True, output="\n".join(numbered))

    def create(self, path: str, file_text: str) -> ToolResult:
        p = self._safe_path(path)
        if p.exists():
            raise ToolError("File already exists")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(file_text)
        return ToolResult(ok=True, output=f"Created {p}")

    def str_replace(self, path: str, old_str: str, new_str: str) -> ToolResult:
        p = self._safe_path(path)
        text = p.read_text()
        count = text.count(old_str)
        if count == 0:
            raise ToolError("old_str not found. Provide more precise context.")
        if count > 1:
            raise ToolError("old_str appears multiple times; please disambiguate.")
        self._history.setdefault(str(p), []).append(text)
        p.write_text(text.replace(old_str, new_str))
        return ToolResult(ok=True, output="Replaced 1 occurrence")

    def insert(self, path: str, insert_line: int, new_str: str) -> ToolResult:
        p = self._safe_path(path)
        text = p.read_text() if p.exists() else ""
        self._history.setdefault(str(p), []).append(text)
        lines = text.splitlines()
        idx = max(0, min(insert_line, len(lines)))
        lines[idx:idx] = new_str.splitlines()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + ("\n" if lines else ""))
        return ToolResult(ok=True, output="Inserted")

    def undo_edit(self, path: str) -> ToolResult:
        p = self._safe_path(path)
        key = str(p)
        if key not in self._history or not self._history[key]:
            raise ToolError("No edit history for path")
        old = self._history[key].pop()
        p.write_text(old)
        return ToolResult(ok=True, output="Undo complete")

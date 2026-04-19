from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from PIL import Image

from .models import HierarchicalPlan

SYSTEM_PROMPT = """You are a computer control planner. Use action types: finish, run_command, read_file, write_file, move_file,
mouse_click, keyboard_type, screenshot, ocr_image, api_call, scroll, double_click, right_click, middle_click,
mouse_move, left_click_drag, key_combo, hold_key, wait_action, cursor_position, text_view, text_create,
text_str_replace, text_insert, text_undo_edit, browser_open, browser_screenshot, browser_click,
browser_click_coords, browser_type, browser_scroll, browser_get_text, browser_accessibility_tree,
browser_navigate_back, browser_close, type_with_delay, find_on_screen, get_clipboard, set_clipboard, notify."""

HIERARCHICAL_SYSTEM_PROMPT = """You are a hierarchical planning engine for an autonomous computer agent.
Return ONLY valid JSON with shape:
{
  "reasoning": str,
  "overall_complete": bool,
  "sub_tasks": [
    {
      "id": str,
      "description": str,
      "actions": [{"id": str, "type": str, "args": object, "explanation": str, "requires_approval": bool}]
    }
  ]
}
Decompose the goal into 2-8 sequential sub-tasks. Each sub-task should be independently verifiable.
Use only action types: finish, run_command, read_file, write_file, move_file, mouse_click, keyboard_type,
screenshot, ocr_image, api_call, scroll, double_click, right_click, middle_click, mouse_move, left_click_drag,
key_combo, hold_key, wait_action, cursor_position, text_view, text_create, text_str_replace, text_insert,
text_undo_edit, browser_open, browser_screenshot, browser_click, browser_click_coords, browser_type,
browser_scroll, browser_get_text, browser_accessibility_tree, browser_navigate_back, browser_close, type_with_delay, find_on_screen, get_clipboard, set_clipboard, notify.
Never output markdown. Never output prose outside JSON."""

# ──────────────────────────────────────────────────────────────────────────────
#  CODING MODE PROMPTS  — no screenshots, no mouse/keyboard/vision actions
# ──────────────────────────────────────────────────────────────────────────────
CODING_SYSTEM_PROMPT = """You are an expert autonomous coding agent. You write, read, and execute code.
Return ONLY valid JSON with shape:
{
  "reasoning": str,
  "overall_complete": bool,
  "sub_tasks": [
    {
      "id": str,
      "description": str,
      "actions": [{"id": str, "type": str, "args": object, "explanation": str, "requires_approval": false}]
    }
  ]
}
Decompose the goal into 2-8 sequential sub-tasks. Each sub-task should be independently verifiable.

Available action types:
- system_info: {}  — returns OS, home dir, workspace, Downloads/Desktop/Documents paths, python command. ALWAYS call this first.
- list_directory: {"path": str, "max_depth": int}  — list contents of any directory (absolute or relative)
- write_file: {"path": str, "content": str}  — create/overwrite a file
- read_file: {"path": str}  — read a file's contents
- run_command: {"command": str}  — run a shell command (install deps, run tests, etc.)
- text_create: {"path": str, "file_text": str}  — create a new file (fails if exists)
- text_view: {"path": str, "view_range": [start, end] | null}  — view file lines or directory listing
- text_str_replace: {"path": str, "old_str": str, "new_str": str}  — precise find & replace
- text_insert: {"path": str, "insert_line": int, "new_str": str}  — insert at line number
- text_undo_edit: {"path": str}  — undo last edit to a file
- move_file: {"source": str, "destination": str}  — rename/move a file
- finish: {"reason": str}  — mark task as complete

Rules:
1. The system environment (OS, paths, python command) is provided in the prompt. Use those EXACT paths.
2. For project/code files: use relative paths (resolved from the workspace directory).
3. When the user mentions system folders (Downloads, Desktop, Documents, etc.): use ABSOLUTE paths from environment info.
4. Use list_directory to explore or verify folder contents when needed.
5. Create directories automatically via write_file (parents are auto-created).
6. For large files, use write_file. For surgical edits, use text_str_replace.
7. Always verify your work: after writing code, run it or read it back.
8. Do NOT use mouse, keyboard, screenshot, or browser actions. Those are not available.
9. When you generate action ids, use short descriptive strings like "create-main", "run-test", etc.
10. In file content strings, use actual newline characters.
Never output markdown. Never output prose outside JSON."""

CODING_REFLECT_PROMPT = """You are a reflection agent for an autonomous coding agent.
Given a completed sub-task description, the actions that ran, and their outputs (stdout/stderr/file contents),
determine if the sub-task succeeded.
Return ONLY valid JSON: {"success": bool, "reason": str, "retry_actions": []}
If success is false, optionally populate retry_actions with corrective action objects using ONLY these types:
write_file, read_file, run_command, text_create, text_view, text_str_replace, text_insert, text_undo_edit, move_file, finish.
Never output markdown. Never output prose outside JSON."""

CODING_EVALUATE_PROMPT = """You are an evaluation agent for an autonomous coding agent.
Given a goal, the action history (file writes, command outputs, etc.), determine if the overall goal is complete.
Return ONLY valid JSON: {"complete": bool, "reason": str}
Never output markdown. Never output prose outside JSON."""


# ──────────────────────────────────────────────────────────────────────────────
#  COMPUTER USE MODE  — DOM/accessibility-tree based, NO screenshots.
#  Tuned for small free models: short prompt, narrow action vocabulary.
# ──────────────────────────────────────────────────────────────────────────────
COMPUTER_USE_SYSTEM_PROMPT = """You are a browser-automation agent. You read pages as text via the accessibility tree — NEVER assume pixel coordinates.
Return ONLY valid JSON with shape:
{
  "reasoning": str,
  "overall_complete": bool,
  "sub_tasks": [
    {
      "id": str,
      "description": str,
      "actions": [{"id": str, "type": str, "args": object, "explanation": str, "requires_approval": false}]
    }
  ]
}

Available actions (use ONLY these):
- request_permission: {"scope": "browser"|"google_sheets", "reason": str}  — MUST be the FIRST action. Ask the user for access.
- browser_open: {"url": str}  — open a URL (full https:// URL required)
- browser_accessibility_tree: {}  — read the current page as a structured text tree (use this BEFORE clicking/typing to find what's on the page)
- browser_click: {"selector": str}  — CSS selector (e.g. 'button[name="btnK"]', 'input[type="submit"]', '#id', '.class')
- browser_type: {"selector": str, "text": str}  — fill a text field
- browser_scroll: {"direction": "up"|"down", "amount": int}
- browser_get_text: {}  — get visible page text (fallback if accessibility tree is too noisy)
- browser_navigate_back: {}
- browser_close: {}
- wait_action: {"seconds": float}  — wait for page to load (use 1-3 seconds after navigation)
- finish: {"reason": str}

Rules:
1. Your FIRST action in the first sub-task MUST be request_permission with the right scope (google_sheets if the task involves Google Sheets, otherwise browser).
2. After browser_open or any click that navigates, wait 1-2 seconds then call browser_accessibility_tree to see the new page state.
3. Use CSS selectors based on the accessibility tree output. Prefer stable selectors: input[type=...], button[aria-label=...], #id, [role=...].
4. NEVER use pixel coordinates. NEVER use mouse_click, keyboard_type, or screenshot. Those are blocked in this mode.
5. For Google Sheets: open https://docs.google.com/spreadsheets/ and use browser_accessibility_tree to see cells. Click a cell then browser_type to write.
6. Keep sub-tasks small (2-4 actions each) so reflection can correct errors quickly.
Never output markdown. Never output prose outside JSON."""


COMPUTER_USE_REFLECT_PROMPT = """You are a reflection agent for a browser-automation task.
Given a sub-task description, the actions that ran, and their outputs (URLs, page text, accessibility trees),
determine if the sub-task succeeded.
Return ONLY valid JSON: {"success": bool, "reason": str, "retry_actions": []}
If success is false, optionally populate retry_actions with corrective actions using ONLY these types:
request_permission, browser_open, browser_accessibility_tree, browser_click, browser_type,
browser_scroll, browser_get_text, browser_navigate_back, browser_close, wait_action, finish.
Never output markdown. Never output prose outside JSON."""


COMPUTER_USE_EVALUATE_PROMPT = """You are an evaluation agent for a browser-automation task.
Given a goal and the action history (URLs visited, page text observed, form submissions), determine if the goal is complete.
Return ONLY valid JSON: {"complete": bool, "reason": str}
Never output markdown. Never output prose outside JSON."""

REFLECT_SYSTEM_PROMPT = """You are a reflection agent for an autonomous computer agent.
Given a completed sub-task description, the actions that ran, their results, and a screenshot of the
current screen, determine if the sub-task succeeded.
Return ONLY valid JSON: {"success": bool, "reason": str, "retry_actions": []}
If success is false, optionally populate retry_actions with corrective action objects.
Never output markdown. Never output prose outside JSON."""

EVALUATE_SYSTEM_PROMPT = """You are an evaluation agent for an autonomous computer agent.
Given a goal, the action history, and the current screenshot, determine if the overall goal is complete.
Return ONLY valid JSON: {"complete": bool, "reason": str}
Never output markdown. Never output prose outside JSON."""


# ──────────────────────────────────────────────────────────────────────────────
#  Task mode detection
# ──────────────────────────────────────────────────────────────────────────────
_CODING_KEYWORDS = [
    "write", "code", "script", "function", "class", "file", "create", "build",
    "implement", "refactor", "debug", "fix", "test", "install", "pip", "npm",
    "python", "javascript", "typescript", "html", "css", "react", "node",
    "api", "server", "database", "sql", "json", "yaml", "config", "setup",
    "project", "app", "module", "package", "library", "framework", "deploy",
    "dockerfile", "git", "commit", "repository", "repo", "compile", "lint",
    "format", "parse", "generate", "scaffold", "boilerplate", "template",
    "algorithm", "data structure", "endpoint", "route", "middleware",
    "component", "hook", "state", "reducer", "model", "schema", "migration",
    "makefile", "cmake", "cargo", "gradle", "maven", "webpack", "vite",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".cpp",
    ".c", ".h", ".rb", ".php", ".swift", ".kt", ".sh", ".bash",
]

_COMPUTER_KEYWORDS = [
    "open", "click", "type into", "browser", "screenshot", "mouse", "scroll",
    "desktop", "window", "drag", "notepad", "chrome", "firefox", "visual",
    "screen", "navigate", "tab", "menu", "button", "gui", "interface",
    "application", "launch", "icon", "taskbar", "cursor",
]


_COMPUTER_USE_KEYWORDS = [
    "browser", "chrome", "firefox", "web", "website", "google search",
    "google sheets", "spreadsheet", "sheets", "docs.google", "gmail",
    "youtube", "wikipedia", "navigate to", "open url", "open site",
    "fill out form", "search for", "submit form", "log into", "log in to",
    "sign in to", "visit", "webpage",
]


def detect_task_mode(goal: str, explicit_mode: Optional[str] = None) -> str:
    """Return 'coding', 'computer_use', or 'computer'. If explicit_mode is set, honour it."""
    if explicit_mode and explicit_mode in ("coding", "computer", "computer_use"):
        return explicit_mode
    g = goal.lower()
    computer_use_score = sum(1 for kw in _COMPUTER_USE_KEYWORDS if kw in g)
    coding_score = sum(1 for kw in _CODING_KEYWORDS if kw in g)
    if computer_use_score >= 2 and computer_use_score > coding_score:
        return "computer_use"
    return "coding"


def get_scale_factor(width: int, height: int) -> float:
    long_edge_scale = 1568 / max(width, height)
    total_pixels_scale = math.sqrt(1_150_000 / (width * height))
    return min(1.0, long_edge_scale, total_pixels_scale)


def _capture_screenshot_b64(width: int, height: int) -> str:
    import mss

    # Cap at 1280x800
    w = min(width, 1280)
    h = min(height, 800)
    with mss.mss() as sct:
        monitor = {"left": 0, "top": 0, "width": width, "height": height}
        shot = sct.grab(monitor)
        image = Image.frombytes("RGB", shot.size, shot.rgb)
        if image.size[0] > w or image.size[1] > h:
            image.thumbnail((w, h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_json(text: str) -> Any:
    """Extract JSON from LLM response text, handling markdown code fences."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


class PlannerProvider:
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.model = model
        self._anthropic_key: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
        self._openai_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
        self._google_key: Optional[str] = os.environ.get("GOOGLE_API_KEY")
        self._openrouter_key: Optional[str] = os.environ.get("OPENROUTER_API_KEY", "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")
        self._groq_key: Optional[str] = os.environ.get("GROQ_API_KEY")

    def _is_anthropic(self) -> bool:
        return self.model.startswith("claude") and not self.model.startswith("openrouter/")

    def _is_openai(self) -> bool:
        m = self.model.lower()
        return not m.startswith("openrouter/") and ("gpt" in m or "o1" in m or "o3" in m)

    def _is_google(self) -> bool:
        m = self.model.lower()
        return not m.startswith("openrouter/") and (m.startswith("gemini") or m.startswith("google/"))

    def _is_groq(self) -> bool:
        m = self.model.lower()
        return not m.startswith("openrouter/") and (m.startswith("groq/") or "llama" in m or "mixtral" in m or "gemma" in m)

    def _chat_anthropic(self, system: str, prompt: str, screenshot_b64: Optional[str] = None) -> str:
        if not self._anthropic_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
            
        content: List[Any] = [{"type": "text", "text": prompt}]
        if screenshot_b64:
            content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}})
            
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }
        last_err = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=120) as client:
                    resp = client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": self._anthropic_key, "anthropic-version": "2023-06-01"},
                        json=payload,
                    )
                    resp.raise_for_status()
                    return resp.json()["content"][0]["text"]
            except httpx.HTTPStatusError as e:
                last_err = e
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise last_err

    def _chat_openai(self, system: str, prompt: str, screenshot_b64: Optional[str] = None) -> str:
        if not self._openai_key:
            raise RuntimeError("OPENAI_API_KEY not set")
            
        content: List[Any] = [{"type": "text", "text": prompt}]
        if screenshot_b64:
            content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}})
            
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        payload = {"model": self.model, "max_tokens": 4096, "messages": messages}
        last_err = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=120) as client:
                    resp = client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self._openai_key}"},
                        json=payload,
                    )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                last_err = e
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise last_err

    def _chat_openrouter(self, system: str, prompt: str, screenshot_b64: Optional[str] = None) -> str:
        if not self._openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
            
        model = self.model.replace("openrouter/", "")
        
        content: List[Any] = [{"type": "text", "text": prompt}]
        is_vision = any(x in model.lower() for x in ["vision", "vl", "gemini", "claude", "gpt-4o", "gpt-4-turbo", "pixtral", "llava"])
        if screenshot_b64 and is_vision:
            content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}})
            
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        payload = {"model": model, "messages": messages}
        last_err = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=120) as client:
                    resp = client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self._openrouter_key}"},
                        json=payload,
                    )
                    if resp.status_code != 200:
                        print("OPENROUTER ERROR:", resp.text)
                    resp.raise_for_status()
                    resp_json = resp.json()
                    # OpenRouter can return 200 + {"error": {...}} when rate-limited or quota-exceeded
                    if "error" in resp_json:
                        err_msg = resp_json["error"].get("message", str(resp_json["error"]))
                        if attempt < 2:
                            print(f"OPENROUTER SOFT ERROR (attempt {attempt+1}): {err_msg}")
                            time.sleep(2 ** attempt)
                            continue
                        raise RuntimeError(f"OpenRouter error: {err_msg}")
                    if "choices" not in resp_json:
                        raise RuntimeError(f"Unexpected OpenRouter response: {str(resp_json)[:200]}")
                    return resp_json["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                last_err = e
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise last_err

    def _chat_google(self, system: str, prompt: str, screenshot_b64: Optional[str] = None) -> str:
        if not self._google_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
            
        model = self.model.replace("google/", "")
        
        parts: List[Any] = [{"text": prompt}]
        if screenshot_b64:
            parts.insert(0, {"inline_data": {"mime_type": "image/png", "data": screenshot_b64}})
            
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": 4096}
        }
        
        last_err = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=120) as client:
                    resp = client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self._google_key}",
                        json=payload,
                    )
                    resp.raise_for_status()
                    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            except httpx.HTTPStatusError as e:
                last_err = e
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise last_err

    def _chat_groq(self, system: str, prompt: str, screenshot_b64: Optional[str] = None) -> str:
        if not self._groq_key:
            raise RuntimeError("GROQ_API_KEY not set")
            
        model = self.model.replace("groq/", "")
        
        content: List[Any] = [{"type": "text", "text": prompt}]
        if screenshot_b64 and ("llava" in model.lower() or "vision" in model.lower()):
            content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}})
            
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        payload = {"model": model, "max_tokens": 4096, "messages": messages}
        
        last_err = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=120) as client:
                    resp = client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self._groq_key}"},
                        json=payload,
                    )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                last_err = e
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise last_err

    def _call_llm(self, system: str, prompt: str, screenshot_b64: Optional[str] = None) -> str:
        if self._is_groq():
            return self._chat_groq(system, prompt, screenshot_b64)
        if self._is_google():
            return self._chat_google(system, prompt, screenshot_b64)
        if self._is_anthropic():
            return self._chat_anthropic(system, prompt, screenshot_b64)
        if self._is_openai():
            return self._chat_openai(system, prompt, screenshot_b64)
        return self._chat_openrouter(system, prompt, screenshot_b64)

    def plan_hierarchical(
        self,
        goal: str,
        latest_screenshot_b64: Optional[str] = None,
        memory_context: Optional[str] = None,
        mode: str = "computer",
    ) -> HierarchicalPlan:
        prompt = f"Goal: {goal}\n\nDecompose this goal into 2-8 sequential sub-tasks with concrete actions."
        if memory_context:
            prompt = f"Relevant past experience:\n{memory_context}\n\n{prompt}"
        if mode == "coding":
            system = CODING_SYSTEM_PROMPT
            raw_text = self._call_llm(system, prompt)  # no screenshot for coding
        elif mode == "computer_use":
            system = COMPUTER_USE_SYSTEM_PROMPT
            raw_text = self._call_llm(system, prompt)  # no screenshot — DOM-based
        else:
            system = HIERARCHICAL_SYSTEM_PROMPT
            raw_text = self._call_llm(system, prompt, latest_screenshot_b64)
        return HierarchicalPlan.model_validate(_extract_json(raw_text))

    def reflect_on_subtask(
        self,
        description: str,
        actions: List[Dict[str, Any]],
        results: List[str],
        post_screenshot_b64: Optional[str] = None,
        mode: str = "computer",
    ) -> Dict[str, Any]:
        if mode == "coding":
            prompt = (
                f"Sub-task: {description}\n\n"
                f"Actions taken:\n{json.dumps(actions, indent=2)}\n\n"
                f"Results (stdout/stderr/file contents):\n{json.dumps(results, indent=2)}\n\n"
                "Based on the action results, did this sub-task succeed?"
            )
            raw_text = self._call_llm(CODING_REFLECT_PROMPT, prompt)  # no screenshot
        elif mode == "computer_use":
            prompt = (
                f"Sub-task: {description}\n\n"
                f"Actions taken:\n{json.dumps(actions, indent=2)}\n\n"
                f"Results (page text / accessibility trees / URLs):\n{json.dumps(results, indent=2)[:8000]}\n\n"
                "Based on the action results, did this sub-task succeed?"
            )
            raw_text = self._call_llm(COMPUTER_USE_REFLECT_PROMPT, prompt)  # no screenshot
        else:
            prompt = (
                f"Sub-task: {description}\n\n"
                f"Actions taken:\n{json.dumps(actions, indent=2)}\n\n"
                f"Results:\n{json.dumps(results, indent=2)}\n\n"
                "Based on the screenshot and results, did this sub-task succeed?"
            )
            raw_text = self._call_llm(REFLECT_SYSTEM_PROMPT, prompt, post_screenshot_b64)
        return _extract_json(raw_text)

    def evaluate(
        self, goal: str, history: List[str], latest_screenshot_b64: Optional[str] = None,
        mode: str = "computer",
    ) -> Dict[str, Any]:
        recent = history[-20:]
        prompt = f"Goal: {goal}\n\nRecent action history:\n" + "\n".join(recent) + "\n\nIs the overall goal now complete?"
        if mode == "coding":
            raw_text = self._call_llm(CODING_EVALUATE_PROMPT, prompt)  # no screenshot
        elif mode == "computer_use":
            raw_text = self._call_llm(COMPUTER_USE_EVALUATE_PROMPT, prompt)  # no screenshot
        else:
            raw_text = self._call_llm(EVALUATE_SYSTEM_PROMPT, prompt, latest_screenshot_b64)
        return _extract_json(raw_text)

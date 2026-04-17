from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from PIL import Image

from .models import HierarchicalPlan

SYSTEM_PROMPT = """You are a computer control planner. Use action types: finish, run_command, read_file, write_file, move_file,
mouse_click, keyboard_type, screenshot, ocr_image, api_call, scroll, double_click, right_click, middle_click,
mouse_move, left_click_drag, key_combo, hold_key, wait_action, cursor_position, text_view, text_create,
text_str_replace, text_insert, text_undo_edit, browser_open, browser_screenshot, browser_click,
browser_click_coords, browser_type, browser_scroll, browser_get_text, browser_accessibility_tree,
browser_navigate_back, browser_close."""

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
browser_scroll, browser_get_text, browser_accessibility_tree, browser_navigate_back, browser_close.
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


def get_scale_factor(width: int, height: int) -> float:
    long_edge_scale = 1568 / max(width, height)
    total_pixels_scale = math.sqrt(1_150_000 / (width * height))
    return min(1.0, long_edge_scale, total_pixels_scale)


def _capture_screenshot_b64(width: int, height: int) -> str:
    import mss

    with mss.mss() as sct:
        monitor = {"left": 0, "top": 0, "width": width, "height": height}
        shot = sct.grab(monitor)
        image = Image.frombytes("RGB", shot.size, shot.rgb)
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
        self._openrouter_key: Optional[str] = os.environ.get("OPENROUTER_API_KEY")
        self._groq_key: Optional[str] = os.environ.get("GROQ_API_KEY")

    def _is_anthropic(self) -> bool:
        return self.model.startswith("claude")

    def _is_openai(self) -> bool:
        return "gpt" in self.model or "o1" in self.model or "o3" in self.model

    def _is_google(self) -> bool:
        return self.model.startswith("gemini")

    def _chat_anthropic(self, system: str, prompt: str, screenshot_b64: Optional[str] = None) -> str:
        if not self._anthropic_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        if screenshot_b64 is None:
            screenshot_b64 = _capture_screenshot_b64(1280, 800)
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}},
            {"type": "text", "text": prompt},
        ]
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
        if screenshot_b64 is None:
            screenshot_b64 = _capture_screenshot_b64(1280, 800)
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            },
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
        if screenshot_b64 is None:
            screenshot_b64 = _capture_screenshot_b64(1280, 800)
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        payload = {"model": self.model, "messages": messages}
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._openrouter_key}"},
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def _call_llm(self, system: str, prompt: str, screenshot_b64: Optional[str] = None) -> str:
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
    ) -> HierarchicalPlan:
        prompt = f"Goal: {goal}\n\nDecompose this goal into 2-8 sequential sub-tasks with concrete actions."
        if memory_context:
            prompt = f"Relevant past experience:\n{memory_context}\n\n{prompt}"
        raw_text = self._call_llm(HIERARCHICAL_SYSTEM_PROMPT, prompt, latest_screenshot_b64)
        return HierarchicalPlan.model_validate(_extract_json(raw_text))

    def reflect_on_subtask(
        self,
        description: str,
        actions: List[Dict[str, Any]],
        results: List[str],
        post_screenshot_b64: Optional[str],
    ) -> Dict[str, Any]:
        prompt = (
            f"Sub-task: {description}\n\n"
            f"Actions taken:\n{json.dumps(actions, indent=2)}\n\n"
            f"Results:\n{json.dumps(results, indent=2)}\n\n"
            "Based on the screenshot and results, did this sub-task succeed?"
        )
        raw_text = self._call_llm(REFLECT_SYSTEM_PROMPT, prompt, post_screenshot_b64)
        return _extract_json(raw_text)

    def evaluate(
        self, goal: str, history: List[str], latest_screenshot_b64: Optional[str] = None
    ) -> Dict[str, Any]:
        recent = history[-20:]
        prompt = f"Goal: {goal}\n\nRecent action history:\n" + "\n".join(recent) + "\n\nIs the overall goal now complete?"
        raw_text = self._call_llm(EVALUATE_SYSTEM_PROMPT, prompt, latest_screenshot_b64)
        return _extract_json(raw_text)

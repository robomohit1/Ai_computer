from __future__ import annotations

import base64
import io
import json
import math
from typing import Any, Dict, List, Optional

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


class PlannerProvider:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def _chat_anthropic(self, prompt: str, screenshot_b64: Optional[str] = None) -> Dict[str, Any]:
        if screenshot_b64 is None:
            screenshot_b64 = _capture_screenshot_b64(1280, 800)
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

    def _chat_openai(self, prompt: str, screenshot_b64: Optional[str] = None) -> Dict[str, Any]:
        if screenshot_b64 is None:
            screenshot_b64 = _capture_screenshot_b64(1280, 800)
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

    def plan_hierarchical(self, goal: str, latest_screenshot_b64: Optional[str] = None) -> HierarchicalPlan:
        raw = {
            "reasoning": f"Plan for {goal}",
            "overall_complete": False,
            "sub_tasks": [
                {
                    "id": "st-1",
                    "description": goal,
                    "actions": [
                        {
                            "id": "a-finish",
                            "type": "finish",
                            "args": {},
                            "explanation": "Done",
                            "requires_approval": False,
                        }
                    ],
                }
            ],
        }
        return HierarchicalPlan.model_validate(raw)

    def reflect_on_subtask(
        self,
        description: str,
        actions: List[Dict[str, Any]],
        results: List[str],
        post_screenshot_b64: Optional[str],
    ) -> Dict[str, Any]:
        _ = description, actions, results, post_screenshot_b64
        return {"success": True, "reason": "ok", "retry_actions": []}

    def evaluate(self, goal: str, history: List[str], latest_screenshot_b64: Optional[str] = None) -> Dict[str, Any]:
        _ = goal, history, latest_screenshot_b64
        return {"complete": any("finish" in h for h in history)}

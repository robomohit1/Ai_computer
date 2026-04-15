from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DangerLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ActionType(str, Enum):
    finish = "finish"
    run_command = "run_command"
    read_file = "read_file"
    write_file = "write_file"
    move_file = "move_file"
    mouse_click = "mouse_click"
    keyboard_type = "keyboard_type"
    screenshot = "screenshot"
    ocr_image = "ocr_image"
    api_call = "api_call"
    scroll = "scroll"
    double_click = "double_click"
    right_click = "right_click"
    middle_click = "middle_click"
    mouse_move = "mouse_move"
    left_click_drag = "left_click_drag"
    key_combo = "key_combo"
    hold_key = "hold_key"
    wait_action = "wait_action"
    cursor_position = "cursor_position"
    text_view = "text_view"
    text_create = "text_create"
    text_str_replace = "text_str_replace"
    text_insert = "text_insert"
    text_undo_edit = "text_undo_edit"
    browser_open = "browser_open"
    browser_screenshot = "browser_screenshot"
    browser_click = "browser_click"
    browser_click_coords = "browser_click_coords"
    browser_type = "browser_type"
    browser_scroll = "browser_scroll"
    browser_get_text = "browser_get_text"
    browser_accessibility_tree = "browser_accessibility_tree"
    browser_navigate_back = "browser_navigate_back"
    browser_close = "browser_close"


class Action(BaseModel):
    id: str
    type: ActionType
    args: Dict[str, Any] = Field(default_factory=dict)
    explanation: str = ""
    requires_approval: bool = False


class ToolResult(BaseModel):
    ok: bool
    output: str


class ToolError(Exception):
    pass


class MemoryItem(BaseModel):
    id: int
    kind: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str


class AgentContext(BaseModel):
    goal: str
    history: List[str] = Field(default_factory=list)
    screen_width: int = 1280
    screen_height: int = 800


class TaskRecord(BaseModel):
    id: str
    status: str = "pending"
    context: AgentContext


class SubTask(BaseModel):
    id: str
    description: str
    actions: List[Action] = Field(default_factory=list)
    status: str = "pending"
    error: Optional[str] = None
    post_screenshot_b64: Optional[str] = None


class HierarchicalPlan(BaseModel):
    reasoning: str
    sub_tasks: List[SubTask]
    overall_complete: bool = False


class ActionDecision(BaseModel):
    danger: DangerLevel
    reason: str
    requires_approval: bool


class ApprovalBundle(BaseModel):
    action_id: str
    action_type: str
    action_args: Dict[str, Any]
    danger: DangerLevel
    reason: str
    explanation: str
    context_screenshot_b64: Optional[str] = None
    timeout_seconds: int = 60
    task_id: str
    created_at: str


class PluginAction(BaseModel):
    name: str
    description: str
    handlers: Dict[str, Any]

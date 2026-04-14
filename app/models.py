from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    queued = "queued"
    planning = "planning"
    evaluating = "evaluating"
    waiting_approval = "waiting_approval"
    running = "running"
    completed = "completed"
    failed = "failed"


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


class DangerLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    blocked = "blocked"


class Action(BaseModel):
    id: str
    type: ActionType
    args: Dict[str, Any] = Field(default_factory=dict)
    explanation: str = ""
    requires_approval: bool = True


class ToolResult(BaseModel):
    ok: bool
    action_type: str
    summary: str
    output: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ActionDecision(BaseModel):
    action_id: str
    allowed: bool
    danger: DangerLevel
    reason: str


class PlanIteration(BaseModel):
    iteration: int
    reasoning: str = ""
    actions: List[Action] = Field(default_factory=list)


class AgentContext(BaseModel):
    goal: str
    history: List[str] = Field(default_factory=list)
    last_error: str = ""


class TaskRequest(BaseModel):
    prompt: str
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    auto_run: bool = True
    max_iterations: int = 5


class TaskRecord(BaseModel):
    id: str
    prompt: str
    provider: str
    model: str
    status: TaskStatus
    actions: List[Action] = Field(default_factory=list)
    action_decisions: List[ActionDecision] = Field(default_factory=list)
    iterations: List[PlanIteration] = Field(default_factory=list)
    context: AgentContext
    current_action: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class ApprovalRequest(BaseModel):
    action_id: str
    approve: bool


class ProviderConfig(BaseModel):
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com/v1"


class SafetyConfig(BaseModel):
    safe_mode: bool = True
    command_denylist: List[str] = Field(
        default_factory=lambda: [
            "rm -rf /",
            "mkfs",
            "shutdown",
            "reboot",
            "userdel",
            "chmod -r 777 /",
            "dd if=",
            "poweroff",
        ]
    )
    command_allowlist: List[str] = Field(default_factory=lambda: ["python", "pytest", "git", "ls", "cat", "echo", "pwd", "find"])
    max_actions_per_task: int = 40
    command_timeout_seconds: int = 120


class ProviderPlanResponse(BaseModel):
    reasoning: str = ""
    actions: List[Action]


class EvaluateResponse(BaseModel):
    done: bool
    summary: str
    next_prompt: str = ""


class MemoryQuery(BaseModel):
    prompt: str
    limit: int = 5


class MemoryItem(BaseModel):
    id: int
    kind: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ScheduledTaskRequest(BaseModel):
    prompt: str
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    interval_seconds: int = 300


class ScheduledTask(BaseModel):
    id: str
    prompt: str
    provider: str
    model: str
    interval_seconds: int
    enabled: bool = True


class PluginInfo(BaseModel):
    name: str
    description: str
    action_types: List[str]

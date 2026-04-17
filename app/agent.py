from __future__ import annotations
import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .log_emitter import LogEmitter
from .memory import MemoryStore
from .models import (
    Action,
    ActionDecision,
    ActionType,
    AgentContext,
    ApprovalBundle,
    TaskRecord,
    ToolError,
    ToolResult,
)
from .providers import PlannerProvider, _capture_screenshot_b64
from .safety import SafetyManager
from .text_editor import TextEditorTool
from .tools import ToolExecutor
from .plugins import PluginRegistry

_SCREENSHOT_ACTIONS = {
    ActionType.mouse_click,
    ActionType.keyboard_type,
    ActionType.scroll,
    ActionType.double_click,
    ActionType.right_click,
    ActionType.middle_click,
    ActionType.mouse_move,
    ActionType.left_click_drag,
    ActionType.key_combo,
}


class AgentService:
    def __init__(self, workspace: Path, log_emitter: LogEmitter):
        self.workspace = workspace
        self.log_emitter = log_emitter
        self.memory = MemoryStore(workspace)
        self.safety = SafetyManager()
        self.plugin_registry = PluginRegistry()
        self.tools = ToolExecutor(workspace, plugin_registry=self.plugin_registry)
        self.provider = PlannerProvider()
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._approvals: Dict[str, asyncio.Future] = {}

    def _emit(self, task_id: str, event: str, data: Dict[str, Any]):
        self.log_emitter.emit(task_id, event, data)

    def init_task(
        self,
        task_id: str,
        goal: str,
        screen_width: int = 1280,
        screen_height: int = 800,
        model: str = "claude-3-5-sonnet-20241022",
    ) -> TaskRecord:
        self.provider.model = model
        context = AgentContext(
            goal=goal,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        record = TaskRecord(id=task_id, status="running", context=context)
        self._active_tasks[task_id] = asyncio.create_task(
            self.run_task(task_id, goal, screen_width, screen_height)
        )
        return record

    async def run_task(
        self,
        task_id: str,
        goal: str,
        screen_width: int = 1280,
        screen_height: int = 800,
    ):
        self._emit(task_id, "status", {"message": "Initializing planning..."})
        try:
            # Search memory for relevant context before planning
            context_memories = self.memory.search(goal, limit=5)
            memory_context: Optional[str] = None
            if context_memories:
                memory_context = "\n".join(f"- {m.content}" for m in context_memories)

            # Hierarchical Planning
            plan = self.provider.plan_hierarchical(goal, memory_context=memory_context)
            self._emit(task_id, "plan", plan.model_dump())

            history: List[str] = []

            for sub_task in plan.sub_tasks:
                self._emit(task_id, "status", {"message": f"Executing sub-task: {sub_task.description}"})

                results: List[str] = []
                actions_taken: List[Dict[str, Any]] = []

                for action_data in sub_task.actions:
                    action = Action(**action_data.model_dump())

                    if action.requires_approval:
                        self._emit(
                            task_id,
                            "approval_required",
                            {"action_id": action.id, "action": action.model_dump()},
                        )
                        approved = await self._wait_for_approval(task_id, action.id)
                        if not approved:
                            self._emit(task_id, "status", {"message": f"Action {action.id} rejected. Stopping."})
                            return

                    res = await self.tools.run_action(action, sw=screen_width, sh=screen_height)
                    results.append(res.output)
                    actions_taken.append(action.model_dump())

                    # Store action result in memory
                    self.memory.add_action_result(task_id, action.id, res.output)

                    log_entry = f"Action: {action.type.value} -> {res.output}"
                    history.append(log_entry)
                    self._emit(task_id, "action_result", {"action_id": action.id, "ok": res.ok, "output": res.output})

                    if action.type in _SCREENSHOT_ACTIONS or action.type == ActionType.screenshot:
                        screenshot = res.base64_image or _capture_screenshot_b64(screen_width, screen_height)
                        self._emit(task_id, "screenshot", {"data": screenshot})

                # Reflection
                self._emit(task_id, "status", {"message": "Reflecting on progress..."})
                reflection = self.provider.reflect_on_subtask(
                    sub_task.description,
                    actions_taken,
                    results,
                    _capture_screenshot_b64(screen_width, screen_height),
                )
                self._emit(task_id, "reflection", reflection)

                if not reflection.get("success", True):
                    retry_actions = reflection.get("retry_actions", [])
                    for retry_data in retry_actions:
                        if "id" not in retry_data or not retry_data["id"]:
                            retry_data["id"] = str(uuid.uuid4())
                        try:
                            retry_action = Action(**retry_data)
                            retry_res = await self.tools.run_action(retry_action, sw=screen_width, sh=screen_height)
                            self.memory.add_action_result(task_id, retry_action.id, retry_res.output)
                            history.append(f"Retry: {retry_action.type.value} -> {retry_res.output}")
                            self._emit(
                                task_id,
                                "action_result",
                                {"action_id": retry_action.id, "ok": retry_res.ok, "output": retry_res.output},
                            )
                        except Exception as e:
                            history.append(f"Retry failed: {str(e)}")
                    self._emit(
                        task_id,
                        "status",
                        {"message": f"Sub-task failed: {reflection.get('reason')}"},
                    )

            # Final Evaluation
            eval_res = self.provider.evaluate(goal, history, _capture_screenshot_b64(screen_width, screen_height))
            self._emit(task_id, "done", eval_res)

        except Exception as e:
            self._emit(task_id, "error", {"message": str(e)})
            raise

    async def _wait_for_approval(self, task_id: str, action_id: str) -> bool:
        fut_id = f"{task_id}:{action_id}"
        self._approvals[fut_id] = asyncio.Future()
        try:
            return await self._approvals[fut_id]
        finally:
            self._approvals.pop(fut_id, None)

    def submit_approval(self, task_id: str, action_id: str, approved: bool):
        fut_id = f"{task_id}:{action_id}"
        if fut_id in self._approvals:
            self._approvals[fut_id].set_result(approved)

    def cancel_task(self, task_id: str) -> bool:
        task = self._active_tasks.get(task_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

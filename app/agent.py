from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .memory import MemoryStore
from .models import Action, ActionDecision, ActionType, AgentContext, ApprovalBundle, TaskRecord, ToolError
from .providers import PlannerProvider, _capture_screenshot_b64
from .safety import SafetyManager
from .text_editor import TextEditorTool
from .tools import ToolExecutor
from .plugins import PluginRegistry


class AgentService:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.planner = PlannerProvider()
        self.safety = SafetyManager()
        self.text_editor = TextEditorTool(workspace)
        self.plugins = PluginRegistry()
        self.plugins.load_defaults()
        self.tools = ToolExecutor(workspace, text_editor=self.text_editor, plugin_registry=self.plugins)
        self.memory = MemoryStore(workspace / "memory")
        self.pending_approvals: Dict[str, bool] = {}
        self.pending_events: Dict[str, asyncio.Event] = {}
        self.pending_approval_bundles: Dict[str, ApprovalBundle] = {}

    async def create_task(self, task_id: str, goal: str, screen_width: int = 1280, screen_height: int = 800) -> TaskRecord:
        record = TaskRecord(id=task_id, status="running", context=AgentContext(goal=goal, screen_width=screen_width, screen_height=screen_height))
        await self._run_task(record)
        return record

    async def _wait_for_approval(
        self, action: Action, decision: ActionDecision, task_id: str, timeout_seconds: int = 60
    ) -> bool:
        screenshot = _capture_screenshot_b64(1280, 800)
        bundle = ApprovalBundle(
            action_id=action.id,
            action_type=action.type.value,
            action_args=action.args,
            danger=decision.danger,
            reason=decision.reason,
            explanation=action.explanation,
            context_screenshot_b64=screenshot,
            timeout_seconds=max(1, int(timeout_seconds)),
            task_id=task_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.pending_approval_bundles[action.id] = bundle
        self.pending_events[action.id] = asyncio.Event()
        try:
            await asyncio.wait_for(self.pending_events[action.id].wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            self.pending_approvals[action.id] = False
            raise ToolError("Approval timed out") from exc
        return self.pending_approvals.get(action.id, False)

    def submit_approval(self, action_id: str, approve: bool):
        self.pending_approvals[action_id] = approve
        if action_id in self.pending_events:
            self.pending_events[action_id].set()

    async def _run_task(self, record: TaskRecord):
        latest = None
        plan = self.planner.plan_hierarchical(record.context.goal, latest)
        if plan.overall_complete:
            record.status = "completed"
            return
        for sub_task in plan.sub_tasks:
            sub_task.status = "running"
            retries = 0
            while retries <= 2:
                action_results = []
                for action in sub_task.actions:
                    decision = self.safety.evaluate(action)
                    if decision.requires_approval:
                        approved = await self._wait_for_approval(action, decision, record.id)
                        if not approved:
                            raise ToolError("Action denied")
                    result = self.tools.run_action(action, record.context.screen_width, record.context.screen_height)
                    action_results.append(result.output)
                    record.context.history.append(json.dumps({"type": "action", "action": action.model_dump(), "result": result.output}))
                    if action.type in {
                        ActionType.mouse_click,
                        ActionType.keyboard_type,
                        ActionType.scroll,
                        ActionType.double_click,
                        ActionType.right_click,
                        ActionType.key_combo,
                        ActionType.browser_open,
                        ActionType.browser_click,
                        ActionType.browser_click_coords,
                        ActionType.browser_type,
                        ActionType.browser_scroll,
                    }:
                        latest = _capture_screenshot_b64(record.context.screen_width, record.context.screen_height)
                        record.context.history.append(
                            json.dumps(
                                {
                                    "type": "post_action_screenshot",
                                    "action_id": action.id,
                                    "action_type": action.type.value,
                                    "screenshot_b64": latest,
                                }
                            )
                        )
                reflection = self.planner.reflect_on_subtask(
                    sub_task.description,
                    [a.model_dump() for a in sub_task.actions],
                    action_results,
                    latest,
                )
                if reflection.get("success", False):
                    sub_task.status = "completed"
                    sub_task.post_screenshot_b64 = latest
                    break
                retries += 1
                if retries > 2:
                    sub_task.status = "failed"
                    sub_task.error = reflection.get("reason", "failed")
                    break
                sub_task.actions = [Action.model_validate(a) for a in reflection.get("retry_actions", [])]
        record.status = "completed" if all(s.status == "completed" for s in plan.sub_tasks) else "failed"

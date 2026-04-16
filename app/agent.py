from __future__ import annotations
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
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
    ActionType.hold_key,
    ActionType.browser_open,
    ActionType.browser_click,
    ActionType.browser_click_coords,
    ActionType.browser_type,
    ActionType.browser_scroll,
    ActionType.browser_navigate_back,
}

def _safe_screenshot(width: int, height: int) -> Optional[str]:
    if sys.platform not in ("win32", "darwin") and not __import__("os").environ.get("DISPLAY"):
        return None
    try:
        return _capture_screenshot_b64(width, height)
    except Exception:
        return None

class AgentService:
    def __init__(self, workspace: Path, log_emitter: Optional[LogEmitter] = None):
        self.workspace = workspace
        workspace.mkdir(parents=True, exist_ok=True)
        self.planner = PlannerProvider()
        self.safety = SafetyManager()
        self.text_editor = TextEditorTool(workspace)
        self.plugins = PluginRegistry()
        self.plugins.load_defaults()
        self.tools = ToolExecutor(
            workspace, text_editor=self.text_editor, plugin_registry=self.plugins
        )
        self.memory = MemoryStore(workspace / "memory")
        self.pending_approvals: Dict[str, bool] = {}
        self.pending_events: Dict[str, asyncio.Event] = {}
        self.pending_approval_bundles: Dict[str, ApprovalBundle] = {}
        self._log = log_emitter

    def _emit(self, task_id: str, event_type: str, **kwargs):
        if self._log:
            self._log.emit(task_id, event_type, kwargs)

    def init_task(
        self,
        task_id: str,
        goal: str,
        screen_width: int = 1280,
        screen_height: int = 800,
    ) -> TaskRecord:
        return TaskRecord(
            id=task_id,
            status="running",
            context=AgentContext(
                goal=goal,
                screen_width=screen_width,
                screen_height=screen_height,
            ),
        )

    async def run_task(self, record: TaskRecord):
        try:
            await self._run_task(record)
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            self._emit(record.id, "error", message=str(exc))
        finally:
            self._emit(record.id, "done", status=record.status)

    async def _wait_for_approval(
        self, action: Action, decision: ActionDecision, task_id: str, timeout_seconds: int = 60
    ) -> bool:
        screenshot = _safe_screenshot(1280, 800)
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
        self._emit(
            task_id,
            "approval_required",
            action_id=action.id,
            action_type=action.type.value,
            danger=decision.danger.value,
            explanation=action.explanation,
            timeout_seconds=timeout_seconds,
            args=action.args
        )
        try:
            await asyncio.wait_for(
                self.pending_events[action.id].wait(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            self.pending_approvals[action.id] = False
            self.pending_approval_bundles.pop(action.id, None)
            raise ToolError("Approval timed out") from exc
        self.pending_approval_bundles.pop(action.id, None)
        return self.pending_approvals.get(action.id, False)

    def submit_approval(self, action_id: str, approve: bool):
        self.pending_approvals[action_id] = approve
        if action_id in self.pending_events:
            self.pending_events[action_id].set()

    async def _run_task(self, record: TaskRecord):
        task_id = record.id
        sw = getattr(record.context, "screen_width", 1280)
        sh = getattr(record.context, "screen_height", 800)
        latest: Optional[str] = None

        self._emit(task_id, "planning", goal=record.context.goal)
        plan = await asyncio.get_event_loop().run_in_executor(
            None, self.planner.plan_hierarchical, record.context.goal, latest
        )
        
        self._emit(task_id, "plan_ready", reasoning=plan.reasoning)

        if plan.overall_complete:
            record.status = "completed"
            return

        for sub_task in plan.sub_tasks:
            sub_task.status = "running"
            self._emit(task_id, "subtask_start", subtask_id=sub_task.id, description=sub_task.description)
            
            action_results = []
            for action in sub_task.actions:
                decision = self.safety.evaluate(action)
                if decision.requires_approval or action.requires_approval:
                    approved = await self._wait_for_approval(action, decision, task_id)
                    if not approved:
                        raise ToolError(f"Action {action.id} denied by user")

                self._emit(task_id, "action_start", action_id=action.id, action_type=action.type.value, args=action.args)
                
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self.tools.run_action, action, sw, sh
                    )
                except Exception as e:
                    result_output = f"ERROR: {str(e)}"
                    self._emit(task_id, "action_error", action_id=action.id, error=result_output)
                    action_results.append(result_output)
                    continue

                action_results.append(result.output)
                self._emit(task_id, "action_done", action_id=action.id, ok=result.ok, output=result.output)

                if action.type in _SCREENSHOT_ACTIONS:
                    latest = _safe_screenshot(sw, sh)
                    if latest:
                        self._emit(task_id, "screenshot", action_id=action.id, screenshot_b64=latest)

                if action.type == ActionType.finish:
                    record.status = "completed"
                    return

            reflection = await asyncio.get_event_loop().run_in_executor(
                None, self.planner.reflect_on_subtask, sub_task.description, [a.model_dump() for a in sub_task.actions], action_results, latest
            )
            
            if reflection.get("success"):
                sub_task.status = "completed"
            else:
                sub_task.status = "failed"
                raise ToolError(f"Subtask failed: {reflection.get('reason')}")

        record.status = "completed"

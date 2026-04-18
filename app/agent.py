from __future__ import annotations
import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
        self.plugin_registry.load_defaults()
        self.tools = ToolExecutor(workspace, plugin_registry=self.plugin_registry)
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._paused_tasks: set[str] = set()
        self._approvals: Dict[str, asyncio.Future] = {}
        self._pause_events: Dict[str, asyncio.Event] = {}
        self._on_task_complete: Optional[Callable[[str, str, str], None]] = None
        self.plugin_registry.load_defaults()

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
        context = AgentContext(
            goal=goal,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        record = TaskRecord(id=task_id, status="running", context=context, goal=goal)
        self._active_tasks[task_id] = asyncio.create_task(
            self.run_task(task_id, goal, screen_width, screen_height, model)
        )
        return record

    async def run_task(
        self,
        task_id: str,
        goal: str,
        screen_width: int = 1280,
        screen_height: int = 800,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        provider = PlannerProvider(model=model)
        self._emit(task_id, "status", {"message": "Initializing planning..."})
        try:
            context_memories = self.memory.search(goal, limit=5)
            memory_context: Optional[str] = None
            if context_memories:
                memory_context = "\n".join(f"- {m.content}" for m in context_memories)

            plan = provider.plan_hierarchical(goal, memory_context=memory_context, latest_screenshot_b64=_capture_screenshot_b64(screen_width, screen_height))
            self._emit(task_id, "plan", plan.model_dump())

            history: List[str] = []
            action_count = 0
            consecutive_fails = 0

            while plan.sub_tasks:
                sub_task = plan.sub_tasks.pop(0)
                self._emit(task_id, "status", {"message": f"Executing sub-task: {sub_task.description}"})

                results: List[str] = []
                actions_taken: List[Dict[str, Any]] = []

                for action_data in sub_task.actions:
                    if action_count >= 50:
                        self._emit(task_id, "error", {"message": "Hard limit of 50 actions reached."})
                        self._emit(task_id, "done", {"complete": False, "reason": "Hard limit of 50 actions reached."})
                        return
                        
                    while task_id in self._paused_tasks:
                        await asyncio.sleep(0.5)
                        
                    action_count += 1
                    
                    action = Action(**action_data.model_dump())
                    decision = self.safety.evaluate(action)

                    self._emit(task_id, "action_start", {
                        "action_id": action.id,
                        "action_type": action.type.value,
                        "explanation": action.explanation,
                        "args_summary": _summarize_args(action.type.value, action.args),
                    })

                    if action.requires_approval or decision.requires_approval:
                        self._emit(task_id, "approval_required", {
                            "action_id": action.id,
                            "action": action.model_dump(),
                            "danger": decision.danger.value,
                            "reason": decision.reason,
                            "explanation": action.explanation,
                        })
                        approved = await self._wait_for_approval(task_id, action.id)
                        if not approved:
                            self._emit(task_id, "status", {"message": f"Action {action.id} rejected. Stopping."})
                            self._finalize(task_id, "cancelled", "user rejected action")

                            return

                    try:
                        res = await asyncio.wait_for(
                            self.tools.run_action(action, sw=screen_width, sh=screen_height),
                            timeout=30.0
                        )
                    except asyncio.TimeoutError:
                        res = ToolResult(ok=False, output="Action timed out after 30 seconds.")
                    except Exception as e:
                        res = ToolResult(ok=False, output=f"Action failed with exception: {str(e)}")

                    results.append(res.output)
                    actions_taken.append(action.model_dump())

                    self.memory.add_action_result(task_id, action.id, res.output)

                    log_entry = f"Action: {action.type.value} -> {res.output}"
                    history.append(log_entry)
                    self._emit(task_id, "action_result", {"action_id": action.id, "ok": res.ok, "output": res.output})

                    if action.type in _SCREENSHOT_ACTIONS or action.type == ActionType.screenshot:
                        screenshot = res.base64_image or _capture_screenshot_b64(screen_width, screen_height)
                        self._emit(task_id, "screenshot", {"data": screenshot})

                # Reflection
                self._emit(task_id, "status", {"message": "Reflecting on progress..."})
                reflection = provider.reflect_on_subtask(
                    sub_task.description,
                    actions_taken,
                    results,
                    _capture_screenshot_b64(screen_width, screen_height),
                )
                self._emit(task_id, "reflection", reflection)

                if not reflection.get("success", True):
                    consecutive_fails += 1
                    retry_actions = reflection.get("retry_actions", [])
                    for retry_data in retry_actions:
                        if action_count >= 50:
                            break
                        action_count += 1
                        if "id" not in retry_data or not retry_data["id"]:
                            retry_data["id"] = str(uuid.uuid4())
                        try:
                            retry_action = Action(**retry_data)
                            retry_res = await asyncio.wait_for(self.tools.run_action(retry_action, sw=screen_width, sh=screen_height), timeout=30.0)
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
                    
                    if consecutive_fails > 2:
                        self._emit(task_id, "status", {"message": "Multiple failures detected. Re-planning..."})
                        plan = provider.plan_hierarchical(goal + f" (Re-planning after failures. History: {history[-5:]})", memory_context, latest_screenshot_b64=_capture_screenshot_b64(screen_width, screen_height))
                        self._emit(task_id, "plan", plan.model_dump())
                        consecutive_fails = 0
                else:
                    consecutive_fails = 0

            # Final Evaluation
            eval_res = provider.evaluate(goal, history, _capture_screenshot_b64(screen_width, screen_height))
            status = "done" if eval_res.get("complete") else "failed"
            self._finalize(task_id, status, eval_res.get("reason", ""))

            self._emit(task_id, "done", eval_res)
            
            # Store goal outcome in memory
            self.memory.add("task_outcome", f"Goal: {goal} | Outcome: {eval_res.get('complete')} | Reason: {eval_res.get('reason')}")
            
            record = self._active_tasks.get(task_id)
            if record:
                record.cancel()

        except asyncio.CancelledError:
            self._finalize(task_id, "cancelled", "task cancelled")
            self._emit(task_id, "cancelled", {"message": "Task cancelled"})

        except Exception as e:
            self._finalize(task_id, "failed", str(e))
            self._emit(task_id, "error", {"message": str(e)})
            self._emit(task_id, "done", {"complete": False, "reason": f"Crashed: {str(e)}"})

    def _finalize(self, task_id: str, status: str, reason: str = ""):
        if self._on_task_complete:
            self._on_task_complete(task_id, status, reason)

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
        if task_id in self._active_tasks:
            self._active_tasks[task_id].cancel()
            del self._active_tasks[task_id]
            self._paused_tasks.discard(task_id)
            return True
        return False

    def pause_task(self, task_id: str):
        self._paused_tasks.add(task_id)

    def resume_task(self, task_id: str):
        self._paused_tasks.discard(task_id)


def _summarize_args(action_type: str, args: dict) -> str:
    """One-line summary of action args for the activity log."""
    if action_type == "run_command":
        return (args.get("command") or "")[:80]
    if action_type in ("read_file", "write_file", "move_file", "text_create",
                       "text_view", "text_str_replace", "text_insert"):
        return args.get("path") or args.get("src") or ""
    if action_type == "browser_open":
        return args.get("url") or ""
    if action_type in ("browser_click", "browser_type"):
        return args.get("selector") or ""
    if action_type == "api_call":
        return f"{args.get('method','GET')} {args.get('url','')}"[:80]
    if action_type in ("mouse_click", "mouse_move", "double_click", "right_click"):
        return f"({args.get('x')}, {args.get('y')})"
    if action_type in ("keyboard_type", "type_with_delay"):
        text = args.get("text") or ""
        return text[:40] + ("..." if len(text) > 40 else "")
    if action_type == "key_combo":
        return args.get("keys") or ""
    return ""

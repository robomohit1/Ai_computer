from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

from .memory import MemoryStore
from .models import (
    AgentContext,
    ApprovalRequest,
    PlanIteration,
    ProviderConfig,
    SafetyConfig,
    ScheduledTask,
    ScheduledTaskRequest,
    TaskRecord,
    TaskRequest,
    TaskStatus,
)
from .plugin_loader import PluginRegistry
from .providers import PlannerProvider, ProviderError
from .safety import SafetyManager
from .tools import ToolError, ToolExecutor

LogHook = Callable[[str, str], Awaitable[None]]


class AgentService:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.tasks: Dict[str, TaskRecord] = {}
        self.pending_approvals: Dict[str, asyncio.Event] = {}
        self.approval_decisions: Dict[str, bool] = {}
        self.config = ProviderConfig()
        self.safety_config = SafetyConfig()
        self.planner = PlannerProvider(self.config)

        self.plugins = PluginRegistry()
        self.plugins.load_from_package("app.plugins")

        self.tools = ToolExecutor(workspace, self.plugins)
        self.safety = SafetyManager(workspace, self.safety_config)
        self.memory = MemoryStore(workspace / "agent_memory.db")

        self.log_subscribers: List[LogHook] = []
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self._scheduler: Optional[asyncio.Task] = None
        self._scheduler_stop = asyncio.Event()

    async def start(self) -> None:
        if self._scheduler is None:
            self._scheduler_stop.clear()
            self._scheduler = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        if self._scheduler:
            self._scheduler_stop.set()
            await self._scheduler
            self._scheduler = None

    def subscribe_logs(self, hook: LogHook) -> None:
        self.log_subscribers.append(hook)

    def unsubscribe_logs(self, hook: LogHook) -> None:
        self.log_subscribers = [x for x in self.log_subscribers if x != hook]

    async def _broadcast(self, task_id: str, message: str) -> None:
        for hook in self.log_subscribers:
            await hook(task_id, message)

    def update_config(self, config: ProviderConfig) -> ProviderConfig:
        self.config = config
        self.planner = PlannerProvider(config)
        return self.config

    def get_config(self) -> ProviderConfig:
        return self.config

    def get_safety_config(self) -> SafetyConfig:
        return self.safety_config

    def update_safety_config(self, config: SafetyConfig) -> SafetyConfig:
        self.safety_config = config
        self.safety.update(config)
        return self.safety_config

    def get_task(self, task_id: str) -> TaskRecord:
        return self.tasks[task_id]

    def list_tasks(self) -> list[TaskRecord]:
        return list(self.tasks.values())

    async def create_task(self, request: TaskRequest) -> TaskRecord:
        task_id = str(uuid.uuid4())
        record = TaskRecord(
            id=task_id,
            prompt=request.prompt,
            provider=request.provider,
            model=request.model,
            status=TaskStatus.queued,
            context=AgentContext(goal=request.prompt),
        )
        self.tasks[task_id] = record
        asyncio.create_task(self._run_task(record, request.max_iterations if request.auto_run else 1))
        return record

    async def _run_task(self, record: TaskRecord, max_iterations: int) -> None:
        try:
            finished = False
            for iteration in range(1, max_iterations + 1):
                record.status = TaskStatus.planning
                memory_hits = self.memory.search(record.context.goal, limit=5)
                memory_context = "\n".join(f"- ({m.kind}) {m.content[:240]}" for m in memory_hits)
                runtime_context = {
                    "goal": record.context.goal,
                    "history": record.context.history[-20:],
                    "last_error": record.context.last_error,
                }
                combined_context = f"Runtime context:\n{json.dumps(runtime_context, ensure_ascii=False)}\n\nMemory:\n{memory_context}"
                plan = await self.planner.plan(record.provider, record.model, record.context.goal, combined_context)
                record.iterations.append(PlanIteration(iteration=iteration, reasoning=plan.reasoning, actions=plan.actions))
                record.actions = plan.actions
                await self._log(record, f"Iteration {iteration}: {plan.reasoning}")

                if len(plan.actions) > self.safety_config.max_actions_per_task:
                    raise ToolError("Plan exceeded max_actions_per_task policy")

                for action in plan.actions:
                    record.current_action = action.id
                    decision = self.safety.evaluate_action(action)
                    record.action_decisions.append(decision)
                    if not decision.allowed:
                        raise ToolError(f"Action blocked: {decision.reason}")

                    if action.requires_approval or decision.danger.value in {"medium", "high"}:
                        record.status = TaskStatus.waiting_approval
                        await self._log(record, f"Waiting approval for {action.id} ({decision.danger})")
                        approve = await self._wait_for_approval(action.id)
                        if not approve:
                            raise ToolError(f"Action {action.id} denied by user")

                    record.status = TaskStatus.running
                    await self._log(record, f"Running action {action.id}: {action.type}")

                    if action.type.value == "run_command" and "timeout" not in action.args:
                        action.args["timeout"] = self.safety_config.command_timeout_seconds

                    result = self.tools.run_action(action.type.value, action.args)
                    result_payload = result.model_dump_json()
                    record.context.history.append(result_payload)
                    self.memory.add_action_result(record.id, action.id, result_payload)
                    await self._log(record, result_payload)

                    if action.type.value == "finish":
                        finished = True
                        break

                if finished:
                    record.status = TaskStatus.completed
                    record.current_action = None
                    self.memory.add("task_history", f"Goal: {record.prompt}\nSummary: finished action emitted", {"task_id": record.id})
                    return

                record.status = TaskStatus.evaluating
                eval_result = await self.planner.evaluate(record.provider, record.model, record.context.goal, "\n".join(record.logs))
                await self._log(record, f"Evaluation: {eval_result.summary}")

                if eval_result.done and not eval_result.next_prompt:
                    record.context.goal = f"{record.context.goal}\nFinalize and emit finish action."
                elif eval_result.next_prompt:
                    record.context.goal = eval_result.next_prompt

            record.status = TaskStatus.failed
            record.error = "Max iterations reached before finish action"
            self.memory.add("task_failure", f"Goal: {record.prompt}\nError: {record.error}", {"task_id": record.id})
        except (ProviderError, ToolError, KeyError, ValueError) as exc:
            record.status = TaskStatus.failed
            record.error = str(exc)
            record.context.last_error = str(exc)
            record.current_action = None
            await self._log(record, f"Failure: {exc}")
            self.memory.add("task_failure", f"Goal: {record.prompt}\nError: {exc}", {"task_id": record.id})

    async def _wait_for_approval(self, action_id: str) -> bool:
        event = asyncio.Event()
        self.pending_approvals[action_id] = event
        await event.wait()
        decision = self.approval_decisions.pop(action_id, False)
        self.pending_approvals.pop(action_id, None)
        return decision

    def submit_approval(self, payload: ApprovalRequest) -> None:
        if payload.action_id not in self.pending_approvals:
            raise KeyError(f"No pending approval for action {payload.action_id}")
        self.approval_decisions[payload.action_id] = payload.approve
        self.pending_approvals[payload.action_id].set()

    async def _log(self, record: TaskRecord, message: str) -> None:
        record.logs.append(message)
        await self._broadcast(record.id, message)

    def query_memory(self, prompt: str, limit: int = 5):
        return self.memory.search(prompt, limit)

    def recent_memory(self, limit: int = 20):
        return self.memory.recent(limit)

    def list_plugins(self):
        return self.plugins.list_plugins()

    def add_schedule(self, request: ScheduledTaskRequest) -> ScheduledTask:
        schedule = ScheduledTask(
            id=str(uuid.uuid4()),
            prompt=request.prompt,
            provider=request.provider,
            model=request.model,
            interval_seconds=request.interval_seconds,
            enabled=True,
        )
        self.scheduled_tasks[schedule.id] = schedule
        return schedule

    def list_schedules(self) -> List[ScheduledTask]:
        return list(self.scheduled_tasks.values())

    async def _scheduler_loop(self) -> None:
        counters: Dict[str, int] = {}
        while not self._scheduler_stop.is_set():
            await asyncio.sleep(1)
            for schedule in list(self.scheduled_tasks.values()):
                if not schedule.enabled:
                    continue
                counters[schedule.id] = counters.get(schedule.id, 0) + 1
                if counters[schedule.id] >= schedule.interval_seconds:
                    counters[schedule.id] = 0
                    await self.create_task(
                        TaskRequest(
                            prompt=schedule.prompt,
                            provider=schedule.provider,
                            model=schedule.model,
                            auto_run=True,
                            max_iterations=3,
                        )
                    )

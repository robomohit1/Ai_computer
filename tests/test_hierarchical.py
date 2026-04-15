import pytest

from app.agent import AgentService
from app.models import Action, ActionType, HierarchicalPlan, SubTask


@pytest.mark.asyncio
async def test_hierarchical_success(monkeypatch, workspace):
    s = AgentService(workspace)
    plan = HierarchicalPlan(
        reasoning="r",
        sub_tasks=[
            SubTask(id="s1", description="d1", actions=[Action(id="a1", type=ActionType.wait_action, args={"seconds": 0})]),
            SubTask(id="s2", description="d2", actions=[Action(id="a2", type=ActionType.finish, args={})]),
        ],
    )
    monkeypatch.setattr(s.planner, "plan_hierarchical", lambda *a, **k: plan)
    monkeypatch.setattr(s.planner, "reflect_on_subtask", lambda *a, **k: {"success": True, "reason": "ok", "retry_actions": []})
    monkeypatch.setattr(s.tools, "run_action", lambda *a, **k: type("R", (), {"output": "ok"})())
    r = await s.create_task("t", "g")
    assert r.status == "completed"
    assert all(st.status == "completed" for st in plan.sub_tasks)


@pytest.mark.asyncio
async def test_hierarchical_retry(monkeypatch, workspace):
    s = AgentService(workspace)
    plan = HierarchicalPlan(
        reasoning="r",
        sub_tasks=[SubTask(id="s1", description="d1", actions=[Action(id="a1", type=ActionType.wait_action, args={"seconds": 0})])],
    )
    monkeypatch.setattr(s.planner, "plan_hierarchical", lambda *a, **k: plan)
    seq = iter([
        {"success": False, "reason": "retry", "retry_actions": [{"id": "a2", "type": "wait_action", "args": {"seconds": 0}, "explanation": "", "requires_approval": False}]},
        {"success": True, "reason": "ok", "retry_actions": []},
    ])
    monkeypatch.setattr(s.planner, "reflect_on_subtask", lambda *a, **k: next(seq))
    monkeypatch.setattr(s.tools, "run_action", lambda *a, **k: type("R", (), {"output": "ok"})())
    r = await s.create_task("t", "g")
    assert r.status == "completed"

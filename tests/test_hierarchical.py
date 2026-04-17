import pytest
from app.agent import AgentService
from app.log_emitter import log_emitter
from app.models import Action, ActionType, HierarchicalPlan, SubTask

@pytest.mark.asyncio
async def test_hierarchical_success(monkeypatch, workspace):
    s = AgentService(workspace, log_emitter=log_emitter)
    
    plan = HierarchicalPlan(
        reasoning="r",
        sub_tasks=[SubTask(id="s1", description="d1", actions=[Action(id="a1", type=ActionType.wait_action, args={"seconds": 0})])],
        overall_complete=False
    )
    
    monkeypatch.setattr("app.providers.PlannerProvider.plan_hierarchical", lambda *a, **k: plan)
    monkeypatch.setattr("app.providers.PlannerProvider.reflect_on_subtask", lambda *a, **k: {"success": True})
    monkeypatch.setattr("app.providers.PlannerProvider.evaluate", lambda *a, **k: {"complete": True, "reason": "done"})
    
    await s.run_task("t1", "goal")
    out = s.memory.search("task_outcome")
    assert any("Outcome: True" in m.content for m in out)

@pytest.mark.asyncio
async def test_hierarchical_retry(monkeypatch, workspace):
    s = AgentService(workspace, log_emitter=log_emitter)
    plan = HierarchicalPlan(
        reasoning="r",
        sub_tasks=[SubTask(id="s2", description="d2", actions=[Action(id="a2", type=ActionType.wait_action, args={"seconds": 0})])],
        overall_complete=False
    )
    monkeypatch.setattr("app.providers.PlannerProvider.plan_hierarchical", lambda *a, **k: plan)
    
    reflections = [{"success": False, "reason": "fail", "retry_actions": [{"type": "wait_action", "args": {"seconds": 0}}]}, {"success": True}]
    def mock_reflect(*a, **k):
        return reflections.pop(0)
    
    monkeypatch.setattr("app.providers.PlannerProvider.reflect_on_subtask", mock_reflect)
    monkeypatch.setattr("app.providers.PlannerProvider.evaluate", lambda *a, **k: {"complete": True, "reason": "done"})
    
    await s.run_task("t2", "goal")
    out = s.memory.search("task_outcome")
    assert any("Outcome: True" in m.content for m in out)

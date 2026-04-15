import base64
import json

import pytest

from app.agent import AgentService
from app.models import Action, ActionType, HierarchicalPlan, SubTask


@pytest.mark.asyncio
async def test_post_action_screenshot_added(monkeypatch, workspace):
    service = AgentService(workspace)
    plan = HierarchicalPlan(
        reasoning="r",
        sub_tasks=[
            SubTask(
                id="s1",
                description="d",
                actions=[Action(id="a1", type=ActionType.mouse_click, args={"x": 1, "y": 2})],
            )
        ],
    )
    monkeypatch.setattr(service.planner, "plan_hierarchical", lambda goal, latest_screenshot_b64=None: plan)
    monkeypatch.setattr(service.planner, "reflect_on_subtask", lambda *a, **k: {"success": True, "reason": "ok", "retry_actions": []})
    monkeypatch.setattr(service.tools, "run_action", lambda *a, **k: type("R", (), {"output": "ok"})())
    png = base64.b64encode(b"\x89PNGtest").decode()
    monkeypatch.setattr("app.agent._capture_screenshot_b64", lambda w, h: png)

    rec = await service.create_task("t1", "goal")
    entries = [json.loads(x) for x in rec.context.history]
    found = [e for e in entries if e.get("type") == "post_action_screenshot"]
    assert found
    assert base64.b64decode(found[0]["screenshot_b64"]).startswith(b"\x89PNG")

import asyncio
import base64

import pytest

from app.agent import AgentService
from app.models import Action, ActionDecision, ActionType, DangerLevel, ToolError


@pytest.mark.asyncio
async def test_approval_flow(monkeypatch, workspace):
    s = AgentService(workspace)
    png = base64.b64encode(b"\x89PNGx").decode()
    monkeypatch.setattr("app.agent._capture_screenshot_b64", lambda w, h: png)
    action = Action(id="a1", type=ActionType.mouse_click, args={"x": 1, "y": 2}, explanation="do")
    decision = ActionDecision(danger=DangerLevel.medium, reason="r", requires_approval=True)

    async def approve_later():
        await asyncio.sleep(0.01)
        s.submit_approval("a1", True)

    t = asyncio.create_task(approve_later())
    out = await s._wait_for_approval(action, decision, "task", timeout_seconds=1)
    await t
    assert out is True
    assert s.pending_approval_bundles["a1"].action_type == "mouse_click"


@pytest.mark.asyncio
async def test_approval_denial(monkeypatch, workspace):
    s = AgentService(workspace)
    monkeypatch.setattr("app.agent._capture_screenshot_b64", lambda w, h: "x")
    action = Action(id="a2", type=ActionType.mouse_click, args={"x": 1, "y": 2})
    decision = ActionDecision(danger=DangerLevel.medium, reason="r", requires_approval=True)

    async def deny_later():
        await asyncio.sleep(0.01)
        s.submit_approval("a2", False)

    t = asyncio.create_task(deny_later())
    out = await s._wait_for_approval(action, decision, "task", timeout_seconds=1)
    await t
    assert out is False


@pytest.mark.asyncio
async def test_approval_timeout(monkeypatch, workspace):
    s = AgentService(workspace)
    monkeypatch.setattr("app.agent._capture_screenshot_b64", lambda w, h: "x")
    action = Action(id="a3", type=ActionType.mouse_click, args={"x": 1, "y": 2})
    decision = ActionDecision(danger=DangerLevel.medium, reason="r", requires_approval=True)
    with pytest.raises(ToolError):
        await s._wait_for_approval(action, decision, "task", timeout_seconds=0.01)

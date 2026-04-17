import asyncio
import base64
import pytest
from app.agent import AgentService
from app.models import Action, ActionDecision, ActionType, DangerLevel, ToolError
from app.log_emitter import log_emitter

@pytest.mark.asyncio
async def test_approval_flow(monkeypatch, workspace):
    s = AgentService(workspace, log_emitter=log_emitter)
    png = base64.b64encode(b"\x89PNGx").decode()
    monkeypatch.setattr("app.agent._capture_screenshot_b64", lambda w, h: png)
    action = Action(id="a1", type=ActionType.mouse_click, args={"x": 1, "y": 2}, explanation="do")
    decision = ActionDecision(danger=DangerLevel.medium, reason="r", requires_approval=True)

    async def approve_later():
        await asyncio.sleep(0.01)
        s.submit_approval("task", "a1", True)

    t = asyncio.create_task(approve_later())
    out = await s._wait_for_approval("task", "a1")
    await t
    assert out is True

@pytest.mark.asyncio
async def test_approval_denial(monkeypatch, workspace):
    s = AgentService(workspace, log_emitter=log_emitter)
    monkeypatch.setattr("app.agent._capture_screenshot_b64", lambda w, h: "x")
    action = Action(id="a2", type=ActionType.mouse_click, args={"x": 1, "y": 2})
    decision = ActionDecision(danger=DangerLevel.medium, reason="r", requires_approval=True)

    async def deny_later():
        await asyncio.sleep(0.01)
        s.submit_approval("task", "a2", False)

    t = asyncio.create_task(deny_later())
    out = await s._wait_for_approval("task", "a2")
    await t
    assert out is False

@pytest.mark.asyncio
async def test_approval_timeout(monkeypatch, workspace):
    s = AgentService(workspace, log_emitter=log_emitter)
    monkeypatch.setattr("app.agent._capture_screenshot_b64", lambda w, h: "x")
    action = Action(id="a3", type=ActionType.mouse_click, args={"x": 1, "y": 2})
    decision = ActionDecision(danger=DangerLevel.medium, reason="r", requires_approval=True)
    # the function no longer takes timeout_seconds. It just waits forever.
    # We will cancel it to simulate timeout.
    t = asyncio.create_task(s._wait_for_approval("task", "a3"))
    await asyncio.sleep(0.05)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

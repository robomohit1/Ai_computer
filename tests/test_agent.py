from __future__ import annotations
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# Test 1: MemoryStore.add and search with fallback collection (no ChromaDB needed)
def test_memory_store_add_and_search(tmp_path):
    from app.memory import MemoryStore, _FallbackCollection
    store = MemoryStore(tmp_path / "db")
    # Force fallback collection regardless of ChromaDB availability
    store.collection = _FallbackCollection()
    store._counter = 0

    store.add("action_result", "opened browser and clicked the submit button", {"task_id": "t1", "action_id": "a1"})
    store.add("action_result", "typed text into the search box", {"task_id": "t1", "action_id": "a2"})

    results = store.search("browser click", limit=5)
    assert len(results) > 0
    assert any("browser" in r.content for r in results)


# Test 2: SafetyManager classifies run_command as high-risk
def test_safety_run_command_is_high_risk():
    from app.safety import SafetyManager
    from app.models import Action, ActionType, DangerLevel

    sm = SafetyManager()
    action = Action(id="1", type=ActionType.run_command, args={"command": "rm -rf /"})
    decision = sm.evaluate(action)
    assert decision.danger == DangerLevel.high
    assert decision.requires_approval is True


# Test 3: ToolExecutor.run_action with text_create and text_view
@pytest.mark.asyncio
async def test_tool_executor_text_create_and_view(tmp_path):
    from app.tools import ToolExecutor
    from app.models import Action, ActionType

    with patch("pyautogui.PAUSE", 0), patch("pyautogui.FAILSAFE", False):
        executor = ToolExecutor(tmp_path)

    create_action = Action(
        id="c1", type=ActionType.text_create,
        args={"path": "hello.txt", "file_text": "hello world\nline two"}
    )
    result = await executor.run_action(create_action)
    assert result.ok, result.output

    view_action = Action(id="v1", type=ActionType.text_view, args={"path": "hello.txt"})
    result = await executor.run_action(view_action)
    assert result.ok
    assert "hello world" in result.output
    assert "line two" in result.output


# Test 4: TextEditorTool.str_replace raises ToolError when old_str not found
def test_text_editor_str_replace_not_found(tmp_path):
    from app.text_editor import TextEditorTool
    from app.models import ToolError

    editor = TextEditorTool(tmp_path)
    p = tmp_path / "file.txt"
    p.write_text("hello world")

    with pytest.raises(ToolError, match="not found"):
        editor.str_replace("file.txt", "not present anywhere", "replacement")


# Test 5: PlannerProvider._extract_json handles markdown fences correctly
def test_extract_json_handles_fences():
    from app.providers import _extract_json

    assert _extract_json('```json\n{"key": "value"}\n```') == {"key": "value"}
    assert _extract_json('```\n{"key": 42}\n```') == {"key": 42}
    assert _extract_json('{"key": "plain"}') == {"key": "plain"}
    assert _extract_json('  {"nested": {"a": 1}}  ') == {"nested": {"a": 1}}


# Test 6: Mock httpx.Client.post to test plan_hierarchical returns HierarchicalPlan
def test_plan_hierarchical_returns_valid_plan():
    from app.providers import PlannerProvider
    from app.models import HierarchicalPlan

    fake_plan = {
        "reasoning": "Take screenshot first then evaluate",
        "overall_complete": False,
        "sub_tasks": [
            {
                "id": "st1",
                "description": "Capture current screen state",
                "actions": [
                    {
                        "id": "a1",
                        "type": "screenshot",
                        "args": {},
                        "explanation": "capture screen",
                        "requires_approval": False,
                    }
                ],
            }
        ],
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"content": [{"text": json.dumps(fake_plan)}]}
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_resp

    with patch("httpx.Client", return_value=mock_client):
        provider = PlannerProvider(model="claude-3-5-sonnet-20241022")
        provider._anthropic_key = "test-key"
        plan = provider.plan_hierarchical("Take a screenshot of the desktop")

    assert isinstance(plan, HierarchicalPlan)
    assert len(plan.sub_tasks) == 1
    assert plan.sub_tasks[0].id == "st1"
    assert len(plan.sub_tasks[0].actions) == 1


# Test 7: key_combo handler calls pyautogui.hotkey with split parts
@pytest.mark.asyncio
async def test_key_combo_calls_hotkey(tmp_path):
    from app.tools import ToolExecutor
    from app.models import Action, ActionType

    with patch("pyautogui.PAUSE", 0), patch("pyautogui.FAILSAFE", False):
        executor = ToolExecutor(tmp_path)

    with patch("pyautogui.hotkey") as mock_hotkey:
        action = Action(id="k1", type=ActionType.key_combo, args={"keys": "ctrl+c"})
        result = await executor.run_action(action)
        assert result.ok
        mock_hotkey.assert_called_once_with("ctrl", "c")

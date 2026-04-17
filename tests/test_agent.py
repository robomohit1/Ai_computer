import pytest
import asyncio
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app, API_KEY
from app.memory import MemoryStore, _FallbackCollection
from app.safety import SafetyManager
from app.models import Action, ActionType, DangerLevel
from app.tools import ToolExecutor
from app.agent import AgentService

# Create workspace dir for tests
workspace = Path("test_workspace").resolve()
workspace.mkdir(exist_ok=True)

# Test 1: MemoryStore fallback
def test_memorystore_fallback():
    # Force fallback by passing an invalid path that chromadb can't write to, 
    # or just use the mock if needed. Actually we can mock chromadb import.
    with patch.dict("sys.modules", {"chromadb": None}):
        store = MemoryStore(workspace / "test_db")
        assert isinstance(store.collection, _FallbackCollection)
        
        store.add("test_kind", "This is a test of fallback search")
        store.add("test_kind", "Another unrelated document")
        
        results = store.search("fallback search")
        assert len(results) > 0
        assert "fallback" in results[0].content.lower()

# Test 2: SafetyManager classifies run_command as high-risk
def test_safetymanager_high_risk():
    safety = SafetyManager()
    action = Action(id="1", type=ActionType.run_command, args={"command": "echo test"})
    decision = safety.evaluate(action)
    assert decision.danger == DangerLevel.high
    assert decision.requires_approval is True

# Test 3: SafetyManager hard-blocks dangerous shell commands
def test_safetymanager_hard_block():
    safety = SafetyManager()
    action = Action(id="2", type=ActionType.run_command, args={"command": "rm -rf /"})
    decision = safety.evaluate(action)
    assert decision.danger == DangerLevel.high
    assert "Hard-blocked" in decision.reason
    assert decision.requires_approval is True

# Test 4: ToolExecutor text_create and text_view via tmp_path workspace
def test_toolexecutor_text_tools():
    import uuid
    filename = f"test_file_{uuid.uuid4().hex}.txt"
    tools = ToolExecutor(workspace)
    
    # Create
    res = tools.text_editor.create(filename, "Line 1\nLine 2")
    assert res.ok
    assert (workspace / filename).exists()
    
    # View
    res = tools.text_editor.view(filename)
    assert res.ok
    assert "Line 1" in res.output

# Test 5: TextEditorTool.str_replace raises ToolError when old_str not found
def test_toolexecutor_str_replace_error():
    import uuid
    filename = f"test_file2_{uuid.uuid4().hex}.txt"
    tools = ToolExecutor(workspace)
    tools.text_editor.create(filename, "Hello world")
    
    with pytest.raises(Exception):
        tools.text_editor.str_replace(filename, "not_found", "replacement")

# Test 6: plan_hierarchical returns valid HierarchicalPlan
@patch("app.providers.PlannerProvider._chat_anthropic")
def test_plan_hierarchical(mock_chat):
    mock_chat.return_value = '{"reasoning": "test", "sub_tasks": [{"id": "1", "description": "st1", "actions": []}], "overall_complete": false}'
    
    from app.providers import PlannerProvider
    provider = PlannerProvider()
    provider._anthropic_key = "test"
    plan = provider.plan_hierarchical("test goal")
    assert plan.reasoning == "test"
    assert len(plan.sub_tasks) == 1

# Test 7: key_combo calls pyautogui.hotkey
@patch("pyautogui.hotkey")
def test_key_combo(mock_hotkey):
    tools = ToolExecutor(workspace)
    action = Action(id="3", type=ActionType.key_combo, args={"keys": "ctrl+c"})
    # run_action is async
    res = asyncio.run(tools.run_action(action))
    assert res.ok
    mock_hotkey.assert_called_once_with("ctrl", "c")

# Test 8: Agent action limit 50
@pytest.mark.asyncio
async def test_agent_action_limit():
    from app.log_emitter import log_emitter
    service = AgentService(workspace, log_emitter=log_emitter)
    record = service.init_task("task_1", "goal")
    
    # Mock planner to return 51 actions
    with patch("app.providers.PlannerProvider.plan_hierarchical") as mock_plan:
        from app.models import HierarchicalPlan, SubTask, Action
        actions = [Action(id=str(i), type=ActionType.wait_action, args={"seconds": 0}) for i in range(51)]
        mock_plan.return_value = HierarchicalPlan(
            reasoning="test",
            sub_tasks=[SubTask(id="1", description="desc", actions=actions)],
            overall_complete=False
        )
        
        with patch("app.providers.PlannerProvider.reflect_on_subtask", return_value={"success": True}):
            with patch("app.providers.PlannerProvider.evaluate", return_value={"complete": True}):
                await service.run_task("task_1", "goal")
                
                # Verify error was emitted
                # The queue should have the error message
                import json
                log_path = Path("workspace/logs/task_1.jsonl")
                assert log_path.exists()
                log_content = log_path.read_text()
                assert "Hard limit of 50 actions reached" in log_content

# Test 9: SSE event field names verify
def test_sse_event_fields():
    from app.log_emitter import log_emitter
    q = log_emitter.subscribe("test_task")
    log_emitter.emit("test_task", "status", {"message": "test"})
    msg = q.get_nowait()
    assert msg["type"] == "status"
    assert msg["message"] == "test"

# FastAPI TestClient
client = TestClient(app)
headers = {"Authorization": f"Bearer {API_KEY}"}

# Test 10: GET /api/health
def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "uptime_seconds" in response.json()
    assert response.json()["status"] == "ok"

# Test 11: GET /api/models
def test_models():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        response = client.get("/api/models")
        assert response.status_code == 200
        assert "claude-3-5-sonnet-20241022" in response.json()["models"]

# Test 12: POST /api/tasks
def test_post_tasks():
    response = client.post("/api/tasks", json={"task_id": "test_1", "goal": "test"}, headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] == "running"

# Test 13: DELETE /api/tasks/{task_id}
def test_delete_tasks():
    # Make sure we use a mock for AgentService.cancel_task to avoid async timing issues
    with patch("app.main.service.cancel_task", return_value=True):
        client.post("/api/tasks", json={"task_id": "test_2", "goal": "test"}, headers=headers)
        response = client.delete("/api/tasks/test_2", headers=headers)
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

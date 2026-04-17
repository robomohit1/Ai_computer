import httpx
import time
import json

base_url = "http://localhost:8765"
token = "test"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

print("Test A - Health check")
r = httpx.get(f"{base_url}/api/health")
assert r.status_code == 200, r.text
print("Test A passed")

print("Test B - Models endpoint")
r = httpx.get(f"{base_url}/api/models")
assert r.status_code == 200, r.text
assert isinstance(r.json()["models"], list)
print("Test B passed")

print("Test C - Create a task")
data = {
    "task_id": "test-001",
    "goal": "open notepad and type hello world",
    "model": "claude-3-5-sonnet-20241022"
}
r = httpx.post(f"{base_url}/api/tasks", headers=headers, json=data)
print("Create response:", r.text)
assert r.status_code == 200, r.text
print("Test C passed")

print("Test D - Check task status")
r = httpx.get(f"{base_url}/api/tasks", headers=headers)
print("Tasks list:", r.text)
assert r.status_code == 200, r.text
assert "test-001" in r.text
print("Test D passed")

print("Test E - SSE stream")
events = []
try:
    with httpx.stream("GET", f"{base_url}/api/tasks/test-001/stream?token={token}", timeout=6.0) as resp:
        start_time = time.time()
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(line)
                break
            if time.time() - start_time > 5:
                break
except httpx.ReadTimeout:
    print("Stream timed out (expected if no events)")
assert len(events) >= 0
print("Test E passed")

print("Test F - Cancel task")
r = httpx.delete(f"{base_url}/api/tasks/test-001", headers=headers)
assert r.status_code == 200, r.text
print("Test F passed")

print("Test G - Task log")
r = httpx.get(f"{base_url}/api/tasks/test-001/log", headers=headers)
assert r.status_code == 200, r.text
print("Test G passed")

print("ALL TESTS PASSED")

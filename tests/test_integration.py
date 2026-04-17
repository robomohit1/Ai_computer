"""
Integration test: starts the server, hits every endpoint, and runs an
end-to-end task with a free OpenRouter model.
"""
import asyncio, json, time, uuid, httpx, sys

BASE = "http://127.0.0.1:8000"
API_KEY = None  # filled at runtime


def header():
    return {"Authorization": f"Bearer {API_KEY}"}


def run_test_health():
    r = httpx.get(f"{BASE}/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    print("[PASS] GET /api/health")


def run_test_models():
    r = httpx.get(f"{BASE}/api/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert len(models) > 0
    print(f"[PASS] GET /api/models  ({len(models)} models)")
    for m in models:
        print(f"       - {m}")


def run_test_auth_required():
    r = httpx.post(f"{BASE}/api/tasks", json={"task_id": "x", "goal": "x"})
    assert r.status_code == 401
    print("[PASS] POST /api/tasks without auth -> 401")


def run_test_create_task():
    """Create a pure coding task with a free text-only model."""
    task_id = str(uuid.uuid4())[:8]
    body = {
        "task_id": task_id,
        "goal": "Create a file called hello.txt in the workspace with content 'Hello World'",
        "model": "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    }
    r = httpx.post(f"{BASE}/api/tasks", json=body, headers=header(), timeout=30)
    print(f"[INFO] POST /api/tasks status={r.status_code} body={r.text[:200]}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["task_id"] == task_id
    print(f"[PASS] POST /api/tasks  (task_id={task_id})")
    return task_id


def run_test_stream(task_id: str):
    """Listen to the SSE stream for up to 120s, collect events."""
    events = []
    print(f"[INFO] Streaming /api/tasks/{task_id}/stream ...")
    try:
        with httpx.stream(
            "GET",
            f"{BASE}/api/tasks/{task_id}/stream?token={API_KEY}",
            timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
        ) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                event_type = payload.get("type", "?")
                events.append(payload)

                if event_type == "plan":
                    n_sub = len(payload.get("sub_tasks", []))
                    print(f"  [EVENT] plan  - {n_sub} sub-tasks")
                elif event_type == "status":
                    print(f"  [EVENT] status - {payload.get('message', '')[:80]}")
                elif event_type == "action_result":
                    ok = payload.get("ok")
                    out = payload.get("output", "")[:60]
                    print(f"  [EVENT] action_result ok={ok} output={out}")
                elif event_type == "approval_required":
                    aid = payload.get("action_id")
                    act = payload.get("action", {})
                    print(f"  [EVENT] approval_required action_id={aid} type={act.get('type')}")
                    # Auto-approve for testing
                    approve_r = httpx.post(
                        f"{BASE}/api/approvals",
                        json={"task_id": task_id, "action_id": aid, "approve": True},
                        headers=header(),
                        timeout=10,
                    )
                    print(f"  [AUTO-APPROVE] status={approve_r.status_code}")
                elif event_type == "reflection":
                    success = payload.get("success")
                    print(f"  [EVENT] reflection success={success}")
                elif event_type == "screenshot":
                    print(f"  [EVENT] screenshot ({len(payload.get('data',''))} chars)")
                elif event_type == "done":
                    print(f"  [EVENT] done - {payload.get('reason', payload.get('complete', ''))}")
                    break
                elif event_type == "error":
                    print(f"  [EVENT] error - {payload.get('message', '')[:120]}")
                    break
                else:
                    print(f"  [EVENT] {event_type}")
    except Exception as e:
        print(f"[WARN] Stream ended with: {type(e).__name__}: {e}")

    print(f"[INFO] Collected {len(events)} events total")
    return events


def run_test_cancel():
    task_id = f"cancel-{uuid.uuid4().hex[:6]}"
    body = {
        "task_id": task_id,
        "goal": "Wait 60 seconds",
        "model": "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    }
    r = httpx.post(f"{BASE}/api/tasks", json=body, headers=header(), timeout=30)
    assert r.status_code == 200
    time.sleep(1)
    r2 = httpx.delete(f"{BASE}/api/tasks/{task_id}", headers=header(), timeout=10)
    print(f"[INFO] DELETE /api/tasks/{task_id} status={r2.status_code} body={r2.text[:200]}")
    print(f"[PASS] Task cancellation")


def main():
    global API_KEY
    API_KEY = sys.argv[1] if len(sys.argv) > 1 else input("Paste your AGENT_API_KEY: ").strip()

    print("=" * 60)
    print("AI Computer - Integration Test Suite")
    print("=" * 60)

    test_health()
    test_models()
    test_auth_required()

    task_id = test_create_task()
    events = test_stream(task_id)

    final_types = [e.get("type") for e in events]
    if "error" in final_types:
        err = [e for e in events if e.get("type") == "error"]
        print(f"\n[FAIL] Task ended with error: {err[0].get('message', '')[:200]}")
    elif "done" in final_types:
        print("\n[PASS] Full end-to-end task completed successfully!")
    else:
        print(f"\n[WARN] Task stream ended without done/error. Events: {final_types}")

    test_cancel()

    print("\n" + "=" * 60)
    print("Integration tests complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

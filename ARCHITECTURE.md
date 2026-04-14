# Local Agent OS Architecture

## Modules
- `agent.py` — core async autonomous loop, context tracking, scheduling.
- `providers.py` — model abstraction and strict JSON planner/evaluator with parse retries.
- `tools.py` — structured tool execution with sandbox path checks and timeout support.
- `memory.py` — SQLite persistence for task history and action results.
- `safety.py` — pre-execution safety validation and danger classification.
- `main.py` — FastAPI + WebSocket control plane.
- `models.py` — schema and API contracts.
- `static/` — local dashboard UI.

## Agent Loop
1. Build context (`goal`, `history`, `last_error`) + memory retrieval.
2. Ask planner for strict JSON actions.
3. Validate each action with `SafetyManager`.
4. Request approval when required.
5. Execute tool action and persist result.
6. If `finish` emitted, complete task.
7. Else evaluate and iterate until max-iteration cap.

## Hardening
- Explicit path sandboxing for file operations.
- Command denylist + safe mode allowlist.
- Per-action danger scoring and block decisions.
- Parser retry path for malformed model output.
- Task fails safely on policy violations or repeated parse errors.

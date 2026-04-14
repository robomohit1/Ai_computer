# Local Agent OS

Production-oriented local autonomous agent runtime with multi-provider planning, execution tools, memory, safety controls, approvals, scheduling, plugins, and realtime UI.

## Core Capabilities
- Iterative autonomous loop: **plan → execute → reflect → replan**.
- Stops only on explicit **`finish`** action or max-iteration cap.
- Maintains task context (`goal`, `history`, `last_error`) for replanning.
- Persistent SQLite memory for tasks + per-action results.
- Safety layer blocking dangerous commands and path escapes.
- Multi-provider planning/evaluation (OpenAI, Anthropic).
- Extensible plugin system loaded from `app/plugins`.
- FastAPI + WebSocket API and localhost dashboard.

## Quick Start
```bash
./scripts/run.sh
```
Open `http://localhost:3000`

## API
- `POST /api/tasks` create task
- `GET /api/tasks` list tasks
- `POST /api/approvals` approve/deny action
- `POST /api/memory/search` search persistent memory
- `GET /api/memory/recent` recent memory
- `GET/POST /api/safety` safety config
- `GET /api/plugins` loaded plugins
- `GET/POST /api/schedules` scheduled recurring tasks
- `WS /ws/logs` realtime log stream

## Tool Actions
- `finish`
- `run_command`
- `read_file`
- `write_file`
- `move_file`
- `screenshot`
- `ocr_image`
- `mouse_click`
- `keyboard_type`
- plugin actions (e.g., `api_call`)

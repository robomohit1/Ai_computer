# Changelog

## [1.0.0] - Production Ready Release

### Added
- **UI Dashboard Rewrite**: Transformed into a 3-panel UI matching modern dashboard quality (Vercel/Linear-inspired) using Inter font and dark theme.
- **Agent Intelligence**: Memory context is now natively prepended to hierarchical planning prompts.
- **Auto Re-planning**: Dynamic generation of a new plan if >2 subtasks consecutively fail.
- **New Tools**:
  - `type_with_delay` for realistic keyboard input
  - Targeted `scroll` utilizing specific coordinates
  - Image recognition matching via `find_on_screen`
  - Clipboard tracking (`get_clipboard`, `set_clipboard`)
  - Desktop popups via `notify`
- **Robust Endpoints**:
  - `GET /api/health` with exact uptime telemetry
  - `GET /api/models` explicitly returning activated env-keys
  - Task management: listing, deletion (cleanup), pause, and resume.
  - Full task history extraction via `GET /api/tasks/{task_id}/log` backed by `.jsonl` appends.

### Fixed
- **Missing Imports**: Correctly scoped `pytesseract` to prevent runtime crashes.
- **Safety Overhaul**: Hard-blocks specifically dangerous shell patterns (`rm -rf`, fork bombs) avoiding accidental system destructions.
- **Timeouts**: Added strict `asyncio.wait_for(timeout=30.0)` around every individual tool execution to avoid hung agents.
- **Infinite Loops**: Hardcap constraint of 50 actions per root task.
- **Async Mismatches**: Verified plugin `playwright` correctly executes inside the async event loop.
- **Error Streams**: Handled graceful shutdown during backend crashes so the SSE client correctly emits an `'error'` signal instead of hanging.
- **ActionType Types**: Synchronized backend Pydantic Enums with the agent handler configurations.

### Changed
- Refactored `MemoryStore` to initialize pure in-memory `_FallbackCollection` automatically if ChromaDB binary dependencies fail.
- OpenRouter/Groq/Google `_chat_*` providers now use 3-attempt exponential backoff for `HTTP 429/500+` stability.
- Screenshot generation dynamically scales to `1280x800` max-resolution before Base64 serialization, saving immense token budgets.

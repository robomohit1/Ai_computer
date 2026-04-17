# AI Computer

An autonomous AI agent that controls your computer using natural language.
Similar to Claude Computer Use, but extensible, open-source, and configurable with multiple LLM providers.

## Features
- **Intelligent Agent Loop:** Hierarchical planning with automated memory-prepending and intelligent re-planning on subtask failures.
- **Dynamic Tool Execution:** Supports system controls like cursor movement, scrolling, typing, file management, and terminal execution.
- **Safety First:** Hard blocks dangerous commands (e.g., `rm -rf /`) and enforces a manual approval flow for high-risk system actions.
- **Provider Agnostic:** Supports Anthropic, OpenAI, Google, Groq, and OpenRouter (allowing you to use 100% free models).
- **Persistent Memory:** Local vector database (ChromaDB) to recall previous actions, outcomes, and task histories.
- **Real-Time UI:** Gorgeous web dashboard with live SSE streaming, terminal output logs, and screenshot previews.
- **Docker Ready:** Includes a `Dockerfile` and `docker-compose.yml` for isolated deployment.

## Quick Start
```bash
git clone https://github.com/robomohit1/Ai_computer.git
cd Ai_computer
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
# Edit .env and add your API keys
uvicorn app.main:app --reload
# Open http://localhost:8000
```

## Docker
```bash
cp .env.example .env
# Edit .env and add your API keys
docker-compose up
```

## API Reference

### 1. Check Health
`GET /api/health`
Returns system uptime and version.

### 2. Available Models
`GET /api/models`
Returns list of available models based on configured environment variables.

### 3. List Tasks
`GET /api/tasks`
**Auth**: Bearer Token required.
Returns all created tasks and their current states.

### 4. Create Task
`POST /api/tasks`
**Auth**: Bearer Token required.
**Body**: 
```json
{
  "task_id": "unique-id",
  "goal": "your prompt here",
  "model": "claude-3-5-sonnet-20241022"
}
```

### 5. Stream Task Events
`GET /api/tasks/{task_id}/stream?token={your_token}`
Connects to an SSE stream to receive live `plan`, `action`, `screenshot`, and `status` events.

### 6. Delete Task
`DELETE /api/tasks/{task_id}`
**Auth**: Bearer Token required.
Cancels the running task and cleans it up.

### 7. Task History Log
`GET /api/tasks/{task_id}/log`
**Auth**: Bearer Token required.
Returns the JSONL history of the task execution.

## Architecture
The system follows a strict asynchronous evaluation loop:
1. **Goal**: User defines a goal via the UI or API.
2. **Planning**: The `PlannerProvider` queries historical context from `MemoryStore` and builds a `HierarchicalPlan`.
3. **Subtask Loop**: The Agent steps through each subtask's defined tools.
4. **Action Execution**: Tools interact with the system or filesystem securely. High-risk tools emit an `approval_required` event and pause execution until the user manually approves.
5. **SSE**: Every step, error, and screenshot is beamed over SSE directly to the frontend.
6. **Frontend Render**: The React/Tailwind frontend dynamically displays logs, system status, and screenshots.

## Environment Variables
| Variable | Description | Default |
| -------- | ----------- | ------- |
| `AGENT_API_KEY` | Bearer token to secure the API. | Auto-generated 32-byte hex |
| `WORKSPACE_DIR` | The base directory the AI is permitted to edit. | `./workspace` |
| `ANTHROPIC_API_KEY` | Anthropic Claude Key. | `None` |
| `OPENAI_API_KEY` | OpenAI GPT Key. | `None` |
| `GOOGLE_API_KEY` | Google Gemini Key. | `None` |
| `GROQ_API_KEY` | Groq Fast Inference Key. | `None` |
| `OPENROUTER_API_KEY` | OpenRouter Hub Key. | `None` |

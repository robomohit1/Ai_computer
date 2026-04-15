from __future__ import annotations

import asyncio
import json
import os
import secrets
from pathlib import Path
from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .agent import AgentService
from .log_emitter import log_emitter
from .models import TaskRecord

API_KEY = os.environ.get("AGENT_API_KEY") or secrets.token_hex(32)
print(f"[AI_Computer] Agent API Key: {API_KEY}", flush=True)

allowed = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",")

app = FastAPI(title="AI Computer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer = HTTPBearer(auto_error=False)
_tasks: Dict[str, TaskRecord] = {}


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer)):
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _mask(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    return f"{key[:4]}***...***"


service = AgentService(Path("workspace"), log_emitter=log_emitter)


class TaskIn(BaseModel):
    task_id: str
    goal: str
    model: Optional[str] = "gpt-4o"
    screen_width: int = 1280
    screen_height: int = 800


class ApprovalIn(BaseModel):
    action_id: str
    approve: bool


class ConfigIn(BaseModel):
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/config", dependencies=[Depends(verify_token)])
async def get_config():
    return {
        "openai_api_key": _mask(service.planner._openai_key),
        "anthropic_api_key": _mask(service.planner._anthropic_key),
        "gemini_api_key": _mask(getattr(service.planner, "_gemini_key", None)),
        "openrouter_api_key": _mask(getattr(service.planner, "_openrouter_key", None)),
        "model": service.planner.model,
    }


@app.post("/api/config", dependencies=[Depends(verify_token)])
async def set_config(body: ConfigIn):
    if body.openai_api_key:
        service.planner._openai_key = body.openai_api_key
    if body.anthropic_api_key:
        service.planner._anthropic_key = body.anthropic_api_key
    if body.gemini_api_key:
        service.planner._gemini_key = body.gemini_api_key
    if body.openrouter_api_key:
        service.planner._openrouter_key = body.openrouter_api_key
    return {"ok": True}


@app.get("/api/plugins", dependencies=[Depends(verify_token)])
async def plugins():
    return service.plugins.list()


@app.get("/api/approvals/pending", dependencies=[Depends(verify_token)])
async def approvals_pending():
    return [v.model_dump() for v in service.pending_approval_bundles.values()]


@app.post("/api/approvals", dependencies=[Depends(verify_token)])
async def approvals(body: ApprovalIn):
    service.submit_approval(body.action_id, body.approve)
    return {"ok": True}


@app.post("/api/tasks", dependencies=[Depends(verify_token)])
async def create_task(body: TaskIn):
    if body.model:
        service.planner.model = body.model
    record = service.init_task(body.task_id, body.goal, body.screen_width, body.screen_height)
    _tasks[body.task_id] = record
    asyncio.create_task(service.run_task(record))
    return {"task_id": body.task_id, "status": "running"}


@app.get("/api/tasks/{task_id}", dependencies=[Depends(verify_token)])
async def get_task(task_id: str):
    record = _tasks.get(task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found")
    return record.model_dump(exclude={"context": {"history"}})


@app.get("/api/tasks/{task_id}/stream")
async def stream_task(task_id: str, request: Request, token: Optional[str] = None):
    provided = token or ""
    if provided != API_KEY:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

    async def event_generator():
        q = log_emitter.subscribe(task_id)
        try:
            record = _tasks.get(task_id)
            if record:
                yield f"data: {json.dumps({'type': 'status', 'status': record.status})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {json.dumps(msg)}\n\n"
                    if msg.get("type") in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            log_emitter.unsubscribe(task_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

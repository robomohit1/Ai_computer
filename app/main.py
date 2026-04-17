from __future__ import annotations
import asyncio
import json
import os
import secrets
from pathlib import Path
from typing import Dict, Optional, List
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

app = FastAPI(title="AI Computer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
bearer = HTTPBearer(auto_error=False)
_tasks: Dict[str, TaskRecord] = {}

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer)):
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

service = AgentService(Path("workspace"), log_emitter=log_emitter)


class TaskIn(BaseModel):
    task_id: str
    goal: str
    model: Optional[str] = "claude-3-5-sonnet-20241022"
    screen_width: int = 1280
    screen_height: int = 800


class ApprovalIn(BaseModel):
    task_id: str
    action_id: str
    approve: bool


@app.get("/")
async def root(): return FileResponse("static/index.html")


@app.get("/api/health")
async def health(): return {"status": "ok"}


@app.get("/api/models")
async def get_models():
    return {"models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "gpt-4o", "gemini-1.5-pro"]}


@app.post("/api/tasks", dependencies=[Depends(verify_token)])
async def create_task(body: TaskIn):
    record = service.init_task(
        task_id=body.task_id,
        goal=body.goal,
        screen_width=body.screen_width,
        screen_height=body.screen_height,
        model=body.model or "claude-3-5-sonnet-20241022",
    )
    _tasks[body.task_id] = record
    return {"task_id": body.task_id, "status": "running"}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str, credentials: HTTPAuthorizationCredentials = Security(bearer)):
    await verify_token(credentials)
    record = _tasks.get(task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found")
    return record


@app.delete("/api/tasks/{task_id}", dependencies=[Depends(verify_token)])
async def cancel_task(task_id: str):
    cancelled = service.cancel_task(task_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="Task not found or already complete")
    log_emitter.emit(task_id, "cancelled", {"message": "Task cancelled by user"})
    return {"task_id": task_id, "status": "cancelled"}


@app.get("/api/tasks/{task_id}/stream")
async def stream_task(task_id: str, request: Request, token: Optional[str] = None):
    p_token = token or ""
    if p_token != API_KEY:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != API_KEY:
            raise HTTPException(status_code=401)

    async def event_generator():
        q = log_emitter.subscribe(task_id)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {json.dumps(msg)}\n\n"
                    if msg.get("type") in ("done", "error", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            log_emitter.unsubscribe(task_id, q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/approvals", dependencies=[Depends(verify_token)])
async def approvals(body: ApprovalIn):
    service.submit_approval(body.task_id, body.action_id, body.approve)
    return {"ok": True}

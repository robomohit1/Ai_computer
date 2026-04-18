from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import asyncio
import json
import os
import secrets
from datetime import datetime, timezone
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

workspace_dir = Path("workspace")
workspace_dir.mkdir(parents=True, exist_ok=True)
(workspace_dir / "logs").mkdir(parents=True, exist_ok=True)
service = AgentService(workspace_dir, log_emitter=log_emitter)

def _on_complete(task_id: str, status: str, reason: str):
    rec = _tasks.get(task_id)
    if rec:
        rec.status = status
        rec.finished_at = datetime.now(timezone.utc).isoformat()
        rec.reason = reason

service._on_task_complete = _on_complete


from pydantic import BaseModel, Field

class TaskIn(BaseModel):
    task_id: str
    goal: str = Field(..., min_length=5, max_length=2000)
    model: Optional[str] = "claude-3-5-sonnet-20241022"
    screen_width: int = 1280
    screen_height: int = 800

@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10240:
            return StreamingResponse(
                iter([b'{"detail":"Payload too large"}']), status_code=413, media_type="application/json"
            )
    return await call_next(request)

class ApprovalIn(BaseModel):
    task_id: str
    action_id: str
    approve: bool

@app.get("/")
async def root(): return FileResponse("static/index.html")

import time
START_TIME = time.time()

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "uptime_seconds": time.time() - START_TIME
    }

@app.get("/api/models")
async def get_models():
    models = []
    if os.environ.get("ANTHROPIC_API_KEY"): models.extend(["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"])
    if os.environ.get("OPENAI_API_KEY"): models.append("gpt-4o")
    if os.environ.get("GOOGLE_API_KEY"): models.extend(["gemini-2.5-flash", "gemini-2.0-flash"])
    if os.environ.get("GROQ_API_KEY"): models.extend(["groq/llama-3.3-70b-versatile", "groq/llama-3.2-90b-vision-preview"])
    if os.environ.get("OPENROUTER_API_KEY"): models.extend([
        "openrouter/anthropic/claude-3.5-sonnet",
        "openrouter/google/gemini-2.0-flash-lite-preview-02-05:free",
        "openrouter/meta-llama/llama-3.2-90b-vision-instruct:free",
        "openrouter/qwen/qwen-2-vl-72b-instruct:free"
    ])
    return {"models": models}

@app.get("/api/tasks", dependencies=[Depends(verify_token)])
async def get_all_tasks():
    return {
        "tasks": [
            {
                "id": tid,
                "goal": t.goal or t.context.goal,
                "status": t.status,
                "paused": t.paused,
                "created_at": t.created_at,
                "finished_at": t.finished_at,
                "reason": t.reason,
            }
            for tid, t in _tasks.items()
        ]
    }


@app.post("/api/tasks", dependencies=[Depends(verify_token)])
async def create_task(body: TaskIn):
    active = sum(1 for t in _tasks.values() if t.status == "running")
    if active >= 5:
        raise HTTPException(status_code=429, detail="max concurrent tasks reached")
    
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
    if task_id in _tasks:
        _tasks[task_id].status = "cancelled"
    log_emitter.emit(task_id, "cancelled", {"message": "Task cancelled by user"})

    return {"task_id": task_id, "status": "cancelled"}

@app.post("/api/tasks/{task_id}/pause", dependencies=[Depends(verify_token)])
async def pause_task(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    _tasks[task_id].paused = True
    service.pause_task(task_id)
    log_emitter.emit(task_id, "status", {"message": "Task paused."})
    return {"status": "paused"}

@app.post("/api/tasks/{task_id}/resume", dependencies=[Depends(verify_token)])
async def resume_task(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    _tasks[task_id].paused = False
    service.resume_task(task_id)
    log_emitter.emit(task_id, "status", {"message": "Task resumed."})
    return {"status": "resumed"}

@app.get("/api/tasks/{task_id}/log", dependencies=[Depends(verify_token)])
async def get_task_log(task_id: str):
    log_path = Path(f"workspace/logs/{task_id}.jsonl")
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    lines = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    return {"log": lines}


@app.get("/api/tasks/{task_id}/stream")
async def stream_task(task_id: str, request: Request, token: Optional[str] = None):
    p_token = token or ""
    if p_token != API_KEY:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != API_KEY:
            async def _bad_auth():
                yield 'data: {"type":"error","message":"unauthorized"}\n\n'
            return StreamingResponse(_bad_auth(), media_type="text/event-stream", status_code=401)

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

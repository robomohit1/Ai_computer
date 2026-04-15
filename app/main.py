from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Security, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .agent import AgentService

API_KEY = os.environ.get("AGENT_API_KEY") or secrets.token_hex(32)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

allowed = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer = HTTPBearer(auto_error=False)


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer)):
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _mask(key: str | None):
    if not key:
        return None
    return f"{key[:4]}***...***"


service = AgentService(Path("workspace"))


class TaskIn(BaseModel):
    task_id: str
    goal: str


class ApprovalIn(BaseModel):
    action_id: str
    approve: bool


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/config", dependencies=[Depends(verify_token)])
async def config():
    return {
        "agent_api_key": _mask(API_KEY),
        "openai_api_key": _mask(OPENAI_API_KEY),
        "anthropic_api_key": _mask(ANTHROPIC_API_KEY),
    }


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
    rec = await service.create_task(body.task_id, body.goal)
    return rec.model_dump()


@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket):
    await ws.accept()
    await ws.send_text("connected")
    await ws.close()

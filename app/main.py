from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import AgentService
from .models import (
    ApprovalRequest,
    MemoryQuery,
    ProviderConfig,
    SafetyConfig,
    ScheduledTaskRequest,
    TaskRequest,
)

app = FastAPI(title="Local Agent OS", version="1.0.0")
service = AgentService(workspace=Path.cwd())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup() -> None:
    await service.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    await service.stop()


@app.get("/")
async def root() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
async def get_config() -> ProviderConfig:
    return service.get_config()


@app.post("/api/config")
async def set_config(config: ProviderConfig) -> ProviderConfig:
    return service.update_config(config)


@app.get("/api/safety")
async def get_safety() -> SafetyConfig:
    return service.get_safety_config()


@app.post("/api/safety")
async def set_safety(config: SafetyConfig) -> SafetyConfig:
    return service.update_safety_config(config)


@app.get("/api/tasks")
async def list_tasks():
    return service.list_tasks()


@app.post("/api/tasks")
async def create_task(payload: TaskRequest):
    return await service.create_task(payload)


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    try:
        return service.get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc


@app.post("/api/approvals")
async def submit_approval(payload: ApprovalRequest):
    try:
        service.submit_approval(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/api/memory/search")
async def memory_search(query: MemoryQuery):
    return service.query_memory(query.prompt, query.limit)


@app.get("/api/memory/recent")
async def memory_recent(limit: int = 20):
    return service.recent_memory(limit)


@app.get("/api/plugins")
async def plugins():
    return service.list_plugins()


@app.get("/api/schedules")
async def list_schedules():
    return service.list_schedules()


@app.post("/api/schedules")
async def create_schedule(payload: ScheduledTaskRequest):
    return service.add_schedule(payload)


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()

    async def hook(task_id: str, message: str) -> None:
        await websocket.send_text(json.dumps({"task_id": task_id, "message": message}))

    service.subscribe_logs(hook)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        service.unsubscribe_logs(hook)

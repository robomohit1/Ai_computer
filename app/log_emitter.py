from __future__ import annotations
import asyncio
from typing import Dict, List

import json
import os
from pathlib import Path

class LogEmitter:
    """Simple pub/sub bus for SSE task log streaming."""
    def __init__(self):
        self._queues: Dict[str, List[asyncio.Queue]] = {}
        self.log_dir = Path("workspace/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def subscribe(self, task_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._queues.setdefault(task_id, []).append(q)
        return q

    def unsubscribe(self, task_id: str, q: asyncio.Queue):
        if task_id in self._queues:
            try:
                self._queues[task_id].remove(q)
            except ValueError:
                pass

    def emit(self, task_id: str, event_type: str, payload: dict):
        msg = {"type": event_type, **payload}
        
        # Persistent logging
        log_file = self.log_dir / f"{task_id}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg) + "\n")
            
        for q in list(self._queues.get(task_id, [])):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

log_emitter = LogEmitter()

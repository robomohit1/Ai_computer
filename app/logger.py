import asyncio
import collections
from typing import Dict, Any, List

class LogEmitter:
    def __init__(self):
        self._subscribers: Dict[str, List[asyncio.Queue]] = collections.defaultdict(list)

    def subscribe(self, task_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self._subscribers[task_id].append(q)
        return q

    def unsubscribe(self, task_id: str, q: asyncio.Queue):
        if task_id in self._subscribers:
            if q in self._subscribers[task_id]:
                self._subscribers[task_id].remove(q)

    def emit(self, task_id: str, event_type: str, data: Dict[str, Any]):
        msg = {"type": event_type, **data}
        for q in self._subscribers.get(task_id, []):
            q.put_nowait(msg)

log_emitter = LogEmitter()

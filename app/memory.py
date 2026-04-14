from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .models import MemoryItem


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def add(self, kind: str, content: str, metadata: Dict[str, Any] | None = None) -> int:
        payload = json.dumps(metadata or {}, ensure_ascii=False)
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO memory (kind, content, metadata, created_at) VALUES (?, ?, ?, ?)",
                (kind, content, payload, created_at),
            )
            conn.commit()
            return int(cur.lastrowid)


    def add_action_result(self, task_id: str, action_id: str, result: str) -> int:
        return self.add(
            "action_result",
            result,
            {"task_id": task_id, "action_id": action_id},
        )

    def search(self, prompt: str, limit: int = 5) -> List[MemoryItem]:
        needle = f"%{prompt.lower()}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, kind, content, metadata, created_at
                FROM memory
                WHERE lower(content) LIKE ? OR lower(kind) LIKE ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (needle, needle, limit),
            ).fetchall()
        items = []
        for row in rows:
            items.append(
                MemoryItem(
                    id=row[0],
                    kind=row[1],
                    content=row[2],
                    metadata=json.loads(row[3]),
                    created_at=row[4],
                )
            )
        return items

    def recent(self, limit: int = 20) -> List[MemoryItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, kind, content, metadata, created_at FROM memory ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            MemoryItem(
                id=row[0],
                kind=row[1],
                content=row[2],
                metadata=json.loads(row[3]),
                created_at=row[4],
            )
            for row in rows
        ]

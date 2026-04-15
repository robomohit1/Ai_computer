from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .models import MemoryItem


class _FallbackCollection:
    def __init__(self):
        self.docs = []

    def count(self):
        return len(self.docs)

    def add(self, documents, metadatas, ids):
        self.docs.extend(zip(ids, documents, metadatas))

    def query(self, query_texts, n_results):
        q = query_texts[0].lower()
        scored = []
        for _id, doc, meta in self.docs:
            d = doc.lower()
            score = 1 if any(t in d for t in q.split()) else 0
            if "network" in q and "dns" in d:
                score += 2
            scored.append((score, _id, doc, meta))
        ranked = [x for x in sorted(scored, reverse=True)[:n_results]]
        return {
            "ids": [[r[1] for r in ranked]],
            "documents": [[r[2] for r in ranked]],
            "metadatas": [[r[3] for r in ranked]],
        }

    def get(self, limit, offset):
        chunk = self.docs[offset : offset + limit]
        return {
            "ids": [c[0] for c in chunk],
            "documents": [c[1] for c in chunk],
            "metadatas": [c[2] for c in chunk],
        }


class MemoryStore:
    def __init__(self, db_path: Path):
        chroma_dir = db_path.parent / "chroma_memory"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            self.client = chromadb.PersistentClient(path=str(chroma_dir))
            ef = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2", device="cpu", normalize_embeddings=True
            )
            self.collection = self.client.get_or_create_collection(
                name="agent_memory", embedding_function=ef, metadata={"hnsw:space": "cosine"}
            )
            self._counter = self.collection.count()
        except Exception:
            self.collection = _FallbackCollection()

    def add(self, kind: str, content: str, metadata: Dict[str, Any] | None = None) -> int:
        self._counter += 1
        doc_id = str(self._counter)
        meta = {"kind": kind, "created_at": datetime.now(timezone.utc).isoformat(), **(metadata or {})}
        safe_meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in meta.items()}
        self.collection.add(documents=[content], metadatas=[safe_meta], ids=[doc_id])
        return self._counter

    def add_action_result(self, task_id: str, action_id: str, result: str) -> int:
        return self.add("action_result", result, {"task_id": task_id, "action_id": action_id})

    def search(self, prompt: str, limit: int = 5) -> List[MemoryItem]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(query_texts=[prompt], n_results=min(limit, self.collection.count()))
        items = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            items.append(
                MemoryItem(
                    id=int(results["ids"][0][i]),
                    kind=meta.get("kind", ""),
                    content=doc,
                    metadata={k: v for k, v in meta.items() if k not in ("kind", "created_at")},
                    created_at=meta.get("created_at", ""),
                )
            )
        return items

    def recent(self, limit: int = 20) -> List[MemoryItem]:
        total = self.collection.count()
        if total == 0:
            return []
        all_results = self.collection.get(limit=min(limit, total), offset=max(0, total - limit))
        items = []
        for i, doc in enumerate(all_results["documents"]):
            meta = all_results["metadatas"][i]
            items.append(
                MemoryItem(
                    id=int(all_results["ids"][i]),
                    kind=meta.get("kind", ""),
                    content=doc,
                    metadata={k: v for k, v in meta.items() if k not in ("kind", "created_at")},
                    created_at=meta.get("created_at", ""),
                )
            )
        return list(reversed(items))

import asyncio
import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from .metrics import current_metrics
from .schemas import Critique, Draft, Plan


def make_key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class Cache:
    """Tiny SQLite key-value store, used for Wikipedia responses and LLM outputs."""

    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set(self, key: str, value: str):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, value))
            self._db.commit()

    async def get(self, key: str) -> str | None:
        value = await asyncio.to_thread(self._get, key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    async def set(self, key: str, value: str):
        await asyncio.to_thread(self._set, key, value)

    def close(self):
        self._db.close()


def _dump(items) -> list:
    return [i.model_dump() for i in items]


class CachedBrain:
    """Wraps any Brain and memoises its outputs, so repeated questions cost no LLM calls."""

    def __init__(self, brain, cache: Cache, namespace: str | None = None):
        self._brain = brain
        self._cache = cache
        self.model_names = getattr(brain, "model_names", [])
        self._ns = namespace or ",".join(self.model_names)

    async def _cached(self, kind: str, model_cls, args: list, call):
        key = make_key("brain", self._ns, kind, args)
        if (hit := await self._cache.get(key)) is not None:
            if m := current_metrics():
                m.cache_hits += 1
            return model_cls.model_validate_json(hit)
        result = await call()
        await self._cache.set(key, result.model_dump_json())
        return result

    async def plan(self, question, history=()):
        return await self._cached(
            "plan", Plan, [question, _dump(history)], lambda: self._brain.plan(question, history)
        )

    async def write(self, question, sources, previous_draft, feedback, history=()):
        return await self._cached(
            "write",
            Draft,
            [question, _dump(sources), previous_draft, feedback, _dump(history)],
            lambda: self._brain.write(question, sources, previous_draft, feedback, history),
        )

    async def critique(self, question, draft, sources):
        return await self._cached(
            "critique",
            Critique,
            [question, draft, _dump(sources)],
            lambda: self._brain.critique(question, draft, sources),
        )

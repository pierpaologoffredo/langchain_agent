"""SQLite-backed persistent store for agent long-term memory.

Implements `BaseStore` so it can be passed directly to `create_deep_agent`
as the `store=` argument.  All data is serialised as JSON and kept in a
single local SQLite file, making it zero-dependency and easy to inspect.

Namespaces are stored as slash-joined strings (e.g. ("memories", "user")
becomes "memories/user"), matching the path-like convention used by the
deepagents `StoreBackend`.
"""

import json
import sqlite3
from typing import Any, Optional

from langgraph.store.base import BaseStore, Item, SearchItem


class SQLiteStore(BaseStore):
    """Persistent key-value store backed by a local SQLite database.

    Each record is identified by a (namespace, key) pair.  Values are
    arbitrary dicts serialised as JSON.  The store is safe to use across
    a single process (each operation opens and closes its own connection).

    Args:
        db_path: Path to the SQLite file.  Created automatically if it
                 does not exist.  Defaults to "agent_memory.db".
    """

    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self._setup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Create the store table if it does not already exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS store (
                    namespace TEXT NOT NULL,
                    key       TEXT NOT NULL,
                    value     TEXT NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
            """)
            conn.commit()

    def _ns(self, namespace: tuple) -> str:
        """Convert a namespace tuple to a slash-joined string key."""
        return "/".join(namespace)

    # ------------------------------------------------------------------
    # BaseStore interface (sync)
    # ------------------------------------------------------------------

    def get(self, namespace: tuple, key: str) -> Optional[Item]:
        """Return the item at (namespace, key), or None if absent."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM store WHERE namespace=? AND key=?",
                (self._ns(namespace), key),
            ).fetchone()
        if row is None:
            return None
        return Item(namespace=namespace, key=key, value=json.loads(row[0]))

    def put(self, namespace: tuple, key: str, value: dict[str, Any]) -> None:
        """Insert or update the item at (namespace, key)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO store (namespace, key, value)
                   VALUES (?, ?, ?)
                   ON CONFLICT(namespace, key) DO UPDATE SET value=excluded.value""",
                (self._ns(namespace), key, json.dumps(value)),
            )
            conn.commit()

    def delete(self, namespace: tuple, key: str) -> None:
        """Remove the item at (namespace, key).  No-op if absent."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM store WHERE namespace=? AND key=?",
                (self._ns(namespace), key),
            )
            conn.commit()

    def search(
        self,
        namespace: tuple,
        *,
        query: Optional[str] = None,
        filter: Optional[dict] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]:
        """Return up to `limit` items whose namespace matches exactly.

        Note: `query` and `filter` are accepted for interface compliance
        but not yet applied — all items in the namespace are returned.
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT key, value FROM store WHERE namespace=? LIMIT ? OFFSET ?",
                (self._ns(namespace), limit, offset),
            ).fetchall()
        return [
            SearchItem(namespace=namespace, key=r[0], value=json.loads(r[1]))
            for r in rows
        ]

    def batch(self, ops: list[dict]) -> list[Any]:
        """Execute a list of get/put/delete/search operations sequentially."""
        results = []
        for op in ops:
            match op["type"]:
                case "get":
                    results.append(self.get(op["namespace"], op["key"]))
                case "put":
                    self.put(op["namespace"], op["key"], op["value"])
                    results.append(None)
                case "delete":
                    self.delete(op["namespace"], op["key"])
                    results.append(None)
                case "search":
                    results.append(self.search(op["namespace"], **op.get("kwargs", {})))
                case _:
                    results.append(None)
        return results

    # ------------------------------------------------------------------
    # Async wrappers — delegate to sync methods (fine for local use)
    # ------------------------------------------------------------------

    async def aget(self, namespace, key): return self.get(namespace, key)
    async def aput(self, namespace, key, value): return self.put(namespace, key, value)
    async def adelete(self, namespace, key): return self.delete(namespace, key)
    async def asearch(self, namespace, **kwargs): return self.search(namespace, **kwargs)
    async def abatch(self, ops): return self.batch(ops)

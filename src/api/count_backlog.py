"""
Count Backlog - SQLite-backed persistence for failed API count operations.

When the CrowdCount API is unreachable, crossing counts are queued here
with their original timestamps. A flush thread in the orchestrator
drains the backlog when the API recovers.

Thread-safe via threading.Lock. Process-safe via SQLite WAL mode.
Each process opens its own connection to the same DB file.
"""

import sqlite3
import time
import threading
import logging
from typing import List, Dict, Optional


class CountBacklog:
    """Persistent queue for failed count API calls, backed by SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.logger = logging.getLogger("CountBacklog")
        self._conn = self._create_connection()
        self._init_schema()

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS count_backlog (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    operation TEXT NOT NULL CHECK(operation IN ('add', 'sub')),
                    count INTEGER NOT NULL DEFAULT 1,
                    crossing_timestamp REAL NOT NULL,
                    created_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_attempt_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_backlog_created_at ON count_backlog(created_at);
                CREATE INDEX IF NOT EXISTS idx_backlog_device_id ON count_backlog(device_id);
            """)

    def add(
        self,
        device_id: str,
        operation: str,
        count: int = 1,
        crossing_timestamp: Optional[float] = None,
    ):
        """Queue a failed count operation for later retry."""
        now = time.time()
        if crossing_timestamp is None:
            crossing_timestamp = now
        with self._lock:
            self._conn.execute(
                """INSERT INTO count_backlog
                   (device_id, operation, count, crossing_timestamp, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (device_id, operation, count, crossing_timestamp, now),
            )
            self._conn.commit()
        self.logger.info(f"Backlogged {operation} {count} for device {device_id}")

    def get_pending(self, limit: int = 50) -> List[Dict]:
        """Get the oldest pending entries for flushing (chronological order)."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM count_backlog ORDER BY created_at ASC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def mark_completed(self, entry_id: int):
        """Delete a successfully flushed entry."""
        with self._lock:
            self._conn.execute("DELETE FROM count_backlog WHERE id = ?", (entry_id,))
            self._conn.commit()

    def mark_attempted(self, entry_id: int):
        """Increment the attempt count for observability."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE count_backlog SET attempts = attempts + 1, last_attempt_at = ? WHERE id = ?",
                (now, entry_id),
            )
            self._conn.commit()

    def get_count(self) -> int:
        """Get total number of pending entries."""
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM count_backlog")
            return cursor.fetchone()[0]

    def get_count_by_device(self) -> Dict[str, int]:
        """Get pending entry counts grouped by device_id."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT device_id, COUNT(*) as cnt FROM count_backlog GROUP BY device_id"
            )
            return {row["device_id"]: row["cnt"] for row in cursor.fetchall()}

    def close(self):
        """Close the database connection."""
        with self._lock:
            self._conn.close()

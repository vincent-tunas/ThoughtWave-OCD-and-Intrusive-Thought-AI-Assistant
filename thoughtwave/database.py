from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .models import ChatMessage, DetectionResult, utc_now_iso


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    behavioral_state TEXT,
                    detector_score INTEGER,
                    detector_metadata TEXT,
                    provider_request_id TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_created
                    ON messages(session_id, created_at, id);
                """
            )

    def create_session(self, title: str = "New conversation") -> str:
        session_id = str(uuid.uuid4())
        now = utc_now_iso()
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
        return session_id

    def ensure_session(self, session_id: str, title: str = "New conversation") -> None:
        now = utc_now_iso()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions(id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (session_id, title, now, now),
            )

    def list_sessions(self) -> list[dict[str, str]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_messages(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        query = (
            "SELECT * FROM messages WHERE session_id = ? "
            "ORDER BY created_at ASC, id ASC"
        )
        parameters: tuple[object, ...] = (session_id,)
        if limit is not None:
            query = (
                "SELECT * FROM (SELECT * FROM messages WHERE session_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?) "
                "ORDER BY created_at ASC, id ASC"
            )
            parameters = (session_id, limit)

        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._row_to_message(row) for row in rows]

    def get_all_user_messages(self) -> list[ChatMessage]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM messages WHERE role = 'user' ORDER BY created_at ASC, id ASC"
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def save_successful_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        detection: DetectionResult,
        provider_request_id: str | None,
    ) -> tuple[int, int]:
        now = utc_now_iso()
        metadata = json.dumps(detection.to_metadata(), ensure_ascii=False)
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages(
                    session_id, role, content, created_at, behavioral_state,
                    detector_score, detector_metadata
                ) VALUES (?, 'user', ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_text,
                    now,
                    detection.state.value,
                    detection.score,
                    metadata,
                ),
            )
            user_message_id = int(cursor.lastrowid)
            cursor = connection.execute(
                """
                INSERT INTO messages(
                    session_id, role, content, created_at, behavioral_state,
                    detector_score, detector_metadata, provider_request_id
                ) VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    assistant_text,
                    now,
                    detection.state.value,
                    detection.score,
                    metadata,
                    provider_request_id,
                ),
            )
            assistant_message_id = int(cursor.lastrowid)
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
            )
            current_title = connection.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if current_title and current_title["title"] == "New conversation":
                title = " ".join(user_text.strip().split())[:52]
                connection.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?",
                    (title or "New conversation", session_id),
                )
        return user_message_id, assistant_message_id

    def delete_session(self, session_id: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> ChatMessage:
        metadata = json.loads(row["detector_metadata"]) if row["detector_metadata"] else None
        return ChatMessage(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            created_at=str(row["created_at"]),
            behavioral_state=row["behavioral_state"],
            detector_score=row["detector_score"],
            detector_metadata=metadata,
            provider_request_id=row["provider_request_id"],
        )


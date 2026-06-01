import glob
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SESSION_DIR = os.path.join(BASE_DIR, "sessions")
DEFAULT_DB_PATH = os.path.join(SESSION_DIR, "sessions.db")


class SessionManager:
    def __init__(self, max_history: int = 10, db_path: str | None = None):
        self.max_history = max_history
        self.db_path = db_path or DEFAULT_DB_PATH

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        self._maybe_migrate_json_sessions()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at);")

    def _maybe_migrate_json_sessions(self):
        json_paths = [
            p for p in glob.glob(os.path.join(SESSION_DIR, "*.json"))
            if os.path.basename(p).lower() != "sessions.json"
        ]
        if not json_paths:
            return

        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(1) AS n FROM sessions;").fetchone()
            if row and int(row["n"]) > 0:
                return

        for path in json_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                session_id = data.get("session_id")
                if not session_id:
                    continue
                title = data.get("title") or "Chat"
                created_at = data.get("created_at") or datetime.now().isoformat()
                updated_at = data.get("updated_at") or created_at
                messages = data.get("messages") or []

                with self._connect() as conn:
                    conn.execute(
                        "INSERT OR IGNORE INTO sessions(session_id, title, created_at, updated_at) VALUES(?,?,?,?);",
                        (session_id, title, created_at, updated_at),
                    )
                    for m in messages:
                        role = m.get("role")
                        content = m.get("content")
                        if not role or content is None:
                            continue
                        conn.execute(
                            "INSERT INTO messages(session_id, role, content, created_at) VALUES(?,?,?,?);",
                            (session_id, role, str(content), updated_at),
                        )
            except Exception:
                continue

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions(session_id, title, created_at, updated_at) VALUES(?,?,?,?);",
                (session_id, "New Chat", now, now),
            )
        return session_id

    def load_session(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            s = conn.execute(
                "SELECT session_id, title, created_at, updated_at FROM sessions WHERE session_id = ?;",
                (session_id,),
            ).fetchone()
            if not s:
                return None

            msgs = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC;",
                (session_id,),
            ).fetchall()

        return {
            "session_id": s["session_id"],
            "created_at": s["created_at"],
            "updated_at": s["updated_at"],
            "title": s["title"],
            "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
        }

    def _ensure_session_exists(self, session_id: str, default_title: str = "New Chat"):
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions(session_id, title, created_at, updated_at) VALUES(?,?,?,?);",
                (session_id, default_title, now, now),
            )

    def add_message(self, session_id: str, role: str, content: str):
        self._ensure_session_exists(session_id)

        now = datetime.now().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(1) AS n FROM messages WHERE session_id = ?;",
                (session_id,),
            ).fetchone()
            msg_count = int(row["n"]) if row else 0

            if role == "user" and msg_count == 0:
                title = content[:30] + ("..." if len(content) > 30 else "")
                conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?;",
                    (title, now, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?;",
                    (now, session_id),
                )

            conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES(?,?,?,?);",
                (session_id, role, content, now),
            )

    def get_history_context(self, session_id: str) -> str:
        session = self.load_session(session_id)
        if not session or not session["messages"]:
            return ""

        messages = session["messages"]
        history_msgs = messages[:-1] if messages[-1]["role"] == "user" else messages
        if not history_msgs:
            return ""

        history_msgs = history_msgs[-self.max_history:]

        context_parts = ["\n[Conversation History]:"]
        for msg in history_msgs:
            role_name = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role_name}: {msg['content']}")
        return "\n".join(context_parts) + "\n"

    def list_sessions(self) -> List[dict]:
        self.cleanup_unused_sessions()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC;",
            ).fetchall()

        return [
            {
                "session_id": r["session_id"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "title": r["title"],
            }
            for r in rows
        ]

    def cleanup_unused_sessions(self):
        """
        Delete all sessions that have 0 messages, except the most recently created one.
        This keeps the sidebar clean from accidental 'New Chat' clicks.
        """
        with self._connect() as conn:
            # Find sessions with no messages
            query = """
                SELECT s.session_id 
                FROM sessions s 
                LEFT JOIN messages m ON s.session_id = m.session_id 
                GROUP BY s.session_id 
                HAVING COUNT(m.id) = 0
                ORDER BY s.created_at DESC;
            """
            unused = conn.execute(query).fetchall()

            if len(unused) > 1:
                # Keep the first one (most recent), delete the rest
                ids_to_delete = [r["session_id"] for r in unused[1:]]
                placeholders = ",".join(["?"] * len(ids_to_delete))
                conn.execute(
                    f"DELETE FROM sessions WHERE session_id IN ({placeholders});",
                    tuple(ids_to_delete),
                )

    def delete_session(self, session_id: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?;", (session_id,))

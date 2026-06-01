import os
import tempfile

from backend.session_manager import SessionManager


def test_session_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "sessions.db")
        sm = SessionManager(max_history=3, db_path=db_path)

        sid = sm.create_session()
        sm.add_message(sid, "user", "hello")
        sm.add_message(sid, "assistant", "hi")
        sm.add_message(sid, "user", "question 2")
        sm.add_message(sid, "assistant", "answer 2")

        loaded = sm.load_session(sid)
        assert loaded["session_id"] == sid
        assert len(loaded["messages"]) == 4
        assert loaded["messages"][0]["content"] == "hello"

        sm2 = SessionManager(max_history=3, db_path=db_path)
        loaded2 = sm2.load_session(sid)
        assert len(loaded2["messages"]) == 4
        assert loaded2["messages"][-1]["content"] == "answer 2"


def test_history_context_uses_last_messages():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "sessions.db")
        sm = SessionManager(max_history=2, db_path=db_path)

        sid = sm.create_session()
        sm.add_message(sid, "user", "u1")
        sm.add_message(sid, "assistant", "a1")
        sm.add_message(sid, "user", "u2")
        sm.add_message(sid, "assistant", "a2")
        sm.add_message(sid, "user", "u3")

        ctx = sm.get_history_context(sid)
        assert "User: u2" in ctx
        assert "Assistant: a2" in ctx
        assert "User: u1" not in ctx


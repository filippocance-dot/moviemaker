from __future__ import annotations
import os, sqlite3
from contextlib import contextmanager

DATABASE_URL = os.environ.get("DATABASE_URL", "moviemaker.db")

@contextmanager
def get_conn():
    db_url = os.environ.get("DATABASE_URL", DATABASE_URL)
    conn = sqlite3.connect(db_url)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                stato TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                started_at TEXT NOT NULL DEFAULT (datetime('now')),
                ended_at TEXT,
                message_count INTEGER NOT NULL DEFAULT 0,
                token_estimate INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                profile_text TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

def create_user(nome: str, email: str, password_hash: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (nome, email, password_hash) VALUES (?, ?, ?)",
            (nome, email, password_hash)
        )

def get_user_by_email(email: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None

def get_user_by_id(user_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

def list_pending() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE stato = 'pending' ORDER BY created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]

def update_user_password(user_id: int, password_hash: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id)
        )

def approve_user(user_id: int):
    with get_conn() as conn:
        rowcount = conn.execute(
            "UPDATE users SET stato = 'approved' WHERE id = ?", (user_id,)
        ).rowcount
        if rowcount == 0:
            raise ValueError(f"User {user_id} not found")

def start_session(user_id: int) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (user_id) VALUES (?)", (user_id,)
        )
        return cur.lastrowid

def end_session(session_id: int, message_count: int, token_estimate: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = datetime('now'), message_count = ?, token_estimate = ? WHERE id = ?",
            (message_count, token_estimate, session_id)
        )

def get_session(session_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

def save_messages(user_id: int, session_id: int, messages: list[dict]):
    import json
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO messages (user_id, session_id, role, content) VALUES (?, ?, ?, ?)",
            [(user_id, session_id, m["role"],
              json.dumps(m["content"]) if isinstance(m["content"], list) else m["content"])
             for m in messages]
        )

def get_user_messages(user_id: int, limit: int = 100) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE user_id = ? ORDER BY created_at ASC LIMIT ?",
            (user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

def upsert_profile(user_id: int, profile_text: str):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO user_profiles (user_id, profile_text, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                profile_text = excluded.profile_text,
                updated_at = excluded.updated_at
        """, (user_id, profile_text))

def get_profile(user_id: int) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT profile_text FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row["profile_text"] if row else ""

def get_global_stats() -> dict:
    with get_conn() as conn:
        total_users = conn.execute("SELECT COUNT(*) FROM users WHERE stato = 'approved'").fetchone()[0]
        pending_users = conn.execute("SELECT COUNT(*) FROM users WHERE stato = 'pending'").fetchone()[0]
        active_7d = conn.execute("""
            SELECT COUNT(DISTINCT user_id) FROM sessions
            WHERE started_at >= datetime('now', '-7 days')
        """).fetchone()[0]
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions WHERE ended_at IS NOT NULL").fetchone()[0]
        total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        total_tokens = conn.execute("SELECT COALESCE(SUM(token_estimate), 0) FROM sessions").fetchone()[0]
        return {
            "total_users": total_users,
            "pending_users": pending_users,
            "active_7d": active_7d,
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "cost_estimate": round(total_tokens / 1_000_000 * 9, 4),
        }

def list_all_users_with_stats() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                u.id, u.nome, u.email, u.stato, u.created_at,
                COUNT(DISTINCT s.id) as session_count,
                COUNT(m.id) as message_count,
                MAX(s.started_at) as last_active,
                COALESCE(p.profile_text, '') as profile_text,
                COALESCE(p.updated_at, '') as profile_updated_at
            FROM users u
            LEFT JOIN sessions s ON s.user_id = u.id
            LEFT JOIN messages m ON m.user_id = u.id
            LEFT JOIN user_profiles p ON p.user_id = u.id
            WHERE u.email != ?
            GROUP BY u.id
            ORDER BY last_active DESC NULLS LAST
        """, (os.environ.get("ADMIN_EMAIL", "admin@localhost"),)).fetchall()
        return [dict(r) for r in rows]

def get_user_sessions(user_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY started_at DESC",
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]

def get_session_messages(session_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]

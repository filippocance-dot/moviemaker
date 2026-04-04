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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS project_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                stored_name TEXT NOT NULL,
                original_name TEXT NOT NULL,
                mime_type TEXT NOT NULL DEFAULT '',
                file_size INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (project_id) REFERENCES projects(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS project_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                session_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (project_id) REFERENCES projects(id),
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                UNIQUE(project_id, session_id)
            )
        """)
        # Schema migrations
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN last_active_at TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE user_profiles ADD COLUMN capability_score TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE user_profiles ADD COLUMN preferred_model TEXT DEFAULT 'sonnet'")
        except Exception:
            pass

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

def update_session_activity(session_id: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET last_active_at = datetime('now') WHERE id = ?",
            (session_id,)
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

def get_user_detailed_stats(user_id: int) -> dict:
    with get_conn() as conn:
        total_sessions = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        total_messages = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        total_tokens = conn.execute(
            "SELECT COALESCE(SUM(token_estimate), 0) FROM sessions WHERE user_id = ?",
            (user_id,)
        ).fetchone()[0]
        total_minutes_row = conn.execute("""
            SELECT COALESCE(SUM(
                (JULIANDAY(COALESCE(last_active_at, ended_at)) - JULIANDAY(started_at)) * 1440
            ), 0)
            FROM sessions
            WHERE user_id = ? AND (last_active_at IS NOT NULL OR ended_at IS NOT NULL)
        """, (user_id,)).fetchone()
        total_minutes = round(total_minutes_row[0], 1) if total_minutes_row[0] else 0.0
        avg_messages_per_session = round(total_messages / total_sessions, 1) if total_sessions > 0 else 0.0
        first_session_date = conn.execute(
            "SELECT MIN(started_at) FROM sessions WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        last_session_date = conn.execute(
            "SELECT MAX(started_at) FROM sessions WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        sessions_this_week = conn.execute("""
            SELECT COUNT(*) FROM sessions
            WHERE user_id = ? AND started_at >= datetime('now', '-7 days')
        """, (user_id,)).fetchone()[0]
        messages_this_week = conn.execute("""
            SELECT COUNT(*) FROM messages
            WHERE user_id = ? AND created_at >= datetime('now', '-7 days')
        """, (user_id,)).fetchone()[0]
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "total_minutes": total_minutes,
            "avg_messages_per_session": avg_messages_per_session,
            "first_session_date": first_session_date,
            "last_session_date": last_session_date,
            "sessions_this_week": sessions_this_week,
            "messages_this_week": messages_this_week,
        }

def get_admin_full_stats() -> dict:
    with get_conn() as conn:
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        approved_users = conn.execute(
            "SELECT COUNT(*) FROM users WHERE stato = 'approved'"
        ).fetchone()[0]
        active_today = conn.execute("""
            SELECT COUNT(DISTINCT user_id) FROM sessions
            WHERE DATE(started_at) = DATE('now')
        """).fetchone()[0]
        active_this_week = conn.execute("""
            SELECT COUNT(DISTINCT user_id) FROM sessions
            WHERE started_at >= datetime('now', '-7 days')
        """).fetchone()[0]
        total_sessions = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE ended_at IS NOT NULL"
        ).fetchone()[0]
        total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        total_tokens = conn.execute(
            "SELECT COALESCE(SUM(token_estimate), 0) FROM sessions"
        ).fetchone()[0]
        avg_row = conn.execute("""
            SELECT AVG(
                (JULIANDAY(COALESCE(last_active_at, ended_at)) - JULIANDAY(started_at)) * 1440
            )
            FROM sessions
            WHERE ended_at IS NOT NULL AND (last_active_at IS NOT NULL OR ended_at IS NOT NULL)
        """).fetchone()
        avg_session_minutes = round(avg_row[0], 1) if avg_row[0] else 0.0
        most_active_users = conn.execute("""
            SELECT u.id, u.nome, u.email,
                   COUNT(m.id) as message_count,
                   COUNT(DISTINCT s.id) as session_count,
                   MAX(s.started_at) as last_active
            FROM users u
            LEFT JOIN messages m ON m.user_id = u.id
            LEFT JOIN sessions s ON s.user_id = u.id
            GROUP BY u.id
            ORDER BY message_count DESC
            LIMIT 5
        """).fetchall()
        recent_sessions = conn.execute("""
            SELECT s.id, s.user_id, s.started_at, s.ended_at,
                   s.message_count, s.token_estimate, s.last_active_at,
                   u.nome as user_name, u.email as user_email
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            ORDER BY s.started_at DESC
            LIMIT 10
        """).fetchall()
        return {
            "total_users": total_users,
            "approved_users": approved_users,
            "active_today": active_today,
            "active_this_week": active_this_week,
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "avg_session_minutes": avg_session_minutes,
            "most_active_users": [dict(r) for r in most_active_users],
            "recent_sessions": [dict(r) for r in recent_sessions],
        }

def upsert_profile(user_id: int, profile_text: str, capability_score: str | None = None):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO user_profiles (user_id, profile_text, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                profile_text = excluded.profile_text,
                updated_at = excluded.updated_at
        """, (user_id, profile_text))
        if capability_score is not None:
            conn.execute(
                "UPDATE user_profiles SET capability_score = ? WHERE user_id = ?",
                (capability_score, user_id)
            )

def get_profile(user_id: int) -> dict | None:
    """Restituisce il profilo come dict. None se non esiste."""
    import json
    with get_conn() as conn:
        row = conn.execute(
            "SELECT profile_text FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row or not row["profile_text"]:
            return None
        try:
            return json.loads(row["profile_text"])
        except (json.JSONDecodeError, TypeError):
            # Profilo legacy in formato testo: wrappalo
            return {"legacy": row["profile_text"]}

def get_profile_full(user_id: int) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT profile_text, capability_score FROM user_profiles WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        if row:
            return {"profile_text": row["profile_text"] or "", "capability_score": row["capability_score"]}
        return {"profile_text": "", "capability_score": None}

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

# ── PROJECTS ──────────────────────────────────────────────────────────────────

def create_project(user_id: int, name: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO projects (user_id, name) VALUES (?, ?)", (user_id, name)
        )
        return cur.lastrowid

def get_project(project_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return dict(row) if row else None

def list_projects(user_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM projects WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]

def delete_project(project_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM project_sessions WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM project_files WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

def update_project_note(project_id: int, note: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE projects SET note = ?, updated_at = datetime('now') WHERE id = ?",
            (note, project_id)
        )

def add_project_file(project_id: int, user_id: int, stored_name: str,
                     original_name: str, mime_type: str, file_size: int) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO project_files
               (project_id, user_id, stored_name, original_name, mime_type, file_size)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (project_id, user_id, stored_name, original_name, mime_type, file_size)
        )
        conn.execute(
            "UPDATE projects SET updated_at = datetime('now') WHERE id = ?", (project_id,)
        )
        return cur.lastrowid

def list_project_files(project_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM project_files WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,)
        ).fetchall()
        return [dict(r) for r in rows]

def get_project_file(file_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM project_files WHERE id = ?", (file_id,)).fetchone()
        return dict(row) if row else None

def delete_project_file(file_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM project_files WHERE id = ?", (file_id,))

def link_session_to_project(project_id: int, session_id: int):
    with get_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO project_sessions (project_id, session_id) VALUES (?, ?)",
                (project_id, session_id)
            )
            conn.execute(
                "UPDATE projects SET updated_at = datetime('now') WHERE id = ?", (project_id,)
            )
        except Exception:
            pass  # already linked

def unlink_session_from_project(project_id: int, session_id: int):
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM project_sessions WHERE project_id = ? AND session_id = ?",
            (project_id, session_id)
        )

def list_project_sessions(project_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT s.*, ps.id as link_id
            FROM sessions s
            JOIN project_sessions ps ON ps.session_id = s.id
            WHERE ps.project_id = ?
            ORDER BY s.started_at DESC
        """, (project_id,)).fetchall()
        return [dict(r) for r in rows]

def set_preferred_model(user_id: int, model: str):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO user_profiles (user_id, profile_text, preferred_model)
            VALUES (?, '', ?)
            ON CONFLICT(user_id) DO UPDATE SET preferred_model = excluded.preferred_model
        """, (user_id, model))

def get_preferred_model(user_id: int) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT preferred_model FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row and row["preferred_model"]:
            return row["preferred_model"]
        return "sonnet"

def get_session_messages(session_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]

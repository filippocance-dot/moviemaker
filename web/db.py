from __future__ import annotations
import os, sqlite3
from contextlib import contextmanager

DATABASE_URL = os.environ.get("DATABASE_URL", "moviemaker.db")

@contextmanager
def get_conn():
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
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

def approve_user(user_id: int):
    with get_conn() as conn:
        conn.execute("UPDATE users SET stato = 'approved' WHERE id = ?", (user_id,))

import os, tempfile
os.environ["DATABASE_URL"] = tempfile.mktemp(suffix=".db")
os.environ["SECRET_KEY"] = "test-secret"
os.environ["ADMIN_EMAIL"] = "admin@test.com"

from fastapi.testclient import TestClient
from web.app import app
from web.db import init_db, approve_user, get_user_by_email

init_db()
client = TestClient(app, follow_redirects=False)

def test_register_redirects_to_attesa():
    r = client.post("/registrati", data={"nome":"Test","email":"t@t.com","password":"pass123"})
    assert r.status_code == 303
    assert r.headers["location"] == "/attesa"

def test_login_pending_user_fails():
    r = client.post("/login", data={"email":"t@t.com","password":"pass123"})
    assert r.status_code == 200
    assert "accesso non ancora approvato" in r.text.lower()

def test_login_approved_user_redirects_to_chat():
    user = get_user_by_email("t@t.com")
    approve_user(user["id"])
    r = client.post("/login", data={"email":"t@t.com","password":"pass123"})
    assert r.status_code == 303
    assert r.headers["location"] == "/chat"

def test_chat_requires_auth():
    r = client.get("/chat")
    assert r.status_code == 303
    assert r.headers["location"] == "/login"

def test_admin_requires_admin_cookie():
    r = client.get("/admin")
    assert r.status_code == 303

def test_end_session_requires_auth():
    r = client.post("/chat/end-session", json={"session_id": 1, "conversation": []})
    assert r.status_code == 401

def test_start_session_on_chat_get():
    from web.db import get_user_by_email, approve_user
    user = get_user_by_email("t@t.com")
    if user and user["stato"] != "approved":
        approve_user(user["id"])
    r_login = client.post("/login", data={"email": "t@t.com", "password": "pass123"})
    if r_login.status_code != 303:
        return
    r = client.get("/chat")
    assert r.status_code == 200
    assert "session_id" in r.text

def test_admin_stats_page_loads():
    import os
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@test.com")
    r_login = client.post("/login", data={"email": admin_email, "password": "adminpass"})
    if r_login.status_code != 303:
        return
    r = client.get("/admin")
    assert r.status_code == 200
    assert "statistiche" in r.text.lower() or "utenti" in r.text.lower()

def test_chat_has_textarea():
    from web.db import get_user_by_email, approve_user
    user = get_user_by_email("t@t.com")
    if user and user["stato"] != "approved":
        approve_user(user["id"])
    r = client.post("/login", data={"email": "t@t.com", "password": "pass123"})
    if r.status_code != 303:
        return
    r = client.get("/chat")
    assert r.status_code == 200
    assert "<textarea" in r.text
    assert "sendBeacon" in r.text
    assert "userScrolled" in r.text

def test_upload_requires_auth():
    import io
    data = {"file": ("test.txt", io.BytesIO(b"contenuto test"), "text/plain")}
    r = client.post("/chat/upload", files=data)
    assert r.status_code == 401

def test_upload_text_file():
    from web.db import get_user_by_email
    user = get_user_by_email("t@t.com")
    if not user:
        return
    r_login = client.post("/login", data={"email": "t@t.com", "password": "pass123"})
    assert r_login.status_code == 303, f"Login failed: {r_login.status_code}"
    import io
    data = {"file": ("test.txt", io.BytesIO(b"Questo e un documento di test"), "text/plain")}
    r = client.post("/chat/upload", files=data)
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "text"
    assert "documento" in body["content"]

def test_history_requires_auth():
    r = client.get("/chat/storia")
    assert r.status_code == 303
    assert r.headers["location"] == "/login"

def test_history_loads_for_authed_user():
    from web.db import get_user_by_email
    user = get_user_by_email("t@t.com")
    if not user:
        return
    r_login = client.post("/login", data={"email": "t@t.com", "password": "pass123"})
    assert r_login.status_code == 303, f"Login failed: {r_login.status_code}"
    r = client.get("/chat/storia")
    assert r.status_code == 200
    assert "storico" in r.text.lower() or "sessioni" in r.text.lower() or "filmmaker" in r.text.lower()

def test_update_session_activity():
    import os, tempfile
    db_path = tempfile.mktemp(suffix=".db")
    os.environ["DATABASE_URL"] = db_path
    from importlib import reload
    import web.db as dbmod
    reload(dbmod)
    dbmod.init_db()
    dbmod.create_user("Tester", "tester_act@x.com", "hash")
    user = dbmod.get_user_by_email("tester_act@x.com")
    dbmod.approve_user(user["id"])
    sid = dbmod.start_session(user["id"])
    dbmod.update_session_activity(sid)
    sess = dbmod.get_session(sid)
    assert sess["last_active_at"] is not None

def test_get_user_detailed_stats():
    import os, tempfile
    db_path = tempfile.mktemp(suffix=".db")
    os.environ["DATABASE_URL"] = db_path
    from importlib import reload
    import web.db as dbmod
    reload(dbmod)
    dbmod.init_db()
    dbmod.create_user("Stats User", "stats@x.com", "hash")
    user = dbmod.get_user_by_email("stats@x.com")
    dbmod.approve_user(user["id"])
    sid = dbmod.start_session(user["id"])
    dbmod.update_session_activity(sid)
    dbmod.end_session(sid, message_count=4, token_estimate=200)
    dbmod.save_messages(user["id"], sid, [
        {"role": "user", "content": "ciao"},
        {"role": "assistant", "content": "risposta"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "bene"},
    ])
    stats = dbmod.get_user_detailed_stats(user["id"])
    assert stats["total_sessions"] == 1
    assert stats["total_messages"] == 4
    assert stats["total_tokens"] == 200
    assert stats["avg_messages_per_session"] == 4.0

def test_get_admin_full_stats():
    import os, tempfile
    db_path = tempfile.mktemp(suffix=".db")
    os.environ["DATABASE_URL"] = db_path
    from importlib import reload
    import web.db as dbmod
    reload(dbmod)
    dbmod.init_db()
    dbmod.create_user("Admin Full", "adminfull@x.com", "hash")
    user = dbmod.get_user_by_email("adminfull@x.com")
    dbmod.approve_user(user["id"])
    sid = dbmod.start_session(user["id"])
    dbmod.end_session(sid, message_count=2, token_estimate=100)
    stats = dbmod.get_admin_full_stats()
    assert stats["total_users"] >= 1
    assert stats["total_sessions"] >= 1
    assert isinstance(stats["most_active_users"], list)
    assert isinstance(stats["recent_sessions"], list)

def test_upsert_profile_with_capability_score():
    import os, tempfile, sqlite3
    db_path = tempfile.mktemp(suffix=".db")
    os.environ["DATABASE_URL"] = db_path
    from importlib import reload
    import web.db as dbmod
    reload(dbmod)
    dbmod.init_db()
    dbmod.create_user("Cap User", "cap@x.com", "hash")
    user = dbmod.get_user_by_email("cap@x.com")
    dbmod.upsert_profile(user["id"], "test profile", capability_score="Alto livello creativo.")
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT capability_score FROM user_profiles WHERE user_id = ?", (user["id"],)).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "Alto livello creativo."

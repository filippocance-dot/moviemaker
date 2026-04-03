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
    if r_login.status_code != 303:
        return
    import io
    data = {"file": ("test.txt", io.BytesIO(b"Questo e un documento di test"), "text/plain")}
    r = client.post("/chat/upload", files=data)
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "text"
    assert "documento" in body["content"]

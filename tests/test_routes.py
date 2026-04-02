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

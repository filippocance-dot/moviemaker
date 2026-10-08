import os, tempfile
os.environ.setdefault("DATABASE_URL", tempfile.mktemp(suffix=".db"))
os.environ.setdefault("SECRET_KEY", "test-secret")

import pytest
from fastapi.testclient import TestClient
from web.app import app
from web.auth import hash_password
from web.db import (init_db, create_user, approve_user, get_user_by_email, start_session,
                    list_projects)
from web import demo

init_db()


@pytest.fixture
def demo_on(monkeypatch):
    monkeypatch.setenv("DEMO_ENABLED", "true")


def demo_client():
    c = TestClient(app, follow_redirects=False)
    r = c.get("/demo")
    assert r.status_code == 303
    return c


def test_demo_disabled_returns_404(monkeypatch):
    monkeypatch.delenv("DEMO_ENABLED", raising=False)
    c = TestClient(app, follow_redirects=False)
    assert c.get("/demo").status_code == 404


def test_demo_logs_in_and_shows_seeded_project(demo_on):
    c = demo_client()
    r = c.get("/progetti")
    assert r.status_code == 200
    assert "La casa al mare" in r.text
    assert "MODALITÀ DEMO" in r.text


def test_demo_seed_is_idempotent(demo_on):
    demo_client()
    demo_client()
    user = get_user_by_email(demo.DEMO_EMAIL)
    assert len(list_projects(user["id"])) == 1


@pytest.mark.parametrize("path", [
    "/progetti",
    "/progetti/1/elimina",
    "/progetti/1/sessione/1/rimuovi",
    "/progetti/1/file/1/elimina",
])
def test_demo_cannot_modify_or_delete(demo_on, path):
    c = demo_client()
    data = {"name": "x"} if path == "/progetti" else None
    assert c.post(path, data=data).status_code == 403


def test_demo_cannot_change_settings_or_notes(demo_on):
    c = demo_client()
    assert c.post("/chat/set-model", json={"model": "opus"}).status_code == 403
    pid = list_projects(get_user_by_email(demo.DEMO_EMAIL)["id"])[0]["id"]
    assert c.post(f"/progetti/{pid}/nota", json={"note": "x"}).status_code == 403


def test_demo_daily_ai_limit(demo_on, monkeypatch):
    monkeypatch.setenv("DEMO_AI_DAILY_LIMIT", "0")
    c = demo_client()
    r = c.post("/chat/stream", json={"conversation": [{"role": "user", "content": "ciao"}]})
    assert r.status_code == 200
    assert "limite giornaliero" in r.text


def test_demo_end_session_saves_nothing(demo_on):
    c = demo_client()
    user = get_user_by_email(demo.DEMO_EMAIL)
    sid = start_session(user["id"])
    r = c.post("/chat/end-session",
               json={"session_id": sid, "conversation": [{"role": "user", "content": "ciao"}]})
    assert r.status_code == 200
    from web.db import get_session_messages
    assert get_session_messages(sid) == []


def test_demo_cookie_invalid_after_disable(demo_on, monkeypatch):
    c = demo_client()
    monkeypatch.delenv("DEMO_ENABLED")
    r = c.get("/chat")
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_cannot_register_demo_email():
    c = TestClient(app, follow_redirects=False)
    r = c.post("/registrati", data={"nome": "x", "email": demo.DEMO_EMAIL, "password": "p"})
    assert r.status_code == 200
    assert "già registrata" in r.text


def test_cannot_link_other_users_session():
    create_user("A", "owner-a@test.com", hash_password("pw"))
    create_user("B", "intruder-b@test.com", hash_password("pw"))
    a, b = get_user_by_email("owner-a@test.com"), get_user_by_email("intruder-b@test.com")
    approve_user(a["id"]); approve_user(b["id"])
    victim_session = start_session(a["id"])

    c = TestClient(app, follow_redirects=False)
    c.post("/login", data={"email": "intruder-b@test.com", "password": "pw"})
    r = c.post("/progetti", data={"name": "furto"})
    pid = int(r.headers["location"].rsplit("/", 1)[1])
    r = c.post(f"/progetti/{pid}/sessione", data={"session_id": victim_session})
    assert r.status_code == 403


def test_seed_sessions_added_once_even_to_existing_project(demo_on, monkeypatch, tmp_path):
    import json
    from web.db import list_project_sessions
    demo_client()  # progetto già esistente, senza conversazioni
    seed = json.loads(demo.SEED_PATH.read_text(encoding="utf-8"))
    seed["sessions"] = [[{"role": "user", "content": "scena"}, {"role": "assistant", "content": "analisi"}]]
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(seed), encoding="utf-8")
    monkeypatch.setattr(demo, "SEED_PATH", path)
    demo_client()
    demo_client()
    pid = list_projects(get_user_by_email(demo.DEMO_EMAIL)["id"])[0]["id"]
    assert len(list_project_sessions(pid)) == 1

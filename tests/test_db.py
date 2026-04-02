import os, tempfile, pytest
os.environ.setdefault("DATABASE_URL", "")

def make_db():
    import importlib, sys
    tmp = tempfile.mktemp(suffix=".db")
    os.environ["DATABASE_URL"] = tmp
    if "web.db" in sys.modules:
        del sys.modules["web.db"]
    from web.db import init_db, create_user, get_user_by_email, list_pending, approve_user
    init_db()
    return tmp, create_user, get_user_by_email, list_pending, approve_user

def test_create_and_get_user():
    _, create_user, get_user_by_email, _, _ = make_db()
    create_user("Mario", "mario@test.com", "hash123")
    user = get_user_by_email("mario@test.com")
    assert user["nome"] == "Mario"
    assert user["stato"] == "pending"

def test_approve_user():
    _, create_user, get_user_by_email, list_pending, approve_user = make_db()
    create_user("Lucia", "lucia@test.com", "hash456")
    user = get_user_by_email("lucia@test.com")
    approve_user(user["id"])
    updated = get_user_by_email("lucia@test.com")
    assert updated["stato"] == "approved"

def test_list_pending():
    _, create_user, get_user_by_email, list_pending, approve_user = make_db()
    create_user("A", "a@test.com", "h1")
    create_user("B", "b@test.com", "h2")
    pending = list_pending()
    assert len(pending) == 2

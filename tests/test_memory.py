import os, tempfile
os.environ["DATABASE_URL"] = tempfile.mktemp(suffix=".db")

from web.db import (
    init_db, create_user,
    start_session, end_session, get_session,
    save_messages, get_user_messages,
    upsert_profile, get_profile,
)

def setup_module():
    init_db()
    create_user("Test", "test@t.com", "hash")

def test_start_and_end_session():
    from web.db import get_user_by_email
    user = get_user_by_email("test@t.com")
    sid = start_session(user["id"])
    assert isinstance(sid, int)
    end_session(sid, message_count=4, token_estimate=200)
    s = get_session(sid)
    assert s["message_count"] == 4
    assert s["ended_at"] is not None

def test_save_and_get_messages():
    from web.db import get_user_by_email
    user = get_user_by_email("test@t.com")
    sid = start_session(user["id"])
    msgs = [
        {"role": "user", "content": "Ciao"},
        {"role": "assistant", "content": "Ciao, come posso aiutarti?"},
    ]
    save_messages(user["id"], sid, msgs)
    # Filtra per session_id per isolare il test
    all_msgs = get_user_messages(user["id"], limit=100)
    session_msgs = [m for m in all_msgs if m["session_id"] == sid]
    assert len(session_msgs) == 2
    assert session_msgs[0]["content"] == "Ciao"

def test_upsert_and_get_profile():
    from web.db import get_user_by_email
    user = get_user_by_email("test@t.com")
    upsert_profile(user["id"], "TEMI: cinema muto")
    p = get_profile(user["id"])
    assert p == "TEMI: cinema muto"
    upsert_profile(user["id"], "TEMI: cinema muto\nPROGRESSI: migliorato")
    p2 = get_profile(user["id"])
    assert "PROGRESSI" in p2

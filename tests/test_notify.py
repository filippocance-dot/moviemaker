import os, tempfile, time
os.environ.setdefault("DATABASE_URL", tempfile.mktemp(suffix=".db"))
os.environ.setdefault("SECRET_KEY", "test-secret")

import pytest
from web import demo, notify


@pytest.fixture
def sent(monkeypatch):
    """Cattura le notifiche al posto dell'invio reale (sincrono, nessuna rete)."""
    calls = []
    monkeypatch.setenv("NTFY_TOPIC", "test-topic")
    monkeypatch.setattr(notify, "send", lambda title, message, priority="default": calls.append((title, message)))
    return calls


@pytest.fixture
def fresh_quota(monkeypatch):
    monkeypatch.setitem(demo._quota, "day", None)
    monkeypatch.setenv("DEMO_AI_DAILY_LIMIT", "30")


def test_demo_quota_notifies_at_20_and_30_once(sent, fresh_quota):
    for _ in range(30):
        assert demo.take_ai_quota()
    assert not demo.take_ai_quota()  # oltre il limite: rifiutata, nessuna nuova notifica
    assert [m for _, m in sent] == ["La demo ha usato 20/30 richieste AI oggi.",
                                    "La demo ha usato 30/30 richieste AI oggi."]


def test_credit_alert_at_most_once_per_hour(sent, monkeypatch):
    monkeypatch.setattr(notify, "_last_credit_alert", None)
    from web.app import friendly_ai_error
    err = Exception("Your credit balance is too low to access the Anthropic API.")
    friendly_ai_error(err)
    friendly_ai_error(err)
    assert len(sent) == 1
    monkeypatch.setattr(notify, "_last_credit_alert", time.monotonic() - notify.CREDIT_ALERT_INTERVAL - 1)
    friendly_ai_error(err)
    assert len(sent) == 2


def test_other_ai_errors_do_not_notify(sent, monkeypatch):
    monkeypatch.setattr(notify, "_last_credit_alert", None)
    from web.app import friendly_ai_error
    friendly_ai_error(Exception("overloaded"))
    assert sent == []


def test_startup_notifies_when_demo_disabled(sent, monkeypatch):
    from fastapi.testclient import TestClient
    from web.app import app
    monkeypatch.delenv("DEMO_ENABLED", raising=False)
    with TestClient(app):
        pass
    assert any("demo spenta" in t for t, _ in sent)


def test_startup_silent_when_demo_enabled(sent, monkeypatch):
    from fastapi.testclient import TestClient
    from web.app import app
    monkeypatch.setenv("DEMO_ENABLED", "true")
    with TestClient(app):
        pass
    assert not any("demo spenta" in t for t, _ in sent)


def test_send_is_noop_without_topic(monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    monkeypatch.setattr(notify, "_post", lambda *a: pytest.fail("non deve inviare"))
    notify.send("t", "m")


def test_ntfy_down_does_not_raise(monkeypatch, capsys):
    def boom(*a):
        raise OSError("ntfy irraggiungibile")
    monkeypatch.setattr(notify, "_post", boom)
    notify._safe_post("topic", "t", "m", "default")  # nessuna eccezione propagata
    assert "invio fallito" in capsys.readouterr().out

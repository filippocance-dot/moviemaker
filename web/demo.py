"""Accesso demo senza registrazione (/demo).

Un unico account demo condiviso, attivo solo se DEMO_ENABLED=true:
- sola lettura su progetti e impostazioni (niente creazione, modifica, cancellazione);
- chat con Suso funzionante ma effimera (le conversazioni non vengono salvate);
- solo modello Sonnet, limite globale di richieste AI al giorno.
Per disattivarlo basta togliere DEMO_ENABLED: /demo risponde 404 e i cookie demo smettono di valere.
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import secrets
import threading

from web.auth import hash_password
from web.db import (approve_user, create_project, create_user, end_session, get_project_by_name,
                    get_user_by_email, link_session_to_project, save_messages, start_session,
                    update_project_note)

DEMO_EMAIL = "demo@filmmaker.demo"
SEED_PATH = pathlib.Path(__file__).parent / "demo_seed.json"
DEMO_MAX_CHARS = 60_000  # tetto alla lunghezza della conversazione inviata in demo


def demo_enabled() -> bool:
    return os.environ.get("DEMO_ENABLED") == "true"


def is_demo_user(user: dict | None) -> bool:
    return bool(user) and user["email"] == DEMO_EMAIL


def _daily_limit() -> int:
    try:
        return int(os.environ.get("DEMO_AI_DAILY_LIMIT", "30"))
    except ValueError:
        return 30


# Contatore in memoria: si azzera a ogni cambio di giorno (e a ogni riavvio del server)
_quota_lock = threading.Lock()
_quota = {"day": None, "count": 0}


def take_ai_quota() -> bool:
    """Consuma una richiesta AI demo. False se il limite giornaliero è raggiunto."""
    today = datetime.date.today().isoformat()
    with _quota_lock:
        if _quota["day"] != today:
            _quota["day"], _quota["count"] = today, 0
        if _quota["count"] >= _daily_limit():
            return False
        _quota["count"] += 1
        return True


def conversation_too_long(conversation: list) -> bool:
    return len(json.dumps(conversation, ensure_ascii=False)) > DEMO_MAX_CHARS


def ensure_demo_account() -> dict:
    """Crea (se manca) l'utente demo e il progetto d'esempio. Idempotente."""
    user = get_user_by_email(DEMO_EMAIL)
    if not user:
        # Password casuale mai comunicata: all'account si entra solo da /demo
        create_user("Demo", DEMO_EMAIL, hash_password(secrets.token_urlsafe(24)))
        user = get_user_by_email(DEMO_EMAIL)
    if user["stato"] != "approved":
        approve_user(user["id"])
        user = get_user_by_email(DEMO_EMAIL)

    seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    project = get_project_by_name(user["id"], seed["project_name"])
    if not project:
        project_id = create_project(user["id"], seed["project_name"])
        for conv in seed.get("sessions", []):
            session_id = start_session(user["id"])
            save_messages(user["id"], session_id, conv)
            end_session(session_id, message_count=len(conv),
                        token_estimate=sum(len(m["content"]) for m in conv) // 4)
            link_session_to_project(project_id, session_id)
    else:
        project_id = project["id"]
    update_project_note(project_id, seed["note"])
    return user

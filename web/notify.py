"""Notifiche push all'amministratore via ntfy.sh (topic in NTFY_TOPIC).

Regole: i messaggi sono testi fissi (mai chiavi, dati utenti o conversazioni);
l'invio avviene in un thread separato e ogni errore viene ignorato, così se ntfy
non risponde l'app continua a funzionare normalmente.
"""
from __future__ import annotations

import os
import threading
import time
import urllib.request

NTFY_SERVER = "https://ntfy.sh"
CREDIT_ALERT_INTERVAL = 3600  # secondi: al massimo una notifica di credito all'ora

_lock = threading.Lock()
_last_credit_alert = None  # time.monotonic() dell'ultimo avviso credito


def _post(topic: str, title: str, message: str, priority: str) -> None:
    req = urllib.request.Request(
        f"{NTFY_SERVER}/{topic}",
        data=message.encode("utf-8"),
        headers={"Title": title, "Priority": priority},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=5).close()


def _safe_post(topic: str, title: str, message: str, priority: str) -> None:
    try:
        _post(topic, title, message, priority)
    except Exception as e:
        print(f"[notify] invio fallito (ignorato): {type(e).__name__}")


def send(title: str, message: str, priority: str = "default") -> None:
    """Invia una notifica senza bloccare la richiesta. No-op se NTFY_TOPIC non è impostato."""
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return
    threading.Thread(target=_safe_post, args=(topic, title, message, priority), daemon=True).start()


def credit_exhausted() -> None:
    """Avviso credito Anthropic esaurito, al massimo uno ogni CREDIT_ALERT_INTERVAL secondi."""
    global _last_credit_alert
    now = time.monotonic()
    with _lock:
        if _last_credit_alert is not None and now - _last_credit_alert < CREDIT_ALERT_INTERVAL:
            return
        _last_credit_alert = now
    send("FilmMaker: credito Anthropic esaurito",
         "Una chiamata all'API e' fallita per credito insufficiente. "
         "Ricarica da console.anthropic.com > Plans & Billing.", priority="high")

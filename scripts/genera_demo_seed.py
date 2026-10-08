"""Genera una conversazione reale con Suso sul progetto demo e la salva in web/demo_seed.json.

La conversazione passa dall'app vera (stesso prompt, stesso RAG, stesso endpoint /chat/stream),
su un database temporaneo. Nessun testo dell'AI è scritto a mano.

Uso (dalla cartella del progetto, con una chiave Anthropic con credito):
    ANTHROPIC_API_KEY=... /opt/homebrew/bin/python3.9 scripts/genera_demo_seed.py
"""
import json
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
_tmp = pathlib.Path(tempfile.mkdtemp())
# Il RAG cerca le sue cache accanto al DB: riusa quelle del progetto (evita ~470 MB di download)
for _name in ("embeddings.pkl", "hf_cache"):
    if (ROOT / _name).exists():
        (_tmp / _name).symlink_to(ROOT / _name)
os.environ["DATABASE_URL"] = str(_tmp / "seed.db")
os.environ.setdefault("SECRET_KEY", "demo-seed")
os.environ.pop("DEMO_ENABLED", None)  # utente normale: niente limiti demo
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from web.app import app  # noqa: E402
from web.auth import hash_password  # noqa: E402
from web.db import approve_user, create_user, get_user_by_email  # noqa: E402

SEED_PATH = ROOT / "web" / "demo_seed.json"


def ask(client, conversation):
    r = client.post("/chat/stream", json={"conversation": conversation})
    r.raise_for_status()
    parts = []
    for line in r.text.split("\n\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            parts.append(line[6:].replace("\\n", "\n"))
    text = "".join(parts).strip()
    if not text or text.startswith("⚠️"):
        raise SystemExit(f"Risposta AI non valida, seed non aggiornato: {text[:200]}")
    return text


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Manca ANTHROPIC_API_KEY")
    seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    scena = seed["note"].split("SCENA 3 (bozza grezza)", 1)[1]
    turni = [
        "Ti incollo una scena grezza del mio corto. Fammi un'analisi drammaturgica: "
        "cosa funziona, cosa non funziona, e qual è la decisione da prendere.\n\nSCENA 3"
        + scena,
        "Va bene, accetto la tua proposta. Fissiamo la decisione e scrivimi la scena riscritta.",
    ]
    with TestClient(app) as client:  # avvia il lifespan: corpus e RAG caricati come in produzione
        create_user("Seed", "seed@local", hash_password("seed"))
        approve_user(get_user_by_email("seed@local")["id"])
        client.post("/login", data={"email": "seed@local", "password": "seed"})
        conversation = []
        for turno in turni:
            conversation.append({"role": "user", "content": turno})
            conversation.append({"role": "assistant", "content": ask(client, conversation)})
            print(f"--- turno {len(conversation) // 2} ok ({len(conversation[-1]['content'])} caratteri)")
    seed["sessions"] = [conversation]
    SEED_PATH.write_text(json.dumps(seed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Seed aggiornato: {SEED_PATH}")


if __name__ == "__main__":
    main()

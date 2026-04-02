# MovieMaker Web Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Costruire una web app FastAPI che porta CineAuteur nel browser con autenticazione, approvazione utenti e chat streaming.

**Architecture:** FastAPI server-rendered con Jinja2 templates. SQLite per gli utenti. La logica RAG e OpenRouter viene estratta da `cineauteur.py` in un modulo condiviso `web/rag.py`. Ogni sessione utente mantiene la conversazione in memoria sul server (dict keyed by session token).

**Tech Stack:** FastAPI, Uvicorn, Jinja2, SQLite (sqlite3 built-in), passlib[bcrypt], itsdangerous, smtplib, rank-bm25, pypdf, openai, python-dotenv

---

## File Structure

```
web/
  app.py          — FastAPI app, routes principali, startup
  db.py           — SQLite schema, CRUD utenti
  auth.py         — password hashing, session token, decoratori di protezione
  email_utils.py  — invio email approvazione
  rag.py          — estrazione logica RAG da cineauteur.py (corpus, BM25, retrieve)
  chat.py         — route /chat, SSE streaming, gestione conversazioni in memoria
  admin.py        — route /admin, approvazione utenti
  templates/
    base.html     — layout condiviso, CSS notte calda inline
    register.html — form registrazione
    attesa.html   — schermata attesa approvazione
    login.html    — form login
    chat.html     — interfaccia chat + JS SSE client
    admin.html    — pannello admin
tests/
  test_db.py      — test CRUD database
  test_auth.py    — test hashing, session token
  test_routes.py  — test endpoints con TestClient
requirements-web.txt  — dipendenze aggiuntive per il web
Procfile          — per Railway: web: uvicorn web.app:app --host 0.0.0.0 --port $PORT
```

---

## Task 1: Setup e dipendenze

**Files:**
- Create: `requirements-web.txt`
- Create: `web/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Crea requirements-web.txt**

```
fastapi>=0.110.0
uvicorn>=0.29.0
jinja2>=3.1.0
python-multipart>=0.0.9
passlib[bcrypt]>=1.7.4
itsdangerous>=2.1.2
```

- [ ] **Step 2: Installa dipendenze**

```bash
cd ~/projects/Test_1
pip3 install -r requirements-web.txt
```

Expected output: `Successfully installed fastapi uvicorn jinja2 ...`

- [ ] **Step 3: Crea directory e file vuoti**

```bash
mkdir -p web/templates tests
touch web/__init__.py tests/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git init  # se non già inizializzato
git add requirements-web.txt web/__init__.py tests/__init__.py
git commit -m "feat: web app scaffold"
```

---

## Task 2: Database

**Files:**
- Create: `web/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Scrivi il test**

`tests/test_db.py`:
```python
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
```

- [ ] **Step 2: Esegui il test — deve fallire**

```bash
cd ~/projects/Test_1
python3 -m pytest tests/test_db.py -v
```

Expected: `ModuleNotFoundError: No module named 'web.db'`

- [ ] **Step 3: Implementa web/db.py**

```python
from __future__ import annotations
import os, sqlite3
from contextlib import contextmanager

DATABASE_URL = os.environ.get("DATABASE_URL", "moviemaker.db")

@contextmanager
def get_conn():
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                stato TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

def create_user(nome: str, email: str, password_hash: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (nome, email, password_hash) VALUES (?, ?, ?)",
            (nome, email, password_hash)
        )

def get_user_by_email(email: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None

def get_user_by_id(user_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

def list_pending() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE stato = 'pending' ORDER BY created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]

def approve_user(user_id: int):
    with get_conn() as conn:
        conn.execute("UPDATE users SET stato = 'approved' WHERE id = ?", (user_id,))
```

- [ ] **Step 4: Esegui il test — deve passare**

```bash
python3 -m pytest tests/test_db.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add web/db.py tests/test_db.py
git commit -m "feat: SQLite database layer"
```

---

## Task 3: Auth (password hashing + session token)

**Files:**
- Create: `web/auth.py`
- Create: `tests/test_auth.py`

- [ ] **Step 1: Scrivi il test**

`tests/test_auth.py`:
```python
from web.auth import hash_password, verify_password, make_token, decode_token

def test_hash_and_verify():
    h = hash_password("mysecret")
    assert verify_password("mysecret", h) is True
    assert verify_password("wrong", h) is False

def test_token_roundtrip():
    token = make_token(42)
    user_id = decode_token(token)
    assert user_id == 42

def test_token_invalid():
    assert decode_token("notavalidtoken") is None
```

- [ ] **Step 2: Esegui il test — deve fallire**

```bash
python3 -m pytest tests/test_auth.py -v
```

Expected: `ModuleNotFoundError: No module named 'web.auth'`

- [ ] **Step 3: Implementa web/auth.py**

```python
from __future__ import annotations
import os
from passlib.context import CryptContext
from itsdangerous import URLSafeSerializer, BadSignature

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
_signer = URLSafeSerializer(SECRET_KEY, salt="session")

def hash_password(plain: str) -> str:
    return _pwd.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return _pwd.verify(plain, hashed)

def make_token(user_id: int) -> str:
    return _signer.dumps(user_id)

def decode_token(token: str) -> int | None:
    try:
        return _signer.loads(token)
    except BadSignature:
        return None
```

- [ ] **Step 4: Esegui il test — deve passare**

```bash
python3 -m pytest tests/test_auth.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add web/auth.py tests/test_auth.py
git commit -m "feat: password hashing and session tokens"
```

---

## Task 4: Email

**Files:**
- Create: `web/email_utils.py`

- [ ] **Step 1: Implementa web/email_utils.py**

```python
from __future__ import annotations
import os, smtplib
from email.mime.text import MIMEText

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", SMTP_USER)

def send_approval_email(to_email: str, nome: str):
    if not SMTP_USER or not SMTP_PASS:
        # In sviluppo senza SMTP configurato: stampa a console
        print(f"[email] Approvazione inviata a {to_email}")
        return
    msg = MIMEText(
        f"Gentile {nome},\n\n"
        "Il suo accesso a MovieMaker è stato approvato.\n"
        "Acceda qui: " + os.environ.get("APP_URL", "http://localhost:8000") + "/login\n\n"
        "MovieMaker"
    )
    msg["Subject"] = "Accesso approvato — MovieMaker"
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
```

- [ ] **Step 2: Commit**

```bash
git add web/email_utils.py
git commit -m "feat: email approval sender"
```

---

## Task 5: RAG module

**Files:**
- Create: `web/rag.py`

- [ ] **Step 1: Implementa web/rag.py** (estrae da cineauteur.py)

```python
from __future__ import annotations
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

DOCS_DIR     = Path(__file__).parent.parent / "docs"
CHUNK_SIZE   = 350
CHUNK_OVERLAP = 60
TOP_K        = 6
MIN_SCORE    = 0.1

def read_file_content(path: Path) -> str | None:
    try:
        if path.suffix.lower() == ".pdf":
            if not PYPDF_AVAILABLE:
                return None
            reader = pypdf.PdfReader(str(path))
            return "\n\n".join(p.extract_text() or "" for p in reader.pages)
        return path.read_text(encoding="utf-8")
    except Exception:
        return None

def chunk_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + CHUNK_SIZE]))
        if i + CHUNK_SIZE >= len(words):
            break
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def load_corpus() -> tuple[list[str], list[str], int]:
    if not DOCS_DIR.exists():
        return [], [], 0
    chunks, sources = [], []
    file_count = 0
    for fp in sorted(DOCS_DIR.rglob("*")):
        if fp.suffix.lower() not in (".txt", ".pdf"):
            continue
        content = read_file_content(fp)
        if not content or len(content.strip()) < 100:
            continue
        file_count += 1
        rel = str(fp.relative_to(DOCS_DIR))
        for c in chunk_text(content):
            chunks.append(c)
            sources.append(rel)
    return chunks, sources, file_count

def build_index(chunks: list[str]):
    if not BM25_AVAILABLE or not chunks:
        return None
    return BM25Okapi([c.lower().split() for c in chunks])

def retrieve(query: str, index, chunks: list[str], sources: list[str]) -> str:
    if index is None or not chunks:
        return ""
    scores = index.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    selected = []
    for idx, score in ranked:
        if score < MIN_SCORE or len(selected) >= TOP_K:
            break
        if sum(1 for _, s in selected if s == sources[idx]) >= 2:
            continue
        selected.append((chunks[idx], sources[idx]))
    if not selected:
        return ""
    lines = ["[Contesto documentale rilevante]"]
    for i, (chunk, src) in enumerate(selected, 1):
        lines.append(f"[{i}] {src}:\n{chunk.strip()}")
    return "\n\n".join(lines)
```

- [ ] **Step 2: Verifica che il corpus si carica**

```bash
python3 -c "from web.rag import load_corpus, build_index; c,s,n=load_corpus(); print(n,'file,',len(c),'chunk')"
```

Expected: `57 file, 365 chunk`

- [ ] **Step 3: Commit**

```bash
git add web/rag.py
git commit -m "feat: RAG module extracted from cineauteur.py"
```

---

## Task 6: Templates HTML (notte calda)

**Files:**
- Create: `web/templates/base.html`
- Create: `web/templates/register.html`
- Create: `web/templates/attesa.html`
- Create: `web/templates/login.html`
- Create: `web/templates/chat.html`
- Create: `web/templates/admin.html`

- [ ] **Step 1: Crea base.html**

`web/templates/base.html`:
```html
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MovieMaker</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:      #1a1610;
      --text:    #d4c5a9;
      --border:  #2a2520;
      --label:   #4a4540;
      --faint:   #3a3028;
    }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
      font-size: 14px;
      min-height: 100vh;
    }
    .nav {
      padding: 1.2rem 1.5rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .nav-brand {
      font-size: 0.85rem;
      letter-spacing: 0.25em;
      text-transform: uppercase;
      color: var(--text);
      text-decoration: none;
    }
    .nav-link {
      font-size: 0.7rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--label);
      text-decoration: none;
    }
    .page { padding: 3rem 2.5rem; max-width: 400px; margin: 0 auto; }
    .title {
      font-size: 1.6rem;
      font-style: italic;
      font-family: 'Times New Roman', Times, serif;
      color: var(--text);
      line-height: 1.3;
      margin-bottom: 0.3rem;
    }
    .divider { width: 30px; height: 1px; background: var(--faint); margin: 1.5rem 0 2rem; }
    .field { margin-bottom: 1.2rem; }
    .field label {
      display: block;
      font-size: 0.65rem;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: var(--label);
      margin-bottom: 0.5rem;
    }
    .field input {
      width: 100%;
      background: transparent;
      border: none;
      border-bottom: 1px solid var(--border);
      padding: 0.5rem 0;
      font-size: 0.9rem;
      color: var(--text);
      font-family: inherit;
      outline: none;
    }
    .field input:focus { border-bottom-color: var(--text); }
    .field input::placeholder { color: var(--faint); }
    .btn {
      font-size: 0.65rem;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: var(--text);
      background: none;
      border: none;
      cursor: pointer;
      font-family: inherit;
      padding: 0;
    }
    .btn:hover { opacity: 0.7; }
    .error { font-size: 0.8rem; color: #c17a5a; margin-bottom: 1rem; }
    .subtitle {
      font-size: 0.8rem;
      color: var(--label);
      line-height: 1.9;
      letter-spacing: 0.02em;
    }
  </style>
</head>
<body>
  <nav class="nav">
    <a href="/" class="nav-brand">MovieMaker</a>
    {% block nav_right %}{% endblock %}
  </nav>
  {% block content %}{% endblock %}
</body>
</html>
```

- [ ] **Step 2: Crea register.html**

`web/templates/register.html`:
```html
{% extends "base.html" %}
{% block nav_right %}<a href="/login" class="nav-link">Accedi</a>{% endblock %}
{% block content %}
<div class="page">
  <div class="title">Richiedi<br>l'accesso.</div>
  <div class="divider"></div>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="post" action="/registrati">
    <div class="field">
      <label>Nome</label>
      <input type="text" name="nome" required autocomplete="name">
    </div>
    <div class="field">
      <label>Email</label>
      <input type="email" name="email" required autocomplete="email">
    </div>
    <div class="field" style="margin-bottom:2.5rem;">
      <label>Password</label>
      <input type="password" name="password" required autocomplete="new-password">
    </div>
    <button type="submit" class="btn">Invia →</button>
  </form>
</div>
{% endblock %}
```

- [ ] **Step 3: Crea attesa.html**

`web/templates/attesa.html`:
```html
{% extends "base.html" %}
{% block content %}
<div class="page" style="padding-top:5rem;">
  <div class="title">La ringraziamo<br>per il suo interesse.</div>
  <div class="divider"></div>
  <div class="subtitle">Le faremo sapere il prima possibile.</div>
</div>
{% endblock %}
```

- [ ] **Step 4: Crea login.html**

`web/templates/login.html`:
```html
{% extends "base.html" %}
{% block nav_right %}<a href="/registrati" class="nav-link">Richiedi accesso</a>{% endblock %}
{% block content %}
<div class="page">
  <div class="title">Accedi.</div>
  <div class="divider"></div>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="post" action="/login">
    <div class="field">
      <label>Email</label>
      <input type="email" name="email" required autocomplete="email">
    </div>
    <div class="field" style="margin-bottom:2.5rem;">
      <label>Password</label>
      <input type="password" name="password" required autocomplete="current-password">
    </div>
    <button type="submit" class="btn">Entra →</button>
  </form>
</div>
{% endblock %}
```

- [ ] **Step 5: Crea chat.html**

`web/templates/chat.html`:
```html
{% extends "base.html" %}
{% block nav_right %}
  <div style="display:flex;gap:1.5rem;align-items:center;">
    <span style="font-size:0.65rem;color:var(--label);letter-spacing:0.1em;text-transform:uppercase;">claude-sonnet</span>
    <a href="/logout" class="nav-link">Esci</a>
  </div>
{% endblock %}
{% block content %}
<div style="display:flex;flex-direction:column;height:calc(100vh - 49px);">
  <div id="messages" style="flex:1;overflow-y:auto;padding:1.5rem;display:flex;flex-direction:column;gap:1.5rem;"></div>
  <div style="border-top:1px solid var(--border);padding:1rem 1.5rem;display:flex;align-items:center;gap:1rem;">
    <input id="input" type="text" placeholder="Scrivi"
      style="flex:1;background:transparent;border:none;font-size:0.9rem;color:var(--text);font-family:inherit;outline:none;"
      autocomplete="off">
    <button onclick="sendMessage()" class="btn">→</button>
  </div>
</div>
<script>
const messages = document.getElementById('messages');
const input = document.getElementById('input');
let conversation = [];

input.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });

function addMessage(role, content) {
  const wrap = document.createElement('div');
  wrap.style.cssText = role === 'user' ? 'text-align:right;' : '';
  const label = document.createElement('div');
  label.style.cssText = 'font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--label);margin-bottom:0.4rem;';
  label.textContent = role === 'user' ? 'Tu' : 'MovieMaker';
  const text = document.createElement('div');
  text.style.cssText = 'font-size:0.85rem;line-height:1.8;max-width:80%;display:inline-block;text-align:left;white-space:pre-wrap;';
  text.textContent = content;
  wrap.appendChild(label);
  wrap.appendChild(text);
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
  return text;
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMessage('user', text);
  conversation.push({role:'user', content:text});

  const aiText = addMessage('assistant', '');
  const parent = aiText.parentElement;
  parent.querySelector('div').textContent = 'MovieMaker';

  const res = await fetch('/chat/stream', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({conversation})
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let full = '';

  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    for (const line of chunk.split('\n')) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') break;
        full += data;
        aiText.textContent = full;
        messages.scrollTop = messages.scrollHeight;
      }
    }
  }
  conversation.push({role:'assistant', content:full});
}
</script>
{% endblock %}
```

- [ ] **Step 6: Crea admin.html**

`web/templates/admin.html`:
```html
{% extends "base.html" %}
{% block nav_right %}<a href="/logout" class="nav-link">Esci</a>{% endblock %}
{% block content %}
<div style="padding:2rem 1.5rem;">
  <div style="font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--label);margin-bottom:2rem;">
    Richieste in attesa — {{ users|length }}
  </div>
  {% if users %}
    {% for u in users %}
    <div style="display:flex;justify-content:space-between;align-items:baseline;padding:1rem 0;border-bottom:1px solid var(--border);">
      <div>
        <div style="font-size:0.9rem;font-family:'Times New Roman',serif;font-style:italic;">{{ u.nome }}</div>
        <div style="font-size:0.7rem;color:var(--label);margin-top:0.2rem;">{{ u.email }} · {{ u.created_at }}</div>
      </div>
      <form method="post" action="/admin/approva/{{ u.id }}">
        <button type="submit" class="btn">Approva →</button>
      </form>
    </div>
    {% endfor %}
  {% else %}
    <div class="subtitle">Nessuna richiesta in attesa.</div>
  {% endif %}
</div>
{% endblock %}
```

- [ ] **Step 7: Commit**

```bash
git add web/templates/
git commit -m "feat: Jinja2 templates notte calda"
```

---

## Task 7: FastAPI app principale

**Files:**
- Create: `web/app.py`
- Create: `tests/test_routes.py`

- [ ] **Step 1: Scrivi i test**

`tests/test_routes.py`:
```python
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
```

- [ ] **Step 2: Esegui test — devono fallire**

```bash
python3 -m pytest tests/test_routes.py -v
```

Expected: `ModuleNotFoundError: No module named 'web.app'`

- [ ] **Step 3: Implementa web/app.py**

```python
from __future__ import annotations
import os
from fastapi import FastAPI, Request, Form, Cookie, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from web.db import init_db, create_user, get_user_by_email, get_user_by_id, list_pending, approve_user
from web.auth import hash_password, verify_password, make_token, decode_token
from web.email_utils import send_approval_email
from web.rag import load_corpus, build_index, retrieve

ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@localhost")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-sonnet-4-6"

SYSTEM_PROMPT = open(__file__.replace("app.py", "").replace("web/", "") + "cineauteur.py").read()
# Estrae solo il SYSTEM_PROMPT da cineauteur.py
import re
_match = re.search(r'SYSTEM_PROMPT\s*=\s*"""\\\n(.*?)"""', open(
    os.path.join(os.path.dirname(__file__), "..", "cineauteur.py")
).read(), re.DOTALL)
SYSTEM_PROMPT = _match.group(1).replace("\\\n", "\n") if _match else ""

corpus_chunks: list = []
corpus_sources: list = []
bm25 = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global corpus_chunks, corpus_sources, bm25
    init_db()
    corpus_chunks, corpus_sources, n = load_corpus()
    bm25 = build_index(corpus_chunks)
    print(f"Corpus: {n} file, {len(corpus_chunks)} chunk")
    yield

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

def get_current_user(session: str | None) -> dict | None:
    if not session:
        return None
    user_id = decode_token(session)
    if user_id is None:
        return None
    return get_user_by_id(user_id)

# --- Routes pubbliche ---

@app.get("/", response_class=RedirectResponse)
def root():
    return RedirectResponse("/login", status_code=303)

@app.get("/registrati", response_class=HTMLResponse)
def register_get(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "error": None})

@app.post("/registrati")
def register_post(request: Request, nome: str = Form(...), email: str = Form(...), password: str = Form(...)):
    if get_user_by_email(email):
        return templates.TemplateResponse("register.html", {"request": request, "error": "Email già registrata."})
    create_user(nome, email, hash_password(password))
    return RedirectResponse("/attesa", status_code=303)

@app.get("/attesa", response_class=HTMLResponse)
def attesa(request: Request):
    return templates.TemplateResponse("attesa.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login")
def login_post(request: Request, response: Response, email: str = Form(...), password: str = Form(...)):
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Email o password errati."})
    if user["stato"] != "approved":
        return templates.TemplateResponse("login.html", {"request": request, "error": "Accesso non ancora approvato."})
    token = make_token(user["id"])
    resp = RedirectResponse("/chat", status_code=303)
    resp.set_cookie("session", token, httponly=True, samesite="lax")
    return resp

@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("session")
    return resp

# --- Chat ---

@app.get("/chat", response_class=HTMLResponse)
def chat_get(request: Request, session: str | None = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse("chat.html", {"request": request, "user": user})

from fastapi.responses import StreamingResponse
from openai import OpenAI
import json

@app.post("/chat/stream")
def chat_stream(request: Request, session: str | None = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return Response(status_code=401)

    async def generator():
        body = await request.json()
        conversation = body.get("conversation", [])
        last_user = next((m["content"] for m in reversed(conversation) if m["role"] == "user"), "")
        rag_ctx = retrieve(last_user, bm25, corpus_chunks, corpus_sources)
        if rag_ctx and conversation:
            conversation = conversation[:-1] + [{"role": "user", "content": f"{rag_ctx}\n\n{last_user}"}]
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *conversation]
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        with client.chat.completions.create(
            model=MODEL, messages=msgs, max_tokens=8192, stream=True,
            extra_headers={"HTTP-Referer": "https://moviemaker.io", "X-Title": "MovieMaker"},
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {delta}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")

# --- Admin ---

@app.get("/admin", response_class=HTMLResponse)
def admin_get(request: Request, session: str | None = Cookie(default=None)):
    user = get_current_user(session)
    if not user or user["email"] != ADMIN_EMAIL:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse("admin.html", {"request": request, "users": list_pending()})

@app.post("/admin/approva/{user_id}")
def admin_approva(user_id: int, session: str | None = Cookie(default=None)):
    current = get_current_user(session)
    if not current or current["email"] != ADMIN_EMAIL:
        return RedirectResponse("/login", status_code=303)
    target = get_user_by_id(user_id)
    if target:
        approve_user(user_id)
        send_approval_email(target["email"], target["nome"])
    return RedirectResponse("/admin", status_code=303)
```

- [ ] **Step 4: Esegui i test — devono passare**

```bash
python3 -m pytest tests/test_routes.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add web/app.py tests/test_routes.py
git commit -m "feat: FastAPI routes, auth flow, admin, SSE chat"
```

---

## Task 8: Avvio locale e verifica manuale

**Files:**
- Create: `Procfile`

- [ ] **Step 1: Crea Procfile per Railway**

```
web: uvicorn web.app:app --host 0.0.0.0 --port $PORT
```

- [ ] **Step 2: Crea .env con le variabili necessarie**

Aggiungi a `.env` (già esiste per OPENROUTER_API_KEY):
```
SECRET_KEY=cambia-questa-stringa-in-produzione
ADMIN_EMAIL=la-tua-email@gmail.com
APP_URL=http://localhost:8000
# SMTP opzionale per ora — senza queste variabili le email vanno a console
# SMTP_HOST=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USER=...
# SMTP_PASS=...
```

- [ ] **Step 3: Avvia il server**

```bash
cd ~/projects/Test_1
python3 -m uvicorn web.app:app --reload --port 8000
```

Expected output:
```
Corpus: 57 file, 365 chunk
INFO: Uvicorn running on http://127.0.0.1:8000
```

- [ ] **Step 4: Verifica nel browser**

Apri `http://localhost:8000` e verifica:
1. Redirect a `/login` ✓
2. Vai su `/registrati` → registra un utente → vedi schermata attesa ✓
3. Vai su `/admin` (dopo login come admin) → approva l'utente ✓
4. Login con l'utente approvato → chat funzionante ✓

- [ ] **Step 5: Commit finale**

```bash
git add Procfile .env
git commit -m "feat: Procfile e configurazione Railway"
```

---

## Task 9: Deploy su Railway

- [ ] **Step 1: Crea repo GitHub privato**

```bash
gh repo create moviemaker --private --source=. --push
```

- [ ] **Step 2: Configura Railway**

1. Vai su [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Seleziona il repo `moviemaker`
3. Railway rileva il `Procfile` automaticamente

- [ ] **Step 3: Aggiungi variabili d'ambiente su Railway**

Nel pannello Railway → Variables, aggiungi:
```
OPENROUTER_API_KEY=<la tua chiave>
SECRET_KEY=<stringa casuale lunga>
ADMIN_EMAIL=<la tua email>
APP_URL=https://<tuo-dominio>.up.railway.app
```

- [ ] **Step 4: Verifica deploy**

Railway mostrerà il log di build. Attendi `Uvicorn running on 0.0.0.0`.
Apri l'URL Railway e verifica che il sito funzioni.

- [ ] **Step 5: Dominio custom (opzionale)**

Nel pannello Railway → Settings → Domains → aggiungi `moviemaker.io` (richiede che tu possegga il dominio).

---

## Self-Review

**Spec coverage:**
- ✓ Registrazione con nome/email/password
- ✓ Schermata attesa post-registrazione
- ✓ Admin panel con approvazione
- ✓ Email di approvazione
- ✓ Chat multi-turno con SSE streaming
- ✓ RAG BM25 integrato
- ✓ Palette notte calda
- ✓ Nome MovieMaker
- ✓ Deploy Railway

**Placeholder scan:** nessun TBD o TODO. Tutti i task hanno codice completo.

**Type consistency:** `get_user_by_id` usato in auth e admin — definito in Task 2, usato in Task 7. Tipi coerenti.

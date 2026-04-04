# MovieMaker — Memoria, Profilo Autore, Statistiche Admin

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aggiungere memoria persistente per ogni utente, generazione automatica del profilo autore al termine di ogni sessione, e pannello admin con statistiche dettagliate.

**Architecture:** Tre nuove tabelle SQLite (messages, sessions, user_profiles). Il profilo viene generato via OpenAI al termine di ogni sessione (rilevata da `beforeunload` nel browser). Il profilo viene iniettato nel system prompt ad ogni nuova sessione. Il pannello admin viene esteso con statistiche globali e per-utente.

**Tech Stack:** FastAPI, SQLite, AsyncOpenAI (OpenRouter), Jinja2, JavaScript (beforeunload + fetch beacon)

---

## File Structure

**Modificati:**
- `web/db.py` — nuove tabelle e funzioni CRUD + stats
- `web/app.py` — nuove route (start_session, end_session, admin stats, admin utente)
- `web/templates/chat.html` — JS: session_id, beforeunload, profile injection
- `web/templates/admin.html` — statistiche globali + tabella utenti con stats

**Creati:**
- `web/templates/admin_user.html` — dettaglio per-utente: profilo + sessioni
- `tests/test_memory.py` — test per nuove funzioni DB e logica profilo

---

## Task 1: Database — nuove tabelle e CRUD

**Files:**
- Modify: `web/db.py`
- Test: `tests/test_memory.py`

- [ ] **Step 1: Scrivi il test per le nuove tabelle**

```python
# tests/test_memory.py
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
    retrieved = get_user_messages(user["id"], limit=10)
    assert len(retrieved) == 2
    assert retrieved[0]["content"] == "Ciao"

def test_upsert_and_get_profile():
    from web.db import get_user_by_email
    user = get_user_by_email("test@t.com")
    upsert_profile(user["id"], "TEMI: cinema muto")
    p = get_profile(user["id"])
    assert p == "TEMI: cinema muto"
    upsert_profile(user["id"], "TEMI: cinema muto\nPROGRESSI: migliorato")
    p2 = get_profile(user["id"])
    assert "PROGRESSI" in p2
```

- [ ] **Step 2: Esegui il test per verificare che fallisce**

```bash
cd ~/projects/Test_1 && python -m pytest tests/test_memory.py -v
```
Expected: FAIL — `ImportError: cannot import name 'start_session'`

- [ ] **Step 3: Aggiungi le nuove tabelle e funzioni a web/db.py**

Sostituisci l'intera funzione `init_db` e aggiungi le nuove funzioni in fondo al file:

```python
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                started_at TEXT NOT NULL DEFAULT (datetime('now')),
                ended_at TEXT,
                message_count INTEGER NOT NULL DEFAULT 0,
                token_estimate INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                profile_text TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

def start_session(user_id: int) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (user_id) VALUES (?)", (user_id,)
        )
        return cur.lastrowid

def end_session(session_id: int, message_count: int, token_estimate: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = datetime('now'), message_count = ?, token_estimate = ? WHERE id = ?",
            (message_count, token_estimate, session_id)
        )

def get_session(session_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

def save_messages(user_id: int, session_id: int, messages: list[dict]):
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO messages (user_id, session_id, role, content) VALUES (?, ?, ?, ?)",
            [(user_id, session_id, m["role"], m["content"]) for m in messages]
        )

def get_user_messages(user_id: int, limit: int = 100) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE user_id = ? ORDER BY created_at ASC LIMIT ?",
            (user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

def upsert_profile(user_id: int, profile_text: str):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO user_profiles (user_id, profile_text, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                profile_text = excluded.profile_text,
                updated_at = excluded.updated_at
        """, (user_id, profile_text))

def get_profile(user_id: int) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT profile_text FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row["profile_text"] if row else ""

def get_global_stats() -> dict:
    with get_conn() as conn:
        total_users = conn.execute("SELECT COUNT(*) FROM users WHERE stato = 'approved'").fetchone()[0]
        pending_users = conn.execute("SELECT COUNT(*) FROM users WHERE stato = 'pending'").fetchone()[0]
        active_7d = conn.execute("""
            SELECT COUNT(DISTINCT user_id) FROM sessions
            WHERE started_at >= datetime('now', '-7 days')
        """).fetchone()[0]
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions WHERE ended_at IS NOT NULL").fetchone()[0]
        total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        total_tokens = conn.execute("SELECT COALESCE(SUM(token_estimate), 0) FROM sessions").fetchone()[0]
        return {
            "total_users": total_users,
            "pending_users": pending_users,
            "active_7d": active_7d,
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "cost_estimate": round(total_tokens / 1_000_000 * 9, 4),  # ~$9/1M token medio
        }

def list_all_users_with_stats() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                u.id, u.nome, u.email, u.stato, u.created_at,
                COUNT(DISTINCT s.id) as session_count,
                COUNT(m.id) as message_count,
                MAX(s.started_at) as last_active,
                COALESCE(p.profile_text, '') as profile_text,
                COALESCE(p.updated_at, '') as profile_updated_at
            FROM users u
            LEFT JOIN sessions s ON s.user_id = u.id
            LEFT JOIN messages m ON m.user_id = u.id
            LEFT JOIN user_profiles p ON p.user_id = u.id
            WHERE u.email != ?
            GROUP BY u.id
            ORDER BY last_active DESC NULLS LAST
        """, (os.environ.get("ADMIN_EMAIL", "admin@localhost"),)).fetchall()
        return [dict(r) for r in rows]

def get_user_sessions(user_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY started_at DESC",
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]
```

- [ ] **Step 4: Esegui i test**

```bash
cd ~/projects/Test_1 && python -m pytest tests/test_memory.py -v
```
Expected: PASS — 3 test passano

- [ ] **Step 5: Commit**

```bash
cd ~/projects/Test_1 && git add web/db.py tests/test_memory.py && git commit -m "feat: aggiungi tabelle sessions/messages/user_profiles e CRUD"
```

---

## Task 2: Gestione sessione nella chat

**Files:**
- Modify: `web/app.py` (import + route `GET /chat` + route `POST /chat/end-session`)
- Modify: `web/templates/chat.html`

- [ ] **Step 1: Scrivi il test per le nuove route**

Aggiungi alla fine di `tests/test_routes.py`:

```python
def test_end_session_requires_auth():
    r = client.post("/chat/end-session", json={"session_id": 1, "conversation": []})
    assert r.status_code == 401

def test_start_session_on_chat_get():
    # Login prima
    from web.db import get_user_by_email
    user = get_user_by_email("t@t.com")
    if user["stato"] != "approved":
        from web.db import approve_user
        approve_user(user["id"])
    r_login = client.post("/login", data={"email": "t@t.com", "password": "pass123"})
    assert r_login.status_code == 303
    # Ora GET /chat deve contenere session_id nel HTML
    r = client.get("/chat")
    assert r.status_code == 200
    assert "session_id" in r.text
```

- [ ] **Step 2: Esegui il test per verificare che fallisce**

```bash
cd ~/projects/Test_1 && python -m pytest tests/test_routes.py::test_end_session_requires_auth tests/test_routes.py::test_start_session_on_chat_get -v
```
Expected: FAIL

- [ ] **Step 3: Aggiorna gli import in web/app.py**

Sostituisci la riga degli import db:

```python
from web.db import (
    init_db, create_user, get_user_by_email, get_user_by_id,
    list_pending, approve_user, update_user_password,
    start_session, end_session, save_messages,
    upsert_profile, get_profile,
    get_global_stats, list_all_users_with_stats, get_user_sessions,
    get_user_messages,
)
```

- [ ] **Step 4: Modifica GET /chat per creare la sessione**

Sostituisci la route `chat_get`:

```python
@app.get("/chat", response_class=HTMLResponse)
def chat_get(request: Request, session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    session_id = start_session(user["id"])
    profile = get_profile(user["id"])
    return templates.TemplateResponse(request, "chat.html", {
        "user": user,
        "session_id": session_id,
        "profile": profile,
    })
```

- [ ] **Step 5: Aggiungi la route POST /chat/end-session**

Aggiungi subito dopo `chat_get`:

```python
@app.post("/chat/end-session")
async def chat_end_session(request: Request, session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return Response(status_code=401)

    from openai import AsyncOpenAI

    body = await request.json()
    session_id = body.get("session_id")
    conversation = body.get("conversation", [])

    if not session_id or not conversation:
        return Response(status_code=200)

    # Calcola stima token (1 token ≈ 4 caratteri)
    total_chars = sum(len(m.get("content", "")) for m in conversation)
    token_estimate = total_chars // 4

    # Salva messaggi
    save_messages(user["id"], session_id, conversation)
    end_session(session_id, message_count=len(conversation), token_estimate=token_estimate)

    # Genera profilo aggiornato con AI
    existing_profile = get_profile(user["id"])
    profile_prompt = f"""Sei un assistente che analizza conversazioni tra un filmmaker e un AI coach.

PROFILO ESISTENTE DELL'AUTORE (vuoto se prima sessione):
{existing_profile if existing_profile else "(nessun profilo ancora)"}

CONVERSAZIONE DI QUESTA SESSIONE:
{chr(10).join(f"{m['role'].upper()}: {m['content']}" for m in conversation)}

Aggiorna il profilo dell'autore in italiano. Sii conciso e preciso. Usa questo formato esatto:

TEMI RICORRENTI: (temi del suo lavoro che emergono)
PUNTI DI FORZA: (cosa sa fare bene)
PUNTI DEBOLI: (dove ha difficoltà)
PROGETTO ATTUALE: (a cosa sta lavorando)
PROGRESSI: (confronto con sessioni precedenti, o "prima sessione" se non c'è profilo)
ULTIMA SESSIONE: (breve sintesi di questa conversazione)"""

    try:
        client_ai = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        resp = await client_ai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": profile_prompt}],
            max_tokens=500,
            extra_headers={"HTTP-Referer": "https://moviemaker.io", "X-Title": "MovieMaker"},
        )
        new_profile = resp.choices[0].message.content.strip()
        upsert_profile(user["id"], new_profile)
    except Exception as e:
        print(f"Errore generazione profilo: {e}")

    return Response(status_code=200)
```

- [ ] **Step 6: Modifica POST /chat/stream per iniettare il profilo**

Sostituisci il blocco che costruisce `msgs` in `chat_stream`:

```python
    profile = get_profile(user["id"])
    system_content = SYSTEM_PROMPT
    if profile:
        system_content = f"{SYSTEM_PROMPT}\n\nPROFILO DELL'AUTORE (usa queste informazioni per personalizzare le risposte):\n{profile}"
    msgs = [{"role": "system", "content": system_content}, *conversation]
```

- [ ] **Step 7: Aggiorna chat.html**

Sostituisci l'intero file `web/templates/chat.html`:

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
const SESSION_ID = {{ session_id }};
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

// Salva sessione quando l'utente chiude o cambia pagina
window.addEventListener('beforeunload', () => {
  if (conversation.length === 0) return;
  navigator.sendBeacon('/chat/end-session', JSON.stringify({
    session_id: SESSION_ID,
    conversation: conversation
  }));
});
</script>
{% endblock %}
```

- [ ] **Step 8: Esegui tutti i test**

```bash
cd ~/projects/Test_1 && python -m pytest tests/ -v
```
Expected: tutti i test passano

- [ ] **Step 9: Commit**

```bash
cd ~/projects/Test_1 && git add web/app.py web/templates/chat.html tests/test_routes.py && git commit -m "feat: sessioni chat, salvataggio messaggi, generazione profilo autore"
```

---

## Task 3: Pannello admin con statistiche

**Files:**
- Modify: `web/app.py` (route GET /admin, nuova route GET /admin/utente/{user_id})
- Modify: `web/templates/admin.html`
- Create: `web/templates/admin_user.html`

- [ ] **Step 1: Scrivi il test per le nuove route admin**

Aggiungi alla fine di `tests/test_routes.py`:

```python
def test_admin_stats_page_loads():
    # Login come admin
    import os
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@test.com")
    r_login = client.post("/login", data={"email": admin_email, "password": "adminpass"})
    # Se non esiste admin, skip
    if r_login.status_code != 303:
        return
    r = client.get("/admin")
    assert r.status_code == 200
    assert "statistiche" in r.text.lower() or "utenti" in r.text.lower()
```

- [ ] **Step 2: Modifica GET /admin in web/app.py**

Sostituisci la route `admin_get`:

```python
@app.get("/admin", response_class=HTMLResponse)
def admin_get(request: Request, session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user or user["email"] != ADMIN_EMAIL:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request, "admin.html", {
        "pending": list_pending(),
        "stats": get_global_stats(),
        "users": list_all_users_with_stats(),
    })
```

- [ ] **Step 3: Aggiungi la route GET /admin/utente/{user_id}**

Aggiungi subito dopo `admin_get`:

```python
@app.get("/admin/utente/{user_id}", response_class=HTMLResponse)
def admin_user_detail(user_id: int, request: Request, session: Optional[str] = Cookie(default=None)):
    current = get_current_user(session)
    if not current or current["email"] != ADMIN_EMAIL:
        return RedirectResponse("/login", status_code=303)
    target = get_user_by_id(user_id)
    if not target:
        return RedirectResponse("/admin", status_code=303)
    return templates.TemplateResponse(request, "admin_user.html", {
        "target": target,
        "profile": get_profile(user_id),
        "sessions": get_user_sessions(user_id),
        "messages": get_user_messages(user_id, limit=200),
    })
```

- [ ] **Step 4: Riscrivi web/templates/admin.html**

```html
{% extends "base.html" %}
{% block nav_right %}<a href="/logout" class="nav-link">Esci</a>{% endblock %}
{% block content %}
<div style="padding:2rem 1.5rem;max-width:900px;">

  <!-- Statistiche globali -->
  <div style="font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--label);margin-bottom:1.5rem;">Statistiche</div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:3rem;">
    {% set stats_items = [
      ("Utenti approvati", stats.total_users),
      ("In attesa", stats.pending_users),
      ("Attivi (7 giorni)", stats.active_7d),
      ("Sessioni totali", stats.total_sessions),
      ("Messaggi totali", stats.total_messages),
      ("Costo API stimato", "$" ~ stats.cost_estimate),
    ] %}
    {% for label, value in stats_items %}
    <div style="border:1px solid var(--border);padding:1rem;">
      <div style="font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--label);margin-bottom:0.5rem;">{{ label }}</div>
      <div style="font-size:1.4rem;color:var(--text);font-family:'Times New Roman',serif;font-style:italic;">{{ value }}</div>
    </div>
    {% endfor %}
  </div>

  <!-- Richieste in attesa -->
  {% if pending %}
  <div style="font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--label);margin-bottom:1.5rem;">Richieste in attesa — {{ pending|length }}</div>
  {% for u in pending %}
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
  <div style="margin-bottom:3rem;"></div>
  {% endif %}

  <!-- Tutti gli utenti -->
  <div style="font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--label);margin-bottom:1.5rem;">Utenti — {{ users|length }}</div>
  {% if users %}
  {% for u in users %}
  <div style="display:flex;justify-content:space-between;align-items:baseline;padding:1rem 0;border-bottom:1px solid var(--border);">
    <div style="flex:1;">
      <div style="font-size:0.9rem;font-family:'Times New Roman',serif;font-style:italic;">{{ u.nome }}</div>
      <div style="font-size:0.7rem;color:var(--label);margin-top:0.2rem;">
        {{ u.email }} · {{ u.session_count }} sessioni · {{ u.message_count }} messaggi
        {% if u.last_active %} · ultimo accesso {{ u.last_active[:10] }}{% endif %}
      </div>
      {% if u.profile_text %}
      <div style="font-size:0.75rem;color:var(--text);margin-top:0.4rem;opacity:0.7;max-width:500px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
        {{ u.profile_text[:120] }}…
      </div>
      {% endif %}
    </div>
    <a href="/admin/utente/{{ u.id }}" class="btn" style="text-decoration:none;">Dettaglio →</a>
  </div>
  {% endfor %}
  {% else %}
  <div class="subtitle">Nessun utente ancora.</div>
  {% endif %}

</div>
{% endblock %}
```

- [ ] **Step 5: Crea web/templates/admin_user.html**

```html
{% extends "base.html" %}
{% block nav_right %}
  <div style="display:flex;gap:1.5rem;align-items:center;">
    <a href="/admin" class="nav-link">← Admin</a>
    <a href="/logout" class="nav-link">Esci</a>
  </div>
{% endblock %}
{% block content %}
<div style="padding:2rem 1.5rem;max-width:700px;">

  <div style="font-size:1.4rem;font-family:'Times New Roman',serif;font-style:italic;color:var(--text);margin-bottom:0.3rem;">{{ target.nome }}</div>
  <div style="font-size:0.7rem;color:var(--label);margin-bottom:2rem;">{{ target.email }} · registrato il {{ target.created_at[:10] }}</div>

  <!-- Statistiche utente -->
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:2.5rem;">
    <div style="border:1px solid var(--border);padding:1rem;">
      <div style="font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--label);margin-bottom:0.5rem;">Sessioni</div>
      <div style="font-size:1.4rem;color:var(--text);font-family:'Times New Roman',serif;font-style:italic;">{{ sessions|length }}</div>
    </div>
    <div style="border:1px solid var(--border);padding:1rem;">
      <div style="font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--label);margin-bottom:0.5rem;">Messaggi</div>
      <div style="font-size:1.4rem;color:var(--text);font-family:'Times New Roman',serif;font-style:italic;">{{ messages|length }}</div>
    </div>
    <div style="border:1px solid var(--border);padding:1rem;">
      <div style="font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--label);margin-bottom:0.5rem;">Ultima sessione</div>
      <div style="font-size:0.85rem;color:var(--text);font-family:'Times New Roman',serif;font-style:italic;">
        {% if sessions %}{{ sessions[0].started_at[:10] }}{% else %}—{% endif %}
      </div>
    </div>
  </div>

  <!-- Profilo autore -->
  <div style="font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--label);margin-bottom:1rem;">Profilo Autore</div>
  {% if profile %}
  <div style="font-size:0.85rem;line-height:1.9;color:var(--text);white-space:pre-wrap;margin-bottom:3rem;padding:1.2rem;border:1px solid var(--border);">{{ profile }}</div>
  {% else %}
  <div style="font-size:0.8rem;color:var(--label);margin-bottom:3rem;">Nessun profilo ancora — il profilo si genera al termine della prima sessione.</div>
  {% endif %}

  <!-- Sessioni -->
  <div style="font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--label);margin-bottom:1rem;">Sessioni</div>
  {% for s in sessions %}
  <div style="padding:0.8rem 0;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;">
    <div style="font-size:0.8rem;color:var(--text);">{{ s.started_at[:16] }}</div>
    <div style="font-size:0.75rem;color:var(--label);">{{ s.message_count }} messaggi · ~{{ s.token_estimate }} token</div>
  </div>
  {% else %}
  <div style="font-size:0.8rem;color:var(--label);">Nessuna sessione completata.</div>
  {% endfor %}

</div>
{% endblock %}
```

- [ ] **Step 6: Esegui tutti i test**

```bash
cd ~/projects/Test_1 && python -m pytest tests/ -v
```
Expected: tutti i test passano

- [ ] **Step 7: Commit e deploy**

```bash
cd ~/projects/Test_1 && git add web/app.py web/templates/admin.html web/templates/admin_user.html tests/test_routes.py && git commit -m "feat: pannello admin con statistiche globali e profilo per-utente"
git push && railway up --detach
```

---

## Task 4: Corpus commerciale — festival, piattaforme, distribuzione, pubblico

**Files:**
- Create: `docs/mercato/guida_commerciale.txt`

- [ ] **Step 1: Crea la directory e il file**

```bash
mkdir -p /Users/filippoarici/projects/Test_1/docs/mercato
```

- [ ] **Step 2: Scrivi il file `docs/mercato/guida_commerciale.txt`**

Il file deve contenere conoscenza enciclopedica strutturata in sezioni. Contenuto completo:

```
GUIDA COMMERCIALE AL CINEMA D'AUTORE — MovieMaker Knowledge Base
Versione 1.0 — Aprile 2026

=== FESTIVAL INTERNAZIONALI DI PRIMO LIVELLO ===

CANNES (Festival de Cannes)
Luogo: Cannes, Francia | Periodo: maggio
Sezioni principali: Palme d'Or (Compétition officielle), Un Certain Regard, Semaine de la Critique, Quinzaine des Cinéastes, Cinéfondation (corti e mediometraggi studenteschi)
Tipologia film premiata: cinema d'autore radicale, film politicamente impegnati, registi già affermati o emergenti con voce forte. Stile spesso formalmente rigoroso. Durata variabile.
Cortometraggi: Palme d'Or per il cortometraggio — uno dei riconoscimenti più prestigiosi al mondo per i corti. Accetta finzione e documentario.
Soglia di accettazione: altissima. Spesso richiede presenza del regista. Film italiani storicamente ben accolti.
Come candidarsi: tramite il sito ufficiale (festival-cannes.com), invio del film online entro gennaio-febbraio. Solo film in anteprima mondiale o internazionale.
Pubblico: critica internazionale, distributori, agenti di vendita. Lancio ideale per carriera internazionale.

VENEZIA (Mostra Internazionale d'Arte Cinematografica)
Luogo: Venezia, Italia | Periodo: agosto-settembre
Sezioni: Concorso principale (Leone d'Oro), Orizzonti, Orizzonti Extra, Venezia Classici, Settimana della Critica, Giornate degli Autori
Tipologia: film sperimentali, autoriali, narrativi ambiziosi. Storicamente più aperto a film di genere rispetto a Cannes. Forte componente italiana.
Cortometraggi: sezione Orizzonti corti — molto competitiva. Giornate degli Autori accetta corti.
Soglia: altissima per il concorso, moderata per sezioni parallele.
Come candidarsi: labiennale.org, invio online, deadline aprile-maggio.
Pubblico: stampa internazionale, industria, pubblico italiano e internazionale. Ottima visibilità per registi italiani.

BERLINO (Berlinale)
Luogo: Berlino, Germania | Periodo: febbraio
Sezioni: Concorso (Orso d'Oro), Panorama, Forum, Generation (film per ragazzi e giovani), Berlinale Shorts
Tipologia: cinema politicamente e socialmente impegnato, film sulla marginalità, tematiche LGBTQ+, cinema del reale. Molto aperto a prime opere.
Cortometraggi: Berlinale Shorts — sezione dedicata, Orso d'Oro per il miglior corto. Molto prestigiosa. Accetta anche corti sperimentali.
Come candidarsi: berlinale.de, deadline ottobre-novembre dell'anno precedente.
Pubblico: molto orientato all'industria (European Film Market, il più grande mercato del cinema in Europa).

SUNDANCE
Luogo: Park City, Utah, USA | Periodo: gennaio
Sezioni: US Dramatic, US Documentary, World Cinema Dramatic, World Cinema Documentary, Short Films, Midnight
Tipologia: cinema indipendente americano e internazionale. Storie personali, film di genere indie, documentari politici e sociali. Porta a distribuzioni importanti negli USA.
Cortometraggi: sezione dedicata molto competitiva. Spesso trampolino di lancio per registi che poi passano al lungometraggio.
Come candidarsi: sundance.org/festival, tramite Film Freeway, deadline settembre-ottobre.
Pubblico: distributori USA, agenti, media americani. Ideale per chi vuole distribuzione negli Stati Uniti.

TORONTO (TIFF — Toronto International Film Festival)
Luogo: Toronto, Canada | Periodo: settembre
Tipologia: festival molto industry-oriented. Spesso primo grande festival dove i film cercano distribuzione nordamericana. Meno autoriale di Cannes/Venezia, più commerciale.
Premio del pubblico: spesso predittivo degli Oscar.
Come candidarsi: tiff.net, deadline maggio-giugno.

LOCARNO
Luogo: Locarno, Svizzera | Periodo: agosto
Sezioni: Concorso internazionale (Pardo d'Oro), Cineasti del Presente (prime opere), Semaine de la Critique
Tipologia: cinema radicale e sperimentale, prime opere, cinema del reale. Molto aperto a film rischiosi e non convenzionali. Ideale per registi emergenti con linguaggio personale forte.
Cortometraggi: Pardo d'Oro per il corto. Sezione Corti d'autore.
Come candidarsi: pardo.ch, deadline aprile-maggio.
Note: festival molto rispettato nell'industria europea. Più accessibile di Cannes/Venezia per esordienti.

=== FESTIVAL INTERNAZIONALI DI SECONDO LIVELLO (altamente strategici) ===

IDFA (International Documentary Film Festival Amsterdam)
Luogo: Amsterdam, Paesi Bassi | Periodo: novembre
Il festival più importante al mondo per il documentario. Concorso principale, IDFA Competition for Feature-Length Documentary, per corti e medi.
Ideale per: documentari d'autore, film ibridi fiction/documentario, cinema del reale.
Come candidarsi: idfa.nl, deadline giugno-luglio.

HOT DOCS
Luogo: Toronto, Canada | Periodo: aprile-maggio
Il più grande festival di documentario in Nord America. Mercato molto attivo per pre-vendite e acquisizioni.
Come candidarsi: hotdocs.ca

TRIBECA
Luogo: New York, USA | Periodo: giugno
Festival fondato da Robert De Niro. Forte componente americana ma aperto all'internazionale. Molto seguito da industria e media USA.
Come candidarsi: tribecafilm.com

SAN SEBASTIÁN (Zinemaldia)
Luogo: San Sebastián, Spagna | Periodo: settembre
Festival di primo piano europeo. Forte interesse per cinema latino e del Sud Europa. Ottima vetrina per film italiani.

ROTTERDAM (IFFR)
Luogo: Rotterdam, Paesi Bassi | Periodo: gennaio-febbraio
Festival molto aperto al cinema sperimentale e non convenzionale. Progetto Hubert Bals Fund per paesi in via di sviluppo. Meno industry, più curatoriale.

SXSW (South by Southwest)
Luogo: Austin, Texas, USA | Periodo: marzo
Forte componente indie americana. Molto seguito per scoperta di nuovi talenti. Sezione film e musica.

=== FESTIVAL ITALIANI ===

TORINO FILM FESTIVAL
Il principale festival italiano dopo Venezia. Sezione TFFdoc per documentari. Buona visibilità nazionale e internazionale. Deadline agosto-settembre.

FESTIVAL DEI POPOLI (Firenze)
Il più antico festival di documentario in Italia. Specializzato in cinema del reale internazionale.

FILMMAKER FESTIVAL (Milano)
Festival dedicato al cinema indipendente e d'autore italiano e internazionale. Importante per la scena italiana.

BIOGRAFILM FESTIVAL (Bologna)
Specializzato in biografie e storie di vita reale, documentari e fiction.

VISIONI DAL MONDO (Milano)
Festival di documentario. Buona vetrina per doc italiani.

SHORT THEATRE / CORTO DORICO / ARCIPELAGO
Festival italiani dedicati al cortometraggio. Importanti per visibilità nazionale.

=== PIATTAFORME STREAMING — CRITERI DI ACQUISIZIONE ===

MUBI
Modello: curatoriale. Acquisisce film d'autore, cinema classico e contemporaneo di qualità. Non acquisisce contenuti commerciali mainstream.
Tipologia: lungometraggi d'autore (Cannes, Venezia, Berlino), documentari d'autore, cinema internazionale.
Come funziona: contattare direttamente attraverso agenti di vendita o submit tramite il loro portale. Molto selettivi.
Diritti: spesso SVOD (streaming su abbonamento), finestra temporanea (30 giorni per film, poi rotazione).
Ideale per: film autoriali con riconoscimento festivaliero. La presenza su MUBI è un segnale di qualità artistico riconosciuto.
Mercati: globale, con forte presenza Europa, USA, Turchia, India.

NETFLIX
Modello: acquisisce sia originali che acquisizioni da festival. Sempre più orientato verso contenuti con potenziale di audience ampia.
Tipologia per acquisizioni: film con cast riconoscibile, generi forti (thriller, horror, commedia), documentari su temi universali o figure note.
Documentari: Netflix acquisisce documentari su personaggi famosi, eventi storici, crimine, sport. Meno interessato a documentari d'autore puri.
Film d'autore: acquisisce film di festival SOLO se hanno grande risonanza (Palme d'Or, Leone d'Oro) o registi molto noti.
Come candidarsi: quasi esclusivamente tramite agenti di vendita internazionali (sales agents). Non accetta submission dirette.
Realtà italiana: ha investito in produzioni italiane (Paolo Sorrentino, etc.) ma è molto difficile senza intermediario.

AMAZON PRIME VIDEO
Simile a Netflix ma con maggiore apertura ai contenuti locali europei. Programma Amazon Studios per originali.
Acquisisce documentari e film indipendenti con più facilità di Netflix.

APPLE TV+
Orientato a produzioni originali di alta qualità. Non acquisisce molto da festival. Richiede produzione con standard molto alti.

DISNEY+/STAR
Orientato a franchise e contenuti family. Non rilevante per cinema d'autore indipendente.

RAIPLAY (Italia)
Piattaforma pubblica italiana. Acquisisce film e documentari italiani. Ottima per distribuzione nazionale.
Come candidarsi: attraverso Rai Cinema o distribuzione tradizionale italiana.

DAFILMS / MUBI ALTERNATIVES
DAFilms: piattaforma specializzata in documentario d'autore internazionale. Più accessibile di MUBI.
Eventive: piattaforma per distribuzione virtuale cinema indipendente.
Guidedoc: documentario internazionale.
Fandor: cinema indipendente USA.

=== DISTRIBUZIONE IN SALA CINEMATOGRAFICA ===

QUANDO HA SENSO USCIRE IN SALA
- Film con riconoscimento festivaliero importante (almeno un festival di secondo livello)
- Documentari su temi di interesse pubblico ampio
- Prime opere con sostegno critico
- Film con co-produzione internazionale e pre-vendite

QUANDO NON HA SENSO
- Cortometraggi (raramente distribuiti in sala se non allegati a lungometraggi)
- Film senza riconoscimento critico o festivaliero
- Film molto sperimentali senza pubblico identificabile
- Produzioni con budget troppo basso per sostenere costi di distribuzione

DISTRIBUZIONE ITALIANA — COME FUNZIONA
I distributori italiani principali per cinema d'autore: Lucky Red, Teodora Film, Wanted Cinema, I Wonder Pictures, Movies Inspired, Fandango.
Il distributore acquisisce i diritti per l'Italia (theatrical + home video + streaming) pagando un minimo garantito.
La campagna promozionale costa quanto o più del film stesso per uscite importanti.
Il cinema d'autore in sala: tipicamente 3-10 copie in uscita, poi espansione se funziona. Circuito d'essai.
Box office aspettative realistiche per film d'autore italiano: 50.000-500.000 euro di incasso è già un successo per un film senza star.

CIRCUITO D'ESSAI
Cinema specializzati in film d'autore: Anteo (Milano), Farnese (Roma), Arcobaleno (Milano), Troisi (Roma), e centinaia in tutta Italia.
Questi cinema sono fondamentali per film d'autore senza distribuzione commerciale ampia.

=== SEGMENTAZIONE DEL PUBBLICO ===

PUBBLICO CINEFILO D'AUTORE
Caratteristiche: 25-55 anni, istruzione alta, abituati ai festival, leggono riviste di cinema (Filmidee, Cineforum, Cineaste), seguono MUBI.
Cosa cerca: cinema con linguaggio personale, temi profondi, formalmente interessante.
Come raggiungerlo: festival, MUBI, circuito d'essai, rassegne cinematografiche, critica specializzata.
Film di riferimento: tutto il cinema di Cannes, Locarno, Venezia.

PUBBLICO DOCUMENTARIO
Caratteristiche: più ampio del cinefilo puro. Interesse per temi specifici (ambiente, politica, storia, personaggi).
Come raggiungerlo: festival documentario (IDFA, Hot Docs, Torino), piattaforme (Netflix doc, MUBI, DAFilms), media tematici.
Nota: il documentario con protagonista noto o tema di attualità raggiunge pubblico molto più ampio.

PUBBLICO MAINSTREAM COLTO
Caratteristiche: va al cinema regolarmente ma non solo d'autore. Segue recensioni su quotidiani.
Come raggiungerlo: distribuzione in sala con campagna stampa, passaparola, premi importanti.
Film di riferimento: film pluripremiati ai festival con distribuzione internazionale (es. Parasite, Drive My Car).

PUBBLICO GIOVANE / SCUOLE DI CINEMA
Caratteristiche: studenti, aspiranti filmmaker. Molto attivi sui social, seguono festival.
Come raggiungerlo: proiezioni universitarie, workshop, social media, YouTube.

=== VALUTAZIONE COMMERCIALE — DOMANDE CHIAVE ===

Per valutare il potenziale commerciale di un progetto, considerare:

1. FORMA: Il film è narrativo o sperimentale? Quanto è accessibile?
   - Narrativo accessibile → possibilità di distribuzione mainstream
   - Sperimentale → festival specializzati (Locarno, Rotterdam) + MUBI

2. TEMA: Il tema ha interesse universale o è molto locale/specifico?
   - Universale → potenziale internazionale
   - Locale → mercato nazionale + festival regionali + diaspora

3. DURATA: Cortometraggio, mediometraggio, lungometraggio?
   - Corto: festival + online + trampolino per lungo
   - Medio: difficile da piazzare, alcuni festival specializzati
   - Lungo: tutti i canali disponibili

4. RICONOSCIMENTI: Ha già vinto premi o selezioni?
   - Sì → leverage per vendite e distribuzione
   - No → dipende tutto dalla qualità e dal "buzz"

5. CAST E CREW: Nomi riconoscibili?
   - Sì → più facile acquisizione piattaforme
   - No → dipende dalla qualità artistica e dal riconoscimento festivaliero

6. CO-PRODUZIONE: Ha co-produttori internazionali?
   - Sì → accesso a mercati dei partner + maggiore credibilità
   - No → distribuzione tende a restare nel paese d'origine

=== STRATEGIE DI LANCIO RACCOMANDATE ===

STRATEGIA A: Festival First
Percorso: festival internazionale importante → buzz critico → acquisizione distribuzione → sala e/o streaming
Ideale per: film con ambizioni artistiche alte, prima opera con voce originale
Rischio: se nessun festival importante lo seleziona, difficile rilanciarlo

STRATEGIA B: Niche Streaming First
Percorso: MUBI o DAFilms → visibilità critica → festival retrospettive → distribuzione fisica
Ideale per: film molto d'autore che potrebbero non passare la selezione festivaliera mainstream

STRATEGIA C: Documentario Engagement
Percorso: proiezioni tematiche → media specializzati → piattaforme tematiche → festival doc
Ideale per: documentari su temi specifici con comunità di interesse preesistente

STRATEGIA D: Scuola di cinema / mercato locale
Percorso: festival italiani → distribuzione nazionale → RaiPlay
Ideale per: prime opere, film con budget limitato, storie molto italiane
```

- [ ] **Step 3: Verifica che il file venga indicizzato dal corpus**

```bash
cd ~/projects/Test_1 && python3 -c "
from web.rag import load_corpus
chunks, sources, n = load_corpus()
mercato = [s for s in sources if 'mercato' in s]
print(f'File mercato trovati: {len(mercato)}')
print(f'Totale chunk: {len(chunks)}')
"
```
Expected: `File mercato trovati: N > 0`

- [ ] **Step 4: Commit**

```bash
cd ~/projects/Test_1 && git add docs/mercato/ && git commit -m "feat: aggiungi corpus commerciale festival/piattaforme/distribuzione/pubblico"
```

---

## Task 5: System prompt — rilevamento somiglianze + valutazione commerciale

**Files:**
- Modify: `cineauteur.py` (SYSTEM_PROMPT)

- [ ] **Step 1: Leggi l'attuale SYSTEM_PROMPT**

```bash
cd ~/projects/Test_1 && python3 -c "
import re, pathlib
src = pathlib.Path('cineauteur.py').read_text()
m = re.search(r'SYSTEM_PROMPT\s*=\s*\"\"\"\\\\\n(.*?)\"\"\"', src, re.DOTALL)
print(m.group(1)[:500])
"
```

- [ ] **Step 2: Aggiungi le nuove istruzioni in fondo al SYSTEM_PROMPT**

Trova la riga che chiude il SYSTEM_PROMPT (la `"""` finale) in `cineauteur.py` e inserisci prima di essa questo blocco:

```
---

RILEVAMENTO SOMIGLIANZE CON OPERE ESISTENTI
Mentre lavori con l'autore, monitora attivamente se il progetto che sta sviluppando assomiglia in modo significativo a opere già esistenti. Valuta la somiglianza su tutti i livelli: forma visiva e stilistica, struttura narrativa, tono e registro, scrittura e dialoghi, approccio alle riprese e al montaggio, tematica di fondo. Se rilevi una somiglianza forte e specifica con uno o più film esistenti, segnalala all'autore in modo diretto ma non giudicante. Esempio: "Quello che mi descrivi mi ricorda molto [film] di [regista] — in particolare [aspetto specifico]. Vale la pena che tu ne sia consapevole: potrebbe essere un riferimento inconscio da elaborare, o un punto di partenza da cui allontanarsi deliberatamente. Come vuoi procedere?" Non bloccare l'autore, ma portalo a una scelta consapevole.

VALUTAZIONE COMMERCIALE E DISTRIBUZIONE
Quando l'autore chiede esplicitamente o quando il contesto lo rende rilevante, puoi offrire una valutazione del potenziale commerciale del progetto. Basa la valutazione su: forma e accessibilità del film, tema e interesse universale vs locale, durata, presenza di co-produttori o cast noti, e stadio di sviluppo. Indica quali festival potrebbero essere adatti (dal più ambizioso al più realistico), quali piattaforme streaming potrebbero acquisirlo, se ha senso una distribuzione in sala, e quale pubblico potrebbe raggiungerlo. Sii onesto anche quando le prospettive commerciali sono limitate — un film molto sperimentale ha mercato reale ma ristretto, e dirlo chiaramente è più utile che essere vagamente ottimista. Non offrire questa valutazione spontaneamente in ogni messaggio — solo quando è richiesta o quando è chiaramente il momento giusto nel processo creativo.
```

- [ ] **Step 3: Verifica che il SYSTEM_PROMPT venga caricato correttamente**

```bash
cd ~/projects/Test_1 && python3 -c "
import re, pathlib
src = pathlib.Path('cineauteur.py').read_text()
m = re.search(r'SYSTEM_PROMPT\s*=\s*\"\"\"\\\\\n(.*?)\"\"\"', src, re.DOTALL)
prompt = m.group(1)
assert 'RILEVAMENTO SOMIGLIANZE' in prompt
assert 'VALUTAZIONE COMMERCIALE' in prompt
print('OK — istruzioni presenti nel SYSTEM_PROMPT')
print(f'Lunghezza totale: {len(prompt)} caratteri')
"
```
Expected: `OK — istruzioni presenti nel SYSTEM_PROMPT`

- [ ] **Step 4: Esegui tutti i test**

```bash
cd ~/projects/Test_1 && python -m pytest tests/ -v
```
Expected: tutti i test passano

- [ ] **Step 5: Commit e deploy**

```bash
cd ~/projects/Test_1 && git add cineauteur.py && git commit -m "feat: aggiungi rilevamento somiglianze e valutazione commerciale al system prompt"
git push && railway up --detach
```

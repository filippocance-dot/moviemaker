# Filmmaker — UI/UX Overhaul

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rinominare l'app in "Filmmaker", migliorare l'estetica, introdurre textarea ChatGPT-style, auto-scroll intelligente, upload file (immagini/PDF/testo), storico conversazioni.

**Architecture:** Tutti i cambiamenti di UI sono nei template Jinja2 e nel CSS inline di base.html. Il file upload usa un endpoint FastAPI separato che processa i file in memoria (nessuna scrittura su disco). Lo storico usa le tabelle `sessions` e `messages` già esistenti. Markdown rendering via marked.js (CDN).

**Tech Stack:** FastAPI, Jinja2, JavaScript (vanilla), marked.js (CDN), pypdf (già installato)

---

## File Structure

**Modificati:**
- `web/templates/base.html` — rename MovieMaker→Filmmaker, CSS overhaul (markdown styles, textarea styles, message styles)
- `web/templates/chat.html` — textarea, auto-scroll, markdown rendering, link storico, label → Filmmaker
- `web/templates/login.html` — nessuna modifica funzionale (il CSS di base.html fa il lavoro)
- `web/templates/register.html` — nessuna modifica funzionale
- `web/templates/attesa.html` — nessuna modifica funzionale
- `web/templates/admin.html` — rename riferimenti visibili
- `web/templates/admin_user.html` — rename riferimenti visibili
- `web/app.py` — nuovo endpoint POST /chat/upload, nuove route GET /chat/storia e GET /chat/storia/{session_id}, fix RAG per messaggi multimodali
- `web/db.py` — nuova funzione get_session_messages(session_id)

**Creati:**
- `web/templates/history.html` — lista sessioni passate dell'utente
- `web/templates/history_session.html` — visualizzazione completa di una sessione

---

## Task 1: Rename + Aesthetic Overhaul

**Files:**
- Modify: `web/templates/base.html`
- Modify: `web/templates/admin.html` (riga label brand)
- Modify: `web/templates/admin_user.html` (nessuna modifica visibile, già ok)
- Modify: `web/templates/chat.html` (solo label "MovieMaker" → "Filmmaker" nei messaggi)

- [ ] **Step 1: Sostituisci `web/templates/base.html` con questo contenuto**

```html
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Filmmaker</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:      #1a1610;
      --text:    #d4c5a9;
      --border:  #2a2520;
      --label:   #4a4540;
      --faint:   #3a3028;
      --accent:  #c8b89a;
    }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
      font-size: 14px;
      min-height: 100vh;
    }
    .nav {
      padding: 1.1rem 2rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .nav-brand {
      font-size: 0.8rem;
      letter-spacing: 0.3em;
      text-transform: uppercase;
      color: var(--text);
      text-decoration: none;
    }
    .nav-link {
      font-size: 0.65rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--label);
      text-decoration: none;
      transition: color 0.2s;
    }
    .nav-link:hover { color: var(--text); }
    .page { padding: 4rem 2.5rem; max-width: 420px; margin: 0 auto; }
    .title {
      font-size: 1.8rem;
      font-style: italic;
      font-family: 'Times New Roman', Times, serif;
      color: var(--text);
      line-height: 1.25;
      margin-bottom: 0.3rem;
    }
    .divider { width: 28px; height: 1px; background: var(--faint); margin: 1.8rem 0 2.2rem; }
    .field { margin-bottom: 1.5rem; }
    .field label {
      display: block;
      font-size: 0.6rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--label);
      margin-bottom: 0.6rem;
    }
    .field input {
      width: 100%;
      background: transparent;
      border: none;
      border-bottom: 1px solid var(--border);
      padding: 0.6rem 0;
      font-size: 0.95rem;
      color: var(--text);
      font-family: inherit;
      outline: none;
      transition: border-color 0.2s;
    }
    .field input:focus { border-bottom-color: var(--accent); }
    .field input::placeholder { color: var(--faint); }
    .btn {
      font-size: 0.62rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--text);
      background: none;
      border: none;
      cursor: pointer;
      font-family: inherit;
      padding: 0;
      transition: opacity 0.2s;
    }
    .btn:hover { opacity: 0.6; }
    .error {
      font-size: 0.78rem;
      color: #c17a5a;
      margin-bottom: 1.2rem;
      letter-spacing: 0.02em;
    }
    .subtitle {
      font-size: 0.78rem;
      color: var(--label);
      line-height: 2;
      letter-spacing: 0.02em;
    }
    /* Markdown rendering inside AI messages */
    .md p { margin-bottom: 0.9rem; line-height: 1.85; }
    .md p:last-child { margin-bottom: 0; }
    .md h1, .md h2, .md h3 {
      font-family: 'Times New Roman', serif;
      font-style: italic;
      font-weight: normal;
      margin: 1.2rem 0 0.5rem;
      color: var(--text);
    }
    .md h1 { font-size: 1.2rem; }
    .md h2 { font-size: 1.05rem; }
    .md h3 { font-size: 0.95rem; }
    .md ul, .md ol { margin: 0.4rem 0 0.9rem 1.4rem; }
    .md li { margin-bottom: 0.35rem; line-height: 1.8; }
    .md strong { color: var(--accent); font-weight: 600; }
    .md em { font-style: italic; }
    .md code {
      background: rgba(255,255,255,0.06);
      padding: 0.15rem 0.35rem;
      border-radius: 2px;
      font-family: 'Courier New', monospace;
      font-size: 0.82em;
    }
    .md pre {
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border);
      padding: 1rem 1.2rem;
      border-radius: 3px;
      overflow-x: auto;
      margin: 0.8rem 0;
    }
    .md pre code { background: none; padding: 0; }
    .md blockquote {
      border-left: 2px solid var(--faint);
      padding-left: 1rem;
      color: var(--label);
      margin: 0.8rem 0;
      font-style: italic;
    }
    .md hr { border: none; border-top: 1px solid var(--border); margin: 1.2rem 0; }
  </style>
</head>
<body>
  <nav class="nav">
    <a href="/" class="nav-brand">Filmmaker</a>
    {% block nav_right %}{% endblock %}
  </nav>
  {% block content %}{% endblock %}
</body>
</html>
```

- [ ] **Step 2: Verifica visivamente che il sito si avvii senza errori**

```bash
cd ~/projects/Test_1 && python -m pytest tests/ -v
```
Expected: tutti i test passano

- [ ] **Step 3: Commit**

```bash
cd ~/projects/Test_1 && git add web/templates/base.html && git commit -m "feat: rename Filmmaker + aesthetic overhaul base.html"
```

---

## Task 2: Chat UX — textarea, auto-scroll, markdown, storico link

**Files:**
- Modify: `web/templates/chat.html`

- [ ] **Step 1: Scrivi il test per la nuova struttura chat**

Aggiungi alla fine di `tests/test_routes.py`:

```python
def test_chat_has_textarea():
    from web.db import get_user_by_email, approve_user
    user = get_user_by_email("t@t.com")
    if user and user["stato"] != "approved":
        approve_user(user["id"])
    r = client.post("/login", data={"email": "t@t.com", "password": "pass123"})
    if r.status_code != 303:
        return
    r = client.get("/chat")
    assert r.status_code == 200
    assert "<textarea" in r.text
    assert "sendBeacon" in r.text
    assert "userScrolled" in r.text
```

- [ ] **Step 2: Verifica che il test fallisce**

```bash
cd ~/projects/Test_1 && python -m pytest tests/test_routes.py::test_chat_has_textarea -v
```
Expected: FAIL

- [ ] **Step 3: Sostituisci `web/templates/chat.html` con questo contenuto**

```html
{% extends "base.html" %}
{% block nav_right %}
  <div style="display:flex;gap:2rem;align-items:center;">
    <a href="/chat/storia" class="nav-link">Storico</a>
    <span style="font-size:0.6rem;color:var(--label);letter-spacing:0.1em;text-transform:uppercase;opacity:0.5;">claude-sonnet</span>
    <a href="/logout" class="nav-link">Esci</a>
  </div>
{% endblock %}
{% block content %}
<style>
  #chat-wrap { display:flex; flex-direction:column; height:calc(100vh - 48px); }
  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 2rem;
    display: flex;
    flex-direction: column;
    gap: 2rem;
    scroll-behavior: smooth;
  }
  .msg { display:flex; flex-direction:column; gap:0.5rem; }
  .msg.user { align-items: flex-end; }
  .msg.ai { align-items: flex-start; max-width: 78%; }
  .msg-label {
    font-size: 0.58rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--label);
  }
  .msg-bubble {
    font-size: 0.88rem;
    line-height: 1.85;
    color: var(--text);
  }
  .msg.user .msg-bubble {
    background: var(--border);
    padding: 0.7rem 1.1rem;
    border-radius: 2px 12px 12px 12px;
    max-width: 70%;
    white-space: pre-wrap;
  }
  .msg.ai .msg-bubble { max-width: 100%; }

  /* Input area */
  #input-area {
    border-top: 1px solid var(--border);
    padding: 1rem 2rem 1.4rem;
    background: var(--bg);
  }
  #input-row {
    display: flex;
    align-items: flex-end;
    gap: 1rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.8rem 1.2rem;
  }
  #input-row:focus-within { border-color: var(--faint); }
  #chat-input {
    flex: 1;
    background: transparent;
    border: none;
    font-size: 0.9rem;
    color: var(--text);
    font-family: inherit;
    outline: none;
    resize: none;
    line-height: 1.65;
    min-height: 24px;
    max-height: 180px;
    overflow-y: auto;
  }
  #chat-input::placeholder { color: var(--label); }
  #attach-btn {
    background: none;
    border: none;
    cursor: pointer;
    color: var(--label);
    padding: 2px 4px;
    font-size: 1rem;
    line-height: 1;
    transition: color 0.2s;
    flex-shrink: 0;
  }
  #attach-btn:hover { color: var(--text); }
  #send-btn {
    background: none;
    border: none;
    cursor: pointer;
    color: var(--text);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: inherit;
    padding: 2px 0;
    transition: opacity 0.2s;
    flex-shrink: 0;
  }
  #send-btn:hover { opacity: 0.6; }
  #attach-preview {
    font-size: 0.7rem;
    color: var(--label);
    margin-bottom: 0.5rem;
    display: none;
  }
  #attach-preview.visible { display: block; }

  /* Scroll-to-bottom button */
  #scroll-btn {
    position: fixed;
    bottom: 100px;
    right: 2rem;
    background: var(--border);
    border: 1px solid var(--faint);
    color: var(--text);
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.5rem 0.8rem;
    cursor: pointer;
    display: none;
    font-family: inherit;
    border-radius: 2px;
    transition: opacity 0.2s;
  }
  #scroll-btn.visible { display: block; }
</style>

<div id="chat-wrap">
  <div id="messages"></div>
  <div id="input-area">
    <div id="attach-preview"></div>
    <div id="input-row">
      <button id="attach-btn" title="Allega file" onclick="document.getElementById('file-input').click()">⌂</button>
      <input type="file" id="file-input" style="display:none"
        accept=".pdf,.txt,.md,.jpg,.jpeg,.png,.webp,.gif">
      <textarea id="chat-input" placeholder="Scrivi un messaggio..." rows="1"></textarea>
      <button id="send-btn" onclick="sendMessage()">Invia →</button>
    </div>
  </div>
</div>
<button id="scroll-btn" onclick="scrollToBottom(true)">↓ In fondo</button>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
const SESSION_ID = {{ session_id }};
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('chat-input');
const attachPreview = document.getElementById('attach-preview');
const scrollBtn = document.getElementById('scroll-btn');
const fileInput = document.getElementById('file-input');

let conversation = [];
let attachment = null; // { type: 'image_url'|'text', ... }
let userScrolled = false;

// Auto-resize textarea
inputEl.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 180) + 'px';
});

// Enter to send, Shift+Enter for newline
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Scroll tracking
messagesEl.addEventListener('scroll', () => {
  const atBottom = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 60;
  userScrolled = !atBottom;
  scrollBtn.classList.toggle('visible', userScrolled);
});

function scrollToBottom(force) {
  if (force || !userScrolled) {
    messagesEl.scrollTop = messagesEl.scrollHeight;
    userScrolled = false;
    scrollBtn.classList.remove('visible');
  }
}

// File attachment
fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file) return;
  attachPreview.textContent = 'Caricamento…';
  attachPreview.classList.add('visible');

  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch('/chat/upload', { method: 'POST', body: formData });
  if (!res.ok) {
    attachPreview.textContent = 'Errore nel caricamento.';
    return;
  }
  attachment = await res.json();
  attachPreview.textContent = `✓ ${file.name} allegato`;
  fileInput.value = '';
});

function addMessage(role, content) {
  const msg = document.createElement('div');
  msg.className = 'msg ' + (role === 'user' ? 'user' : 'ai');

  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = role === 'user' ? 'Tu' : 'Filmmaker';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble' + (role === 'ai' ? ' md' : '');

  if (typeof content === 'string') {
    bubble.textContent = content;
  }

  msg.appendChild(label);
  msg.appendChild(bubble);
  messagesEl.appendChild(msg);
  scrollToBottom();
  return bubble;
}

function renderMarkdown(el, text) {
  el.innerHTML = marked.parse(text);
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text && !attachment) return;

  inputEl.value = '';
  inputEl.style.height = 'auto';

  // Costruisci il contenuto del messaggio utente
  let userContent;
  if (attachment && attachment.type === 'image_url') {
    userContent = [
      { type: 'text', text: text || 'Analizza questa immagine.' },
      { type: 'image_url', image_url: { url: attachment.url } }
    ];
  } else if (attachment && attachment.type === 'text') {
    userContent = (text ? text + '\n\n' : '') + '[Documento allegato]\n' + attachment.content;
  } else {
    userContent = text;
  }

  // Mostra messaggio utente
  const displayText = text || (attachment ? '[file allegato]' : '');
  const userBubble = addMessage('user', displayText);

  // Reset allegato
  attachment = null;
  attachPreview.textContent = '';
  attachPreview.classList.remove('visible');

  conversation.push({ role: 'user', content: userContent });

  // Risposta AI
  const aiBubble = addMessage('ai', '');
  let full = '';

  const res = await fetch('/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conversation })
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    for (const line of chunk.split('\n')) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') break;
        full += data;
        aiBubble.textContent = full; // plain text during streaming
        scrollToBottom();
      }
    }
  }

  // Render markdown dopo completamento
  renderMarkdown(aiBubble, full);
  conversation.push({ role: 'assistant', content: full });
}

window.addEventListener('beforeunload', () => {
  if (conversation.length === 0) return;
  navigator.sendBeacon('/chat/end-session', new Blob(
    [JSON.stringify({ session_id: SESSION_ID, conversation: conversation })],
    { type: 'application/json' }
  ));
});
</script>
{% endblock %}
```

- [ ] **Step 4: Esegui il test**

```bash
cd ~/projects/Test_1 && python -m pytest tests/test_routes.py::test_chat_has_textarea -v
```
Expected: PASS

- [ ] **Step 5: Esegui tutti i test**

```bash
cd ~/projects/Test_1 && python -m pytest tests/ -v
```
Expected: tutti passano

- [ ] **Step 6: Commit**

```bash
cd ~/projects/Test_1 && git add web/templates/chat.html tests/test_routes.py && git commit -m "feat: chat textarea ChatGPT-style, auto-scroll intelligente, markdown rendering"
```

---

## Task 3: File Upload

**Files:**
- Modify: `web/app.py` — nuovo endpoint POST /chat/upload, fix RAG per contenuto multimodale
- Modify: `web/db.py` — nuova funzione get_session_messages

- [ ] **Step 1: Scrivi il test per l'upload**

Aggiungi alla fine di `tests/test_routes.py`:

```python
def test_upload_requires_auth():
    import io
    data = {"file": ("test.txt", io.BytesIO(b"contenuto test"), "text/plain")}
    r = client.post("/chat/upload", files=data)
    assert r.status_code == 401

def test_upload_text_file():
    from web.db import get_user_by_email
    user = get_user_by_email("t@t.com")
    if not user:
        return
    r_login = client.post("/login", data={"email": "t@t.com", "password": "pass123"})
    if r_login.status_code != 303:
        return
    import io
    data = {"file": ("test.txt", io.BytesIO(b"Questo e un documento di test"), "text/plain")}
    r = client.post("/chat/upload", files=data)
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "text"
    assert "documento" in body["content"]
```

- [ ] **Step 2: Verifica che i test falliscono**

```bash
cd ~/projects/Test_1 && python -m pytest tests/test_routes.py::test_upload_requires_auth tests/test_routes.py::test_upload_text_file -v
```
Expected: FAIL

- [ ] **Step 3: Aggiungi `get_session_messages` a `web/db.py`**

Aggiungi in fondo a `web/db.py`:

```python
def get_session_messages(session_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]
```

- [ ] **Step 4: Aggiungi gli import necessari in `web/app.py`**

In cima a `web/app.py`, assicurati che questi import esistano (aggiungili se mancano):

```python
from fastapi import FastAPI, Request, Form, Cookie, Response, UploadFile, File
```

Aggiungi anche `get_session_messages` agli import da `web.db`:
```python
from web.db import (
    init_db, create_user, get_user_by_email, get_user_by_id,
    list_pending, approve_user, update_user_password,
    start_session, end_session, save_messages, get_session,
    upsert_profile, get_profile,
    get_global_stats, list_all_users_with_stats, get_user_sessions,
    get_user_messages, get_session_messages,
)
```

- [ ] **Step 5: Aggiungi l'endpoint `POST /chat/upload` in `web/app.py`**

Aggiungi subito prima della sezione `# --- Admin ---`:

```python
@app.post("/chat/upload")
async def chat_upload(
    request: Request,
    file: UploadFile = File(...),
    session: Optional[str] = Cookie(default=None)
):
    user = get_current_user(session)
    if not user:
        return Response(status_code=401)

    content_type = file.content_type or ""
    data = await file.read()

    # Immagini → base64 per vision model
    if content_type.startswith("image/"):
        import base64
        b64 = base64.b64encode(data).decode()
        return {"type": "image_url", "url": f"data:{content_type};base64,{b64}"}

    # PDF → estrazione testo con pypdf
    if content_type == "application/pdf" or file.filename.endswith(".pdf"):
        import io
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return {"type": "text", "content": text[:20000]}  # cap a 20k caratteri
        except Exception:
            return Response(status_code=422, content="PDF non leggibile")

    # Testo plain / markdown
    try:
        text = data.decode("utf-8", errors="replace")
        return {"type": "text", "content": text[:20000]}
    except Exception:
        return Response(status_code=422, content="Formato non supportato")
```

- [ ] **Step 6: Correggi il RAG in `chat_stream` per gestire contenuto multimodale**

In `web/app.py`, nella route `chat_stream`, trova:
```python
    last_user = next((m["content"] for m in reversed(conversation) if m["role"] == "user"), "")
```

Sostituisci con:
```python
    def _extract_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        return ""

    last_user = next((_extract_text(m["content"]) for m in reversed(conversation) if m["role"] == "user"), "")
```

- [ ] **Step 7: Esegui i test**

```bash
cd ~/projects/Test_1 && python -m pytest tests/ -v
```
Expected: tutti i test passano

- [ ] **Step 8: Commit**

```bash
cd ~/projects/Test_1 && git add web/app.py web/db.py tests/test_routes.py && git commit -m "feat: upload file (immagini/PDF/testo), fix RAG per contenuto multimodale"
```

---

## Task 4: Storico Conversazioni

**Files:**
- Modify: `web/app.py` — nuove route GET /chat/storia e GET /chat/storia/{session_id}
- Create: `web/templates/history.html`
- Create: `web/templates/history_session.html`

- [ ] **Step 1: Scrivi il test per le nuove route storico**

Aggiungi alla fine di `tests/test_routes.py`:

```python
def test_history_requires_auth():
    r = client.get("/chat/storia")
    assert r.status_code == 303
    assert r.headers["location"] == "/login"

def test_history_loads_for_authed_user():
    from web.db import get_user_by_email
    user = get_user_by_email("t@t.com")
    if not user:
        return
    r_login = client.post("/login", data={"email": "t@t.com", "password": "pass123"})
    if r_login.status_code != 303:
        return
    r = client.get("/chat/storia")
    assert r.status_code == 200
    assert "storico" in r.text.lower() or "sessioni" in r.text.lower() or "filmmaker" in r.text.lower()
```

- [ ] **Step 2: Verifica che i test falliscono**

```bash
cd ~/projects/Test_1 && python -m pytest tests/test_routes.py::test_history_requires_auth tests/test_routes.py::test_history_loads_for_authed_user -v
```
Expected: FAIL — route non esiste

- [ ] **Step 3: Aggiungi le route in `web/app.py`**

Aggiungi subito dopo la route `chat_end_session` e prima della sezione `# --- Admin ---`:

```python
@app.get("/chat/storia", response_class=HTMLResponse)
def chat_history(request: Request, session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    sessions = get_user_sessions(user["id"])
    # Filtra solo sessioni concluse con almeno un messaggio
    sessions = [s for s in sessions if s["ended_at"] and s["message_count"] > 0]
    return templates.TemplateResponse(request, "history.html", {
        "user": user,
        "sessions": sessions,
    })

@app.get("/chat/storia/{session_id}", response_class=HTMLResponse)
def chat_history_session(session_id: int, request: Request, session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    # Verifica che la sessione appartenga all'utente
    sess = get_session(session_id)
    if not sess or sess["user_id"] != user["id"]:
        return RedirectResponse("/chat/storia", status_code=303)
    msgs = get_session_messages(session_id)
    return templates.TemplateResponse(request, "history_session.html", {
        "user": user,
        "sess": sess,
        "messages": msgs,
    })
```

- [ ] **Step 4: Crea `web/templates/history.html`**

```html
{% extends "base.html" %}
{% block nav_right %}
  <div style="display:flex;gap:2rem;align-items:center;">
    <a href="/chat" class="nav-link">← Chat</a>
    <a href="/logout" class="nav-link">Esci</a>
  </div>
{% endblock %}
{% block content %}
<div style="padding:2.5rem 2rem;max-width:680px;margin:0 auto;">

  <div style="font-size:1.6rem;font-family:'Times New Roman',serif;font-style:italic;color:var(--text);margin-bottom:0.3rem;">
    Le tue conversazioni.
  </div>
  <div style="width:28px;height:1px;background:var(--faint);margin:1.5rem 0 2.5rem;"></div>

  {% if sessions %}
  {% for s in sessions %}
  <a href="/chat/storia/{{ s.id }}" style="text-decoration:none;display:block;">
    <div style="padding:1.2rem 0;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:baseline;transition:opacity 0.15s;" onmouseover="this.style.opacity='0.7'" onmouseout="this.style.opacity='1'">
      <div>
        <div style="font-size:0.88rem;color:var(--text);font-family:'Times New Roman',serif;font-style:italic;">
          {{ s.started_at[:10] }}
          <span style="font-size:0.7rem;color:var(--label);font-style:normal;margin-left:0.5rem;">{{ s.started_at[11:16] }}</span>
        </div>
        <div style="font-size:0.7rem;color:var(--label);margin-top:0.3rem;letter-spacing:0.02em;">
          {{ s.message_count }} messaggi
        </div>
      </div>
      <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--label);">Leggi →</div>
    </div>
  </a>
  {% endfor %}
  {% else %}
  <div class="subtitle">Nessuna conversazione salvata ancora.</div>
  {% endif %}

</div>
{% endblock %}
```

- [ ] **Step 5: Crea `web/templates/history_session.html`**

```html
{% extends "base.html" %}
{% block nav_right %}
  <div style="display:flex;gap:2rem;align-items:center;">
    <a href="/chat/storia" class="nav-link">← Storico</a>
    <a href="/chat" class="nav-link">Chat</a>
    <a href="/logout" class="nav-link">Esci</a>
  </div>
{% endblock %}
{% block content %}
<style>
  .h-msg { display:flex; flex-direction:column; gap:0.4rem; margin-bottom:2rem; }
  .h-msg.user { align-items: flex-end; }
  .h-msg.ai { align-items: flex-start; max-width: 78%; }
  .h-label { font-size:0.58rem; letter-spacing:0.18em; text-transform:uppercase; color:var(--label); }
  .h-bubble { font-size:0.88rem; line-height:1.85; color:var(--text); }
  .h-msg.user .h-bubble {
    background: var(--border);
    padding: 0.7rem 1.1rem;
    border-radius: 2px 12px 12px 12px;
    max-width: 70%;
    white-space: pre-wrap;
  }
</style>

<div style="padding:2.5rem 2rem;max-width:680px;margin:0 auto;">

  <div style="font-size:1.1rem;font-family:'Times New Roman',serif;font-style:italic;color:var(--text);margin-bottom:0.2rem;">
    {{ sess.started_at[:10] }} — {{ sess.started_at[11:16] }}
  </div>
  <div style="font-size:0.7rem;color:var(--label);margin-bottom:0.3rem;">{{ sess.message_count }} messaggi</div>
  <div style="width:28px;height:1px;background:var(--faint);margin:1.5rem 0 2.5rem;"></div>

  {% for m in messages %}
  <div class="h-msg {{ 'user' if m.role == 'user' else 'ai' }}">
    <div class="h-label">{{ 'Tu' if m.role == 'user' else 'Filmmaker' }}</div>
    <div class="h-bubble {{ 'md' if m.role == 'assistant' else '' }}" data-content="{{ m.content | e }}">
      {{ m.content }}
    </div>
  </div>
  {% endfor %}

</div>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
// Render markdown per messaggi AI
document.querySelectorAll('.h-bubble.md').forEach(el => {
  const raw = el.getAttribute('data-content');
  if (raw) el.innerHTML = marked.parse(raw);
});
</script>
{% endblock %}
```

- [ ] **Step 6: Esegui tutti i test**

```bash
cd ~/projects/Test_1 && python -m pytest tests/ -v
```
Expected: tutti i test passano

- [ ] **Step 7: Commit e deploy**

```bash
cd ~/projects/Test_1 && git add web/app.py web/db.py web/templates/history.html web/templates/history_session.html tests/test_routes.py && git commit -m "feat: storico conversazioni con rendering markdown"
git push && railway up --detach
```

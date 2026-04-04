# AI Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migliorare la qualità dell'AI su 5 fronti: system prompt, RAG semantico, profilo JSON strutturato, summarizzazione sessione, e model switcher.

**Architecture:** Ogni task è indipendente e deployabile separatamente. Il RAG semantico sostituisce BM25 con sentence-transformers (multilingual, zero costo aggiuntivo). Il profilo passa da testo libero a JSON strutturato. La summarizzazione riduce il context window per sessioni lunghe. Il model switcher aggiunge un toggle sonnet/opus nella chat.

**Tech Stack:** FastAPI, SQLite, sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2), OpenRouter (claude-sonnet-4-6 / claude-opus-4-6), Python 3.13

---

## Task 1: Miglioramento System Prompt

**Files:**
- Modify: `cineauteur.py` (SYSTEM_PROMPT, righe 107-169)

- [ ] **Step 1: Sostituisci il SYSTEM_PROMPT in cineauteur.py**

Sostituisci l'intero blocco `SYSTEM_PROMPT = """\..."""` con:

```python
SYSTEM_PROMPT = """\
Sei un consulente creativo per lo sviluppo di film. Il tuo compito è aiutare l'autore a costruire il progetto più vero che è capace di fare — non quello più corretto, non quello più vendibile, quello più suo.

---

REGOLA PRINCIPALE: UNA DOMANDA ALLA VOLTA.

Non fare mai più di una domanda per messaggio. Se hai dubbi su più cose, scegli quella che sblocca il passo successivo. Quando fai una domanda, rendila concreta: non "cosa vuoi raccontare?" ma "descrivi la scena più importante del film in tre righe — cosa si vede, chi c'è, cosa succede."

---

QUANDO RICEVI MATERIALE GREZZO (appunti, file, testi disorganizzati):

Non fare domande subito. Prima restituisci quello che vedi:

1. Identifica l'idea con più energia — quella che torna, quella che ha urgenza.
2. Identifica i fili ricorrenti — immagini, ossessioni, temi che appaiono in forme diverse.
3. Identifica la coerenza nascosta — anche negli appunti più caotici c'è una logica interna.

Poi presenta quello che hai trovato con chiarezza. Solo dopo, se serve, fai UNA domanda per avanzare nella direzione più promettente.

---

COME FAR AVANZARE IL PROGETTO:

Ogni tua risposta deve spostare qualcosa. Usa questa bussola — non come protocollo rigido, ma per capire dove sei:

SPECIFICITÀ → cosa si vede esattamente? Che corpo, che luce, che suono, che spazio?
SCELTE CONCRETE → camera fissa o in movimento? Durata del piano? Distanza dal soggetto?
TENSIONE → cosa cambia? Cosa si accumula? Cosa disturba?
NECESSITÀ → perché questo film deve esistere oggi? Cosa lo rende impossibile da non fare?

Quando qualcosa è vago, chiedi di renderlo visibile: "Mostrami questa scena come se la stessi descrivendo a un direttore della fotografia."

---

RIFERIMENTI A FILM E REGISTI:

Non citare mai film o registi come prima risposta. Usali — con parsimonia, 1-2 al massimo — solo quando stai già lavorando su qualcosa di specifico e il riferimento aiuta a chiarire una direzione concreta, non a validare un'idea. Esempio corretto: "Quello che descrivi — il silenzio tra i due personaggi, la distanza fisica — ricorda come Haneke usa il fuori campo in Niente da nascondere. Ha senso per te, o vuoi costruire una distanza diversa?"

---

QUANDO QUALCOSA NON FUNZIONA:

Dillo chiaramente, senza ammorbidire. Poi proponi subito come rafforzarlo. Non lasciare l'autore con un problema senza una direzione. Esempio: "Questo personaggio non ha ancora una necessità interna — sappiamo cosa fa ma non perché non possa fare altrimenti. Come diventerebbe impossibile per lui non farlo?"

---

DIVIETI ASSOLUTI:

- Mai elenchi di domande
- Mai linguaggio accademico (dispositivo, sguardo, postura etica, diegesi, soggettività)
- Mai "questa cosa esiste già" come prima reazione
- Mai rendere l'idea più giusta — renderla più vera
- Mai rispondere solo con elogi senza far avanzare il lavoro

---

OBIETTIVO DI OGNI CONVERSAZIONE:

L'autore deve uscire con un passo concreto in avanti — una scena da scrivere, una scelta da prendere, una domanda a cui sa già rispondere ma non aveva ancora formulato. Non con più domande di quante ne aveva all'inizio.

Rispondi sempre in italiano.

---

SOMIGLIANZE CON OPERE ESISTENTI:
Se emerge una somiglianza forte, segnalala — solo dopo aver lavorato sull'idea, come osservazione neutra: "Quello che descrivi ha qualcosa di [film/autore] — in particolare [aspetto]. Vale la pena saperlo. Come vuoi procedere?"

DISTRIBUZIONE E MERCATO:
Solo se richiesto esplicitamente. Sii onesto anche quando le prospettive sono difficili.
"""
```

- [ ] **Step 2: Commit**

```bash
cd ~/projects/Test_1
git add cineauteur.py
git commit -m "feat: improve system prompt — concrete questions, visual anchoring, stronger directives"
```

---

## Task 2: RAG Semantico con Embeddings

Sostituisce BM25 con ricerca semantica usando `sentence-transformers`. Gli embedding vengono calcolati una volta all'avvio e salvati su disco. Fallback a BM25 se sentence-transformers non è disponibile.

**Files:**
- Modify: `requirements.txt`
- Modify: `web/rag.py`
- Modify: `web/app.py` (lifespan + chat/stream)

- [ ] **Step 1: Aggiungi dipendenza a requirements.txt**

Aggiungi in fondo a `requirements.txt`:
```
sentence-transformers>=3.0.0
```

- [ ] **Step 2: Riscrivi web/rag.py con supporto embedding**

Sostituisci l'intero contenuto di `web/rag.py`:

```python
from __future__ import annotations
import os
import pickle
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

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

DOCS_DIR      = Path(__file__).parent.parent / "docs"
CHUNK_SIZE    = 350
CHUNK_OVERLAP = 60
TOP_K         = 6
MIN_SCORE     = 0.1
EMBED_CACHE   = Path(os.environ.get("DATABASE_URL", "moviemaker.db")).parent / "embeddings.pkl"
MODEL_CACHE   = Path(os.environ.get("DATABASE_URL", "moviemaker.db")).parent / "hf_cache"
EMBED_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"

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
    """Build BM25 index (fallback)."""
    if not BM25_AVAILABLE or not chunks:
        return None
    return BM25Okapi([c.lower().split() for c in chunks])

def build_embedding_index(chunks: list[str]) -> dict | None:
    """Build or load semantic embedding index. Returns dict with model + embeddings."""
    if not EMBEDDINGS_AVAILABLE or not chunks:
        return None
    os.makedirs(MODEL_CACHE, exist_ok=True)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODEL_CACHE)
    # Load from cache if exists and chunk count matches
    if EMBED_CACHE.exists():
        try:
            with open(EMBED_CACHE, "rb") as f:
                cached = pickle.load(f)
            if cached.get("chunk_count") == len(chunks):
                print(f"Embedding index caricato da cache ({len(chunks)} chunk)")
                # Ricarica il modello in memoria (non serializzato nel pickle)
                cached["model_obj"] = SentenceTransformer(EMBED_MODEL)
                return cached
        except Exception:
            pass
    print(f"Generazione embedding per {len(chunks)} chunk (prima volta, può richiedere qualche minuto)...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
    data = {"model": EMBED_MODEL, "model_obj": model, "embeddings": embeddings, "chunk_count": len(chunks)}
    try:
        with open(EMBED_CACHE, "wb") as f:
            pickle.dump(data, f)
        print("Embedding index salvato in cache.")
    except Exception as e:
        print(f"Impossibile salvare embedding cache: {e}")
    return data

def retrieve(query: str, index, chunks: list[str], sources: list[str],
             embed_index: dict | None = None) -> str:
    """Retrieve relevant chunks. Uses semantic search if available, else BM25."""
    if not chunks:
        return ""

    if embed_index is not None and EMBEDDINGS_AVAILABLE:
        return _retrieve_semantic(query, chunks, sources, embed_index)

    if index is None:
        return ""
    return _retrieve_bm25(query, index, chunks, sources)

def _retrieve_semantic(query: str, chunks: list[str], sources: list[str], embed_index: dict) -> str:
    import numpy as np
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODEL_CACHE)
    # Usa il modello già caricato in memoria se disponibile
    model = embed_index.get("model_obj") or SentenceTransformer(EMBED_MODEL)
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    embeddings = embed_index["embeddings"]
    scores = embeddings @ query_vec  # cosine similarity (vectors are normalized)
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

def _retrieve_bm25(query: str, index, chunks: list[str], sources: list[str]) -> str:
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

- [ ] **Step 3: Aggiorna web/app.py — lifespan per caricare embedding index**

Nel lifespan di `app.py`, aggiorna le variabili globali e il caricamento:

```python
# In cima al file, aggiungi dopo le altre variabili globali:
corpus_chunks: list = []
corpus_sources: list = []
bm25 = None
embed_index = None  # NUOVO
```

Nel lifespan, dopo `bm25 = build_index(corpus_chunks)`:
```python
    global embed_index
    from web.rag import build_embedding_index
    embed_index = build_embedding_index(corpus_chunks)
    if embed_index:
        print("RAG semantico attivo (sentence-transformers)")
    else:
        print("RAG BM25 attivo (fallback)")
```

- [ ] **Step 4: Aggiorna la chiamata retrieve in /chat/stream**

In `app.py`, nella route `/chat/stream`, sostituisci:
```python
rag_ctx = retrieve(last_user, bm25, corpus_chunks, corpus_sources)
```
con:
```python
rag_ctx = retrieve(last_user, bm25, corpus_chunks, corpus_sources, embed_index=embed_index)
```

- [ ] **Step 5: Commit**

```bash
cd ~/projects/Test_1
git add requirements.txt web/rag.py web/app.py
git commit -m "feat: semantic RAG with sentence-transformers, BM25 fallback"
```

---

## Task 3: Profilo Autore come JSON Strutturato

Il profilo passa da stringa di testo libero a JSON con campi fissi. Migrazione non-distruttiva: profili esistenti vengono wrappati in un campo `legacy`.

**Files:**
- Modify: `web/app.py` (profile generation prompt + injection)
- Modify: `web/db.py` (get_profile restituisce dict)

- [ ] **Step 1: Aggiorna get_profile in db.py**

Sostituisci la funzione `get_profile`:
```python
def get_profile(user_id: int) -> dict | None:
    """Restituisce il profilo come dict. None se non esiste."""
    import json
    with get_conn() as conn:
        row = conn.execute(
            "SELECT profile_text FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row or not row["profile_text"]:
            return None
        try:
            return json.loads(row["profile_text"])
        except (json.JSONDecodeError, TypeError):
            # Profilo legacy in formato testo: wrappalo
            return {"legacy": row["profile_text"]}
```

- [ ] **Step 2: Aggiorna il prompt di generazione profilo in app.py**

In `/chat/end-session`, sostituisci il `profile_prompt`:

```python
    existing_profile = get_profile(user["id"])
    import json

    profile_prompt = f"""Analizza questa conversazione tra un filmmaker e un AI coach e aggiorna il profilo dell'autore.

PROFILO ESISTENTE (JSON, None se prima sessione):
{json.dumps(existing_profile, ensure_ascii=False, indent=2) if existing_profile else "null"}

CONVERSAZIONE:
{chr(10).join(f"{m['role'].upper()}: {m['content']}" for m in conversation if isinstance(m.get('content'), str))}

Restituisci SOLO un oggetto JSON valido, senza testo prima o dopo, con questi campi esatti:

{{
  "progetti_attivi": ["lista dei progetti cinematografici in corso"],
  "temi_ricorrenti": ["temi, ossessioni, motivi che tornano nel lavoro"],
  "punti_di_forza": ["cosa sa fare bene come autore"],
  "aree_di_sviluppo": ["dove ha ancora margine di crescita"],
  "stile_preferito": "descrizione breve del suo approccio estetico/narrativo",
  "livello_tecnico": "principiante | intermedio | avanzato",
  "ultima_sessione": "sintesi in 2-3 frasi di questa conversazione",
  "progressi": "confronto con sessioni precedenti, o 'prima sessione' se profilo era null"
}}"""
```

Poi aggiorna il salvataggio del profilo generato:
```python
        new_profile_text = resp.choices[0].message.content.strip()
        # Verifica che sia JSON valido, altrimenti salva come legacy
        try:
            json.loads(new_profile_text)
        except json.JSONDecodeError:
            new_profile_text = json.dumps({"legacy": new_profile_text}, ensure_ascii=False)
        upsert_profile(user["id"], new_profile_text)
```

- [ ] **Step 3: Aggiorna l'injection del profilo nel system message**

In `/chat/stream`, sostituisci il blocco di costruzione del system message:

```python
    import json
    profile = get_profile(user["id"])
    system_content = SYSTEM_PROMPT
    if profile:
        if "legacy" in profile:
            profile_str = profile["legacy"]
        else:
            profile_str = f"""Progetti attivi: {', '.join(profile.get('progetti_attivi', [])) or 'nessuno'}
Temi ricorrenti: {', '.join(profile.get('temi_ricorrenti', [])) or 'nessuno'}
Punti di forza: {', '.join(profile.get('punti_di_forza', [])) or 'non ancora definiti'}
Aree di sviluppo: {', '.join(profile.get('aree_di_sviluppo', [])) or 'non ancora definite'}
Stile: {profile.get('stile_preferito', '')}
Livello tecnico: {profile.get('livello_tecnico', '')}
Ultima sessione: {profile.get('ultima_sessione', '')}"""
        system_content = f"{SYSTEM_PROMPT}\n\nPROFILO DELL'AUTORE:\n{profile_str}"
```

- [ ] **Step 4: Commit**

```bash
cd ~/projects/Test_1
git add web/app.py web/db.py
git commit -m "feat: structured JSON author profile with migration support"
```

---

## Task 4: Summarizzazione Sessione per Conversazioni Lunghe

Dopo 8 scambi (16 messaggi), i messaggi più vecchi vengono sostituiti con un riassunto. Riduce il context window e migliora il focus dell'AI.

**Files:**
- Modify: `web/app.py` (route /chat/stream)

- [ ] **Step 1: Aggiungi funzione di summarizzazione in app.py**

Aggiungi questa funzione prima delle routes, dopo la definizione di SYSTEM_PROMPT:

```python
async def _summarize_old_messages(messages: list[dict], keep_recent: int = 8) -> list[dict]:
    """
    Se la conversazione supera keep_recent*2 messaggi, riassume i più vecchi.
    Restituisce una conversazione abbreviata: [summary_msg] + ultimi keep_recent scambi.
    """
    if len(messages) <= keep_recent * 2:
        return messages

    from openai import AsyncOpenAI
    old_msgs = messages[:-keep_recent * 2]
    recent_msgs = messages[-keep_recent * 2:]

    old_text = "\n".join(
        f"{m['role'].upper()}: {m['content'] if isinstance(m['content'], str) else '[file allegato]'}"
        for m in old_msgs
    )
    summary_prompt = f"""Riassumi questa parte di conversazione tra un filmmaker e un AI coach in 3-5 frasi. 
Includi: i progetti discussi, le decisioni prese, i punti chiave emersi. 
Sii concreto e preciso — questo riassunto sostituirà la storia della conversazione per continuare il lavoro.

CONVERSAZIONE DA RIASSUMERE:
{old_text}"""

    try:
        client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=300,
            extra_headers={"HTTP-Referer": "https://moviemaker.io", "X-Title": "Filmmaker"},
        )
        summary_text = resp.choices[0].message.content.strip()
        summary_msg = {
            "role": "assistant",
            "content": f"[Riassunto conversazione precedente]\n{summary_text}"
        }
        return [summary_msg] + recent_msgs
    except Exception as e:
        print(f"Errore summarizzazione: {e}")
        return messages  # fallback: usa conversazione completa
```

- [ ] **Step 2: Usa la summarizzazione in /chat/stream**

In `/chat/stream`, prima di costruire `msgs`, aggiungi:

```python
    # Comprimi conversazioni lunghe
    conversation = await _summarize_old_messages(conversation, keep_recent=8)

    profile = get_profile(user["id"])
    # ... resto del codice invariato
```

- [ ] **Step 3: Commit**

```bash
cd ~/projects/Test_1
git add web/app.py
git commit -m "feat: session summarization for long conversations (>16 messages)"
```

---

## Task 5: Model Switcher (Sonnet / Opus)

Aggiunge un toggle nella chat per scegliere tra claude-sonnet-4-6 (veloce, economico) e claude-opus-4-6 (più lento, migliore per progetti complessi). La scelta viene salvata nel profilo utente e persiste tra le sessioni.

**Files:**
- Modify: `web/db.py` (aggiungi colonna preferred_model)
- Modify: `web/app.py` (leggi modello preferito, nuova route per cambiarlo)
- Modify: `web/templates/chat.html` (toggle UI)

- [ ] **Step 1: Aggiungi colonna preferred_model in db.py**

In `init_db()`, dopo l'ultimo `try/except` per le migrazioni:
```python
        try:
            conn.execute("ALTER TABLE user_profiles ADD COLUMN preferred_model TEXT DEFAULT 'sonnet'")
        except Exception:
            pass
```

Aggiungi questa funzione in db.py:
```python
def set_preferred_model(user_id: int, model: str):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO user_profiles (user_id, profile_text, preferred_model)
            VALUES (?, '', ?)
            ON CONFLICT(user_id) DO UPDATE SET preferred_model = excluded.preferred_model
        """, (user_id, model))

def get_preferred_model(user_id: int) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT preferred_model FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row and row["preferred_model"]:
            return row["preferred_model"]
        return "sonnet"
```

- [ ] **Step 2: Aggiungi route per cambiare modello in app.py**

Aggiungi import in cima al file (nella sezione imports da db):
```python
from web.db import (
    ...  # aggiungi:
    set_preferred_model, get_preferred_model,
)
```

Aggiungi route dopo le routes della chat:
```python
@app.post("/chat/set-model")
async def set_model(request: Request, session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return Response(status_code=401)
    body = await request.json()
    model_choice = body.get("model", "sonnet")
    if model_choice not in ("sonnet", "opus"):
        return Response(status_code=400)
    set_preferred_model(user["id"], model_choice)
    return Response(status_code=200)
```

- [ ] **Step 3: Usa il modello preferito in /chat/stream e /chat/end-session**

In `app.py`, in cima a `/chat/stream`, dopo `user = get_current_user(session)`:
```python
    preferred = get_preferred_model(user["id"])
    active_model = "anthropic/claude-opus-4-6" if preferred == "opus" else "anthropic/claude-sonnet-4-6"
```

Poi in `generate()`, sostituisci `model=MODEL` con `model=active_model`.

Fai lo stesso nel generate_welcome() e nell'end-session (usa sempre `active_model`).

- [ ] **Step 4: Aggiungi toggle nella chat UI (chat.html)**

Nel `{% block nav_right %}` di chat.html, sostituisci:
```html
<span class="nav-badge">claude opus</span>
```
con:
```html
<button id="model-toggle" onclick="toggleModel()" class="nav-badge" style="cursor:pointer;border:none;background:none;font-family:var(--font);">
  claude <span id="model-name">sonnet</span>
</button>
```

Aggiungi questo JavaScript prima della chiusura `</script>`:
```javascript
// Model switcher
let currentModel = 'sonnet';

async function toggleModel() {
  currentModel = currentModel === 'sonnet' ? 'opus' : 'sonnet';
  document.getElementById('model-name').textContent = currentModel;
  await fetch('/chat/set-model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: currentModel })
  });
}

// Carica preferenza dal server all'avvio
(async () => {
  // Il modello preferito viene passato dal template
  currentModel = '{{ preferred_model }}';
  document.getElementById('model-name').textContent = currentModel;
})();
```

- [ ] **Step 5: Passa preferred_model al template in app.py**

In `/chat` GET route, aggiungi `preferred_model` al context:
```python
    return templates.TemplateResponse(request, "chat.html", {
        "user": user,
        "session_id": session_id,
        "profile": profile,
        "is_first_session": profile is None,
        "preferred_model": get_preferred_model(user["id"]),  # NUOVO
    })
```

- [ ] **Step 6: Commit**

```bash
cd ~/projects/Test_1
git add web/app.py web/db.py web/templates/chat.html
git commit -m "feat: model switcher — sonnet/opus toggle in chat UI"
```

---

## Deploy finale

- [ ] **Push e deploy su Railway**

```bash
cd ~/projects/Test_1
git push
railway up --detach
```

Attendi il completamento del build. Al primo avvio con sentence-transformers, il download del modello (~270MB) richiede 2-3 minuti. I log mostreranno:
```
Generazione embedding per N chunk (prima volta, può richiedere qualche minuto)...
Embedding index salvato in cache.
RAG semantico attivo (sentence-transformers)
```

Nelle esecuzioni successive:
```
Embedding index caricato da cache (N chunk)
RAG semantico attivo (sentence-transformers)
```

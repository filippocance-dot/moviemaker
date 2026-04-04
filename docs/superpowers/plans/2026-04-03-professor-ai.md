# Professor AI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone FastAPI web app at `/Users/filippoarici/projects/Professor/` with a dual-panel interface — left for working with Claude, right for a professor AI that explains what's happening and remembers what it has already taught.

**Architecture:** Single FastAPI app, SQLite on Railway `/data` volume, two Claude Opus instances per session (left=work, right=professor), BM25 RAG on 13 corpus files, learning profile updated after every session via beacon. Single-password access (no registration).

**Tech Stack:** Python 3.11+, FastAPI, Jinja2, SQLite, openai SDK (OpenRouter), rank_bm25, itsdangerous, bcrypt, Railway

---

## File Map

```
/Users/filippoarici/projects/Professor/
├── professor.py              # PROFESSOR_SYSTEM_PROMPT constant
├── requirements.txt
├── railway.toml
├── web/
│   ├── __init__.py           # empty
│   ├── app.py                # all FastAPI routes
│   ├── db.py                 # sqlite init + all db functions
│   ├── auth.py               # password check + session token
│   ├── rag.py                # BM25 corpus loader (copied pattern from FilmMaker)
│   └── templates/
│       ├── base.html         # CSS design system, nav
│       ├── login.html        # password form
│       ├── studio.html       # dual-panel main interface
│       ├── storia.html       # session history list
│       └── profilo.html      # learning profile viewer
└── docs/
    ├── come_funziona_claude/
    │   ├── context_window.txt
    │   ├── elaborazione_istruzioni.txt
    │   ├── limiti_allucinazioni.txt
    │   └── variabilita_risposte.txt
    ├── prompting/
    │   ├── struttura_prompt_efficace.txt
    │   ├── few_shot_prompting.txt
    │   ├── chain_of_thought.txt
    │   ├── system_prompt.txt
    │   └── iterazione.txt
    ├── costi_modelli/
    │   ├── opus_sonnet_haiku.txt
    │   ├── pricing_token.txt
    │   ├── prompt_caching.txt
    │   ├── batch_api.txt
    │   └── api_vs_interfaccia.txt
    └── apprendimento/
        ├── metodo_feynman.txt
        ├── modelli_mentali.txt
        ├── spaced_repetition.txt
        └── imitare_vs_capire.txt
```

---

## Task 1: Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `railway.toml`
- Create: `web/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
jinja2==3.1.4
python-multipart==0.0.9
openai==1.40.0
rank-bm25==0.2.2
bcrypt==4.1.3
itsdangerous==2.2.0
```

- [ ] **Step 2: Create railway.toml**

```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn web.app:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/"
healthcheckTimeout = 30
restartPolicyType = "on_failure"
```

- [ ] **Step 3: Create web/__init__.py (empty)**

```python
```

- [ ] **Step 4: Initialize git**

```bash
cd /Users/filippoarici/projects/Professor
git init
echo "__pycache__/" > .gitignore
echo "*.pyc" >> .gitignore
echo ".env" >> .gitignore
echo "*.db" >> .gitignore
git add requirements.txt railway.toml web/__init__.py .gitignore
git commit -m "chore: initial scaffolding"
```

---

## Task 2: Database module

**Files:**
- Create: `web/db.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_db.py
import os, sys, pytest
os.environ["DATABASE_URL"] = ":memory:"
sys.path.insert(0, "/Users/filippoarici/projects/Professor")
from web.db import init_db, start_session, end_session, save_message_left, save_message_right, get_messages_left, get_messages_right, get_session, get_learning_profile, update_learning_profile, list_sessions

def test_session_lifecycle():
    init_db()
    sid = start_session()
    assert sid > 0
    msgs = get_messages_left(sid)
    assert msgs == []
    save_message_left(sid, "user", "ciao")
    save_message_right(sid, "assistant", "risposta")
    assert len(get_messages_left(sid)) == 1
    assert len(get_messages_right(sid)) == 1
    end_session(sid, "riassunto test")
    sess = get_session(sid)
    assert sess["summary"] == "riassunto test"
    assert sess["ended_at"] is not None

def test_learning_profile():
    init_db()
    p = get_learning_profile()
    assert p is not None
    update_learning_profile({"prompting": "capisce few-shot"})
    p2 = get_learning_profile()
    assert p2["prompting"] == "capisce few-shot"

def test_list_sessions():
    init_db()
    sid = start_session()
    end_session(sid, "test")
    sessions = list_sessions()
    assert len(sessions) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/filippoarici/projects/Professor
python -m pytest tests/test_db.py -v 2>&1 | head -20
```
Expected: FAIL with "ModuleNotFoundError" or "No module named 'web.db'"

- [ ] **Step 3: Write web/db.py**

```python
from __future__ import annotations
import os, sqlite3
from contextlib import contextmanager

DATABASE_URL = os.environ.get("DATABASE_URL", "professor.db")

@contextmanager
def get_conn():
    db_url = os.environ.get("DATABASE_URL", DATABASE_URL)
    conn = sqlite3.connect(db_url)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL DEFAULT (datetime('now')),
                ended_at TEXT,
                summary TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages_left (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages_right (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_profile (
                id INTEGER PRIMARY KEY DEFAULT 1,
                come_funziona_claude TEXT NOT NULL DEFAULT '',
                prompting TEXT NOT NULL DEFAULT '',
                costi_modelli TEXT NOT NULL DEFAULT '',
                apprendimento TEXT NOT NULL DEFAULT '',
                pattern_negativi TEXT NOT NULL DEFAULT '',
                concetti_consolidati TEXT NOT NULL DEFAULT '',
                concetti_aperti TEXT NOT NULL DEFAULT '',
                ultima_sessione TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        # Ensure the single learning_profile row exists
        conn.execute("""
            INSERT OR IGNORE INTO learning_profile (id) VALUES (1)
        """)

def start_session() -> int:
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO sessions DEFAULT VALUES")
        return cur.lastrowid

def end_session(session_id: int, summary: str = "") -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = datetime('now'), summary = ? WHERE id = ?",
            (summary, session_id)
        )

def get_session(session_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

def list_sessions() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE ended_at IS NOT NULL ORDER BY started_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

def save_message_left(session_id: int, role: str, content: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages_left (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )

def save_message_right(session_id: int, role: str, content: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages_right (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )

def get_messages_left(session_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages_left WHERE session_id = ? ORDER BY id",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]

def get_messages_right(session_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages_right WHERE session_id = ? ORDER BY id",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]

def get_learning_profile() -> dict:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM learning_profile WHERE id = 1").fetchone()
        return dict(row) if row else {}

def update_learning_profile(fields: dict) -> None:
    """Update named fields on the single learning_profile row."""
    if not fields:
        return
    allowed = {
        "come_funziona_claude", "prompting", "costi_modelli", "apprendimento",
        "pattern_negativi", "concetti_consolidati", "concetti_aperti", "ultima_sessione"
    }
    safe = {k: v for k, v in fields.items() if k in allowed}
    if not safe:
        return
    set_clause = ", ".join(f"{k} = ?" for k in safe)
    set_clause += ", updated_at = datetime('now')"
    with get_conn() as conn:
        conn.execute(
            f"UPDATE learning_profile SET {set_clause} WHERE id = 1",
            list(safe.values())
        )
```

- [ ] **Step 4: Create tests directory and run tests**

```bash
cd /Users/filippoarici/projects/Professor
mkdir -p tests
touch tests/__init__.py
python -m pytest tests/test_db.py -v
```
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add web/db.py tests/
git commit -m "feat: database module with 4 tables and all CRUD functions"
```

---

## Task 3: Auth module

**Files:**
- Create: `web/auth.py`

- [ ] **Step 1: Write test**

```python
# tests/test_auth.py
import os, sys
os.environ.setdefault("ACCESS_PASSWORD", "testpass")
os.environ.setdefault("SECRET_KEY", "test-secret")
sys.path.insert(0, "/Users/filippoarici/projects/Professor")
from web.auth import check_password, make_token, decode_token

def test_password_check():
    assert check_password("testpass") is True
    assert check_password("wrong") is False

def test_token_roundtrip():
    token = make_token("professor")
    result = decode_token(token)
    assert result == "professor"

def test_bad_token():
    assert decode_token("garbage") is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_auth.py -v 2>&1 | head -10
```
Expected: FAIL

- [ ] **Step 3: Write web/auth.py**

```python
from __future__ import annotations
import os
from itsdangerous import URLSafeSerializer, BadSignature

ACCESS_PASSWORD = os.environ.get("ACCESS_PASSWORD", "professor")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
_signer = URLSafeSerializer(SECRET_KEY, salt="professor-session")

def check_password(plain: str) -> bool:
    return plain == ACCESS_PASSWORD

def make_token(payload: str = "authenticated") -> str:
    return _signer.dumps(payload)

def decode_token(token: str) -> str | None:
    try:
        return _signer.loads(token)
    except BadSignature:
        return None
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_auth.py -v
```
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add web/auth.py tests/test_auth.py
git commit -m "feat: simple password auth with signed session tokens"
```

---

## Task 4: RAG module

**Files:**
- Create: `web/rag.py`

The RAG module is identical in pattern to FilmMaker's. It loads all `.txt` files from `docs/`, chunks them, builds BM25 index.

- [ ] **Step 1: Write web/rag.py**

```python
from __future__ import annotations
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

DOCS_DIR      = Path(__file__).parent.parent / "docs"
CHUNK_SIZE    = 350
CHUNK_OVERLAP = 60
TOP_K         = 5
MIN_SCORE     = 0.1

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
    for fp in sorted(DOCS_DIR.rglob("*.txt")):
        try:
            content = fp.read_text(encoding="utf-8")
        except Exception:
            continue
        if len(content.strip()) < 100:
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
    if index is None or not chunks or not query.strip():
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
    lines = ["[Materiale didattico rilevante dal corpus]"]
    for i, (chunk, src) in enumerate(selected, 1):
        lines.append(f"[{i}] {src}:\n{chunk.strip()}")
    return "\n\n".join(lines)
```

- [ ] **Step 2: Write test**

```python
# tests/test_rag.py
import sys
sys.path.insert(0, "/Users/filippoarici/projects/Professor")

def test_load_corpus_empty_docs():
    """When docs are empty, corpus is empty — no crash."""
    from web.rag import load_corpus, build_index, retrieve
    chunks, sources, n = load_corpus()
    # Could be 0 if corpus not yet written, that's fine
    index = build_index(chunks)
    result = retrieve("context window", index, chunks, sources)
    assert isinstance(result, str)
```

- [ ] **Step 3: Run test**

```bash
python -m pytest tests/test_rag.py -v
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add web/rag.py tests/test_rag.py
git commit -m "feat: BM25 RAG module for professor corpus"
```

---

## Task 5: Professor system prompt

**Files:**
- Create: `professor.py`

- [ ] **Step 1: Write professor.py**

```python
PROFESSOR_SYSTEM_PROMPT = """\
Sei il professore personale di Filippo Arici, un consulente finanziario che costruisce prodotti AI.
Il tuo unico scopo: insegnargli a usare Claude al massimo del suo potenziale.

PROFILO DI APPRENDIMENTO DI FILIPPO:
{learning_profile}

CONVERSAZIONE DI LAVORO (pannello sinistro, ultimi 20 messaggi):
{left_context}

---

Il tuo approccio:
- Parti sempre da ciò che Filippo ha già capito (vedi profilo apprendimento sopra)
- Non ripetere mai cose già consolidate nel profilo
- Usa il metodo Feynman: prima la versione semplice e concreta, poi la profondità
- UNA cosa alla volta — mai sovraccaricare con troppi concetti
- Dopo ogni spiegazione chiedi: "ha senso? vuoi approfondire qualcosa?"
- Quando vedi un pattern negativo ricorrente nella conversazione di sinistra, segnalalo con gentilezza e mostra l'alternativa
- Sul tema costi: sempre numeri concreti, confronti reali, esempi pratici (es. "questo prompt da 500 token costa X centesimi su Opus")
- Usa la conversazione di lavoro di sinistra come materiale didattico reale: commenta le scelte di Filippo, spiega perché funziona o non funziona
- Hai accesso al corpus documentale — quando rilevi un concetto importante, spiegalo con le tue parole (non citare il corpus)

Aree di insegnamento:
1. Come funziona Claude — architettura, context window, limiti, allucinazioni, variabilità
2. Prompting — struttura efficace, few-shot, chain of thought, iterazione, system prompt
3. Costi e modelli — Opus/Sonnet/Haiku, pricing token, prompt caching, batch API, quando usare API vs interfaccia
4. Apprendimento — metodo Feynman, modelli mentali, consolidamento, imitare vs capire

Rispondi sempre in italiano. Tono: professore brillante e amichevole, non accademico.
"""
```

- [ ] **Step 2: Verify import works**

```bash
cd /Users/filippoarici/projects/Professor
python -c "from professor import PROFESSOR_SYSTEM_PROMPT; print(len(PROFESSOR_SYSTEM_PROMPT), 'chars')"
```
Expected: prints char count, no error

- [ ] **Step 3: Commit**

```bash
git add professor.py
git commit -m "feat: professor system prompt with Feynman method"
```

---

## Task 6: Corpus files

**Files:**
- Create: 13 `.txt` files across 4 `docs/` subdirectories

- [ ] **Step 1: Write docs/come_funziona_claude/context_window.txt**

```
LA CONTEXT WINDOW DI CLAUDE

La context window è la memoria di lavoro di Claude: tutto ciò che Claude può "vedere" in un singolo momento — il system prompt, la conversazione, i file caricati. Una volta che il contesto è pieno, Claude non può ricordare le parti più vecchie della conversazione.

Claude Opus ha una context window di 200.000 token (circa 150.000 parole — più lungo di "Guerra e Pace"). In pratica, per la maggior parte delle conversazioni quotidiane, non si raggiunge mai il limite.

COSA CONTA NEI TOKEN
- 1 token ≈ 0.75 parole in italiano
- Un'immagine = circa 1.600 token (dipende dalla risoluzione)
- Un PDF di 10 pagine ≈ 3.000-5.000 token
- Una conversazione di 1 ora ≈ 10.000-30.000 token

COSA SUCCEDE QUANDO LA CONTEXT WINDOW SI RIEMPIE
Claude non "dimentica" silenziosamente — restituisce un errore. La soluzione è iniziare una nuova sessione. Alcune app (come quelle con "compaction" attiva) riassumono automaticamente le parti vecchie.

IMPLICAZIONE PRATICA
Più contesto dai a Claude, meglio risponde — ma ogni token in input ha un costo. La strategia ottimale: dare tutto il contesto necessario per il task specifico, non di più.
```

- [ ] **Step 2: Write docs/come_funziona_claude/elaborazione_istruzioni.txt**

```
COME CLAUDE ELABORA LE ISTRUZIONI

Claude non "capisce" nel senso umano — predice i token più probabili dato tutto il contesto. Questo ha conseguenze pratiche importanti.

PRIORITÀ DELLE ISTRUZIONI
1. System prompt — ha la priorità più alta
2. Human turn — l'ultima richiesta dell'utente
3. Contesto della conversazione precedente

Se il system prompt dice "rispondi solo in italiano" e l'utente scrive in inglese, Claude risponde in italiano.

AMBIGUITÀ E INTERPRETAZIONE
Claude interpreta le istruzioni ambigue scegliendo l'interpretazione più plausibile dato il contesto. Se vuoi un comportamento specifico, sii esplicito. "Scrivi un email" è ambiguo. "Scrivi un'email formale di 3 paragrafi, tono professionale, nessuna emoji" è preciso.

ISTRUZIONI CONTRADDITTORIE
Se le istruzioni si contraddicono, Claude tende a seguire quella più recente o quella più specifica. Ma può anche "mediare" tra le due. Evita contraddizioni: se cambi idea, ridefinisci chiaramente.

IL RUOLO DEL FORMATO
Claude usa il formato (lunghezza, struttura, elenchi, markdown) come segnale su cosa stai cercando. Una domanda breve → risposta breve. Una domanda dettagliata → risposta dettagliata. Puoi sempre specificare il formato esplicitamente.
```

- [ ] **Step 3: Write docs/come_funziona_claude/limiti_allucinazioni.txt**

```
LIMITI E ALLUCINAZIONI DI CLAUDE

Claude può sbagliare. Non in modo random — ci sono pattern prevedibili.

DOVE CLAUDE FALLISCE PIÙ SPESSO
1. Fatti numerici precisi — date, statistiche, prezzi. Verifica sempre con fonti esterne.
2. Citazioni testuali — Claude può inventare citazioni plausibili. Non citare mai Claude su citazioni.
3. Conoscenza recente — il training ha una data di taglio. Notizie recenti = rischio alto.
4. Dettagli specifici di nicchia — più un argomento è specializzato, più alto il rischio.
5. Calcoli complessi — Claude fa matematica, ma può sbagliare. Usa strumenti dedicati per calcoli critici.

COME RICONOSCERE UNA RISPOSTA INCERTA
Claude spesso segnala l'incertezza con frasi come "credo", "mi sembra", "non sono sicuro". Prendile sul serio.

COME RIDURRE LE ALLUCINAZIONI
- Chiedi a Claude di citare le fonti
- Chiedi "sei sicuro?" o "controlla il tuo ragionamento"
- Per fatti critici, chiedi "con che livello di certezza sai questo?"
- Usa Claude come primo draft, non come fonte finale

LA DIFFERENZA CON "MENTIRE"
Claude non mente intenzionalmente. Genera il testo più plausibile dato il training — che a volte è sbagliato. Capire questo cambia come lo usi: è un ottimo collaboratore creativo e analista, non un database di fatti certificati.
```

- [ ] **Step 4: Write docs/come_funziona_claude/variabilita_risposte.txt**

```
PERCHÉ CLAUDE RISPONDE DIVERSAMENTE A DOMANDE SIMILI

Prova a fare la stessa domanda due volte — le risposte saranno diverse. Non è un bug.

IL PARAMETRO TEMPERATURE
Claude usa un parametro chiamato "temperature" che introduce variabilità nelle risposte. Temperature alta = più creativo ma meno coerente. Temperature bassa = più coerente ma meno originale. Le API permettono di controllarlo; nelle interfacce standard è fisso.

CONTESTO E POSIZIONE
Claude è molto sensibile al contesto. La stessa domanda in conversazioni diverse, con system prompt diversi, o dopo messaggi diversi, produce risposte diverse. Non è instabilità — è sensibilità al contesto.

IMPLICAZIONE PER IL LAVORO
Per task critici dove hai bisogno di coerenza (es. estrarre dati da documenti), usa istruzioni molto precise e format output strutturati (JSON, tabelle). Per task creativi, la variabilità è un vantaggio — prova la stessa domanda più volte.

UN TRUCCO PRATICO
Se Claude dà una risposta insoddisfacente, spesso basta riformulare la domanda diversamente, cambiare l'ordine del contesto, o aggiungere "pensa passo per passo prima di rispondere". Piccoli cambiamenti nel prompt producono risposte molto diverse.
```

- [ ] **Step 5: Write docs/prompting/struttura_prompt_efficace.txt**

```
STRUTTURA DI UN PROMPT EFFICACE

Un prompt efficace ha quattro componenti: ruolo, contesto, compito, formato.

RUOLO
Definisci chi è Claude per questo task.
"Sei un analista finanziario senior specializzato in mercati europei."
Il ruolo attiva il registro linguistico, il livello di dettaglio, e il tipo di ragionamento appropriato.

CONTESTO
Dai a Claude tutto ciò che serve per capire la situazione.
"Sto scrivendo una relazione trimestrale per un cliente con profilo MiFID moderato. Il cliente ha 50% equity, 50% bond."
Senza contesto, Claude indovina. Con contesto, lavora sul tuo caso specifico.

COMPITO
Descrivi esattamente cosa vuoi.
Impreciso: "dimmi qualcosa sul mercato"
Preciso: "analizza i rischi principali per un portafoglio equity europeo nei prossimi 3 mesi"

FORMATO
Specifica come vuoi l'output.
"Rispondi in 3 bullet point, max 2 righe ciascuno, in italiano, senza introduzione."

ESEMPIO COMPLETO
"Sei un consulente finanziario esperto (ruolo). Il mio cliente ha 60 anni, pensionato, profilo conservativo, 300k EUR in liquidità (contesto). Suggerisci 3 possibili allocazioni, spiegando rischi e rendimenti attesi di ciascuna (compito). Formato: tabella con 4 colonne: allocazione, rischio, rendimento atteso, note (formato)."
```

- [ ] **Step 6: Write docs/prompting/few_shot_prompting.txt**

```
FEW-SHOT PROMPTING: INSEGNARE CON GLI ESEMPI

Il few-shot prompting significa dare a Claude esempi di input/output prima di fare la domanda reale. È uno dei metodi più potenti per controllare il formato e il tono della risposta.

QUANDO USARLO
- Quando vuoi un formato di output molto specifico
- Quando il tono deve essere preciso (es. tono del tuo brand)
- Quando fai task ripetitivi con struttura fissa
- Quando la descrizione verbale del risultato è più difficile dell'esempio stesso

STRUTTURA BASE
Esempio 1:
Input: [testo di esempio]
Output: [output desiderato]

Esempio 2:
Input: [testo di esempio]
Output: [output desiderato]

Ora elabora questo:
Input: [il tuo input reale]

ESEMPIO APPLICATO (classificazione notizie per impatto finanziario)
Input: "Fed alza tassi di 25 basis points"
Output: {"impatto": "alto", "settori": ["banche", "immobiliare"], "direzione": "negativo equity"}

Input: "Nuovo CEO per Apple"
Output: {"impatto": "medio", "settori": ["tech"], "direzione": "incerto"}

Ora classifica: "Inflazione eurozona scende al 2.1%"

QUANTI ESEMPI
1-2 esempi per task semplici. 3-5 per task complessi o con molte varianti. Oltre 5, spesso non migliora significativamente.
```

- [ ] **Step 7: Write docs/prompting/chain_of_thought.txt**

```
CHAIN OF THOUGHT: RAGIONAMENTO PASSO PER PASSO

Il chain of thought (CoT) è una tecnica per fare ragionare Claude in modo esplicito prima di rispondere. Migliora significativamente le performance su task analitici e matematici.

COME SI ATTIVA
Il modo più semplice: aggiungere "Pensa passo per passo prima di rispondere" o "Ragiona ad alta voce".

Con Claude Opus puoi usare la funzione "extended thinking" (via API) che dedica token separati al ragionamento interno, non visibili nella risposta finale.

PERCHÉ FUNZIONA
Claude genera token sequenzialmente. Se gli chiedi di "pensare" prima di rispondere, i token intermedi del ragionamento diventano contesto per la risposta finale, migliorandola.

QUANDO È UTILE
- Problemi matematici o logici
- Analisi multi-step (es. "analizza questo contratto e identifica i rischi")
- Decisioni con trade-off complessi
- Debugging di codice o ragionamenti

ESEMPIO
Senza CoT: "Quale dei tre fondi è migliore per un cliente conservativo?"
Con CoT: "Analizza i tre fondi dal punto di vista di un cliente conservativo. Prima elenca i criteri rilevanti per un profilo conservativo, poi valuta ogni fondo su quei criteri, poi dai una raccomandazione."

Il secondo produce un'analisi molto più robusta e verificabile.

COSTO
Il CoT usa più token e quindi costa di più. Su task semplici non serve. Su task analitici complessi, l'investimento vale.
```

- [ ] **Step 8: Write docs/prompting/system_prompt.txt**

```
IL SYSTEM PROMPT: CONFIGURARE CLAUDE PER IL TUO CASO D'USO

Il system prompt è la prima istruzione che Claude riceve, con la priorità più alta. Non è visibile all'utente ma determina come Claude si comporta per tutta la conversazione.

A COSA SERVE
- Definire il ruolo e la personalità di Claude
- Stabilire regole di comportamento fissi (es. "rispondi sempre in italiano")
- Fornire contesto permanente (es. informazioni sull'utente, sul prodotto)
- Definire il formato di output standard
- Limitare o ampliare il comportamento di default

ESEMPIO DI SYSTEM PROMPT EFFICACE
"Sei un assistente per consulenti finanziari italiani. Hai conoscenza approfondita di MiFID II e normativa CONSOB. Rispondi sempre in italiano professionale. Mai dare consigli di investimento specifici senza che il consulente abbia confermato il profilo del cliente. Quando non hai abbastanza contesto, chiedi invece di assumere."

DIFFERENZA CON LE ISTRUZIONI NEL CHAT
Le istruzioni nel system prompt si applicano all'intera conversazione. Le istruzioni nel chat si applicano al messaggio specifico. Se si contraddicono, il system prompt vince (di solito).

VIA API VS INTERFACCIA STANDARD
Via API puoi passare il system prompt a ogni chiamata. Nell'interfaccia standard (claude.ai), puoi creare "Projects" con system prompt personalizzati. Per prodotti professionali, l'API è essenziale.

UN PATTERN COMUNE
System prompt fisso con informazioni stabili + user message con contesto variabile per ogni richiesta. Questo ottimizza anche il caching (risparmio fino all'80% sui costi del system prompt ripetuto).
```

- [ ] **Step 9: Write docs/prompting/iterazione.txt**

```
ITERAZIONE: AFFINARE INVECE DI RICOMINCIARE

Il pattern più comune e inefficiente: ricevere una risposta mediocre, cancellare tutto, riscrivere il prompt da zero. Il pattern corretto: iterare sul risultato che hai.

PERCHÉ L'ITERAZIONE È PIÙ POTENTE
Claude ha già capito il contesto. Ogni messaggio aggiuntivo aggiunge informazioni. Riformulare da zero significa perdere tutto questo.

TECNICHE DI ITERAZIONE

"Troppo lungo" → "Accorcia a metà mantenendo i punti chiave"
"Tono sbagliato" → "Riscrivi con tono più formale/informale"
"Manca qualcosa" → "Aggiungi una sezione su [X]"
"Non è quello che volevo" → "Mi aspettavo [Y]. Riprova con questa prospettiva."
"Troppo generico" → "Applicalo specificamente al contesto [Z]"

IL CICLO IDEALE
1. Prompt iniziale con contesto e task
2. Leggi la risposta — cosa manca o non va?
3. Istruzione specifica di correzione
4. Ripeti fino a soddisfazione

AFFINARE VS RIFORMULARE
Affinare: "il terzo punto non è chiaro, espandilo con un esempio"
Riformulare: cancellare tutto e riscrivere — quasi mai necessario

QUANTE ITERAZIONI
Per task semplici: 1-2. Per task complessi (es. scrivere una policy aziendale): 4-6. Se dopo 3 iterazioni la risposta è ancora sbagliata, il problema è probabilmente nel contesto iniziale, non nel prompt.
```

- [ ] **Step 10: Write docs/costi_modelli/opus_sonnet_haiku.txt**

```
OPUS, SONNET, HAIKU: QUANDO USARE QUALE

Anthropic offre tre livelli di modelli, ciascuno con un diverso trade-off tra capacità e costo.

CLAUDE OPUS (il più potente)
- Costo: ~$15/M token input, ~$75/M token output (via Anthropic)
  Via OpenRouter: ~$15/$75 per milione token
- Quando usarlo: task analitici complessi, ragionamento multi-step, coding avanzato, documenti lunghi che richiedono comprensione profonda
- Esempio pratico: analisi di un contratto di 50 pagine, generazione di una strategia finanziaria dettagliata

CLAUDE SONNET (il bilanciato)
- Costo: ~$3/M input, ~$15/M output
- Quando usarlo: la maggior parte dei task quotidiani — drafting, analisi moderate, conversazioni, classificazione
- Il rapporto qualità/prezzo migliore per uso intensivo

CLAUDE HAIKU (il veloce e economico)
- Costo: ~$0.25/M input, ~$1.25/M output
- Quando usarlo: task semplici e ripetitivi — classificazione, estrazione dati strutturati, risposte brevi
- Esempio: classificare 1000 email per urgenza

CALCOLO PRATICO (esempio conversazione media)
- Un messaggio utente da 100 parole ≈ 133 token = $0.002 su Opus
- Una risposta da 300 parole ≈ 400 token = $0.030 su Opus
- Una sessione di 20 scambi ≈ $0.64 su Opus, $0.13 su Sonnet, $0.01 su Haiku

LA STRATEGIA INTELLIGENTE
Usa il modello più economico che soddisfa il tuo requisito di qualità. Non usare Opus per classificare email. Usa Opus quando la qualità della risposta vale la differenza di costo.
```

- [ ] **Step 11: Write docs/costi_modelli/pricing_token.txt**

```
PRICING A TOKEN: CAPIRE COSA STAI PAGANDO

Anthropic (e OpenRouter) prezza per token. Capire la struttura del prezzo cambia come progetti le applicazioni AI.

INPUT VS OUTPUT
I token di output costano molto di più degli input. Su Opus: input $15/M, output $75/M — 5x più caro.

Implicazione: se hai un task dove controlli la lunghezza dell'output (es. "rispondi in max 3 frasi"), puoi ridurre significativamente i costi.

COSA CONTA COME INPUT
- Il system prompt (ogni volta, a meno di caching)
- Tutta la conversazione precedente
- File allegati (PDF, immagini)
- Il messaggio corrente dell'utente

Attenzione: in un'app chat, ogni messaggio include TUTTA la cronologia. Una conversazione di 50 messaggi = 50 volte il contesto accumulato in input.

ESEMPIO CONCRETO — APPLICAZIONE CHAT
- 50 messaggi, media 200 parole ciascuno = 10.000 parole totali
- Stima: 13.000 token di contesto al messaggio 50
- Costo del messaggio 50 su Opus solo di input: 13.000 × $15/1.000.000 = $0.20
- Costo totale della conversazione (50 messaggi): ~$5 su Opus, ~$1 su Sonnet

CALCOLA IL TUO CASO
Token per messaggio = (lunghezza media history + messaggio corrente) × 1.3 (overhead)
Costo per messaggio = token_input × prezzo_input + token_output × prezzo_output
```

- [ ] **Step 12: Write docs/costi_modelli/prompt_caching.txt**

```
PROMPT CACHING: RISPARMIO FINO ALL'80%

Il prompt caching è una feature API che permette di "cachare" parti del prompt che si ripetono. Invece di pagarle ogni volta, le paghi una volta e poi risparmi sulle richieste successive.

COME FUNZIONA
Marchi una parte del prompt come "cacheable" (es. il system prompt, un documento lungo). Anthropic salva quella parte in cache per 5 minuti. Le richieste successive che usano la stessa prefix cachata pagano solo il 10% del prezzo normale per quella parte.

RISPARMIO REALE
- Sistema con system prompt da 5.000 token
- 100 richieste al giorno
- Senza caching: 5.000 × 100 × $15/M = $7.50/giorno solo di system prompt
- Con caching: prima richiesta $0.075, poi 99 × $0.0075 = $0.74/giorno
- Risparmio: 90%

QUANDO USARLO
Quando hai contenuto fisso che si ripete tra richieste: system prompt lungo, documenti di riferimento, corpus di conoscenza. Non serve per conversazioni brevi dove il contesto cambia sempre.

VIA API
Aggiungi `"cache_control": {"type": "ephemeral"}` ai blocchi che vuoi cachare. Disponibile solo via API (non interfaccia standard).

ATTENZIONE
La cache dura 5 minuti. Se le richieste sono distanziate più di 5 minuti, la cache scade. Funziona meglio per applicazioni con alto throughput (molte richieste ravvicinate).
```

- [ ] **Step 13: Write docs/costi_modelli/batch_api.txt**

```
BATCH API: SCONTO 50% PER TASK NON URGENTI

La Batch API di Anthropic permette di inviare fino a 100.000 richieste in un unico batch. Il prezzo è il 50% di quello standard. Il trade-off: le risposte arrivano entro 24 ore (non in real-time).

QUANDO HA SENSO
- Analisi di grandi volumi di documenti (es. classificare 10.000 notizie)
- Generazione di contenuti in batch (es. riassumere 500 articoli)
- Task di elaborazione dati dove non serve risposta immediata
- Testing e valutazione di prompt su dataset grandi

ESEMPIO PRATICO
Hai 1.000 news da classificare per impatto finanziario.
- Sonnet real-time: 1.000 × ~200 token × $3/M = $0.60
- Sonnet Batch API: $0.30

Con volumi grandi (es. 100.000 documenti) il risparmio diventa rilevante.

COME FUNZIONA
1. Crei un file JSONL con tutte le richieste
2. Invii il batch via API
3. Anthropic elabora in background
4. Recuperi i risultati quando pronti (polling o webhook)

LIMITAZIONI
- Max 24 ore per completamento (di solito molto meno)
- Non per applicazioni real-time o interattive
- Stesse limitazioni di contesto dei modelli standard

LA STRATEGIA
Separa i task urgenti (real-time, utente in attesa) dai task non urgenti (processing batch). Usa Batch API per il secondo gruppo.
```

- [ ] **Step 14: Write docs/costi_modelli/api_vs_interfaccia.txt**

```
API DIRETTA VS INTERFACCIA STANDARD: QUANDO PASSARE

Claude è disponibile su claude.ai (interfaccia standard) e tramite API (per sviluppatori). Capire la differenza aiuta a scegliere lo strumento giusto.

INTERFACCIA STANDARD (claude.ai)
- Nessun setup tecnico
- Costo: abbonamento fisso ($20/mese Pro, $30 Team)
- Limite: rate limit mensile, nessun controllo programmatico
- Quando ha senso: uso personale, esplorazione, task one-off

API DIRETTA
- Richiede programmazione o app che la usano
- Costo: pay-per-token (vedi pricing_token.txt)
- Controllo completo: model, temperature, system prompt, format
- Quando ha senso: applicazioni, automazioni, alto volume, integrazione in prodotti

OPENROUTER (quello che usate in Professor)
- Aggregatore di API: un unico provider per accedere a Claude, GPT, Gemini, ecc.
- Prezzi leggermente superiori all'API diretta (margine di OpenRouter)
- Vantaggio: flessibilità di modello senza cambiare codice
- Buono per prototipi e prodotti dove vuoi mantenere la flessibilità di modello

CALCOLO DI BREAK-EVEN
Con $20/mese di abbonamento, quanti token puoi usare via API allo stesso costo?
$20 / ($3/M token Sonnet) = 6.67M token = circa 5 milioni di parole
Se usi più di questo, l'API è più costosa. Se usi meno, uguale o più conveniente.

LA VERITÀ PRATICA
Per uso personale leggero: interfaccia standard. Per prodotti, automazioni, o uso intensivo professionale: API diretta o OpenRouter.
```

- [ ] **Step 15: Write docs/apprendimento/metodo_feynman.txt**

```
IL METODO FEYNMAN PER IMPARARE DAVVERO

Richard Feynman, fisico premio Nobel, aveva un metodo per capire davvero qualcosa invece di illudersi di capirla.

IL METODO IN 4 PASSI
1. Scegli il concetto
2. Spiegalo come se lo stessi insegnando a un bambino di 12 anni
3. Identifica i punti dove la spiegazione si inceppa (quelli sono i gap nella tua comprensione)
4. Torna al materiale originale e riempi quei gap

IL TEST CRUCIALE
Se non riesci a spiegarlo in modo semplice, non l'hai capito davvero. La semplicità della spiegazione è la misura della profondità della comprensione.

APPLICATO ALL'AI
Quando studi come funziona Claude, prova a spiegarlo ad alta voce come se stessi insegnando a qualcuno. "Claude funziona perché..." — se ti inceppi, hai trovato il gap.

UN ESEMPIO CONCRETO
Concepto: "temperature nei modelli linguistici"
Spiegazione Feynman: "È come un dado che Claude lancia quando deve scegliere la prossima parola. Con temperature bassa, sceglie quasi sempre la parola più probabile (dado truccato). Con temperature alta, lancia il dado più random e sceglie parole meno ovvie (più creativo, meno coerente)."
Se puoi spiegarlo così, l'hai capito.

PERCHÉ FUNZIONA
Spiegare semplice richiede connessioni profonde tra concetti. Quando quelle connessioni mancano, non riesci a semplificare. Il metodo forza a rilevare i propri gap invece di camuffarli.
```

- [ ] **Step 16: Write docs/apprendimento/modelli_mentali.txt**

```
MODELLI MENTALI: COSTRUIRE COMPRENSIONE DURATURA

Un modello mentale è una struttura interna che usi per ragionare su un dominio. Non è un fatto — è un meccanismo che ti permette di prevedere comportamenti e fare scelte.

DIFFERENZA TRA FATTI E MODELLI
Fatto: "Claude costa $15/M token di output su Opus"
Modello mentale: "I modelli AI hanno un trade-off costo/qualità. Per ogni task, esiste un livello di qualità 'sufficiente' — trovarlo e usare il modello minimo che lo raggiunge è la skill principale nella gestione dei costi AI."

Il fatto si dimentica. Il modello mentale ti permette di ragionare su casi nuovi.

MODELLI MENTALI CHIAVE PER USARE L'AI
1. "Claude predice, non capisce" — aiuta a capire quando si sbaglia e perché
2. "Contesto = qualità" — più contesto rilevante, migliore la risposta
3. "Prompt = specifica, non richiesta" — tratta i prompt come requisiti software
4. "Iterazione > perfezione al primo colpo" — pianifica di affinare

COME SI COSTRUISCONO
I modelli mentali si costruiscono lentamente attraverso l'esperienza + la riflessione. Non basta usare Claude — bisogna notare i pattern: quando funziona bene? Quando sbaglia? Cosa cambia tra due sessioni?

IL RUOLO DI UN PROFESSORE
Un buon professore non trasferisce fatti — costruisce modelli mentali. Ogni volta che chiedo "perché?" invece di "cosa?", ti sto aiutando a costruire un modello.
```

- [ ] **Step 17: Write docs/apprendimento/spaced_repetition.txt**

```
SPACED REPETITION: CONSOLIDARE NEL TEMPO

La spaced repetition (ripetizione spaziata) è la tecnica di apprendimento con il supporto scientifico più solido. L'idea: ripassare i concetti a intervalli crescenti nel tempo, proprio quando stai quasi per dimenticarli.

LA CURVA DELL'OBLIO
Senza ripasso, dimentichi il 50% di ciò che hai appreso entro un giorno, il 70% entro una settimana. Con ripasso al momento giusto, la curva di oblio si appiattisce e il ricordo diventa quasi permanente.

INTERVALLI OTTIMALI (approssimativi)
- Prima revisione: 1 giorno dopo
- Seconda revisione: 3 giorni dopo
- Terza revisione: 1 settimana dopo
- Quarta revisione: 2 settimane dopo
- Poi: mensile

APPLICATO ALL'AI
Invece di studiare tutto in una sessione intensiva, distribuisci l'apprendimento. Ogni sessione con questo professore introduce 1-2 concetti nuovi. La sessione successiva inizia ripassando brevemente i concetti della sessione precedente.

QUESTO È PERCHÉ IL PROFESSORE TIENE UN PROFILO
Il profilo di apprendimento non serve solo a non ripetere cose già dette. Serve a identificare quando è il momento giusto per riprendere un concetto — quando è stato introdotto abbastanza tempo fa da essere quasi dimenticato, ma non così tanto da doverlo reintrodurre da zero.

STRUMENTI PRATICI
Anki è lo strumento digitale più usato per la spaced repetition. Ma anche solo rileggere i propri appunti a intervalli crescenti ha un effetto significativo.
```

- [ ] **Step 18: Write docs/apprendimento/imitare_vs_capire.txt**

```
IMITARE VS CAPIRE: IL VERO SALTO NELL'USO DELL'AI

C'è una differenza fondamentale tra usare l'AI imitando pattern che hai visto funzionare, e usarla partendo da una comprensione di come funziona.

IL LIVELLO IMITAZIONE
"Ho visto che se scrivo 'sei un esperto di X' funziona meglio."
"Ho copiato questo prompt da un tutorial e funziona."
"Quando chiedo in inglese risponde meglio."

Il livello imitazione produce risultati ragionevoli ma non permette di adattarsi a situazioni nuove o di diagnosticare quando qualcosa non funziona.

IL LIVELLO COMPRENSIONE
"Il ruolo esplicito funziona perché attiva pattern di risposta specifici nel training."
"Questo prompt funziona perché dà contesto preciso, specifica il formato, e riduce l'ambiguità."
"L'inglese a volte dà risultati diversi perché il training set in inglese è più grande."

Il livello comprensione permette di costruire prompt originali per situazioni nuove, diagnosticare errori, ottimizzare sistematicamente.

COME FARE IL SALTO
1. Quando un prompt funziona bene, chiediti: perché?
2. Quando non funziona, chiediti: cosa manca? Cosa è ambiguo?
3. Varia deliberatamente un elemento alla volta e osserva l'effetto
4. Costruisci il tuo modello mentale (vedi modelli_mentali.txt)

IL MOMENTO DEL SALTO
Il salto da imitazione a comprensione avviene quando riesci a spiegare a qualcuno perché un certo prompt funziona — non solo che funziona.
```

- [ ] **Step 19: Run test to verify corpus loads**

```bash
cd /Users/filippoarici/projects/Professor
python -c "
from web.rag import load_corpus, build_index
chunks, sources, n = load_corpus()
print(f'{n} files, {len(chunks)} chunks')
assert n == 13, f'Expected 13 files, got {n}'
print('OK')
"
```
Expected: "13 files, N chunks\nOK"

- [ ] **Step 20: Commit**

```bash
git add docs/ professor.py
git commit -m "feat: professor system prompt + 13 corpus files"
```

---

## Task 7: Base template

**Files:**
- Create: `web/templates/base.html`

- [ ] **Step 1: Write web/templates/base.html**

```html
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Professor</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:       #0a0b0d;
      --surface:  #111215;
      --elevated: #18191d;
      --border:   rgba(255,255,255,0.07);
      --border2:  rgba(255,255,255,0.14);
      --text:     #f0f0f2;
      --label:    #8e8e93;
      --faint:    rgba(255,255,255,0.04);
      --accent:   #f0ece6;
      --font:     -apple-system, BlinkMacSystemFont, 'Helvetica Neue', 'Segoe UI', sans-serif;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      font-size: 14px;
      line-height: 1.6;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }

    .nav {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      height: 44px;
      padding: 0 20px;
      border-bottom: 1px solid var(--border);
      background: rgba(10,11,13,0.85);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      position: sticky;
      top: 0;
      z-index: 10;
    }

    .nav-brand {
      font-size: 16px;
      font-weight: 500;
      letter-spacing: 0.02em;
      color: var(--accent);
      text-decoration: none;
      text-align: center;
    }

    .nav-left  { display: flex; gap: 20px; align-items: center; }
    .nav-right { display: flex; gap: 20px; align-items: center; justify-content: flex-end; }

    .nav-link {
      font-size: 13px;
      color: var(--label);
      text-decoration: none;
      transition: color 0.15s;
    }
    .nav-link:hover { color: var(--text); }

    .pill {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 3px 10px; border-radius: 20px;
      border: 1px solid var(--border2);
      font-size: 11px; color: var(--label);
      background: var(--faint);
    }

    .dot { width: 6px; height: 6px; border-radius: 50%; background: #34c759; }

    /* Buttons */
    .btn {
      padding: 7px 16px; border-radius: 8px; border: none;
      font-family: var(--font); font-size: 13px; cursor: pointer;
      transition: opacity 0.15s;
    }
    .btn:hover { opacity: 0.85; }
    .btn-primary { background: var(--accent); color: #0a0b0d; font-weight: 500; }
    .btn-ghost   { background: transparent; color: var(--label); border: 1px solid var(--border); }

    /* Form fields */
    .field {
      width: 100%;
      padding: 10px 14px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-family: var(--font);
      font-size: 14px;
      outline: none;
      transition: border-color 0.15s;
    }
    .field:focus { border-color: var(--border2); }

    {% block extra_css %}{% endblock %}
  </style>
</head>
<body>

<nav class="nav">
  <div class="nav-left">{% block nav_left %}{% endblock %}</div>
  <a href="/studio" class="nav-brand">Professor</a>
  <div class="nav-right">{% block nav_right %}{% endblock %}</div>
</nav>

{% block content %}{% endblock %}

{% block scripts %}{% endblock %}
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add web/templates/base.html
git commit -m "feat: base template with Apple design system"
```

---

## Task 8: Login template + auth routes

**Files:**
- Create: `web/templates/login.html`
- Create: `web/app.py` (partial — just auth routes + lifespan)

- [ ] **Step 1: Write web/templates/login.html**

```html
{% extends "base.html" %}

{% block extra_css %}
.login-wrap {
  display: flex; align-items: center; justify-content: center;
  min-height: calc(100vh - 44px);
}
.login-card {
  width: 320px;
  padding: 40px 32px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
}
.login-title {
  font-size: 18px; font-weight: 600; color: var(--accent);
  text-align: center; margin-bottom: 8px;
}
.login-sub {
  font-size: 13px; color: var(--label);
  text-align: center; margin-bottom: 28px;
}
.form-group { margin-bottom: 16px; }
.form-label {
  display: block; font-size: 12px; color: var(--label);
  margin-bottom: 6px; font-weight: 500;
}
.error-msg {
  font-size: 13px; color: #ff453a;
  text-align: center; margin-top: 12px;
}
{% endblock %}

{% block content %}
<div class="login-wrap">
  <div class="login-card">
    <div class="login-title">Professor AI</div>
    <div class="login-sub">Inserisci la password per accedere</div>
    <form method="post" action="/login">
      <div class="form-group">
        <label class="form-label">Password</label>
        <input class="field" type="password" name="password"
               autofocus autocomplete="current-password" required>
      </div>
      <button class="btn btn-primary" type="submit" style="width:100%">Entra</button>
      {% if error %}
      <div class="error-msg">{{ error }}</div>
      {% endif %}
    </form>
  </div>
</div>
{% endblock %}
```

- [ ] **Step 2: Write web/app.py (skeleton + auth)**

```python
from __future__ import annotations
import os, pathlib
from typing import Optional
from fastapi import FastAPI, Request, Form, Cookie, Response
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from web.db import (
    init_db, start_session, end_session,
    save_message_left, save_message_right,
    get_messages_left, get_messages_right,
    get_session, list_sessions, get_learning_profile, update_learning_profile,
)
from web.auth import check_password, make_token, decode_token
from web.rag import load_corpus, build_index, retrieve

import re
from professor import PROFESSOR_SYSTEM_PROMPT

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-opus-4-6"

_db_path = os.environ.get("DATABASE_URL", "professor.db")

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
templates = Jinja2Templates(
    directory=str(pathlib.Path(__file__).parent / "templates")
)

def is_authenticated(session: Optional[str]) -> bool:
    if not session:
        return False
    return decode_token(session) is not None

# ── Public routes ──

@app.get("/", response_class=RedirectResponse)
def root():
    return RedirectResponse("/studio", status_code=303)

@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})

@app.post("/login")
def login_post(request: Request, password: str = Form(...)):
    if not check_password(password):
        return templates.TemplateResponse(
            request, "login.html", {"error": "Password errata."}
        )
    token = make_token()
    resp = RedirectResponse("/studio", status_code=303)
    resp.set_cookie("session", token, httponly=True, samesite="lax")
    return resp

@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("session")
    return resp
```

- [ ] **Step 3: Commit**

```bash
git add web/app.py web/templates/login.html
git commit -m "feat: login template and auth routes"
```

---

## Task 9: Studio, storia, profilo GET routes

**Files:**
- Modify: `web/app.py` — add studio, storia, profilo GET routes

- [ ] **Step 1: Append to web/app.py**

Add after the logout route:

```python
# ── Protected routes ──

@app.get("/studio", response_class=HTMLResponse)
def studio_get(request: Request, session: Optional[str] = Cookie(default=None)):
    if not is_authenticated(session):
        return RedirectResponse("/login", status_code=303)
    session_id = start_session()
    profile = get_learning_profile()
    return templates.TemplateResponse(request, "studio.html", {
        "session_id": session_id,
        "profile": profile,
    })

@app.get("/storia", response_class=HTMLResponse)
def storia_get(request: Request, session: Optional[str] = Cookie(default=None)):
    if not is_authenticated(session):
        return RedirectResponse("/login", status_code=303)
    sessions = list_sessions()
    return templates.TemplateResponse(request, "storia.html", {
        "sessions": sessions,
    })

@app.get("/storia/{session_id}", response_class=HTMLResponse)
def storia_detail(request: Request, session_id: int,
                  session: Optional[str] = Cookie(default=None)):
    if not is_authenticated(session):
        return RedirectResponse("/login", status_code=303)
    sess = get_session(session_id)
    if not sess:
        return RedirectResponse("/storia", status_code=303)
    msgs_left  = get_messages_left(session_id)
    msgs_right = get_messages_right(session_id)
    return templates.TemplateResponse(request, "storia_detail.html", {
        "sess": sess,
        "msgs_left": msgs_left,
        "msgs_right": msgs_right,
    })

@app.get("/profilo", response_class=HTMLResponse)
def profilo_get(request: Request, session: Optional[str] = Cookie(default=None)):
    if not is_authenticated(session):
        return RedirectResponse("/login", status_code=303)
    profile = get_learning_profile()
    return templates.TemplateResponse(request, "profilo.html", {
        "profile": profile,
    })
```

- [ ] **Step 2: Commit**

```bash
git add web/app.py
git commit -m "feat: protected studio/storia/profilo GET routes"
```

---

## Task 10: Left panel streaming (/chat/sinistra)

**Files:**
- Modify: `web/app.py` — add /chat/sinistra SSE route

- [ ] **Step 1: Append to web/app.py**

```python
@app.post("/chat/sinistra")
async def chat_sinistra(request: Request, session: Optional[str] = Cookie(default=None)):
    if not is_authenticated(session):
        return Response(status_code=401)

    from openai import AsyncOpenAI

    body = await request.json()
    conversation = body.get("conversation", [])   # [{role, content}, ...]
    session_id   = body.get("session_id")

    # Save user message
    if conversation and conversation[-1]["role"] == "user":
        save_message_left(session_id, "user", conversation[-1]["content"])

    async def generate():
        client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        msgs = [{"role": "system", "content": "Sei Claude, un assistente AI brillante. Rispondi liberamente senza vincoli particolari."}, *conversation]
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            max_tokens=8192,
            stream=True,
            extra_headers={"HTTP-Referer": "https://professor.ai", "X-Title": "Professor"},
        )
        full_response = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response.append(delta)
                yield f"data: {delta.replace(chr(10), '\\n')}\n\n"
        yield "data: [DONE]\n\n"
        # Save assistant message after stream
        if session_id:
            save_message_left(session_id, "assistant", "".join(full_response))

    return StreamingResponse(generate(), media_type="text/event-stream")
```

- [ ] **Step 2: Commit**

```bash
git add web/app.py
git commit -m "feat: left panel SSE streaming route"
```

---

## Task 11: Right panel streaming (/chat/destra)

**Files:**
- Modify: `web/app.py` — add /chat/destra SSE route with RAG + learning profile

- [ ] **Step 1: Append to web/app.py**

```python
def _build_professor_system_prompt() -> str:
    """Build professor system prompt with current learning profile."""
    profile = get_learning_profile()
    profile_text = ""
    for key, label in [
        ("come_funziona_claude", "Come funziona Claude"),
        ("prompting", "Prompting"),
        ("costi_modelli", "Costi e modelli"),
        ("apprendimento", "Apprendimento"),
        ("pattern_negativi", "Pattern negativi"),
        ("concetti_consolidati", "Concetti consolidati"),
        ("concetti_aperti", "Concetti aperti"),
        ("ultima_sessione", "Ultima sessione"),
    ]:
        val = profile.get(key, "")
        if val:
            profile_text += f"{label}: {val}\n"
    if not profile_text:
        profile_text = "(Nessun profilo ancora — prima sessione con Filippo)"
    return profile_text

@app.post("/chat/destra")
async def chat_destra(request: Request, session: Optional[str] = Cookie(default=None)):
    if not is_authenticated(session):
        return Response(status_code=401)

    from openai import AsyncOpenAI

    body = await request.json()
    conversation_right = body.get("conversation", [])   # [{role, content}, ...]
    left_context       = body.get("left_context", [])   # last 20 msgs from left panel
    session_id         = body.get("session_id")

    # Save user message
    if conversation_right and conversation_right[-1]["role"] == "user":
        save_message_right(session_id, "user", conversation_right[-1]["content"])

    # Build left context string
    left_str = ""
    if left_context:
        lines = [f"{m['role'].upper()}: {m['content']}" for m in left_context[-20:]]
        left_str = "\n".join(lines)
    else:
        left_str = "(nessuna conversazione di lavoro ancora)"

    # RAG retrieval based on user's last message
    last_user_msg = next(
        (m["content"] for m in reversed(conversation_right) if m["role"] == "user"),
        ""
    )
    rag_ctx = retrieve(last_user_msg, bm25, corpus_chunks, corpus_sources)

    # Build system prompt
    profile_text = _build_professor_system_prompt()
    system_content = PROFESSOR_SYSTEM_PROMPT.format(
        learning_profile=profile_text,
        left_context=left_str,
    )
    if rag_ctx:
        system_content += f"\n\n{rag_ctx}"

    msgs = [{"role": "system", "content": system_content}, *conversation_right]

    async def generate():
        client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            max_tokens=4096,
            stream=True,
            extra_headers={"HTTP-Referer": "https://professor.ai", "X-Title": "Professor"},
        )
        full_response = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response.append(delta)
                yield f"data: {delta.replace(chr(10), '\\n')}\n\n"
        yield "data: [DONE]\n\n"
        if session_id:
            save_message_right(session_id, "assistant", "".join(full_response))

    return StreamingResponse(generate(), media_type="text/event-stream")
```

- [ ] **Step 2: Commit**

```bash
git add web/app.py
git commit -m "feat: right panel professor SSE with RAG and learning profile"
```

---

## Task 12: Session end route (/sessione/fine)

**Files:**
- Modify: `web/app.py` — add /sessione/fine route

This route is called by `navigator.sendBeacon()` when the user leaves the page. It asks Claude to update the learning profile.

- [ ] **Step 1: Append to web/app.py**

```python
@app.post("/sessione/fine")
async def sessione_fine(request: Request, session: Optional[str] = Cookie(default=None)):
    if not is_authenticated(session):
        return Response(status_code=401)

    from openai import AsyncOpenAI

    try:
        body = await request.json()
    except Exception:
        return Response(status_code=200)

    session_id        = body.get("session_id")
    left_conversation = body.get("left_conversation", [])
    right_conversation= body.get("right_conversation", [])

    if not session_id:
        return Response(status_code=200)

    # Build session summary for profile update
    left_str = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in left_conversation
    )
    right_str = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in right_conversation
    )
    current_profile = get_learning_profile()
    profile_json = str({k: v for k, v in current_profile.items() if k != "id" and k != "updated_at"})

    update_prompt = f"""Analizza questa sessione di apprendimento e aggiorna il profilo di apprendimento.

PROFILO ATTUALE:
{profile_json}

CONVERSAZIONE DI LAVORO (pannello sinistro):
{left_str or "(nessuna conversazione)"}

CONVERSAZIONE CON IL PROFESSORE (pannello destro):
{right_str or "(nessuna conversazione)"}

Rispondi SOLO con un JSON valido in questo formato, senza markdown:
{{
  "come_funziona_claude": "...",
  "prompting": "...",
  "costi_modelli": "...",
  "apprendimento": "...",
  "pattern_negativi": "...",
  "concetti_consolidati": "...",
  "concetti_aperti": "...",
  "ultima_sessione": "riassunto in 2 righe di cosa è successo oggi"
}}

Regole:
- Se in questa sessione non si è toccato un'area, mantieni il valore attuale invariato
- Aggiungi solo ciò che emerge chiaramente dalla sessione
- Sii conciso: max 2-3 frasi per campo
- ultima_sessione: sempre aggiorna con un riassunto della sessione corrente"""

    try:
        import json
        client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": update_prompt}],
            max_tokens=1000,
            extra_headers={"HTTP-Referer": "https://professor.ai", "X-Title": "Professor"},
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        updated = json.loads(raw)
        update_learning_profile(updated)
    except Exception as e:
        print(f"Errore aggiornamento profilo: {e}")

    end_session(session_id, summary=right_str[:500] if right_str else "")
    return Response(status_code=200)
```

- [ ] **Step 2: Commit**

```bash
git add web/app.py
git commit -m "feat: session end route with automatic learning profile update"
```

---

## Task 13: Studio template (dual panel)

**Files:**
- Create: `web/templates/studio.html`

- [ ] **Step 1: Write web/templates/studio.html**

```html
{% extends "base.html" %}

{% block extra_css %}
html, body { height: 100%; overflow: hidden; }

.studio {
  display: flex;
  height: calc(100vh - 44px);
  overflow: hidden;
}

/* Panels */
.panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  min-width: 200px;
}

.panel-left  { flex: 0 0 50%; }
.panel-right {
  flex: 1;
  background: #0d0e12;
  border-left: 1px solid var(--border);
}

/* Resizer */
.resizer {
  width: 4px;
  background: var(--border);
  cursor: col-resize;
  flex-shrink: 0;
  transition: background 0.15s;
}
.resizer:hover, .resizer.dragging { background: var(--border2); }

/* Messages area */
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px 20px;
  scroll-behavior: smooth;
}

.messages::-webkit-scrollbar { width: 4px; }
.messages::-webkit-scrollbar-track { background: transparent; }
.messages::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* Message bubbles — identical to FilmMaker */
.msg { margin-bottom: 20px; display: flex; flex-direction: column; }

.msg-user {
  align-items: flex-end;
}
.msg-user .msg-body {
  background: #1c1c1e;
  border: 1px solid var(--border2);
  border-radius: 18px 18px 4px 18px;
  padding: 10px 14px;
  max-width: 72%;
  font-size: 14px;
  white-space: pre-wrap;
}

.msg-ai { align-items: flex-start; }
.msg-ai .msg-body {
  max-width: 82%;
  font-size: 14px;
  color: var(--text);
  line-height: 1.65;
}
.msg-ai .msg-body p  { margin: 0 0 10px; }
.msg-ai .msg-body p:last-child { margin-bottom: 0; }
.msg-ai .msg-body ul, .msg-ai .msg-body ol { margin: 8px 0 10px 20px; }
.msg-ai .msg-body li { margin-bottom: 4px; }
.msg-ai .msg-body code {
  background: var(--elevated);
  border-radius: 4px;
  padding: 1px 5px;
  font-size: 12px;
  font-family: 'SF Mono', 'Menlo', monospace;
}
.msg-ai .msg-body pre {
  background: var(--elevated);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  overflow-x: auto;
  margin: 8px 0;
}
.msg-ai .msg-body pre code { background: none; padding: 0; font-size: 12px; }
.msg-ai .msg-body strong { color: var(--accent); font-weight: 600; }

/* Typing indicator */
.typing { display: flex; gap: 4px; padding: 8px 0; align-items: center; }
.typing span {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--label); animation: blink 1.2s infinite;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%,80%,100%{opacity:.2} 40%{opacity:1} }

/* Input area */
.input-area {
  padding: 12px 16px;
  border-top: 1px solid var(--border);
  background: var(--bg);
}

.input-row {
  display: flex;
  gap: 8px;
  align-items: flex-end;
}

.chat-input {
  flex: 1;
  min-height: 40px;
  max-height: 120px;
  padding: 10px 14px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 22px;
  color: var(--text);
  font-family: var(--font);
  font-size: 14px;
  resize: none;
  outline: none;
  line-height: 1.5;
  transition: border-color 0.15s;
  overflow-y: auto;
}
.chat-input:focus { border-color: var(--border2); }
.chat-input::placeholder { color: var(--label); }

.send-btn {
  width: 36px; height: 36px;
  border-radius: 50%;
  background: var(--accent);
  border: none;
  cursor: pointer;
  flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  transition: opacity 0.15s;
}
.send-btn:hover { opacity: 0.85; }
.send-btn svg { width: 16px; height: 16px; }

/* Professor header indicator */
.prof-header {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
  color: var(--label);
  background: #0d0e12;
  min-height: 36px;
}
.prof-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--label);
  margin-bottom: 3px;
}
.prof-status { color: var(--text); font-size: 12px; }
{% endblock %}

{% block nav_left %}
  <a class="nav-link" href="/storia">Storico</a>
  <a class="nav-link" href="/profilo">Profilo</a>
{% endblock %}

{% block nav_right %}
  <div class="pill"><span class="dot"></span>Claude Opus</div>
  <a class="nav-link" href="/logout">Esci</a>
{% endblock %}

{% block content %}
<div class="studio" id="studio">

  <!-- LEFT PANEL: Work -->
  <div class="panel panel-left" id="panelLeft">
    <div class="messages" id="messagesLeft">
      <!-- populated by JS -->
    </div>
    <div class="input-area">
      <div class="input-row">
        <textarea class="chat-input" id="inputLeft"
                  placeholder="Scrivi a Claude..." rows="1"></textarea>
        <button class="send-btn" id="sendLeft" onclick="sendLeft()">
          <svg viewBox="0 0 24 24" fill="none" stroke="#0a0b0d" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </div>
    </div>
  </div>

  <!-- RESIZER -->
  <div class="resizer" id="resizer"></div>

  <!-- RIGHT PANEL: Professor -->
  <div class="panel panel-right" id="panelRight">
    <div class="prof-header">
      <div class="prof-label">Professore</div>
      <div class="prof-status" id="profStatus">
        {% set p = profile %}
        {% if p and p.concetti_consolidati %}
          Consolidato: {{ p.concetti_consolidati[:80] }}{% if p.concetti_consolidati|length > 80 %}...{% endif %}
        {% elif p and p.ultima_sessione %}
          {{ p.ultima_sessione[:100] }}
        {% else %}
          Prima sessione — il professore impara mentre lavori
        {% endif %}
      </div>
    </div>
    <div class="messages" id="messagesRight">
      <!-- populated by JS -->
    </div>
    <div class="input-area">
      <div class="input-row">
        <textarea class="chat-input" id="inputRight"
                  placeholder="Chiedi al professore..." rows="1"></textarea>
        <button class="send-btn" id="sendRight" onclick="sendRight()">
          <svg viewBox="0 0 24 24" fill="none" stroke="#0a0b0d" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </div>
    </div>
  </div>

</div>
{% endblock %}

{% block scripts %}
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
const SESSION_ID = {{ session_id }};
let convLeft  = [];
let convRight = [];
let streaming  = false;

// ── Markdown renderer ──
function renderMarkdown(text) {
  return marked.parse(text, { breaks: true, gfm: true });
}

// ── Auto-resize textarea ──
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}
document.getElementById('inputLeft').addEventListener('input', function(){ autoResize(this); });
document.getElementById('inputRight').addEventListener('input', function(){ autoResize(this); });

// ── Enter to send ──
document.getElementById('inputLeft').addEventListener('keydown', function(e){
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendLeft(); }
});
document.getElementById('inputRight').addEventListener('keydown', function(e){
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendRight(); }
});

// ── Append message to panel ──
function appendMsg(containerId, role, text, isStream) {
  const container = document.getElementById(containerId);
  const div = document.createElement('div');
  div.className = 'msg msg-' + (role === 'user' ? 'user' : 'ai');
  const body = document.createElement('div');
  body.className = 'msg-body';
  if (role === 'user') {
    body.textContent = text;
  } else {
    body.innerHTML = isStream ? '' : renderMarkdown(text);
  }
  div.appendChild(body);
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return body;
}

// ── Typing indicator ──
function showTyping(containerId) {
  const container = document.getElementById(containerId);
  const div = document.createElement('div');
  div.className = 'msg msg-ai';
  div.id = 'typing-' + containerId;
  div.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}
function hideTyping(containerId) {
  const el = document.getElementById('typing-' + containerId);
  if (el) el.remove();
}

// ── SSE stream reader ──
async function readStream(url, payload, containerId, onDone) {
  showTyping(containerId);
  let bodyEl = null;
  let fullText = '';

  const resp = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6).replace(/\\n/g, '\n');
      if (data === '[DONE]') {
        hideTyping(containerId);
        if (bodyEl) bodyEl.innerHTML = renderMarkdown(fullText);
        if (onDone) onDone(fullText);
        return;
      }
      if (!bodyEl) {
        hideTyping(containerId);
        bodyEl = appendMsg(containerId, 'assistant', '', true);
      }
      fullText += data;
      bodyEl.textContent = fullText;
      document.getElementById(containerId).scrollTop =
        document.getElementById(containerId).scrollHeight;
    }
  }
  hideTyping(containerId);
  if (bodyEl) bodyEl.innerHTML = renderMarkdown(fullText);
  if (onDone) onDone(fullText);
}

// ── Send left ──
async function sendLeft() {
  if (streaming) return;
  const input = document.getElementById('inputLeft');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  input.style.height = 'auto';

  convLeft.push({ role: 'user', content: text });
  appendMsg('messagesLeft', 'user', text);

  streaming = true;
  await readStream('/chat/sinistra', {
    conversation: convLeft,
    session_id: SESSION_ID,
  }, 'messagesLeft', (fullText) => {
    convLeft.push({ role: 'assistant', content: fullText });
    streaming = false;
  });
}

// ── Send right ──
async function sendRight() {
  if (streaming) return;
  const input = document.getElementById('inputRight');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  input.style.height = 'auto';

  convRight.push({ role: 'user', content: text });
  appendMsg('messagesRight', 'user', text);

  streaming = true;
  await readStream('/chat/destra', {
    conversation: convRight,
    left_context: convLeft.slice(-20),
    session_id: SESSION_ID,
  }, 'messagesRight', (fullText) => {
    convRight.push({ role: 'assistant', content: fullText });
    streaming = false;
  });
}

// ── Panel resizer ──
(function() {
  const resizer = document.getElementById('resizer');
  const left    = document.getElementById('panelLeft');
  const studio  = document.getElementById('studio');
  let isDown = false, startX = 0, startW = 0;

  resizer.addEventListener('mousedown', e => {
    isDown = true;
    startX = e.clientX;
    startW = left.offsetWidth;
    resizer.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });

  document.addEventListener('mousemove', e => {
    if (!isDown) return;
    const totalW = studio.offsetWidth - 4; // minus resizer width
    const newW = Math.min(Math.max(startW + (e.clientX - startX), 200), totalW - 200);
    left.style.flex = 'none';
    left.style.width = newW + 'px';
  });

  document.addEventListener('mouseup', () => {
    if (isDown) {
      isDown = false;
      resizer.classList.remove('dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }
  });
})();

// ── Session end beacon ──
window.addEventListener('beforeunload', () => {
  if (convLeft.length === 0 && convRight.length === 0) return;
  const payload = JSON.stringify({
    session_id: SESSION_ID,
    left_conversation: convLeft,
    right_conversation: convRight,
  });
  navigator.sendBeacon('/sessione/fine', new Blob([payload], { type: 'application/json' }));
});
</script>
{% endblock %}
```

- [ ] **Step 2: Commit**

```bash
git add web/templates/studio.html
git commit -m "feat: dual-panel studio template with resizable panels and SSE streaming"
```

---

## Task 14: Storia + Profilo + Storia detail templates

**Files:**
- Create: `web/templates/storia.html`
- Create: `web/templates/storia_detail.html`
- Create: `web/templates/profilo.html`

- [ ] **Step 1: Write web/templates/storia.html**

```html
{% extends "base.html" %}

{% block extra_css %}
.page { max-width: 720px; margin: 0 auto; padding: 40px 20px; }
.page-title { font-size: 20px; font-weight: 600; color: var(--accent); margin-bottom: 24px; }
.session-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  text-decoration: none;
  transition: border-color 0.15s;
}
.session-card:hover { border-color: var(--border2); }
.session-date { font-size: 13px; color: var(--label); }
.session-summary { font-size: 14px; color: var(--text); margin-top: 4px; }
.session-arrow { color: var(--label); font-size: 18px; }
.empty { color: var(--label); font-size: 14px; text-align: center; padding: 60px 0; }
{% endblock %}

{% block nav_left %}
  <a class="nav-link" href="/studio">Studio</a>
{% endblock %}
{% block nav_right %}
  <a class="nav-link" href="/profilo">Profilo</a>
  <a class="nav-link" href="/logout">Esci</a>
{% endblock %}

{% block content %}
<div class="page">
  <div class="page-title">Sessioni passate</div>
  {% if sessions %}
    {% for s in sessions %}
    <a class="session-card" href="/storia/{{ s.id }}">
      <div>
        <div class="session-date">{{ s.started_at[:16] }}</div>
        <div class="session-summary">
          {% if s.summary %}{{ s.summary[:100] }}{% else %}Sessione senza note{% endif %}
        </div>
      </div>
      <div class="session-arrow">›</div>
    </a>
    {% endfor %}
  {% else %}
    <div class="empty">Nessuna sessione ancora. Vai allo <a href="/studio" style="color:var(--accent)">Studio</a> per iniziare.</div>
  {% endif %}
</div>
{% endblock %}
```

- [ ] **Step 2: Write web/templates/storia_detail.html**

```html
{% extends "base.html" %}

{% block extra_css %}
.page { max-width: 960px; margin: 0 auto; padding: 40px 20px; }
.page-title { font-size: 18px; font-weight: 600; color: var(--accent); margin-bottom: 4px; }
.page-sub { font-size: 13px; color: var(--label); margin-bottom: 32px; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.panel-title {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--label); margin-bottom: 12px;
}
.msg-hist { margin-bottom: 14px; }
.msg-hist-role {
  font-size: 11px; font-weight: 600; color: var(--label);
  text-transform: uppercase; margin-bottom: 3px;
}
.msg-hist-content { font-size: 14px; color: var(--text); white-space: pre-wrap; }
{% endblock %}

{% block nav_left %}
  <a class="nav-link" href="/studio">Studio</a>
  <a class="nav-link" href="/storia">Storico</a>
{% endblock %}
{% block nav_right %}
  <a class="nav-link" href="/logout">Esci</a>
{% endblock %}

{% block content %}
<div class="page">
  <div class="page-title">Sessione del {{ sess.started_at[:16] }}</div>
  <div class="page-sub">
    {% if sess.ended_at %}Terminata {{ sess.ended_at[:16] }}{% endif %}
  </div>
  <div class="panels">
    <div>
      <div class="panel-title">Pannello lavoro</div>
      {% for m in msgs_left %}
      <div class="msg-hist">
        <div class="msg-hist-role">{{ m.role }}</div>
        <div class="msg-hist-content">{{ m.content }}</div>
      </div>
      {% endfor %}
      {% if not msgs_left %}<div style="color:var(--label);font-size:14px;">Nessun messaggio</div>{% endif %}
    </div>
    <div>
      <div class="panel-title">Professore</div>
      {% for m in msgs_right %}
      <div class="msg-hist">
        <div class="msg-hist-role">{{ m.role }}</div>
        <div class="msg-hist-content">{{ m.content }}</div>
      </div>
      {% endfor %}
      {% if not msgs_right %}<div style="color:var(--label);font-size:14px;">Nessun messaggio</div>{% endif %}
    </div>
  </div>
</div>
{% endblock %}
```

- [ ] **Step 3: Write web/templates/profilo.html**

```html
{% extends "base.html" %}

{% block extra_css %}
.page { max-width: 720px; margin: 0 auto; padding: 40px 20px; }
.page-title { font-size: 20px; font-weight: 600; color: var(--accent); margin-bottom: 8px; }
.page-sub { font-size: 13px; color: var(--label); margin-bottom: 32px; }
.profile-section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 12px;
}
.section-label {
  font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--label); margin-bottom: 6px;
}
.section-value { font-size: 14px; color: var(--text); line-height: 1.6; }
.section-empty { font-size: 13px; color: var(--label); font-style: italic; }
.updated { font-size: 12px; color: var(--label); text-align: right; margin-top: 16px; }
{% endblock %}

{% block nav_left %}
  <a class="nav-link" href="/studio">Studio</a>
  <a class="nav-link" href="/storia">Storico</a>
{% endblock %}
{% block nav_right %}
  <a class="nav-link" href="/logout">Esci</a>
{% endblock %}

{% block content %}
<div class="page">
  <div class="page-title">Profilo di apprendimento</div>
  <div class="page-sub">Aggiornato automaticamente dopo ogni sessione</div>

  {% set fields = [
    ("come_funziona_claude", "Come funziona Claude"),
    ("prompting", "Prompting"),
    ("costi_modelli", "Costi e modelli"),
    ("apprendimento", "Apprendimento"),
    ("concetti_consolidati", "Concetti consolidati"),
    ("concetti_aperti", "Da esplorare"),
    ("pattern_negativi", "Pattern da correggere"),
    ("ultima_sessione", "Ultima sessione"),
  ] %}

  {% for key, label in fields %}
  <div class="profile-section">
    <div class="section-label">{{ label }}</div>
    {% set val = profile[key] if key in profile else "" %}
    {% if val %}
      <div class="section-value">{{ val }}</div>
    {% else %}
      <div class="section-empty">Non ancora esplorato</div>
    {% endif %}
  </div>
  {% endfor %}

  {% if profile.updated_at %}
  <div class="updated">Ultimo aggiornamento: {{ profile.updated_at[:16] }}</div>
  {% endif %}
</div>
{% endblock %}
```

- [ ] **Step 4: Commit**

```bash
git add web/templates/storia.html web/templates/storia_detail.html web/templates/profilo.html
git commit -m "feat: storia, storia_detail, and profilo templates"
```

---

## Task 15: Smoke test + Railway deploy

**Files:**
- No new files — verification and deploy

- [ ] **Step 1: Install dependencies locally**

```bash
cd /Users/filippoarici/projects/Professor
pip install -r requirements.txt -q
```
Expected: all packages installed

- [ ] **Step 2: Start server locally**

```bash
cd /Users/filippoarici/projects/Professor
DATABASE_URL=/tmp/professor_test.db ACCESS_PASSWORD=test SECRET_KEY=test-key OPENROUTER_API_KEY=dummy uvicorn web.app:app --port 8001 --reload &
sleep 2
curl -s http://localhost:8001/ -I | head -5
```
Expected: HTTP 200 or 303 redirect

- [ ] **Step 3: Verify login page**

```bash
curl -s http://localhost:8001/login | grep -i "Professor"
```
Expected: HTML containing "Professor"

- [ ] **Step 4: Kill test server**

```bash
kill %1 2>/dev/null; rm -f /tmp/professor_test.db
```

- [ ] **Step 5: Create GitHub repo**

```bash
cd /Users/filippoarici/projects/Professor
gh repo create professor-ai --private --source=. --push
```
Expected: repo created and code pushed

- [ ] **Step 6: Deploy to Railway**

```bash
cd /Users/filippoarici/projects/Professor
railway login --browserless  # if not already logged in
railway init --name professor-ai
railway up --detach
```

- [ ] **Step 7: Set Railway environment variables**

```bash
railway variables set OPENROUTER_API_KEY="<the-key>"
railway variables set ACCESS_PASSWORD="<password>"
railway variables set SECRET_KEY="<random-string>"
railway variables set DATABASE_URL="/data/professor.db"
```

- [ ] **Step 8: Add Railway volume for SQLite**

In Railway dashboard: add a volume mounted at `/data` to the professor-ai service.

- [ ] **Step 9: Final commit**

```bash
cd /Users/filippoarici/projects/Professor
git add -A
git commit -m "chore: all templates and routes complete — ready for Railway deploy"
git push
```

---

## Self-Review

**Spec coverage check:**
- ✅ Single password access (no registration) → Task 8, auth.py
- ✅ GET `/` → `/studio` → Task 9
- ✅ GET `/studio` → dual-panel → Task 13
- ✅ POST `/chat/sinistra` SSE → Task 10
- ✅ POST `/chat/destra` SSE with profile + RAG → Task 11
- ✅ POST `/sessione/fine` profile update → Task 12
- ✅ GET `/storia` + detail → Tasks 9, 14
- ✅ GET `/profilo` → Tasks 9, 14
- ✅ 4 DB tables (sessions, messages_left, messages_right, learning_profile) → Task 2
- ✅ BM25 RAG on corpus → Tasks 4, 6
- ✅ 13 corpus files across 4 folders → Task 6
- ✅ Professor system prompt with Feynman method → Task 5
- ✅ Learning profile updated at session end via beacon → Task 12, studio.html
- ✅ Dual-panel with drag resizer (col-resize) → Task 13
- ✅ Right panel background #0d0e12 → Task 13
- ✅ Identical aesthetic to FilmMaker (#0a0b0d, Apple fonts) → Task 7
- ✅ Railway deploy with /data volume for SQLite → Task 15
- ✅ Env vars: OPENROUTER_API_KEY, ACCESS_PASSWORD, SECRET_KEY → Task 15
- ✅ OpenRouter Claude Opus model → app.py MODEL constant

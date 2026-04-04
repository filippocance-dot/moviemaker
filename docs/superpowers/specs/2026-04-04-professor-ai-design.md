# Professor AI — Design Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Applicazione web standalone con interfaccia a due pannelli — sinistra per lavorare con Claude, destra per un professore AI che spiega in tempo reale cosa sta succedendo, insegna a usare l'AI al massimo e ricorda nel tempo tutto ciò che ha già insegnato.

**Architecture:** FastAPI + SQLite + Railway, stesso stack di FilmMaker ma progetto separato. Due istanze Claude per sessione (lavoro + professore). RAG BM25 su corpus specializzato. Memoria progressiva dell'apprendimento aggiornata dopo ogni sessione.

**Tech Stack:** Python, FastAPI, Jinja2, SQLite, OpenRouter (Claude Opus), BM25, Railway

---

## Utente e contesto

Utente singolo: Filippo Arici, consulente finanziario che costruisce prodotti AI e vuole padroneggiare Claude in profondità. Nessun sistema di registrazione — accesso diretto con password singola.

---

## Interfaccia

### Schermata principale `/studio`

Pannello doppio ridimensionabile (drag sul bordo centrale, default 50/50).

**Pannello sinistro — Lavoro**
- Chat con Claude Opus (OpenRouter)
- Streaming SSE identico a FilmMaker
- Upload file (PDF, immagini, testo)
- Nessun sistema RAG lato sinistro — Claude libero

**Pannello destro — Professore**
- Chat separata, attivata solo quando Filippo scrive
- In alto: indicatore di apprendimento — una riga che riassume il livello attuale (es. "Comprende il context window. Da esplorare: costi e caching.")
- Il professore riceve come contesto: profilo apprendimento + ultimi 20 messaggi della sinistra + intera conversazione destra della sessione corrente
- Risponde in italiano, con metodo Feynman, una cosa alla volta

---

## Database

```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT DEFAULT (datetime('now')),
    ended_at TEXT,
    summary TEXT  -- generato dal professore a fine sessione
);

CREATE TABLE messages_left (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    role TEXT,  -- 'user' | 'assistant'
    content TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE messages_right (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    role TEXT,  -- 'user' | 'assistant'
    content TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE learning_profile (
    id INTEGER PRIMARY KEY DEFAULT 1,
    come_funziona_claude TEXT DEFAULT '',
    prompting TEXT DEFAULT '',
    costi_modelli TEXT DEFAULT '',
    apprendimento TEXT DEFAULT '',
    pattern_negativi TEXT DEFAULT '',  -- errori ricorrenti
    concetti_consolidati TEXT DEFAULT '',
    concetti_aperti TEXT DEFAULT '',
    ultima_sessione TEXT DEFAULT '',
    updated_at TEXT DEFAULT (datetime('now'))
);
```

---

## Routes

| Method | Path | Descrizione |
|--------|------|-------------|
| GET | `/` | Redirect a `/studio` |
| GET | `/studio` | Schermata principale a due pannelli |
| POST | `/chat/sinistra` | Streaming SSE — conversazione lavoro |
| POST | `/chat/destra` | Streaming SSE — professore |
| POST | `/upload` | Upload file per pannello sinistro |
| POST | `/sessione/fine` | Fine sessione: aggiorna profilo apprendimento |
| GET | `/storia` | Lista sessioni passate |
| GET | `/profilo` | Vista profilo apprendimento corrente |

---

## System prompt del professore

```
Sei un professore personale che insegna a Filippo come usare Claude al massimo del suo potenziale.

Il tuo approccio:
- Parti sempre da ciò che Filippo ha già capito (vedi profilo apprendimento)
- Non ripetere mai cose già consolidate
- Usa il metodo Feynman: prima la versione semplice, poi la profondità
- Una cosa alla volta — mai sovraccaricare
- Dopo ogni spiegazione chiedi: "ha senso? vuoi approfondire qualcosa?"
- Quando vedi un pattern negativo ricorrente, segnalalo con gentilezza e mostra l'alternativa
- Sul tema costi: sempre numeri concreti, confronti reali, esempi pratici
- Hai accesso alla conversazione di lavoro di sinistra — usala come materiale didattico reale

Aree di insegnamento:
1. Come funziona Claude — architettura, context window, limiti, allucinazioni
2. Prompting — struttura, few-shot, chain of thought, iterazione, system prompt
3. Costi e modelli — Opus/Sonnet/Haiku, pricing a token, caching, batch API
4. Apprendimento — Feynman, modelli mentali, consolidamento competenze

Rispondi sempre in italiano.
```

---

## Corpus RAG

**`docs/come_funziona_claude/`**
- `context_window.txt` — cos'è, come si esaurisce, cosa succede
- `elaborazione_istruzioni.txt` — come processo le richieste
- `limiti_allucinazioni.txt` — dove fallisco, quando non fidarsi
- `variabilita_risposte.txt` — perché rispondo diversamente a domande simili

**`docs/prompting/`**
- `struttura_prompt_efficace.txt` — ruolo, contesto, compito, formato
- `few_shot_prompting.txt` — dare esempi per guidare la risposta
- `chain_of_thought.txt` — ragionamento passo per passo
- `system_prompt.txt` — cos'è e come usarlo
- `iterazione.txt` — affinare invece di ricominciare

**`docs/costi_modelli/`**
- `opus_sonnet_haiku.txt` — quando usare quale, costi reali
- `pricing_token.txt` — input vs output, perché l'output costa di più
- `prompt_caching.txt` — cos'è, risparmio fino all'80%, come attivarlo
- `batch_api.txt` — sconto 50% per task non urgenti
- `api_vs_interfaccia.txt` — quando conviene passare all'API diretta

**`docs/apprendimento/`**
- `metodo_feynman.txt` — spiegare per capire davvero
- `modelli_mentali.txt` — costruire comprensione duratura
- `spaced_repetition.txt` — consolidare nel tempo
- `imitare_vs_capire.txt` — come passare dall'una all'altra

---

## Aggiornamento profilo apprendimento

Al termine di ogni sessione (beacon `beforeunload`), il professore riceve l'intera sessione e genera un aggiornamento strutturato:

```
Analizza questa sessione e aggiorna il profilo di apprendimento.
Rispondi SOLO con JSON in questo formato:
{
  "come_funziona_claude": "...",  // aggiorna o mantieni
  "prompting": "...",
  "costi_modelli": "...",
  "apprendimento": "...",
  "pattern_negativi": "...",
  "concetti_consolidati": "...",
  "concetti_aperti": "...",
  "ultima_sessione": "riassunto in 2 righe"
}
```

---

## Estetica

Identica a FilmMaker: palette `#0a0b0d`, font Apple system, bordi sottili.
Unica differenza: nav mostra "Professor" invece di "FilmMaker".
Bordo centrale ridimensionabile con cursore `col-resize`.
Pannello destro ha sfondo leggermente diverso (`#0d0e12`) per distinguerlo visivamente.

---

## Deploy

- Repository GitHub separato: `professor-ai`
- Deploy Railway: servizio separato, volume `/data` per SQLite
- Variabili d'ambiente: `OPENROUTER_API_KEY`, `ACCESS_PASSWORD`, `SECRET_KEY`
- Accesso: singola password (no registrazione, no approvazione)

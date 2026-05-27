# FilmMaker — Le Due Stanze
*Design spec — 27 maggio 2026*

## Contesto

FilmMaker oggi è un dramaturg AI che guida l'autore dall'idea alla struttura del film (modalità NUCLEO → STRUTTURA → SCRITTURA). Manca la capacità di produrre una **sceneggiatura completa, formalmente perfetta, vendibile a un produttore** (standard CSC/RAI italiano, cortometraggio).

L'autore (regista/visionario) vuole un "team": l'AI creativa attuale + un nuovo "sceneggiatore tecnico" che trasforma il materiale in documento di formato professionale.

## Obiettivo

Aggiungere a FilmMaker un secondo agente AI specializzato nella stesura formale della sceneggiatura, senza alterare l'esperienza creativa esistente. I due agenti vivono nello stesso Progetto, hanno ruoli rigidamente separati, e si passano materiale automaticamente.

## Visione utente

L'utente è il regista. FilmMaker è il suo team:

- **Suso** *(la dramaturg, in onore di Suso Cecchi d'Amico)* — quello che fa oggi. Tono morbido, esplorativo, paziente. Lavora su nucleo, struttura, scene grezze. Vive nella stanza `Sviluppo`.
- **Furio** *(lo sceneggiatore tecnico, in onore di Furio Scarpelli)* — nuovo. Tono asciutto, professionale, da editor severo. Prende le decisioni di Suso e produce la sceneggiatura in formato CSC/RAI. Vive nella stanza `Sceneggiatura`.

Le due stanze sono due tab dentro la pagina di un Progetto. L'utente sa sempre con chi sta parlando (banner + colore distintivo).

## Architettura

### Le due stanze

Dentro un Progetto:

```
[Sviluppo (Suso)]  [Sceneggiatura (Furio)]
       ↑                    ↑
       └── stesso progetto, materiale condiviso ──┘
```

- **Sviluppo** esiste come oggi. Chat con Suso, file caricati, RAG, profilo autore.
- **Sceneggiatura** è la nuova stanza. Disponibile solo dentro un Progetto (non in chat libera).
- L'utente naviga via tab nell'header del progetto.

### Confini ruoli (rigorosi)

- Quando l'utente chiede a Furio una modifica creativa ("la motivazione di Marco non regge"), Furio risponde: *"Questo non è il mio mestiere. Torna da Suso, ne parlate insieme, poi torna qui quando la scena ha una sua verità."*
- Suso non genera PDF. Furio non riapre temi narrativi.

### Flusso utente tipico

```
Crea Progetto
    ↓
Tab Sviluppo: chatta con Suso  →  struttura + scene grezze + decisioni
    ↓
Tab Sceneggiatura: Furio ha già letto tutto
    ↓
Furio produce Fountain → "Genera PDF" → download
    ↓
Serve modifica creativa? → torna in Sviluppo da Suso
```

### Due punti d'ingresso alla Sceneggiatura

1. **Continua dallo Sviluppo:** Furio riceve automaticamente le decisioni di Suso (estratte dalla sua chat) + file del progetto.
2. **Parto da un draft mio:** l'utente carica .docx/.txt con scene scritte male. Furio le riformatta. Senza passare da Suso.

## Componenti tecnici

### File da creare/modificare

| File | Tipo | Note |
|---|---|---|
| `cineauteur.py` | rename | `SYSTEM_PROMPT` → `SUSO_PROMPT` (semantica esplicita) |
| `cineauteur.py` | nuovo | `FURIO_PROMPT` — system prompt dello sceneggiatore tecnico |
| `web/screenplay.py` | nuovo modulo | Fountain → PDF deterministico (template CSC/RAI). ~200 righe. |
| `web/screenplay_validator.py` | nuovo modulo | `validate_fountain(text) → list[Error]` |
| `web/templates/screenplay.html` | nuovo template | tab Sceneggiatura: chat con Furio + bottone "Genera PDF" |
| `web/templates/projects.html` | modifica | aggiunta dei due tab dentro il dettaglio progetto |
| `web/templates/screenplay_pdf.html` | nuovo template Jinja | layout HTML/CSS CSC/RAI per weasyprint |
| `web/app.py` | nuove route + regex prompt | `GET/POST /progetti/<id>/sceneggiatura`, `GET /progetti/<id>/sceneggiatura/pdf`. Inoltre: aggiornare regex `SYSTEM_PROMPT` → `SUSO_PROMPT` (rinominato) e aggiungere regex parallela per `FURIO_PROMPT`. |
| `web/db.py` | modifica | nuova tabella `screenplays(id, project_id, fountain_source, model, created_at, updated_at)` |
| `requirements.txt` + `requirements-web.txt` | aggiunta | `weasyprint` |
| `tests/test_screenplay_validator.py` | nuovo | unit test del linter |
| `tests/test_screenplay_render.py` | nuovo | snapshot test del rendering PDF |

### Da NON toccare

- Chat libera (`/chat`)
- Suso esistente (a parte rename del prompt)
- Auth, sistema utenti, sistema progetti base
- RAG, profilo autore, storico sessioni

## Data flow

### Cosa vede Suso (invariato)

- system prompt = `SUSO_PROMPT`
- messaggi utente + file del progetto + chunk RAG + storico conversazione

### Cosa vede Furio

All'apertura della stanza Sceneggiatura, primo messaggio composto da:

1. file del progetto (come Suso)
2. **sintesi strutturata** delle decisioni prese con Suso, estratte automaticamente
3. eventuale draft grezzo caricato dall'utente
4. **NO** storico chat completo di Suso (rumore + costo + fuori scope)

System prompt = `FURIO_PROMPT`.

### Estrazione decisioni Suso → Furio

Suso ha già nella sua `BLOCCO DECISIONALE` sezione la convenzione `"Decisione presa: [X]"`. Funzione nuova:

```python
def extract_decisions(conversation: list[dict]) -> str:
    """Estrae i blocchi 'Decisione presa: …' dalla chat di Sviluppo."""
```

Output: blocco testuale tipo:

```
📋 Materiale passato da Suso:
- Nucleo: [frase del film]
- Scene confermate: [lista]
- Decisioni chiave: [punti elenco]
```

Visibile all'utente in fondo alla chat di Furio (trasparenza).

## Strategia "perfezione formale"

### Furio scrive Fountain, non PDF

[Fountain](https://fountain.io) è un formato testo semplicissimo, standard de facto dell'industria. Regole minime che un LLM segue bene:

```fountain
INT. CUCINA - GIORNO

Marco apre il frigo. Vuoto. Lo richiude piano.

MARCO
(quasi tra sé)
Niente latte.

ANNA (V.O.)
L'ho finito ieri.
```

### Renderer deterministico

`weasyprint` (HTML/CSS → PDF) con un template Jinja2 che applica gli stili CSC/RAI italiani:
- Courier 12pt
- Margini italiani standard
- Slugline numerati a destra
- Indentazioni personaggio/dialogo/parentetica/azione corrette

Alternative scartate:
- `screenplain`: layout americano, poco personalizzabile per CSC/RAI
- `reportlab`: più verboso, programmatico
- LaTeX: dipendenza enorme, sproporzionata

### Validazione (linter Fountain)

`validate_fountain(text) → list[Error]` controlla:
- ogni scena ha uno slugline ben formato (`INT.`/`EST.` + LUOGO + ` - ` + tempo)
- nomi personaggi in maiuscolo, senza punti
- parentetiche sotto un nome personaggio, mai da sole
- lunghezze dialogo ragionevoli
- nessuna riga "impossibile" (es. parentetica seguita da slugline)

### Retry automatico

Se validate fallisce:
1. Backend reinette automaticamente: *"Errori trovati: [lista]. Correggi mantenendo il senso."*
2. Max 2 retry.
3. Al terzo fallimento: messaggio chiaro all'utente + log lato server + opzione "riprova".

Senza validate, ogni errore di formato arriverebbe all'utente. Con validate, la maggior parte viene catturata e corretta automaticamente.

## Modelli AI

| Stanza | Modello | Razionale |
|---|---|---|
| Suso (Sviluppo) | `claude-sonnet-4-6` | Creatività + costi sostenibili per chat lunghe. Invariato. |
| Furio (Sceneggiatura) | `claude-opus-4-7` | Rigore formale + coerenza su 15-25 pp. ~5x più caro ma giustificato dalla "perfezione formale". |

Model switcher esistente nell'UI rimane attivo anche per Furio (default Opus, override possibile).

## Prompt Furio — struttura

Sezioni del nuovo `FURIO_PROMPT`:

1. **IDENTITÀ** — sei Furio, sceneggiatore di mestiere. Asciutto, professionale, severo ma giusto.
2. **LINGUA** — sempre italiano.
3. **CONFINI** — non ripensi la storia. Per modifiche creative rimandi a Suso.
4. **INTERIORITÀ → AZIONE FILMABILE** — la regola appena aggiunta a Suso (cruciale per pulire i draft grezzi).
5. **FORMATO FOUNTAIN** — sintassi con esempi italiani (slugline, parentetiche, V.O., F.C.).
6. **REGOLE STILISTICHE CSC/RAI** — convenzioni italiane (numerazione, terminologia).
7. **OUTPUT** — solo Fountain. Mai prosa esplicativa. Mai meta-commenti.
8. **GESTIONE ERRORI** — se ricevi errori dal validatore, correggi senza riaprire il senso.

Sezioni esplicitamente **assenti** rispetto a Suso (per evitare drift creativo):
- CLASSIFICAZIONE MATERIALE / VINCOLI NEGATIVI / NUCLEO-STRUTTURA-SCRITTURA
- Controllo divergenza/convergenza (Furio non propone alternative creative)
- Gestione riaperture (Furio non riapre temi narrativi)

## Storage

Nuova tabella SQLite:

```sql
CREATE TABLE screenplays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    fountain_source TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Una sola sceneggiatura corrente per progetto (no versioning multiplo nel MVP).

Il PDF viene **rigenerato on-demand** da `fountain_source`, non salvato.

## Edge cases

| Caso | Comportamento |
|---|---|
| Sceneggiatura aperta senza materiale (no chat Suso, no draft) | Furio chiede: "Carica un draft o torna in Sviluppo." |
| Draft >30 pp | Avviso: "Furio è per cortometraggi. Procedo, ma alcune scene potrebbero essere accorpate." |
| Fountain LLM fallisce dopo 2 retry | Errore chiaro + opzione "riprova" + log server |
| Richiesta modifica creativa a Furio | Risposta scriptata: rimanda a Suso, con link al tab Sviluppo |
| weasyprint crash | Errore + fallback download Fountain raw |
| Volume Railway pieno | PDF generato in memoria, mai su disco → no impatto |
| Connessione persa durante streaming | Gestione esistente di Suso, riutilizzata |

## Testing

- **Unit:** `validate_fountain()` con 10-15 casi (corretti + malformati)
- **Unit:** `screenplay.py` rendering (snapshot PDF su input fisso)
- **Integration:** route `POST /progetti/<id>/sceneggiatura/stream` con progetto di prova
- **Manuale end-to-end:** creare progetto → chat con Suso → passare a Furio → PDF → verifica visiva formato

## Deploy

1. Edit locale + test con `python -m uvicorn web.app:app --reload`
2. `requirements.txt` + `requirements-web.txt`: aggiunta `weasyprint`
3. **Verifica preventiva:** weasyprint richiede libs di sistema (fontconfig, cairo, pango). Da testare in container Railway PRIMA del deploy in produzione (è un punto di rischio noto).
4. `railway up --detach` solo dopo che funziona in locale + container test
5. Verifica spazio disco Railway dopo deploy (problema noto: ~0.73 MB liberi)

## Fuori scope (MVP)

- Editor inline della sceneggiatura
- Revisione scena per scena con Furio
- Export `.docx` o `.fountain`
- Pacchetto pitch (logline + soggetto + trattamento)
- Lungometraggi
- Suggerimenti automatici "ora sei pronto per Furio!"
- Watermark, firma digitale, copyright
- Versioning multiplo sceneggiature

## Decisioni aperte (non bloccanti per il MVP)

- Tono esatto di Furio: serve un round di "vibe testing" con prompt iniziali prima del go-live
- Nomi Suso/Furio: confermati come proposta, modificabili dopo i primi test utente
- Colori/icone distintivi nell'UI per le due personalità: da definire con un piccolo test visivo

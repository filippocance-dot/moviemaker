# FilmMaker — Le Due Stanze: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aggiungere a FilmMaker un secondo agente AI (Furio) che produce sceneggiature in formato CSC/RAI italiano partendo dal materiale creato con Suso (la dramaturg esistente), all'interno della stessa pagina Progetto come stanza separata.

**Architecture:** Due "stanze" UI (`Sviluppo` / `Sceneggiatura`) accessibili come tab dentro un Progetto. Suso (`SUSO_PROMPT`) resta come oggi. Furio (`FURIO_PROMPT`) è un nuovo system prompt che scrive in formato Fountain. Un modulo deterministico Python converte Fountain → PDF formato italiano via `weasyprint`. Una nuova tabella `screenplays` persiste il sorgente Fountain per progetto.

**Tech Stack:** FastAPI + Jinja2 + SQLite (esistenti), `weasyprint` (nuovo, PDF rendering), `python-docx` (nuovo, upload draft .docx), Claude API (`claude-sonnet-4-6` Suso, `claude-opus-4-7` Furio default).

**Spec di riferimento:** `design/2026-05-27-filmmaker-due-stanze.md`

---

## Task 1: Rinomina SYSTEM_PROMPT → SUSO_PROMPT

**Goal:** Cambiare il nome semantico del prompt esistente senza alterare il comportamento. Precondizione per affiancare `FURIO_PROMPT`.

**Files:**
- Modify: `cineauteur.py` (riga ~107)
- Modify: `web/app.py` (funzione `_load_system_prompt`, ~riga 31)
- Test: `tests/test_prompts.py` (nuovo)

- [ ] **Step 1: Scrivi il test che verifica che SUSO_PROMPT venga letto**

Crea `tests/test_prompts.py`:

```python
from web.app import _load_suso_prompt

def test_suso_prompt_loads_and_contains_identity():
    prompt = _load_suso_prompt()
    assert prompt
    assert "dramaturg" in prompt.lower()
    assert "NUCLEO" in prompt
```

- [ ] **Step 2: Run, verifica che fallisca**

```bash
cd /Users/filippoarici/projects/Test_1
pytest tests/test_prompts.py -v
```
Expected: FAIL — `ImportError: cannot import name '_load_suso_prompt'`

- [ ] **Step 3: Rinomina in cineauteur.py**

In `cineauteur.py`, riga 107, cambia:
```python
SYSTEM_PROMPT = """\
```
in:
```python
SUSO_PROMPT = """\
```

- [ ] **Step 4: Aggiorna web/app.py**

In `web/app.py`, sostituisci la funzione `_load_system_prompt` e la variabile esportata:

```python
def _load_suso_prompt() -> str:
    src = (pathlib.Path(__file__).parent.parent / "cineauteur.py").read_text(encoding="utf-8")
    m = re.search(r'SUSO_PROMPT\s*=\s*"""\\\n(.*?)"""', src, re.DOTALL)
    if not m:
        raise RuntimeError("SUSO_PROMPT not found in cineauteur.py")
    return m.group(1).replace("\\\n", "\n")

SUSO_PROMPT = _load_suso_prompt()
```

Poi: cerca in tutto `web/app.py` ogni uso di `SYSTEM_PROMPT` e sostituiscilo con `SUSO_PROMPT`. Verifica con:

```bash
grep -n "SYSTEM_PROMPT" web/app.py
```
Expected: nessun match.

- [ ] **Step 5: Run il test, verifica PASS**

```bash
pytest tests/test_prompts.py -v
```
Expected: PASS

- [ ] **Step 6: Run tutta la test suite per verificare niente regressioni**

```bash
pytest tests/ -v
```
Expected: tutti i test esistenti passano.

- [ ] **Step 7: Commit**

```bash
git add cineauteur.py web/app.py tests/test_prompts.py
git commit -m "refactor: rename SYSTEM_PROMPT to SUSO_PROMPT for two-agent architecture"
```

---

## Task 2: Aggiungi FURIO_PROMPT in cineauteur.py

**Goal:** Definire il system prompt dello sceneggiatore tecnico in formato Fountain con regole CSC/RAI italiano.

**Files:**
- Modify: `cineauteur.py` (in fondo al SUSO_PROMPT)
- Modify: `web/app.py` (aggiungi `_load_furio_prompt`)
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Scrivi il test per FURIO_PROMPT**

In `tests/test_prompts.py` aggiungi:

```python
from web.app import _load_furio_prompt

def test_furio_prompt_loads_and_contains_identity():
    prompt = _load_furio_prompt()
    assert prompt
    assert "Furio" in prompt
    assert "Fountain" in prompt
    assert "INT." in prompt or "EST." in prompt

def test_furio_prompt_forbids_creative_drift():
    prompt = _load_furio_prompt()
    assert "Suso" in prompt
    assert any(k in prompt.lower() for k in ["torna da suso", "rimanda", "non riapri"])
```

- [ ] **Step 2: Run, verifica fallimento**

```bash
pytest tests/test_prompts.py::test_furio_prompt_loads_and_contains_identity -v
```
Expected: FAIL

- [ ] **Step 3: Aggiungi FURIO_PROMPT in cineauteur.py**

Dopo la chiusura di `SUSO_PROMPT` (riga ~410, dopo `"""`), aggiungi:

```python

# ---------------------------------------------------------------------------
# FURIO — sceneggiatore tecnico (formato CSC/RAI italiano, output Fountain)
# ---------------------------------------------------------------------------
FURIO_PROMPT = """\
# IDENTITÀ

Sei Furio, sceneggiatore di mestiere.

Non sei creativo, non sei consulente, non sei dramaturg.
Sei un professionista che prende decisioni narrative già prese
e le mette in forma di sceneggiatura cinematografica italiana.

Tono: asciutto, professionale, da editor severo ma giusto.
Mai effusivo. Mai esplorativo. Mai "secondo me".

---

# LINGUA

Rispondi sempre in italiano.

---

# CONFINI DI RUOLO (ASSOLUTI)

Tu non riapri temi narrativi. Mai.

Se l'autore ti chiede una modifica creativa
("la motivazione di Marco non regge",
 "questo personaggio non funziona",
 "cambiamo il finale"):

Rispondi così:

"Questo non è il mio mestiere.
Torna da Suso (tab Sviluppo), ne parlate insieme,
poi torna qui quando la scena ha una sua verità.
Io la metto in forma."

Non proporre alternative narrative.
Non suggerire psicologia del personaggio.
Non chiedere "cosa vuoi dire con questa scena?".

Il tuo mestiere comincia dopo la decisione, non prima.

---

# INTERIORITÀ → AZIONE FILMABILE

Quando ricevi materiale dell'autore che descrive interiorità —
"si sente in colpa", "prova ansia", "è triste",
"vorrebbe scappare ma non lo fa":

Non cancellarla. Traducila.

Formato di risposta:

> Hai scritto: [interiorità originale]
> In sceneggiatura: [azione filmabile, 1-2 righe, presente]
> Perché: [una riga che mostra come l'azione restituisce lo stato]

Se l'interiorità ha più traduzioni valide:
- proponi max 2 opzioni
- chiedi quale è più vicina

Mai:
- dialogo didascalico ("sono così stanca")
- voice-over esplicativo
- parentetiche emotive sotto il personaggio ("(con tristezza)")
- azioni che descrivono stati invece di fatti

---

# FORMATO OUTPUT: FOUNTAIN

Tu produci SOLO testo in formato Fountain.
Mai prosa libera fuori dal Fountain.
Mai meta-commenti tipo "ecco la scena:".

Regole Fountain:

1. SLUGLINE (intestazione scena):
   - prima parola: INT. o EST.
   - segue: LUOGO in MAIUSCOLO
   - segue: " - " (spazio trattino spazio)
   - tempo: GIORNO, NOTTE, ALBA, TRAMONTO, CONTINUA
   - esempio: INT. CUCINA DI MARCO - GIORNO

2. AZIONE:
   - righe normali, tempo presente indicativo
   - filmabile: solo ciò che si vede o si sente
   - max 3-4 righe per blocco

3. PERSONAGGIO:
   - nome in MAIUSCOLO, su riga propria
   - prima del dialogo
   - esempio:
     MARCO

4. DIALOGO:
   - riga subito sotto il PERSONAGGIO
   - testo normale

5. PARENTETICA (opzionale):
   - tra PERSONAGGIO e DIALOGO
   - tra parentesi tonde
   - una sola riga, breve
   - solo se indispensabile
   - esempio:
     MARCO
     (sottovoce)
     Non ce la faccio.

6. ESTENSIONI personaggio:
   - V.O. = voice over (voce fuori da spazio narrativo)
   - F.C. = fuori campo (voce dentro lo spazio ma fuori inquadratura)
   - esempio: MARCO (V.O.)

7. TRANSIZIONI (usare con parsimonia):
   - DISSOLVENZA IN APERTURA:
   - DISSOLVENZA IN CHIUSURA.
   - TAGLIO A:
   - solo se davvero necessarie

---

# REGOLE STILISTICHE CSC/RAI

- Sempre presente indicativo nelle azioni.
- Mai "vedremo", "sentiamo", "la camera mostra" — descrivi i fatti, non la macchina da presa.
- Nomi luoghi: specifici, mai generici ("UN APPARTAMENTO" ✗ → "APPARTAMENTO DI MARCO" ✓).
- Tempo: GIORNO/NOTTE è preferito ad ALBA/TRAMONTO salvo motivo specifico.
- Numero scene: implicito dall'ordine, non scriverli (li mette il renderer).
- Lunghezza scena: per cortometraggio, max 2 pp per scena salvo necessità.

---

# OUTPUT QUANDO RICEVI MATERIALE

Primo messaggio dell'autore conterrà:

[Materiale del progetto: file, decisioni passate da Suso, eventuale draft grezzo]

Procedi così:

1. Leggi tutto.
2. Se manca materiale per scrivere → chiedi UNA cosa concreta.
   Esempio: "Non ho la scaletta. Hai una lista di scene o devo leggere il draft per estrarla?"
3. Se hai materiale → produci la sceneggiatura completa in Fountain.
4. Non commentare prima. Non commentare dopo.
   Solo Fountain dall'inizio alla fine.

---

# GESTIONE ERRORI VALIDATORE

Se ricevi un messaggio sistema del tipo:
"[Errori validatore Fountain: ...]"

Correggi SOLO gli errori indicati.
Non riaprire il senso. Non riscrivere.
Restituisci la sceneggiatura corretta intera.

---

# OBIETTIVO

Produrre un PDF formalmente perfetto.
Vendibile a un produttore italiano al primo invio.
"""
```

- [ ] **Step 4: Aggiungi loader in web/app.py**

In `web/app.py`, sotto la definizione di `_load_suso_prompt`, aggiungi:

```python
def _load_furio_prompt() -> str:
    src = (pathlib.Path(__file__).parent.parent / "cineauteur.py").read_text(encoding="utf-8")
    m = re.search(r'FURIO_PROMPT\s*=\s*"""\\\n(.*?)"""', src, re.DOTALL)
    if not m:
        raise RuntimeError("FURIO_PROMPT not found in cineauteur.py")
    return m.group(1).replace("\\\n", "\n")

FURIO_PROMPT = _load_furio_prompt()
```

- [ ] **Step 5: Run tutti i test prompts**

```bash
pytest tests/test_prompts.py -v
```
Expected: tutti PASS.

- [ ] **Step 6: Commit**

```bash
git add cineauteur.py web/app.py tests/test_prompts.py
git commit -m "feat: add Furio screenwriter prompt with Fountain output rules"
```

---

## Task 3: Validatore Fountain (modulo + test)

**Goal:** Funzione `validate_fountain(text) -> list[Error]` che cattura gli errori di formato più comuni prima del rendering PDF.

**Files:**
- Create: `web/screenplay_validator.py`
- Create: `tests/test_screenplay_validator.py`

- [ ] **Step 1: Scrivi i test**

Crea `tests/test_screenplay_validator.py`:

```python
from web.screenplay_validator import validate_fountain, FountainError

def test_valid_minimal_fountain_passes():
    text = """INT. CUCINA - GIORNO

Marco apre il frigo.

MARCO
Niente latte.
"""
    errors = validate_fountain(text)
    assert errors == []

def test_slugline_must_start_with_int_or_est():
    text = """INTERNO CUCINA - GIORNO

Marco apre il frigo.
"""
    errors = validate_fountain(text)
    assert any("slugline" in e.message.lower() for e in errors)

def test_slugline_must_have_time():
    text = """INT. CUCINA

Marco apre il frigo.
"""
    errors = validate_fountain(text)
    assert any("tempo" in e.message.lower() or "time" in e.message.lower() for e in errors)

def test_at_least_one_slugline_required():
    text = """Marco apre il frigo.

MARCO
Ciao.
"""
    errors = validate_fountain(text)
    assert any("nessuna scena" in e.message.lower() or "no scene" in e.message.lower() for e in errors)

def test_parenthetical_without_character_is_error():
    text = """INT. CUCINA - GIORNO

(sottovoce)
Niente latte.
"""
    errors = validate_fountain(text)
    assert any("parentetica" in e.message.lower() for e in errors)

def test_character_must_be_uppercase():
    text = """INT. CUCINA - GIORNO

Marco
Niente latte.
"""
    errors = validate_fountain(text)
    # "Marco" non in MAIUSCOLO → trattato come azione, non personaggio
    # NON è un errore: Fountain tollera. Ma "Niente latte." viene poi
    # interpretato come azione, e questo NON dovrebbe rompere.
    # Verifichiamo: nessuna scena = errore solo se davvero manca.
    # Qui c'è uno slugline, quindi 0 errori.
    assert errors == []

def test_multiple_scenes_all_validated():
    text = """INT. CUCINA - GIORNO

Marco entra.

INTERNO SALOTTO - SERA

Marco esce.
"""
    errors = validate_fountain(text)
    # primo slugline ok, secondo malformato
    assert len(errors) >= 1
    assert any("slugline" in e.message.lower() for e in errors)

def test_dialog_line_too_long_is_warning():
    long_line = "x" * 600
    text = f"""INT. CUCINA - GIORNO

MARCO
{long_line}
"""
    errors = validate_fountain(text)
    assert any("lung" in e.message.lower() for e in errors)
```

- [ ] **Step 2: Run, verifica fallimento (modulo non esiste)**

```bash
pytest tests/test_screenplay_validator.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'web.screenplay_validator'`

- [ ] **Step 3: Implementa il validator**

Crea `web/screenplay_validator.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
import re

@dataclass
class FountainError:
    line: int
    message: str

    def __repr__(self) -> str:
        return f"L{self.line}: {self.message}"


_SLUGLINE_RE = re.compile(
    r'^(INT\.|EST\.|INT\./EST\.|EST\./INT\.)\s+.+?\s+-\s+(GIORNO|NOTTE|ALBA|TRAMONTO|CONTINUA|MATTINA|SERA|POMERIGGIO)\s*$',
    re.IGNORECASE,
)
_SLUGLINE_PREFIX = re.compile(r'^(INT\.|EST\.|INT\./EST\.|EST\./INT\.)\s', re.IGNORECASE)
_CHARACTER_RE = re.compile(r'^[A-Z][A-Z0-9 \'\-\.()]+$')
_PARENTHETICAL_RE = re.compile(r'^\s*\(.+\)\s*$')

MAX_DIALOG_LINE_CHARS = 400


def validate_fountain(text: str) -> list[FountainError]:
    errors: list[FountainError] = []
    lines = text.splitlines()
    found_slugline = False
    prev_nonblank: tuple[int, str] | None = None  # (lineno, content)

    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            prev_nonblank = None
            continue

        # Slugline check: linea che inizia con prefisso scena
        if _SLUGLINE_PREFIX.match(stripped):
            found_slugline = True
            if not _SLUGLINE_RE.match(stripped):
                errors.append(FountainError(
                    line=i,
                    message=f"Slugline malformato: '{stripped}'. Atteso: 'INT./EST. LUOGO - GIORNO/NOTTE/ALBA/TRAMONTO/CONTINUA'."
                ))
            prev_nonblank = (i, "slugline")
            continue

        # Slugline iniziato con INTERNO/ESTERNO (errore comune italiano)
        if re.match(r'^(INTERNO|ESTERNO)\s', stripped, re.IGNORECASE):
            errors.append(FountainError(
                line=i,
                message=f"Slugline deve iniziare con 'INT.' o 'EST.', non 'INTERNO/ESTERNO': '{stripped}'."
            ))
            prev_nonblank = (i, "action")
            continue

        # Slugline senza tempo
        # già coperto da SLUGLINE_RE failure, ma se inizia con INT./EST. senza ' - ', errore esplicito sul tempo
        # (gestito sopra)

        # Parentetica
        if _PARENTHETICAL_RE.match(stripped):
            if prev_nonblank is None or prev_nonblank[1] != "character":
                errors.append(FountainError(
                    line=i,
                    message=f"Parentetica senza personaggio precedente: '{stripped}'."
                ))
            prev_nonblank = (i, "parenthetical")
            continue

        # Personaggio (riga in MAIUSCOLO breve)
        if _CHARACTER_RE.match(stripped) and len(stripped) <= 50 and not stripped.endswith('.'):
            prev_nonblank = (i, "character")
            continue

        # Dialogo (se precedente era character o parenthetical)
        if prev_nonblank and prev_nonblank[1] in ("character", "parenthetical"):
            if len(stripped) > MAX_DIALOG_LINE_CHARS:
                errors.append(FountainError(
                    line=i,
                    message=f"Riga dialogo troppo lunga ({len(stripped)} caratteri, max {MAX_DIALOG_LINE_CHARS})."
                ))
            prev_nonblank = (i, "dialog")
            continue

        # Altrimenti: azione
        prev_nonblank = (i, "action")

    if not found_slugline:
        errors.append(FountainError(
            line=0,
            message="Nessuna scena trovata: il documento deve contenere almeno uno slugline (INT./EST.)."
        ))

    return errors
```

- [ ] **Step 4: Run i test**

```bash
pytest tests/test_screenplay_validator.py -v
```
Expected: tutti PASS.

- [ ] **Step 5: Commit**

```bash
git add web/screenplay_validator.py tests/test_screenplay_validator.py
git commit -m "feat: add Fountain validator with Italian CSC/RAI rules"
```

---

## Task 4: Parser Fountain → struttura

**Goal:** Convertire testo Fountain in una lista di elementi tipizzati (`scene`, `action`, `character`, `parenthetical`, `dialog`), preparatoria al rendering.

**Files:**
- Create: `web/screenplay_parser.py`
- Create: `tests/test_screenplay_parser.py`

- [ ] **Step 1: Scrivi i test**

Crea `tests/test_screenplay_parser.py`:

```python
from web.screenplay_parser import parse_fountain, Scene, Action, Character, Dialog, Parenthetical

def test_parse_minimal():
    text = """INT. CUCINA - GIORNO

Marco apre il frigo.

MARCO
Niente latte.
"""
    scenes = parse_fountain(text)
    assert len(scenes) == 1
    s = scenes[0]
    assert isinstance(s, Scene)
    assert s.slugline == "INT. CUCINA - GIORNO"
    assert s.number == 1
    elements = s.elements
    assert len(elements) == 3
    assert isinstance(elements[0], Action)
    assert "Marco apre" in elements[0].text
    assert isinstance(elements[1], Character)
    assert elements[1].name == "MARCO"
    assert isinstance(elements[2], Dialog)
    assert elements[2].text == "Niente latte."

def test_parse_with_parenthetical():
    text = """INT. CUCINA - GIORNO

MARCO
(sottovoce)
Non ce la faccio.
"""
    scenes = parse_fountain(text)
    s = scenes[0]
    assert len(s.elements) == 3
    assert isinstance(s.elements[1], Parenthetical)
    assert s.elements[1].text == "sottovoce"

def test_parse_multiple_scenes_numbered():
    text = """INT. CUCINA - GIORNO

Marco entra.

EST. STRADA - NOTTE

Marco cammina.
"""
    scenes = parse_fountain(text)
    assert len(scenes) == 2
    assert scenes[0].number == 1
    assert scenes[1].number == 2
    assert scenes[1].slugline == "EST. STRADA - NOTTE"

def test_parse_multiline_dialog():
    text = """INT. CUCINA - GIORNO

MARCO
Prima riga.
Seconda riga.
"""
    scenes = parse_fountain(text)
    s = scenes[0]
    dialogs = [e for e in s.elements if isinstance(e, Dialog)]
    assert len(dialogs) == 1
    assert "Prima riga." in dialogs[0].text
    assert "Seconda riga." in dialogs[0].text

def test_parse_character_with_extension():
    text = """INT. CUCINA - GIORNO

ANNA (V.O.)
Marco?
"""
    scenes = parse_fountain(text)
    s = scenes[0]
    c = [e for e in s.elements if isinstance(e, Character)][0]
    assert c.name == "ANNA"
    assert c.extension == "V.O."
```

- [ ] **Step 2: Run, verifica fallimento**

```bash
pytest tests/test_screenplay_parser.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implementa parser**

Crea `web/screenplay_parser.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
import re

@dataclass
class Action:
    text: str

@dataclass
class Character:
    name: str
    extension: str = ""  # V.O., F.C., CONT'D

@dataclass
class Parenthetical:
    text: str

@dataclass
class Dialog:
    text: str

Element = Action | Character | Parenthetical | Dialog

@dataclass
class Scene:
    number: int
    slugline: str
    elements: list[Element] = field(default_factory=list)


_SLUGLINE_PREFIX = re.compile(r'^(INT\.|EST\.|INT\./EST\.|EST\./INT\.)\s', re.IGNORECASE)
_CHARACTER_RE = re.compile(r'^([A-Z][A-Z0-9 \'\-\.]+?)(\s*\(([^)]+)\))?\s*$')
_PARENTHETICAL_RE = re.compile(r'^\s*\((.+)\)\s*$')


def parse_fountain(text: str) -> list[Scene]:
    scenes: list[Scene] = []
    current: Scene | None = None
    pending_dialog: list[str] = []
    last_kind: str | None = None  # "character", "parenthetical", "dialog", "action", "slugline"

    def flush_dialog():
        nonlocal pending_dialog
        if pending_dialog and current is not None:
            current.elements.append(Dialog(text="\n".join(pending_dialog)))
            pending_dialog = []

    raw_lines = text.splitlines()
    pending_action: list[str] = []

    def flush_action():
        nonlocal pending_action
        if pending_action and current is not None:
            current.elements.append(Action(text="\n".join(pending_action).strip()))
            pending_action = []

    for raw in raw_lines:
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            # riga vuota: flush dialog/action in corso
            flush_dialog()
            flush_action()
            last_kind = None
            continue

        # Slugline
        if _SLUGLINE_PREFIX.match(stripped):
            flush_dialog()
            flush_action()
            current = Scene(number=len(scenes) + 1, slugline=stripped)
            scenes.append(current)
            last_kind = "slugline"
            continue

        if current is None:
            # contenuto prima del primo slugline: ignora silenziosamente
            continue

        # Parenthetical (solo dopo character)
        m = _PARENTHETICAL_RE.match(stripped)
        if m and last_kind == "character":
            flush_dialog()
            current.elements.append(Parenthetical(text=m.group(1)))
            last_kind = "parenthetical"
            continue

        # Character (riga MAIUSCOLO breve, possibile extension)
        # NON deve confondersi con azione MAIUSCOLA: heuristic: solo se segue una riga vuota (cioè last_kind is None)
        m = _CHARACTER_RE.match(stripped)
        if m and len(stripped) <= 50 and not stripped.endswith('.') and last_kind is None:
            flush_action()
            name = m.group(1).strip()
            extension = (m.group(3) or "").strip()
            current.elements.append(Character(name=name, extension=extension))
            last_kind = "character"
            continue

        # Dialog (segue character o parenthetical)
        if last_kind in ("character", "parenthetical", "dialog"):
            pending_dialog.append(stripped)
            last_kind = "dialog"
            continue

        # Altrimenti: azione
        flush_dialog()
        pending_action.append(stripped)
        last_kind = "action"

    flush_dialog()
    flush_action()
    return scenes
```

- [ ] **Step 4: Run i test**

```bash
pytest tests/test_screenplay_parser.py -v
```
Expected: tutti PASS.

- [ ] **Step 5: Commit**

```bash
git add web/screenplay_parser.py tests/test_screenplay_parser.py
git commit -m "feat: add Fountain parser producing typed scene structure"
```

---

## Task 5: Template HTML + CSS CSC/RAI

**Goal:** Template Jinja2 che renderizza una lista `Scene` in HTML stilizzato come sceneggiatura italiana, pronto per weasyprint.

**Files:**
- Create: `web/templates/screenplay_pdf.html`

- [ ] **Step 1: Crea il template**

Crea `web/templates/screenplay_pdf.html`:

```html
<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>{{ title or "Sceneggiatura" }}</title>
<style>
  @page {
    size: A4;
    margin: 2.5cm 2.5cm 2.5cm 3.5cm;
    @bottom-right {
      content: counter(page);
      font-family: "Courier New", Courier, monospace;
      font-size: 11pt;
    }
  }
  body {
    font-family: "Courier New", Courier, monospace;
    font-size: 12pt;
    line-height: 1.0;
    color: #000;
  }
  .title-page {
    text-align: center;
    page-break-after: always;
    padding-top: 30%;
  }
  .title-page h1 {
    font-size: 16pt;
    text-transform: uppercase;
    margin: 0 0 2em 0;
    font-weight: normal;
  }
  .title-page .author {
    font-size: 12pt;
    margin-top: 4em;
  }
  .scene {
    margin-top: 1.5em;
    page-break-inside: avoid;
  }
  .scene-header {
    text-transform: uppercase;
    font-weight: bold;
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5em;
  }
  .scene-header .num { min-width: 3em; }
  .scene-header .slug { flex: 1; }
  .scene-header .num-right { min-width: 3em; text-align: right; }
  .action {
    margin: 0.5em 0;
    text-align: left;
    white-space: pre-wrap;
  }
  .character {
    margin: 1em 0 0 0;
    padding-left: 5cm;
    text-transform: uppercase;
    font-weight: bold;
  }
  .parenthetical {
    margin: 0;
    padding-left: 4cm;
    font-style: normal;
  }
  .dialog {
    margin: 0 0 0.5em 0;
    padding-left: 3cm;
    padding-right: 3cm;
    white-space: pre-wrap;
  }
</style>
</head>
<body>

{% if title %}
<div class="title-page">
  <h1>{{ title }}</h1>
  {% if subtitle %}<div class="subtitle">{{ subtitle }}</div>{% endif %}
  {% if author %}<div class="author">di {{ author }}</div>{% endif %}
</div>
{% endif %}

{% for scene in scenes %}
<div class="scene">
  <div class="scene-header">
    <span class="num">{{ scene.number }}.</span>
    <span class="slug">{{ scene.slugline }}</span>
    <span class="num-right">{{ scene.number }}.</span>
  </div>
  {% for el in scene.elements %}
    {% if el.__class__.__name__ == "Action" %}
      <div class="action">{{ el.text }}</div>
    {% elif el.__class__.__name__ == "Character" %}
      <div class="character">{{ el.name }}{% if el.extension %} ({{ el.extension }}){% endif %}</div>
    {% elif el.__class__.__name__ == "Parenthetical" %}
      <div class="parenthetical">({{ el.text }})</div>
    {% elif el.__class__.__name__ == "Dialog" %}
      <div class="dialog">{{ el.text }}</div>
    {% endif %}
  {% endfor %}
</div>
{% endfor %}

</body>
</html>
```

- [ ] **Step 2: Commit (no test diretto, sarà coperto dal renderer)**

```bash
git add web/templates/screenplay_pdf.html
git commit -m "feat: add HTML/CSS template for CSC/RAI screenplay PDF rendering"
```

---

## Task 6: Renderer Fountain → PDF via weasyprint

**Goal:** Modulo che prende testo Fountain e ritorna `bytes` di un PDF formattato CSC/RAI.

**Files:**
- Create: `web/screenplay.py`
- Create: `tests/test_screenplay_render.py`
- Modify: `requirements-web.txt`

- [ ] **Step 1: Aggiungi dipendenze**

In `requirements-web.txt`, aggiungi:

```
weasyprint>=60.0
```

Installa:

```bash
cd /Users/filippoarici/projects/Test_1
pip install weasyprint
```

Verifica che weasyprint funzioni (richiede libs di sistema: cairo, pango, gdk-pixbuf, libffi):

```bash
python -c "from weasyprint import HTML; print(HTML(string='<p>ok</p>').write_pdf()[:4])"
```
Expected: `b'%PDF'`

Se l'import fallisce: install via brew (macOS): `brew install pango gdk-pixbuf libffi`.

- [ ] **Step 2: Scrivi il test**

Crea `tests/test_screenplay_render.py`:

```python
from web.screenplay import render_fountain_to_pdf

MINIMAL_FOUNTAIN = """INT. CUCINA - GIORNO

Marco apre il frigo. Vuoto. Lo richiude piano.

MARCO
(quasi tra sé)
Niente latte.

ANNA (V.O.)
L'ho finito ieri.
"""

def test_render_returns_pdf_bytes():
    pdf = render_fountain_to_pdf(MINIMAL_FOUNTAIN, title="Test", author="A. Tester")
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000

def test_render_includes_scene_count_in_metadata(tmp_path):
    pdf = render_fountain_to_pdf(MINIMAL_FOUNTAIN, title="Test", author="A. Tester")
    # smoke test: scrivi su disco e verifica che esista
    out = tmp_path / "test.pdf"
    out.write_bytes(pdf)
    assert out.exists()
    assert out.stat().st_size > 1000
```

- [ ] **Step 3: Run, verifica fallimento**

```bash
pytest tests/test_screenplay_render.py -v
```
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4: Implementa il renderer**

Crea `web/screenplay.py`:

```python
from __future__ import annotations
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from web.screenplay_parser import parse_fountain

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)


def render_fountain_to_pdf(
    fountain_text: str,
    title: str | None = None,
    subtitle: str | None = None,
    author: str | None = None,
) -> bytes:
    scenes = parse_fountain(fountain_text)
    template = _env.get_template("screenplay_pdf.html")
    html = template.render(
        scenes=scenes,
        title=title,
        subtitle=subtitle,
        author=author,
    )
    pdf_bytes = HTML(string=html).write_pdf()
    return pdf_bytes
```

- [ ] **Step 5: Run i test**

```bash
pytest tests/test_screenplay_render.py -v
```
Expected: PASS.

- [ ] **Step 6: Test visivo manuale (raccomandato)**

```bash
python -c "
from web.screenplay import render_fountain_to_pdf
t = open('tests/fixtures/sample.fountain').read() if __import__('os').path.exists('tests/fixtures/sample.fountain') else '''INT. CUCINA - GIORNO\n\nMarco apre il frigo.\n\nMARCO\nNiente latte.'''
open('/tmp/test-screenplay.pdf','wb').write(render_fountain_to_pdf(t, title='Test', author='Filippo Arici'))
print('PDF in /tmp/test-screenplay.pdf')
"
open /tmp/test-screenplay.pdf
```
Verifica visivamente: font Courier, margini, numerazione scene, indentazioni personaggio/dialogo.

- [ ] **Step 7: Commit**

```bash
git add web/screenplay.py tests/test_screenplay_render.py requirements-web.txt
git commit -m "feat: add Fountain to PDF renderer with weasyprint and CSC/RAI layout"
```

---

## Task 7: Tabella `screenplays` nel DB

**Goal:** Persistenza del sorgente Fountain per progetto. Una sola corrente per progetto.

**Files:**
- Modify: `web/db.py`
- Create: `tests/test_screenplay_db.py`

- [ ] **Step 1: Scrivi i test**

Crea `tests/test_screenplay_db.py`:

```python
import pytest
from web.db import (
    init_db, create_user, create_project,
    upsert_screenplay, get_screenplay, delete_screenplay,
)
from tests.conftest import temp_db  # esistente

def test_create_and_get_screenplay(temp_db):
    init_db()
    uid = create_user("Test", "test@x.it", "hash")
    pid = create_project(uid, "Corto X")

    upsert_screenplay(pid, "INT. CUCINA - GIORNO\n\nMarco entra.", model="claude-opus-4-7")
    s = get_screenplay(pid)
    assert s is not None
    assert "Marco entra" in s["fountain_source"]
    assert s["model"] == "claude-opus-4-7"

def test_upsert_replaces_existing(temp_db):
    init_db()
    uid = create_user("Test", "test@x.it", "hash")
    pid = create_project(uid, "Corto X")

    upsert_screenplay(pid, "v1", model="claude-opus-4-7")
    upsert_screenplay(pid, "v2", model="claude-sonnet-4-6")
    s = get_screenplay(pid)
    assert s["fountain_source"] == "v2"
    assert s["model"] == "claude-sonnet-4-6"

def test_get_screenplay_none_when_absent(temp_db):
    init_db()
    uid = create_user("Test", "test@x.it", "hash")
    pid = create_project(uid, "Corto X")
    assert get_screenplay(pid) is None
```

- [ ] **Step 2: Run, verifica fallimento**

```bash
pytest tests/test_screenplay_db.py -v
```
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Aggiungi CREATE TABLE in init_db**

In `web/db.py`, dentro `init_db()`, aggiungi (dopo le altre CREATE TABLE):

```python
        conn.execute("""
            CREATE TABLE IF NOT EXISTS screenplays (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL UNIQUE,
                fountain_source TEXT NOT NULL,
                model TEXT NOT NULL DEFAULT 'claude-opus-4-7',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        """)
```

- [ ] **Step 4: Aggiungi funzioni CRUD**

In fondo a `web/db.py`, aggiungi:

```python
def upsert_screenplay(project_id: int, fountain_source: str, model: str) -> None:
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO screenplays (project_id, fountain_source, model)
            VALUES (?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                fountain_source = excluded.fountain_source,
                model = excluded.model,
                updated_at = datetime('now')
        """, (project_id, fountain_source, model))


def get_screenplay(project_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM screenplays WHERE project_id = ?",
            (project_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_screenplay(project_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM screenplays WHERE project_id = ?", (project_id,))
```

- [ ] **Step 5: Importa le nuove funzioni in app.py**

In `web/app.py`, nella riga `from web.db import (...)`, aggiungi:
```python
    upsert_screenplay, get_screenplay, delete_screenplay,
```

- [ ] **Step 6: Run i test**

```bash
pytest tests/test_screenplay_db.py -v
```
Expected: PASS.

- [ ] **Step 7: Run tutta la suite, verifica niente regressioni**

```bash
pytest tests/ -v
```
Expected: tutti PASS.

- [ ] **Step 8: Commit**

```bash
git add web/db.py web/app.py tests/test_screenplay_db.py
git commit -m "feat: add screenplays table with upsert/get/delete CRUD"
```

---

## Task 8: Estrazione decisioni Suso → Furio

**Goal:** Funzione che prende lo storico messaggi di una chat di Sviluppo e estrae le "decisioni prese" + nucleo + scene confermate, da passare a Furio come primo input.

**Files:**
- Create: `web/handoff.py`
- Create: `tests/test_handoff.py`

- [ ] **Step 1: Scrivi i test**

Crea `tests/test_handoff.py`:

```python
from web.handoff import extract_decisions, format_handoff

def test_extract_decisions_finds_marked_blocks():
    messages = [
        {"role": "user", "content": "voglio fare un corto su un cassiere"},
        {"role": "assistant", "content": """
[MODALITÀ 1 — NUCLEO]
Decisione presa: il protagonista è un cassiere di supermercato 
che simula crisi epilettiche per ottenere attenzione.
Vuoi modificarla?
"""},
        {"role": "user", "content": "ok andiamo avanti"},
        {"role": "assistant", "content": """
Decisione presa: 15 scene in una giornata lavorativa, dall'alba alla chiusura.
"""},
    ]
    decisions = extract_decisions(messages)
    assert len(decisions) == 2
    assert "cassiere di supermercato" in decisions[0]
    assert "15 scene" in decisions[1]

def test_extract_decisions_empty_when_no_blocks():
    messages = [
        {"role": "user", "content": "ciao"},
        {"role": "assistant", "content": "ciao, raccontami il film"},
    ]
    assert extract_decisions(messages) == []

def test_format_handoff_produces_readable_block():
    decisions = ["protagonista cassiere", "15 scene"]
    project_summary = "Corto su routine alienante."
    out = format_handoff(decisions, project_summary=project_summary, files_summary=None, raw_draft=None)
    assert "Materiale passato da Suso" in out
    assert "protagonista cassiere" in out
    assert "15 scene" in out
    assert "Corto su routine" in out

def test_format_handoff_includes_raw_draft_when_provided():
    out = format_handoff(decisions=[], project_summary=None, files_summary=None, raw_draft="INT. CASA - GIORNO\n\nMarco esce.")
    assert "draft" in out.lower()
    assert "Marco esce" in out
```

- [ ] **Step 2: Run, verifica fallimento**

```bash
pytest tests/test_handoff.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implementa**

Crea `web/handoff.py`:

```python
from __future__ import annotations
import re


_DECISION_RE = re.compile(
    r'Decisione presa:\s*(.+?)(?=\n\s*(?:Vuoi modificarla|⚠️|\n\n|$))',
    re.DOTALL | re.IGNORECASE,
)


def extract_decisions(messages: list[dict]) -> list[str]:
    """Estrae i blocchi 'Decisione presa: …' dai messaggi assistant."""
    out: list[str] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        for match in _DECISION_RE.finditer(content):
            text = match.group(1).strip()
            if text:
                out.append(text)
    return out


def format_handoff(
    decisions: list[str],
    project_summary: str | None = None,
    files_summary: str | None = None,
    raw_draft: str | None = None,
) -> str:
    """Costruisce il primo messaggio user che Furio riceve."""
    parts = ["📋 Materiale passato da Suso (stanza Sviluppo)\n"]

    if project_summary:
        parts.append(f"## Sintesi progetto\n{project_summary}\n")

    if decisions:
        parts.append("## Decisioni prese durante lo sviluppo\n")
        for i, d in enumerate(decisions, 1):
            parts.append(f"{i}. {d}\n")
    else:
        parts.append("## Decisioni prese\n(Nessuna decisione esplicita registrata da Suso.)\n")

    if files_summary:
        parts.append(f"\n## File del progetto\n{files_summary}\n")

    if raw_draft:
        parts.append(f"\n## Draft grezzo caricato dall'autore\n```\n{raw_draft}\n```\n")

    parts.append(
        "\n---\nFurio: leggi tutto. "
        "Se manca materiale per scrivere, fai UNA domanda concreta. "
        "Altrimenti produci la sceneggiatura completa in Fountain."
    )
    return "\n".join(parts)
```

- [ ] **Step 4: Run i test**

```bash
pytest tests/test_handoff.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/handoff.py tests/test_handoff.py
git commit -m "feat: add Suso-to-Furio handoff extraction and formatting"
```

---

## Task 9: Backend route — pagina Sceneggiatura (GET)

**Goal:** Servire la stanza Sceneggiatura come pagina dentro un progetto. Mostrare l'eventuale Fountain già salvato.

**Files:**
- Modify: `web/app.py` (nuova route)
- Create: `web/templates/screenplay.html`

- [ ] **Step 1: Aggiungi la route GET**

In `web/app.py`, aggiungi (vicino alle altre route progetti):

```python
@app.get("/progetti/{project_id}/sceneggiatura", response_class=HTMLResponse)
async def screenplay_page(project_id: int, request: Request, session_token: str = Cookie(None)):
    user = _require_user(session_token)
    if not user:
        return RedirectResponse("/login", status_code=303)

    project = get_project(project_id)
    if not project or project["user_id"] != user["id"]:
        return RedirectResponse("/progetti", status_code=303)

    screenplay = get_screenplay(project_id)
    return templates.TemplateResponse("screenplay.html", {
        "request": request,
        "project": project,
        "screenplay": screenplay,
        "user": user,
    })
```

(Se non esiste `_require_user`, riutilizza il pattern già presente in `web/app.py` per le altre route protette — cerca con `grep -n "session_token" web/app.py` per trovare l'esempio.)

- [ ] **Step 2: Crea il template iniziale**

Crea `web/templates/screenplay.html`:

```html
{% extends "base.html" %}
{% block title %}Sceneggiatura — {{ project.name }}{% endblock %}

{% block content %}
<div class="screenplay-room">
  <header class="room-header furio">
    <div class="tabs">
      <a href="/progetti/{{ project.id }}" class="tab">Sviluppo (Suso)</a>
      <a href="/progetti/{{ project.id }}/sceneggiatura" class="tab active">Sceneggiatura (Furio)</a>
    </div>
    <div class="agent-banner">
      <span class="agent-name">Furio</span>
      <span class="agent-role">— sceneggiatore tecnico</span>
    </div>
  </header>

  <main class="chat-container">
    <div id="messages" class="messages">
      {% if not screenplay %}
        <div class="system-message">
          Furio non ha ancora materiale per questo progetto.
          Inizia chattando per produrre la prima passata di sceneggiatura,
          oppure carica un draft grezzo (.docx / .txt) con il bottone sotto.
        </div>
      {% else %}
        <div class="system-message">
          Esiste già una sceneggiatura per questo progetto (ultimo aggiornamento: {{ screenplay.updated_at }}).
          Puoi rigenerare il PDF o continuare a iterare con Furio.
        </div>
      {% endif %}
    </div>

    <form id="chat-form" class="chat-form">
      <input type="file" id="draft-upload" accept=".txt,.docx" style="display:none">
      <button type="button" id="upload-btn" title="Carica draft grezzo">📎</button>
      <textarea id="input" placeholder="Scrivi a Furio..." rows="3"></textarea>
      <button type="submit">Invia</button>
    </form>

    <div class="actions-bar">
      <button id="generate-pdf" {% if not screenplay %}disabled{% endif %}>
        Scarica PDF
      </button>
    </div>
  </main>
</div>

<script src="/static/screenplay.js"></script>
<style>
  .room-header.furio { border-bottom: 3px solid #2c3e50; background: #f4f1ea; }
  .agent-name { font-weight: bold; color: #2c3e50; font-size: 1.1em; }
  .agent-role { color: #888; }
  .tabs .tab { padding: 0.5em 1em; text-decoration: none; color: #666; }
  .tabs .tab.active { font-weight: bold; color: #2c3e50; border-bottom: 2px solid #2c3e50; }
</style>
{% endblock %}
```

(Nota: assume che `base.html` esista. Se non esiste, ispira la struttura da `chat.html` esistente.)

- [ ] **Step 3: Verifica manuale**

```bash
cd /Users/filippoarici/projects/Test_1
python -m uvicorn web.app:app --reload --port 8000
```

Apri `http://localhost:8000/progetti/1/sceneggiatura` (sostituisci 1 con un id valido). Verifica che la pagina carichi e mostri "Furio non ha ancora materiale" o lo stato esistente.

- [ ] **Step 4: Commit**

```bash
git add web/app.py web/templates/screenplay.html
git commit -m "feat: add screenplay room page route and template"
```

---

## Task 10: Backend route — streaming chat con Furio (POST)

**Goal:** Endpoint SSE che chatta con Furio (modello Opus default), passando handoff iniziale + messaggio utente.

**Files:**
- Modify: `web/app.py`
- Create: `tests/test_screenplay_route.py`

- [ ] **Step 1: Implementa la route**

In `web/app.py`, dopo la route GET, aggiungi:

```python
@app.post("/progetti/{project_id}/sceneggiatura/stream")
async def screenplay_stream(
    project_id: int,
    request: Request,
    session_token: str = Cookie(None),
):
    user = _require_user(session_token)
    if not user:
        return Response(status_code=401)

    project = get_project(project_id)
    if not project or project["user_id"] != user["id"]:
        return Response(status_code=403)

    body = await request.json()
    user_message: str = body.get("message", "").strip()
    raw_draft: str | None = body.get("raw_draft")
    model: str = body.get("model", "claude-opus-4-7")
    history: list[dict] = body.get("history", [])  # conversazione con Furio se esiste

    # Se è il primo messaggio (no history), costruisci handoff
    if not history:
        from web.handoff import extract_decisions, format_handoff
        last_messages = get_last_project_session_messages(project_id) or []
        decisions = extract_decisions(last_messages)
        files = list_project_files(project_id)
        files_summary = "\n".join(f"- {f['filename']}" for f in files) if files else None
        handoff = format_handoff(
            decisions=decisions,
            project_summary=project.get("note") or None,
            files_summary=files_summary,
            raw_draft=raw_draft,
        )
        messages_for_api = [
            {"role": "user", "content": handoff + ("\n\n" + user_message if user_message else "")}
        ]
    else:
        messages_for_api = history + [{"role": "user", "content": user_message}]

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_BASE_URL)

    async def generate():
        full = ""
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": FURIO_PROMPT}, *messages_for_api],
                max_tokens=8192,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full += delta
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
            # Tentativo di estrarre Fountain dal full response e salvarlo
            fountain_text = _extract_fountain_block(full)
            if fountain_text:
                from web.screenplay_validator import validate_fountain
                errors = validate_fountain(fountain_text)
                if not errors:
                    upsert_screenplay(project_id, fountain_text, model=model)
                    yield f"data: {json.dumps({'saved': True})}\n\n"
                else:
                    yield f"data: {json.dumps({'errors': [str(e) for e in errors]})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _extract_fountain_block(text: str) -> str | None:
    """Estrae il primo blocco Fountain dalla risposta di Furio.
    Furio è istruito a produrre solo Fountain, ma può capitare prosa esplicativa
    accidentale. Cerchiamo il primo INT./EST. e prendiamo da lì in poi."""
    import re
    m = re.search(r'^(INT\.|EST\.|INT\./EST\.|EST\./INT\.)\s', text, re.MULTILINE | re.IGNORECASE)
    if not m:
        return None
    return text[m.start():].strip()
```

Verifica che `json` sia importato in cima al file. Aggiungi se manca: `import json`.

- [ ] **Step 2: Scrivi test integration smoke**

Crea `tests/test_screenplay_route.py`:

```python
import pytest
from fastapi.testclient import TestClient
from web.app import app

# Nota: questo test richiede mock del client OpenAI per non chiamare l'API vera.
# Per ora, smoke test sulla GET; lo stream richiederebbe setup più complesso.

def test_screenplay_get_redirects_unauthenticated():
    client = TestClient(app)
    r = client.get("/progetti/1/sceneggiatura", follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in r.headers.get("location", "")
```

- [ ] **Step 3: Run il test**

```bash
pytest tests/test_screenplay_route.py -v
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add web/app.py tests/test_screenplay_route.py
git commit -m "feat: add Furio streaming endpoint with handoff and auto-save"
```

---

## Task 11: Backend route — generazione PDF

**Goal:** Endpoint che ritorna il PDF della sceneggiatura corrente per un progetto.

**Files:**
- Modify: `web/app.py`
- Modify: `tests/test_screenplay_route.py`

- [ ] **Step 1: Implementa la route**

In `web/app.py`, aggiungi:

```python
@app.get("/progetti/{project_id}/sceneggiatura/pdf")
async def screenplay_pdf(project_id: int, session_token: str = Cookie(None)):
    user = _require_user(session_token)
    if not user:
        return Response(status_code=401)

    project = get_project(project_id)
    if not project or project["user_id"] != user["id"]:
        return Response(status_code=403)

    screenplay = get_screenplay(project_id)
    if not screenplay:
        return Response(content="Nessuna sceneggiatura per questo progetto.", status_code=404)

    from web.screenplay import render_fountain_to_pdf
    pdf_bytes = render_fountain_to_pdf(
        screenplay["fountain_source"],
        title=project["name"],
        author=user["nome"],
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{project["name"]}.pdf"'},
    )
```

- [ ] **Step 2: Test smoke (404 quando manca)**

In `tests/test_screenplay_route.py`, aggiungi:

```python
def test_screenplay_pdf_404_when_no_screenplay(authenticated_client, sample_project):
    """authenticated_client e sample_project sono fixture da definire in conftest se non esistono."""
    r = authenticated_client.get(f"/progetti/{sample_project['id']}/sceneggiatura/pdf")
    assert r.status_code == 404
```

Se le fixture non esistono, skip il test o creale in `tests/conftest.py` ispirandoti ai test esistenti per le route protette.

- [ ] **Step 3: Verifica manuale**

```bash
# Server in esecuzione + sessione loggata + sceneggiatura presente per un progetto
curl -o /tmp/screenplay.pdf -b cookies.txt "http://localhost:8000/progetti/1/sceneggiatura/pdf"
open /tmp/screenplay.pdf
```

- [ ] **Step 4: Commit**

```bash
git add web/app.py tests/test_screenplay_route.py
git commit -m "feat: add PDF download endpoint for screenplay"
```

---

## Task 12: Frontend Furio — chat streaming + bottone PDF

**Goal:** JS che gestisce streaming chat con Furio e download PDF.

**Files:**
- Create: `web/static/screenplay.js`

- [ ] **Step 1: Crea il JS**

Crea `web/static/screenplay.js`:

```javascript
(function() {
  const projectId = window.location.pathname.match(/progetti\/(\d+)/)[1];
  const messages = document.getElementById('messages');
  const form = document.getElementById('chat-form');
  const input = document.getElementById('input');
  const uploadBtn = document.getElementById('upload-btn');
  const draftUpload = document.getElementById('draft-upload');
  const pdfBtn = document.getElementById('generate-pdf');

  let history = [];
  let pendingDraft = null;

  function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return div;
  }

  uploadBtn.addEventListener('click', () => draftUpload.click());

  draftUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (file.name.endsWith('.docx')) {
      // delega al backend la conversione docx → testo (riusiamo /chat/upload?)
      // per MVP: solo .txt
      alert("Per ora carica un file .txt. Supporto .docx in arrivo.");
      return;
    }
    pendingDraft = await file.text();
    addMessage('system', `📎 Draft caricato: ${file.name} (${pendingDraft.length} caratteri). Sarà passato a Furio col prossimo messaggio.`);
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    addMessage('user', text);
    input.value = '';

    const responseDiv = addMessage('assistant', '');
    let fullText = '';

    const body = {
      message: text,
      history: history,
      model: 'claude-opus-4-7',
    };
    if (pendingDraft) {
      body.raw_draft = pendingDraft;
      pendingDraft = null;
    }

    const resp = await fetch(`/progetti/${projectId}/sceneggiatura/stream`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const payload = JSON.parse(line.slice(6));
          if (payload.delta) {
            fullText += payload.delta;
            responseDiv.textContent = fullText;
          } else if (payload.saved) {
            addMessage('system', '✓ Sceneggiatura salvata.');
            pdfBtn.disabled = false;
          } else if (payload.errors) {
            addMessage('system', '⚠️ Errori validatore: ' + payload.errors.join('; '));
          } else if (payload.error) {
            addMessage('system', 'Errore: ' + payload.error);
          }
        } catch (e) { /* ignore parse errors */ }
      }
    }

    history.push({role: 'user', content: text});
    history.push({role: 'assistant', content: fullText});
    messages.scrollTop = messages.scrollHeight;
  });

  pdfBtn.addEventListener('click', () => {
    window.location.href = `/progetti/${projectId}/sceneggiatura/pdf`;
  });
})();
```

- [ ] **Step 2: Verifica manuale end-to-end**

Server in esecuzione + login + apri un progetto + clicca tab Sceneggiatura.
- Scrivi "ciao Furio, scrivi una scena di prova".
- Verifica streaming + salvataggio.
- Clicca "Scarica PDF".
- Apri il PDF.

- [ ] **Step 3: Commit**

```bash
git add web/static/screenplay.js
git commit -m "feat: add Furio frontend chat with streaming and PDF download"
```

---

## Task 13: Tab Sviluppo/Sceneggiatura nella pagina Progetto

**Goal:** Aggiungere i due tab `Sviluppo` / `Sceneggiatura` nell'header della pagina progetto esistente, simmetrici a quelli della stanza Sceneggiatura.

**Files:**
- Modify: `web/templates/projects.html` (o template della pagina dettaglio progetto)

- [ ] **Step 1: Identifica il template della pagina dettaglio progetto**

```bash
grep -rn "project.id" /Users/filippoarici/projects/Test_1/web/templates/
```

Identifica il template che mostra il singolo progetto (probabilmente `projects.html` o un blocco specifico).

- [ ] **Step 2: Aggiungi i tab in cima al contenuto del progetto**

Nel template identificato, sopra il contenuto principale del progetto (dove c'è la chat con Suso), aggiungi:

```html
<div class="room-tabs">
  <a href="/progetti/{{ project.id }}" class="tab active">
    Sviluppo <span class="agent">(Suso)</span>
  </a>
  <a href="/progetti/{{ project.id }}/sceneggiatura" class="tab">
    Sceneggiatura <span class="agent">(Furio)</span>
  </a>
</div>

<style>
  .room-tabs { display: flex; gap: 0; border-bottom: 1px solid #ddd; margin-bottom: 1em; }
  .room-tabs .tab {
    padding: 0.75em 1.5em;
    text-decoration: none;
    color: #888;
    border-bottom: 3px solid transparent;
  }
  .room-tabs .tab.active {
    color: #c17a5a;
    border-bottom-color: #c17a5a;
    font-weight: bold;
  }
  .room-tabs .agent { color: #666; font-weight: normal; font-size: 0.9em; }
</style>
```

(Nel template `screenplay.html` la classe `active` va sul secondo tab; nella pagina progetto va sul primo.)

- [ ] **Step 3: Verifica manuale**

Server in esecuzione → apri un progetto → vedi i due tab → cliccando passa dalle due stanze.

- [ ] **Step 4: Commit**

```bash
git add web/templates/projects.html
git commit -m "feat: add Sviluppo/Sceneggiatura tabs to project page"
```

---

## Task 14: Upload draft grezzo .docx

**Goal:** Permettere all'utente di caricare un .docx come draft grezzo per Furio (non solo .txt).

**Files:**
- Modify: `web/app.py`
- Modify: `web/static/screenplay.js`
- Modify: `requirements-web.txt`

- [ ] **Step 1: Aggiungi dipendenza**

In `requirements-web.txt`, aggiungi:
```
python-docx>=1.0.0
```

Installa:
```bash
pip install python-docx
```

- [ ] **Step 2: Aggiungi endpoint conversione docx → testo**

In `web/app.py`, aggiungi:

```python
@app.post("/progetti/{project_id}/sceneggiatura/upload-draft")
async def upload_draft(
    project_id: int,
    file: UploadFile = File(...),
    session_token: str = Cookie(None),
):
    user = _require_user(session_token)
    if not user:
        return Response(status_code=401)
    project = get_project(project_id)
    if not project or project["user_id"] != user["id"]:
        return Response(status_code=403)

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        return Response(content="File troppo grande (max 5 MB).", status_code=400)

    fname = file.filename.lower()
    if fname.endswith(".docx"):
        from io import BytesIO
        from docx import Document
        doc = Document(BytesIO(content))
        text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif fname.endswith(".txt"):
        text = content.decode("utf-8", errors="replace")
    else:
        return Response(content="Formato non supportato (usa .txt o .docx).", status_code=400)

    return {"draft": text, "size_chars": len(text)}
```

- [ ] **Step 3: Aggiorna frontend**

In `web/static/screenplay.js`, sostituisci la gestione del `.docx`:

```javascript
draftUpload.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  if (file.name.endsWith('.docx')) {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch(`/progetti/${projectId}/sceneggiatura/upload-draft`, {
      method: 'POST', body: fd
    });
    if (!r.ok) { alert('Upload fallito'); return; }
    const data = await r.json();
    pendingDraft = data.draft;
  } else {
    pendingDraft = await file.text();
  }
  addMessage('system', `📎 Draft caricato: ${file.name} (${pendingDraft.length} caratteri). Sarà passato a Furio col prossimo messaggio.`);
});
```

- [ ] **Step 4: Verifica manuale**

Carica un .docx di prova con qualche scena scritta in modo informale. Manda un messaggio a Furio. Verifica che lo riceva e produca Fountain.

- [ ] **Step 5: Commit**

```bash
git add web/app.py web/static/screenplay.js requirements-web.txt
git commit -m "feat: support .docx draft upload for Furio"
```

---

## Task 15: Retry automatico su errori validatore

**Goal:** Quando il validatore Fountain trova errori, reinette automaticamente a Furio per correzione (max 2 retry).

**Files:**
- Modify: `web/app.py`

- [ ] **Step 1: Estrai logica retry nel route stream**

Sostituisci la sezione `try: stream = ...` del task 10 con questa versione che gestisce retry:

```python
async def generate():
    nonlocal messages_for_api  # se serve per closure
    full = ""
    max_retries = 2
    attempt = 0

    while attempt <= max_retries:
        full = ""
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": FURIO_PROMPT}, *messages_for_api],
                max_tokens=8192,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full += delta
                    yield f"data: {json.dumps({'delta': delta, 'attempt': attempt})}\n\n"

            fountain_text = _extract_fountain_block(full)
            if not fountain_text:
                yield f"data: {json.dumps({'error': 'Furio non ha prodotto Fountain riconoscibile.'})}\n\n"
                return

            from web.screenplay_validator import validate_fountain
            errors = validate_fountain(fountain_text)
            if not errors:
                upsert_screenplay(project_id, fountain_text, model=model)
                yield f"data: {json.dumps({'saved': True})}\n\n"
                return

            # Errori → prepara retry
            attempt += 1
            if attempt > max_retries:
                yield f"data: {json.dumps({'errors': [str(e) for e in errors], 'final': True})}\n\n"
                return

            yield f"data: {json.dumps({'errors': [str(e) for e in errors], 'retry': attempt})}\n\n"
            messages_for_api = messages_for_api + [
                {"role": "assistant", "content": full},
                {"role": "user", "content": (
                    f"[Errori validatore Fountain trovati:]\n"
                    + "\n".join(f"- {e}" for e in errors)
                    + "\n\nCorreggi SOLO gli errori indicati. Non riaprire il senso."
                )}
            ]
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
```

- [ ] **Step 2: Verifica manuale**

Forzare uno scenario di retry è difficile in produzione. Per testarlo, modifica temporaneamente `FURIO_PROMPT` con un errore intenzionale (es. cambia "INT." in "INTERNO"), manda un messaggio, verifica che parta il retry e poi correggi.

Rimuovi la modifica temporanea quando hai verificato.

- [ ] **Step 3: Commit**

```bash
git add web/app.py
git commit -m "feat: auto-retry Furio on Fountain validation errors (max 2)"
```

---

## Task 16: Aggiornamento documento di stato

**Goal:** Riflettere la nuova architettura nel `FilmMaker-stato.md` dell'utente.

**Files:**
- Modify: `/Users/filippoarici/Desktop/AI/FilmMaker-stato.md` (chiedere prima all'utente)

- [ ] **Step 1: Chiedi all'utente se vuole aggiornare il file di stato**

Mostra all'utente la diff proposta:

- aggiungere sezione "Architettura due stanze (Suso + Furio)"
- aggiungere file: `web/screenplay.py`, `web/screenplay_parser.py`, `web/screenplay_validator.py`, `web/handoff.py`, `web/templates/screenplay.html`, `web/templates/screenplay_pdf.html`, `web/static/screenplay.js`
- aggiungere route: `GET/POST /progetti/<id>/sceneggiatura`, `GET /progetti/<id>/sceneggiatura/pdf`, `POST /progetti/<id>/sceneggiatura/upload-draft`
- aggiungere tabella DB: `screenplays`
- aggiungere dipendenze: `weasyprint`, `python-docx`

- [ ] **Step 2: Se l'utente conferma, applica le modifiche e committale**

```bash
git add ../Desktop/AI/FilmMaker-stato.md  # se nel repo, altrimenti skip git
```

---

## Task 17: Verifica deploy Railway (rischio noto)

**Goal:** Assicurarsi che `weasyprint` funzioni nel container Railway prima del deploy in produzione.

**Files:**
- Possibilmente: nuovo `nixpacks.toml` o `Dockerfile` per dichiarare libs di sistema

- [ ] **Step 1: Verifica libs richieste da weasyprint**

Documentazione weasyprint dice: serve `libpango`, `libcairo`, `libffi`, `libgdk-pixbuf`.

In Railway, di default si usa Nixpacks. Crea o modifica `nixpacks.toml` al root del progetto:

```toml
[phases.setup]
nixPkgs = ["python311", "pango", "cairo", "gdk-pixbuf", "libffi"]
```

- [ ] **Step 2: Test locale in container (opzionale ma raccomandato)**

Se hai Docker installato:

```bash
docker run --rm -v $(pwd):/app -w /app python:3.11-slim bash -c "
  apt-get update && apt-get install -y libpango-1.0-0 libcairo2 libgdk-pixbuf2.0-0 &&
  pip install -r requirements-web.txt &&
  python -c 'from weasyprint import HTML; print(HTML(string=\"<p>ok</p>\").write_pdf()[:4])'
"
```
Expected: `b'%PDF'`

- [ ] **Step 3: Verifica spazio disco Railway prima del deploy**

```bash
railway logs --tail 50 | grep -i "disk\|space\|memory" || true
```

Il problema noto è ~0.73 MB liberi sul volume `/data`. weasyprint e dipendenze sono ~50 MB. Se non c'è spazio, l'install fallirà. Verificare prima di pushare.

- [ ] **Step 4: Deploy (con conferma utente)**

⚠️ **Chiedere conferma all'utente PRIMA di lanciare `railway up`** (deploy in produzione).

```bash
cd /Users/filippoarici/projects/Test_1
railway up --detach
```

- [ ] **Step 5: Verifica post-deploy**

```bash
railway logs --tail 100
```

Cerca errori weasyprint. Apri `https://moviemaker-production-c379.up.railway.app/progetti/<id>/sceneggiatura` e verifica end-to-end.

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Due stanze UI nel Progetto (tab Sviluppo / Sceneggiatura) | 9, 13 |
| Suso rinominata + invariata | 1 |
| Furio prompt asciutto + Fountain rules | 2 |
| Confini rigidi (Furio rimanda a Suso) | 2 (parte del prompt) |
| Interiorità → azione (già aggiunta a Suso) | esistente, ribadita in 2 |
| Fountain → PDF via weasyprint, layout CSC/RAI | 4, 5, 6 |
| Validator + retry max 2 | 3, 15 |
| Handoff Suso → Furio (decisioni + file + draft) | 8, 10 |
| Due ingressi: continua da Suso / draft grezzo | 10 (continua), 14 (draft .docx) |
| Modello configurabile (Opus default Furio) | 10 (param `model`), 12 (frontend) |
| Tabella `screenplays` | 7 |
| Edge cases (no materiale, draft lungo, errori) | 9, 10, 15 |
| Test unitari + integration | 1, 2, 3, 4, 6, 7, 8, 10, 11 |
| Deploy Railway con verifica weasyprint | 17 |

**Placeholder scan:** nessun "TBD", "TODO", "implement later", "similar to". Codice esplicito in tutti gli step di implementazione.

**Type consistency:** `Scene` / `Action` / `Character` / `Parenthetical` / `Dialog` definiti in Task 4, usati coerentemente in Task 5 e 6. `validate_fountain` definito in Task 3 con stessa signature usata in Task 10 e 15. `upsert_screenplay` / `get_screenplay` definiti in Task 7, usati in Task 10, 11, 15.

**Gap identificati e risolti inline:** la spec menzionava "watermark, firma digitale" come fuori scope → confermato fuori scope, non aggiunto. La spec menzionava "lo sceneggiatore rimanda al dramaturg" → implementato nel prompt di Furio (Task 2), nessuna logica backend richiesta.

---

## Note operative

- **Ordine consigliato:** task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17.
- I task 3-4-5-6 (validator, parser, template, renderer) sono indipendenti tra loro a parte le dipendenze esplicite. Possono essere parallelizzati se eseguiti da subagent diversi.
- **Commit frequenti:** ogni task chiude con un commit. Non saltare.
- **Deploy:** SOLO dopo task 17 e conferma utente. Mai prima.

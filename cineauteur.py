#!/usr/bin/env python3
"""
CineAuteur — Assistente AI per il cinema d'autore.
Usa OpenRouter per accedere a modelli multipli a costi ridotti.

Uso:
  python cineauteur.py [--model MODEL]

Esempi:
  python cineauteur.py
  python cineauteur.py --model google/gemini-2.5-pro
  python cineauteur.py --model deepseek/deepseek-r1

Comandi in chat:
  /load <percorso>   Carica un file (sceneggiatura, trattamento, PDF)
  /files             Mostra i file caricati nel contesto
  /clear             Rimuove i file dal contesto
  /reset             Azzera la conversazione (file e corpus rimangono)
  /model [nome]      Mostra il modello attivo o cambialo al volo
  /corpus            Statistiche del corpus documentale
  /help              Mostra questo messaggio
  /quit              Esci
"""

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime
from openai import OpenAI

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
from rich.rule import Rule
from rich.live import Live
from rich.theme import Theme

# ---------------------------------------------------------------------------
# UI — tema colori
# ---------------------------------------------------------------------------
_theme = Theme({
    "ai.name":    "#C17A5A bold",   # terracotta
    "ai.text":    "#F0E6D3",        # avorio caldo
    "user.arrow": "#D4A853 bold",   # oro ambra
    "sys":        "#5A5A5A",        # grigio muto
    "sys.bold":   "#7A7A7A bold",
    "err":        "#C14F4F",        # rosso morbido
    "sep":        "#2E2E2E",        # separatore scuro
})
console = Console(theme=_theme, highlight=False)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# RAG: dipendenze opzionali
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Modelli disponibili su OpenRouter (aggiornati a 2026)
# ---------------------------------------------------------------------------
MODELS = {
    # alias rapidi → ID completo OpenRouter
    "sonnet":   "anthropic/claude-sonnet-4-6",
    "opus":     "anthropic/claude-opus-4-6",
    "gemini":   "google/gemini-2.5-pro",
    "flash":    "google/gemini-2.0-flash-001",
    "deepseek": "deepseek/deepseek-r1",
    "qwen":     "qwen/qwq-32b",
    "llama":    "meta-llama/llama-3.3-70b-instruct",
}

# Modello di default — ottimo rapporto qualità/prezzo per analisi filmiche
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Configurazione RAG
# ---------------------------------------------------------------------------
DOCS_DIR     = Path(__file__).parent / "docs"
CHUNK_SIZE   = 350
CHUNK_OVERLAP= 60
TOP_K        = 6
MIN_SCORE    = 0.1

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
# IDENTITÀ

Sei un dramaturg AI per lo sviluppo di progetti cinematografici.

Non sei una chat generica.
Sei un sistema che guida l'autore dalla prima idea alla sceneggiatura completa.

Il tuo obiettivo è portare il progetto a una forma chiusa, coerente e realizzabile.

---

# LINGUA

Rispondi sempre in italiano.

---

# CONTESTO OPERATIVO

L'utente può:
- caricare file (note, PDF, sceneggiature, immagini)
- usare /progetto (sintesi del materiale)
- avere uno storico e un profilo autore

Regole:
1. Se ci sono file → leggili sempre prima
2. Usa /progetto come base principale
3. Non chiedere informazioni già presenti
4. Se il materiale è contraddittorio → segnalalo

---

# CLASSIFICAZIONE DEL MATERIALE (STEP 0)

Prima di qualsiasi analisi o costruzione:

Devi determinare la natura del materiale ricevuto.

Classifica sempre in:

1. Decisioni già prese
2. Ipotesi / esplorazioni
3. Idee scartate

Se NON è chiaro:
- NON procedere alla costruzione
- chiedi esplicitamente all'autore: "questo materiale è definitivo o esplorativo?"

Regole:
- non trattare mai automaticamente il materiale come valido
- non costruire strutture su materiale non validato
- considera le note come grezze per default

Se questo passaggio non viene rispettato:
- qualsiasi costruzione successiva è da considerarsi non valida
- devi fermarti e tornare alla classificazione

Obiettivo: evitare di costruire su basi instabili.

---

# VINCOLI NEGATIVI (PRIORITÀ MASSIMA)

Quando lavori su progetti non narrativi o anti-narrativi:

È VIETATO:
- applicare strutture classiche (inizio, sviluppo, climax, risoluzione)
- costruire escalation emotiva
- ordinare le scene per intensità crescente
- creare relazioni causa-effetto tra scene
- trasformare il materiale in arco psicologico

Devi:
- seguire il principio dichiarato dall'autore
- verificare che ogni scelta sia coerente con quel principio

Se non riesci:
- fermati
- segnala il problema
- chiedi chiarimento

---

# FILTRAGGIO DEL MATERIALE

Quando analizzi file o note devi distinguere:

1. Decisioni già prese
2. Idee in dubbio
3. Idee scartate

Regole:
- usa SOLO le decisioni per costruire
- non includere automaticamente i dubbi
- ignora completamente gli scarti
- se non è chiaro → chiedi quali immagini sono "vive"

---

# NON AGGIUNGERE — ESTRARRE

- NON inventare elementi nuovi
- NON completare in modo creativo
- NON migliorare aggiungendo contenuto

Devi:
- estrarre
- ordinare
- chiarire

Se manca qualcosa:
- segnalarlo, non inventarlo

---

# MODALITÀ DI LAVORO (AUTO-INFERITE)

Dichiara sempre la fase all'inizio (se la risposta è >3 righe).

---

## FASE 1 — NUCLEO

Attiva quando il materiale è confuso.

Output:
- 1 frase del film
- 1 immagine centrale
- 1 meccanismo formale

---

## FASE 2 — STRUTTURA

Attiva quando esiste un'idea ma non una struttura chiusa.

Output:
- lista scene (max 6-8)
- ordine
- funzione concreta

---

## FASE 3 — SCRITTURA

Attiva quando la struttura è definita.

Output:
- una scena completa, concreta e filmabile

---

# CONTROLLO DIVERGENZA / CONVERGENZA

Per ogni decisione:

Se NON hai abbastanza informazioni:
- proponi max 2 opzioni
- entrambe concrete
- fai UNA domanda

Se hai abbastanza informazioni:
- indica la scelta migliore
- spiega perché
- chiedi conferma

Se l'autore non decide:
- scegli tu e procedi

Dopo la scelta:
- non riaprire automaticamente alternative

---

# GESTIONE RIAPERTURE

Quando l'autore vuole modificare una decisione:

1. Segnala la riapertura
2. Spiega cosa cambia

Offri:

A) Testare senza cambiare struttura
B) Sostituire definitivamente

Se A:
- confronto rapido
- scelta finale

Se B:
- aggiorna e continua

---

# CONTROLLO QUALITÀ (PRIMA DELLA CHIUSURA)

Prima di considerare una decisione come definitiva:

Devi verificare che sia necessaria e coerente.

Per ogni decisione importante:

1. chiedi:
   - questa scelta è indispensabile o sostituibile?
   - cosa aggiunge realmente al film?

2. identifica:
   - se è un'immagine forte ma gratuita
   - se è coerente con il principio del film

3. se debole:
   - segnala il problema
   - proponi un miglioramento minimo (non riscrivere tutto)

Solo dopo questa verifica:
- la decisione può essere confermata e bloccata

Obiettivo: evitare chiusure veloci ma superficiali.

---

# BLOCCO DECISIONALE

Dopo ogni decisione importante:

"Decisione presa: [X]
Vuoi modificarla?
⚠️ modificarla riapre la struttura"

---

# PROGRESSIONE

Mantieni sempre visibile:
- Nucleo: ✓ / in corso / ✗
- Struttura: ✓ / in corso / ✗
- Scrittura: ✓ / in corso / ✗

---

# MODALITÀ CRITICA

Attiva:
1. su richiesta
2. se c'è un problema strutturale evidente

Se attivata senza richiesta:
- dichiaralo

Output:
- 3 problemi concreti
- 1 miglioramento

Non usarla su dettagli minori.

---

# DOMANDE

Solo domande concrete e filmabili:

- "cosa fa fisicamente?"
- "dove è il telefono?"
- "la camera si muove?"

Evita:
- psicologia
- teoria
- interpretazioni astratte

---

# OUTPUT (HEADER CONDIZIONALE)

Se la risposta è lunga:

FASE:
OBIETTIVO:
OUTPUT:

Ometti se:
- risposta breve
- semplice conferma

---

# PRIORITÀ

1. far avanzare il progetto
2. produrre output concreti
3. evitare loop

---

# OBIETTIVO FINALE

Portare l'autore da:
"non so cosa fare"

a:
"ho una sceneggiatura completa e girabile"
"""

# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

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


def load_corpus(docs_dir: Path) -> tuple[list[str], list[str], int]:
    if not docs_dir.exists():
        return [], [], 0
    chunks, sources = [], []
    file_count = 0
    for fp in sorted(docs_dir.rglob("*")):
        if fp.suffix.lower() not in (".txt", ".pdf"):
            continue
        content = read_file_content(fp)
        if not content or len(content.strip()) < 100:
            continue
        file_count += 1
        rel = str(fp.relative_to(docs_dir))
        for c in chunk_text(content):
            chunks.append(c)
            sources.append(rel)
    return chunks, sources, file_count


def build_index(chunks: list[str]):
    if not BM25_AVAILABLE or not chunks:
        return None
    return BM25Okapi([c.lower().split() for c in chunks])


def retrieve(query: str, index, chunks, sources) -> str:
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


# ---------------------------------------------------------------------------
# File utente
# ---------------------------------------------------------------------------

def load_user_file(filepath: str) -> tuple[str, str]:
    # Normalizza il percorso: rimuove virgolette, unescape spazi (drag & drop da Finder)
    fp = filepath.strip().strip('"').strip("'")
    fp = fp.replace("\\ ", " ")   # "annabel\ project.pdf" → "annabel project.pdf"
    path = Path(fp).expanduser().resolve()
    if not path.exists():
        if path.suffix.lower() == ".pages":
            raise ValueError("Formato .pages non supportato. Apri Pages → File → Esporta in → PDF, poi ricarica il PDF.")
        raise FileNotFoundError(f"File non trovato: {path}")
    content = read_file_content(path)
    if content is None:
        raise ValueError("Formato non supportato. Usa .pdf o .txt")
    return path.name, content


def looks_like_path(text: str) -> bool:
    """Restituisce True se il testo sembra un percorso file trascinato dal Finder."""
    t = text.strip().strip('"').strip("'").replace("\\ ", " ")
    if not t:
        return False
    p = Path(t.replace("\\ ", " ")).expanduser()
    return (t.startswith("/") or t.startswith("~")) and p.suffix.lower() in {
        ".pdf", ".txt", ".md", ".pages", ".docx"
    }


def build_file_injection(project_files, files_in_context) -> str | None:
    new = {k: v for k, v in project_files.items() if k not in files_in_context}
    if not new:
        return None
    parts = ["[File del progetto]"]
    for name, content in new.items():
        parts.append(f"=== {name} ===\n{content}\n=== fine {name} ===")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def resolve_model(name: str) -> str:
    """Risolve alias brevi nel nome completo OpenRouter."""
    return MODELS.get(name.lower(), name)


def print_banner(model: str, corpus_files: int, corpus_chunks: int):
    console.print()
    console.print("  [ai.name]CineAuteur[/]  [sys]—  cinema d'autore[/]")
    console.print(f"  [sys]{model}[/]")
    console.print(f"  [sys]{corpus_files} file · {corpus_chunks} chunk  ·  /help per i comandi[/]")
    console.print(Rule(style="sep"))
    console.print()


def print_help(current_model: str):
    console.print()
    console.print(f"  [sys.bold]modello attivo[/]  [sys]{current_model}[/]")
    console.print()
    lines = [
        ("/load [i]<percorso>[/i]", "carica sceneggiatura, trattamento o PDF"),
        ("/progetto",               "sintesi strutturata del progetto caricato"),
        ("/files",                  "file del progetto caricati"),
        ("/clear",                  "rimuove i file dal contesto"),
        ("/reset",                  "azzera la conversazione"),
        ("/save [i][nome][/i]",     "esporta conversazione in .md"),
        ("/model [i][nome][/i]",    "cambia modello  (sonnet · opus · gemini · flash · deepseek · qwen · llama)"),
        ("/corpus",                 "statistiche corpus"),
        ("/quit",                   "esci"),
    ]
    for cmd, desc in lines:
        console.print(f"  [ai.name]{cmd:<28}[/] [sys]{desc}[/]")
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args, _ = parser.parse_known_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Errore: OPENROUTER_API_KEY non impostata.", file=sys.stderr)
        print("Ottieni una chiave su https://openrouter.ai/keys", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )

    current_model = resolve_model(args.model)

    # Carica corpus
    print("  Caricamento corpus...", end="", flush=True)
    corpus_chunks, corpus_sources, corpus_file_count = load_corpus(DOCS_DIR)
    bm25 = build_index(corpus_chunks)
    rag_active = bm25 is not None
    console.print(f" [sys]{len(corpus_chunks)} chunk da {corpus_file_count} file.[/]")

    # Stato conversazione
    conversation: list[dict] = []
    project_files: dict[str, str] = {}
    files_in_context: set[str] = set()

    print_banner(current_model, corpus_file_count, len(corpus_chunks))

    def stream_response(messages: list[dict]) -> str:
        """Esegue la chiamata API con streaming e rendering markdown live."""
        full_response = ""
        console.print()
        console.print("[ai.name]CineAuteur[/]")
        try:
            with client.chat.completions.create(
                model=current_model,
                messages=messages,
                max_tokens=8192,
                stream=True,
                extra_headers={"HTTP-Referer": "https://cineauteur.local", "X-Title": "CineAuteur"},
            ) as stream:
                with Live(
                    Markdown(""),
                    console=console,
                    refresh_per_second=12,
                    vertical_overflow="visible",
                ) as live:
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            full_response += delta
                            live.update(Markdown(full_response))
            console.print()
            console.print(Rule(style="sep"))
            console.print()
        except Exception as e:
            console.print(f"\n[err]  errore: {e}[/]\n")
            full_response = ""
        return full_response

    while True:
        try:
            arrow = Text("› ", style="user.arrow")
            console.print(arrow, end="")
            user_input = input("").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[sys]arrivederci.[/]")
            break

        if not user_input:
            continue

        # --- Drag & drop: percorso file trascinato dal Finder ---
        if looks_like_path(user_input):
            try:
                name, content = load_user_file(user_input)
                project_files[name] = content
                kb    = len(content.encode()) / 1024
                words = len(content.split())
                lines = content.count("\n") + 1
                console.print(f"\n  [ai.name]{name}[/]  [sys]{kb:.1f} KB · {words:,} parole · {lines:,} righe[/]")
                if words > 15000:
                    console.print(f"  [sys]documento lungo — /model gemini consigliato (1M token)[/]")
                console.print(f"  [sys]usa /progetto per la sintesi iniziale.[/]\n")
            except Exception as e:
                console.print(f"\n[err]  {e}[/]\n")
            continue

        # --- Comandi ---
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd == "/quit":
                console.print("[sys]arrivederci.[/]")
                break
            elif cmd == "/help":
                print_help(current_model)
            elif cmd == "/model":
                if arg:
                    current_model = resolve_model(arg)
                    console.print(f"\n[sys]  modello  [/][ai.name]{current_model}[/]\n")
                else:
                    console.print(f"\n[sys]  modello attivo  [/][ai.name]{current_model}[/]\n")
            elif cmd == "/corpus":
                if rag_active:
                    console.print()
                    console.print(f"  [sys]{corpus_file_count} file · {len(corpus_chunks)} chunk[/]")
                    for folder, count in sorted(Counter(
                        s.split("/")[0] for s in corpus_sources
                    ).items()):
                        console.print(f"  [sys]  {folder}/  ·  {count} chunk[/]")
                    console.print()
                else:
                    console.print("\n[sys]  corpus non disponibile.[/]\n")
            elif cmd == "/files":
                if project_files:
                    console.print(f"\n  [sys]file caricati ({len(project_files)})[/]")
                    for name in project_files:
                        mark = "[ai.name]✓[/]" if name in files_in_context else "[sys]·[/]"
                        console.print(f"  {mark} [sys]{name}  ({len(project_files[name]):,} car.)[/]")
                    console.print()
                else:
                    console.print("\n[sys]  nessun file caricato.[/]\n")
            elif cmd == "/clear":
                project_files.clear()
                files_in_context.clear()
                console.print("\n[sys]  file rimossi.[/]\n")
            elif cmd == "/reset":
                conversation.clear()
                files_in_context.clear()
                console.print("\n[sys]  conversazione azzerata.[/]\n")
            elif cmd == "/progetto":
                if not project_files:
                    console.print("\n[sys]  nessun file caricato. usa /load prima.[/]\n")
                else:
                    file_inj = build_file_injection(project_files, set())
                    files_in_context.update(project_files.keys())
                    prompt = (
                        f"{file_inj}\n\n"
                        "Hai appena ricevuto il materiale del progetto. "
                        "Leggi tutto. Poi: "
                        "1. Determina in quale modalità siamo (NUCLEO / STRUTTURA / SCRITTURA) "
                        "in base allo stato reale del materiale. Dichiaralo esplicitamente. "
                        "2. Produci l'output obbligatorio di quella modalità. "
                        "Non fare domande su cose già presenti nel materiale."
                    )
                    conversation.append({"role": "user", "content": prompt})
                    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *conversation]
                    resp = stream_response(msgs)
                    if resp:
                        conversation.append({"role": "assistant", "content": resp})
                    else:
                        conversation.pop()
                continue
            elif cmd == "/save":
                if not conversation:
                    console.print("\n[sys]  nessuna conversazione da salvare.[/]\n")
                else:
                    ts = datetime.now().strftime("%Y%m%d_%H%M")
                    fname = arg if arg else f"sessione_{ts}"
                    if not fname.endswith(".md"):
                        fname += ".md"
                    save_path = Path(__file__).parent / fname
                    lines = [f"# CineAuteur — {ts}\n\n"]
                    for msg in conversation:
                        if msg["role"] == "user":
                            content = msg["content"]
                            if "---" in content:
                                content = content.split("---")[-1].strip()
                            lines.append(f"**Tu:** {content}\n\n")
                        elif msg["role"] == "assistant":
                            lines.append(f"**CineAuteur:** {msg['content']}\n\n---\n\n")
                    save_path.write_text("".join(lines), encoding="utf-8")
                    console.print(f"\n[sys]  conversazione salvata in [/][ai.name]{save_path.name}[/]\n")
            elif cmd == "/load":
                if not arg:
                    console.print("[sys]  uso: /load <percorso>[/]\n")
                else:
                    try:
                        name, content = load_user_file(arg)
                        project_files[name] = content
                        kb    = len(content.encode()) / 1024
                        words = len(content.split())
                        lines = content.count("\n") + 1
                        console.print(f"\n  [ai.name]{name}[/]  [sys]{kb:.1f} KB · {words:,} parole · {lines:,} righe[/]")
                        if words > 15000:
                            console.print(f"  [sys]documento lungo — /model gemini consigliato (1M token)[/]")
                        console.print(f"  [sys]usa /progetto per la sintesi iniziale.[/]\n")
                    except Exception as e:
                        console.print(f"\n[err]  errore: {e}[/]\n")
            else:
                console.print(f"\n[sys]  comando non riconosciuto. /help per la lista.[/]\n")
            continue

        # --- Costruzione messaggio utente ---
        rag_ctx  = retrieve(user_input, bm25, corpus_chunks, corpus_sources) if rag_active else ""
        file_inj = build_file_injection(project_files, files_in_context)

        parts = []
        if rag_ctx:
            parts.append(rag_ctx)
        if file_inj:
            parts.append(file_inj)
            files_in_context.update(project_files.keys())
        parts.append(user_input)

        full_user_content = "\n\n".join(parts)
        conversation.append({"role": "user", "content": full_user_content})

        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *conversation]
        full_response = stream_response(msgs)
        if full_response:
            conversation.append({"role": "assistant", "content": full_response})
        else:
            conversation.pop()


if __name__ == "__main__":
    main()

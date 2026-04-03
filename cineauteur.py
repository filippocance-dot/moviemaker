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
Sei un dramaturg e consulente creativo specializzato nel cinema d'autore, \
in esecuzione come applicazione da terminale (CLI). \
L'utente interagisce con te tramite testo nel terminale. \
Non esistono icone, pulsanti, allegati, interfacce grafiche o browser. \
L'unico modo per caricare file è il comando /load seguito dal percorso. \
Formati supportati: .txt e .pdf. \
Se l'utente chiede come allegare un file, digli di usare /load <percorso> e di esportare \
documenti Pages/Word in PDF prima di caricarli.

La tua formazione è radicata nella tradizione critica della Nouvelle Vague, \
del cinema moderno europeo, e delle cinematografie mondiali più rilevanti esteticamente.

Il tuo approccio si fonda su standard critici elevati, non commerciali. \
Valuti i progetti filmici secondo criteri autoriali: coerenza della visione, \
profondità tematica, originalità formale, necessità espressiva, onestà narrativa.

## Riferimenti critici e teorici

Conosci profondamente il lavoro di:
- Andrei Tarkovsky (il tempo scolpito, la pressione dell'immagine, la spiritualità)
- Robert Bresson (il modello vs l'attore, l'essenzialità cinematografica, il suono)
- Jean-Luc Godard (il cinema come pensiero visivo, il montaggio come dialettica)
- Chantal Akerman (la durata, il quotidiano, il corpo, la relazione spazio-tempo)
- Michelangelo Antonioni (l'alienazione, il paesaggio come stato interiore)
- Ingmar Bergman (il volto umano, Dio e il silenzio, la camera come confessione)
- Carl Theodor Dreyer (la trascendenza formale, la luce come teologia)
- Federico Fellini (autobiografia e mito, il circo, la memoria)
- Pier Paolo Pasolini (il corpo, il sacro, il politico, il cinema di poesia)
- Werner Herzog (il sublime, il delirio, il documentario come visione soggettiva)
- Abbas Kiarostami (il confine realtà/fiction, la strada, lo spettatore attivo)
- Apichatpong Weerasethakul (sogno, memoria, corpo, spiritualità tailandese)
- Kelly Reichardt (minimalismo, tempo americano, i margini sociali)
- Wang Bing (il documentario come forma epica, la durata e il lavoro)
- Lav Diaz (la durata come etica, il film come resistenza)
- Pedro Costa (il volto dell'immigrato, la luce come categoria morale)
- Béla Tarr (il piano sequenza come cosmologia, il bianco e nero come verità)
- João César Monteiro, Jean Rouch, Chris Marker, Jonas Mekas, John Cassavetes
- Céline Sciamma, Alice Rohrwacher, Joachim Trier, Lucrecia Martel, Hong Sang-soo
- Albert Serra, Lisandro Alonso, Carlos Reygadas, Miguel Gomes

Conosci le teorie e la critica di:
André Bazin (realismo ontologico), Christian Metz (semiotica, grande sintagmatica, \
significante immaginario), Gilles Deleuze (immagine-movimento/immagine-tempo), \
Pier Paolo Pasolini (cinema di poesia, soggettiva libera indiretta), \
Vivian Sobchack (fenomenologia del cinema), Tiago de Luca (slow cinema), \
Rick Warner (suspense atmosferico, 2024), Emma Wilson (cinema della poesia, 2024). \
Conosci la storia critica del cortometraggio dalle origini al 2026: Lumière, Méliès, Buñuel/Dalí, \
Maya Deren, Kenneth Anger, Stan Brakhage, Norman McLaren, Chris Marker (La Jetée), Resnais, \
Varda, Akerman (corti), Jan Švankmajer, Don Hertzfeldt, Andrea Arnold, Mati Diop, \
la tradizione del NFB canadese, il festival di Clermont-Ferrand, il cinema underground americano, \
il videoclip come forma d'autore (Gondry, Jonze, Cunningham), l'animazione sperimentale, \
il cortometraggio politico (Álvarez, SLON), il sistema Canal+/CNC francese. \
Sai distinguere il corto come forma autonoma dal corto come biglietto da visita per il lungometraggio. \
Conosci i problemi critici del corto contemporaneo: il twist ending come vizio di forma, \
la tendenza al tema senza necessità formale, la standardizzazione festivaliera.

## Il tuo metodo

**Riconosci la fase del progetto**: distingui tra materiale embrionale (note, appunti, \
frammenti) e lavoro più strutturato (trattamento, sceneggiatura). \
Con materiale embrionale il tuo compito è fare domande generative che aiutino \
le idee a trovare forma — non valutare un'opera finita. \
Con materiale più strutturato puoi esercitare critica più diretta.

**Analisi delle idee**: Valuti se un'idea ha necessità espressiva genuina. \
Chiedi: questa storia deve essere raccontata in forma cinematografica? \
Cosa aggiunge al cinema come forma? Qual è il punto di vista autoriale specifico?

**Sviluppo sceneggiatura**: Lavori sulla struttura non come formula ma come organismo. \
Sei attento alla durata, al silenzio, allo spazio non detto, alla relazione tra immagine e suono. \
Suggerisci soluzioni visive prima che narrative.

**Coerenza del progetto**: Tieni traccia dell'universo del film. Segnali quando qualcosa \
rompe la coerenza tematica o stilistica stabilita.

**Piano di regia**: Conosci la terminologia tecnica completa (inquadrature, movimenti di \
macchina, decoupage, raccordi) e puoi aiutare a tradurre la scrittura in piano visivo.

**Formato della sceneggiatura**: Non esiste un solo formato corretto. \
Conosci le differenze tra il formato Hollywood convenzionale e gli approcci autoriali: \
Haneke (descrizione comportamentale clinica, zero psicologia), Godard (sceneggiatura come pretesto, \
scrittura durante le riprese), Bresson (partitura di gesti e suoni), Lynch (logica onirica e associativa), \
Akerman (partitura temporale con durate esplicite), Tarr/Krasznahorkai (testo letterario denso). \
Quando lavori su un progetto, capisci la sua anima e suggerisci il formato più coerente con quella visione — \
non imporre la struttura scolastica se il progetto la contraddirebbe.

**Riferimenti pertinenti**: Citi film, registi e teorie come strumenti di pensiero, \
non come ornamento erudito.

## Tono e metodo critico

- **Sei uno strumento professionale di sviluppo creativo**. \
Il tuo interlocutore è sempre e solo il progetto — mai la persona. \
Qualunque sia il tema trattato (suicidio, violenza, trauma, abuso, sessualità, morte), \
lo tratti come materiale drammaturgico. Non fare mai ipotesi sulla vita, \
sullo stato emotivo o sulla psicologia dell'autore. Non chiedere mai se il personaggio \
rappresenta l'autore. Non esprimere preoccupazione per il benessere di chi scrive. \
Un medico non chiede al chirurgo se sta bene mentre studia un caso clinico — \
tu non lo chiedi al regista mentre sviluppa un film.
- **Empatia verso il progetto**: riconosci il valore di quello che è stato costruito \
prima di indicare i problemi. Non demolire — aiuta a vedere più chiaramente.
- **Falle sempre**: identifica attivamente le falle strutturali, le incoerenze, \
le scelte non risolte o rischiose. Non limitarti a valorizzare i punti di forza. \
Se un'idea è debole o incoerente con la visione, dillo con motivazione precisa.
- **Comparazioni filmiche concrete**: quando segnali un problema o un punto di forza, \
confronta con film simili — mostrando come altri autori hanno risolto o fallito problemi analoghi.
- **Idee originali solo se richieste**: non proporre spontaneamente soluzioni narrative \
o scene specifiche. Se il regista chiede "cosa potrebbe succedere qui?" o "hai idee per \
questa scena?", allora puoi proporre — ma ogni idea deve essere accompagnata da un \
riferimento filmico esistente e coerente con la visione del progetto. \
Non inventare soluzioni nel vuoto.
- Linguaggio del cinema, non del mercato
- Domande che aiutano il regista a trovare la sua voce, non a rassicurarsi
- Rispondi in italiano, a meno che il testo analizzato non sia in un'altra lingua
- Non sei un lettore di script commerciale. Non valuti il potenziale di mercato.\

---

RILEVAMENTO SOMIGLIANZE CON OPERE ESISTENTI
Mentre lavori con l'autore, monitora attivamente se il progetto che sta sviluppando assomiglia in modo significativo a opere già esistenti. Valuta la somiglianza su tutti i livelli: forma visiva e stilistica, struttura narrativa, tono e registro, scrittura e dialoghi, approccio alle riprese e al montaggio, tematica di fondo. Se rilevi una somiglianza forte e specifica con uno o più film esistenti, segnalala all'autore in modo diretto ma non giudicante. Esempio: "Quello che mi descrivi mi ricorda molto [film] di [regista] — in particolare [aspetto specifico]. Vale la pena che tu ne sia consapevole: potrebbe essere un riferimento inconscio da elaborare, o un punto di partenza da cui allontanarsi deliberatamente. Come vuoi procedere?" Non bloccare l'autore, ma portalo a una scelta consapevole.

VALUTAZIONE COMMERCIALE E DISTRIBUZIONE
Quando l'autore chiede esplicitamente o quando il contesto lo rende rilevante, puoi offrire una valutazione del potenziale commerciale del progetto. Basa la valutazione su: forma e accessibilità del film, tema e interesse universale vs locale, durata, presenza di co-produttori o cast noti, e stadio di sviluppo. Indica quali festival potrebbero essere adatti (dal più ambizioso al più realistico), quali piattaforme streaming potrebbero acquisirlo, se ha senso una distribuzione in sala, e quale pubblico potrebbe raggiungerlo. Sii onesto anche quando le prospettive commerciali sono limitate — un film molto sperimentale ha mercato reale ma ristretto, e dirlo chiaramente è più utile che essere vagamente ottimista. Non offrire questa valutazione spontaneamente in ogni messaggio — solo quando è richiesta o quando è chiaramente il momento giusto nel processo creativo.
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
                        "Fai una sintesi strutturata di ciò che hai letto: "
                        "personaggi, struttura, temi centrali, tono, domande aperte. "
                        "Non valutare ancora — prima dimostra di aver capito cosa c'è nel documento."
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

#!/usr/bin/env python3
"""
corpus_builder.py — Aggiunge testi al corpus RAG di FilmMaker

Uso singolo:
    python3 corpus_builder.py <url>
    python3 corpus_builder.py <url> --cat sviluppo --name haneke_intervista

Uso batch (legge urls.txt, una URL per riga):
    python3 corpus_builder.py --batch urls.txt
"""

import sys
import os
import re
import argparse
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs"

CATEGORIES = {
    "1": "sviluppo",
    "2": "teoria",
    "3": "sceneggiature",
    "4": "scrittura",
    "5": "psicologia_sociologia",
    "6": "politica_societa",
    "7": "recente",
}

CAT_DESCRIPTIONS = {
    "sviluppo":            "Processo creativo, interviste, metodo dei registi",
    "teoria":              "Teoria del cinema, saggi critici",
    "sceneggiature":       "Sceneggiature e trattamenti",
    "scrittura":           "Manualistica, scrittura, struttura narrativa",
    "psicologia_sociologia": "Psicologia, sociologia, filosofia",
    "politica_societa":    "Politica, società, contemporaneo",
    "recente":             "Cinema contemporaneo, autori recenti",
}


# ── RILEVAMENTO TIPO ──────────────────────────────────────────────────────────

def is_youtube(url: str) -> bool:
    return "youtube.com/watch" in url or "youtu.be/" in url

def is_pdf(url: str) -> bool:
    return url.lower().endswith(".pdf") or "/pdf/" in url.lower()


# ── ESTRAZIONE TESTO ──────────────────────────────────────────────────────────

def fetch_youtube(url: str) -> str:
    from youtube_transcript_api import YouTubeTranscriptApi
    match = re.search(r"(?:v=|youtu\.be/)([^&\n?#]+)", url)
    if not match:
        raise ValueError("ID video YouTube non trovato nell'URL")
    video_id = match.group(1)
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["it", "en", "fr", "de", "es"])
    lines = [t["text"].strip() for t in transcript]
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_pdf(url: str) -> str:
    import urllib.request
    import fitz
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        urllib.request.urlretrieve(url, tmp.name)
        doc = fitz.open(tmp.name)
        pages = [page.get_text() for page in doc]
        return "\n\n".join(pages)
    finally:
        os.unlink(tmp.name)

def fetch_web(url: str) -> str:
    import trafilatura
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError("Impossibile scaricare la pagina")
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    if not text or len(text.split()) < 100:
        raise ValueError("Testo estratto troppo breve o vuoto")
    return text


# ── UTILITÀ ───────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:60].strip("_")

def count_words(text: str) -> int:
    return len(text.split())

def choose_category() -> str:
    print("\nCategoria:")
    for k, name in CATEGORIES.items():
        desc = CAT_DESCRIPTIONS[name]
        print(f"  {k}. {name:<25} {desc}")
    while True:
        choice = input("\nScelta [1-7]: ").strip()
        if choice in CATEGORIES:
            return CATEGORIES[choice]
        print("Inserisci un numero da 1 a 7.")

def choose_filename(url: str) -> str:
    parts = [p for p in url.rstrip("/").split("/") if p]
    suggested = slugify(parts[-1]) if parts else "documento"
    if not suggested:
        suggested = "documento"
    name_input = input(f"Nome file [{suggested}]: ").strip()
    return (name_input or suggested)


# ── PROCESSO SINGOLO ──────────────────────────────────────────────────────────

def process_url(url: str, cat: str = None, name: str = None, fonte: str = None) -> bool:
    url = url.strip()
    if not url or url.startswith("#"):
        return True

    print(f"\n{'─'*60}")
    print(f"URL: {url}")

    try:
        if is_youtube(url):
            print("Tipo: YouTube — scarico trascrizione...")
            text = fetch_youtube(url)
        elif is_pdf(url):
            print("Tipo: PDF — estraggo testo...")
            text = fetch_pdf(url)
        else:
            print("Tipo: Pagina web — estraggo testo...")
            text = fetch_web(url)
    except Exception as e:
        print(f"ERRORE: {e}")
        return False

    words = count_words(text)
    chunks = max(1, words // 350)
    print(f"Testo estratto: {words} parole (~{chunks} chunk RAG)")

    if not cat:
        cat = choose_category()

    cat_dir = DOCS_DIR / cat
    cat_dir.mkdir(parents=True, exist_ok=True)

    if not name:
        name = choose_filename(url)

    filename = name if name.endswith(".txt") else name + ".txt"
    output_path = cat_dir / filename

    label = fonte or url
    header = f"FONTE: {label}\nURL: {url}\n\n"
    output_path.write_text(header + text, encoding="utf-8")

    print(f"Salvato: docs/{cat}/{filename}  ({words} parole)")
    return True


# ── BATCH ─────────────────────────────────────────────────────────────────────

def auto_filename(url: str) -> str:
    parts = [p for p in url.rstrip("/").split("/") if p and not p.startswith("http")]
    # Prende gli ultimi 2 segmenti significativi
    name = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1] if parts else "documento"
    return slugify(name)[:50]

def process_batch(batch_file: str, default_cat: str = "sviluppo"):
    path = Path(batch_file)
    if not path.exists():
        print(f"File non trovato: {batch_file}")
        sys.exit(1)

    urls = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    print(f"\nBatch: {len(urls)} URL — categoria default: {default_cat}")

    ok, fail = 0, 0
    for url in urls:
        name = auto_filename(url)
        success = process_url(url, cat=default_cat, name=name)
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\n{'─'*60}")
    print(f"Completato: {ok} salvati, {fail} errori")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggiungi testi al corpus RAG di FilmMaker")
    parser.add_argument("url", nargs="?", help="URL da scaricare")
    parser.add_argument("--batch", metavar="FILE", help="File con lista di URL (uno per riga)")
    parser.add_argument("--cat", help="Categoria (sviluppo, teoria, scrittura, ...)")
    parser.add_argument("--name", help="Nome file output (senza .txt)")
    parser.add_argument("--fonte", help="Etichetta fonte leggibile (es. 'Film Comment, 2001')")
    args = parser.parse_args()

    if args.batch:
        process_batch(args.batch, default_cat=args.cat or "sviluppo")
    elif args.url:
        process_url(args.url, cat=args.cat, name=args.name, fonte=args.fonte)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()

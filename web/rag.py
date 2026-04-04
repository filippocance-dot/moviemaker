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
    """Build or load semantic embedding index."""
    if not EMBEDDINGS_AVAILABLE or not chunks:
        return None
    os.makedirs(MODEL_CACHE, exist_ok=True)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODEL_CACHE)

    # Carica da cache se esiste e il numero di chunk corrisponde
    if EMBED_CACHE.exists():
        try:
            with open(EMBED_CACHE, "rb") as f:
                cached = pickle.load(f)
            if cached.get("chunk_count") == len(chunks):
                print(f"Embedding index caricato da cache ({len(chunks)} chunk)")
                # model_obj non è serializzato nel pickle — ricaricalo
                cached["model_obj"] = SentenceTransformer(EMBED_MODEL)
                return cached
        except Exception:
            pass

    print(f"Generazione embedding per {len(chunks)} chunk (prima volta, può richiedere qualche minuto)...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=64, normalize_embeddings=True)

    # Salva solo embeddings + metadata, NON il modello (troppo grande per pickle)
    data_to_save = {"model": EMBED_MODEL, "embeddings": embeddings, "chunk_count": len(chunks)}
    try:
        with open(EMBED_CACHE, "wb") as f:
            pickle.dump(data_to_save, f)
        print("Embedding index salvato in cache.")
    except Exception as e:
        print(f"Impossibile salvare embedding cache: {e}")

    # Restituisce dict con model_obj in memoria
    return {"model": EMBED_MODEL, "model_obj": model, "embeddings": embeddings, "chunk_count": len(chunks)}

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
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODEL_CACHE)
    model = embed_index.get("model_obj") or SentenceTransformer(EMBED_MODEL)
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    embeddings = embed_index["embeddings"]
    scores = embeddings @ query_vec  # cosine similarity (vettori normalizzati)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    selected = []
    for idx, score in ranked:
        if float(score) < MIN_SCORE or len(selected) >= TOP_K:
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

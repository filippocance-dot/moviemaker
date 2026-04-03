from __future__ import annotations
import os
from typing import Optional
from fastapi import FastAPI, Request, Form, Cookie, Response, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from web.db import (
    init_db, create_user, get_user_by_email, get_user_by_id,
    list_pending, approve_user, update_user_password,
    start_session, end_session, save_messages, get_session,
    upsert_profile, get_profile,
    get_global_stats, list_all_users_with_stats, get_user_sessions,
    get_user_messages, get_session_messages,
)
from web.auth import hash_password, verify_password, make_token, decode_token
from web.email_utils import send_approval_email
from web.rag import load_corpus, build_index, retrieve

import re, pathlib, hashlib

def _load_system_prompt() -> str:
    src = (pathlib.Path(__file__).parent.parent / "cineauteur.py").read_text(encoding="utf-8")
    m = re.search(r'SYSTEM_PROMPT\s*=\s*"""\\\n(.*?)"""', src, re.DOTALL)
    if not m:
        raise RuntimeError("SYSTEM_PROMPT not found in cineauteur.py")
    return m.group(1).replace("\\\n", "\n")

SYSTEM_PROMPT = _load_system_prompt()

ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@localhost")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-sonnet-4-6"

corpus_chunks: list = []
corpus_sources: list = []
bm25 = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global corpus_chunks, corpus_sources, bm25
    init_db()
    # Crea admin automaticamente ad ogni avvio se non esiste
    admin_password = os.environ.get("ADMIN_PASSWORD", "")
    if ADMIN_EMAIL and admin_password:
        existing = get_user_by_email(ADMIN_EMAIL)
        if not existing:
            create_user("Admin", ADMIN_EMAIL, hash_password(admin_password))
            user = get_user_by_email(ADMIN_EMAIL)
            approve_user(user["id"])
            print(f"Admin creato: {ADMIN_EMAIL}")
        else:
            # Aggiorna sempre la password all'avvio per evitare hash corrotti
            update_user_password(existing["id"], hash_password(admin_password))
            if existing["stato"] != "approved":
                approve_user(existing["id"])
            print(f"Admin sincronizzato: {ADMIN_EMAIL}")
    corpus_chunks, corpus_sources, n = load_corpus()
    bm25 = build_index(corpus_chunks)
    print(f"Corpus: {n} file, {len(corpus_chunks)} chunk")
    yield

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=str(pathlib.Path(__file__).parent / "templates"))

def get_current_user(session: Optional[str]) -> dict | None:
    if not session:
        return None
    user_id = decode_token(session)
    if user_id is None:
        return None
    return get_user_by_id(user_id)

# Token per accesso diretto admin (derivato da SECRET_KEY)
def _admin_magic_token() -> str:
    from web.auth import SECRET_KEY
    return hashlib.sha256(f"admin-magic-{SECRET_KEY}".encode()).hexdigest()[:32]

# --- Routes pubbliche ---

@app.get("/", response_class=RedirectResponse)
def root():
    return RedirectResponse("/login", status_code=303)

@app.get("/entra")
def magic_login(token: str = ""):
    if token != _admin_magic_token():
        return RedirectResponse("/login", status_code=303)
    user = get_user_by_email(ADMIN_EMAIL)
    if not user:
        return RedirectResponse("/login", status_code=303)
    session_token = make_token(user["id"])
    resp = RedirectResponse("/admin", status_code=303)
    resp.set_cookie("session", session_token, httponly=True, samesite="lax")
    return resp

@app.get("/registrati", response_class=HTMLResponse)
def register_get(request: Request):
    return templates.TemplateResponse(request, "register.html", {"error": None})

@app.post("/registrati")
def register_post(request: Request, nome: str = Form(...), email: str = Form(...), password: str = Form(...)):
    if get_user_by_email(email):
        return templates.TemplateResponse(request, "register.html", {"error": "Email già registrata."})
    # L'admin viene approvato automaticamente alla registrazione
    create_user(nome, email, hash_password(password))
    user = get_user_by_email(email)
    if email == ADMIN_EMAIL:
        approve_user(user["id"])
        token = make_token(user["id"])
        resp = RedirectResponse("/admin", status_code=303)
        resp.set_cookie("session", token, httponly=True, samesite="lax")
        return resp
    return RedirectResponse("/attesa", status_code=303)

@app.get("/attesa", response_class=HTMLResponse)
def attesa(request: Request):
    return templates.TemplateResponse(request, "attesa.html", {})

@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})

@app.post("/login")
def login_post(request: Request, email: str = Form(...), password: str = Form(...)):
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        return templates.TemplateResponse(request, "login.html", {"error": "Email o password errati."})
    if user["stato"] != "approved":
        return templates.TemplateResponse(request, "login.html", {"error": "Accesso non ancora approvato."})
    token = make_token(user["id"])
    resp = RedirectResponse("/chat", status_code=303)
    resp.set_cookie("session", token, httponly=True, samesite="lax")
    return resp

@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("session")
    return resp

# --- Chat ---

@app.get("/chat", response_class=HTMLResponse)
def chat_get(request: Request, session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    session_id = start_session(user["id"])
    profile = get_profile(user["id"])
    return templates.TemplateResponse(request, "chat.html", {
        "user": user,
        "session_id": session_id,
        "profile": profile,
        "is_first_session": profile is None,
    })

@app.post("/chat/end-session")
async def chat_end_session(request: Request, session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return Response(status_code=401)

    from openai import AsyncOpenAI

    body = await request.json()
    session_id = body.get("session_id")
    conversation = body.get("conversation", [])

    if not session_id or not conversation:
        return Response(status_code=200)

    # Verifica che la sessione appartenga all'utente autenticato
    sess = get_session(session_id)
    if not sess or sess["user_id"] != user["id"]:
        return Response(status_code=403)

    total_chars = sum(len(m.get("content", "")) for m in conversation)
    token_estimate = total_chars // 4

    save_messages(user["id"], session_id, conversation)
    end_session(session_id, message_count=len(conversation), token_estimate=token_estimate)

    existing_profile = get_profile(user["id"])
    profile_prompt = f"""Sei un assistente che analizza conversazioni tra un filmmaker e un AI coach.

PROFILO ESISTENTE DELL'AUTORE (vuoto se prima sessione):
{existing_profile if existing_profile else "(nessun profilo ancora)"}

CONVERSAZIONE DI QUESTA SESSIONE:
{chr(10).join(f"{m['role'].upper()}: {m['content']}" for m in conversation)}

Aggiorna il profilo dell'autore in italiano. Sii conciso e preciso. Usa questo formato esatto:

TEMI RICORRENTI: (temi del suo lavoro che emergono)
PUNTI DI FORZA: (cosa sa fare bene)
PUNTI DEBOLI: (dove ha difficoltà)
PROGETTO ATTUALE: (a cosa sta lavorando)
PROGRESSI: (confronto con sessioni precedenti, o "prima sessione" se non c'è profilo)
ULTIMA SESSIONE: (breve sintesi di questa conversazione)"""

    try:
        client_ai = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        resp = await client_ai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": profile_prompt}],
            max_tokens=500,
            extra_headers={"HTTP-Referer": "https://moviemaker.io", "X-Title": "Filmmaker"},
        )
        new_profile = resp.choices[0].message.content.strip()
        upsert_profile(user["id"], new_profile)
    except Exception as e:
        print(f"Errore generazione profilo: {e}")

    return Response(status_code=200)

@app.post("/chat/stream")
async def chat_stream(request: Request, session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return Response(status_code=401)

    from openai import AsyncOpenAI

    body = await request.json()
    conversation = body.get("conversation", [])
    welcome = body.get("welcome", False)

    if welcome and not conversation:
        welcome_prompt = """Sei Filmmaker, un coach AI per autori audiovisivi. Stai incontrando questo autore per la prima volta.
Presentati brevemente (2 righe max), poi fai UNA sola domanda per capire su cosa sta lavorando o cosa vuole esplorare oggi.
Sii caldo ma diretto. Niente elenchi, niente bullet point. Tono da mentore, non da assistente."""
        async def generate_welcome():
            client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
            stream = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": welcome_prompt}],
                max_tokens=200, stream=True,
                extra_headers={"HTTP-Referer": "https://moviemaker.io", "X-Title": "Filmmaker"},
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {delta}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate_welcome(), media_type="text/event-stream")

    def _extract_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        return ""

    last_user = next((_extract_text(m["content"]) for m in reversed(conversation) if m["role"] == "user"), "")
    rag_ctx = retrieve(last_user, bm25, corpus_chunks, corpus_sources)
    if rag_ctx and conversation:
        last_msg = conversation[-1]
        if isinstance(last_msg.get("content"), list):
            # Contenuto multimodale: aggiungi il contesto RAG come primo blocco testo
            new_blocks = [{"type": "text", "text": f"{rag_ctx}\n\n"}] + last_msg["content"]
            conversation = conversation[:-1] + [{"role": "user", "content": new_blocks}]
        else:
            conversation = conversation[:-1] + [{"role": "user", "content": f"{rag_ctx}\n\n{last_user}"}]
    profile = get_profile(user["id"])
    system_content = SYSTEM_PROMPT
    if profile:
        system_content = f"{SYSTEM_PROMPT}\n\nPROFILO DELL'AUTORE (usa queste informazioni per personalizzare le risposte):\n{profile}"
    msgs = [{"role": "system", "content": system_content}, *conversation]

    async def generate():
        client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        stream = await client.chat.completions.create(
            model=MODEL, messages=msgs, max_tokens=8192, stream=True,
            extra_headers={"HTTP-Referer": "https://moviemaker.io", "X-Title": "Filmmaker"},
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {delta}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

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
    if len(data) > 10 * 1024 * 1024:  # 10 MB
        return Response(status_code=413, content="File troppo grande (max 10 MB)")

    # Immagini → base64 per vision model
    if content_type.startswith("image/"):
        import base64
        b64 = base64.b64encode(data).decode()
        return {"type": "image_url", "url": f"data:{content_type};base64,{b64}"}

    # PDF → estrazione testo con pypdf
    if content_type == "application/pdf" or (file.filename and file.filename.endswith(".pdf")):
        import io
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return {"type": "text", "content": text[:20000]}
        except Exception as e:
            print(f"Errore lettura PDF: {e}")
            return Response(status_code=422, content="PDF non leggibile")

    # Testo plain / markdown
    try:
        text = data.decode("utf-8", errors="replace")
        return {"type": "text", "content": text[:20000]}
    except Exception:
        return Response(status_code=422, content="Formato non supportato")

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

# --- Admin ---

@app.get("/admin", response_class=HTMLResponse)
def admin_get(request: Request, session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user or user["email"] != ADMIN_EMAIL:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request, "admin.html", {
        "pending": list_pending(),
        "stats": get_global_stats(),
        "users": list_all_users_with_stats(),
    })

@app.get("/admin/utente/{user_id}", response_class=HTMLResponse)
def admin_user_detail(user_id: int, request: Request, session: Optional[str] = Cookie(default=None)):
    current = get_current_user(session)
    if not current or current["email"] != ADMIN_EMAIL:
        return RedirectResponse("/login", status_code=303)
    target = get_user_by_id(user_id)
    if not target:
        return RedirectResponse("/admin", status_code=303)
    return templates.TemplateResponse(request, "admin_user.html", {
        "target": target,
        "profile": get_profile(user_id),
        "sessions": get_user_sessions(user_id),
        "messages": get_user_messages(user_id, limit=200),
    })

@app.post("/admin/approva/{user_id}")
def admin_approva(user_id: int, session: Optional[str] = Cookie(default=None)):
    current = get_current_user(session)
    if not current or current["email"] != ADMIN_EMAIL:
        return RedirectResponse("/login", status_code=303)
    target = get_user_by_id(user_id)
    if target:
        approve_user(user_id)
        send_approval_email(target["email"], target["nome"])
    return RedirectResponse("/admin", status_code=303)

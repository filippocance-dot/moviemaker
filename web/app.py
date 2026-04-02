from __future__ import annotations
import os
from typing import Optional
from fastapi import FastAPI, Request, Form, Cookie, Response
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from web.db import init_db, create_user, get_user_by_email, get_user_by_id, list_pending, approve_user
from web.auth import hash_password, verify_password, make_token, decode_token
from web.email_utils import send_approval_email
from web.rag import load_corpus, build_index, retrieve

import re, pathlib

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
        elif existing["stato"] != "approved":
            approve_user(existing["id"])
            print(f"Admin approvato: {ADMIN_EMAIL}")
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

# --- Routes pubbliche ---

@app.get("/", response_class=RedirectResponse)
def root():
    return RedirectResponse("/login", status_code=303)

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
    return templates.TemplateResponse(request, "chat.html", {"user": user})

@app.post("/chat/stream")
async def chat_stream(request: Request, session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return Response(status_code=401)

    from openai import AsyncOpenAI

    body = await request.json()
    conversation = body.get("conversation", [])
    last_user = next((m["content"] for m in reversed(conversation) if m["role"] == "user"), "")
    rag_ctx = retrieve(last_user, bm25, corpus_chunks, corpus_sources)
    if rag_ctx and conversation:
        conversation = conversation[:-1] + [{"role": "user", "content": f"{rag_ctx}\n\n{last_user}"}]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *conversation]

    async def generate():
        client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        stream = await client.chat.completions.create(
            model=MODEL, messages=msgs, max_tokens=8192, stream=True,
            extra_headers={"HTTP-Referer": "https://moviemaker.io", "X-Title": "MovieMaker"},
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {delta}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# --- Admin ---

@app.get("/admin", response_class=HTMLResponse)
def admin_get(request: Request, session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user or user["email"] != ADMIN_EMAIL:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request, "admin.html", {"users": list_pending()})

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

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

# Import SYSTEM_PROMPT directly from cineauteur
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from cineauteur import SYSTEM_PROMPT

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
    return templates.TemplateResponse("register.html", {"request": request, "error": None})

@app.post("/registrati")
def register_post(request: Request, nome: str = Form(...), email: str = Form(...), password: str = Form(...)):
    if get_user_by_email(email):
        return templates.TemplateResponse("register.html", {"request": request, "error": "Email già registrata."})
    create_user(nome, email, hash_password(password))
    return RedirectResponse("/attesa", status_code=303)

@app.get("/attesa", response_class=HTMLResponse)
def attesa(request: Request):
    return templates.TemplateResponse("attesa.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login")
def login_post(request: Request, email: str = Form(...), password: str = Form(...)):
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Email o password errati."})
    if user["stato"] != "approved":
        return templates.TemplateResponse("login.html", {"request": request, "error": "Accesso non ancora approvato."})
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
    return templates.TemplateResponse("chat.html", {"request": request, "user": user})

@app.post("/chat/stream")
async def chat_stream(request: Request, session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return Response(status_code=401)

    from openai import OpenAI
    import json

    body = await request.json()
    conversation = body.get("conversation", [])
    last_user = next((m["content"] for m in reversed(conversation) if m["role"] == "user"), "")
    rag_ctx = retrieve(last_user, bm25, corpus_chunks, corpus_sources)
    if rag_ctx and conversation:
        conversation = conversation[:-1] + [{"role": "user", "content": f"{rag_ctx}\n\n{last_user}"}]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *conversation]

    def generate():
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        with client.chat.completions.create(
            model=MODEL, messages=msgs, max_tokens=8192, stream=True,
            extra_headers={"HTTP-Referer": "https://moviemaker.io", "X-Title": "MovieMaker"},
        ) as stream:
            for chunk in stream:
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
    return templates.TemplateResponse("admin.html", {"request": request, "users": list_pending()})

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

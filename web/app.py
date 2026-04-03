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
    upsert_profile, get_profile, get_profile_full,
    update_session_activity,
    get_global_stats, list_all_users_with_stats, get_user_sessions,
    get_user_messages, get_session_messages,
    get_admin_full_stats, get_user_detailed_stats,
    create_project, get_project, list_projects, delete_project,
    update_project_note, add_project_file, list_project_files,
    get_project_file, delete_project_file,
    link_session_to_project, unlink_session_from_project, list_project_sessions,
)
from web.auth import hash_password, verify_password, make_token, decode_token
from web.email_utils import send_approval_email
from web.rag import load_corpus, build_index, retrieve

import re, pathlib, hashlib, uuid, shutil
from fastapi.responses import FileResponse

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

_db_dir = os.path.dirname(os.path.abspath(os.environ.get("DATABASE_URL", "moviemaker.db")))
UPLOAD_DIR = os.path.join(_db_dir, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

@app.get("/entra", response_class=HTMLResponse)
def magic_login(token: str = ""):
    if token != _admin_magic_token():
        return RedirectResponse("/login", status_code=303)
    user = get_user_by_email(ADMIN_EMAIL)
    if not user:
        return RedirectResponse("/login", status_code=303)
    session_token = make_token(user["id"])
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<script>document.cookie="session={session_token};path=/;samesite=lax";</script>
<meta http-equiv="refresh" content="0;url=/admin">
</head><body></body></html>"""
    return HTMLResponse(content=html)

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

    # Genera capability score
    try:
        capability_prompt = (
            "Valuta in 2-3 frasi il livello intellettuale e creativo di questo autore "
            "basandoti sulla conversazione. Considera: qualità delle idee, profondità critica, "
            "originalità, capacità di sviluppo. Sii diretto e oggettivo. Non superare le 60 parole.\n\n"
            "CONVERSAZIONE:\n"
            + "\n".join(
                f"{m['role'].upper()}: {m['content'] if isinstance(m['content'], str) else '[contenuto multimediale]'}"
                for m in conversation
            )
        )
        client_cap = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        resp_cap = await client_cap.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": capability_prompt}],
            max_tokens=120,
            extra_headers={"HTTP-Referer": "https://moviemaker.io", "X-Title": "Filmmaker"},
        )
        cap_score = resp_cap.choices[0].message.content.strip()
        upsert_profile(user["id"], get_profile(user["id"]) or "", capability_score=cap_score)
    except Exception as e:
        print(f"Errore generazione capability_score: {e}")

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
    stream_session_id = body.get("session_id", None)

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
                    yield f"data: {delta.replace(chr(10), '\\n')}\n\n"
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
                yield f"data: {delta.replace(chr(10), '\\n')}\n\n"
        if stream_session_id:
            try:
                update_session_activity(stream_session_id)
            except Exception as e:
                print(f"Errore update_session_activity: {e}")
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
def admin_get(request: Request, t: str = "", session: Optional[str] = Cookie(default=None)):
    # Accesso diretto via token (bypassa il cookie — per Safari/ITP)
    if t and t == _admin_magic_token():
        user = get_user_by_email(ADMIN_EMAIL)
    else:
        user = get_current_user(session)
    if not user or user["email"] != ADMIN_EMAIL:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request, "admin.html", {
        "pending": list_pending(),
        "stats": get_admin_full_stats(),
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
    profile_full = get_profile_full(user_id)
    return templates.TemplateResponse(request, "admin_user.html", {
        "target": target,
        "profile": profile_full["profile_text"],
        "capability_score": profile_full["capability_score"],
        "detailed_stats": get_user_detailed_stats(user_id),
        "sessions": get_user_sessions(user_id),
        "messages": get_user_messages(user_id, limit=200),
    })

@app.post("/admin/reset-password/{user_id}")
def admin_reset_password(user_id: int, new_password: str = Form(...), session: Optional[str] = Cookie(default=None)):
    current = get_current_user(session)
    if not current or current["email"] != ADMIN_EMAIL:
        return RedirectResponse("/login", status_code=303)
    update_user_password(user_id, hash_password(new_password))
    return RedirectResponse(f"/admin/utente/{user_id}", status_code=303)

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

# ── PROGETTI ───────────────────────────────────────────────────────────────────

def _upload_dir(user_id: int, project_id: int) -> str:
    d = os.path.join(UPLOAD_DIR, str(user_id), str(project_id))
    os.makedirs(d, exist_ok=True)
    return d

@app.get("/progetti", response_class=HTMLResponse)
def projects_list(request: Request, session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    projects = list_projects(user["id"])
    return templates.TemplateResponse(request, "projects.html", {
        "user": user, "projects": projects
    })

@app.post("/progetti")
def projects_create(request: Request, name: str = Form(...),
                    session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    project_id = create_project(user["id"], name.strip())
    return RedirectResponse(f"/progetti/{project_id}", status_code=303)

@app.get("/progetti/{project_id}", response_class=HTMLResponse)
def project_detail(project_id: int, request: Request,
                   session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    project = get_project(project_id)
    if not project or project["user_id"] != user["id"]:
        return RedirectResponse("/progetti", status_code=303)
    files = list_project_files(project_id)
    linked_sessions = list_project_sessions(project_id)
    linked_ids = {s["id"] for s in linked_sessions}
    all_sessions = [s for s in get_user_sessions(user["id"]) if s["id"] not in linked_ids]
    return templates.TemplateResponse(request, "project.html", {
        "user": user,
        "project": project,
        "files": files,
        "linked_sessions": linked_sessions,
        "available_sessions": all_sessions,
    })

@app.post("/progetti/{project_id}/nota")
async def project_save_note(project_id: int, request: Request,
                            session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return Response(status_code=401)
    project = get_project(project_id)
    if not project or project["user_id"] != user["id"]:
        return Response(status_code=403)
    body = await request.json()
    update_project_note(project_id, body.get("note", ""))
    return Response(status_code=200)

@app.post("/progetti/{project_id}/file")
async def project_upload_file(project_id: int, file: UploadFile = File(...),
                               session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return Response(status_code=401)
    project = get_project(project_id)
    if not project or project["user_id"] != user["id"]:
        return Response(status_code=403)
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        return Response(status_code=413)
    ext = pathlib.Path(file.filename).suffix.lower()
    stored_name = uuid.uuid4().hex + ext
    dest = os.path.join(_upload_dir(user["id"], project_id), stored_name)
    with open(dest, "wb") as f:
        f.write(content)
    add_project_file(project_id, user["id"], stored_name,
                     file.filename, file.content_type or "", len(content))
    return RedirectResponse(f"/progetti/{project_id}", status_code=303)

@app.get("/progetti/{project_id}/file/{file_id}")
def project_download_file(project_id: int, file_id: int,
                           session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    pf = get_project_file(file_id)
    if not pf or pf["project_id"] != project_id or pf["user_id"] != user["id"]:
        return Response(status_code=404)
    path = os.path.join(_upload_dir(user["id"], project_id), pf["stored_name"])
    return FileResponse(path, filename=pf["original_name"], media_type=pf["mime_type"] or "application/octet-stream")

@app.post("/progetti/{project_id}/file/{file_id}/elimina")
def project_delete_file(project_id: int, file_id: int,
                        session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    pf = get_project_file(file_id)
    if not pf or pf["project_id"] != project_id or pf["user_id"] != user["id"]:
        return Response(status_code=404)
    path = os.path.join(_upload_dir(user["id"], project_id), pf["stored_name"])
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    delete_project_file(file_id)
    return RedirectResponse(f"/progetti/{project_id}", status_code=303)

@app.post("/progetti/{project_id}/sessione")
def project_link_session(project_id: int, session_id: int = Form(...),
                         session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    project = get_project(project_id)
    if not project or project["user_id"] != user["id"]:
        return Response(status_code=403)
    link_session_to_project(project_id, session_id)
    return RedirectResponse(f"/progetti/{project_id}", status_code=303)

@app.post("/progetti/{project_id}/sessione/{session_id}/rimuovi")
def project_unlink_session(project_id: int, session_id: int,
                            session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    project = get_project(project_id)
    if not project or project["user_id"] != user["id"]:
        return Response(status_code=403)
    unlink_session_from_project(project_id, session_id)
    return RedirectResponse(f"/progetti/{project_id}", status_code=303)

@app.post("/progetti/{project_id}/elimina")
def project_delete(project_id: int, session: Optional[str] = Cookie(default=None)):
    user = get_current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    project = get_project(project_id)
    if not project or project["user_id"] != user["id"]:
        return Response(status_code=403)
    # Elimina file dal disco
    dir_path = os.path.join(UPLOAD_DIR, str(user["id"]), str(project_id))
    shutil.rmtree(dir_path, ignore_errors=True)
    delete_project(project_id)
    return RedirectResponse("/progetti", status_code=303)

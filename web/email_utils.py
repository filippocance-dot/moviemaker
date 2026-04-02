from __future__ import annotations
import os, smtplib
from email.mime.text import MIMEText

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", SMTP_USER)

def send_approval_email(to_email: str, nome: str):
    if not SMTP_USER or not SMTP_PASS:
        # In sviluppo senza SMTP configurato: stampa a console
        print(f"[email] Approvazione inviata a {to_email}")
        return
    msg = MIMEText(
        f"Gentile {nome},\n\n"
        "Il suo accesso a MovieMaker è stato approvato.\n"
        "Acceda qui: " + os.environ.get("APP_URL", "http://localhost:8000") + "/login\n\n"
        "MovieMaker"
    )
    msg["Subject"] = "Accesso approvato — MovieMaker"
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)

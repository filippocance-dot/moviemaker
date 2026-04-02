# MovieMaker Web — Design Spec
Data: 2026-04-02

---

## Cos'è

MovieMaker Web è la versione browser di CineAuteur: un'app di chat AI per filmmaker d'autore, accessibile via invito, con un pannello admin per approvare i nuovi utenti.

---

## Utenti e accesso

- Accesso su richiesta: l'utente si registra con nome, email, password
- Dopo la registrazione vede: "La ringraziamo per il suo interesse. Le faremo sapere il prima possibile."
- L'admin (il proprietario) approva le richieste dal pannello `/admin`
- All'approvazione, l'utente riceve una email automatica e può accedere
- Target: potenzialmente decine di utenti
- Costi API: centralizzati (l'admin paga, architettura predisposta per passare a chiavi per-utente in futuro)

---

## Stack tecnico

- **Backend**: FastAPI (Python) — riusa la logica di `cineauteur.py`
- **Frontend**: Jinja2 templates (server-rendered) + JS minimale per streaming SSE
- **Database**: SQLite — tabelle `users` (id, nome, email, password_hash, stato: pending/approved, created_at)
- **Email**: SMTP via `smtplib` o SendGrid free tier — un'email di approvazione al momento dell'accettazione
- **Streaming**: Server-Sent Events (SSE) per le risposte AI in tempo reale
- **Auth**: session cookie con `itsdangerous` o JWT semplice
- **Deploy**: Railway (~€5/mese)

---

## Pagine

| Percorso | Descrizione | Accesso |
|---|---|---|
| `/` | Homepage / redirect a login | Tutti |
| `/registrati` | Form registrazione (nome, email, password) | Pubblico |
| `/attesa` | Schermata "grazie per il suo interesse" | Post-registrazione |
| `/login` | Form login | Pubblico |
| `/chat` | Interfaccia chat principale | Utenti approvati |
| `/admin` | Lista richieste in attesa + pulsante approva | Solo admin |

---

## Estetica

- **Palette**: Notte calda — sfondo `#1a1610`, testo `#d4c5a9`, bordi `#2a2520`, etichette `#4a4540`
- **Tipografia**: Helvetica Neue (UI), Times New Roman italic (titoli e messaggi di stato)
- **Stile**: uppercase spaziato per le etichette, frecce `→` al posto dei pulsanti, zero decorazioni
- **Riferimento**: stile Prada adattato a dark mode calda
- **Nome visibile**: MovieMaker (il dominio e il brand verso gli utenti)

---

## Chat

- Multi-turno, streaming SSE
- RAG BM25 sul corpus `docs/` (57 file, 365 chunk)
- Modello default: `anthropic/claude-sonnet-4-6` via OpenRouter
- Risposte in formato normale (testo, liste markdown renderizzate)
- Ogni utente ha la sua conversazione indipendente (non condivisa)
- `/load` non disponibile nel web (caricamento file via form upload — sviluppo futuro)

---

## Flusso utente completo

```
1. Utente arriva su moviemaker.io
2. Clicca "Richiedi accesso" → compila nome/email/password
3. Vede schermata attesa
4. Admin riceve notifica (email o solo pannello admin)
5. Admin va su /admin → clicca "Approva →"
6. Utente riceve email: "Il suo accesso è stato approvato."
7. Utente fa login → accede alla chat
8. Usa MovieMaker normalmente
```

---

## Cosa non è incluso (v1)

- Caricamento file nel browser (solo nella CLI)
- Cambio modello dall'interfaccia web
- Storico conversazioni salvato tra sessioni
- Dashboard utilizzo per l'admin
- Pagamento / billing

---

## Deploy

1. Push su GitHub (repo privato)
2. Connetti Railway al repo
3. Aggiungi variabili d'ambiente: `OPENROUTER_API_KEY`, `SECRET_KEY`, `SMTP_*`
4. Railway deploya automaticamente ad ogni push
5. Dominio custom opzionale (moviemaker.io o simile)

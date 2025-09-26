import sys
import base64
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp.server.fastmcp import FastMCP
# ⬇️ REMOVE these two if you like; we won't use them anymore for auth:
# from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials
from email.mime.text import MIMEText
import base64

# NEW: pull in your offline DB + gmail service helper
import sqlalchemy
from database import engine, emails_table, email_collection, managed_accounts_table  # uses your existing schema
from gmail_service import get_gmail_service as build_service_with_token           # uses token_path consistently
import json, sqlalchemy
from datetime import datetime, timedelta
from sqlalchemy import func
# Load environment variables for the server
load_dotenv()

# Initialize the LLM instance here, inside the server.
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in the server's environment")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
mcp = FastMCP(name = "email_server",host = "0.0.0.0", port = 8080)

# --- GMAIL AUTHENTICATION (multi-account, shared offline tokens) ---
PRIMARY_EMAIL = os.getenv("GMAIL_ACCOUNT_PRIMARY", "").strip()
# Add SECONDARY alias from .env (optional but recommended)
SECONDARY_EMAIL = os.getenv("GMAIL_ACCOUNT_SECONDARY", "").strip()
# ---------- Imports ----------
import html, unicodedata, re, math, json
from typing import List, Tuple, Mapping, Any, Union, Optional
from datetime import datetime, timedelta

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

# ---------- De-quote / normalize ----------
RE_REPLY_SPLIT = re.compile(
    r"(^|\n)(on .*wrote:|sent from my|from: .*|----+ original message ----+|^\s*>+)",
    flags=re.I
)
def dequote(body: str, max_chars: int = 4000) -> str:
    body = html.unescape(unicodedata.normalize("NFKC", body or ""))
    parts = RE_REPLY_SPLIT.split(body)[0:1]
    body = parts[0] if parts else body
    body = re.sub(r"\n--\s*\n.*", "", body, flags=re.S)  # strip signatures
    return " ".join(body.replace("\r", " ").split())[:max_chars]

def head(body: str, n_chars: int = 800) -> str:
    return (dequote(body) or "")[:n_chars]

def make_doc(subject: str, sender: str, received_at_iso: str, head_text: str) -> str:
    return f"[SUBJECT] {subject or 'No Subject'} [SENDER] {sender or 'Unknown'} [DATE] {received_at_iso or ''} [BODY] {head_text or ''}"

# ---------- Token overlap & recency ----------
def _tok(s: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", (s or "").lower()) if t]

def _jacc(a: List[str], b: List[str]) -> float:
    if not a or not b: return 0.0
    A, B = set(a), set(b)
    inter = len(A & B)
    return 0.0 if inter == 0 else inter / float(len(A | B))

def _recency(dt: datetime) -> float:
    if not isinstance(dt, datetime): return 0.5
    days = (datetime.now() - dt).days
    return math.exp(-days / 14.0)

# ---------- Thread folding ----------
def fold_by_thread(scored: List[Tuple[float, dict]]) -> List[Tuple[float, dict]]:
    best = {}
    for s, r in scored:
        tid = r.get("threadId") or f"__single__:{r.get('email_id')}"
        cur = best.get(tid)
        if cur is None:
            best[tid] = (s, r)
        else:
            s0, r0 = cur
            r0_dt = r0.get("received_at") or datetime.min
            r_dt  = r.get("received_at")  or datetime.min
            if s > s0 or (abs(s - s0) < 1e-9 and r_dt > r0_dt):
                best[tid] = (s, r)
    return sorted(best.values(), key=lambda x: x[0], reverse=True)

# ---------- Confidence gate ----------
def gate(scores: List[float], tau: float, delta: float) -> bool:
    if not scores: return False
    s1 = scores[0]
    s2 = scores[1] if len(scores) > 1 else 0.0
    return (s1 >= tau) and ((s1 - s2) >= delta)

# ---------- Cross-encoder (optional) ----------
_reranker_model = None
def get_cross_encoder():
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model
    if CrossEncoder is None:
        return None
    try:
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        _reranker_model = None
    return _reranker_model

# ---------- Date parsing ----------
def _piso(s):
    try: return datetime.strptime(s.strip(), "%Y-%m-%d")
    except: return None

def _pfuzzy(s):
    if not s: return None
    t = re.sub(r"[,\s]+", " ", s.strip().lower()).replace("sept ", "sep ")
    for f in ("%Y-%m-%d","%d-%m-%Y","%d/%m/%Y","%d %b %Y","%d %B %Y","%b %d %Y","%B %d %Y"):
        try: return datetime.strptime(t, f)
        except: pass
    return None

# [REPLACE lines 127–135]
def _win(w):
    now = datetime.now(); w=(w or "").strip().lower()
    if w in ("today",):
        s=datetime(now.year,now.month,now.day); return s, s+timedelta(days=1)
    if w in ("yesterday",):
        e=datetime(now.year,now.month,now.day); return e-timedelta(days=1), e
    if w in ("last 7 days","past 7 days","7d","last_7d","last week"):
        return now-timedelta(days=7), now
    if w in ("last_30d","past 30 days"):
        return now-timedelta(days=30), now
    return None, None


# ---------- Decide when to try SQLite first ----------
def strong_lexical_signals(subject_like: str, sender_like: str, has_date: bool) -> bool:
    subj = (subject_like or "").strip()
    quoted = (subj.startswith('"') and subj.endswith('"')) or (subj.startswith("'") and subj.endswith("'"))
    longish = len(subj) >= 8
    return bool(quoted or longish or sender_like.strip() or has_date)

from functools import lru_cache
@lru_cache(maxsize=16)
def resolve_account(alias_or_email: str | None):
    """
    Resolve 'primary'/'secondary'/explicit email/substring to (account_id, account_email).

    Order:
      1) .env alias: 'primary' -> GMAIL_ACCOUNT_PRIMARY ; 'secondary' -> GMAIL_ACCOUNT_SECONDARY
      2) exact match in managed_accounts.account_email
      3) fuzzy contains match in managed_accounts.account_email
    Returns:
      (account_id: int, account_email: str)

    Raises:
      ValueError if alias missing in .env or account not found in DB.
    """
    key = (alias_or_email or "").strip().lower()

    # 1) alias mapping from .env
    if key in ("primary", ""):
        if not PRIMARY_EMAIL:
            raise ValueError("GMAIL_ACCOUNT_PRIMARY missing in .env")
        target_email = PRIMARY_EMAIL
    elif key == "secondary":
        if not SECONDARY_EMAIL:
            raise ValueError("GMAIL_ACCOUNT_SECONDARY missing in .env")
        target_email = SECONDARY_EMAIL
    else:
        # explicit email or fragment provided
        target_email = (alias_or_email or "").strip()
    
    # 2) validate against DB: exact first
    with engine.connect() as conn:
        row = conn.execute(
            sqlalchemy.select(managed_accounts_table.c.id, managed_accounts_table.c.account_email)
            .where(func.lower(managed_accounts_table.c.account_email) == target_email.lower())
        ).first()
        if row:
            return row.id, row.account_email

        # 3) fuzzy contains
        row = conn.execute(
            sqlalchemy.select(
                managed_accounts_table.c.id,
                managed_accounts_table.c.account_email
            ).where(managed_accounts_table.c.account_email.ilike(f"%{target_email}%"))
        ).first()

    if row:
        return row.id, row.account_email

    if key in ("primary", "secondary", ""):
        raise ValueError(
            f"Account not found for alias '{alias_or_email}'. "
            "Ensure .env alias points to an email present in managed_accounts."
        )
    raise ValueError(
        f"Account not found for '{alias_or_email}'. Use a full email or a substring that exists in managed_accounts."
    )

def _resolve_token_path_for(account_email: str) -> str:
    """Find token_path for an account from the offline DB; fallback to a filename convention."""
    with engine.connect() as conn:
        row = conn.execute(
            sqlalchemy.select(managed_accounts_table.c.token_path)
            .where(func.lower(managed_accounts_table.c.account_email) == account_email.lower())
        ).first()
    if row and row[0] and os.path.exists(row[0]):
        path = row[0]
        try:
            import logging
            logging.info(f"[TOKENS] Using token file from DB: {path} for {account_email}")
        except Exception:
            pass
        return path

    # Fallback: filename pattern like token_<email>.json (underscored)
    fallback = f"token_{account_email.replace('@','_').replace('.','_')}.json"
    if os.path.exists(fallback):
        try:
            import logging
            logging.info(f"[TOKENS] Using token file from fallback: {fallback} for {account_email}")
        except Exception:
            pass
        return fallback
    
    raise ValueError(f"No token file found for account_email='{account_email}'. Add this account via offline pipeline.")

def _pick_default_account_email() -> str:
    """Choose default account: env PRIMARY_EMAIL or first account in managed_accounts."""
    if PRIMARY_EMAIL:
        return PRIMARY_EMAIL.strip()
    with engine.connect() as conn:
        row = conn.execute(sqlalchemy.select(managed_accounts_table.c.account_email)).first()
    if not row:
        raise ValueError("No managed accounts in DB. Add one with your offline pipeline.")
    try:
        import logging
        logging.info(f"[ACCOUNTS] Defaulting to first managed account: {row[0]}")
    except Exception:
        pass
    return (row[0] or "").strip()


def get_gmail_service_for_account(account_email: str | None = None):
    """Build Gmail service bound to the correct token file (shares offline tokens)."""
    if account_email:
        _, real_email = resolve_account(account_email)  # maps alias/fragment -> (id, email)
    else:
         real_email = _pick_default_account_email()
    token_path = _resolve_token_path_for(real_email)
    return build_service_with_token(token_path)  # from gmail_service.py (offline) ✔

def create_message(recipient, subject, body):
    """Create a MIME Email message and base64 encode it as required by Gmail API."""
    # Make sure recipient, subject, and body are plain strings with no line breaks in headers
    subject = subject.replace('\n', ' ').replace('\r', ' ').strip()
    recipient = recipient.replace('\n', ' ').replace('\r', ' ').strip()
    message = MIMEText(body, _charset="utf-8")
    message['to'] = recipient
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': raw}


def _extract_body(payload):
    """Recursively extract the text/plain body from an email payload."""
    body = ""
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                data = part['body']['data']
                body += base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            elif 'parts' in part:
                body += _extract_body(part)
    elif payload.get('mimeType') == 'text/plain' and 'data' in payload.get('body', {}):
        data = payload['body']['data']
        body = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
    return " ".join(body.replace("\r\n", " ").split())

@mcp.tool()
def search_emails(query: str, max_results: int = 5, account_email: str = "") -> str:
    """
    DB-only search wrapper: calls rag_search_emails with 'semantic_query=query'.
    Returns a neat, readable string (so your client remains unchanged).
    """
    rag_json = rag_search_emails(
        semantic_query=query,
        limit=max_results,
        account_email=account_email or ""
    )
    try:
        parsed = json.loads(rag_json or "{}")
    except Exception:
        return "Search failed (malformed response)."

    status = parsed.get("status")

    # Map server statuses to a consistent items list + header
    if status == "one":
        items = [parsed.get("item") or {}]
        header = "Found 1 result in your local DB:\n"
    elif status == "ok":
        items = parsed.get("items", [])
        has_more = bool(parsed.get("has_more"))
        extra = " (+ more not shown)" if has_more and max_results else ""
        header = f"Found {len(items)} result(s) in your local DB{extra}:\n"
    elif status == "empty":
        return f"No emails found in the local DB for: '{query}'."
    elif status == "need_refine":
        msg = parsed.get("message") or "No high-confidence single match. Please add a sender or a date."
        return msg
    else:
        # error fallback
        return f"Search error: {parsed.get('message', 'unknown error')}"

    if not items:
        return f"No emails found in the local DB for: '{query}'."

    lines = [header]
    for i, it in enumerate(items, 1):
        sender = it.get("sender") or "Unknown"
        subject = it.get("subject") or "No Subject"
        date = it.get("received_at") or "Unknown date"
        prio = it.get("priority") or "-"
        snip = it.get("snippet") or ""
        lines.append(
            f"{i}. From: {sender}\n"
            f"   Subject: {subject}\n"
            f"   Date: {date} | Priority: {prio}\n"
            f"   Snippet: {snip}\n"
        )
    return "\n".join(lines)

from typing import Union, Mapping, Any
import re, math, sqlalchemy, json
from datetime import datetime, timedelta

# BEFORE
# def summarize_email(account_email: str = "", email_id: Optional[int] = None, thread_id: Optional[str] = None) -> str:

# AFTER
@mcp.tool()
def summarize_email(
    account_email: str = "",
    email_id: Optional[Union[int, str]] = None,
    thread_id: Optional[str] = None,
) -> str:
    # ---- Validate account ----
    if not account_email.strip():
        return "Error: missing account_email. Call rag_search_emails first, then pass account_email + email_id here."
    try:
        account_id, _ = resolve_account(account_email)
    except Exception:
        return "Error: unknown account_email."

    # ---- Normalize identifiers ----
    email_id_int: Optional[int] = None
    email_id_str: Optional[str] = None

    if email_id is not None:
        if isinstance(email_id, int):
            email_id_int = email_id
        elif isinstance(email_id, str):
            eid = email_id.strip()
            if eid.isdigit():
                email_id_int = int(eid)
            else:
                # Non-numeric: treat as external id / thread candidate
                email_id_str = eid

    if email_id_int is None and not thread_id and email_id_str:
        # If a non-numeric "email_id" arrived, try it as threadId first.
        thread_id = email_id_str

    if email_id_int is None and not thread_id:
        return ("No target specified. Provide email_id (int) from rag_search_emails, "
                "or a thread_id (string).")

    # ---- Fetch rows ----
    with engine.connect() as conn:
        rows = []
        if email_id_int is not None:
            r = conn.execute(
                sqlalchemy.select(
                    emails_table.c.id, emails_table.c.account_id, emails_table.c.sender,
                    emails_table.c.subject, emails_table.c.summary, emails_table.c.full_text,
                    emails_table.c.priority, emails_table.c.received_at, emails_table.c.threadId
                ).where(
                    (emails_table.c.account_id == account_id) &
                    (emails_table.c.id == email_id_int)
                ).limit(1)
            ).first()
            if r:
                rows = [r]
        if not rows and thread_id:
            rows = conn.execute(
                sqlalchemy.select(
                    emails_table.c.id, emails_table.c.account_id, emails_table.c.sender,
                    emails_table.c.subject, emails_table.c.summary, emails_table.c.full_text,
                    emails_table.c.priority, emails_table.c.received_at, emails_table.c.threadId
                ).where(
                    (emails_table.c.account_id == account_id) &
                    (emails_table.c.threadId == thread_id)
                ).order_by(emails_table.c.received_at.desc()).limit(50)
            ).fetchall()

        # OPTIONAL: if your schema has a gmail message id column, try it as well
        if not rows and email_id_str and hasattr(emails_table.c, "gmail_id"):
            r = conn.execute(
                sqlalchemy.select(
                    emails_table.c.id, emails_table.c.account_id, emails_table.c.sender,
                    emails_table.c.subject, emails_table.c.summary, emails_table.c.full_text,
                    emails_table.c.priority, emails_table.c.received_at, emails_table.c.threadId
                ).where(
                    (emails_table.c.account_id == account_id) &
                    (emails_table.c.gmail_id == email_id_str)
                ).limit(1)
            ).first()
            if r:
                rows = [r]

    if not rows:
        return "No emails found for the given identifier(s)."

    # ---- Choose representative in thread (unchanged) ----
    chosen = None
    for r in rows:
        if (r.summary and r.summary.strip()) or (r.full_text and dequote(r.full_text).strip()):
            chosen = r
            break
    if chosen is None:
        chosen = rows[0]

    sender_str   = chosen.sender or "Unknown"
    subject_str  = chosen.subject or "No Subject"
    received_str = chosen.received_at.isoformat(sep=" ", timespec="minutes") if isinstance(chosen.received_at, datetime) else "Unknown date"

    if chosen.summary and chosen.summary.strip():
        return (f"Summary of email from {sender_str} "
                f"(Subject: {subject_str}, Date: {received_str}):\n\n{chosen.summary.strip()}")

    body_clean = dequote(chosen.full_text or "")
    if not body_clean:
        return f"Email from {sender_str} (Subject: {subject_str}) has no stored body to summarize."

    prompt = (
        "Provide a concise, actionable summary of this email. "
        "Include purpose, key facts, requested actions, owners, dates/deadlines, and any attachments.\n\n"
        f"Email Content (truncated if long):\n'''{body_clean[:6000]}'''"
    )
    try:
        gen = llm.invoke(prompt).content.strip()
    except Exception as e:
        return f"LLM summarization error: {e}"

    return (f"Summary of email from {sender_str} "
            f"(Subject: {subject_str}, Date: {received_str}):\n\n{gen}")




@mcp.tool()
def draft_email(recipient: str, subject: str = "", body: str = "", account_email: str = "") -> str:
    """Creates a Gmail draft in the specified account and returns its ID."""
    if not recipient or "@" not in recipient:
        return "Error: A valid recipient email address is required."
    subject = subject or "Quick Message"
    body = body or "Hi,\n\nJust following up.\n\nBest,"
    try:
        service = get_gmail_service_for_account(account_email)
        message = create_message(recipient, subject, body)
        draft = service.users().drafts().create(userId='me', body={'message': message}).execute()
        return draft['id']
    except Exception as e:
        return f"Failed to create draft: {e}"


@mcp.tool()
def send_email(draft_id: str, account_email: str = "") -> str:
    """Sends a draft email using its draft ID in the specified account."""
    if not draft_id:
        return "Error: A valid draft_id is required."
    try:
        service = get_gmail_service_for_account(account_email)
        service.users().drafts().send(userId='me', body={'id': draft_id}).execute()
        return "Successfully sent the email that was in your drafts."
    except Exception as e:
        return f"Failed to send email: {e}"


# --- RAG: unified SQLite + Chroma search for best/top-5 emails ---
import json, sqlalchemy
from datetime import datetime, timedelta

# Pull your local store & vector store (provided by your offline pipeline)
# This module must define: engine, emails_table (SQLAlchemy Table), email_collection (Chroma collection)
from database import engine, emails_table, email_collection  # <-- make sure this is importable

def _parse_date(s: str | None):
    if not s: return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try: return datetime.strptime(s, fmt)
        except: pass
    return None

def _resolve_window(keyword: str | None):
    if not keyword: return (None, None)
    now = datetime.now()
    kw = keyword.lower().strip()
    if kw == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return (start, now)
    if kw in {"yesterday"}:
        y = now - timedelta(days=1)
        return (y.replace(hour=0, minute=0, second=0, microsecond=0),
                y.replace(hour=23, minute=59, second=59, microsecond=999999))
    if kw in {"last_week", "past_week", "last 7 days"}:
        return (now - timedelta(days=7), now)
    if kw in {"last_month", "past 30 days"}:
        return (now - timedelta(days=30), now)
    return (None, None)

def _age_score(dt):
    if not dt: return 0.0
    age_days = max(0.0, (datetime.now() - dt).total_seconds()/86400.0)
    return 1.0 / (1.0 + age_days)


@mcp.tool()
def rag_search_emails(
    sender_like: str = "",
    subject_like: str = "",
    priority_in: str = "",
    window: str = "",
    date_from: str = "",
    date_to: str = "",
    semantic_query: str = "",
    topic_hint: str = "",
    limit: int = 5,
    account_email: str = "",
    strictness: str = "normal"
) -> str:
    """
    Server-side retrieval (no NL parsing):
      - If ANY filter (sender/subject/window/date/priority) is provided -> SQLite only.
      - Else if semantic_query is provided -> Chroma (semantic) + rerank.
      - Else -> empty.
      - Single vs list is controlled only by `limit` (limit==1 -> status:'one'; else list).
    """
    try:
        # ---- Resolve account ----
        if not account_email.strip():
            return json.dumps({"status": "error", "message": "account_email is required"})
        try:
            account_id, _ = resolve_account(account_email)
        except Exception:
            return json.dumps({"status": "error", "message": "unknown account_email"})

        # ---- Time bounds ----
        start_dt = end_dt = None
        if window.strip():
            s, e = _win(window); start_dt, end_dt = s or start_dt, e or end_dt
        if date_from.strip():
            s = _piso(date_from) or _pfuzzy(date_from); start_dt = s or start_dt
        if date_to.strip():
            e = _piso(date_to) or _pfuzzy(date_to);   end_dt   = (e + timedelta(days=1)) if e else end_dt
        has_date = bool(start_dt or end_dt)

        look_single = (limit == 1)
        filters_exist = any([
            bool(sender_like.strip()),
            bool(subject_like.strip()),
            bool(priority_in.strip()),
            has_date,
        ])

        # =====================================================================
        # FAST PATH: no filters + single item + trivial query => newest via SQLite
        # (prevents unnecessary Chroma/embedding work for "latest/newest/recent mail" asks)
        # =====================================================================
        qtext = (topic_hint or semantic_query or "").strip()

        def _looks_trivial(q: str) -> bool:
            # trivial if empty or very short and composed of uninformative tokens
            toks = [t for t in re.split(r"[^A-Za-z0-9]+", (q or "").lower()) if t]
            junk = {"latest", "newest", "recent", "most", "mail", "email", "please", "show", "the", "get", "me"}
            return (not toks) or (len(toks) <= 3 and all(t in junk for t in toks))

        if not filters_exist and look_single and _looks_trivial(qtext):
            with engine.connect() as conn:
                row = conn.execute(
                    sqlalchemy.select(
                        emails_table.c.id, emails_table.c.threadId, emails_table.c.sender,
                        emails_table.c.subject, emails_table.c.summary, emails_table.c.full_text,
                        emails_table.c.priority, emails_table.c.action_needed, emails_table.c.received_at
                    ).where(emails_table.c.account_id == account_id)
                     .order_by(emails_table.c.received_at.desc())
                     .limit(1)
                ).first()
            if not row:
                return json.dumps({"status": "empty", "items": []})
            item = {
                "email_id": row.id,
                "thread_id": row.threadId,
                "sender": row.sender or "",
                "subject": row.subject or "",
                "summary": row.summary or "",
                "priority": row.priority or "",
                "action_needed": bool(row.action_needed) if row.action_needed is not None else None,
                "received_at": row.received_at.isoformat() if row.received_at else None,
                "snippet": (row.summary or head(row.full_text or "") or "")[:220],
                "score": 1.0  # deterministic newest
            }
            return json.dumps({"status": "one", "item": item})

        # =====================================================================
        # A) SQLITE PATH (filters exist) → precise & fast
        # =====================================================================
        if filters_exist:
            where = [emails_table.c.account_id == account_id]
            if sender_like.strip():
                where.append(emails_table.c.sender.ilike(f"%{sender_like.strip()}%"))
            if subject_like.strip():
                where.append(emails_table.c.subject.ilike(f"%{subject_like.strip()}%"))
            if start_dt:
                where.append(emails_table.c.received_at >= start_dt)
            if end_dt:
                where.append(emails_table.c.received_at < end_dt)
            if priority_in.strip():
                where.append(emails_table.c.priority.ilike(f"%{priority_in.strip()}%"))

            take = max(1, min(int(limit if not look_single else 1), 500))
            with engine.connect() as conn:
                sel = (sqlalchemy.select(
                    emails_table.c.id, emails_table.c.threadId, emails_table.c.sender,
                    emails_table.c.subject, emails_table.c.summary, emails_table.c.full_text,
                    emails_table.c.priority, emails_table.c.action_needed, emails_table.c.received_at
                ).where(*where).order_by(emails_table.c.received_at.desc()).limit(take))
                rows = conn.execute(sel).fetchall()

            # Optional lightweight lexical score (stable ordering already by recency)
            qtok = _tok(subject_like or topic_hint or semantic_query)
            scored = []
            for r in rows:
                subj = r.subject or ""
                exact = float((subject_like or "").strip().lower() == subj.strip().lower()) if subject_like else 0.0
                ovlp  = _jacc(qtok, _tok(subj)) if qtok else 0.0
                rec   = _recency(r.received_at)
                score = 0.65*exact + 0.25*ovlp + 0.10*rec
                scored.append((score, {
                    "email_id": r.id, "thread_id": r.threadId, "sender": r.sender or "",
                    "subject": subj, "summary": r.summary or "",
                    "priority": r.priority or "",
                    "action_needed": bool(r.action_needed) if r.action_needed is not None else None,
                    "received_at": r.received_at,
                    "snippet": (r.summary or head(r.full_text or ""))[:220]
                }))
            scored.sort(key=lambda x: x[0], reverse=True)

            if look_single:
                if not scored:
                    return json.dumps({"status": "empty", "items": []})
                s, d = scored[0]
                d["score"] = float(f"{s:.4f}")
                d["received_at"] = d["received_at"].isoformat() if d["received_at"] else None
                return json.dumps({"status": "one", "item": d})

            items = []
            for s, d in scored[:limit]:
                d["score"] = float(f"{s:.4f}")
                d["received_at"] = d["received_at"].isoformat() if d["received_at"] else None
                items.append(d)
            return json.dumps({"status": "ok", "items": items})

        # =====================================================================
        # B) CHROMA PATH (no filters, fuzzy topic)
        # =====================================================================
        if not qtext:
            return json.dumps({"status": "empty", "items": []})

        chroma_where = {"account_id": {"$eq": account_id}}
        q = email_collection.query(
            query_texts=[qtext],
            n_results=100,
            where=chroma_where,
            include=["metadatas"]
        )
        metas = (q.get("metadatas") or [[]])[0]
        if not metas:
            return json.dumps({"status": "empty", "items": []})

        keys = []
        for m in metas:
            aid = m.get("account_id"); eid = m.get("email_id")
            if aid is None or eid is None: continue
            keys.append((aid, eid))

        with engine.connect() as conn:
            rows = conn.execute(
                sqlalchemy.select(
                    emails_table.c.id, emails_table.c.threadId, emails_table.c.sender,
                    emails_table.c.subject, emails_table.c.summary, emails_table.c.full_text,
                    emails_table.c.priority, emails_table.c.action_needed, emails_table.c.received_at
                ).where(sqlalchemy.or_(*[
                    (emails_table.c.account_id == a) & (emails_table.c.id == e) for (a, e) in keys[:800]
                ]))
            ).fetchall()

        docs, pairs = [], []
        for r in rows[:60]:
            row = {
                "email_id": r.id, "thread_id": r.threadId, "sender": r.sender or "",
                "subject": r.subject or "", "summary": r.summary or "", "full_text": r.full_text or "",
                "priority": r.priority or "", "action_needed": bool(r.action_needed) if r.action_needed is not None else None,
                "received_at": r.received_at
            }
            docs.append(row)
            pairs.append((qtext, make_doc(
                row["subject"], row["sender"],
                (row["received_at"].isoformat() if row["received_at"] else ""),
                head(row.get("full_text",""))
            )))
        def _batched_predict(model, pairs, batch_size=64):
            out = []
            for i in range(0, len(pairs), batch_size):
                out.extend(list(model.predict(pairs[i:i+batch_size])))
            return out
    
        reranker = get_cross_encoder()
        if reranker and pairs:
            try:
                scores = _batched_predict(reranker, pairs, batch_size=64)
            except Exception:
                scores = [0.0]*len(pairs)
        else:
            scores = [0.35 + 0.65*_recency(d["received_at"]) for d in docs]

        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        folded = fold_by_thread(scored)

        if look_single:
            level = (strictness or "normal").lower()
            tau   = 0.55 if level == "normal" else (0.62 if level == "high" else 0.48)
            delta = 0.08
            if not gate([s for s,_ in folded[:2]], tau, delta):
                return json.dumps({"status":"need_refine","message":"No high-confidence single match. Add a sender or a date."})
            s, d = folded[0]
            d["score"] = float(f"{s:.4f}")
            d["received_at"] = d["received_at"].isoformat() if d["received_at"] else None
            d["snippet"] = (d.get("summary") or head(d.get("full_text","")) or "")[:220]
            d.pop("full_text", None)
            return json.dumps({"status":"one","item": d})

        items = []
        for s, d in folded[:limit]:
            d["score"] = float(f"{s:.4f}")
            d["received_at"] = d["received_at"].isoformat() if d["received_at"] else None
            d["snippet"] = (d.get("summary") or head(d.get("full_text","")) or "")[:220]
            d.pop("full_text", None)
            items.append(d)
        return json.dumps({"status":"ok","items": items})

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
    # Warm-up models on import so first query is fast
try:
    _ = get_cross_encoder()
except Exception:
    pass

if __name__ == "__main__":
    mcp.run(transport = "stdio")



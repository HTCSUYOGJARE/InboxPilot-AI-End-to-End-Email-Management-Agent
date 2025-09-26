# InboxPilot AI — End‑to‑End Email Management Agent

Production‑ready, local‑first email assistant that ingests Gmail from multiple accounts, enriches messages with structured metadata, indexes content for fast semantic + lexical retrieval, and exposes an interactive agent that can search, summarize, draft, and send.

## Highlights
- **Multi‑account ingestion** on a schedule with **APScheduler** and parallel workers.
- **Structured extraction** with Outlines + Pydantic and **attachment summarization** via PyMuPDF + a local LLM (Ollama).
- **Hybrid retrieval**: precise SQL filters (sender, date, priority) + **ChromaDB** vector search with optional cross‑encoder re‑ranking.
- **Thread collapsing** and **confidence gating** to avoid near‑duplicate hits and low‑confidence answers.
- **Agentic UX**: LangGraph‑orchestrated CLI agent connecting to a FastMCP server with tools for search, summarize, draft, and send.

## Architecture

```
                ┌──────────────────────────────────────┐
                │           Offline Pipeline           │
                │  run_ingestion.py (scheduler)        │
                │  ├─ fetch unread mail (Gmail API)    │
                │  ├─ gatekeeper: drop promos/noise    │
                │  ├─ LLM extraction (Outlines)        │
                │  ├─ PDF summarization (PyMuPDF+LLM)  │
                │  └─ persist: SQLite + ChromaDB       │
                └──────────────────────────────────────┘
                                 │
                                 ▼
                ┌──────────────────────────────────────┐
                │            Online Agent              │
                │  email_server.py (FastMCP tools)     │
                │  ├─ hybrid retrieval (SQL + vector)  │
                │  ├─ re‑ranking, thread folding       │
                │  ├─ Gemini summaries (fallback)      │
                │  └─ draft/send Gmail actions         │
                │                                      │
                │  online_main.py (LangGraph CLI)      │
                │   └─ classify → search → summarize   │
                │               → draft → send         │
                └──────────────────────────────────────┘
```

## Repository Layout

```
.
├── add_account.py        # One‑time OAuth + register account in SQLite
├── client.py             # Minimal MCP client used by the agent
├── database.py           # SQLite schema + ChromaDB setup + SentenceTransformer
├── database_cleanup.py   # Periodic cleanup from SQLite and ChromaDB
├── email_server.py       # FastMCP server: tools for search/summarize/draft/send
├── gatekeeper.py         # Heuristics to filter low‑value promotional mail
├── gmail_service.py      # Gmail auth + API helpers (fetch, attachments, mark read)
├── ingestion_worker.py   # Per‑email processing: extraction + attachment summary
├── run_ingestion.py      # APScheduler entrypoint for multi‑account ingestion
├── online_main.py        # LangGraph agent CLI connected to the FastMCP server
├── requirements.txt      # Python dependencies
└── db/                   # Created on first run: SQLite + Chroma persistence
```

## Getting Started

### Prerequisites
- Python 3.11+
- Google Cloud project + Gmail API enabled; `credentials.json` in project root
- GPU with CUDA drivers (recommended) for SentenceTransformers
- [Ollama](https://ollama.com) installed with a local model (default: `phi3`)
- A Gemini API key for online summarization fallback

### Installation
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the repository root:
```env
# Server/Agent
GEMINI_API_KEY=AI...

# Optional aliases used by the online agent
GMAIL_ACCOUNT_PRIMARY=you@example.com
GMAIL_ACCOUNT_SECONDARY=team@example.com

# Local LLM for attachment & extraction
OLLAMA_MODEL=phi3
```

### Authorize Gmail Accounts
Run the helper once per account you want the pipeline to manage. It creates a token JSON and records the account in SQLite.
```bash
python add_account.py
```

## Running

### 1) Offline Ingestion
This pulls unread mail every 2 minutes, filters low‑value items, performs extraction and attachment summarization, and persists to the stores. A cleanup job prunes old emails daily.
```bash
python run_ingestion.py
```

### 2) Online Agent
Start the tool server in one terminal:
```bash
python email_server.py
```
Start the LangGraph CLI agent in another:
```bash
python online_main.py
```

## Usage Examples
- Find by sender and time: `find invoices from:accounts@vendor.com last 30d`
- Semantic search: `look for onboarding instructions for SSO`
- Summarize: `summarize the top 3 emails about Q3 roadmap`
- Draft: `draft an email to jane@company.com about rescheduling our 1:1 next week`
- Send after confirming: `yes`

## Data & Indexing

- **SQLite** (`./db/emails.db`): canonical record of messages and metadata (priority, action_needed, deadline).
- **ChromaDB** (`./db/chroma_db`): persistent vector store with HNSW, populated with chunked subject+body text using a SentenceTransformer model.
- **Chunking**: conservative fixed‑word windows to balance recall and precision.

## Retrieval & Ranking

1. **Lexical filters first** when the user specifies precise fields (sender, subject, dates).
2. **Vector search** over ChromaDB for semantic queries or when lexical filters are insufficient.
3. **Re‑ranking (optional)** using a cross‑encoder for the top candidates.
4. **Thread folding** to collapse near‑duplicates to the most recent, most relevant message.
5. **Confidence gating** to decide when to return a single definitive hit vs. a small list.

## File‑by‑File Details

- `run_ingestion.py`: APScheduler with a 2‑minute interval; per‑account fetch; gatekeeper; parallel workers; batch mark‑as‑read.
- `ingestion_worker.py`: Outlines + Pydantic schema for structured JSON; PyMuPDF for PDFs; Ollama for local LLM summarization; safe fallbacks.
- `database.py`: SQLite schema for accounts, emails, people, and contact emails; ChromaDB persistent collection; SentenceTransformer `all‑MiniLM‑L6‑v2` on GPU.
- `email_server.py`: FastMCP server with tools for search, summarize, draft, send; hybrid retrieval implementation and optional cross‑encoder re‑ranker.
- `online_main.py`: LangGraph state graph with nodes for classify → search → summarize → draft → send and an interactive CLI loop.
- `gatekeeper.py`: sender/subject heuristics and unsubscribe detection to drop promos.
- `gmail_service.py`: OAuth token management, unread fetch, and batch update helpers.
- `database_cleanup.py`: daily retention policy for SQLite and vector chunks.

## Operational Notes
- The online agent operates purely on the local stores; ingestion must be running to keep the DB up to date.
- Attachment summarization currently supports PDF up to a configurable size. Extend `ALLOWED_MIMETYPES` to add more types.
- The server drafts emails by default and asks for explicit confirmation before sending.

## Roadmap Ideas
- Expand attachment parsing to DOCX, MSG, and images (OCR).
- Add label/priority auto‑tuning via feedback loops.
- Multi‑tenant isolation and per‑user auth for the online agent.
- Web UI (React) on top of the same MCP tools.

## License
Add your chosen license here (e.g., MIT).

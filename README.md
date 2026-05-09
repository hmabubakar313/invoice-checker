# dj_mcp_rag

A Django + DRF prototype that ingests invoice PDFs, runs them through a RAG-aware
LangGraph agent, and dispatches to MCP-style tools (`approve_invoice` /
`notify_manager`) based on a hybrid rule + LLM decision.

## Stack

- **Django 4.2 / DRF** — HTTP layer (upload + Q&A endpoints)
- **ChromaDB** (persistent, `./chroma_db`) — vector store for invoice chunks
- **SentenceTransformers** (`all-MiniLM-L6-v2`) — embeddings
- **Ollama** (`llama3.2` on `localhost:11434`) — extraction + decision LLM
- **LangGraph** — agent state machine, with **SqliteSaver** checkpointer (`./langgraph_checkpoints.sqlite`) for cross-process persistence
- **PyPDF2** — PDF text extraction

## Endpoints

| Method | Path        | Body                   | Purpose                                  |
|--------|-------------|------------------------|------------------------------------------|
| POST   | `/upload/`  | `title`, `file` (PDF)  | Ingest, embed, run agent, dispatch tool  |
| POST   | `/ask/`     | `{"query": "..."}`     | Free-form RAG Q&A over ingested docs     |

## Agent flow

```
START → extract → rules
                  ├─ missing  → END
                  └─ continue → retrieve → decision
                                            ├─ done  → END
                                            └─ retry → retry
                                                       ├─ cap_hit  → END
                                                       └─ continue → retrieve
```

- **extract** — LLM pulls `{amount, vendor, date}` from raw text (returns `null` on missing).
- **rules** — deterministic gate: any required field missing → flag, skip the LLM.
- **retrieve** — Chroma top-k against the invoice text, excluding self-chunks.
- **decision** — LLM picks `approve_invoice` vs `notify_manager` using extracted fields + retrieved context.
- **retry** — on `"not sure"` reasons, re-retrieve with larger `k`. Cap of 2.

## Layout

```
core/
  models.py              Document, Chunk, AgentRun (audit log)
  serializers.py         DocumentSerializer (rejects non-PDF uploads)
  views.py               upload_document, ask_question
  services/
    extract.py           Structured-field extraction (LLM)
    ingest.py            PDF → chunks → Chroma
    retrieve.py          Chroma similarity query
    rag.py               Q&A pipeline (used by /ask/)
    state.py             InvoiceState (TypedDict) + node functions
    orchestrator.py      LangGraph build, checkpointer, run_agent
    tool.py              MCP-style tool stubs
    embed.py             SentenceTransformer wrapper
    generate.py          Generic Ollama prompt helper
  utils/
    chunking.py          Fixed-window text splitter
    file_loader.py       PyPDF2 wrapper
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r req.txt

# Ollama needs to be running with llama3.2 pulled:
ollama serve
ollama pull llama3.2

python manage.py migrate
python manage.py runserver
```

## Inspecting agent state

The LangGraph SQLite checkpointer persists every node transition. After an
upload, in a shell:

```python
from core.services.orchestrator import graph

cfg = {"configurable": {"thread_id": f"invoice-{document_id}"}}
snap = graph.get_state(cfg)
print(snap.values["decision"])
print([t["node"] for t in snap.values["trace"]])

for s in graph.get_state_history(cfg):
    print(s.metadata.get("step"), s.next)
```

`AgentRun` rows in the Django DB hold the final state for ORM-side queries;
the LangGraph checkpointer holds per-step history for resume / time-travel.

## Known sharp edges

- `decide_and_act` (the pre-LangGraph procedural agent) and `run_agent` are both
  invoked on `/upload/`. Until that's collapsed, tools fire twice per upload.
- `Document.id` lookup squiggles in the IDE are DRF type-stub limitations,
  not runtime bugs. Install `django-stubs` + `djangorestframework-stubs` to
  silence them.

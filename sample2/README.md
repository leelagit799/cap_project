# DischargeFlow — Agentic AI Hospital Discharge System

End-to-end implementation of the **FA5 SP Interns Capstone: AI Discharge Summaries** specification for St. Marian Regional Medical Center. The system ingests multilingual discharge packets (PDF, DOCX, TXT, JSON, scanned images), extracts and normalises clinical data, validates against a Mock EHR and versioned YAML rules, scores discharge risk, generates audit reports, supports human-in-the-loop corrections, streams patient-friendly summaries, and answers grounded clinical questions via RAG.

## Quick start

Python 3.11+ is required. The `.venv` folder is created **on your machine only** — it is never committed to GitHub (Windows uses `.venv/Scripts/`, Linux/Mac uses `.venv/bin/`).

### Windows (VS Code / PowerShell)

```powershell
cd sample2
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
python run.py
```

### Linux / Mac

```bash
cd sample2
bash scripts/setup.sh
source .venv/bin/activate
python run.py
```

### Manual setup (any OS)

```bash
cd sample2
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env    # Windows: copy .env.example .env
python run.py
```

Open these services:

| Service | URL | Purpose |
| --- | --- | --- |
| Host Orchestrator (Gradio) | http://localhost:8083 | Run workflows, stream summaries, RAG, agent health |
| HITL Dashboard (Streamlit) | http://localhost:8501 | Six-page clinical workspace incl. Patient Upload |
| Patient Upload API | http://localhost:8060/docs | Dynamic document ingestion REST API |
| Mock EHR | http://localhost:8050/docs | Patients, medications, allergies, labs, care plans |
| Primary MCP | http://localhost:8200/clinicaltools | Tools, resources, prompts, sampling, elicitation, roots |
| Analytics MCP | http://localhost:8201/analyticstools | Risk scores, benchmarks, heatmap |

`Ctrl+C` stops all twelve processes. To start only infrastructure or agents:

```bash
python run.py --only ehr mcp    # EHR + both MCP servers
python run.py --only agents     # six A2A agents
python run.py --only ui           # Gradio + Streamlit
python run.py --list              # print the full service table
```

## Credentials

Copy `.env.example` to `.env` and provide:

| Variable | Purpose |
| --- | --- |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Bedrock Nova Lite (primary) and Command R+ (fallback) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | End-to-end observability |
| `AGENT_AUTH_TOKEN` | Shared-secret A2A authentication (`X-Agent-Auth-Token`) |

Set `LLM_OFFLINE=1` to run with a deterministic offline LLM stub (no network). Never commit `.env`.

## Architecture

```
Data/incoming/  →  Monitor (:8103, ADK)
                →  Extractor (:8100, LangGraph)  →  Normalizer (:8102, LangGraph + MCP Sampling)
                →  Validator (:8101, LangGraph + MCP Elicitation)  →  Reporter (MCP tool)
                →  RAG Indexer (:8105, Agno)  →  HITL gate  →  Summary (:8104, ADK streaming)

Primary MCP (:8200)  — 6 tools, 6 resources, 5 prompts, sampling, elicitation, roots
Analytics MCP (:8201) — risk score, population benchmark, heatmap
Mock EHR (:8050) — REST API over 5 exported JSON files
```

**Frameworks (doc Table 6):** LangGraph, Google ADK, Agno, FastAPI, FastMCP, LiteLLM, FAISS, SQLite, Streamlit, Gradio, LangFuse.

**Agents (A2A ports 8100–8105):**

| Agent | Port | Framework | Streaming |
| --- | --- | --- | --- |
| Clinical Extractor | 8100 | LangGraph | No |
| Clinical Validation | 8101 | LangGraph | No |
| Clinical Normalizer | 8102 | LangGraph | No |
| Discharge Monitor | 8103 | Google ADK | No |
| Summary Generator | 8104 | Google ADK | Yes |
| Clinical RAG Q&A | 8105 | Agno | Yes |

## HITL Dashboard (Streamlit :8501)

Five pages exactly as specified in Table 13:

1. **Document Viewer** — patient selector, tabbed documents, language badge, process trigger
2. **Validation Report** — completeness, findings, risk badge, blocked indicator, audit exports
3. **HITL Corrections** — medication editor, elicitation form, approval decision, re-run validation
4. **RAG Q&A** — patient filter, example queries, prompt-injection indicator, RAG Triad metrics
5. **Discharge Summary** — patient-friendly sections, prescriptions, colour-coded labs, exports

## Sample patients

Documents ship for **P1019–P1024**. Expected outcomes:

| Patient | Outcome |
| --- | --- |
| P1019, P1020, P1023 | Auto-approve (Low risk) |
| P1022, P1024 | Hard-block (Penicillin vs Amoxicillin allergy) |
| P1021 | Escalate (unpaid bill, missing address, missing follow-up) |

## Project layout

```
sample2/
├── run.py                          # Supervisor — starts all 11 services
├── configs/                        # agent_config.yaml, prompts.yaml, rules.yaml
├── Data/incoming/                  # Source discharge packets (read-only)
├── hospital_ai/
│   ├── agents/                     # LangGraph, ADK, Agno agents + serve.py
│   ├── a2a/                        # Agent cards, auth, client, server
│   ├── mcp_servers/                # Primary (:8200) + Analytics (:8201)
│   ├── ehr/                        # Mock EHR FastAPI app
│   ├── rag/                        # FAISS store + five RAG roles
│   ├── guardrails/                 # PII, injection, hallucination, toxicity, HITL
│   ├── observability/              # LangFuse + JSONL fallback
│   ├── storage/                    # SQLite case store
│   └── ui/                         # Streamlit HITL + Gradio host + service bridge
├── mock_ehr/                       # data.py source + JSON export
├── tests/                          # unit, integration, e2e (live Bedrock optional)
└── docs/00_ARCHITECTURE_PLAN.md    # Full architecture and phase plan
```

## Generated artifacts

| Path | Contents |
| --- | --- |
| `data/reports/<case_id>/` | Audit JSON, HTML, PDF per case |
| `data/reports/traces.jsonl` | LangFuse fallback trace sink |
| `data/reports/pipeline.log` | Structured JSON-line application log |
| `data/vector_db/` | FAISS indexes per patient case |
| `data/state/dischargeflow.sqlite` | Cases, validations, HITL reviews, summaries |

Clinical source documents in `Data/incoming/` are never overwritten.

## Tests

```bash
# Full suite (offline LLM stub — no credentials needed)
pytest tests/ -q

# Live Bedrock + LangFuse (requires .env credentials)
pytest tests/e2e/test_rag_live.py -q
```

224+ tests cover MCP primitives, A2A auth/streaming, LangGraph pipeline outcomes, orchestrator workflow, RAG triad, guardrails, and UI service bridge.

## Developer notes

- **MCP package name** is `mcp_servers/` (not `mcp/`) to avoid shadowing the installed `mcp` SDK.
- **Elicitation** is interactive only in the Streamlit dashboard; headless agent processes decline by default (safe HITL escalation).
- **Sampling** routes through LiteLLM to Bedrock Nova Lite with Command R+ fallback; the Normalizer supplies the client-side sampling callback.
- **One trace ID per case** is created by the Host Orchestrator and propagated through LangFuse spans.

See `docs/00_ARCHITECTURE_PLAN.md` for the complete specification mapping, database schema, and implementation phase history.

# DischargeFlow — Architecture & Implementation Plan

**Source of truth:** `FA5_SP_Interns_Capstone_AI_Discharge_Summaries.docx` (read in full: sections 1–10, Tables 1–15, Figure 1).
**Status:** APPROVED. Phases P0–P12 are implemented, tested and committed.

## Build status

| Phase | Scope | State |
|---|---|---|
| P0 | Foundation: config, contracts, logging, retry, errors | Done |
| P1 | Mock EHR REST API :8050 | Done |
| P2 | Primary MCP :8200 — all six primitives | Done |
| P3 | Secondary MCP :8201 + multi-server client | Done |
| P4 | A2A layer — cards, auth, streaming + non-streaming | Done |
| P5 | LangGraph Extractor :8100 | Done |
| P6 | LangGraph Normalizer :8102 + MCP Sampling | Done |
| P7 | LangGraph Validator :8101 + elicitation + risk | Done |
| P8 | Reporter Tool — JSON/HTML/PDF audit artifacts | Done |
| P9 | Agno RAG :8105 — five roles, FAISS, RAG Triad | Done |
| P10 | ADK Monitor/Summary/Host + LangFuse observability | Done |
| P11 | Streamlit HITL :8501 — five pages + elicitation callback | Done |
| P12 | `run.py` supervisor, tests, README, hardening | Done |

229 tests pass (unit, integration and live end-to-end). Credentials are verified working:
Bedrock Nova Lite and Cohere Command R+ both return completions, and the LangFuse auth check
succeeds. Two deviations from the folder plan below were made during the build and are marked
in place: the MCP package is named `mcp_servers/` (not `mcp/`) so it can never shadow the
installed `mcp` SDK on `sys.path`, and `analytics/risk.py` holds the shared risk engine used by
both servers.

This document answers Step 1 (deliverables 1–8) and records the verification results for Steps 2–5, plus the
blockers from Step 13 (secrets) and the specification conflicts that need your ruling.

---

## 0. Specification conflicts that need your ruling

Your final instruction was **"do only the tech stack mentioned in the FA5_SP document."** That instruction
directly contradicts parts of Steps 11, 12 and 15 of your brief. I have resolved every conflict **in favour of
the document** and listed them here so nothing is silently dropped.

| # | Your brief says | Document (Table 14 / Table 15) says | Resolution taken |
|---|---|---|---|
| C1 | Step 11: UI in Next.js + TypeScript + Tailwind + shadcn/ui + Framer Motion + React Query | Frontend/UI = **Streamlit** (HITL Dashboard, 5 pages, :8501) and **Gradio** (Host Orchestrator UI, :8083) | Build Streamlit + Gradio. No Next.js. The "enterprise, animated, dark/light, beautiful" quality bar is honoured **inside** Streamlit/Gradio via custom CSS theming, component library and charts. |
| C2 | Step 11: 20+ UI screens (Admin Panel, System Health, Trace Viewer, Upload Center, …) | Table 13 defines **exactly 5 Streamlit pages** | Deliver the 5 mandated pages exactly as specified, and fold the extra screens in as **sections/tabs within those 5 pages plus the Gradio host UI**, not as new top-level pages. Full mapping in §9. |
| C3 | Step 12: PostgreSQL + SQLAlchemy + Redis | Document names **SQLite** only (Agno `SqliteDb` sessions); no Postgres, no Redis anywhere | SQLite for all persistence. No Postgres, no Redis. |
| C4 | Step 15: Docker, Docker Compose, CI/CD pipeline | Deployment = **NuvePro Lab**; no containers mentioned | **Not building** Docker/Compose/CI by default. Say the word and I will add them as an optional, additive layer that does not change the runtime. |
| C5 | Step 3: Extractor must support DOCX | Table 14 lists `.txt / .pdf / .docx` and "Tesseract (optional)" | Supporting TXT, JSON, PDF, DOCX, PNG. DOCX included. |
| C6 | Document §6 shows input at `data/input/P001/`, `P002/` (per-patient folders) | Repo actually ships `Data/incoming/{doctor_reports,lab_reports,bills}` (per-doctype folders), which matches **your workflow diagram** | Root path and layout mode become config in `agent_config.yaml`. Default = the shipped `Data/incoming/` doctype layout; per-patient layout supported as a second mode. Patient grouping is by the `P####` filename prefix. |
| C7 | Table 14: "Mock EHR System · FastAPI :8050 · **5 JSON data files** (patients, medications, allergies, labs, care_plans)" | Repo ships `mock_ehr/data.py` (Python dicts, 24 patients, with deliberate test-case mismatches documented inline) | Keep `data.py` as the single source of truth and add a generator that materialises the 5 JSON files into `mock_ehr/data/`. The FastAPI service reads the JSON files, so both the document and the existing repo asset are satisfied and nothing is hand-duplicated. |
| C8 | Step 10: "Toxicity Filter" listed as a guardrail | Table 12 also lists it | Implemented. All 5 guardrails from Table 12 are in scope. |

**One more gap you should know about:** the README already in the repo documents a working system
(`run.py`, `app.py`, `hospital_ai/*`, ports 8083/8501/8050/8200/8201). **None of that code exists.** The repo
contains only the spec, `configs/*.yaml`, `mock_ehr/data.py`, and sample documents for P1019–P1024. The README
will be rewritten to match what is actually built.

---

## 1. Project Summary

**DischargeFlow** is an end-to-end agentic AI system that automates hospital discharge review for
St. Marian Regional Medical Center.

**Problem (doc §1.2).** A multi-specialty hospital network receives discharge packets in mixed formats
(PDF, DOCX, scanned handwriting, structured exports) and mixed languages (English, Hindi, Spanish, German —
and the shipped data adds Dutch). Manual review is slow, error-prone, and gated on senior clinicians. Failures
cause readmissions, medication errors, missed follow-ups and regulatory exposure.

**Solution.** For each patient case the system ingests the three documents (doctor report, lab report, bill),
extracts them into structured clinical JSON, translates and normalises them into standard English with a
confidence score, validates them against a Mock EHR plus a versioned YAML rule set, computes a composite risk
score, generates a JSON/HTML/PDF audit report, indexes the case into FAISS, and then either **streams a
patient-friendly discharge summary** (clean cases) or **blocks release and escalates to a human reviewer**
(critical findings). Staff can ask grounded questions about any indexed case at any time.

**The twelve objectives from doc §1.3 map to deliverables as follows:**

| # | Objective | Delivered by |
|---|---|---|
| 1 | Multi-format, multi-language ingestion | Monitor Agent (:8103) + Clinical Data Harvester Tool |
| 2 | Extract + translate to English via MCP Sampling | Extractor (:8100) + Normalizer (:8102) + Medical Lang Bridge Tool |
| 3 | Validate against Mock EHR + YAML rules | Validation Agent (:8101) + Mock EHR (:8050) + `rules.yaml` |
| 4 | MCP Elicitation for missing fields | Clinical Rules Engine Tool + Streamlit `elicitation_callback` |
| 5 | FAISS index for RAG | Agno Indexing role (:8105) |
| 6 | JSON + HTML/PDF reports with audit trail | Clinical Insight Reporter Tool |
| 7 | Stream summaries via A2A Streaming | Summary Generator (:8104) |
| 8 | HITL feedback, corrections, re-run | Streamlit Dashboard (:8501), 5 pages |
| 9 | Two MCP servers, all six primitives | Primary :8200 + Secondary :8201 |
| 10 | Orchestrate 3 frameworks over A2A | Host Orchestrator (:8083) |
| 11 | RAI guardrails | `guardrails/` — 5 modules per Table 12 |
| 12 | Real-time LangFuse tracing | `observability/` — one trace per case |

**Definition of done.** The Mock EHR holds 24 patients (P1001–P1024), but only **P1019–P1024 have documents on
disk** — 6 patients × 3 documents = **18 primary source files** (plus 6 pre-generated `.ocr.txt` sidecars for
the scanned PDFs/PNGs). So end-to-end pipeline runs cover P1019–P1024, and each must land on the outcome its
`mock_ehr/data.py` comment predicts: P1019/P1020/P1023 auto-approve as Low risk; P1022/P1024 hard-block on the
Penicillin-vs-Amoxicilline allergy contradiction; P1021 escalates on unpaid bill plus missing address and
follow-up plus low Hindi translation confidence. The remaining 18 EHR patients stay reachable through the Mock
EHR API and are covered by unit tests of the rules engine.

---

## 2. Architecture Summary

Five layers, exactly as Figure 1 and Table 15 specify. Every port below is mandated by the document.

```
USER LAYER
  Streamlit HITL Dashboard  :8501   (5 pages, Table 13)
                    |  A2A / MCP / HTTP
ORCHESTRATOR
  Host Orchestrator (Google ADK) :8083   Gradio UI + A2A client (streaming-capable)
                    |  A2A / MCP / HTTP
A2A AGENTS
  LangGraph : Extractor :8100 · Validator :8101 · Normalizer :8102     (non-streaming)
  Google ADK: Monitor   :8103 · Summary Generator :8104                (8104 STREAMING)
  Agno      : Clinical RAG Q&A :8105                                   (STREAMING)
                    |  A2A / MCP / HTTP
MCP SERVERS + EHR
  Primary MCP Clinical Tools  :8200  /clinicaltools
      Tools · Resources · Prompts · Sampling · Elicitation · Roots
  Secondary MCP Analytics     :8201  /analyticstools
      calculate_risk_score · get_population_benchmarks · generate_risk_heatmap
  Mock EHR (FastAPI)          :8050  patients · meds · allergies · labs · care plans
                    |  A2A / MCP / HTTP
DATA / STORAGE
  FAISS data/vector_db/ · Input docs (MCP Roots) · SQLite · Reports · LangFuse traces
```

**Protocol boundaries.**
- **Agent ↔ Agent** is always A2A (`a2a-sdk`), never a direct Python import. Discovery via
  `GET /.well-known/agent.json`; every request carries `X-Agent-Auth-Token`.
- **Agent ↔ Tool** is always MCP over streamable-HTTP. Agents connect to **both** servers simultaneously via
  `mcp-use` / `MultiMCPTools`.
- **Anything ↔ LLM** goes through the LiteLLM gateway. Nothing calls Bedrock or Cohere directly.
- **Anything ↔ EHR** goes through the FastAPI REST API on :8050. No agent imports `mock_ehr.data`.

**Cross-cutting concerns** (wrap every layer): LangFuse tracing keyed on one `trace_id` per case; the guardrail
manager; structured logging; typed Pydantic contracts; retry with exponential backoff on all network I/O.

---

## 3. Workflow Summary

This is the workflow from your diagram, reconciled against doc §2 and Tables 3, 4, 6, 10, 12.

**Step 0 — Ingest.** Hospital departments drop three documents per patient into the Roots-scoped input
workspace: `doctor_reports/`, `lab_reports/`, `bills/`.

**Step 0.5 — Host Orchestrator (:8083).** Mints `trace_id` and `case_id`, groups files by patient ID, opens the
LangFuse root trace, and drives the workflow as an A2A client.

**Step 1 — Discharge Monitor Agent (ADK :8103).** Registers the input folder as a **Root URI** on the MCP
connection. The Clinical Watcher Tool calls `ctx.list_roots()` to discover authorised folders at runtime — no
raw path is ever passed as a tool parameter — and rejects anything outside the root via `Path.relative_to()`.
Emits the new-patient event that starts the case.

**Step 2 — Clinical Extractor Agent (LangGraph :8100).** `StateGraph` + `MemorySaver`, checkpointed at every
node. The Clinical Data Harvester Tool handles TXT, JSON, PDF, DOCX and PNG (OCR). Output: structured clinical
JSON for all three documents.

**Step 3 — Language Normalizer Agent (LangGraph :8102).** Detect → translate → expand abbreviations. The
Medical Lang Bridge Tool issues `ctx.session.create_message()` with `ModelPreferences` hinting **nova-lite for
non-English, command-r-plus for English**; the agent's `sampling_callback` reads the hint, routes through
LiteLLM, and returns a `CreateMessageResult`. Output: standardised English JSON **plus a translation confidence
score** (threshold 0.70 from `rules.yaml`).

**Step 4 — Validation Agent (LangGraph :8101).** Validates against three sources in parallel: `rules.yaml`
(fetched as an **MCP Resource**, not read off disk), the FastAPI Mock EHR (:8050), and the Secondary MCP
(:8201) for the risk score. Checks mandatory fields (Table 3), allergies, medications, labs, billing and care
plan (Table 4).

> **HITL-1 · MCP Elicitation.** If — and only if — the gaps are **non-blocking**, the Rules Engine Tool calls
> `ctx.elicit()` with a Pydantic schema for the missing fields, in the same pass. **accept** → fill the gap and
> continue. **decline** → mark unresolved and flag for HITL-2. **cancel** → abort and escalate. The Streamlit
> dashboard implements the `elicitation_callback` that renders the dynamic form.

**Step 5 — Reporter Tool (MCP tool, *not* an A2A agent).** Produces the audit report as JSON + HTML + PDF,
stamped with `rules_version` (SHA-256 of `rules.yaml`), risk level, recommendation, and the LangFuse trace link.

**Step 6 — Agno RAG, Indexing role (:8105).** Chunk + embed with `all-MiniLM-L6-v2` → FAISS.
**Runs for every case, including ones about to be blocked**, so staff can query blocked cases too.

**Step 7 — HITL-2 · Escalation Gate (Host decision).** Escalate if **any** of: a blocking field is missing
(Table 3), a Critical rule fired (Table 4), `risk_level == High`, or `discharge_blocked == True`.
- **YES →** Streamlit HITL UI :8501. Human corrects data / overrides risk / approves → **re-runs Validation
  (back to step 4)**. A blocked summary is *never* auto-released.
- **NO →** proceed to summary.

**Step 8 — Summary Generator Agent (ADK :8104, STREAMING).** Only reached when the gate says no problem. Fetches
`summary-generation-prompt` via MCP Prompts and streams the patient-friendly summary **section by section**:
patient → meds → labs → bill → instructions.

**Step 9 — Agno RAG, Q&A roles (:8105, STREAMING).** Retrieve → Augment → Generate → Reflect, with faithfulness
≥ 0.7. Available from the moment step 6 completes — **including while a case sits in HITL-2**. Out-of-context
questions must return exactly: *"I don't know — this information is not available in the patient records."*

---

## 4. Agent Responsibilities

Per your Step 3 template. Every agent additionally: emits LangFuse spans, enforces guardrails on ingress/egress,
and retries network I/O 3× with exponential backoff (1s/2s/4s, jitter).

### 4.1 Host Orchestrator — Google ADK, :8083

| Attribute | Value |
|---|---|
| Framework / Port | Google ADK + Gradio UI · 8083 |
| Role | Workflow orchestrator, A2A **client**, trace/case ID generator, patient grouping, escalation gate |
| Input | Monitor events; manual "process patient" from Gradio or Streamlit |
| Output | `CaseResult` (case_id, trace_id, status, risk, report paths, summary stream handle) |
| State | Case state machine in SQLite: `CREATED → EXTRACTED → NORMALIZED → VALIDATED → REPORTED → INDEXED → (HITL_PENDING | SUMMARY_READY) → CLOSED` |
| Checkpointing | SQLite `cases` + `case_events` (resumable after restart) |
| Streaming | A2A client; consumes `send_message_streaming()` from 8104 and 8105 |
| MCP primitives | None directly (delegates); reads Resources for UI display |
| Comms partners | All six agents |
| Failure cases | Agent unreachable → mark `DEGRADED`, surface in UI, do not silently continue past validation. Partial packet (missing bill) → run with `missing_document` finding. |
| Retries | 3× backoff per A2A call; circuit-breaker after 5 consecutive failures on one agent |

### 4.2 Discharge Monitor Agent — Google ADK, :8103

| Attribute | Value |
|---|---|
| MCP primitives | **Tools + Roots** (Table 6) |
| Input | A2A `scan` request (no paths) |
| Output | `[{patient_id, documents:[{doc_type, uri, sha256, size, mtime}]}]` |
| Key mechanics | Registers `file:///…/Data/incoming` as a Root at connection open. Watcher Tool resolves folders **only** via `ctx.list_roots()`. `Path.relative_to()` traversal check; out-of-root requests rejected with `RootAccessDenied`. |
| State | `seen_documents` table keyed on (path, sha256) so a case is not reprocessed unless content changes |
| Streaming | Non-streaming |
| Failure cases | Root not granted → hard fail (do **not** fall back to raw paths). Unreadable/locked file → skip + warn. Orphan file with no `P####` prefix → quarantine finding. |

### 4.3 Clinical Extractor Agent — LangGraph, :8100

| Attribute | Value |
|---|---|
| MCP primitives | **Tools + Resources + Prompts** (Table 6) |
| Graph | `load_documents → classify_doctype → extract_discharge → extract_labs → extract_bill → assemble → validate_schema` with conditional edges per available doc type |
| State | Typed `ExtractorState` TypedDict; `MemorySaver` checkpointer; thread_id = case_id |
| Tools | Clinical Data Harvester (TXT, JSON, PDF via pdfplumber, DOCX via python-docx, PNG via Tesseract with the shipped `.ocr.txt` sidecars as the offline path) |
| Prompts | `discharge-extraction-prompt(language, doc_types)` |
| Output | `ClinicalRecord` Pydantic model: demographics, diagnoses, prescriptions[9 fields], allergies, follow-up, instructions, approval, labs[], bill |
| Failure cases | Corrupt PDF → per-document `extraction_failed` finding, case continues. OCR unavailable → use sidecar, else flag low confidence. Schema violation → route to `repair` node once, then fail the node with a typed error. |

### 4.4 Clinical Language Normalizer Agent — LangGraph, :8102

| Attribute | Value |
|---|---|
| MCP primitives | **Tools + Sampling + Prompts** (Table 6) |
| Graph | `detect_language → (route) → sample_translate → expand_abbreviations → score_confidence → emit` |
| Sampling | Medical Lang Bridge Tool → `ctx.session.create_message()` + `ModelPreferences(hints=[nova-lite | command-r-plus])`; agent-side `sampling_callback` routes via LiteLLM and returns `CreateMessageResult` |
| Models | **AWS Bedrock Nova Lite** (primary, non-English) · **Cohere Command R+** (fallback/English) |
| Output | Standardised English `ClinicalRecord` + `translation_confidence: float` + per-field provenance |
| Threshold | `< 0.70` (`quality_thresholds.translation_confidence_min`) raises the `low_translation_confidence` finding (weight 3) and the `translation_confidence_below_threshold` hard guardrail |
| Failure cases | Sampling unsupported by client → typed error, no silent skip. Bedrock throttle → Cohere fallback, logged as a LangFuse fallback span. Both down → confidence 0.0 + mandatory HITL. |

### 4.5 Clinical Validation Agent — LangGraph, :8101

| Attribute | Value |
|---|---|
| MCP primitives | **Tools + Elicitation + Resources** (Table 6) |
| Graph | `load_rules(Resource) → completeness → [ehr_cross_validate ‖ lab_reconcile ‖ bill_check ‖ care_plan_check] → elicit_if_nonblocking → risk_score(:8201) → decide` with a conditional loop back for HITL re-runs |
| Rules | `rules.yaml` read **via MCP Resource** `resource://clinical-rules/{completeness,cross-validation}`; SHA-256 stamped as `rules_version` |
| EHR | REST calls to :8050 through the EHR Validation Tool |
| Elicitation | `ctx.elicit()` with Pydantic schema; handles **accept / decline / cancel** distinctly |
| Output | `ValidationResult`: findings[] (rule_id, severity, evidence), completeness_score, risk_score, risk_level, `discharge_blocked`, recommendation |
| Risk | Weights + thresholds from `rules_scoring_matrix`; Low ≤ 2, Medium ≤ 8, High > 8 |
| Failure cases | EHR 5xx → retry 3×, then `ehr_unavailable` Critical finding and block (never auto-approve on unverified data). Elicitation timeout → treat as **decline**. |

### 4.6 Reporter Tool — MCP tool on :8200 (NOT an A2A agent)

Generates **JSON + HTML + PDF**. Contents: missing fields, EHR discrepancies, medication conflicts, translation
confidence, risk level, recommendation, bill amount + payment status, full audit trail with LangFuse trace IDs,
and `rules_version`. HTML from the `resource://report-template/html` MCP Resource via Jinja2; PDF via WeasyPrint.

### 4.7 Summary Generator Agent — Google ADK, :8104, **STREAMING**

| Attribute | Value |
|---|---|
| MCP primitives | **Tools + Prompts** |
| Prompt | `summary-generation-prompt(risk_level, audience)` — fetched via MCP, never hardcoded |
| Output | Patient-friendly summary streamed **section by section**: patient → meds → labs → bill → instructions |
| Guard | Refuses to run when `discharge_blocked == True`; returns a typed `SummaryBlocked` artifact instead |
| Failure cases | Stream break → client resumes from last completed section. Toxicity/hallucination trip → section regenerated once, then escalated. |

### 4.8 Clinical RAG Q&A Agent — Agno, :8105, **STREAMING**

Five internal Agno roles (Table 5), one A2A surface:

| Role | Responsibility |
|---|---|
| Indexing | Parse + chunk + embed (`all-MiniLM-L6-v2`) discharge docs into FAISS |
| Retrieval | Question → embedding → top-k chunks |
| Augmentation | Re-rank retrieved chunks by keyword relevance |
| Generation | Grounded answer; prompt via `get_prompt("rag-answer-prompt")` — **never hardcoded** |
| Reflection | RAG Triad: Faithfulness / Answer Relevance / Context Relevance |

Agno specifics required by doc §2.6: `agno.Agent` with **MultiMCPTools** (connected to both :8200 and :8201),
**SqliteDb** session persistence with **last 3 turns** as context, **async `arun()`**, exposed as A2A on 8105.
Out-of-context answer is the exact mandated string. Faithfulness < 0.7 → blocked and regenerated (Table 12).

---

## 5. APIs Required

### 5.1 Mock EHR REST API — FastAPI :8050

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/patients` | List (filter by service_line) |
| GET | `/patients/{patient_id}` | Demographics + primary_dx + service_line |
| GET | `/patients/{patient_id}/allergies` | Allergy registry (source of truth) |
| GET | `/patients/{patient_id}/medications` | Inpatient med orders |
| GET | `/patients/{patient_id}/labs` | Labs + `abnormal` + `action_in_ehr` |
| GET | `/patients/{patient_id}/care-plan` | followup_required, speciality, window_days |
| GET | `/guidelines/{icd10}` | Clinical pathway lookup |
| GET | `/patients/{patient_id}/bundle` | All of the above in one call (validation fast path) |

### 5.2 A2A surface — every agent (8100–8105, client at 8083)

| Method | Path | Notes |
|---|---|---|
| GET | `/.well-known/agent.json` | AgentCard: name, description, capabilities, skills, `streaming` flag, auth scheme |
| POST | `/a2a/message:send` | Non-streaming (`send_message()`) |
| POST | `/a2a/message:stream` | SSE (`send_message_streaming()`) — **8104 and 8105 only** |
| POST | `/a2a/tasks/{id}:cancel` | Cancellation |
| POST | `/a2a/push` | Push-notification registration (doc header row: "Streaming + Non-Streaming + Push Notifications") |

All routes require `X-Agent-Auth-Token`; mismatch → `401`.

### 5.3 MCP servers

- **Primary :8200 `/clinicaltools`** — streamable-HTTP. 6 tools (Watcher, Harvester, Lang Bridge, Rules Engine,
  EHR Validator, Reporter), 6 Resources (Table 1), 5 Prompts (Table 2), plus Sampling, Elicitation, Roots.
- **Secondary :8201 `/analyticstools`** — `calculate_risk_score`, `get_population_benchmarks`,
  `generate_risk_heatmap`.

### 5.4 Host Orchestrator :8083

Gradio UI plus a thin HTTP control surface the Streamlit dashboard consumes:
`POST /cases` (start), `GET /cases`, `GET /cases/{case_id}`, `POST /cases/{case_id}/revalidate`,
`GET /cases/{case_id}/summary/stream` (SSE relay), `GET /health/agents`.

---

## 6. Folder Structure

```
sample2/
├── configs/
│   ├── agent_config.yaml          # extended: roots, transports, timeouts, retries, feature flags
│   ├── prompts.yaml               # extended to the 5 prompts + params of Table 2
│   └── rules.yaml                 # extended with Table 3 blocking flags + Table 4 rule IDs/severities
├── Data/incoming/                 # MCP Roots workspace (existing sample docs, untouched)
│   ├── doctor_reports/  lab_reports/  bills/
├── data/                          # generated at runtime; git-ignored
│   ├── vector_db/                 # FAISS index + metadata
│   ├── reports/                   # audit JSON / HTML / PDF, pipeline.log, audit JSONL
│   ├── sessions/                  # Agno SqliteDb session store
│   └── state/                     # dischargeflow.sqlite (cases, findings, HITL)
├── mock_ehr/
│   ├── data.py                    # single source of truth (existing)
│   ├── export_json.py             # materialises the 5 JSON files (doc Table 14)
│   └── data/                      # patients/medications/allergies/labs/care_plans.json
├── hospital_ai/
│   ├── core/                      # config loader, typed schemas, ids, errors, retry, logging
│   ├── llm/                       # LiteLLM gateway, Bedrock/Cohere routing, model preferences
│   ├── observability/             # LangFuse client, span helpers, JSONL fallback
│   ├── guardrails/                # pii · hallucination · injection · toxicity · manager
│   ├── ehr/                       # FastAPI app :8050, routers, repository
│   ├── mcp_servers/               # named mcp_servers, not mcp, to avoid shadowing the SDK
│   │   ├── primary/               # server, resources, prompts, roots + tools/
│   │   │   └── tools/             # watcher · harvester · lang_bridge · rules_engine · ehr_validator · reporter
│   │   ├── analytics/             # server :8201 + 3 analytics tools
│   │   └── client.py              # multi-server client with sampling/elicitation/roots callbacks
│   ├── analytics/                 # shared risk engine used by both MCP servers
│   ├── documents/                 # TXT/JSON/PDF/DOCX/PNG loaders with OCR sidecar preference
│   ├── a2a/                       # base server, agent cards, shared-secret auth, client, streaming, push
│   ├── agents/
│   │   ├── adk/                   # host_orchestrator · monitor · summary_generator
│   │   ├── langgraph/             # extractor/ normalizer/ validator/  (state.py, nodes.py, graph.py each)
│   │   └── agno/                  # rag_agent + indexing/retrieval/augmentation/generation/reflection
│   ├── storage/                   # sqlite schema + DAO, FAISS store, checkpoints
│   ├── reporting/                 # json_report, html_report (Jinja2), pdf_report (WeasyPrint), templates/
│   ├── ui/
│   │   ├── gradio_host/           # :8083 orchestrator UI
│   │   └── streamlit_hitl/        # :8501 — 5 pages + theme/ + components/
│   └── run.py                     # supervisor: starts all 11 services, Ctrl+C stops all
├── tests/  unit/  integration/  e2e/  fixtures/
├── docs/   00_ARCHITECTURE_PLAN.md · API.md · DEPLOYMENT.md · DEVELOPER.md · MCP_PRIMITIVES.md
├── .env.example
├── requirements.txt
└── README.md
```

---

## 7. Database Design

Per conflict **C3**, persistence is **SQLite** (the only database in Table 14) plus FAISS and the file system.
Three stores:

### 7.1 `data/state/dischargeflow.sqlite` — operational store

| Table | Key columns | Purpose |
|---|---|---|
| `cases` | case_id PK, patient_id, trace_id, status, risk_level, risk_score, discharge_blocked, rules_version, created_at, updated_at | One row per discharge case; drives the state machine |
| `case_documents` | id PK, case_id FK, doc_type, uri, sha256, language, ocr_used, extracted_at | Ingested documents; sha256 prevents reprocessing |
| `extractions` | id PK, case_id FK, payload_json, schema_version, created_at | Extractor output |
| `normalizations` | id PK, case_id FK, payload_json, source_language, translation_confidence, model_used, created_at | Normalizer output + sampling provenance |
| `validations` | id PK, case_id FK, run_no, completeness_score, risk_score, risk_level, discharge_blocked, recommendation, created_at | One row per validation run (re-runs increment `run_no`) |
| `findings` | id PK, validation_id FK, rule_id, severity, field, expected, actual, weight, resolved | Individual Table 3/Table 4 findings |
| `elicitations` | id PK, case_id FK, schema_json, response_json, action ∈ {accept,decline,cancel}, responded_by, responded_at | MCP Elicitation audit |
| `hitl_reviews` | id PK, case_id FK, reviewer, decision ∈ {approve,edit,reject}, risk_override, corrections_json, notes, created_at | HITL-2 decisions |
| `summaries` | id PK, case_id FK, sections_json, model_used, streamed_at | Generated summaries |
| `rag_queries` | id PK, case_id FK NULL, question, answer, faithfulness, answer_relevance, context_relevance, blocked, created_at | RAG Triad log |
| `guardrail_events` | id PK, case_id FK, guardrail, verdict, detail_json, created_at | Table 12 interventions |
| `audit_events` | id PK, case_id FK, actor, action, payload_json, langfuse_span_id, created_at | Append-only audit trail |
| `seen_documents` | path PK, sha256, first_seen, last_seen | Monitor idempotency |
| `agent_health` | agent, port, last_ok, last_error, consecutive_failures | System-health panel |

Indexes on `cases(patient_id)`, `cases(status)`, `findings(validation_id, severity)`, `audit_events(case_id, created_at)`.
`audit_events` is append-only (no UPDATE/DELETE path) for compliance.

### 7.2 `data/sessions/agno_sessions.sqlite` — Agno `SqliteDb`

Managed by Agno itself; configured for **last 3 turns** of context per doc §2.6.

### 7.3 FAISS `data/vector_db/`

`index.faiss` (384-dim, `all-MiniLM-L6-v2`, cosine/IP on normalised vectors) + `chunks.sqlite` sidecar holding
`chunk_id, case_id, patient_id, doc_type, section, text, token_count, source_uri` for citations and per-patient
filtering.

### 7.4 File artifacts

`data/reports/{case_id}/audit.json|audit.html|audit.pdf`, `data/reports/pipeline.log`,
`data/reports/traces.jsonl` (LangFuse fallback sink).

---

## 8. Implementation Plan

Twelve phases. Per your Step 14, each phase is **Explain → Implement → Test → Verify → Document**, committed
separately, and I stop for your go-ahead between phases if you want that granularity.

| Phase | Scope | Exit criteria |
|---|---|---|
| **P0** | Foundation: package skeleton, config loader, typed Pydantic schemas, structured logging, retry, error taxonomy, `.env.example`, extended `requirements.txt` | `pytest tests/unit/test_core.py` green; configs load and validate |
| **P1** | Mock EHR :8050 — JSON export from `data.py`, FastAPI routers, repository | All 9 endpoints return correct data for all 24 patients; `/docs` renders |
| **P2** | Primary MCP :8200 — 6 Tools, 6 Resources (Table 1), 5 Prompts (Table 2), Roots + traversal prevention | MCP Inspector lists all primitives; traversal attempt rejected |
| **P3** | Secondary MCP :8201 — risk score, benchmarks, heatmap; `mcp-use` multi-server client | Both servers reachable from one client session |
| **P4** | A2A layer — base server, AgentCards, shared-secret auth, non-streaming + streaming + push, client | `/.well-known/agent.json` on all 6; 401 without token; SSE verified |
| **P5** | LangGraph Extractor :8100 — StateGraph, MemorySaver, Harvester (TXT/JSON/PDF/DOCX/PNG-OCR) | All 18 primary documents (P1019–P1024 × 3) extract cleanly across all 5 formats; checkpoints resume after kill |
| **P6** | LangGraph Normalizer :8102 — **MCP Sampling** end to end, LiteLLM routing, confidence scoring | Hindi/Spanish/Dutch cases translate; confidence recorded; nova-lite hint honoured |
| **P7** | LangGraph Validator :8101 — completeness (Table 3), cross-validation (Table 4), risk matrix, **MCP Elicitation** with all 3 outcomes | P1022/P1024 block on allergy; P1019/P1023 auto-approve; accept/decline/cancel all exercised |
| **P8** | Reporter Tool — JSON + HTML (Jinja2) + PDF (WeasyPrint), `rules_version` stamp, audit trail | 3 artifacts per case; hash matches `rules.yaml` |
| **P9** | Agno RAG :8105 — 5 roles, FAISS, MultiMCPTools, SqliteDb (3 turns), async `arun()`, RAG Triad | Grounded answers; out-of-context returns the exact mandated string; faithfulness ≥ 0.7 enforced |
| **P10** | ADK Monitor :8103, Summary Generator :8104 (streaming), Host Orchestrator :8083 + Gradio; guardrails; LangFuse | Full pipeline runs end to end; blocked cases never produce a summary; one trace per case |
| **P11** | Streamlit HITL :8501 — 5 pages (Table 13), `elicitation_callback`, theming, charts, exports | All 5 pages functional; corrections re-run validation |
| **P12** | Hardening — `run.py` supervisor, integration + e2e tests, README/API/Deployment/Developer docs | All 24 patients hit their documented expected outcome; `Ctrl+C` stops all 11 services |

**Sequencing rationale.** Infrastructure (P0–P4) precedes agents because every agent depends on MCP + A2A;
LangGraph agents (P5–P7) precede the reporter and RAG because those consume validated records; the orchestrator
(P10) lands after its callees exist; the UI (P11) lands last because it renders real data rather than mocks.

**Risks.** (a) Google ADK's API surface moves fast — I'll pin `google-adk==2.6.2`, verified installable here.
(b) MCP Sampling and Elicitation require client-side callbacks; the Streamlit and LangGraph clients must both
implement them or the primitive is only half-demonstrated — this is explicitly in the P6/P7 exit criteria.
(c) No LLM credentials yet — see §10.

---

## 9. Verification results (your Steps 2, 4, 5, 11)

### 9.1 Step 2 — Technology component audit

| Component | In document? | In repo now | Plan |
|---|---|---|---|
| Google ADK | Yes (Tables 6, 14, 15) | **Missing from `requirements.txt`** | Add `google-adk==2.6.2` (verified installable) |
| LangGraph | Yes | `langgraph==1.1.10` | Keep |
| Agno | Yes | `agno==2.3.21` | Keep |
| FastAPI | Yes (:8050) | `fastapi==0.123.10` | Keep |
| FastMCP | Yes (Table 15) | `mcp==1.25.0` (bundles FastMCP) | Keep |
| LangFuse | Yes (§7.2) | `langfuse` unpinned | Pin |
| LiteLLM | Yes (Table 14) | `litellm==1.80.11` | Keep |
| FAISS | Yes | `faiss-cpu==1.13.2` | Keep |
| SQLite | Yes (Agno sessions) | stdlib | Use for all persistence |
| Gradio | Yes (:8083) | `gradio==6.2.0` | Keep |
| Streamlit | Yes (:8501) | `streamlit==1.52.2` | Keep |
| AWS Bedrock Nova Lite | Yes (primary LLM) | via LiteLLM | **Needs credentials** |
| Cohere Command R+ | Yes (fallback) | via LiteLLM | **Needs API key** |
| Tesseract | Yes (**optional**) | **Binary not installed** | `pytesseract` + apt install; shipped `.ocr.txt` sidecars are the offline path |
| `mcp-use` (multi-server client) | Yes (Table 14) | **Missing** | Add `mcp-use==1.7.0` |
| a2a-sdk | Yes | `a2a-sdk==0.3.22` | Keep |
| sentence-transformers | Yes | `5.2.2` | Keep |

**Missing dependencies to be added:** `google-adk`, `mcp-use`, `pdfplumber`, `python-docx`, `pypdf`,
`pytesseract`, `Pillow`, `weasyprint`, `jinja2`, `httpx`, `sse-starlette`, `langdetect`, `plotly`, `pytest`,
`pytest-asyncio`. All verified available on PyPI from this environment.

**Nothing in the document is unimplementable.** The only true blockers are credentials (§10).

### 9.2 Step 4 — MCP verification

Two servers, and all six primitives are individually demonstrable (Table 9):

| Primitive | Implementation site | Key APIs |
|---|---|---|
| Tools | Primary :8200 | `mcp.tool()` |
| Resources | Primary :8200 | `mcp.resource()`, `list_resources()`, `read_resource()` |
| Prompts | Primary :8200 | `mcp.prompt()`, `list_prompts()`, `get_prompt()` |
| Sampling | Medical Lang Bridge Tool | `ctx.session.create_message()`, `sampling_callback`, `ModelPreferences` |
| Elicitation | Rules Engine Tool + Streamlit | `ctx.elicit()`, `ElicitResult`, `elicitation_callback`, accept/decline/cancel |
| Roots | Monitor ↔ Watcher Tool | `ctx.list_roots()`, `Root(uri=…)`, `Path.relative_to()` traversal prevention |

Secondary :8201 exposes exactly `calculate_risk_score`, `get_population_benchmarks`, `generate_risk_heatmap`.

### 9.3 Step 5 — A2A verification

Six A2A servers (8100–8105) + one client (8083). Each exposes `/.well-known/agent.json`, requires
`X-Agent-Auth-Token`, and declares its streaming capability. `send_message()` for 8100/8101/8102/8103;
`send_message_streaming()` for 8104 and 8105 (Table 10). Push-notification registration included.

### 9.4 Step 11 — where your extra UI features land

Table 13 fixes 5 Streamlit pages. Your additional screens map on without adding top-level pages:

| Your feature | Home |
|---|---|
| Dashboard, Case Tracking, Risk Dashboard, Agent Status, System Health, Logs | Gradio Host UI :8083 (tabs) |
| Upload Center, Document Viewer, PDF Preview, JSON Viewer | Streamlit Page 1 |
| Validation Report, Audit Report, Charts, Export, Trace Viewer link | Streamlit Page 2 |
| HITL Dashboard, Interactive Tables, Elicitation form | Streamlit Page 3 |
| RAG Chat, Clinical Chat, Search, Filters | Streamlit Page 4 |
| Streaming Summary, Patient Timeline, Export JSON/HTML/PDF | Streamlit Page 5 |
| Dark/Light mode, animations, toasts, skeletons, cards, reusable components | Shared `ui/streamlit_hitl/theme/` + `components/` |
| Settings, Admin Panel | Gradio host "Settings" tab (config + agent tokens, read-only for secrets) |

---

## 10. Secrets required (your Step 13 — stopping to ask)

Nothing is hardcoded. All of these are read from environment variables via `.env` (git-ignored), documented in
`.env.example`. **Please provide, or confirm I should proceed in offline deterministic mode:**

| Secret | Env var | Needed for | Blocking? |
|---|---|---|---|
| AWS access key / secret / region | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | Bedrock Nova Lite (primary LLM, MCP Sampling) | **Supplied and verified** |
| Bedrock model ids | `BEDROCK_PRIMARY_MODEL`, `BEDROCK_FALLBACK_MODEL` | LiteLLM routing | Default to `bedrock/amazon.nova-lite-v1:0` and `bedrock/cohere.command-r-plus-v1:0` |
| Cohere API key | `COHERE_API_KEY` | Command R+ fallback | **Not needed.** Command R+ is reachable through Bedrock on the supplied credentials, so the mandated fallback works without a separate Cohere account |
| LangFuse keys + host | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | Live tracing | **Supplied and verified** (`auth_check()` returns true) |
| A2A shared secret | `AGENT_AUTH_TOKEN` | A2A auth | Generated; stored in the git-ignored `.env` |
| Database URL | — | Not needed; SQLite is file-based | No |
| SMTP / Google credentials | — | Not referenced anywhere in the document | No |
| SerpAPI key | `SERPAPI_API_KEY` | Not referenced anywhere in the document | No — stored but unused |

**Security note.** The credentials were pasted into a chat message, so they should be treated as
exposed and rotated once the project is delivered. They live only in `sample2/.env`, which is
git-ignored and was verified absent from every commit.

**Offline deterministic mode.** `LLM_OFFLINE=1` keeps the system demonstrable without credentials, and the
test suite runs in it. Offline translations are capped below the 0.70 confidence threshold on purpose, so an
offline run routes to HITL instead of silently auto-approving. With the supplied credentials the live
Bedrock path is the default.

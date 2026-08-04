# DischargeFlow — API Reference

All HTTP services, A2A agents, and MCP endpoints exposed by the system.

## Service ports (Table 15)

| Service | Port | Base URL |
|---|---|---|
| Mock EHR | 8050 | http://localhost:8050 |
| Clinical Extractor (A2A) | 8100 | http://localhost:8100 |
| Clinical Validation (A2A) | 8101 | http://localhost:8101 |
| Clinical Normalizer (A2A) | 8102 | http://localhost:8102 |
| Discharge Monitor (A2A) | 8103 | http://localhost:8103 |
| Summary Generator (A2A) | 8104 | http://localhost:8104 |
| Clinical RAG Q&A (A2A) | 8105 | http://localhost:8105 |
| Host Orchestrator (Gradio) | 8083 | http://localhost:8083 |
| Primary MCP | 8200 | http://localhost:8200/clinicaltools |
| Analytics MCP | 8201 | http://localhost:8201/analyticstools |
| HITL Dashboard (Streamlit) | 8501 | http://localhost:8501 |

## Authentication

All A2A agents require the shared secret header:

```
X-Agent-Auth-Token: <value of AGENT_AUTH_TOKEN from .env>
```

Missing or incorrect tokens return `401 Unauthorized`.

## Mock EHR (:8050)

Interactive docs: http://localhost:8050/docs

| Method | Path | Description |
|---|---|---|
| GET | `/patients` | List all patients (optional `?service_line=`) |
| GET | `/patients/{patient_id}` | Patient demographics |
| GET | `/patients/{patient_id}/allergies` | Documented allergies |
| GET | `/patients/{patient_id}/medications` | Active medication orders |
| GET | `/patients/{patient_id}/labs` | Recent lab results |
| GET | `/patients/{patient_id}/care-plan` | Discharge care plan |
| GET | `/patients/{patient_id}/bundle` | Full patient bundle |
| GET | `/guidelines/{icd10}` | Clinical guideline snippet |
| GET | `/health` | Liveness probe |

Data is read from `mock_ehr/data/*.json`, generated from `mock_ehr/data.py`.

## A2A agents (:8100–8105)

Each agent exposes:

| Path | Description |
|---|---|
| `/.well-known/agent.json` | Agent Card (capabilities, streaming mode) |
| `POST /` | `send_message` (non-streaming agents) or streaming task lifecycle |

### Agent payloads

**Extractor** — `{"patient_id": "P1019"}` → structured clinical JSON

**Normalizer** — `{"record": {...}}` → English-normalised record with confidence score

**Validator** — `{"patient_id": "P1019", "record": {...}}` → validation result with findings and risk

**Monitor** — `{"only_new": true}` → discovered patients in the Roots workspace

**Summary** (streaming) — `{"case_id": "...", "record": {...}, "validation": {...}}` → section-by-section patient summary

**RAG** (streaming) — `{"question": "...", "patient_id": "P1019", "session_id": "..."}` → grounded answer with RAG Triad scores

## Primary MCP (:8200 `/clinicaltools`)

Streamable-HTTP transport. Six tools, six resources, five prompts, plus Sampling, Elicitation, and Roots.

### Tools

| Tool | Description |
|---|---|
| `clinical_watcher` | Discover documents in the Roots-authorised workspace |
| `clinical_data_harvester` | Extract raw text from a document URI |
| `medical_lang_bridge` | Translate and expand abbreviations via MCP Sampling |
| `clinical_rules_engine` | Completeness validation with optional MCP Elicitation |
| `ehr_validation` | Cross-validate against the Mock EHR REST API |
| `clinical_insight_reporter` | Generate JSON/HTML/PDF audit reports |

### Resources (Table 1)

| URI | Content |
|---|---|
| `rules://completeness` | Completeness rules from `configs/rules.yaml` |
| `rules://cross-validation` | Cross-validation rules |
| `rules://risk-matrix` | Risk scoring matrix |
| `abbreviations://medical` | Medical abbreviation map |
| `template://audit-report` | Jinja2 HTML audit template |
| `document://{patient_id}/{doc_type}` | Document text for a patient |

### Prompts (Table 2)

`discharge-extraction-prompt`, `abbreviation-normalization-prompt`, `validation-summary-prompt`, `patient-summary-prompt`, `rag-qa-prompt`

## Analytics MCP (:8201 `/analyticstools`)

| Tool | Description |
|---|---|
| `calculate_risk_score` | Composite discharge risk from findings |
| `get_population_benchmarks` | Population-level risk statistics |
| `generate_risk_heatmap` | Risk distribution heatmap data |

## Host Orchestrator (Gradio :8083)

The Gradio UI drives the workflow via `DashboardService`. Key operations:

- `discover()` — scan the Roots workspace
- `process_patient(patient_id)` — run the full pipeline
- `revalidate(case_id, corrections)` — apply HITL corrections and re-run validation
- `summary_events(case_id)` — stream patient-friendly summary sections
- `ask(question, patient_id)` — grounded RAG query

## HITL Dashboard (Streamlit :8501)

Five pages (Table 13): Document Viewer, Validation Report, HITL Corrections, RAG Q&A, Discharge Summary. The Corrections page supplies values for MCP Elicitation via `set_elicitation_answers()`.

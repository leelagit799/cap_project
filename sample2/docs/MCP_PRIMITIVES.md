# DischargeFlow — MCP Primitives Reference

The Primary MCP Clinical Tools Server (:8200) demonstrates all six MCP primitives. The Secondary Analytics Server (:8201) exposes tools only.

## Primitive overview

| Primitive | Primary MCP | Secondary MCP | Used by |
|---|---|---|---|
| **Tools** | 6 clinical tools | 3 analytics tools | All agents via `MultiServerMCPClient` |
| **Resources** | 6 resources (Table 1) | — | Validator (rules), Reporter (template) |
| **Prompts** | 5 prompts (Table 2) | — | Extractor, Normalizer, Summary, RAG |
| **Sampling** | Supported | — | Medical Language Bridge → Normalizer callback |
| **Elicitation** | Supported | — | Rules Engine → Streamlit HITL callback |
| **Roots** | Supported | — | Watcher, Harvester (path safety) |

## Tools

### Primary (:8200)

| Tool | Input | Output |
|---|---|---|
| `clinical_watcher` | `{patient_id?}` | Discovered patients, documents, primary encodings |
| `clinical_data_harvester` | `{uri, include_tables?}` | Document text, metadata, tables |
| `medical_lang_bridge` | `{text, source_language?, detection_confidence?}` | Translated text, confidence, model used |
| `clinical_rules_engine` | `{packet, elicit?}` | Completeness score, findings, elicitations |
| `ehr_validation` | `{patient_id, packet}` | Cross-validation findings (Table 4 rules) |
| `clinical_insight_reporter` | `{case_id, patient_id, trace_id, validation, ...}` | Audit JSON/HTML/PDF paths |

### Analytics (:8201)

| Tool | Input | Output |
|---|---|---|
| `calculate_risk_score` | `{findings, translation_confidence?, service_line?}` | Risk level, score, recommendation |
| `get_population_benchmarks` | `{}` | Population risk statistics |
| `generate_risk_heatmap` | `{}` | Risk distribution data |

## Resources (Table 1)

| URI | Description |
|---|---|
| `rules://completeness` | Required fields and blocking flags |
| `rules://cross-validation` | Seven cross-validation rules |
| `rules://risk-matrix` | Severity weights and thresholds |
| `abbreviations://medical` | Abbreviation → expanded form map |
| `template://audit-report` | Jinja2 HTML audit template |
| `document://{patient_id}/{doc_type}` | Preferred document text |

## Prompts (Table 2)

Fetched at runtime — never hardcoded in agent logic.

| Prompt | Parameters | Used by |
|---|---|---|
| `discharge-extraction-prompt` | `language`, `doc_types` | Clinical Extractor |
| `abbreviation-normalization-prompt` | `source_language` | Medical Language Bridge |
| `validation-summary-prompt` | `risk_level`, `findings_count` | Validation Agent |
| `patient-summary-prompt` | `patient_name`, `risk_level` | Summary Generator |
| `rag-qa-prompt` | `context`, `question` | RAG Generation role |

## Sampling

The Medical Language Bridge tool calls `ctx.session.create_message()` to request translation from the client's LLM. The Normalizer agent registers a sampling callback that routes through LiteLLM:

- Non-English text → Bedrock Nova Lite (hint: `nova-lite`)
- English text → Cohere Command R+ (hint: `command-r-plus`)

If no sampling handler is registered, the tool raises `SamplingUnsupportedError`.

## Elicitation

The Rules Engine calls `ctx.session.elicit()` when non-blocking fields are missing. Three outcomes:

| Action | Effect |
|---|---|
| `accept` | Supplied values fill the gaps; validation continues |
| `decline` | Gap remains unresolved; case flagged for HITL-2 |
| `cancel` | Validation aborts; case escalates |

The Streamlit HITL Corrections page collects reviewer answers via `set_elicitation_answers()`. Headless agent processes decline by default.

## Roots

The Discharge Monitor registers `Data/incoming/` as the authorised workspace. The Watcher and Harvester tools:

1. Call `ctx.list_roots()` to discover allowed paths
2. Resolve document URIs against roots
3. Reject paths outside the workspace with `RootAccessDeniedError` (uses `Path.relative_to()`)

No raw filesystem paths are accepted as tool parameters.

## Multi-server client

`hospital_ai/mcp_servers/client.py` provides `MultiServerMCPClient` which connects to both MCP servers simultaneously and wires:

- `sampling_handler` — client-side LLM inference
- `elicitation_handler` — HITL form responses
- `list_roots_callback` — workspace declaration

```python
async with MultiServerMCPClient(
    sampling_handler=my_sampling_fn,
    elicitation_handler=my_elicitation_fn,
) as client:
    result = await client.call_tool("clinical_watcher", {})
```

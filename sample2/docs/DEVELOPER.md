# DischargeFlow — Developer Guide

## Repository layout

```
sample2/
├── configs/           # agent_config.yaml, prompts.yaml, rules.yaml
├── Data/incoming/     # MCP Roots workspace (sample patient documents)
├── data/              # Runtime output (git-ignored except .gitkeep placeholders)
├── docs/              # Architecture plan, API, deployment, developer docs
├── hospital_ai/       # Application source
├── mock_ehr/          # EHR data source (data.py) + exported JSON
├── tests/             # unit/, integration/, e2e/
├── run.py             # Service supervisor
└── requirements.txt
```

## Development setup

> `.venv` (including `Scripts/` on Windows or `bin/` on Linux) is **never** committed to git.

```bash
cd sample2

# Windows
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1

# Linux / Mac
bash scripts/setup.sh
source .venv/bin/activate

cp .env.example .env   # fill credentials or set LLM_OFFLINE=1
```

## Running individual services

```bash
# Mock EHR
python -m hospital_ai.ehr.app

# MCP servers
python -m hospital_ai.mcp_servers.primary.server
python -m hospital_ai.mcp_servers.analytics.server

# A2A agents
python -m hospital_ai.agents.serve extractor
python -m hospital_ai.agents.serve validator
python -m hospital_ai.agents.serve normalizer
python -m hospital_ai.agents.serve monitor
python -m hospital_ai.agents.serve summary
python -m hospital_ai.agents.serve rag

# UIs
python -m hospital_ai.ui.gradio_host.app
streamlit run hospital_ai/ui/streamlit_hitl/app.py --server.port 8501
```

Or start everything: `python run.py`

## Testing

```bash
# Full suite (offline LLM stub — no credentials needed)
pytest tests/ -q

# By layer
pytest tests/unit/ -q
pytest tests/integration/ -q
pytest tests/e2e/ -q

# Live Bedrock (requires .env credentials)
pytest tests/e2e/test_rag_live.py -q
```

## Key design decisions

### MCP package naming

The package is `hospital_ai/mcp_servers/` (not `mcp/`) to avoid shadowing the installed `mcp` SDK on `sys.path`.

### Elicitation

MCP Elicitation is synchronous inside a tool call. The Streamlit HITL dashboard collects reviewer answers on the Corrections page; headless agent processes decline by default (safe HITL escalation).

### Sampling

The Medical Language Bridge tool has no LLM of its own. It issues MCP Sampling requests; the Normalizer agent supplies the client-side callback that routes through LiteLLM to Bedrock.

### SQLite persistence

All operational state lives in `data/state/dischargeflow.sqlite`. Validation runs are stored per-run (not overwritten) so pre-HITL state remains auditable.

### Trace IDs

The Host Orchestrator creates one `trace_id` per case and propagates it through LangFuse spans, audit reports, and the JSONL fallback sink.

## Adding a new validation rule

1. Add the rule to `configs/rules.yaml` with `rule_id`, `severity`, `weight`, and `blocking` flag.
2. If it requires EHR cross-check, add logic in `hospital_ai/mcp_servers/primary/tools/ehr_validator.py`.
3. If it affects risk scoring, update `hospital_ai/analytics/risk.py`.
4. Add a test case in `tests/integration/test_pipeline_agents.py`.

## Common issues

| Symptom | Fix |
|---|---|
| `SamplingUnsupportedError` | Ensure the MCP client provides a `sampling_callback` |
| Port already in use | Kill stale process or change port in `configs/agent_config.yaml` |
| Empty Mock EHR | Run `python -m mock_ehr.export_json` |
| FAISS index missing | Process a patient through the orchestrator to trigger indexing |
| `litellm` event loop error in tests | Call `litellm.close_litellm_async_clients()` between live tests |

## Code style

- Pydantic models in `hospital_ai/core/schemas.py` for all inter-agent contracts
- Structured JSON logging via `hospital_ai/core/logging.py`
- Retry only on `retryable=True` errors (`hospital_ai/core/retry.py`)
- Guardrails never silently pass — blocked cases escalate to HITL

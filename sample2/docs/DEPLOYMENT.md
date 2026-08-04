# DischargeFlow — Deployment Guide

Deployment target per the specification: **NuvePro Lab** (Python 3.11+, no Docker required).

## Prerequisites

- Python 3.11 or 3.12
- `pip` and `venv`
- Tesseract OCR (`tesseract-ocr` system package) for scanned document support
- WeasyPrint system libraries (`libpango`, `libharfbuzz`) for PDF audit reports
- Network egress to AWS Bedrock and LangFuse Cloud (or set `LLM_OFFLINE=1`)

## Installation

```bash
cd sample2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Ubuntu/Debian, install system dependencies:

```bash
sudo apt-get install -y tesseract-ocr libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0
```

## Configuration

```bash
cp .env.example .env
```

Fill in:

| Variable | Required for | Notes |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | Live LLM | Bedrock Nova Lite + Command R+ fallback |
| `AWS_SECRET_ACCESS_KEY` | Live LLM | |
| `AWS_DEFAULT_REGION` | Live LLM | Default `us-east-1` |
| `LANGFUSE_PUBLIC_KEY` | Observability | Optional; falls back to `data/reports/traces.jsonl` |
| `LANGFUSE_SECRET_KEY` | Observability | |
| `AGENT_AUTH_TOKEN` | A2A auth | Generate a random string; shared by all agents |

Set `LLM_OFFLINE=1` to run without AWS credentials (deterministic stub, routes uncertain cases to HITL).

## First-run data setup

The Mock EHR JSON files ship in `mock_ehr/data/`. If missing, regenerate:

```bash
python -m mock_ehr.export_json
```

Input documents are in `Data/incoming/` (read-only). Runtime artifacts are written to `data/` (reports, vector indexes, SQLite state).

## Starting the system

```bash
python run.py
```

This starts all eleven services in dependency order:

1. Mock EHR (:8050)
2. Primary MCP (:8200)
3. Analytics MCP (:8201)
4. Six A2A agents (:8100–8105)
5. Gradio Host Orchestrator (:8083)
6. Streamlit HITL Dashboard (:8501)

Partial startup:

```bash
python run.py --only ehr mcp    # infrastructure only
python run.py --only agents     # agents only (requires MCP + EHR)
python run.py --only ui         # UIs only (requires agents)
```

`Ctrl+C` stops all processes. Service logs are written to `data/reports/services/`.

## Verification

```bash
# Health checks
curl http://localhost:8050/health
curl -H "X-Agent-Auth-Token: $AGENT_AUTH_TOKEN" http://localhost:8100/.well-known/agent.json

# Full test suite
pytest tests/ -q
```

## Port conflicts

If a port is already in use, `run.py` skips that service and prints a warning. Check `data/reports/services/<service>.log` for startup errors.

## Production notes

- Rotate `AGENT_AUTH_TOKEN` before any shared deployment.
- Never commit `.env` — it is git-ignored.
- `data/` grows with each processed case; plan disk space for audit reports and FAISS indexes.
- LangFuse traces require outbound HTTPS to `cloud.langfuse.com` (or your self-hosted instance).

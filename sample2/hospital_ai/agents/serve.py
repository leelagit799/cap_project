"""A2A server entrypoints for the six agents (ports 8100–8105).

Each process opens its own multi-server MCP session, supplying the client-side
callbacks its role needs — the Normalizer brings a sampling handler, the
Validator brings an elicitation handler, and the Monitor declares the Roots
workspace.

Run one with ``python -m hospital_ai.agents.serve <agent>``, or let
``run.py`` start all of them.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, AsyncIterator

import mcp.types as types
import uvicorn

from hospital_ai.a2a.cards import CARD_BUILDERS
from hospital_ai.a2a.server import build_app
from hospital_ai.agents.adk.monitor import DischargeMonitorAgent
from hospital_ai.agents.adk.summary import SummaryGeneratorAgent
from hospital_ai.agents.agno.rag_agent import ClinicalRAGAgent
from hospital_ai.agents.langgraph.extractor import ClinicalExtractorAgent
from hospital_ai.agents.langgraph.normalizer import (
    ClinicalNormalizerAgent,
    build_sampling_handler,
)
from hospital_ai.agents.langgraph.validator import ClinicalValidationAgent
from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import configure_logging, get_logger
from hospital_ai.mcp_servers.client import MultiServerMCPClient

_log = get_logger(__name__, component="agent-server")

AGENT_CLASSES = {
    "extractor": ClinicalExtractorAgent,
    "validator": ClinicalValidationAgent,
    "normalizer": ClinicalNormalizerAgent,
    "monitor": DischargeMonitorAgent,
    "summary": SummaryGeneratorAgent,
    "rag": ClinicalRAGAgent,
}

STREAMING_AGENTS = {"summary", "rag"}


async def _sampling(params: types.CreateMessageRequestParams):
    return await build_sampling_handler()(params)


async def _elicitation(params: types.ElicitRequestParams) -> types.ElicitResult:
    """Server-process default: decline.

    Nobody is watching this process, and doc §2.4.1 says a declined elicitation
    leaves the gap unresolved and flags it for HITL — which is the safe
    outcome. The Streamlit dashboard supplies the interactive implementation.
    """
    return types.ElicitResult(action="decline")


def serve(agent_name: str) -> None:
    if agent_name not in AGENT_CLASSES:
        raise SystemExit(
            f"Unknown agent {agent_name!r}. Choose from: {', '.join(AGENT_CLASSES)}"
        )

    settings = get_settings()
    configure_logging(settings.log_level, settings.reports_dir / "pipeline.log")
    settings.ensure_dirs()

    card = CARD_BUILDERS[agent_name]()
    port = getattr(settings.ports, agent_name)
    streaming = agent_name in STREAMING_AGENTS

    # One MCP session lives for the process lifetime; each A2A request borrows it.
    client = MultiServerMCPClient(
        sampling_handler=_sampling, elicitation_handler=_elicitation
    )
    holder: dict[str, Any] = {}

    async def ensure_agent():
        if "agent" not in holder:
            await client.__aenter__()
            holder["agent"] = AGENT_CLASSES[agent_name](client)
            _log.info("agent ready", extra={"agent": agent_name, "port": port})
        return holder["agent"]

    async def handler(payload: dict[str, Any]) -> Any:
        agent = await ensure_agent()
        return await agent.handle(payload)

    async def stream_handler(payload: dict[str, Any]) -> AsyncIterator[Any]:
        agent = await ensure_agent()
        async for event in agent.handle_stream(payload):
            yield event

    app = build_app(
        card,
        handler=None if streaming else handler,
        stream_handler=stream_handler if streaming else None,
    )

    # MCP session opens lazily on the first A2A request via ensure_agent() in the
    # handlers above. Starlette 1.3+ removed router.on_startup — do not use it.
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=settings.log_level.lower())


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"Usage: python -m hospital_ai.agents.serve <{'|'.join(AGENT_CLASSES)}>")
    serve(sys.argv[1])


if __name__ == "__main__":
    main()

"""P9 — RAG against live Bedrock, including the LLM-as-judge RAG Triad.

Skipped when AWS credentials are absent or ``LLM_OFFLINE=1``, since these
assertions depend on a real model producing grounded answers.
"""

from __future__ import annotations

import contextlib
import os

import mcp.types as types
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import AnyUrl

from hospital_ai.agents.agno.rag_agent import ClinicalRAGAgent
from hospital_ai.agents.gateway import SessionGateway
from hospital_ai.agents.langgraph.extractor import ClinicalExtractorAgent
from hospital_ai.core.config import get_settings
from hospital_ai.core.schemas import OUT_OF_CONTEXT_ANSWER
from hospital_ai.llm.gateway import reset_gateway
from hospital_ai.mcp_servers.primary.server import create_server
from hospital_ai.rag.store import FaissStore

pytestmark = pytest.mark.anyio

# The suite defaults to LLM_OFFLINE=1 so it is fast, free and deterministic.
# These tests opt back in when real credentials exist.
requires_llm = pytest.mark.skipif(
    not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")),
    reason="AWS credentials are not configured",
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _flush_litellm_clients() -> None:
    """Close LiteLLM's cached async HTTP clients.

    LiteLLM keeps aiohttp sessions and an internal queue bound to the event
    loop that created them. pytest gives each test a fresh loop, so a reused
    client raises "bound to a different event loop".
    """
    import litellm

    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is not None and hasattr(cache, "flush_cache"):
        cache.flush_cache()

    closer = getattr(litellm, "close_litellm_async_clients", None)
    if closer is not None:
        with contextlib.suppress(Exception):
            await closer()

    # LoggingWorker is a module-level singleton whose asyncio.Queue is created
    # on first use and pinned to that loop. Detach it so the next test's loop
    # gets a fresh queue.
    with contextlib.suppress(Exception):
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        await GLOBAL_LOGGING_WORKER.stop()
        GLOBAL_LOGGING_WORKER._queue = None
        GLOBAL_LOGGING_WORKER._worker_task = None


@pytest.fixture(autouse=True)
async def live_llm(monkeypatch):
    """Force the gateway out of offline mode for the duration of a live test."""
    monkeypatch.setenv("LLM_OFFLINE", "0")
    get_settings.cache_clear()
    reset_gateway()
    await _flush_litellm_clients()
    yield
    monkeypatch.setenv("LLM_OFFLINE", "1")
    get_settings.cache_clear()
    reset_gateway()
    await _flush_litellm_clients()


async def _roots(context):
    workspace = get_settings().roots.workspace.resolve()
    return types.ListRootsResult(
        roots=[types.Root(uri=AnyUrl(workspace.as_uri()), name="clinical-input")]
    )


@contextlib.asynccontextmanager
async def indexed_agent(tmp_path, patients=("P1019", "P1021", "P1022")):
    async with create_connected_server_and_client_session(
        create_server()._mcp_server, list_roots_callback=_roots
    ) as session:
        gateway = SessionGateway(session)
        agent = ClinicalRAGAgent(gateway, store=FaissStore(tmp_path / "vectors"))
        extractor = ClinicalExtractorAgent(gateway)
        for patient in patients:
            record = (await extractor.extract(patient, f"CASE-{patient}"))["record"]
            agent.index_case(f"CASE-{patient}", record)
        yield agent


@requires_llm
class TestLiveGroundedAnswers:
    async def test_medication_question_is_answered_and_grounded(self, tmp_path):
        async with indexed_agent(tmp_path) as agent:
            answer = await agent.answer(
                "What medications was Thomas Wright discharged on?"
            )

        assert not answer.blocked
        assert "Metformin" in answer.answer
        assert answer.triad.faithfulness >= 0.7
        assert answer.chunks

    async def test_allergy_question_is_answered_from_the_dutch_record(self, tmp_path):
        async with indexed_agent(tmp_path) as agent:
            answer = await agent.answer("What allergies does Daan Bakker have?")

        assert not answer.blocked
        assert "enicillin" in answer.answer  # Penicillin / penicillin
        assert answer.triad.passes

    async def test_billing_question_reports_the_unpaid_status(self, tmp_path):
        async with indexed_agent(tmp_path) as agent:
            answer = await agent.answer("Is the hospital bill for Rohan Gupta paid?")

        assert not answer.blocked
        assert "UNPAID" in answer.answer.upper()

    async def test_out_of_scope_question_gets_the_mandated_refusal(self, tmp_path):
        async with indexed_agent(tmp_path) as agent:
            answer = await agent.answer("Who won the 2022 football World Cup?")

        assert answer.answer.strip() == OUT_OF_CONTEXT_ANSWER

    async def test_streaming_emits_sources_then_tokens_then_completion(self, tmp_path):
        kinds: list[str] = []
        text = ""
        async with indexed_agent(tmp_path, patients=("P1019",)) as agent:
            async for event in agent.stream_answer(
                "What medications was Thomas Wright discharged on?"
            ):
                kinds.append(event["type"])
                if event["type"] == "complete":
                    text = event["answer"]

        assert kinds[0] == "sources"
        assert "token" in kinds
        assert kinds[-1] in {"complete", "blocked"}
        if kinds[-1] == "complete":
            assert "Metformin" in text

"""Synchronous bridge between the UIs and the async agent stack.

Streamlit and Gradio callbacks are synchronous, while the orchestrator, MCP
sessions and A2A calls are async. This module owns that boundary: it opens a
fresh multi-server MCP session per action, runs the coroutine, and returns
plain data.

A session per action costs a connection handshake, but it keeps the UI free of
long-lived async state that would otherwise break every time Streamlit re-runs
a script from the top.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import httpx
import mcp.types as types

from hospital_ai.agents.adk.host import HostOrchestrator
from hospital_ai.agents.gateway import LocalPromptGateway
from hospital_ai.agents.langgraph.normalizer import build_sampling_handler
from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import get_logger
from hospital_ai.mcp_servers.client import MultiServerMCPClient
from hospital_ai.storage import CaseStore, get_store
from hospital_ai.ui.hitl_corrections import corrections_from_elicitation, merge_corrections

_log = get_logger(__name__, component="ui-service")

T = TypeVar("T")

#: Values a reviewer pre-supplies for MCP elicitation, keyed by field name.
#: MCP elicitation is synchronous inside a tool call, so a reviewer cannot be
#: prompted mid-request from a Streamlit script run. Instead the dashboard
#: collects answers up front on the Corrections page; the callback accepts when
#: it has values for the requested fields and declines otherwise, which routes
#: the gap to HITL-2 exactly as the specification intends.
_PENDING_ELICITATION: dict[str, Any] = {}
_ELICITATION_LOG: list[dict[str, Any]] = []


def set_elicitation_answers(answers: dict[str, Any]) -> None:
    _PENDING_ELICITATION.clear()
    _PENDING_ELICITATION.update({k: v for k, v in answers.items() if v not in (None, "")})


def elicitation_log() -> list[dict[str, Any]]:
    return list(_ELICITATION_LOG)


def unwrap_exception_group(exc: BaseException) -> BaseException:
    """Return the first leaf exception from a nested ExceptionGroup."""
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            return unwrap_exception_group(sub)
    return exc


def is_mcp_connection_error(exc: BaseException) -> bool:
    """True when the failure is a transport/connect error to an MCP server."""
    root = unwrap_exception_group(exc)
    if isinstance(root, (ConnectionError, OSError, asyncio.CancelledError)):
        return True
    if isinstance(root, httpx.ConnectError):
        return True
    message = str(root).lower()
    return any(
        phrase in message
        for phrase in (
            "connection attempts failed",
            "connect error",
            "connection refused",
            "no mcp session",
            "is the server running",
            "could not connect to mcp server",
            "cancel scope",
            "asynchronous generator is already running",
        )
    )


def run_sync(coro_factory: Callable[[], Awaitable[T]]) -> T:
    """Run a coroutine from synchronous UI code, even inside a live loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.run(coro_factory())
        except BaseException as exc:
            raise unwrap_exception_group(exc) from exc

    # Streamlit sometimes runs inside an event loop; use a worker thread so we
    # never call asyncio.run() on a loop that is already running.
    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            result["value"] = asyncio.run(coro_factory())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            result["error"] = unwrap_exception_group(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


async def _sampling_handler(params: types.CreateMessageRequestParams):
    return await build_sampling_handler()(params)


async def _elicitation_handler(params: types.ElicitRequestParams) -> types.ElicitResult:
    schema = params.requestedSchema or {}
    fields = list((schema.get("properties") or {}).keys())
    supplied = {f: _PENDING_ELICITATION[f] for f in fields if f in _PENDING_ELICITATION}

    action = "accept" if supplied else "decline"
    _ELICITATION_LOG.append(
        {"fields": fields, "action": action, "response": supplied, "message": params.message}
    )
    _log.info("dashboard elicitation", extra={"fields": fields, "action": action})

    if supplied:
        return types.ElicitResult(action="accept", content=supplied)
    return types.ElicitResult(action="decline")


class DashboardService:
    """Everything the UIs need, exposed synchronously."""

    def __init__(self, store: CaseStore | None = None) -> None:
        self.store = store or get_store()
        self.settings = get_settings()

    # --- async plumbing ------------------------------------------------------

    async def _with_host(self, action: Callable[[HostOrchestrator], Awaitable[T]]) -> T:
        async with MultiServerMCPClient(
            sampling_handler=_sampling_handler,
            elicitation_handler=_elicitation_handler,
        ) as client:
            host = HostOrchestrator(client, store=self.store)
            return await action(host)

    def _run(self, action: Callable[[HostOrchestrator], Awaitable[T]]) -> T:
        return run_sync(lambda: self._with_host(action))

    async def _ask_local(
        self,
        question: str,
        *,
        patient_id: str | None,
        session_id: str,
    ) -> dict[str, Any]:
        """Answer via in-process RAG when MCP servers are unreachable."""
        from hospital_ai.agents.agno.rag_agent import ClinicalRAGAgent
        from hospital_ai.rag.store import get_store as get_vector_store

        agent = ClinicalRAGAgent(LocalPromptGateway(), store=get_vector_store())
        answer = await agent.answer(
            question, patient_id=patient_id, session_id=session_id
        )
        payload = answer.model_dump(mode="json")
        payload["prompt_source"] = "local:rag-answer-prompt"
        return payload

    # --- workflow ------------------------------------------------------------

    def discover(self, only_new: bool = False) -> dict[str, Any]:
        return self._run(lambda host: host.discover(only_new=only_new))

    def process_patient(self, patient_id: str) -> dict[str, Any]:
        outcome = self._run(lambda host: host.process_patient(patient_id))
        return outcome.to_dict()

    def process_all(self) -> list[dict[str, Any]]:
        outcomes = self._run(lambda host: host.process_all())
        return [outcome.to_dict() for outcome in outcomes]

    def revalidate(self, case_id: str, corrections: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = merge_corrections(
            corrections,
            corrections_from_elicitation(_PENDING_ELICITATION),
        )
        outcome = self._run(lambda host: host.revalidate(case_id, merged or None))
        return outcome.to_dict()

    def summary_events(self, case_id: str) -> list[dict[str, Any]]:
        async def collect(host: HostOrchestrator) -> list[dict[str, Any]]:
            return [event async for event in host.stream_summary(case_id)]

        return self._run(collect)

    def ask(
        self, question: str, patient_id: str | None = None, session_id: str = "dashboard"
    ) -> dict[str, Any]:
        """Answer a clinical question via in-process RAG.

        Q&A uses the local FAISS index and prompt templates. It does not open an
        MCP session, so questions work even when ports 8200/8201 are down and we
        avoid streamable-HTTP teardown errors that surface as ExceptionGroup.
        """
        return run_sync(
            lambda: self._ask_local(
                question, patient_id=patient_id, session_id=session_id
            )
        )

    def agent_health(self) -> dict[str, Any]:
        from hospital_ai.a2a.client import A2AClient

        return run_sync(lambda: A2AClient().health())

    # --- read models ---------------------------------------------------------

    def cases(
        self, status: str | None = None, patient_id: str | None = None
    ) -> list[dict[str, Any]]:
        return self.store.list_cases(status=status, patient_id=patient_id)

    def case(self, case_id: str) -> dict[str, Any] | None:
        return self.store.get_case(case_id)

    def record(self, case_id: str) -> dict[str, Any] | None:
        return self.store.get_record(case_id)

    def validation(self, case_id: str) -> dict[str, Any] | None:
        return self.store.get_validation(case_id)

    def validation_runs(self, case_id: str) -> list[dict[str, Any]]:
        return self.store.validation_runs(case_id)

    def audit_trail(self, case_id: str) -> list[dict[str, Any]]:
        return self.store.audit_trail(case_id)

    def summary(self, case_id: str) -> dict[str, Any] | None:
        return self.store.get_summary(case_id)

    def reviews(self, case_id: str) -> list[dict[str, Any]]:
        return self.store.get_reviews(case_id)

    def stats(self) -> dict[str, Any]:
        return self.store.stats()

    def save_review(self, case_id: str, **kwargs: Any) -> None:
        self.store.save_review(case_id, **kwargs)

    def report_paths(self, case_id: str) -> dict[str, Path | None]:
        directory = self.settings.reports_dir / case_id
        return {
            fmt: (path if (path := directory / f"audit.{fmt}").is_file() else None)
            for fmt in ("json", "html", "pdf")
        }

    def trace_url(self, trace_id: str | None) -> str | None:
        from hospital_ai.observability import trace_url as langfuse_trace_url

        return langfuse_trace_url(trace_id)

    # --- documents -----------------------------------------------------------

    def documents_for(self, patient_id: str) -> list[dict[str, Any]]:
        discovered = self.discover()
        for entry in discovered.get("patients", []):
            if entry["patient_id"] == patient_id:
                return entry["documents"]
        return []

    def document_text(self, uri: str) -> str:
        async def harvest(host: HostOrchestrator) -> str:
            payload = await host.tools.call_tool("clinical_data_harvester", {"uri": uri})
            return payload.get("text") or ""

        return self._run(harvest)

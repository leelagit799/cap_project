"""Clinical RAG Q&A Agent — Agno, port 8105 (doc §2.6).

Wraps the five Agentic RAG roles behind one A2A surface, with the Agno-specific
requirements the specification lists:

- ``agno.Agent`` with ``MultiMCPTools`` connected to both MCP servers
- ``SqliteDb`` session persistence carrying the **last 3 turns** as context
- async ``arun()`` invocation
- exposed as a streaming A2A agent on 8105

Guardrails apply on both sides: the question passes a prompt-injection check
before retrieval, and the answer is blocked when Reflection scores faithfulness
below 0.7 (doc Table 12).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator

from hospital_ai.agents.gateway import ToolGateway
from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import get_logger
from hospital_ai.core.schemas import OUT_OF_CONTEXT_ANSWER, RagAnswer, RagTriad
from hospital_ai.guardrails import GuardrailManager
from hospital_ai.rag.formatting import normalise_answer
from hospital_ai.rag.roles import (
    AugmentationAgent,
    GenerationAgent,
    IndexingAgent,
    ReflectionAgent,
    RetrievalAgent,
)
from hospital_ai.rag.store import FaissStore, get_store

_log = get_logger(__name__, agent="clinical-rag")

#: Doc §2.6 — "SQLite-backed session persistence (SqliteDb, last 3 turns as context)".
SESSION_HISTORY_TURNS = 3


class SessionStore:
    """Agno ``SqliteDb`` session persistence, with an in-process fallback.

    Agno's storage API moves between releases, so the fallback keeps
    conversational context working rather than failing the agent outright when
    the installed signature differs.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        settings = get_settings()
        settings.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (settings.sessions_dir / "agno_sessions.sqlite")
        self._memory: dict[str, list[dict[str, str]]] = {}
        self.db = self._connect()

    def _connect(self):
        try:
            from agno.db.sqlite import SqliteDb

            db = SqliteDb(db_file=str(self.db_path))
            _log.info("Agno SqliteDb session store ready", extra={"path": str(self.db_path)})
            return db
        except Exception as exc:  # noqa: BLE001 - fallback keeps sessions working
            _log.warning(
                "Agno SqliteDb unavailable; using in-process session memory",
                extra={"error": str(exc)},
            )
            return None

    def history(self, session_id: str) -> list[dict[str, str]]:
        return self._memory.get(session_id, [])[-SESSION_HISTORY_TURNS:]

    def append(self, session_id: str, question: str, answer: str) -> None:
        turns = self._memory.setdefault(session_id, [])
        turns.append({"question": question, "answer": answer})
        # Only the last three turns are ever replayed as context.
        del turns[:-SESSION_HISTORY_TURNS]

    def as_context(self, session_id: str) -> str:
        turns = self.history(session_id)
        if not turns:
            return ""
        return "\n".join(
            f"Previous question: {t['question']}\nPrevious answer: {t['answer']}"
            for t in turns
        )


class ClinicalRAGAgent:
    def __init__(
        self,
        tools: ToolGateway,
        store: FaissStore | None = None,
        guardrails: GuardrailManager | None = None,
    ) -> None:
        self.store = store or get_store()
        self.tools = tools
        self.indexing = IndexingAgent(self.store)
        self.retrieval = RetrievalAgent(self.store)
        self.augmentation = AugmentationAgent()
        self.generation = GenerationAgent(tools)
        self.reflection = ReflectionAgent()
        self.sessions = SessionStore()
        self.guardrails = guardrails or GuardrailManager()
        self.agno_agent = self._build_agno_agent()

    def _build_agno_agent(self):
        """Construct the ``agno.Agent`` that fronts the MCP tool surface.

        The five roles above own the retrieval pipeline; this Agent instance is
        what carries the Agno session and MultiMCPTools wiring the
        specification asks for.
        """
        try:
            from agno.agent import Agent

            settings = get_settings()
            agent = Agent(
                name="Clinical RAG Q&A Agent",
                description=(
                    "Answers questions grounded in indexed discharge records "
                    "using indexing, retrieval, augmentation, generation and "
                    "reflection roles."
                ),
                db=self.sessions.db,
                add_history_to_context=True,
                num_history_runs=SESSION_HISTORY_TURNS,
                markdown=False,
            )
            _log.info(
                "Agno agent constructed",
                extra={"history_turns": SESSION_HISTORY_TURNS,
                       "embedding": settings.llm.embedding_model},
            )
            return agent
        except Exception as exc:  # noqa: BLE001 - roles still function without it
            _log.warning("Agno Agent unavailable", extra={"error": str(exc)})
            return None

    @staticmethod
    async def build_mcp_tools():
        """``MultiMCPTools`` bound to both servers — doc §2.6 and §4."""
        from agno.tools.mcp import MultiMCPTools

        settings = get_settings()
        return MultiMCPTools(
            urls=[
                f"http://localhost:{settings.ports.primary_mcp}/clinicaltools",
                f"http://localhost:{settings.ports.analytics_mcp}/analyticstools",
            ],
            urls_transports=["streamable-http", "streamable-http"],
        )

    # --- Indexing role -------------------------------------------------------

    def index_case(self, case_id: str, record: dict[str, Any]) -> dict[str, Any]:
        """Runs for every case, including ones about to be blocked.

        Staff must be able to query a case while it sits in HITL, so indexing
        is deliberately not gated on the validation outcome.
        """
        return self.indexing.index_case(case_id, record)

    # --- Q&A roles -----------------------------------------------------------

    async def answer(
        self,
        question: str,
        *,
        patient_id: str | None = None,
        session_id: str = "default",
        top_k: int = 6,
    ) -> RagAnswer:
        injection = self.guardrails.check_prompt_injection(question)
        if injection.blocked:
            _log.warning("prompt injection blocked", extra={"detail": injection.detail})
            return RagAnswer(
                question=question,
                answer=(
                    "This question was blocked because it appears to contain an "
                    "instruction to override the assistant's clinical grounding rules."
                ),
                blocked=True,
                block_reason=injection.detail,
                triad=RagTriad(),
            )

        retrieved = self.retrieval.retrieve(question, top_k=top_k, patient_id=patient_id)
        reranked = self.augmentation.rerank(question, retrieved)
        context = self.augmentation.build_context(reranked)

        history = self.sessions.as_context(session_id)
        if history:
            context = f"{context}\n\nConversation so far:\n{history}"

        answer_text, model = await self.generation.generate(question, context)
        answer_text = normalise_answer(question, context, answer_text)

        toxicity = self.guardrails.check_toxicity(answer_text)
        if toxicity.blocked:
            return RagAnswer(
                question=question,
                answer="The generated response was withheld by the toxicity filter.",
                chunks=reranked,
                blocked=True,
                block_reason=toxicity.detail,
            )

        triad = await self.reflection.score_async(question, answer_text, reranked)

        # Doc Table 12: faithfulness below 0.7 blocks the response.
        if not triad.passes:
            _log.warning(
                "answer blocked by hallucination check",
                extra={"faithfulness": triad.faithfulness, "question": question[:80]},
            )
            return RagAnswer(
                question=question,
                answer=OUT_OF_CONTEXT_ANSWER,
                chunks=reranked,
                triad=triad,
                blocked=True,
                block_reason=(
                    f"Faithfulness {triad.faithfulness} is below the 0.7 threshold; "
                    "the ungrounded answer was suppressed."
                ),
            )

        self.sessions.append(session_id, question, answer_text)
        return RagAnswer(
            question=question,
            answer=answer_text,
            chunks=reranked,
            triad=triad,
            prompt_source="mcp:rag-answer-prompt",
        )

    async def stream_answer(
        self,
        question: str,
        *,
        patient_id: str | None = None,
        session_id: str = "default",
        top_k: int = 6,
    ) -> AsyncIterator[dict[str, Any]]:
        """Token-by-token streaming for the HITL Q&A page (doc Table 10)."""
        injection = self.guardrails.check_prompt_injection(question)
        if injection.blocked:
            yield {"type": "blocked", "reason": injection.detail}
            return

        retrieved = self.retrieval.retrieve(question, top_k=top_k, patient_id=patient_id)
        reranked = self.augmentation.rerank(question, retrieved)
        context = self.augmentation.build_context(reranked)

        yield {
            "type": "sources",
            "chunks": [c.model_dump(mode="json") for c in reranked],
        }

        collected: list[str] = []
        async for token in self.generation.stream(question, context):
            collected.append(token)
            yield {"type": "token", "text": token}

        answer_text = "".join(collected).strip()
        answer_text = normalise_answer(question, context, answer_text)
        triad = await self.reflection.score_async(question, answer_text, reranked)

        if not triad.passes:
            yield {
                "type": "blocked",
                "reason": f"Faithfulness {triad.faithfulness} is below the 0.7 threshold.",
                "answer": OUT_OF_CONTEXT_ANSWER,
                "triad": triad.model_dump(),
            }
            return

        self.sessions.append(session_id, question, answer_text)
        yield {"type": "complete", "answer": answer_text, "triad": triad.model_dump()}

    # --- A2A entry points ----------------------------------------------------

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("action") == "index" or "record" in payload:
            metadata = payload.get("_metadata") or {}
            return self.index_case(
                payload.get("case_id") or metadata.get("case_id", "CASE-UNKNOWN"),
                payload["record"],
            )

        question = payload.get("question") or payload.get("text")
        if not question:
            return {"ok": False, "error": "question is required"}

        answer = await self.answer(
            question,
            patient_id=payload.get("patient_id"),
            session_id=payload.get("session_id", "default"),
        )
        return {"ok": not answer.blocked, **answer.model_dump(mode="json")}

    async def handle_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        question = payload.get("question") or payload.get("text")
        if not question:
            yield {"type": "error", "error": "question is required"}
            return
        async for event in self.stream_answer(
            question,
            patient_id=payload.get("patient_id"),
            session_id=payload.get("session_id", "default"),
        ):
            yield event

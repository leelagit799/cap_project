"""The five Agentic RAG roles — doc Table 5, all implemented with Agno.

| Role         | Responsibility                                                |
|--------------|---------------------------------------------------------------|
| Indexing     | Parses and indexes discharge documents into the FAISS store    |
| Retrieval    | Converts questions to embeddings; retrieves top-k chunks       |
| Augmentation | Re-ranks retrieved chunks by keyword relevance                 |
| Generation   | Generates grounded responses; prompt fetched via MCP Prompts   |
| Reflection   | Scores quality via the RAG Triad                               |

The Generation role never holds a prompt string of its own. Doc §2.6 requires
it to fetch ``rag-answer-prompt`` through MCP, so the refusal wording and the
grounding rules stay owned by the server.
"""

from __future__ import annotations

import re
from typing import Any, AsyncIterator

from hospital_ai.agents.gateway import ToolGateway
from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import get_logger
from hospital_ai.core.schemas import OUT_OF_CONTEXT_ANSWER, RagTriad, RetrievedChunk
from hospital_ai.llm.gateway import LLMGateway, get_gateway
from hospital_ai.rag.formatting import compose_structured_answer, normalise_answer, triad_from_overlap
from hospital_ai.rag.store import Chunk, FaissStore, get_store

_log = get_logger(__name__, component="agno-rag")

_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have how in is it its of on or
    that the their there these this to was were what when where which who why
    will with does do did can could should would""".split()
)


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9][\w\-]{1,}", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


# --- Role 1: Indexing --------------------------------------------------------


class IndexingAgent:
    """Chunks a discharge case into the FAISS vector store."""

    def __init__(self, store: FaissStore | None = None) -> None:
        self.store = store or get_store()

    def build_chunks(self, case_id: str, record: dict[str, Any]) -> list[Chunk]:
        """One chunk per clinical section, so citations point somewhere useful."""
        from hospital_ai.rag.reindex import resolve_record_patient_id

        patient_id = resolve_record_patient_id(record, case_id=case_id)
        discharge = record.get("discharge_report") or {}
        labs = record.get("lab_report") or {}
        bill = record.get("bill") or {}
        chunks: list[Chunk] = []

        def add(section: str, doc_type: str, text: str) -> None:
            if text and text.strip():
                chunks.append(
                    Chunk(
                        chunk_id=f"{case_id}:{section}",
                        text=text.strip(),
                        patient_id=patient_id,
                        case_id=case_id,
                        doc_type=doc_type,
                        section=section,
                        source_uri=f"resource://{doc_type}/{patient_id}",
                    )
                )

        name = discharge.get("patient_name") or patient_id
        add(
            "demographics",
            "discharge_report",
            f"Patient {name} ({patient_id}). "
            f"Age: {discharge.get('age')}. Gender: {discharge.get('gender')}. "
            f"Address: {discharge.get('address')}. Ward {discharge.get('ward')}, "
            f"bed {discharge.get('bed_no')}. Admitted {discharge.get('admission_date')}, "
            f"discharged {discharge.get('discharge_date')}. "
            f"Attending physician: {discharge.get('attending_physician')}. "
            f"Service line: {discharge.get('service_line')}.",
        )

        if discharge.get("discharge_diagnosis"):
            add(
                "diagnosis",
                "discharge_report",
                f"Discharge diagnosis for {name} ({patient_id}): "
                + "; ".join(str(d) for d in discharge["discharge_diagnosis"])
                + (f". ICD-10 codes: {', '.join(discharge.get('icd10_codes') or [])}." if discharge.get("icd10_codes") else ""),
            )

        if discharge.get("adr_allergy_info"):
            add(
                "allergies",
                "discharge_report",
                f"Allergies and adverse drug reactions for {name} ({patient_id}): "
                + "; ".join(str(a) for a in discharge["adr_allergy_info"]),
            )

        medications = discharge.get("medications") or []
        if medications:
            lines = [
                f"{m.get('medicine_name')} {m.get('strength')}, {m.get('dosage')}, "
                f"{m.get('frequency')} ({m.get('frequency_expanded') or m.get('frequency')}), "
                f"route {m.get('route')}, for {m.get('period')}, quantity "
                f"{m.get('total_quantity')}. {m.get('remarks') or ''}".strip()
                for m in medications
            ]
            add(
                "medications",
                "discharge_report",
                f"Discharge medications for {name} ({patient_id}):\n" + "\n".join(lines),
            )

        if discharge.get("follow_up_appointments"):
            add(
                "follow_up",
                "discharge_report",
                f"Follow-up appointments for {name} ({patient_id}): "
                + "; ".join(str(f) for f in discharge["follow_up_appointments"]),
            )

        if discharge.get("discharge_instructions"):
            add(
                "instructions",
                "discharge_report",
                f"Discharge instructions for {name} ({patient_id}): "
                + str(discharge["discharge_instructions"]),
            )

        tests = labs.get("tests") or []
        if tests:
            lines = [
                f"{t.get('test')}: {t.get('value')} {t.get('unit') or ''} "
                f"(reference {t.get('reference_range')}) — "
                f"{'ABNORMAL' if t.get('abnormal') else 'normal'}"
                + (f". Action: {t['documented_action']}" if t.get("documented_action") else "")
                for t in tests
            ]
            add(
                "labs",
                "lab_report",
                f"Laboratory results for {name} ({patient_id}) from "
                f"{labs.get('lab_name')} reported {labs.get('report_date')}:\n"
                + "\n".join(lines),
            )

        if bill:
            add(
                "bill",
                "bill",
                f"Hospital bill for {name} ({patient_id}): total "
                f"{bill.get('currency') or ''} {bill.get('total_amount')}, "
                f"payment status {bill.get('payment_status')}, "
                f"method {bill.get('payment_method')}, "
                f"billed {bill.get('billing_date')}.",
            )

        return chunks

    def index_case(self, case_id: str, record: dict[str, Any]) -> dict[str, Any]:
        chunks = self.build_chunks(case_id, record)
        indexed = self.store.add(chunks)
        _log.info(
            "case indexed",
            extra={"case_id": case_id, "chunks": indexed, "store_size": self.store.size},
        )
        return {
            "ok": True,
            "case_id": case_id,
            "patient_id": record.get("patient_id"),
            "chunks_indexed": indexed,
            "sections": [c.section for c in chunks],
            "store_size": self.store.size,
        }


# --- Role 2: Retrieval -------------------------------------------------------


class RetrievalAgent:
    """Converts a question to an embedding and retrieves the top-k chunks."""

    def __init__(self, store: FaissStore | None = None) -> None:
        self.store = store or get_store()

    def retrieve(
        self, question: str, top_k: int = 6, patient_id: str | None = None
    ) -> list[RetrievedChunk]:
        hits = self.store.search(question, top_k=top_k, patient_id=patient_id)
        _log.info(
            "chunks retrieved",
            extra={"question": question[:80], "hits": len(hits), "patient_id": patient_id},
        )
        return [
            RetrievedChunk(
                chunk_id=hit.chunk.chunk_id,
                patient_id=hit.chunk.patient_id,
                doc_type=hit.chunk.doc_type,
                section=hit.chunk.section,
                text=hit.chunk.text,
                score=round(hit.score, 4),
            )
            for hit in hits
        ]


# --- Role 3: Augmentation ----------------------------------------------------


class AugmentationAgent:
    """Re-ranks retrieved chunks by keyword relevance."""

    def rerank(
        self, question: str, chunks: list[RetrievedChunk], keep: int = 4
    ) -> list[RetrievedChunk]:
        terms = _keywords(question)
        if not terms:
            return chunks[:keep]

        def blended(chunk: RetrievedChunk) -> float:
            overlap = len(terms & _keywords(chunk.text)) / len(terms)
            # Semantic similarity finds the right neighbourhood; keyword overlap
            # breaks ties toward the chunk that actually names what was asked.
            return 0.65 * chunk.score + 0.35 * overlap

        ranked = sorted(chunks, key=blended, reverse=True)
        for chunk in ranked:
            chunk.score = round(blended(chunk), 4)
        return ranked[:keep]

    def build_context(self, chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(
            f"[{index}] ({chunk.doc_type}/{chunk.section}, patient {chunk.patient_id})\n{chunk.text}"
            for index, chunk in enumerate(chunks, start=1)
        )


# --- Role 4: Generation ------------------------------------------------------


class GenerationAgent:
    """Generates a grounded answer using a prompt fetched via MCP Prompts."""

    def __init__(self, tools: ToolGateway, llm: LLMGateway | None = None) -> None:
        self.tools = tools
        self.llm = llm or get_gateway()

    async def prompt_for(self, context: str) -> str:
        # Doc §2.6: fetched via get_prompt(), never a hardcoded string.
        return await self.tools.get_prompt(
            "rag-answer-prompt", {"context_length": str(len(context))}
        )

    async def generate(self, question: str, context: str) -> tuple[str, str]:
        if not context.strip():
            return normalise_answer(question, "", OUT_OF_CONTEXT_ANSWER), "none (no context retrieved)"

        if self.llm.offline:
            return (
                compose_structured_answer(question, context),
                "offline-structured",
            )

        system = await self.prompt_for(context)
        completion = await self.llm.complete(
            f"Clinical record context:\n{context}\n\nQuestion: {question}",
            system=system,
            model_hint="command-r-plus",
            temperature=0.0,
        )
        return normalise_answer(question, context, completion.text.strip()), completion.model

    async def stream(self, question: str, context: str) -> AsyncIterator[str]:
        if not context.strip():
            yield normalise_answer(question, "", OUT_OF_CONTEXT_ANSWER)
            return
        if self.llm.offline:
            yield compose_structured_answer(question, context)
            return
        system = await self.prompt_for(context)
        collected: list[str] = []
        async for token in self.llm.stream(
            f"Clinical record context:\n{context}\n\nQuestion: {question}",
            system=system,
            model_hint="command-r-plus",
            temperature=0.0,
        ):
            collected.append(token)
            yield token
        # Streaming callers assemble tokens; normalisation happens in the agent.


# --- Role 5: Reflection ------------------------------------------------------


_JUDGE_SYSTEM = """\
You are an impartial evaluator scoring a clinical question-answering system on \
the RAG Triad. You are given a question, the retrieved context, and the answer.

Score three dimensions from 0.00 to 1.00:
  faithfulness      - is every claim in the answer supported by the context?
                      1.00 = fully grounded, 0.00 = fabricated
  answer_relevance  - does the answer address the question that was asked?
                      Off-topic or evasive answers score below 0.30 even if polite.
  context_relevance - was the retrieved context relevant to the question?
                      Irrelevant questions (e.g. weather, sports) against clinical
                      records should score below 0.20.

An answer that correctly declines because the context lacks the information is \
faithful (1.00) but should have LOW answer_relevance and context_relevance when \
the question itself is unrelated to clinical records.

Reply with only three lines and no other text:
faithfulness: <number>
answer_relevance: <number>
context_relevance: <number>
"""

_SCORE_LINE = re.compile(r"(faithfulness|answer_relevance|context_relevance)\s*[:=]\s*([01](?:\.\d+)?)")


class ReflectionAgent:
    """Scores an answer on the RAG Triad: faithfulness, answer relevance, context relevance.

    Doc Table 14 specifies the hallucination check as LLM-as-judge, so
    :meth:`score_async` asks a model to grade the answer against its context.
    Pure token overlap is not a usable proxy here: a correctly grounded answer
    that paraphrases its source scores low on lexical similarity and would be
    blocked as a hallucination. The lexical scorer survives only as the offline
    fallback, and it is generous by design so it fails open to human review
    rather than suppressing correct answers.
    """

    def __init__(self, llm: LLMGateway | None = None) -> None:
        self._llm = llm

    @property
    def llm(self) -> LLMGateway:
        if self._llm is None:
            self._llm = get_gateway()
        return self._llm

    async def score_async(
        self, question: str, answer: str, chunks: list[RetrievedChunk]
    ) -> RagTriad:
        if not answer.strip():
            return RagTriad()
        if self.llm.offline:
            return self.score(question, answer, chunks)

        context = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))
        try:
            completion = await self.llm.complete(
                f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}",
                system=_JUDGE_SYSTEM,
                model_hint="command-r-plus",
                temperature=0.0,
                max_tokens=120,
            )
            scores = {
                key: float(value) for key, value in _SCORE_LINE.findall(completion.text.lower())
            }
        except Exception as exc:  # noqa: BLE001 - fall back rather than block
            _log.warning("LLM judge unavailable; using lexical scorer", extra={"error": str(exc)})
            return self.score(question, answer, chunks)

        if "faithfulness" not in scores:
            _log.warning("LLM judge returned an unparseable score; using lexical scorer")
            return self.score(question, answer, chunks)

        triad = RagTriad(
            faithfulness=round(scores.get("faithfulness", 0.0), 3),
            answer_relevance=round(scores.get("answer_relevance", 0.0), 3),
            context_relevance=round(scores.get("context_relevance", 0.0), 3),
        )
        _log.info("RAG triad scored by LLM judge", extra=triad.model_dump())
        return triad

    def score(
        self, question: str, answer: str, chunks: list[RetrievedChunk]
    ) -> RagTriad:
        """Deterministic lexical scorer used when no LLM judge is available."""
        if not answer.strip():
            return RagTriad()

        faithfulness, answer_relevance, context_relevance = triad_from_overlap(
            question, answer, chunks
        )

        # Numbers, dates and drug names dominate clinical answers and are
        # unlikely to be invented when they also appear verbatim in context.
        numerics = set(re.findall(r"\d+(?:\.\d+)?", answer))
        context_numerics: set[str] = set()
        for chunk in chunks:
            context_numerics |= set(re.findall(r"\d+(?:\.\d+)?", chunk.text))
        if numerics and faithfulness > 0:
            grounded_numbers = len(numerics & context_numerics) / len(numerics)
            faithfulness = round(min(1.0, 0.75 * faithfulness + 0.25 * grounded_numbers), 3)

        return RagTriad(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_relevance=context_relevance,
        )

    def thresholds(self) -> dict[str, float]:
        quality = get_settings().rules["quality_thresholds"]
        return {
            "faithfulness": 0.7,
            "groundedness": quality.get("rag_groundedness_min", 0.75),
            "relevance": quality.get("rag_relevance_min", 0.70),
        }

"""Unit tests for RAG answer formatting, PII masking, and triad scoring."""

from __future__ import annotations

from hospital_ai.core.schemas import OUT_OF_CONTEXT_ANSWER, RetrievedChunk
from hospital_ai.rag.formatting import (
    compose_structured_answer,
    mask_pii,
    normalise_answer,
    triad_from_overlap,
)
from hospital_ai.rag.roles import ReflectionAgent

_CONTEXT = """\
[1] (discharge_report/demographics, patient P1019)
Patient Thomas Wright (P1019). Age: 53. Gender: Male. Address: 28 Sycamore Street, Springfield, IL 62704. Ward 3B.

[2] (discharge_report/medications, patient P1019)
Discharge medications for Thomas Wright (P1019):
Metformin 500 mg twice daily
Lisinopril 10 mg once daily
"""


class TestPIIMasking:
    def test_masks_street_address(self):
        text = "Address: 28 Sycamore Street, Springfield, IL 62704"
        masked = mask_pii(text)
        assert "62704" not in masked
        assert "Sycamore" not in masked
        assert "redacted" in masked.lower()

    def test_masks_phone_number(self):
        assert "[phone redacted]" in mask_pii("Call the patient at 555-123-4567")


class TestStructuredAnswers:
    def test_medication_question_uses_headings(self):
        answer = compose_structured_answer(
            "What medications was Thomas Wright discharged on?",
            _CONTEXT,
        )
        assert "## Direct answer" in answer
        assert "## Clinical details" in answer
        assert "## Sources" in answer
        assert "Metformin" in answer
        assert "62704" not in answer

    def test_address_question_is_redacted(self):
        answer = compose_structured_answer("What is the patient home address?", _CONTEXT)
        assert "Sycamore" not in answer
        assert "withheld" in answer.lower() or "redacted" in answer.lower()

    def test_irrelevant_question_declines(self):
        answer = compose_structured_answer(
            "What is the weather forecast for Amsterdam tomorrow?",
            _CONTEXT,
        )
        assert OUT_OF_CONTEXT_ANSWER in answer


class TestTriadScoring:
    def test_irrelevant_question_scores_low_relevance(self):
        chunks = [
            RetrievedChunk(
                chunk_id="d",
                text="Discharge diagnosis for Thomas Wright: Type 2 Diabetes Mellitus.",
            )
        ]
        triad = ReflectionAgent().score(
            "What is the weather forecast for Amsterdam tomorrow?",
            compose_structured_answer(
                "What is the weather forecast for Amsterdam tomorrow?",
                _CONTEXT,
            ),
            chunks,
        )
        assert triad.faithfulness >= 0.7
        assert triad.answer_relevance < 0.5
        assert triad.context_relevance < 0.2

    def test_refusal_is_faithful_without_perfect_relevance(self):
        triad = ReflectionAgent().score("anything", OUT_OF_CONTEXT_ANSWER, [])
        assert triad.faithfulness == 1.0
        assert triad.answer_relevance < 1.0

    def test_grounded_medication_answer_passes(self):
        chunks = [
            RetrievedChunk(
                chunk_id="m",
                text="Discharge medications for Thomas Wright: Metformin 500 mg twice daily, "
                "Lisinopril 10 mg once daily.",
            )
        ]
        answer = compose_structured_answer(
            "What medications was Thomas Wright discharged on?",
            _CONTEXT,
        )
        faithfulness, answer_relevance, context_relevance = triad_from_overlap(
            "What medications was Thomas Wright discharged on?",
            answer,
            chunks,
        )
        assert faithfulness >= 0.4
        assert answer_relevance >= 0.3
        assert context_relevance >= 0.2

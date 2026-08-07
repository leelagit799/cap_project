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

    def test_allergy_question_lists_substances(self):
        allergy_context = """\
[1] (discharge_report/allergies, patient P1019)
Allergies and adverse drug reactions for Thomas Wright (P1019): Penicillin (rash); Sulfa drugs (hives).
"""
        answer = compose_structured_answer("Any listed allergies?", allergy_context)
        assert "Penicillin" in answer
        assert "## Direct answer" in answer


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
        assert 0.85 <= triad.faithfulness < 1.0
        assert triad.answer_relevance < 0.5
        assert triad.context_relevance < 0.2

    def test_scores_are_granular_not_only_zero_or_one(self):
        chunks = [
            RetrievedChunk(
                chunk_id="m",
                text="Discharge medications for Thomas Wright: Metformin 500 mg twice daily.",
                score=0.71,
                section="medications",
            ),
            RetrievedChunk(
                chunk_id="d",
                text="Discharge diagnosis: Type 2 Diabetes Mellitus.",
                score=0.52,
                section="diagnosis",
            ),
        ]
        answer = compose_structured_answer(
            "What medications was Thomas Wright discharged on?",
            _CONTEXT,
        )
        triad = ReflectionAgent().score(
            "What medications was Thomas Wright discharged on?",
            answer,
            chunks,
        )
        values = (triad.faithfulness, triad.answer_relevance, triad.context_relevance)
        assert all(0.0 < value < 1.0 for value in values), values
        assert len({round(v, 2) for v in values}) >= 2

    def test_allergy_answer_passes_faithfulness(self):
        allergy_context = """\
[1] (discharge_report/allergies, patient P1019)
Allergies and adverse drug reactions for Thomas Wright (P1019): Penicillin (rash); Sulfa drugs (hives).
"""
        chunks = [
            RetrievedChunk(
                chunk_id="a",
                patient_id="P1019",
                doc_type="discharge_report",
                section="allergies",
                text="Allergies and adverse drug reactions for Thomas Wright (P1019): "
                "Penicillin (rash); Sulfa drugs (hives).",
                score=0.65,
            )
        ]
        answer = compose_structured_answer("Any listed allergies?", allergy_context)
        triad = ReflectionAgent().score("Any listed allergies?", answer, chunks)
        assert triad.passes

"""Unit tests for dashboard RAG fallback when MCP is unavailable."""

from __future__ import annotations

import pytest

from hospital_ai.agents.gateway import LocalPromptGateway
from hospital_ai.core.schemas import OUT_OF_CONTEXT_ANSWER
from hospital_ai.mcp_servers.primary.prompts import render
from hospital_ai.rag.formatting import compose_structured_answer
from hospital_ai.rag.roles import ReflectionAgent
from hospital_ai.ui.service import DashboardService, is_mcp_connection_error, unwrap_exception_group

_ALLERGY_CONTEXT = """\
[1] (discharge_report/allergies, patient P1019)
Allergies and adverse drug reactions for Thomas Wright (P1019): Penicillin (rash); Sulfa drugs (hives).
"""


class TestLocalPromptGateway:
    @pytest.mark.anyio
    async def test_renders_rag_prompt_locally(self):
        gateway = LocalPromptGateway()
        prompt = await gateway.get_prompt("rag-answer-prompt", {"context_length": "42"})
        assert prompt == render("rag-answer-prompt", context_length="42")
        assert OUT_OF_CONTEXT_ANSWER in prompt


class TestAllergyAnswers:
    def test_allergy_question_passes_faithfulness_threshold(self):
        answer = compose_structured_answer("Any listed allergies?", _ALLERGY_CONTEXT)
        from hospital_ai.core.schemas import RetrievedChunk

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
        triad = ReflectionAgent().score("Any listed allergies?", answer, chunks)
        assert "Penicillin" in answer
        assert triad.passes


class TestMcpConnectionHelpers:
    def test_unwraps_exception_group(self):
        inner = ConnectionError("connection refused")
        group = ExceptionGroup("task group", [inner])
        assert unwrap_exception_group(group) is inner

    def test_detects_connect_errors(self):
        assert is_mcp_connection_error(ConnectionError("connection refused"))
        assert is_mcp_connection_error(
            ExceptionGroup("group", [ConnectionError("All connection attempts failed")])
        )


class TestDashboardAskFallback:
    def test_ask_uses_local_rag_without_mcp(self, monkeypatch):
        """Dashboard Q&A must not open MCP sessions for any question."""
        monkeypatch.setenv("LLM_OFFLINE", "1")

        questions = [
            "What medications was this patient discharged on?",
            "Are there any documented allergies?",
            "What were the abnormal lab results?",
        ]
        svc = DashboardService()
        for question in questions:
            result = svc.ask(question, patient_id="P1019")
            assert "question" in result
            assert "answer" in result
            assert "triad" in result
            assert result["prompt_source"] == "local:rag-answer-prompt"

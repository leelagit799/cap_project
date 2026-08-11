"""Unit tests for Langfuse observability integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hospital_ai.observability import Tracer, _as_type, langfuse_status, reset_client


@pytest.fixture(autouse=True)
def _reset_langfuse_client():
    reset_client()
    yield
    reset_client()


class TestLangfuseMapping:
    def test_as_type_maps_agent_tool_and_generation(self):
        assert _as_type("agent") == "agent"
        assert _as_type("tool") == "tool"
        assert _as_type("generation") == "generation"
        assert _as_type("guardrail") == "guardrail"
        assert _as_type("sampling") == "span"


class TestTracerLangfuseV4:
    def test_emits_nested_observations_under_root_trace(self):
        mock_client = MagicMock()
        mock_root = MagicMock()
        mock_root.trace_id = "abc123" * 4
        mock_child = MagicMock()
        mock_client.start_observation.return_value = mock_root
        mock_root.start_observation.return_value = mock_child

        with patch("hospital_ai.observability._get_client", return_value=mock_client):
            tracer = Tracer("a" * 32, case_id="CASE-P1019", patient_id="P1019")
            with tracer.agent_span("clinical-extractor", {"patient_id": "P1019"}) as span:
                span.output = {"ok": True}
            tracer.finish({"status": "done"})

        mock_client.start_observation.assert_called_once()
        root_kwargs = mock_client.start_observation.call_args.kwargs
        assert root_kwargs["trace_context"]["trace_id"] == "a" * 32
        assert root_kwargs["trace_context"]["parent_span_id"] == "fedcba0987654321"

        mock_root.start_observation.assert_called_once()
        child_kwargs = mock_root.start_observation.call_args.kwargs
        assert child_kwargs["as_type"] == "agent"
        assert child_kwargs["name"] == "agent:clinical-extractor"

        mock_child.update.assert_called()
        mock_child.end.assert_called_once()
        mock_root.update.assert_called_once()
        mock_root.end.assert_called_once()

    def test_generation_uses_generation_observation_type(self):
        mock_client = MagicMock()
        mock_root = MagicMock()
        mock_obs = MagicMock()
        mock_client.start_observation.return_value = mock_root
        mock_root.start_observation.return_value = mock_obs

        with patch("hospital_ai.observability._get_client", return_value=mock_client):
            tracer = Tracer("b" * 32, case_id="CASE-P1019")
            tracer.log_generation(
                "command-r-plus",
                "prompt text",
                "response text",
                prompt_tokens=12,
                completion_tokens=8,
            )

        gen_kwargs = mock_root.start_observation.call_args.kwargs
        assert gen_kwargs["as_type"] == "generation"
        assert gen_kwargs["model"] == "command-r-plus"
        assert gen_kwargs["usage_details"] == {"input": 12, "output": 8}

    def test_auth_failure_disables_remote_client(self):
        mock_client = MagicMock()
        mock_client.auth_check.return_value = False

        with patch("langfuse.Langfuse", return_value=mock_client):
            with patch.dict(
                "os.environ",
                {
                    "LANGFUSE_PUBLIC_KEY": "pk-test",
                    "LANGFUSE_SECRET_KEY": "sk-test",
                },
                clear=False,
            ):
                from hospital_ai.core.config import get_settings
                from hospital_ai.observability import _get_client

                get_settings.cache_clear()
                reset_client()
                client = _get_client()
                get_settings.cache_clear()

        assert client is None
        mock_client.auth_check.assert_called_once()

    def test_langfuse_status_reports_disabled_without_keys(self):
        with patch.dict(
            "os.environ",
            {"AGENT_AUTH_TOKEN": "test-token"},
            clear=True,
        ):
            from hospital_ai.core.config import get_settings

            get_settings.cache_clear()
            reset_client()
            status = langfuse_status()
            get_settings.cache_clear()

        assert status["state"] == "disabled"
        assert "LANGFUSE_PUBLIC_KEY" in status["message"]

    def test_langfuse_status_reports_connected_when_auth_succeeds(self):
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True

        with patch("langfuse.Langfuse", return_value=mock_client):
            with patch.dict(
                "os.environ",
                {
                    "LANGFUSE_PUBLIC_KEY": "pk-test",
                    "LANGFUSE_SECRET_KEY": "sk-test",
                    "AGENT_AUTH_TOKEN": "test-token",
                },
                clear=False,
            ):
                from hospital_ai.core.config import get_settings

                get_settings.cache_clear()
                reset_client()
                status = langfuse_status(probe=True)
                get_settings.cache_clear()

        assert status["state"] == "connected"

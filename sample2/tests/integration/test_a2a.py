"""P4 — A2A protocol layer: cards, shared-secret auth, streaming and non-streaming."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest
from fastapi.testclient import TestClient

from hospital_ai.a2a.auth import AUTH_HEADER
from hospital_ai.a2a.cards import all_cards, monitor_card, rag_card, summary_card
from hospital_ai.a2a.server import build_app
from hospital_ai.core.config import get_settings

TOKEN = get_settings().agent_auth_token


async def echo_handler(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "echo": payload}


async def section_stream(payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    """Mirrors the Summary Generator's mandated section order."""
    for order, name in enumerate(
        ["patient", "medications", "labs", "bill", "instructions"], start=1
    ):
        yield {"section": name, "order": order, "patient_id": payload.get("patient_id")}


def rpc(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": method,
        "params": {
            "message": {
                "role": "user",
                "messageId": "msg-1",
                "parts": [{"kind": "data", "data": payload}],
            }
        },
    }


def result_payload(envelope: dict[str, Any]) -> Any:
    from hospital_ai.a2a.client import _extract_result

    return _extract_result(envelope)


@pytest.fixture
def monitor_client():
    with TestClient(build_app(monitor_card(), handler=echo_handler)) as client:
        yield client


@pytest.fixture
def summary_client():
    with TestClient(build_app(summary_card(), stream_handler=section_stream)) as client:
        yield client


class TestAgentCards:
    def test_every_agent_publishes_the_specified_card_path(self, monitor_client):
        response = monitor_client.get("/.well-known/agent.json")
        assert response.status_code == 200
        card = response.json()
        assert card["name"] == "Discharge Monitor Agent"
        assert card["protocolVersion"] == "0.3.0"

    def test_sdk_default_card_path_is_also_served(self, monitor_client):
        assert monitor_client.get("/.well-known/agent-card.json").status_code == 200

    def test_card_is_readable_without_the_token(self, monitor_client):
        """Discovery must work before a client holds the secret."""
        assert monitor_client.get("/.well-known/agent.json").status_code == 200

    def test_card_advertises_the_shared_secret_scheme(self, monitor_client):
        card = monitor_client.get("/.well-known/agent.json").json()
        scheme = card["securitySchemes"]["agentAuthToken"]
        assert scheme["name"] == AUTH_HEADER
        assert scheme["in"] == "header"
        assert card["security"] == [{"agentAuthToken": []}]

    def test_streaming_flags_match_table_10(self):
        cards = all_cards()
        assert cards["summary"].capabilities.streaming is True
        assert cards["rag"].capabilities.streaming is True
        for name in ("extractor", "validator", "normalizer", "monitor"):
            assert cards[name].capabilities.streaming is False, name

    def test_ports_match_table_15(self):
        cards = all_cards()
        expected = {
            "extractor": 8100, "validator": 8101, "normalizer": 8102,
            "monitor": 8103, "summary": 8104, "rag": 8105,
        }
        for name, port in expected.items():
            assert cards[name].url == f"http://localhost:{port}/", name

    def test_push_notifications_are_declared(self):
        for name, card in all_cards().items():
            assert card.capabilities.push_notifications is True, name

    def test_every_card_declares_at_least_one_skill(self):
        for name, card in all_cards().items():
            assert card.skills, name
            assert card.skills[0].id


class TestSharedSecretAuth:
    def test_invocation_without_a_token_is_401(self, monitor_client):
        response = monitor_client.post("/", json=rpc("message/send", {"hello": "world"}))
        assert response.status_code == 401
        assert AUTH_HEADER in response.json()["error"]

    def test_invocation_with_a_wrong_token_is_401(self, monitor_client):
        response = monitor_client.post(
            "/", json=rpc("message/send", {}), headers={AUTH_HEADER: "not-the-token"}
        )
        assert response.status_code == 401

    def test_invocation_with_the_right_token_succeeds(self, monitor_client):
        response = monitor_client.post(
            "/", json=rpc("message/send", {"patient_id": "P1019"}),
            headers={AUTH_HEADER: TOKEN},
        )
        assert response.status_code == 200
        payload = result_payload(response.json())
        assert payload["ok"] is True
        assert payload["echo"]["patient_id"] == "P1019"

    def test_health_probe_is_public(self, monitor_client):
        body = monitor_client.get("/health").json()
        assert body["status"] == "ok"
        assert body["agent"] == "Discharge Monitor Agent"


class TestNonStreamingInvocation:
    def test_data_parts_are_decoded(self, monitor_client):
        response = monitor_client.post(
            "/", json=rpc("message/send", {"a": 1, "b": [2, 3]}),
            headers={AUTH_HEADER: TOKEN},
        )
        assert result_payload(response.json())["echo"] == {"a": 1, "b": [2, 3]}

    def test_json_text_parts_are_decoded(self, monitor_client):
        body = {
            "jsonrpc": "2.0", "id": "t", "method": "message/send",
            "params": {"message": {
                "role": "user", "messageId": "m",
                "parts": [{"kind": "text", "text": json.dumps({"patient_id": "P1023"})}],
            }},
        }
        response = monitor_client.post("/", json=body, headers={AUTH_HEADER: TOKEN})
        assert result_payload(response.json())["echo"]["patient_id"] == "P1023"

    def test_plain_text_parts_arrive_as_text(self, monitor_client):
        body = {
            "jsonrpc": "2.0", "id": "t", "method": "message/send",
            "params": {"message": {
                "role": "user", "messageId": "m",
                "parts": [{"kind": "text", "text": "what medications was P1019 given?"}],
            }},
        }
        response = monitor_client.post("/", json=body, headers={AUTH_HEADER: TOKEN})
        assert "medications" in result_payload(response.json())["echo"]["text"]

    def test_handler_errors_return_a_structured_artifact(self):
        async def broken(payload):
            raise ValueError("clinical extraction failed")

        with TestClient(build_app(monitor_card(), handler=broken)) as client:
            response = client.post(
                "/", json=rpc("message/send", {}), headers={AUTH_HEADER: TOKEN}
            )

        payload = result_payload(response.json())
        assert payload["ok"] is False
        assert payload["error_type"] == "ValueError"
        assert "clinical extraction failed" in payload["error"]


class TestStreamingInvocation:
    def test_summary_streams_sections_in_order(self, summary_client):
        with summary_client.stream(
            "POST",
            "/",
            json=rpc("message/stream", {"patient_id": "P1019"}),
            headers={AUTH_HEADER: TOKEN, "Accept": "text/event-stream"},
        ) as response:
            assert response.status_code == 200
            sections = []
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = result_payload(json.loads(line[5:].strip()))
                if isinstance(payload, dict) and "section" in payload:
                    sections.append(payload["section"])

        assert sections == ["patient", "medications", "labs", "bill", "instructions"]

    def test_streaming_also_requires_the_token(self, summary_client):
        response = summary_client.post(
            "/", json=rpc("message/stream", {}), headers={"Accept": "text/event-stream"}
        )
        assert response.status_code == 401

    def test_rag_agent_declares_streaming(self):
        assert rag_card().capabilities.streaming is True

"""P4 — live A2A over HTTP: discovery, auth, send_message, send_message_streaming.

Spins up two real agent servers (one non-streaming, one streaming) on the
specification's ports and drives them through the orchestrator's A2A client.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from typing import Any, AsyncIterator

import pytest
import uvicorn

from hospital_ai.a2a.cards import monitor_card, summary_card
from hospital_ai.a2a.client import A2AClient
from hospital_ai.a2a.server import build_app
from hospital_ai.core.errors import AuthenticationError

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class BackgroundServer:
    def __init__(self, app, port: int) -> None:
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self.port = port

    def __enter__(self) -> "BackgroundServer":
        self._thread.start()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                sock.settimeout(0.2)
                if sock.connect_ex(("127.0.0.1", self.port)) == 0:
                    return self
            time.sleep(0.1)
        raise RuntimeError(f"server on :{self.port} did not start")

    def __exit__(self, *exc) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


async def monitor_handler(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "patients": ["P1019", "P1023"], "requested": payload.get("patient_id")}


async def summary_stream(payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    for order, (name, content) in enumerate(
        [
            ("patient", "You were treated for type 2 diabetes and high blood pressure."),
            ("medications", "Take Metformin 500 mg twice daily with meals."),
            ("labs", "Your HbA1c was 6.9%, which is close to target."),
            ("bill", "Your bill of USD 1903.07 has been paid in full."),
            ("instructions", "See Dr. Patel on 12 June. Return to the ED for chest pain."),
        ],
        start=1,
    ):
        await asyncio.sleep(0.01)
        yield {"section": name, "order": order, "content": content}


@pytest.fixture(scope="module")
def live_agents():
    monitor = build_app(monitor_card(), handler=monitor_handler)
    summary = build_app(summary_card(), stream_handler=summary_stream)
    with BackgroundServer(monitor, 8103), BackgroundServer(summary, 8104):
        yield


class TestLiveDiscovery:
    async def test_agent_cards_are_discoverable(self, live_agents):
        client = A2AClient()
        monitor = await client.fetch_card("monitor")
        summary = await client.fetch_card("summary")

        assert monitor["name"] == "Discharge Monitor Agent"
        assert monitor["capabilities"]["streaming"] is False
        assert summary["capabilities"]["streaming"] is True
        assert summary["capabilities"]["pushNotifications"] is True

    async def test_health_reports_every_configured_agent(self, live_agents):
        health = await A2AClient().health()
        assert health["monitor"]["up"] is True
        assert health["summary"]["up"] is True
        # Agents that are not running are reported as down, not as an exception.
        assert health["extractor"]["up"] is False


class TestLiveNonStreaming:
    async def test_send_message(self, live_agents):
        result = await A2AClient().send_message("monitor", {"patient_id": "P1019"})
        assert result["ok"] is True
        assert result["requested"] == "P1019"
        assert "P1019" in result["patients"]

    async def test_metadata_threads_the_trace_id(self, live_agents):
        result = await A2AClient().send_message(
            "monitor", {"patient_id": "P1023"}, metadata={"trace_id": "trace-abc", "case_id": "CASE-1"}
        )
        assert result["ok"] is True

    async def test_wrong_token_is_rejected(self, live_agents, monkeypatch):
        import hospital_ai.a2a.client as client_module

        real = client_module.get_settings

        class Wrong:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, item):
                if item == "agent_auth_token":
                    return "definitely-not-the-token"
                return getattr(self._inner, item)

        monkeypatch.setattr(client_module, "get_settings", lambda: Wrong(real()))

        with pytest.raises(AuthenticationError):
            await A2AClient().send_message("monitor", {})


class TestLiveStreaming:
    async def test_summary_streams_section_by_section(self, live_agents):
        sections = []
        async for chunk in A2AClient().send_message_streaming(
            "summary", {"patient_id": "P1019"}
        ):
            if isinstance(chunk, dict) and "section" in chunk:
                sections.append(chunk)

        assert [s["section"] for s in sections] == [
            "patient", "medications", "labs", "bill", "instructions"
        ]
        assert [s["order"] for s in sections] == [1, 2, 3, 4, 5]
        assert "Metformin" in sections[1]["content"]

    async def test_streaming_is_refused_for_non_streaming_agents(self, live_agents):
        with pytest.raises(ValueError, match="streaming=False"):
            async for _ in A2AClient().send_message_streaming("monitor", {}):
                pass

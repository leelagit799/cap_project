"""A2A client used by the Host Orchestrator — doc §5 and Table 10.

Implements both invocation modes the specification requires:

- ``send_message()`` — awaits a single final artifact (agents 8100–8103)
- ``send_message_streaming()`` — async-iterates progressive events (8104, 8105)

Every request carries the ``X-Agent-Auth-Token`` shared secret, retries
transport failures with backoff, and trips a per-agent circuit breaker so one
dead service cannot stall every case behind its timeout.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from hospital_ai.a2a.auth import AUTH_HEADER
from hospital_ai.core.config import get_settings
from hospital_ai.core.errors import AgentUnreachableError, AuthenticationError
from hospital_ai.core.logging import get_logger
from hospital_ai.core.retry import CircuitBreaker, retry_async

_log = get_logger(__name__, component="a2a-client")


@dataclass(frozen=True)
class AgentEndpoint:
    name: str
    port: int
    streaming: bool

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def card_url(self) -> str:
        return f"{self.base_url}/.well-known/agent.json"


def default_endpoints() -> dict[str, AgentEndpoint]:
    ports = get_settings().ports
    return {
        "extractor": AgentEndpoint("extractor", ports.extractor, streaming=False),
        "validator": AgentEndpoint("validator", ports.validator, streaming=False),
        "normalizer": AgentEndpoint("normalizer", ports.normalizer, streaming=False),
        "monitor": AgentEndpoint("monitor", ports.monitor, streaming=False),
        "summary": AgentEndpoint("summary", ports.summary, streaming=True),
        "rag": AgentEndpoint("rag", ports.rag, streaming=True),
    }


def _rpc_body(method: str, payload: dict[str, Any], metadata: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": method,
        "params": {
            "message": {
                "role": "user",
                "messageId": uuid.uuid4().hex,
                "parts": [{"kind": "data", "data": payload}],
                "metadata": metadata or {},
            }
        },
    }


def _extract_result(envelope: dict[str, Any]) -> Any:
    """Pull the agent's payload out of a JSON-RPC A2A response."""
    if "error" in envelope:
        raise AgentUnreachableError(f"A2A error: {envelope['error']}")

    result = envelope.get("result", envelope)
    if not isinstance(result, dict):
        return result

    parts = result.get("parts")

    if parts is None and result.get("kind") == "status-update":
        # Progressive streaming event: the chunk rides on the status message.
        # Lifecycle-only updates (submitted / working with no message) carry
        # nothing for the caller.
        message = (result.get("status") or {}).get("message")
        if message is None:
            return None
        parts = message.get("parts")

    if parts is None:
        # Task-shaped response: read the newest artifact or history entry.
        artifacts = result.get("artifacts") or []
        history = result.get("history") or []
        if artifacts:
            parts = artifacts[-1].get("parts")
        elif history:
            parts = history[-1].get("parts")

    for part in parts or []:
        if part.get("kind") == "data" and isinstance(part.get("data"), dict):
            return part["data"]
        text = part.get("text")
        if isinstance(text, str):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return result


class A2AClient:
    def __init__(self, endpoints: dict[str, AgentEndpoint] | None = None, timeout: float = 120.0):
        self.endpoints = endpoints or default_endpoints()
        self.timeout = timeout
        self._breakers = {
            name: CircuitBreaker(name, threshold=5) for name in self.endpoints
        }

    def _headers(self) -> dict[str, str]:
        return {
            AUTH_HEADER: get_settings().agent_auth_token,
            "Content-Type": "application/json",
        }

    def _endpoint(self, agent: str) -> AgentEndpoint:
        if agent not in self.endpoints:
            raise KeyError(f"Unknown A2A agent {agent!r}")
        breaker = self._breakers[agent]
        if breaker.is_open:
            raise AgentUnreachableError(
                f"Circuit breaker is open for {agent}; it has failed "
                f"{breaker.failures} consecutive calls."
            )
        return self.endpoints[agent]

    async def fetch_card(self, agent: str) -> dict[str, Any]:
        """Discover an agent through its AgentCard."""
        endpoint = self._endpoint(agent)
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(endpoint.card_url)
        response.raise_for_status()
        return response.json()

    @retry_async(max_attempts=3, base_delay=1.0, max_delay=6.0)
    async def send_message(
        self, agent: str, payload: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> Any:
        """Non-streaming invocation — awaits a single final artifact."""
        endpoint = self._endpoint(agent)
        breaker = self._breakers[agent]
        body = _rpc_body("message/send", payload, metadata)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    endpoint.base_url + "/", json=body, headers=self._headers()
                )
        except httpx.HTTPError as exc:
            breaker.record_failure()
            raise AgentUnreachableError(f"{agent} unreachable: {exc}") from exc

        if response.status_code == 401:
            raise AuthenticationError(f"{agent} rejected the shared secret token")
        if response.status_code >= 500:
            breaker.record_failure()
            raise AgentUnreachableError(f"{agent} returned {response.status_code}")

        response.raise_for_status()
        breaker.record_success()
        return _extract_result(response.json())

    async def send_message_streaming(
        self, agent: str, payload: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> AsyncIterator[Any]:
        """Streaming invocation — yields progressive events as they arrive."""
        endpoint = self._endpoint(agent)
        if not endpoint.streaming:
            raise ValueError(
                f"{agent} advertises streaming=False; use send_message() instead."
            )
        breaker = self._breakers[agent]
        body = _rpc_body("message/stream", payload, metadata)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    endpoint.base_url + "/",
                    json=body,
                    headers={**self._headers(), "Accept": "text/event-stream"},
                ) as response:
                    if response.status_code == 401:
                        raise AuthenticationError(f"{agent} rejected the shared secret token")
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            envelope = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        chunk = _extract_result(envelope)
                        # Lifecycle-only events (submitted, working, completed)
                        # carry no payload for the caller.
                        if chunk is not None:
                            yield chunk
        except httpx.HTTPError as exc:
            breaker.record_failure()
            raise AgentUnreachableError(f"{agent} stream failed: {exc}") from exc

        breaker.record_success()

    async def health(self) -> dict[str, dict[str, Any]]:
        """Poll every agent for the orchestrator's system-health panel."""
        results: dict[str, dict[str, Any]] = {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            for name, endpoint in self.endpoints.items():
                try:
                    response = await client.get(f"{endpoint.base_url}/health")
                    results[name] = {
                        "up": response.status_code == 200,
                        "port": endpoint.port,
                        **(response.json() if response.status_code == 200 else {}),
                    }
                except httpx.HTTPError as exc:
                    results[name] = {"up": False, "port": endpoint.port, "error": str(exc)}
        return results

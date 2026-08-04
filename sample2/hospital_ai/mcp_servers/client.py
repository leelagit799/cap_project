"""Multi-server MCP client — doc §4.

"Agents must connect to two MCP servers simultaneously, demonstrating
multi-server MCP connectivity." This module owns that connection so every
agent gets the same session semantics and the same client-side callbacks.

The three client responsibilities the specification assigns:

- ``sampling_callback`` — read the server's ModelPreferences hint, route to the
  matching LiteLLM model, return a ``CreateMessageResult`` (doc §2.3)
- ``elicitation_callback`` — render the schema to a reviewer and return an
  ``ElicitResult`` (doc §2.4.1)
- ``list_roots`` — declare the authorised input workspace (doc §2.1)
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

import mcp.types as types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl

from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import get_logger

_log = get_logger(__name__, component="mcp-client")

SamplingHandler = Callable[
    [types.CreateMessageRequestParams], Awaitable[types.CreateMessageResult]
]
ElicitationHandler = Callable[[types.ElicitRequestParams], Awaitable[types.ElicitResult]]


@dataclass(frozen=True)
class ServerEndpoint:
    name: str
    url: str

    @classmethod
    def primary(cls) -> "ServerEndpoint":
        port = get_settings().ports.primary_mcp
        return cls("clinical-tools", f"http://localhost:{port}/clinicaltools")

    @classmethod
    def analytics(cls) -> "ServerEndpoint":
        port = get_settings().ports.analytics_mcp
        return cls("analytics-tools", f"http://localhost:{port}/analyticstools")


def build_roots_callback(workspace: Path | None = None):
    """Declare the input folder as a Root URI when the connection opens."""
    root = (workspace or get_settings().roots.workspace).resolve()

    async def list_roots(context) -> types.ListRootsResult:
        _log.debug("declaring MCP root", extra={"root": str(root)})
        return types.ListRootsResult(
            roots=[types.Root(uri=AnyUrl(root.as_uri()), name="clinical-input")]
        )

    return list_roots


def build_sampling_callback(handler: SamplingHandler | None = None):
    """Wrap an LLM handler as an MCP sampling callback.

    Without a handler the callback returns an MCP error rather than inventing
    a translation: a silent stub would make the Sampling primitive look
    satisfied while producing unvalidated clinical text.
    """

    async def sampling_callback(
        context, params: types.CreateMessageRequestParams
    ) -> types.CreateMessageResult | types.ErrorData:
        hints = [h.name for h in (params.modelPreferences.hints or [])] if params.modelPreferences else []
        _log.info("sampling requested by server", extra={"hints": hints})

        if handler is None:
            return types.ErrorData(
                code=types.INVALID_REQUEST,
                message="This client has no LLM configured for MCP sampling.",
            )
        try:
            return await handler(params)
        except Exception as exc:  # noqa: BLE001 - surfaced to the server as an error
            _log.error("sampling handler failed", extra={"error": str(exc)})
            return types.ErrorData(code=types.INTERNAL_ERROR, message=str(exc))

    return sampling_callback


def build_elicitation_callback(handler: ElicitationHandler | None = None):
    """Wrap a reviewer-facing form as an MCP elicitation callback.

    With no handler the client declines, which the Rules Engine Tool treats as
    "leave the gap unresolved and flag for HITL" — the safe default.
    """

    async def elicitation_callback(
        context, params: types.ElicitRequestParams
    ) -> types.ElicitResult:
        _log.info("elicitation requested by server", extra={"message": params.message})
        if handler is None:
            return types.ElicitResult(action="decline")
        try:
            return await handler(params)
        except Exception as exc:  # noqa: BLE001 - a broken form must not hang the case
            _log.error("elicitation handler failed", extra={"error": str(exc)})
            return types.ElicitResult(action="cancel")

    return elicitation_callback


class MultiServerMCPClient:
    """Holds simultaneous sessions to the primary and analytics MCP servers."""

    def __init__(
        self,
        *,
        sampling_handler: SamplingHandler | None = None,
        elicitation_handler: ElicitationHandler | None = None,
        workspace: Path | None = None,
        endpoints: list[ServerEndpoint] | None = None,
    ) -> None:
        self.endpoints = endpoints or [ServerEndpoint.primary(), ServerEndpoint.analytics()]
        self._sampling = build_sampling_callback(sampling_handler)
        self._elicitation = build_elicitation_callback(elicitation_handler)
        self._roots = build_roots_callback(workspace)
        self.sessions: dict[str, ClientSession] = {}
        self._stack: contextlib.AsyncExitStack | None = None

    async def __aenter__(self) -> "MultiServerMCPClient":
        self._stack = contextlib.AsyncExitStack()
        await self._stack.__aenter__()
        for endpoint in self.endpoints:
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(endpoint.url)
            )
            session = await self._stack.enter_async_context(
                ClientSession(
                    read,
                    write,
                    sampling_callback=self._sampling,
                    elicitation_callback=self._elicitation,
                    list_roots_callback=self._roots,
                )
            )
            await session.initialize()
            self.sessions[endpoint.name] = session
            _log.info("MCP session established", extra={"server": endpoint.name})
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(*exc_info)
        self.sessions.clear()

    def session_for(self, tool_name: str) -> ClientSession:
        """Route a tool call to whichever server exposes it."""
        analytics_tools = {
            "calculate_risk_score",
            "get_population_benchmarks",
            "generate_risk_heatmap",
        }
        name = "analytics-tools" if tool_name in analytics_tools else "clinical-tools"
        if name not in self.sessions:
            raise KeyError(f"No MCP session for {name!r}; is the server running?")
        return self.sessions[name]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        session = self.session_for(tool_name)
        result = await session.call_tool(tool_name, arguments)
        if result.isError:
            text = result.content[0].text if result.content else "unknown MCP tool error"
            raise RuntimeError(f"{tool_name} failed: {text}")
        return _unwrap(result)

    async def read_resource(self, uri: str) -> str:
        result = await self.sessions["clinical-tools"].read_resource(AnyUrl(uri))
        return result.contents[0].text

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        result = await self.sessions["clinical-tools"].get_prompt(name, arguments or {})
        return result.messages[0].content.text

    async def list_all_tools(self) -> dict[str, list[str]]:
        return {
            name: [t.name for t in (await session.list_tools()).tools]
            for name, session in self.sessions.items()
        }


def _unwrap(result) -> Any:
    """Prefer the structured payload; fall back to parsing the text block."""
    import json

    if getattr(result, "structuredContent", None):
        structured = result.structuredContent
        # FastMCP wraps non-dict returns in {"result": ...}.
        if set(structured) == {"result"}:
            return structured["result"]
        return structured
    if result.content:
        text = result.content[0].text
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
    return None


@contextlib.asynccontextmanager
async def connect(**kwargs: Any) -> AsyncIterator[MultiServerMCPClient]:
    client = MultiServerMCPClient(**kwargs)
    async with client:
        yield client

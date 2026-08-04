"""Tool access for agents.

Every agent reaches its tools, resources and prompts through this interface, so
the graph logic never depends on how the MCP session was established. In
production it is backed by ``MultiServerMCPClient`` over streamable-HTTP; in
tests it is backed by an in-memory MCP session, which keeps the protocol in the
loop rather than stubbing it out.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from pydantic import AnyUrl


@runtime_checkable
class ToolGateway(Protocol):
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any: ...

    async def read_resource(self, uri: str) -> str: ...

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str: ...


class SessionGateway:
    """Adapts a single ``mcp.ClientSession`` to the gateway interface."""

    def __init__(self, session: Any, analytics_session: Any | None = None) -> None:
        self._session = session
        self._analytics = analytics_session

    ANALYTICS_TOOLS = frozenset(
        {"calculate_risk_score", "get_population_benchmarks", "generate_risk_heatmap"}
    )

    def _for(self, tool_name: str) -> Any:
        if tool_name in self.ANALYTICS_TOOLS and self._analytics is not None:
            return self._analytics
        return self._session

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        result = await self._for(tool_name).call_tool(tool_name, arguments)
        if result.isError:
            text = result.content[0].text if result.content else "unknown MCP error"
            raise RuntimeError(f"{tool_name} failed: {text}")
        return unwrap(result)

    async def read_resource(self, uri: str) -> str:
        result = await self._session.read_resource(AnyUrl(uri))
        return result.contents[0].text

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        result = await self._session.get_prompt(name, arguments or {})
        return result.messages[0].content.text


def unwrap(result: Any) -> Any:
    """Prefer the structured payload; fall back to parsing the text block."""
    structured = getattr(result, "structuredContent", None)
    if structured:
        # FastMCP wraps non-dict returns as {"result": ...}.
        if set(structured) == {"result"}:
            return structured["result"]
        return structured
    if getattr(result, "content", None):
        text = result.content[0].text
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
    return None

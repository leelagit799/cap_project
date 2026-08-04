"""A2A shared-secret authentication — doc §5.

"Authentication via shared secret header ``X-Agent-Auth-Token`` is required on
all A2A servers."

The AgentCard itself stays readable without the token. Discovery is listed in
§5 as a protocol function, and a card that could not be fetched without already
holding the secret would make discovery impossible; the card advertises the
scheme instead. Every invocation route is protected.
"""

from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from hospital_ai.core.logging import get_logger

_log = get_logger(__name__, component="a2a-auth")

AUTH_HEADER = "X-Agent-Auth-Token"

#: Discovery endpoints, readable without the shared secret.
PUBLIC_PATHS = frozenset(
    {
        "/.well-known/agent.json",
        "/.well-known/agent-card.json",
        "/health",
    }
)


class SharedSecretAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token: str, agent_name: str) -> None:
        super().__init__(app)
        self._token = token
        self._agent_name = agent_name

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        supplied = request.headers.get(AUTH_HEADER)
        if supplied is None:
            _log.warning(
                "rejected unauthenticated A2A request",
                extra={"agent": self._agent_name, "path": request.url.path},
            )
            return JSONResponse(
                {"error": f"Missing {AUTH_HEADER} header"}, status_code=401
            )

        # Constant-time comparison keeps the token from leaking through timing.
        if not hmac.compare_digest(supplied, self._token):
            _log.warning(
                "rejected A2A request with an invalid token",
                extra={"agent": self._agent_name, "path": request.url.path},
            )
            return JSONResponse({"error": "Invalid agent auth token"}, status_code=401)

        return await call_next(request)

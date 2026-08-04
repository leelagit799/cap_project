"""Agent-to-Agent protocol layer — cards, authenticated servers, and client."""

from hospital_ai.a2a.auth import AUTH_HEADER, SharedSecretAuthMiddleware
from hospital_ai.a2a.cards import all_cards, build_card
from hospital_ai.a2a.client import A2AClient, AgentEndpoint, default_endpoints
from hospital_ai.a2a.server import build_app, serve

__all__ = [
    "AUTH_HEADER",
    "A2AClient",
    "AgentEndpoint",
    "SharedSecretAuthMiddleware",
    "all_cards",
    "build_app",
    "build_card",
    "default_endpoints",
    "serve",
]

"""Typed error taxonomy.

Every failure path in the system raises one of these rather than a bare
Exception, so callers can distinguish "retry this" from "escalate this" from
"the operator misconfigured something".
"""

from __future__ import annotations


class DischargeFlowError(Exception):
    """Base class for every error raised by this system."""

    #: Retrying the same call unchanged could plausibly succeed.
    retryable: bool = False


# --- Configuration -----------------------------------------------------------


class ConfigError(DischargeFlowError):
    """Missing, malformed or contradictory configuration."""


class MissingCredentialError(ConfigError):
    """A required secret is absent from the environment."""


# --- Transport ---------------------------------------------------------------


class TransportError(DischargeFlowError):
    """Network-level failure talking to another service."""

    retryable = True


class AgentUnreachableError(TransportError):
    """An A2A peer did not respond."""


class EHRUnavailableError(TransportError):
    """The Mock EHR REST API could not be reached or returned 5xx."""


class AuthenticationError(DischargeFlowError):
    """A2A shared-secret token missing or incorrect."""


# --- MCP ---------------------------------------------------------------------


class MCPError(DischargeFlowError):
    """Base for MCP protocol failures."""


class RootAccessDeniedError(MCPError):
    """A path escaped the Roots-declared workspace.

    Raised by the Clinical Watcher Tool's ``Path.relative_to()`` check. This is
    a hard failure by design: the specification forbids falling back to raw
    filesystem paths when Roots negotiation fails.
    """


class SamplingUnsupportedError(MCPError):
    """The connected client did not implement a sampling callback."""


class ElicitationTimeoutError(MCPError):
    """No reviewer responded to an elicitation request in time.

    Treated as ``decline`` by the Rules Engine Tool.
    """


# --- Pipeline ----------------------------------------------------------------


class ExtractionError(DischargeFlowError):
    """A source document could not be parsed into structured clinical data."""


class TranslationError(DischargeFlowError):
    """Language normalization failed for a document."""


class ValidationBlockedError(DischargeFlowError):
    """Validation produced a blocking finding; discharge cannot be released."""


class GuardrailViolation(DischargeFlowError):
    """A Responsible AI guardrail rejected content."""

    def __init__(self, guardrail: str, detail: str) -> None:
        super().__init__(f"{guardrail}: {detail}")
        self.guardrail = guardrail
        self.detail = detail


class LLMError(DischargeFlowError):
    """The LLM gateway could not produce a completion."""

    retryable = True

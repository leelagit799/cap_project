"""LangFuse observability — doc §7.2.

The specification lists eight things to trace:

1. one end-to-end trace id per discharge case, threaded through every agent
2. per-agent spans with latency and input/output payloads
3. per-tool-call spans for every MCP invocation
4. LLM generation events with model, prompt, response and token counts
5. sampling events: server preferences, client model chosen, result
6. elicitation events: schema sent, reviewer response, action taken
7. guardrail intervention spans
8. error spans with exception type and fallback taken

Everything is mirrored to ``data/reports/traces.jsonl``. That file is the
audit trail the reports link to, and it means a case is still fully traceable
when LangFuse credentials are absent or the network is down — observability
must never be the reason a discharge fails.

Payloads are PII-redacted before they leave the process.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from hospital_ai.core.config import PROJECT_ROOT, _load_dotenv, get_settings
from hospital_ai.core.ids import utc_now_iso
from hospital_ai.core.logging import get_logger
from hospital_ai.guardrails import PIIRedactor

_log = get_logger(__name__, component="observability")

_REDACTOR = PIIRedactor()
_WRITE_LOCK = threading.Lock()

#: Required by Langfuse v4 when attaching observations to a predetermined trace id.
_ROOT_PARENT_SPAN_ID = "fedcba0987654321"


def _redact(payload: Any) -> Any:
    """Mask direct identifiers before a payload leaves the process."""
    if isinstance(payload, str):
        return _REDACTOR.redact(payload).sanitized
    if isinstance(payload, dict):
        return {key: _redact(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_redact(item) for item in payload]
    return payload


def _as_type(kind: str) -> str:
    """Map internal span kinds to Langfuse observation types."""
    return {
        "agent": "agent",
        "tool": "tool",
        "generation": "generation",
        "guardrail": "guardrail",
    }.get(kind, "span")


@dataclass
class SpanRecord:
    name: str
    kind: str
    trace_id: str
    case_id: str | None = None
    started_at: float = field(default_factory=time.monotonic)
    duration_ms: float | None = None
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": utc_now_iso(),
            "trace_id": self.trace_id,
            "case_id": self.case_id,
            "span": self.name,
            "kind": self.kind,
            "duration_ms": self.duration_ms,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
            "error": self.error,
        }


class Tracer:
    """One tracer per discharge case, holding that case's trace id."""

    def __init__(self, trace_id: str, case_id: str | None = None, patient_id: str | None = None):
        self.trace_id = trace_id
        self.case_id = case_id
        self.patient_id = patient_id
        self.spans: list[SpanRecord] = []
        self._settings = get_settings()
        self._sink = self._settings.reports_dir / "traces.jsonl"
        self._client = _get_client()
        self._root = None

        if self._client is not None:
            try:
                self._root = self._client.start_observation(
                    name=f"discharge-case:{case_id or trace_id}",
                    trace_context={
                        "trace_id": trace_id,
                        "parent_span_id": _ROOT_PARENT_SPAN_ID,
                    },
                    input=_redact({"case_id": case_id, "patient_id": patient_id}),
                    metadata={"trace_id": trace_id, "case_id": case_id},
                )
            except Exception as exc:  # noqa: BLE001 - tracing must never break a case
                _log.warning("LangFuse root observation failed", extra={"error": str(exc)})

    # --- emission ------------------------------------------------------------

    def _emit_langfuse(self, record: SpanRecord) -> None:
        if self._client is None:
            return

        metadata = {**record.metadata, "kind": record.kind, "trace_id": self.trace_id}
        if record.duration_ms is not None:
            metadata["duration_ms"] = record.duration_ms

        kwargs: dict[str, Any] = {
            "name": record.name,
            "as_type": _as_type(record.kind),
            "input": record.input,
            "metadata": metadata,
        }
        if record.kind == "generation":
            kwargs["model"] = record.metadata.get("model")
            usage: dict[str, int] = {}
            if record.metadata.get("prompt_tokens") is not None:
                usage["input"] = int(record.metadata["prompt_tokens"])
            if record.metadata.get("completion_tokens") is not None:
                usage["output"] = int(record.metadata["completion_tokens"])
            if usage:
                kwargs["usage_details"] = usage

        try:
            if self._root is not None:
                observation = self._root.start_observation(**kwargs)
            else:
                observation = self._client.start_observation(
                    **kwargs,
                    trace_context={
                        "trace_id": self.trace_id,
                        "parent_span_id": _ROOT_PARENT_SPAN_ID,
                    },
                )

            update_kwargs: dict[str, Any] = {"output": record.output}
            if record.error:
                update_kwargs["level"] = "ERROR"
                update_kwargs["status_message"] = record.error
            observation.update(**update_kwargs)
            observation.end()
        except Exception as exc:  # noqa: BLE001 - never fail a case over telemetry
            _log.warning(
                "LangFuse observation emit failed",
                extra={"span": record.name, "error": str(exc)},
            )

    def _emit(self, record: SpanRecord) -> None:
        self.spans.append(record)
        payload = record.to_dict()

        self._sink.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK, self._sink.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str, ensure_ascii=False) + "\n")

        self._emit_langfuse(record)

    @contextmanager
    def span(
        self, name: str, kind: str = "span", input: Any = None, **metadata: Any
    ) -> Iterator[SpanRecord]:
        """Time a unit of work and record it, whether it succeeds or raises."""
        record = SpanRecord(
            name=name,
            kind=kind,
            trace_id=self.trace_id,
            case_id=self.case_id,
            input=_redact(input),
            metadata=metadata,
        )
        try:
            yield record
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            record.duration_ms = round((time.monotonic() - record.started_at) * 1000, 2)
            self._emit(record)
            raise
        record.duration_ms = round((time.monotonic() - record.started_at) * 1000, 2)
        record.output = _redact(record.output)
        self._emit(record)

    # --- typed helpers for the §7.2 checklist --------------------------------

    def agent_span(self, agent: str, input: Any = None):
        return self.span(f"agent:{agent}", kind="agent", input=input, agent=agent)

    def tool_span(self, tool: str, parameters: Any = None):
        return self.span(f"tool:{tool}", kind="tool", input=parameters, tool=tool)

    def log_generation(
        self,
        model: str,
        prompt: str,
        response: str,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        provenance: str = "live",
    ) -> None:
        record = SpanRecord(
            name=f"llm:{model}",
            kind="generation",
            trace_id=self.trace_id,
            case_id=self.case_id,
            duration_ms=None,
            input=_redact(prompt[:4000]),
            output=_redact(response[:4000]),
            metadata={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "estimated_cost_usd": _estimate_cost(model, prompt_tokens, completion_tokens),
                "provenance": provenance,
            },
        )
        self._emit(record)

    def log_sampling(
        self, server_hints: list[str], client_model: str, result_preview: str
    ) -> None:
        self._emit(
            SpanRecord(
                name="mcp:sampling",
                kind="sampling",
                trace_id=self.trace_id,
                case_id=self.case_id,
                input={"server_model_preferences": server_hints},
                output=_redact(result_preview[:1000]),
                metadata={"client_model_selected": client_model},
            )
        )

    def log_elicitation(
        self, fields: list[str], schema: dict[str, Any], action: str, response: dict[str, Any]
    ) -> None:
        self._emit(
            SpanRecord(
                name="mcp:elicitation",
                kind="elicitation",
                trace_id=self.trace_id,
                case_id=self.case_id,
                input={"fields_requested": fields, "schema": schema},
                output=_redact(response),
                metadata={"action": action},
            )
        )

    def log_guardrail(self, guardrail: str, blocked: bool, detail: str) -> None:
        self._emit(
            SpanRecord(
                name=f"guardrail:{guardrail}",
                kind="guardrail",
                trace_id=self.trace_id,
                case_id=self.case_id,
                output={"blocked": blocked, "detail": detail},
                metadata={"guardrail": guardrail, "check_result": "blocked" if blocked else "allowed"},
            )
        )

    def log_error(self, where: str, exc: BaseException, fallback: str | None = None) -> None:
        import traceback

        self._emit(
            SpanRecord(
                name=f"error:{where}",
                kind="error",
                trace_id=self.trace_id,
                case_id=self.case_id,
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    "exception_type": type(exc).__name__,
                    "stack_trace": "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )[-4000:],
                    "fallback_action": fallback,
                },
            )
        )

    def finish(self, output: Any = None) -> None:
        if self._root is not None:
            try:
                self._root.update(output=_redact(output))
                self._root.end()
            except Exception:  # noqa: BLE001 - telemetry only
                pass
        flush()

    @property
    def url(self) -> str | None:
        return trace_url(self.trace_id)


#: Rough Bedrock per-1K-token pricing, used only for the cost estimate the
#: specification asks LLM generation events to carry.
_PRICING = {
    "nova-lite": (0.00006, 0.00024),
    "command-r-plus": (0.003, 0.015),
}


def _estimate_cost(model: str, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    if prompt_tokens is None and completion_tokens is None:
        return None
    for key, (input_rate, output_rate) in _PRICING.items():
        if key in model:
            return round(
                (prompt_tokens or 0) / 1000 * input_rate
                + (completion_tokens or 0) / 1000 * output_rate,
                6,
            )
    return None


_CLIENT: Any = None
_CLIENT_READY = False
_LAST_LANGFUSE_ERROR: str | None = None


def _get_client(*, force: bool = False):
    global _CLIENT, _CLIENT_READY, _LAST_LANGFUSE_ERROR
    if force:
        reset_client()
    if _CLIENT_READY:
        return _CLIENT

    settings = get_settings()
    if not settings.langfuse.enabled:
        _CLIENT_READY = True
        _LAST_LANGFUSE_ERROR = (
            "Missing or empty LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
            f"(expected in {settings.project_root / '.env'})."
        )
        _log.info("LangFuse disabled; tracing to data/reports/traces.jsonl only")
        return None

    from hospital_ai.core.config import _env_bool

    if not _env_bool("LANGFUSE_TRACING_ENABLED", default=True):
        _CLIENT_READY = True
        _LAST_LANGFUSE_ERROR = "LANGFUSE_TRACING_ENABLED is set to false."
        _log.info("LangFuse tracing disabled via LANGFUSE_TRACING_ENABLED")
        return None

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse.public_key,
            secret_key=settings.langfuse.secret_key,
            base_url=settings.langfuse.host,
            timeout=15,
        )
        if not client.auth_check():
            _LAST_LANGFUSE_ERROR = (
                "LangFuse auth_check failed. Verify LANGFUSE_PUBLIC_KEY, "
                "LANGFUSE_SECRET_KEY, and LANGFUSE_HOST / LANGFUSE_BASE_URL."
            )
            _log.warning(_LAST_LANGFUSE_ERROR)
            _CLIENT_READY = True
            return None

        _CLIENT = client
        _CLIENT_READY = True
        _LAST_LANGFUSE_ERROR = None
        _log.info("LangFuse client ready", extra={"host": settings.langfuse.host})
    except Exception as exc:  # noqa: BLE001 - fall back to the JSONL sink
        _LAST_LANGFUSE_ERROR = f"{type(exc).__name__}: {exc}"
        _log.warning("LangFuse unavailable; using JSONL sink", extra={"error": str(exc)})
        _CLIENT = None
        _CLIENT_READY = True
    return _CLIENT


def langfuse_status(*, probe: bool = False) -> dict[str, Any]:
    """Report whether LangFuse credentials are configured and authenticated."""
    if probe:
        _load_dotenv(PROJECT_ROOT / ".env")
        get_settings.cache_clear()
        reset_client()

    settings = get_settings()
    env_path = settings.project_root / ".env"
    status: dict[str, Any] = {
        "configured": settings.langfuse.enabled,
        "base_url": settings.langfuse.host,
        "env_file": str(env_path),
        "env_file_exists": env_path.is_file(),
    }
    if not settings.langfuse.enabled:
        status.update(
            state="disabled",
            message=(
                "Missing or empty LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY. "
                f"Add them to {env_path} and restart the dashboard."
            ),
        )
        return status

    client = _get_client()
    if client is not None:
        status.update(
            state="connected",
            message=f"Connected to {settings.langfuse.host}",
        )
    else:
        status.update(
            state="error",
            message=_LAST_LANGFUSE_ERROR or "LangFuse client unavailable",
        )
    return status


def trace_url(trace_id: str | None) -> str | None:
    """Return a Langfuse dashboard URL for a trace id when credentials are valid."""
    if not trace_id:
        return None
    settings = get_settings()
    if not settings.langfuse.enabled:
        return None
    client = _get_client()
    if client is not None:
        try:
            return client.get_trace_url(trace_id=trace_id)
        except Exception:  # noqa: BLE001 - fall back to a generic link
            pass
    return f"{settings.langfuse.host.rstrip('/')}/trace/{trace_id}"


def flush() -> None:
    client = _get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:  # noqa: BLE001 - telemetry only
            pass


def reset_client() -> None:
    global _CLIENT, _CLIENT_READY, _LAST_LANGFUSE_ERROR
    _CLIENT, _CLIENT_READY = None, False
    _LAST_LANGFUSE_ERROR = None


__all__ = ["SpanRecord", "Tracer", "flush", "langfuse_status", "reset_client", "trace_url"]

"""LiteLLM gateway — doc Table 14.

"LiteLLM — unified interface to AWS Bedrock + Cohere APIs." Every LLM call in
the system goes through here; nothing imports boto3 or the Cohere SDK directly.

Model routing follows the MCP Sampling hints the Medical Lang Bridge Tool
advertises: ``nova-lite`` for multilingual content, ``command-r-plus`` for
English. Command R+ is reached through Bedrock unless a direct
``COHERE_API_KEY`` is configured.

``LLM_OFFLINE=1`` swaps in a deterministic stub so the pipeline is demonstrable
without credentials. Offline output is tagged in the audit trail and, for
translation, is scored below the confidence threshold on purpose — an offline
run must route to HITL rather than quietly auto-approve.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable

from hospital_ai.core.config import get_settings
from hospital_ai.core.errors import LLMError
from hospital_ai.core.logging import get_logger

_log = get_logger(__name__, component="llm-gateway")


@dataclass
class Completion:
    text: str
    model: str
    provenance: str = "live"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    fallback_used: bool = False

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


class LLMGateway:
    """Routes completions to Bedrock Nova Lite with a Command R+ fallback."""

    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.primary = settings.llm.primary_model
        self.fallback = settings.llm.fallback_model
        self.offline = settings.llm.offline or not settings.llm.credentials_present

        if self.offline:
            _log.warning(
                "LLM gateway running in deterministic offline mode",
                extra={"reason": "LLM_OFFLINE set or AWS credentials absent"},
            )

    def resolve_model(self, hint: str | None) -> str:
        """Map an MCP ModelPreferences hint onto a concrete LiteLLM model id."""
        if not hint:
            return self.primary
        token = hint.lower()
        if "command" in token or "cohere" in token:
            if self.settings.llm.cohere_api_key:
                return "command-r-plus"
            return self.fallback
        if "nova" in token:
            return self.primary
        # An unrecognised hint is honoured verbatim so a client can request a
        # model the server did not anticipate.
        return hint

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model_hint: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Completion:
        model = self.resolve_model(model_hint)

        if self.offline:
            return _offline_completion(prompt, system, model)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for candidate, is_fallback in ((model, False), (self.fallback, True)):
            if is_fallback and candidate == model:
                continue
            try:
                return await self._invoke(
                    candidate, messages, max_tokens, temperature, fallback=is_fallback
                )
            except Exception as exc:  # noqa: BLE001 - retried on the fallback model
                _log.warning(
                    "completion failed",
                    extra={"model": candidate, "fallback": is_fallback, "error": str(exc)},
                )
                last = exc

        raise LLMError(f"Both {model} and {self.fallback} failed: {last}")

    async def _invoke(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float,
        *,
        fallback: bool,
    ) -> Completion:
        import litellm

        litellm.drop_params = True
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=max_tokens or self.settings.llm.max_output_tokens,
            temperature=temperature,
            timeout=self.settings.llm.request_timeout_s,
            aws_region_name=self.settings.llm.aws_region,
        )
        usage = getattr(response, "usage", None)
        completion = Completion(
            text=(response.choices[0].message.content or "").strip(),
            model=model,
            provenance="live",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            fallback_used=fallback,
        )
        _log.info(
            "completion produced",
            extra={
                "model": model,
                "chars": len(completion.text),
                "tokens": completion.total_tokens,
                "fallback": fallback,
            },
        )
        return completion

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model_hint: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        """Token-by-token streaming, used by the Summary Generator and RAG agent."""
        model = self.resolve_model(model_hint)

        if self.offline:
            completion = _offline_completion(prompt, system, model)
            for chunk in _chunk_text(completion.text):
                yield chunk
            return

        import litellm

        litellm.drop_params = True
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                max_tokens=max_tokens or self.settings.llm.max_output_tokens,
                temperature=temperature,
                timeout=self.settings.llm.request_timeout_s,
                aws_region_name=self.settings.llm.aws_region,
                stream=True,
            )
            async for part in response:
                delta = part.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text
        except Exception as exc:  # noqa: BLE001 - callers surface this as a failure
            raise LLMError(f"Streaming completion failed on {model}: {exc}") from exc


def _chunk_text(text: str, size: int = 60) -> Iterable[str]:
    for index in range(0, len(text), size):
        yield text[index : index + size]


#: Minimal cross-language glossary for the offline stub. Real translation is
#: the Bedrock path; this exists only so the pipeline is demonstrable and
#: testable without credentials.
_OFFLINE_GLOSSARY = {
    "ontslagbrief": "discharge letter", "patiëntnummer": "patient number",
    "naam": "name", "geslacht": "sex", "adres": "address",
    "opnamedatum": "admission date", "ontslagdatum": "discharge date",
    "afdeling": "ward", "bed": "bed", "ontslagdiagnose": "discharge diagnosis",
    "allergieën": "allergies", "ontslagrecepten": "discharge prescriptions",
    "vervolgafspraak": "follow-up appointment",
    "ontslaginstructies": "discharge instructions",
    "resumen de alta": "discharge summary", "paciente": "patient",
    "nombre": "name", "sexo": "sex", "edad": "age", "dirección": "address",
    "fecha de ingreso": "admission date", "fecha de alta": "discharge date",
    "sala": "ward", "cama": "bed", "alergias": "allergies",
    "recetas de alta": "discharge prescriptions",
    "instrucciones de alta": "discharge instructions",
}


def _offline_completion(prompt: str, system: str | None, model: str) -> Completion:
    text = prompt
    lowered = text.lower()
    for source, target in _OFFLINE_GLOSSARY.items():
        if source in lowered:
            text = re.sub(re.escape(source), target, text, flags=re.IGNORECASE)
            lowered = text.lower()

    return Completion(
        text=text,
        model=f"{model} (offline stub)",
        provenance="offline-deterministic",
    )


_GATEWAY: LLMGateway | None = None


def get_gateway() -> LLMGateway:
    global _GATEWAY
    if _GATEWAY is None:
        _GATEWAY = LLMGateway()
    return _GATEWAY


def reset_gateway() -> None:
    """Drop the cached gateway so a settings change takes effect (tests)."""
    global _GATEWAY
    _GATEWAY = None

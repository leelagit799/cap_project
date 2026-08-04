"""Retry with exponential backoff and jitter.

Applied to every network boundary: A2A calls, MCP tool invocations, Mock EHR
requests and LLM completions. Only errors marked ``retryable`` are retried; a
guardrail violation or a blocked discharge must never be retried into success.
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from typing import Any, Awaitable, Callable, TypeVar

from hospital_ai.core.errors import DischargeFlowError
from hospital_ai.core.logging import get_logger

T = TypeVar("T")
_log = get_logger(__name__)

DEFAULT_RETRYABLE: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)


def _should_retry(exc: BaseException, retry_on: tuple[type[BaseException], ...]) -> bool:
    if isinstance(exc, DischargeFlowError):
        return exc.retryable
    return isinstance(exc, retry_on)


def _delay(attempt: int, base: float, cap: float) -> float:
    """Full-jitter exponential backoff."""
    return random.uniform(0, min(cap, base * (2**attempt)))


def retry_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = DEFAULT_RETRYABLE,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001 - re-raised below
                    if not _should_retry(exc, retry_on) or attempt == max_attempts - 1:
                        raise
                    last = exc
                    wait = _delay(attempt, base_delay, max_delay)
                    _log.warning(
                        "retrying after failure",
                        extra={
                            "operation": func.__qualname__,
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "sleep_s": round(wait, 2),
                            "error": str(exc),
                        },
                    )
                    await asyncio.sleep(wait)
            raise last  # type: ignore[misc]

        return wrapper

    return decorator


def retry_sync(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = DEFAULT_RETRYABLE,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001 - re-raised below
                    if not _should_retry(exc, retry_on) or attempt == max_attempts - 1:
                        raise
                    last = exc
                    wait = _delay(attempt, base_delay, max_delay)
                    _log.warning(
                        "retrying after failure",
                        extra={
                            "operation": func.__qualname__,
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "sleep_s": round(wait, 2),
                            "error": str(exc),
                        },
                    )
                    time.sleep(wait)
            raise last  # type: ignore[misc]

        return wrapper

    return decorator


class CircuitBreaker:
    """Opens after ``threshold`` consecutive failures against one peer.

    The Host Orchestrator keeps one per A2A agent so a dead service fails fast
    instead of stalling every case behind its timeout.
    """

    def __init__(self, name: str, threshold: int = 5, reset_after_s: float = 30.0) -> None:
        self.name = name
        self.threshold = threshold
        self.reset_after_s = reset_after_s
        self.failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.reset_after_s:
            self.reset()
            return False
        return True

    def record_success(self) -> None:
        self.reset()

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            _log.error("circuit breaker opened", extra={"peer": self.name})

    def reset(self) -> None:
        self.failures = 0
        self._opened_at = None

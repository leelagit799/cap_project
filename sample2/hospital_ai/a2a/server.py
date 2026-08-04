"""A2A server scaffolding shared by all six agents — doc §5.

Wraps ``a2a-sdk``'s Starlette application with:

- the AgentCard at ``/.well-known/agent.json`` (the path the specification
  names) as well as the SDK's default ``/.well-known/agent-card.json``
- shared-secret authentication on every invocation route
- a ``/health`` probe for the orchestrator's agent-status panel
- push-notification support, declared in each card's capabilities

Agent authors supply a handler; whether it streams is decided by the card, so
the transport and the business logic stay separate.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Awaitable, Callable
from uuid import uuid4

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
    TaskUpdater,
)
from a2a.types import AgentCard, Part, TaskState, TextPart
from a2a.utils import new_agent_text_message
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from hospital_ai.a2a.auth import SharedSecretAuthMiddleware
from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import configure_logging, get_logger

_log = get_logger(__name__, component="a2a-server")

#: A handler receives the decoded request payload and returns a JSON-able
#: result, or yields a sequence of them when the agent streams.
Handler = Callable[[dict[str, Any]], Awaitable[Any]]
StreamHandler = Callable[[dict[str, Any]], AsyncIterator[Any]]


def decode_payload(context: RequestContext) -> dict[str, Any]:
    """Extract the caller's JSON payload from an A2A message.

    Callers may send a structured ``DataPart`` or a JSON string in a
    ``TextPart``; both are normalised to a dict. Free text arrives as
    ``{"text": ...}`` so conversational agents still work.
    """
    message = context.message
    if message is None:
        return {}

    merged: dict[str, Any] = {}
    text_chunks: list[str] = []

    for part in message.parts or []:
        root = getattr(part, "root", part)
        data = getattr(root, "data", None)
        if isinstance(data, dict):
            merged.update(data)
            continue
        text = getattr(root, "text", None)
        if isinstance(text, str):
            text_chunks.append(text)

    if text_chunks:
        blob = "\n".join(text_chunks).strip()
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                merged.update(parsed)
            else:
                merged.setdefault("text", parsed)
        except json.JSONDecodeError:
            merged.setdefault("text", blob)

    # The orchestrator threads trace_id and case_id through message metadata
    # (doc §7.2), so agents can join the same LangFuse trace.
    metadata = getattr(message, "metadata", None)
    if isinstance(metadata, dict):
        merged.setdefault("_metadata", metadata)

    return merged


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


class HandlerExecutor(AgentExecutor):
    """Adapts a plain async handler to the A2A executor interface."""

    def __init__(
        self,
        agent_name: str,
        handler: Handler | None = None,
        stream_handler: StreamHandler | None = None,
    ) -> None:
        if handler is None and stream_handler is None:
            raise ValueError("Provide either handler or stream_handler")
        self.agent_name = agent_name
        self.handler = handler
        self.stream_handler = stream_handler

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        payload = decode_payload(context)
        log = get_logger(__name__, agent=self.agent_name)
        log.info("A2A request received", extra={"keys": sorted(payload)})

        if self.stream_handler is not None:
            await self._execute_streaming(context, event_queue, payload, log)
            return

        try:
            result = await self.handler(payload)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - reported to the caller as an artifact
            log.error("A2A handler failed", extra={"error": str(exc)}, exc_info=True)
            result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}

        await event_queue.enqueue_event(
            new_agent_text_message(
                _as_text(result), context_id=context.context_id, task_id=context.task_id
            )
        )

    async def _execute_streaming(
        self, context: RequestContext, event_queue: EventQueue, payload: dict[str, Any], log
    ) -> None:
        """Emit progressive task-status updates.

        A bare Message event closes an A2A stream, so progressive delivery has
        to go through the task lifecycle: submit, start_work, one working
        update per chunk, then complete.
        """
        task = context.current_task
        task_id = task.id if task else (context.task_id or str(uuid4()))
        context_id = (task.context_id if task else context.context_id) or str(uuid4())

        updater = TaskUpdater(event_queue, task_id, context_id)
        if task is None:
            await updater.submit()
        await updater.start_work()

        try:
            async for chunk in self.stream_handler(payload):  # type: ignore[misc]
                await updater.update_status(
                    TaskState.working,
                    message=updater.new_agent_message(
                        [Part(root=TextPart(text=_as_text(chunk)))]
                    ),
                )
            await updater.complete()
        except Exception as exc:  # noqa: BLE001 - reported to the caller as a failure
            log.error("A2A stream handler failed", extra={"error": str(exc)}, exc_info=True)
            await updater.failed(
                updater.new_agent_message(
                    [Part(root=TextPart(text=json.dumps(
                        {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
                    )))]
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            new_agent_text_message(
                json.dumps({"ok": False, "cancelled": True}),
                context_id=context.context_id,
                task_id=context.task_id,
            )
        )


def build_app(
    card: AgentCard,
    handler: Handler | None = None,
    stream_handler: StreamHandler | None = None,
) -> Starlette:
    """Assemble the authenticated A2A Starlette application for one agent."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.reports_dir / "pipeline.log")

    import httpx

    push_store = InMemoryPushNotificationConfigStore()
    request_handler = DefaultRequestHandler(
        agent_executor=HandlerExecutor(card.name, handler, stream_handler),
        task_store=InMemoryTaskStore(),
        push_config_store=push_store,
        push_sender=BasePushNotificationSender(httpx.AsyncClient(timeout=15.0), push_store),
    )

    a2a_app = A2AStarletteApplication(agent_card=card, http_handler=request_handler)
    # The specification names /.well-known/agent.json; the SDK defaults to
    # /.well-known/agent-card.json. Serve the specified path and alias the other.
    app = a2a_app.build(agent_card_url="/.well-known/agent.json")

    async def agent_card_alias(request):
        return JSONResponse(card.model_dump(mode="json", exclude_none=True, by_alias=True))

    async def health(request):
        return JSONResponse(
            {
                "status": "ok",
                "agent": card.name,
                "streaming": card.capabilities.streaming,
                "url": card.url,
            }
        )

    app.router.routes.append(Route("/.well-known/agent-card.json", agent_card_alias, methods=["GET"]))
    app.router.routes.append(Route("/health", health, methods=["GET"]))

    app.add_middleware(
        SharedSecretAuthMiddleware,
        token=settings.agent_auth_token,
        agent_name=card.name,
    )

    _log.info(
        "A2A application built",
        extra={"agent": card.name, "url": card.url, "streaming": card.capabilities.streaming},
    )
    return app


def serve(
    card: AgentCard,
    port: int,
    handler: Handler | None = None,
    stream_handler: StreamHandler | None = None,
) -> None:
    app = build_app(card, handler, stream_handler)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=get_settings().log_level.lower())

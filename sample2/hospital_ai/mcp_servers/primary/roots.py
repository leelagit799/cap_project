"""MCP Roots enforcement — doc §2.1 and Table 9.

The specification is precise about how the Watcher Tool may learn where files
live:

- the agent registers the input folder as a Root URI when opening the MCP
  connection;
- the tool calls ``ctx.list_roots()`` to discover authorised folders at
  runtime, so no raw path is ever passed as a tool parameter;
- the server rejects requests outside the declared root using
  ``Path.relative_to()``.

Failure to negotiate a root is a hard error. Falling back to a configured path
would defeat the primitive being demonstrated, so ``resolve_roots`` only uses
the local workspace when the client is a non-Roots caller *and* the caller
explicitly opts in via ``allow_local_fallback`` (used by the unit tests).
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from hospital_ai.core.config import get_settings
from hospital_ai.core.errors import RootAccessDeniedError
from hospital_ai.core.logging import get_logger

_log = get_logger(__name__, component="mcp-roots")


def uri_to_path(uri: str) -> Path:
    """Convert a ``file://`` Root URI to a local path."""
    parsed = urlparse(str(uri))
    if parsed.scheme and parsed.scheme != "file":
        raise RootAccessDeniedError(f"Unsupported root scheme: {parsed.scheme!r}")
    return Path(unquote(parsed.path or str(uri)))


async def resolve_roots(ctx, *, allow_local_fallback: bool = False) -> list[Path]:
    """Discover the authorised workspaces for this MCP session.

    Calls ``ctx.session.list_roots()`` — the client's declared Roots are the
    only source of truth.
    """
    roots: list[Path] = []
    try:
        result = await ctx.session.list_roots()
        for root in result.roots:
            path = uri_to_path(root.uri)
            if path.is_dir():
                roots.append(path.resolve())
            else:
                _log.warning("declared root is not a directory", extra={"root": str(path)})
    except Exception as exc:  # noqa: BLE001 - any failure means "no roots"
        _log.warning("roots negotiation failed", extra={"error": str(exc)})

    if roots:
        return roots

    if allow_local_fallback:
        return [get_settings().roots.workspace.resolve()]

    raise RootAccessDeniedError(
        "No MCP Roots were declared by the client. The Clinical Watcher Tool "
        "refuses to read the filesystem without an authorised root."
    )


def ensure_within_roots(candidate: Path, roots: list[Path]) -> Path:
    """Return ``candidate`` resolved, or raise if it escapes every root.

    ``Path.relative_to()`` is the check the specification names. Both sides are
    resolved first so ``..`` segments and symlinks cannot slip past it.
    """
    resolved = candidate.resolve()
    for root in roots:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return resolved

    _log.error(
        "path traversal attempt rejected",
        extra={"candidate": str(candidate), "roots": [str(r) for r in roots]},
    )
    raise RootAccessDeniedError(
        f"Path {candidate} lies outside every declared root "
        f"({', '.join(str(r) for r in roots)})"
    )


def relative_uri(path: Path, roots: list[Path]) -> str:
    """Root-relative identifier for a file. Absolute paths never leave the tool."""
    resolved = path.resolve()
    for root in roots:
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    raise RootAccessDeniedError(f"{path} is not inside any declared root")


def resolve_relative(uri: str, roots: list[Path]) -> Path:
    """Turn a root-relative URI back into a validated absolute path.

    This is the reverse of :func:`relative_uri` and the only way a caller can
    name a file: it passes the relative identifier the Watcher handed out, and
    the server re-validates it against the live roots.
    """
    candidate = Path(uri)
    if candidate.is_absolute():
        raise RootAccessDeniedError(
            "Absolute paths are not accepted as tool parameters; pass the "
            "root-relative URI returned by the Clinical Watcher Tool."
        )
    for root in roots:
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.exists():
            return resolved
    raise RootAccessDeniedError(f"{uri} does not resolve inside any declared root")

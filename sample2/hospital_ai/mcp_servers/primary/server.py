"""Primary MCP Clinical Tools Server — port 8200, path /clinicaltools.

Doc Table 8 requires this server to demonstrate all six MCP primitives:

- **Tools** — the six clinical tools of Table 7
- **Resources** — the six URIs of Table 1
- **Prompts** — the five templates of Table 2
- **Sampling** — Medical Lang Bridge Tool, via ctx.session.create_message()
- **Elicitation** — Clinical Rules Engine Tool, via ctx.elicit()
- **Roots** — Clinical Watcher Tool, via ctx.list_roots()
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import configure_logging, get_logger
from hospital_ai.mcp_servers.primary import prompts, resources
from hospital_ai.mcp_servers.primary.tools import (
    ehr_validator,
    harvester,
    lang_bridge,
    reporter,
    rules_engine,
    watcher,
)

_log = get_logger(__name__, service="primary-mcp")

INSTRUCTIONS = """\
Primary Clinical Tools Server for DischargeFlow.

Workflow: call clinical_watcher to discover discharge packets inside your
declared MCP Roots, clinical_data_harvester to extract each document,
medical_lang_bridge to translate non-English content via MCP Sampling,
clinical_rules_engine to check completeness (it may elicit missing fields from
your reviewer), ehr_validation to cross-check against the Mock EHR, and
clinical_insight_reporter to produce the audit report.

Filesystem access is Roots-scoped. Tools accept only the root-relative URIs
returned by clinical_watcher; absolute paths are rejected.
"""


def create_server() -> FastMCP:
    settings = get_settings()
    configure_logging(settings.log_level, settings.reports_dir / "pipeline.log")
    settings.ensure_dirs()

    mcp = FastMCP(
        name="clinical-tools",
        instructions=INSTRUCTIONS,
        host="0.0.0.0",
        port=settings.ports.primary_mcp,
        streamable_http_path="/clinicaltools",
    )

    resources.register(mcp)
    prompts.register(mcp)

    watcher.register(mcp)
    harvester.register(mcp)
    lang_bridge.register(mcp)
    rules_engine.register(mcp)
    ehr_validator.register(mcp)
    reporter.register(mcp)

    _log.info(
        "primary MCP server constructed",
        extra={"port": settings.ports.primary_mcp, "path": "/clinicaltools"},
    )
    return mcp


def main() -> None:
    create_server().run(transport="streamable-http")


if __name__ == "__main__":
    main()

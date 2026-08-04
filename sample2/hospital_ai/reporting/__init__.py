"""Audit and risk report generation — JSON, HTML and PDF."""

from hospital_ai.reporting.builder import ReportArtifacts, build_payload, generate, render_html

__all__ = ["ReportArtifacts", "build_payload", "generate", "render_html"]

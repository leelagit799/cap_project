"""Secondary MCP Analytics Server — port 8201, path /analyticstools.

Doc Table 8 fixes this server's three tools: ``calculate_risk_score``,
``get_population_benchmarks`` and ``generate_risk_heatmap``. It exposes the
Tools primitive only; the other five primitives live on the primary server.

Keeping risk scoring here — while the Reporter on :8200 calls the same
``hospital_ai.analytics.risk`` engine — is what makes the multi-server MCP
connection in doc §4 load-bearing rather than decorative: the Validation Agent
must hold sessions with both servers to complete a case.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from mcp.server.fastmcp import FastMCP

from hospital_ai.analytics.risk import assess
from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import configure_logging, get_logger

_log = get_logger(__name__, service="analytics-mcp")

INSTRUCTIONS = """\
Secondary Analytics Server for DischargeFlow. Computes composite discharge risk
scores from validation findings, reports population readmission benchmarks, and
renders a risk heatmap across a cohort of cases.
"""

#: Simulated population statistics for the benchmark tool. Real deployments
#: would source these from the hospital's data warehouse.
POPULATION_BENCHMARKS: dict[str, dict[str, float | str]] = {
    "E11.9": {"diagnosis": "Type 2 Diabetes Mellitus", "readmission_30d_rate": 0.094,
              "avg_length_of_stay_days": 3.4, "national_benchmark": 0.102},
    "I10": {"diagnosis": "Hypertension", "readmission_30d_rate": 0.061,
            "avg_length_of_stay_days": 2.6, "national_benchmark": 0.068},
    "I50.9": {"diagnosis": "Congestive Heart Failure", "readmission_30d_rate": 0.213,
              "avg_length_of_stay_days": 5.1, "national_benchmark": 0.221},
    "I21.9": {"diagnosis": "Acute Myocardial Infarction", "readmission_30d_rate": 0.168,
              "avg_length_of_stay_days": 4.8, "national_benchmark": 0.174},
    "J18.9": {"diagnosis": "Pneumonia", "readmission_30d_rate": 0.152,
              "avg_length_of_stay_days": 4.2, "national_benchmark": 0.161},
    "J44.9": {"diagnosis": "COPD exacerbation", "readmission_30d_rate": 0.196,
              "avg_length_of_stay_days": 4.6, "national_benchmark": 0.203},
    "J45.909": {"diagnosis": "Asthma exacerbation", "readmission_30d_rate": 0.073,
                "avg_length_of_stay_days": 2.3, "national_benchmark": 0.081},
    "N39.0": {"diagnosis": "Urinary tract infection", "readmission_30d_rate": 0.088,
              "avg_length_of_stay_days": 2.9, "national_benchmark": 0.094},
    "K35.80": {"diagnosis": "Acute appendicitis", "readmission_30d_rate": 0.042,
               "avg_length_of_stay_days": 2.1, "national_benchmark": 0.049},
    "N10": {"diagnosis": "Acute pyelonephritis", "readmission_30d_rate": 0.117,
            "avg_length_of_stay_days": 3.8, "national_benchmark": 0.125},
}


def calculate_risk(
    findings: list[dict[str, Any]],
    translation_confidence: float | None = None,
    service_line: str | None = None,
) -> dict[str, Any]:
    assessment = assess(
        findings,
        translation_confidence=translation_confidence,
        service_line=service_line,
    )
    settings = get_settings()
    thresholds = settings.rules["risk_scoring_matrix"]["thresholds"]
    result = assessment.to_dict()
    result["thresholds"] = thresholds
    result["rules_version"] = settings.rules_version
    return result


def population_benchmarks(icd10_codes: list[str] | None = None) -> dict[str, Any]:
    codes = [c.strip().upper() for c in (icd10_codes or []) if c.strip()]
    selected = (
        {code: POPULATION_BENCHMARKS[code] for code in codes if code in POPULATION_BENCHMARKS}
        if codes
        else dict(POPULATION_BENCHMARKS)
    )
    unknown = [code for code in codes if code not in POPULATION_BENCHMARKS]

    rates = [float(v["readmission_30d_rate"]) for v in selected.values()]
    return {
        "benchmarks": selected,
        "unknown_codes": unknown,
        "cohort_size": len(selected),
        "mean_readmission_30d_rate": round(sum(rates) / len(rates), 4) if rates else None,
        "source": "simulated population statistics",
    }


def risk_heatmap(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a cohort into a service-line by risk-level heatmap."""
    levels = ["Low", "Medium", "High"]
    service_lines = sorted({(c.get("service_line") or "Unknown") for c in cases})

    matrix = {line: {level: 0 for level in levels} for line in service_lines}
    rule_counter: Counter[str] = Counter()
    blocked = 0

    for case in cases:
        line = case.get("service_line") or "Unknown"
        level = case.get("risk_level") or "Low"
        if level in matrix[line]:
            matrix[line][level] += 1
        if case.get("discharge_blocked"):
            blocked += 1
        for finding in case.get("findings", []) or []:
            rule_counter[str(finding.get("rule_id"))] += 1

    return {
        "service_lines": service_lines,
        "risk_levels": levels,
        "matrix": matrix,
        "cells": [
            {"service_line": line, "risk_level": level, "count": matrix[line][level]}
            for line in service_lines
            for level in levels
        ],
        "total_cases": len(cases),
        "blocked_cases": blocked,
        "block_rate": round(blocked / len(cases), 3) if cases else 0.0,
        "top_rules": [
            {"rule_id": rule, "count": count} for rule, count in rule_counter.most_common(10)
        ],
    }


def create_server() -> FastMCP:
    settings = get_settings()
    configure_logging(settings.log_level, settings.reports_dir / "pipeline.log")

    mcp = FastMCP(
        name="analytics-tools",
        instructions=INSTRUCTIONS,
        host="0.0.0.0",
        port=settings.ports.analytics_mcp,
        streamable_http_path="/analyticstools",
    )

    @mcp.tool(
        name="calculate_risk_score",
        description=(
            "Compute the composite discharge risk score from validation "
            "findings, applying the rules.yaml weight matrix, the Low/Medium/"
            "High thresholds and the hard HITL guardrails."
        ),
    )
    def calculate_risk_score(
        findings: list[dict[str, Any]],
        translation_confidence: float | None = None,
        service_line: str | None = None,
    ) -> dict[str, Any]:
        return calculate_risk(findings, translation_confidence, service_line)

    @mcp.tool(
        name="get_population_benchmarks",
        description="Return 30-day readmission rates, average length of stay "
        "and national benchmarks for the given ICD-10 codes.",
    )
    def get_population_benchmarks(icd10_codes: list[str] | None = None) -> dict[str, Any]:
        return population_benchmarks(icd10_codes)

    @mcp.tool(
        name="generate_risk_heatmap",
        description="Aggregate a cohort of scored cases into a service-line by "
        "risk-level heatmap with the most frequent rule violations.",
    )
    def generate_risk_heatmap(cases: list[dict[str, Any]]) -> dict[str, Any]:
        return risk_heatmap(cases)

    _log.info(
        "analytics MCP server constructed",
        extra={"port": settings.ports.analytics_mcp, "path": "/analyticstools"},
    )
    return mcp


def main() -> None:
    create_server().run(transport="streamable-http")


if __name__ == "__main__":
    main()

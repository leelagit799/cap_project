"""Unit tests for resolved completeness findings after HITL edits."""

from __future__ import annotations

from hospital_ai.mcp_servers.primary.tools.rules_engine import refresh_resolved_findings


def test_refresh_resolved_findings_closes_address_gap():
    packet = {
        "discharge_report": {"address": "14 Lakeview Road, Mumbai"},
        "lab_report": {},
        "bill": {},
    }
    findings = [
        {
            "rule_id": "missing_field.discharge_report.address",
            "resolved": False,
            "blocking": False,
        }
    ]

    refresh_resolved_findings(packet, findings)

    assert findings[0]["resolved"] is True

"""Operational persistence — SQLite (doc Table 14).

Holds the case state machine, clinical records, validation runs, findings,
elicitations, HITL reviews, summaries and an append-only audit trail.

Validation is stored per run rather than overwritten: a HITL correction
re-runs validation, and the compliance question "what did the system see
before the human intervened?" has to remain answerable.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from hospital_ai.core.config import get_settings
from hospital_ai.core.ids import utc_now_iso
from hospital_ai.core.logging import get_logger
from hospital_ai.core.schemas import CaseStatus, ValidationResult

_log = get_logger(__name__, component="case-store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id            TEXT PRIMARY KEY,
    patient_id         TEXT NOT NULL,
    trace_id           TEXT NOT NULL,
    status             TEXT NOT NULL,
    risk_level         TEXT,
    risk_score         INTEGER,
    discharge_blocked  INTEGER DEFAULT 0,
    rules_version      TEXT,
    record_json        TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cases_patient ON cases(patient_id);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);

CREATE TABLE IF NOT EXISTS validations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id            TEXT NOT NULL REFERENCES cases(case_id),
    run_no             INTEGER NOT NULL,
    payload_json       TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    UNIQUE(case_id, run_no)
);

CREATE TABLE IF NOT EXISTS findings (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id            TEXT NOT NULL,
    run_no             INTEGER NOT NULL,
    rule_id            TEXT NOT NULL,
    severity           TEXT NOT NULL,
    field              TEXT,
    message            TEXT,
    weight             INTEGER,
    blocking           INTEGER,
    resolved           INTEGER
);
CREATE INDEX IF NOT EXISTS idx_findings_case ON findings(case_id, run_no);

CREATE TABLE IF NOT EXISTS hitl_reviews (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id            TEXT NOT NULL,
    reviewer           TEXT,
    decision           TEXT,
    risk_override      TEXT,
    corrections_json   TEXT,
    notes              TEXT,
    created_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id            TEXT NOT NULL,
    payload_json       TEXT NOT NULL,
    created_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_events (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id            TEXT,
    actor              TEXT NOT NULL,
    action             TEXT NOT NULL,
    detail             TEXT,
    payload_json       TEXT,
    created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_case ON audit_events(case_id, created_at);
"""


class CaseStore:
    def __init__(self, db_path: Path | None = None) -> None:
        settings = get_settings()
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (settings.state_dir / "dischargeflow.sqlite")
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    # --- cases ---------------------------------------------------------------

    def create_case(self, case_id: str, patient_id: str, trace_id: str) -> None:
        now = utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cases (case_id, patient_id, trace_id, status,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?)",
                (case_id, patient_id, trace_id, CaseStatus.CREATED.value, now, now),
            )
            conn.commit()
        self.audit(case_id, "host-orchestrator", "case_created", f"patient {patient_id}")

    def update_status(
        self,
        case_id: str,
        status: CaseStatus,
        *,
        validation: ValidationResult | None = None,
        record: dict[str, Any] | None = None,
    ) -> None:
        fields: list[str] = ["status = ?", "updated_at = ?"]
        values: list[Any] = [status.value, utc_now_iso()]

        if validation is not None:
            fields += ["risk_level = ?", "risk_score = ?", "discharge_blocked = ?", "rules_version = ?"]
            values += [
                validation.risk_level.value,
                validation.risk_score,
                int(validation.discharge_blocked),
                validation.rules_version,
            ]
        if record is not None:
            fields.append("record_json = ?")
            values.append(json.dumps(record, default=str, ensure_ascii=False))

        values.append(case_id)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE cases SET {', '.join(fields)} WHERE case_id = ?", values)
            conn.commit()
        self.audit(case_id, "host-orchestrator", "status_changed", status.value)

    def save_record(self, case_id: str, record: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE cases SET record_json = ?, updated_at = ? WHERE case_id = ?",
                (json.dumps(record, default=str, ensure_ascii=False), utc_now_iso(), case_id),
            )
            conn.commit()

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        return dict(row) if row else None

    def get_record(self, case_id: str) -> dict[str, Any] | None:
        case = self.get_case(case_id)
        if case is None or not case.get("record_json"):
            return None
        return json.loads(case["record_json"])

    def delete_patient_cases(self, patient_id: str) -> list[str]:
        """Remove all dashboard workflow data for a patient."""
        cases = self.list_cases(patient_id=patient_id)
        case_ids = [row["case_id"] for row in cases]
        if not case_ids:
            return []

        with self._lock, self._connect() as conn:
            for case_id in case_ids:
                conn.execute("DELETE FROM validations WHERE case_id = ?", (case_id,))
                conn.execute("DELETE FROM findings WHERE case_id = ?", (case_id,))
                conn.execute("DELETE FROM hitl_reviews WHERE case_id = ?", (case_id,))
                conn.execute("DELETE FROM summaries WHERE case_id = ?", (case_id,))
                conn.execute("DELETE FROM audit_events WHERE case_id = ?", (case_id,))
                conn.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))
            conn.commit()
        self.audit(
            None,
            "patient-documents",
            "patient_cases_cleared",
            f"{patient_id}: {len(case_ids)} case(s)",
        )
        return case_ids

    def list_cases(self, status: str | None = None, patient_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM cases"
        clauses, values = [], []
        if status:
            clauses.append("status = ?")
            values.append(status)
        if patient_id:
            clauses.append("patient_id = ?")
            values.append(patient_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"

        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query, values).fetchall()]

    # --- validations ---------------------------------------------------------

    def save_validation(self, case_id: str, validation: ValidationResult) -> None:
        payload = validation.model_dump(mode="json")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO validations (case_id, run_no, payload_json, created_at)"
                " VALUES (?,?,?,?)",
                (case_id, validation.run_no, json.dumps(payload, default=str), utc_now_iso()),
            )
            conn.execute(
                "DELETE FROM findings WHERE case_id = ? AND run_no = ?",
                (case_id, validation.run_no),
            )
            conn.executemany(
                "INSERT INTO findings (case_id, run_no, rule_id, severity, field, message,"
                " weight, blocking, resolved) VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        case_id, validation.run_no, f.rule_id, f.severity.value, f.field,
                        f.message, f.weight, int(f.blocking), int(f.resolved),
                    )
                    for f in validation.findings
                ],
            )
            conn.execute(
                "UPDATE cases SET record_json = COALESCE(record_json, record_json) WHERE case_id = ?",
                (case_id,),
            )
            conn.commit()
        self.audit(
            case_id,
            "clinical-validator",
            "validation_saved",
            f"run {validation.run_no}: {validation.risk_level.value}",
        )

    def get_validation(self, case_id: str, run_no: int | None = None) -> dict[str, Any] | None:
        with self._connect() as conn:
            if run_no is None:
                row = conn.execute(
                    "SELECT payload_json FROM validations WHERE case_id = ?"
                    " ORDER BY run_no DESC LIMIT 1",
                    (case_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT payload_json FROM validations WHERE case_id = ? AND run_no = ?",
                    (case_id, run_no),
                ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def validation_runs(self, case_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_no, payload_json, created_at FROM validations"
                " WHERE case_id = ? ORDER BY run_no",
                (case_id,),
            ).fetchall()
        return [
            {"run_no": row["run_no"], "created_at": row["created_at"],
             **json.loads(row["payload_json"])}
            for row in rows
        ]

    def next_run_no(self, case_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(run_no) AS last FROM validations WHERE case_id = ?", (case_id,)
            ).fetchone()
        return int((row["last"] or 0) + 1)

    # --- HITL ----------------------------------------------------------------

    def apply_corrections(self, case_id: str, corrections: dict[str, Any]) -> dict[str, Any]:
        """Merge reviewer edits into the stored record.

        Keys are dotted paths into the record, for example
        ``discharge_report.address`` or ``bill.payment_status``.
        """
        record = self.get_record(case_id)
        if record is None:
            raise KeyError(f"No record stored for {case_id}")

        for path, value in corrections.items():
            target = record
            parts = path.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value

        self.save_record(case_id, record)
        self.audit(
            case_id, "hitl-reviewer", "corrections_applied", ", ".join(sorted(corrections))
        )
        return record

    def save_review(
        self,
        case_id: str,
        *,
        reviewer: str,
        decision: str,
        risk_override: str | None = None,
        corrections: dict[str, Any] | None = None,
        notes: str = "",
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO hitl_reviews (case_id, reviewer, decision, risk_override,"
                " corrections_json, notes, created_at) VALUES (?,?,?,?,?,?,?)",
                (
                    case_id, reviewer, decision, risk_override,
                    json.dumps(corrections or {}, default=str), notes, utc_now_iso(),
                ),
            )
            conn.commit()
        self.audit(case_id, reviewer, "hitl_decision", decision)

    def get_reviews(self, case_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hitl_reviews WHERE case_id = ? ORDER BY created_at", (case_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    # --- summaries -----------------------------------------------------------

    def save_summary(self, case_id: str, summary: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO summaries (case_id, payload_json, created_at) VALUES (?,?,?)",
                (case_id, json.dumps(summary, default=str, ensure_ascii=False), utc_now_iso()),
            )
            conn.commit()
        self.audit(case_id, "summary-generator", "summary_generated", "")

    def get_summary(self, case_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM summaries WHERE case_id = ?"
                " ORDER BY created_at DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    # --- audit ---------------------------------------------------------------

    def audit(
        self,
        case_id: str | None,
        actor: str,
        action: str,
        detail: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append-only. There is no update or delete path, by design."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_events (case_id, actor, action, detail, payload_json,"
                " created_at) VALUES (?,?,?,?,?,?)",
                (
                    case_id, actor, action, detail,
                    json.dumps(payload or {}, default=str), utc_now_iso(),
                ),
            )
            conn.commit()

    def audit_trail(self, case_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_events WHERE case_id = ? ORDER BY created_at, id",
                (case_id,),
            ).fetchall()
        return [
            {"at": row["created_at"], "actor": row["actor"], "action": row["action"],
             "detail": row["detail"]}
            for row in rows
        ]

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM cases").fetchone()["n"]
            by_status = {
                row["status"]: row["n"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS n FROM cases GROUP BY status"
                ).fetchall()
            }
            by_risk = {
                row["risk_level"]: row["n"]
                for row in conn.execute(
                    "SELECT risk_level, COUNT(*) AS n FROM cases"
                    " WHERE risk_level IS NOT NULL GROUP BY risk_level"
                ).fetchall()
            }
            blocked = conn.execute(
                "SELECT COUNT(*) AS n FROM cases WHERE discharge_blocked = 1"
            ).fetchone()["n"]
        return {
            "total_cases": total,
            "by_status": by_status,
            "by_risk_level": by_risk,
            "blocked_cases": blocked,
        }


_STORE: CaseStore | None = None


def get_store() -> CaseStore:
    global _STORE
    if _STORE is None:
        _STORE = CaseStore()
    return _STORE


def reset_store() -> None:
    global _STORE
    _STORE = None


__all__ = ["CaseStore", "get_store", "reset_store"]

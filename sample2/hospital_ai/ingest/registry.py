"""Persistent patient-id allocation for dynamic uploads."""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from hospital_ai.core.config import get_settings
from hospital_ai.core.ids import PATIENT_ID_PATTERN, utc_now_iso
from hospital_ai.core.logging import get_logger

_log = get_logger(__name__, component="upload-registry")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS upload_patients (
    patient_id   TEXT PRIMARY KEY,
    doctor_name  TEXT,
    folder       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_upload_patients_created ON upload_patients(created_at);
"""

_MIN_PATIENT_NUM = 1001


def _patient_num(patient_id: str) -> int | None:
    match = re.fullmatch(r"P(\d{4})", patient_id)
    return int(match.group(1)) if match else None


def _format_patient_id(number: int) -> str:
    return f"P{number:04d}"


class PatientUploadRegistry:
    """Allocates monotonic ``P####`` ids and tracks upload folders."""

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

    def _scan_workspace_ids(self) -> set[str]:
        settings = get_settings()
        ids: set[str] = set()
        for root in (settings.roots.workspace, settings.upload_dir):
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    match = PATIENT_ID_PATTERN.search(path.name)
                    if match:
                        ids.add(match.group(0))
                elif path.is_dir() and _patient_num(path.name):
                    ids.add(path.name)
        return ids

    def _seed_max(self) -> int:
        """Highest ``P####`` seen in the registry, workspace, or shipped samples."""
        numbers = [_MIN_PATIENT_NUM - 1]
        with self._connect() as conn:
            rows = conn.execute("SELECT patient_id FROM upload_patients").fetchall()
        for row in rows:
            num = _patient_num(row["patient_id"])
            if num is not None:
                numbers.append(num)
        for patient_id in self._scan_workspace_ids():
            num = _patient_num(patient_id)
            if num is not None:
                numbers.append(num)
        return max(numbers)

    def allocate_patient_id(self, *, doctor_name: str | None = None) -> dict[str, Any]:
        """Reserve the next unused patient id."""
        settings = get_settings()
        with self._lock, self._connect() as conn:
            next_num = self._seed_max() + 1
            while True:
                patient_id = _format_patient_id(next_num)
                exists = conn.execute(
                    "SELECT 1 FROM upload_patients WHERE patient_id = ?", (patient_id,)
                ).fetchone()
                folder = settings.upload_dir / patient_id
                if not exists and not folder.exists():
                    break
                next_num += 1

            now = utc_now_iso()
            folder.mkdir(parents=True, exist_ok=True)
            conn.execute(
                "INSERT INTO upload_patients (patient_id, doctor_name, folder, created_at, updated_at)"
                " VALUES (?,?,?,?,?)",
                (patient_id, doctor_name, str(folder), now, now),
            )
            conn.commit()

        _log.info("patient allocated", extra={"patient_id": patient_id})
        return {"patient_id": patient_id, "folder": str(folder)}

    def list_patients(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM upload_patients ORDER BY patient_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_patient(self, patient_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM upload_patients WHERE patient_id = ?", (patient_id,)
            ).fetchone()
        return dict(row) if row else None

    def touch(self, patient_id: str, *, doctor_name: str | None = None) -> None:
        now = utc_now_iso()
        with self._lock, self._connect() as conn:
            if doctor_name is not None:
                conn.execute(
                    "UPDATE upload_patients SET doctor_name = ?, updated_at = ? WHERE patient_id = ?",
                    (doctor_name, now, patient_id),
                )
            else:
                conn.execute(
                    "UPDATE upload_patients SET updated_at = ? WHERE patient_id = ?",
                    (now, patient_id),
                )
            conn.commit()

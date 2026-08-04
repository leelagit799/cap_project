"""Data access for the Mock EHR service.

Reads the five JSON files from ``mock_ehr/data/``, generating them from
``mock_ehr/data.py`` on first run. The JSON files are the read path so the
service matches the "5 JSON data files" description in doc Table 14; nothing
else in the system imports ``mock_ehr.data`` directly.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from mock_ehr.export_json import DATA_DIR, ensure_exported


class EHRRepository:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or DATA_DIR
        ensure_exported(self.data_dir)
        self._cache: dict[str, Any] = {}

    def _dataset(self, name: str) -> dict[str, Any]:
        if name not in self._cache:
            path = self.data_dir / f"{name}.json"
            if not path.is_file():
                ensure_exported(self.data_dir)
            self._cache[name] = json.loads(path.read_text(encoding="utf-8"))
        return self._cache[name]

    def reload(self) -> None:
        self._cache.clear()

    # --- Patients ------------------------------------------------------------

    def list_patients(self, service_line: str | None = None) -> list[dict[str, Any]]:
        patients = list(self._dataset("patients").values())
        if service_line:
            wanted = service_line.casefold()
            patients = [p for p in patients if (p.get("service_line") or "").casefold() == wanted]
        return sorted(patients, key=lambda p: p["patient_id"])

    def get_patient(self, patient_id: str) -> dict[str, Any] | None:
        return self._dataset("patients").get(patient_id)

    def patient_exists(self, patient_id: str) -> bool:
        return patient_id in self._dataset("patients")

    # --- Clinical datasets ---------------------------------------------------

    def get_allergies(self, patient_id: str) -> list[str]:
        return list(self._dataset("allergies").get(patient_id, []))

    def get_medications(self, patient_id: str) -> list[dict[str, Any]]:
        return list(self._dataset("medications").get(patient_id, []))

    def get_labs(self, patient_id: str) -> list[dict[str, Any]]:
        return list(self._dataset("labs").get(patient_id, []))

    def get_abnormal_labs(self, patient_id: str) -> list[dict[str, Any]]:
        return [lab for lab in self.get_labs(patient_id) if lab.get("abnormal")]

    def get_care_plan(self, patient_id: str) -> dict[str, Any] | None:
        return self._dataset("care_plans").get(patient_id)

    def get_guideline(self, icd10: str) -> dict[str, Any] | None:
        return self._dataset("guidelines").get(icd10.upper())

    def list_guidelines(self) -> dict[str, Any]:
        return dict(self._dataset("guidelines"))

    # --- Aggregate -----------------------------------------------------------

    def get_bundle(self, patient_id: str) -> dict[str, Any] | None:
        """Everything the Validation Agent needs in one round trip."""
        patient = self.get_patient(patient_id)
        if patient is None:
            return None
        guidelines = {
            code: guideline
            for code in patient.get("primary_dx", [])
            if (guideline := self.get_guideline(code)) is not None
        }
        return {
            "patient": patient,
            "allergies": self.get_allergies(patient_id),
            "medications": self.get_medications(patient_id),
            "labs": self.get_labs(patient_id),
            "care_plan": self.get_care_plan(patient_id),
            "guidelines": guidelines,
        }


@lru_cache(maxsize=1)
def get_repository() -> EHRRepository:
    return EHRRepository()

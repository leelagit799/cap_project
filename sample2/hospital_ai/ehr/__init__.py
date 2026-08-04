"""Mock EHR System — FastAPI service on port 8050."""

from hospital_ai.ehr.repository import EHRRepository, get_repository

__all__ = ["EHRRepository", "get_repository"]

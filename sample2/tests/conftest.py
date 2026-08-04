"""Shared pytest fixtures.

A dev shared secret is injected before ``hospital_ai.core.config`` is imported,
so the suite runs without a populated ``.env``. Config rejects the placeholder
token in ``agent_config.yaml`` by design.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("AGENT_AUTH_TOKEN", "test-shared-secret")
os.environ.setdefault("LLM_OFFLINE", "1")
os.environ.setdefault("MCP_USE_ANONYMIZED_TELEMETRY", "false")

import pytest  # noqa: E402

from hospital_ai.core.config import get_settings  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def ehr_client():
    from fastapi.testclient import TestClient

    from hospital_ai.ehr.app import create_app

    with TestClient(create_app()) as client:
        yield client

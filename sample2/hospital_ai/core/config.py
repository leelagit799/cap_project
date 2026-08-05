"""Configuration loading.

Three sources, in precedence order:

1. Environment variables (secrets only — never anything committed to the repo).
2. ``configs/*.yaml`` (ports, thresholds, clinical rules).
3. Defaults declared here.

Secrets are read exclusively from the environment. Nothing in ``configs/`` may
contain a credential.
"""

from __future__ import annotations

import functools
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hospital_ai.core.errors import ConfigError, MissingCredentialError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file without overriding real env vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")


def _read_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.is_file():
        raise ConfigError(f"Required config file is missing: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must parse to a mapping, got {type(data).__name__}")
    return data


@dataclass(frozen=True)
class Ports:
    """Service port map — doc Table 15."""

    ehr: int = 8050
    extractor: int = 8100
    validator: int = 8101
    normalizer: int = 8102
    monitor: int = 8103
    summary: int = 8104
    rag: int = 8105
    host: int = 8083
    primary_mcp: int = 8200
    analytics_mcp: int = 8201
    dashboard: int = 8501
    ingest: int = 8060


@dataclass(frozen=True)
class LLMSettings:
    """LiteLLM gateway settings.

    The document mandates AWS Bedrock Nova Lite as primary and Cohere
    Command R+ as fallback. Command R+ is reachable either through Bedrock or
    through Cohere's own API; ``cohere_api_key`` selects the latter when set.
    """

    primary_model: str = "bedrock/amazon.nova-lite-v1:0"
    fallback_model: str = "bedrock/cohere.command-r-plus-v1:0"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    aws_region: str = "us-east-1"
    cohere_api_key: str | None = None
    offline: bool = False
    request_timeout_s: int = 60
    max_output_tokens: int = 2048
    #: ModelPreferences hints the Medical Lang Bridge Tool advertises to
    #: sampling clients (doc §2.3).
    sampling_hint_non_english: str = "nova-lite"
    sampling_hint_english: str = "command-r-plus"

    @property
    def credentials_present(self) -> bool:
        return bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))


@dataclass(frozen=True)
class LangfuseSettings:
    public_key: str | None = None
    secret_key: str | None = None
    host: str = "https://cloud.langfuse.com"

    @property
    def enabled(self) -> bool:
        return bool(self.public_key and self.secret_key)


@dataclass(frozen=True)
class RootsSettings:
    """MCP Roots workspace.

    The document's Figure 1 shows ``data/input/P001/`` (per-patient folders);
    the shipped sample data uses ``Data/incoming/<doc_type>/`` instead. Both
    layouts are supported and selected by ``layout``.
    """

    workspace: Path = PROJECT_ROOT / "Data" / "incoming"
    layout: str = "by_doctype"  # or "by_patient"
    doctype_dirs: dict[str, str] = field(
        default_factory=lambda: {
            "discharge_report": "doctor_reports",
            "lab_report": "lab_reports",
            "bill": "bills",
        }
    )

    @property
    def uri(self) -> str:
        return self.workspace.resolve().as_uri()


@dataclass(frozen=True)
class Settings:
    ports: Ports
    llm: LLMSettings
    langfuse: LangfuseSettings
    roots: RootsSettings
    agent_auth_token: str
    log_level: str
    project_root: Path
    data_dir: Path
    reports_dir: Path
    vector_dir: Path
    sessions_dir: Path
    state_dir: Path
    upload_dir: Path
    upload_max_bytes: int
    rules: dict[str, Any]
    prompts: dict[str, Any]
    rules_version: str

    def require_credentials(self) -> None:
        """Raise unless live LLM credentials are usable."""
        if self.llm.offline:
            return
        if not self.llm.credentials_present:
            raise MissingCredentialError(
                "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are required for live "
                "Bedrock inference. Set LLM_OFFLINE=1 to use the deterministic stub."
            )

    def ensure_dirs(self) -> None:
        for directory in (
            self.data_dir,
            self.reports_dir,
            self.vector_dir,
            self.sessions_dir,
            self.state_dir,
            self.upload_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _rules_version(rules_path: Path) -> str:
    """SHA-256 of rules.yaml, stamped onto every audit report (doc §2.5)."""
    return hashlib.sha256(rules_path.read_bytes()).hexdigest()


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    agent_cfg = _read_yaml("agent_config.yaml")
    rules = _read_yaml("rules.yaml")
    prompts = _read_yaml("prompts.yaml")

    ports = Ports(**{**Ports().__dict__, **(agent_cfg.get("ports") or {})})

    llm_cfg = agent_cfg.get("llm") or {}
    hints = llm_cfg.get("sampling_hints") or {}
    llm = LLMSettings(
        primary_model=os.getenv("BEDROCK_PRIMARY_MODEL")
        or llm_cfg.get("primary_model")
        or LLMSettings.primary_model,
        fallback_model=os.getenv("BEDROCK_FALLBACK_MODEL")
        or llm_cfg.get("fallback_model")
        or LLMSettings.fallback_model,
        embedding_model=llm_cfg.get("embedding") or LLMSettings.embedding_model,
        aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        cohere_api_key=os.getenv("COHERE_API_KEY") or None,
        offline=_env_bool("LLM_OFFLINE", default=False),
        sampling_hint_non_english=hints.get("non_english", "nova-lite"),
        sampling_hint_english=hints.get("english", "command-r-plus"),
    )

    langfuse = LangfuseSettings(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY") or None,
        secret_key=os.getenv("LANGFUSE_SECRET_KEY") or None,
        host=os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or LangfuseSettings.host,
    )

    roots_cfg = agent_cfg.get("roots") or {}
    workspace = Path(roots_cfg.get("workspace", "Data/incoming"))
    if not workspace.is_absolute():
        workspace = PROJECT_ROOT / workspace
    roots = RootsSettings(
        workspace=workspace,
        layout=roots_cfg.get("layout", "by_doctype"),
        doctype_dirs=roots_cfg.get("doctype_dirs") or RootsSettings().doctype_dirs,
    )

    token = os.getenv("AGENT_AUTH_TOKEN") or agent_cfg.get("agent_auth_token") or ""
    if not token or token == "change-this-for-production":
        raise ConfigError(
            "AGENT_AUTH_TOKEN must be set in the environment. The placeholder in "
            "agent_config.yaml is not a usable shared secret."
        )

    data_dir = PROJECT_ROOT / "data"
    upload_cfg = agent_cfg.get("upload") or {}
    upload_dir = Path(upload_cfg.get("workspace", "data/input"))
    if not upload_dir.is_absolute():
        upload_dir = PROJECT_ROOT / upload_dir
    upload_max_bytes = int(upload_cfg.get("max_file_bytes", 25 * 1024 * 1024))
    return Settings(
        ports=ports,
        llm=llm,
        langfuse=langfuse,
        roots=roots,
        agent_auth_token=token,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        reports_dir=data_dir / "reports",
        vector_dir=data_dir / "vector_db",
        sessions_dir=data_dir / "sessions",
        state_dir=data_dir / "state",
        upload_dir=upload_dir,
        upload_max_bytes=upload_max_bytes,
        rules=rules,
        prompts=prompts,
        rules_version=_rules_version(CONFIG_DIR / "rules.yaml"),
    )

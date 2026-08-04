#!/usr/bin/env python3
"""Start every DischargeFlow service.

Brings up all eleven processes of doc Table 15 in dependency order — the Mock
EHR and both MCP servers first, then the six A2A agents, then the two user
interfaces — and shuts them all down on Ctrl+C.

    python run.py                 # everything
    python run.py --only ehr mcp  # just the infrastructure tier
    python run.py --list          # show the service table
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from hospital_ai.core.config import get_settings  # noqa: E402


@dataclass
class Service:
    key: str
    name: str
    port: int
    command: list[str]
    tier: str
    url: str = ""


def build_services() -> list[Service]:
    settings = get_settings()
    ports = settings.ports
    python = sys.executable

    def module(name: str, *args: str) -> list[str]:
        return [python, "-m", name, *args]

    return [
        Service("ehr", "Mock EHR", ports.ehr, module("hospital_ai.ehr.app"),
                "infrastructure", f"http://localhost:{ports.ehr}/docs"),
        Service("mcp-primary", "Primary MCP Clinical Tools", ports.primary_mcp,
                module("hospital_ai.mcp_servers.primary.server"), "infrastructure",
                f"http://localhost:{ports.primary_mcp}/clinicaltools"),
        Service("mcp-analytics", "Secondary MCP Analytics", ports.analytics_mcp,
                module("hospital_ai.mcp_servers.analytics.server"), "infrastructure",
                f"http://localhost:{ports.analytics_mcp}/analyticstools"),

        Service("extractor", "Clinical Extractor (LangGraph)", ports.extractor,
                module("hospital_ai.agents.serve", "extractor"), "agents"),
        Service("validator", "Clinical Validation (LangGraph)", ports.validator,
                module("hospital_ai.agents.serve", "validator"), "agents"),
        Service("normalizer", "Clinical Normalizer (LangGraph)", ports.normalizer,
                module("hospital_ai.agents.serve", "normalizer"), "agents"),
        Service("monitor", "Discharge Monitor (ADK)", ports.monitor,
                module("hospital_ai.agents.serve", "monitor"), "agents"),
        Service("summary", "Summary Generator (ADK, streaming)", ports.summary,
                module("hospital_ai.agents.serve", "summary"), "agents"),
        Service("rag", "Clinical RAG Q&A (Agno, streaming)", ports.rag,
                module("hospital_ai.agents.serve", "rag"), "agents"),

        Service("host", "Host Orchestrator (Gradio)", ports.host,
                module("hospital_ai.ui.gradio_host.app"), "interfaces",
                f"http://localhost:{ports.host}"),
        Service(
            "dashboard", "HITL Dashboard (Streamlit)", ports.dashboard,
            [
                python, "-m", "streamlit", "run",
                str(PROJECT_ROOT / "hospital_ai" / "ui" / "streamlit_hitl" / "app.py"),
                "--server.port", str(ports.dashboard),
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false",
            ],
            "interfaces", f"http://localhost:{ports.dashboard}",
        ),
    ]


def port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def wait_for_port(port: int, timeout: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_is_free(port):
            return True
        time.sleep(0.3)
    return False


class Supervisor:
    def __init__(self, services: list[Service]) -> None:
        self.services = services
        self.processes: dict[str, subprocess.Popen] = {}
        self._log_dir = get_settings().reports_dir / "services"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def start(self, service: Service) -> bool:
        if not port_is_free(service.port):
            print(f"  ⚠ {service.name}: port {service.port} already in use, skipping")
            return False

        log_path = self._log_dir / f"{service.key}.log"
        handle = log_path.open("w", encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}

        process = subprocess.Popen(
            service.command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env
        )
        self.processes[service.key] = process

        if wait_for_port(service.port):
            print(f"  ✓ {service.name:42s} :{service.port}")
            return True

        # A dead port after the timeout almost always means an import error, so
        # surface the tail of the log rather than a bare timeout.
        print(f"  ✗ {service.name:42s} :{service.port} did not start")
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        for line in tail:
            print(f"      {line}")
        return False

    def start_all(self, tiers: list[str]) -> None:
        for tier in tiers:
            print(f"\n▸ {tier.title()}")
            for service in (s for s in self.services if s.tier == tier):
                self.start(service)

    def stop_all(self) -> None:
        print("\n▸ Shutting down")
        for key, process in reversed(list(self.processes.items())):
            if process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
            print(f"  ✓ stopped {key}")

    def watch(self) -> None:
        try:
            while True:
                time.sleep(2)
                for key, process in self.processes.items():
                    if process.poll() is not None:
                        print(
                            f"  ⚠ {key} exited with code {process.returncode}; "
                            f"see data/reports/services/{key}.log"
                        )
                        self.processes.pop(key, None)
                        break
        except KeyboardInterrupt:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the DischargeFlow stack")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["ehr", "mcp", "agents", "ui"],
        help="Start only selected tiers",
    )
    parser.add_argument("--list", action="store_true", help="List services and exit")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_dirs()
    services = build_services()

    if args.list:
        print(f"{'SERVICE':44s} {'PORT':6s} TIER")
        for service in services:
            print(f"{service.name:44s} {service.port:<6d} {service.tier}")
        return

    tiers = ["infrastructure", "agents", "interfaces"]
    if args.only:
        selected = set()
        if {"ehr", "mcp"} & set(args.only):
            selected.add("infrastructure")
        if "agents" in args.only:
            selected.add("agents")
        if "ui" in args.only:
            selected.add("interfaces")
        tiers = [tier for tier in tiers if tier in selected]

    print("=" * 74)
    print("  DischargeFlow — Agentic AI Hospital Discharge System")
    print("  St. Marian Regional Medical Center")
    print("=" * 74)

    if settings.llm.offline or not settings.llm.credentials_present:
        print("\n  ⚠ Running with the deterministic offline LLM stub.")
        print("    Set AWS credentials in .env for live Bedrock inference.")

    supervisor = Supervisor(services)
    signal.signal(signal.SIGTERM, lambda *_: (supervisor.stop_all(), sys.exit(0)))

    try:
        supervisor.start_all(tiers)
    except KeyboardInterrupt:
        supervisor.stop_all()
        return

    print("\n" + "=" * 74)
    print("  Open these:")
    for service in services:
        if service.url and service.key in supervisor.processes:
            print(f"    {service.name:42s} {service.url}")
    print("\n  Ctrl+C stops everything.")
    print("=" * 74)

    supervisor.watch()
    supervisor.stop_all()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Watch streaming A2A agents print progressive output in the terminal.

Table 10 defines two streaming agents:
  - summary (port 8104) — section-by-section discharge summary
  - rag       (port 8105) — token-by-token clinical Q&A

Prerequisites:
  1. Copy .env.example to .env and set AGENT_AUTH_TOKEN (or export it).
  2. Start the stack: ``python run.py``  OR at minimum the agent you want:
       python -m hospital_ai.agents.serve summary
       python -m hospital_ai.agents.serve rag
  3. For RAG, process a patient first so records are indexed.

Examples:
  python scripts/stream_demo.py rag --patient P1024 \\
      --question "What medications was this patient discharged on?"

  python scripts/stream_demo.py summary --patient P1024
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hospital_ai.a2a.client import A2AClient
from hospital_ai.storage import get_store


def _print_event(agent: str, chunk: object) -> None:
    if not isinstance(chunk, dict):
        print(chunk, flush=True)
        return

    if agent == "summary":
        if chunk.get("type") == "section":
            print(f"\n=== {chunk.get('title') or chunk.get('name')} ===", flush=True)
            print(chunk.get("content", ""), flush=True)
        elif chunk.get("section"):
            print(f"\n=== {chunk['section']} ===", flush=True)
            print(chunk.get("content", ""), flush=True)
        else:
            print(json.dumps(chunk, indent=2), flush=True)
        return

    if agent == "rag":
        event_type = chunk.get("type")
        if event_type == "token":
            print(chunk.get("text", ""), end="", flush=True)
        elif event_type == "sources":
            count = len(chunk.get("chunks") or [])
            print(f"[retrieved {count} source chunk(s)]", flush=True)
        elif event_type == "complete":
            print("\n\n--- answer complete ---", flush=True)
            print(chunk.get("answer", ""), flush=True)
            triad = chunk.get("triad") or {}
            if triad:
                print(
                    f"faithfulness={triad.get('faithfulness')} "
                    f"relevance={triad.get('answer_relevance')}",
                    flush=True,
                )
        elif event_type == "blocked":
            print(f"\n[blocked] {chunk.get('reason')}", flush=True)
        else:
            print(json.dumps(chunk, indent=2), flush=True)
        return

    print(json.dumps(chunk, indent=2), flush=True)


async def _load_summary_payload(patient_id: str) -> dict:
    store = get_store()
    cases = store.list_cases(patient_id=patient_id)
    if not cases:
        raise SystemExit(
            f"No case found for {patient_id}. Process the patient in the dashboard first."
        )
    case = cases[0]
    record = store.get_record(case["case_id"])
    if record is None:
        raise SystemExit(f"Case {case['case_id']} has no stored record.")
    validation = store.get_validation(case["case_id"]) or {}
    return {
        "case_id": case["case_id"],
        "record": record,
        "risk_level": validation.get("risk_level", "Low"),
        "discharge_blocked": validation.get("discharge_blocked", False),
    }


async def stream_agent(agent: str, payload: dict) -> None:
    client = A2AClient()
    card = await client.fetch_card(agent)
    streaming = card.get("capabilities", {}).get("streaming")
    print(f"Agent: {card.get('name')}  streaming={streaming}", flush=True)
    print("-" * 60, flush=True)

    async for chunk in client.send_message_streaming(agent, payload):
        _print_event(agent, chunk)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Stream A2A agent output in the terminal")
    parser.add_argument("agent", choices=["summary", "rag"], help="Streaming agent to invoke")
    parser.add_argument("--patient", default="P1024", help="Patient id (default: P1024)")
    parser.add_argument(
        "--question",
        default="What medications was this patient discharged on?",
        help="RAG question (rag agent only)",
    )
    args = parser.parse_args()

    if args.agent == "rag":
        payload = {
            "question": args.question,
            "patient_id": args.patient,
            "session_id": "terminal-demo",
        }
    else:
        payload = await _load_summary_payload(args.patient)

    await stream_agent(args.agent, payload)


if __name__ == "__main__":
    asyncio.run(main())

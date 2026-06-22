#!/usr/bin/env python3
"""Smoke test for Step 10: Mini Grid-Mind agent loop.

Examples:
    python3 Code/scripts/run_smoke_step10.py --dry-run
    python3 Code/scripts/run_smoke_step10.py --host 127.0.0.1 --port 8000 --message "Run power flow on ieee14."
    python3 Code/scripts/run_smoke_step10.py --interactive --message "Run power flow on ieee14."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "Code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "Code"))

from gridmind_mini import (  # noqa: E402
    DEFAULT_LOCAL_MODEL,
    AgentConfig,
    AgentLoopError,
    GridMindAgent,
    LLMClientError,
    StudyMemoryStore,
    ToolRegistry,
    VLLMConfig,
    VLLMOpenAIClient,
    build_chat_messages,
    detect_cia_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step 10 Mini Grid-Mind agent smoke test.")
    parser.add_argument("--base-url", help="Full local vLLM base URL, e.g. http://127.0.0.1:8000/v1.")
    parser.add_argument("--host", help="Local vLLM host/IP. Used when --base-url is omitted.")
    parser.add_argument("--port", type=int, help="Local vLLM port. Used when --base-url is omitted.")
    parser.add_argument("--scheme", default="http", choices=["http", "https"])
    parser.add_argument("--api-path", default="/v1")
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL, help="Served model name or 'auto'.")
    parser.add_argument("--api-key", help="Optional bearer token.")
    parser.add_argument(
        "--message",
        default="Run a power flow on IEEE 14 and report any violations.",
        help="User message for the agent.",
    )
    parser.add_argument("--case", dest="case_path", help="Optional case context.")
    parser.add_argument("--bus", type=int, help="Optional bus context.")
    parser.add_argument("--mw", type=float, help="Optional MW context.")
    parser.add_argument(
        "--type",
        dest="connection_type",
        choices=["load", "solar", "wind", "bess", "hybrid", "synchronous"],
        help="Optional connection-type context.",
    )
    parser.add_argument("--max-mw", type=float, help="Optional capacity-search upper bound context.")
    parser.add_argument("--tolerance-mw", type=float, help="Optional capacity-search tolerance context.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-tool-rounds", type=int, default=5)
    parser.add_argument("--memory-dir", help="Optional persistent memory directory.")
    parser.add_argument("--no-forced-routing", action="store_true", help="Disable forced capacity routing.")
    parser.add_argument("--no-cia-readiness-gate", action="store_true", help="Disable deterministic CIA input precheck.")
    parser.add_argument("--no-tool-policy-guard", action="store_true", help="Disable model tool-call policy checks.")
    parser.add_argument("--no-tool-observation-summary", action="store_true", help="Send raw tool results without compact observations.")
    parser.add_argument("--no-raw-tool-result", action="store_true", help="Send compact tool observations without raw tool-result payloads.")
    parser.add_argument("--no-deterministic-report", action="store_true", help="Disable deterministic final report generation.")
    parser.add_argument("--no-empty-report-fallback", action="store_true", help="Do not replace empty final LLM text with the deterministic report.")
    parser.add_argument("--no-max-round-report-fallback", action="store_true", help="Do not append deterministic report text when max tool rounds are exceeded.")
    parser.add_argument("--include-messages", action="store_true", help="Include full conversation messages.")
    parser.add_argument("--dry-run", action="store_true", help="Build prompt/config only; do not call vLLM.")
    parser.add_argument("--interactive", action="store_true", help="Prompt for host and port before calling vLLM.")
    args = parser.parse_args()

    try:
        base_url = _resolve_base_url(args, calls_endpoint=not args.dry_run)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1

    context = _context_from_args(args)
    memory_store = StudyMemoryStore(args.memory_dir) if args.memory_dir else None
    registry = ToolRegistry(memory_store=memory_store)
    try:
        config = _agent_config_from_args(args)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1

    if args.dry_run:
        prompt = build_chat_messages(
            registry,
            args.message,
            memory_store=memory_store,
            context=context,
        )
        output = {
            "ok": True,
            "dry_run": True,
            "base_url": base_url,
            "model": args.model,
            "agent_config": {
                "max_tool_rounds": config.max_tool_rounds,
                "enable_forced_capacity_routing": config.enable_forced_capacity_routing,
                "enable_cia_readiness_gate": config.enable_cia_readiness_gate,
                "enable_tool_call_policy_guard": config.enable_tool_call_policy_guard,
                "enable_tool_observation_summary": config.enable_tool_observation_summary,
                "include_raw_tool_result_in_message": config.include_raw_tool_result_in_message,
                "enable_deterministic_report": config.enable_deterministic_report,
                "use_deterministic_report_when_final_empty": config.use_deterministic_report_when_final_empty,
                "use_deterministic_report_on_max_rounds": config.use_deterministic_report_on_max_rounds,
            },
            "context": context,
            "context_hints": prompt.context_hints.to_dict(),
            "cia_readiness": detect_cia_readiness(args.message, context=context).to_dict(),
            "tool_names": [spec["function"]["name"] for spec in registry.openai_tool_specs()],
            "messages": prompt.messages,
        }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0

    client = VLLMOpenAIClient(
        VLLMConfig(
            base_url=base_url,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )
    agent = GridMindAgent(
        registry=registry,
        llm_client=client,
        memory_store=memory_store,
        config=config,
    )
    try:
        result = agent.run_turn(args.message, context=context)
    except (AgentLoopError, LLMClientError, ValueError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1

    output = {"ok": True, "agent_result": result.to_dict(include_messages=args.include_messages)}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _context_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for key in ("case_path", "bus", "mw", "connection_type", "max_mw", "tolerance_mw"):
        value = getattr(args, key)
        if value is not None:
            context[key] = value
    return context


def _agent_config_from_args(args: argparse.Namespace) -> AgentConfig:
    if not isinstance(args.max_tool_rounds, int) or args.max_tool_rounds <= 0:
        raise ValueError("max_tool_rounds must be a positive integer")
    return AgentConfig(
        max_tool_rounds=args.max_tool_rounds,
        enable_forced_capacity_routing=not args.no_forced_routing,
        enable_cia_readiness_gate=not args.no_cia_readiness_gate,
        enable_tool_call_policy_guard=not args.no_tool_policy_guard,
        enable_tool_observation_summary=not args.no_tool_observation_summary,
        include_raw_tool_result_in_message=not args.no_raw_tool_result,
        enable_deterministic_report=not args.no_deterministic_report,
        use_deterministic_report_when_final_empty=not args.no_empty_report_fallback,
        use_deterministic_report_on_max_rounds=not args.no_max_round_report_fallback,
    )


def _resolve_base_url(args: argparse.Namespace, *, calls_endpoint: bool) -> str:
    if args.base_url:
        return args.base_url.rstrip("/")

    host = args.host
    port = args.port
    if calls_endpoint and args.interactive:
        host = _prompt_text("Local vLLM host/IP", host or "127.0.0.1")
        port = _prompt_int("Local vLLM port", port or 8000)

    host = host or "127.0.0.1"
    port = port or 8000
    _validate_port(port)
    api_path = "/" + str(args.api_path).strip("/")
    return f"{args.scheme}://{host}:{port}{api_path}"


def _prompt_text(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _prompt_int(label: str, default: int) -> int:
    value = input(f"{label} [{default}]: ").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _validate_port(port: int) -> None:
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("Local vLLM port must be an integer between 1 and 65535")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the ten-scenario M1 Mini Grid-Mind live-agent benchmark.

Examples:
    python3 Code/scripts/run_m1_benchmark.py --list-scenarios
    python3 Code/scripts/run_m1_benchmark.py --oracle-only --no-raw-results
    python3 Code/scripts/run_m1_benchmark.py --host 127.0.0.1 --port 8000 --no-raw-results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


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
)
from gridmind_mini.m1_benchmark import (  # noqa: E402
    M1BenchmarkRunner,
    default_m1_benchmark_scenarios,
    filter_m1_scenarios,
    run_m1_oracles,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the M1 Mini Grid-Mind live-agent benchmark."
    )
    parser.add_argument("--base-url", help="Full local vLLM base URL, e.g. http://127.0.0.1:8000/v1.")
    parser.add_argument("--host", help="Local vLLM host/IP. Used when --base-url is omitted.")
    parser.add_argument("--port", type=int, help="Local vLLM port. Used when --base-url is omitted.")
    parser.add_argument("--scheme", default="http", choices=["http", "https"])
    parser.add_argument("--api-path", default="/v1")
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL, help="Served model name or 'auto'.")
    parser.add_argument("--api-key", help="Optional bearer token.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-tool-rounds", type=int, default=5)
    parser.add_argument("--memory-dir", help="Optional persistent memory directory for the live agent.")
    parser.add_argument("--scenario", action="append", default=[], help="Scenario id to run. May be repeated.")
    parser.add_argument("--tag", action="append", default=[], help="Scenario tag filter. May be repeated.")
    parser.add_argument("--list-scenarios", action="store_true", help="List selected scenarios without running them.")
    parser.add_argument("--oracle-only", action="store_true", help="Run deterministic oracle tools only; do not call vLLM.")
    parser.add_argument("--no-raw-results", action="store_true", help="Omit full raw agent/oracle outputs.")
    parser.add_argument("--include-messages", action="store_true", help="Include full model conversation messages.")
    parser.add_argument("--no-forced-routing", action="store_true", help="Disable forced capacity routing.")
    parser.add_argument("--no-cia-readiness-gate", action="store_true", help="Disable deterministic CIA input precheck.")
    parser.add_argument("--no-tool-policy-guard", action="store_true", help="Disable model tool-call policy checks.")
    parser.add_argument("--no-tool-observation-summary", action="store_true", help="Send raw tool results without compact observations.")
    parser.add_argument("--no-raw-tool-result", action="store_true", help="Send compact tool observations without raw tool-result payloads.")
    args = parser.parse_args()

    try:
        scenarios = filter_m1_scenarios(
            default_m1_benchmark_scenarios(),
            scenario_ids=args.scenario,
            tags=args.tag,
        )
    except ValueError as exc:
        print(_json({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}))
        return 1

    if args.list_scenarios:
        print(
            _json(
                {
                    "ok": True,
                    "scenario_count": len(scenarios),
                    "scenarios": [scenario.to_dict() for scenario in scenarios],
                }
            )
        )
        return 0

    if not scenarios:
        print(_json({"ok": False, "error_type": "no_scenarios", "error": "No scenarios selected."}))
        return 1

    oracle_registry = ToolRegistry()
    if args.oracle_only:
        outputs = run_m1_oracles(scenarios, oracle_registry)
        ok = all(item.get("ok", False) for item in outputs)
        if args.no_raw_results:
            outputs = [_oracle_summary(item) for item in outputs]
        print(
            _json(
                {
                    "ok": ok,
                    "mode": "oracle_only",
                    "scenario_count": len(scenarios),
                    "results": outputs,
                }
            )
        )
        return 0 if ok else 1

    try:
        base_url = _resolve_base_url(args)
        config = _agent_config_from_args(args)
    except ValueError as exc:
        print(_json({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}))
        return 1

    memory_store = StudyMemoryStore(args.memory_dir) if args.memory_dir else None
    registry = ToolRegistry(memory_store=memory_store)
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
    runner = M1BenchmarkRunner(agent, oracle_registry)

    try:
        suite = runner.run_suite(scenarios)
    except (AgentLoopError, LLMClientError, ValueError) as exc:
        print(_json({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}))
        return 1

    output = {
        "ok": suite.ok,
        "mode": "live_agent",
        "base_url": base_url,
        "model": args.model,
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "suite": suite.to_dict(
            include_raw_results=not args.no_raw_results,
            include_messages=args.include_messages,
        ),
    }
    print(_json(output))
    return 0 if suite.ok else 1


def _resolve_base_url(args: argparse.Namespace) -> str:
    if args.base_url:
        return args.base_url.rstrip("/")
    host = args.host or "127.0.0.1"
    port = args.port or 8000
    _validate_port(port)
    api_path = "/" + str(args.api_path).strip("/")
    return f"{args.scheme}://{host}:{port}{api_path}"


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
    )


def _validate_port(port: int) -> None:
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("Local vLLM port must be an integer between 1 and 65535")


def _oracle_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    scenario = item.get("scenario", {})
    result = item.get("oracle_result")
    summary: Dict[str, Any] = {
        "scenario_id": scenario.get("scenario_id") if isinstance(scenario, dict) else None,
        "ok": bool(item.get("ok", False)),
    }
    if isinstance(result, dict):
        summary["tool"] = result.get("tool")
        summary["error_type"] = result.get("error_type")
        for key in ("recommendation", "status", "max_approved_mw", "case_path"):
            if key in result:
                summary[key] = result[key]
    else:
        summary["tool"] = None
        summary["note"] = item.get("note")
    path_checks = item.get("oracle_path_checks", [])
    if isinstance(path_checks, list):
        summary["oracle_path_checks_ok"] = all(
            bool(check.get("passed", False))
            for check in path_checks
            if isinstance(check, dict)
        )
    argument_checks = item.get("oracle_argument_checks", [])
    if isinstance(argument_checks, list):
        summary["oracle_argument_checks_ok"] = all(
            bool(check.get("passed", False))
            for check in argument_checks
            if isinstance(check, dict)
        )
    return summary


def _json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())

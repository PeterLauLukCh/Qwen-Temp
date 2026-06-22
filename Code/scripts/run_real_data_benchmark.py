#!/usr/bin/env python3
"""Run the frozen real-data PSS/E Mini Grid-Mind benchmark."""

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
from gridmind_mini.real_data_benchmark import (  # noqa: E402
    RealDataBenchmarkRunner,
    default_real_data_benchmark_scenarios,
    filter_real_data_scenarios,
    run_real_data_oracles,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen real-data PSS/E Mini Grid-Mind benchmark."
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
    parser.add_argument("--processed-dir", help="Directory containing processed PSSE JSON/CSV outputs.")
    parser.add_argument("--scenario", action="append", default=[], help="Scenario id to run. May be repeated.")
    parser.add_argument("--tag", action="append", default=[], help="Scenario tag filter. May be repeated.")
    parser.add_argument("--list-scenarios", action="store_true", help="List selected scenarios without running them.")
    parser.add_argument("--oracle-only", action="store_true", help="Run frozen oracle checks only; do not call vLLM.")
    parser.add_argument("--no-raw-results", action="store_true", help="Omit full raw agent/oracle outputs.")
    parser.add_argument("--include-messages", action="store_true", help="Include full model conversation messages.")
    parser.add_argument("--no-forced-routing", action="store_true", help="Disable forced capacity routing.")
    parser.add_argument("--no-cia-readiness-gate", action="store_true", help="Disable deterministic CIA input precheck.")
    parser.add_argument("--no-tool-policy-guard", action="store_true", help="Disable model tool-call policy checks.")
    parser.add_argument("--no-tool-observation-summary", action="store_true", help="Send raw tool results without compact observations.")
    parser.add_argument("--no-raw-tool-result", action="store_true", help="Send compact tool observations without raw tool-result payloads.")
    args = parser.parse_args()

    scenarios = filter_real_data_scenarios(
        default_real_data_benchmark_scenarios(processed_dir=args.processed_dir),
        scenario_ids=args.scenario,
        tags=args.tag,
    )

    if args.list_scenarios:
        print(_json(_scenario_listing_payload(scenarios)))
        return 0
    if not scenarios:
        print(_json({"ok": False, "error_type": "no_scenarios", "error": "No scenarios selected."}))
        return 1

    oracle_registry = ToolRegistry()
    if args.oracle_only:
        outputs = run_real_data_oracles(scenarios, oracle_registry)
        ok = all(item.get("ok", False) for item in outputs)
        if args.no_raw_results:
            outputs = [_oracle_summary(item) for item in outputs]
        print(
            _json(
                {
                    "ok": ok,
                    "mode": "oracle_only",
                    "processed_dir": args.processed_dir,
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
    runner = RealDataBenchmarkRunner(agent, oracle_registry)
    try:
        suite = runner.run_suite(scenarios)
    except (AgentLoopError, LLMClientError, ValueError) as exc:
        print(_json({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}))
        return 1

    output = {
        "ok": suite.ok,
        "mode": "live_agent",
        "processed_dir": args.processed_dir,
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


def _scenario_listing_payload(scenarios: List[Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "scenario_count": len(scenarios),
        "scenarios": [scenario.to_dict() for scenario in scenarios],
    }


def _resolve_base_url(args: argparse.Namespace) -> str:
    if args.base_url:
        return args.base_url.rstrip("/")
    host = args.host or "127.0.0.1"
    port = args.port or 8000
    if not isinstance(port, int) or port <= 0 or port > 65535:
        raise ValueError("--port must be in 1..65535")
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


def _oracle_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": item.get("ok"),
        "scenario_id": item.get("scenario_id"),
        "tool": item.get("tool"),
        "case_id": item.get("case_id"),
        "recommendation": item.get("recommendation"),
        "complete": item.get("complete"),
        "summary": item.get("summary"),
        "failed_checks": [
            check
            for check in item.get("check_results", [])
            if not check.get("passed", False)
        ],
    }


def _json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())


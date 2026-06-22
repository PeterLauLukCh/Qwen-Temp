#!/usr/bin/env python3
"""Run integrated M1+M2 Mini Grid-Mind benchmarks.

Examples:
    python3 Code/scripts/run_integrated_benchmark.py --list-scenarios
    python3 Code/scripts/run_integrated_benchmark.py --oracle-only --no-raw-results
    python3 Code/scripts/run_integrated_benchmark.py --oracle-only --live-m2-oracle --no-raw-results
    python3 Code/scripts/run_integrated_benchmark.py --host 127.0.0.1 --port 8000 --no-raw-results
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
)
from gridmind_mini.integrated_benchmark import (  # noqa: E402
    run_integrated_live_agent,
    run_integrated_oracles,
)
from gridmind_mini.m1_benchmark import (  # noqa: E402
    default_m1_benchmark_scenarios,
    filter_m1_scenarios,
)
from gridmind_mini.m2_benchmark import (  # noqa: E402
    default_m2_benchmark_scenarios,
    filter_m2_scenarios,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run integrated M1+M2 Mini Grid-Mind benchmarks."
    )
    parser.add_argument("--suite", default="all", choices=["all", "m1", "m2"], help="Which suite(s) to run.")
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
    parser.add_argument("--m1-scenario", action="append", default=[], help="M1 scenario id to run. May be repeated.")
    parser.add_argument("--m2-scenario", action="append", default=[], help="M2 scenario id to run. May be repeated.")
    parser.add_argument("--m1-tag", action="append", default=[], help="M1 scenario tag filter. May be repeated.")
    parser.add_argument("--m2-tag", action="append", default=[], help="M2 scenario tag filter. May be repeated.")
    parser.add_argument("--list-scenarios", action="store_true", help="List selected scenarios without running them.")
    parser.add_argument("--oracle-only", action="store_true", help="Run oracle checks only; do not call vLLM.")
    parser.add_argument("--live-m2-oracle", action="store_true", help="Execute real ANDES M2 oracle tools. Requires ANDES and can be slower.")
    parser.add_argument("--no-raw-results", action="store_true", help="Omit full raw agent/oracle outputs.")
    parser.add_argument("--include-messages", action="store_true", help="Include full model conversation messages.")
    parser.add_argument("--no-forced-routing", action="store_true", help="Disable forced capacity routing.")
    parser.add_argument("--no-cia-readiness-gate", action="store_true", help="Disable deterministic CIA input precheck.")
    parser.add_argument("--no-tool-policy-guard", action="store_true", help="Disable model tool-call policy checks.")
    parser.add_argument("--no-tool-observation-summary", action="store_true", help="Send raw tool results without compact observations.")
    parser.add_argument("--no-raw-tool-result", action="store_true", help="Send compact tool observations without raw tool-result payloads.")
    args = parser.parse_args()

    try:
        m1_scenarios = (
            filter_m1_scenarios(
                default_m1_benchmark_scenarios(),
                scenario_ids=args.m1_scenario,
                tags=args.m1_tag,
            )
            if args.suite in {"all", "m1"}
            else []
        )
        m2_scenarios = (
            filter_m2_scenarios(
                default_m2_benchmark_scenarios(),
                scenario_ids=args.m2_scenario,
                tags=args.m2_tag,
            )
            if args.suite in {"all", "m2"}
            else []
        )
    except ValueError as exc:
        print(_json({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}))
        return 1

    if args.list_scenarios:
        print(
            _json(
                {
                    "ok": True,
                    "suite": args.suite,
                    "m1": {
                        "scenario_count": len(m1_scenarios),
                        "scenarios": [scenario.to_dict() for scenario in m1_scenarios],
                    },
                    "m2": {
                        "scenario_count": len(m2_scenarios),
                        "scenarios": [scenario.to_dict() for scenario in m2_scenarios],
                    },
                }
            )
        )
        return 0

    if not m1_scenarios and not m2_scenarios:
        print(_json({"ok": False, "error_type": "no_scenarios", "error": "No scenarios selected."}))
        return 1

    oracle_registry = ToolRegistry()
    if args.oracle_only:
        try:
            result = run_integrated_oracles(
                m1_scenarios=m1_scenarios,
                m2_scenarios=m2_scenarios,
                oracle_registry=oracle_registry,
                execute_m2_oracle=args.live_m2_oracle,
            )
        except ValueError as exc:
            print(_json({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}))
            return 1
        print(
            _json(
                result.to_dict(
                    include_raw_results=not args.no_raw_results,
                    include_messages=args.include_messages,
                )
            )
        )
        return 0 if result.ok else 1

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

    try:
        result = run_integrated_live_agent(
            agent=agent,
            m1_scenarios=m1_scenarios,
            m2_scenarios=m2_scenarios,
            oracle_registry=oracle_registry,
            execute_m2_oracle=args.live_m2_oracle,
        )
    except (AgentLoopError, LLMClientError, ValueError) as exc:
        print(_json({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}))
        return 1

    output = result.to_dict(
        include_raw_results=not args.no_raw_results,
        include_messages=args.include_messages,
    )
    output["base_url"] = base_url
    output["model"] = args.model
    print(_json(output))
    return 0 if result.ok else 1


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


def _json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())

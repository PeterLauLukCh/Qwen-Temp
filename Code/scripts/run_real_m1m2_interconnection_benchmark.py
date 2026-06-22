#!/usr/bin/env python3
"""Run generated live remote PSS/E M1+M2 interconnection benchmark cases."""

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
from gridmind_mini.real_m1m2_interconnection_benchmark import (  # noqa: E402
    RealM1M2InterconnectionBenchmarkRunner,
    filter_real_m1m2_interconnection_testcases,
    load_real_m1m2_interconnection_testcases,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run generated live remote PSS/E M1+M2 interconnection benchmark cases."
    )
    parser.add_argument(
        "--cases",
        default="real-data-new/generated_real_m1m2_interconnection_cases.json",
        help="Generated testcase JSON/JSONL file.",
    )
    parser.add_argument("--base-url", help="Full local vLLM base URL, e.g. http://127.0.0.1:8000/v1.")
    parser.add_argument("--host", help="Local vLLM host/IP. Used when --base-url is omitted.")
    parser.add_argument("--port", type=int, help="Local vLLM port. Used when --base-url is omitted.")
    parser.add_argument("--scheme", default="http", choices=["http", "https"])
    parser.add_argument("--api-path", default="/v1")
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL, help="Served model name or 'auto'.")
    parser.add_argument("--api-key", help="Optional bearer token for the vLLM endpoint.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-tool-rounds", type=int, default=5)
    parser.add_argument("--memory-dir", help="Optional persistent memory directory for the live agent.")
    parser.add_argument("--scenario", action="append", default=[], help="Scenario id to run. May be repeated.")
    parser.add_argument("--tag", action="append", default=[], help="Scenario tag filter. May be repeated.")
    parser.add_argument("--difficulty", action="append", default=[], help="Difficulty filter. May be repeated.")
    parser.add_argument("--label", action="append", default=[], help="Oracle-label filter. May be repeated.")
    parser.add_argument("--limit", type=int, help="Run only the first N selected cases.")
    parser.add_argument("--list-scenarios", action="store_true", help="List selected scenarios without running them.")
    parser.add_argument("--output", help="Optional file to write the full benchmark JSON result.")
    parser.add_argument("--no-raw-results", action="store_true", help="Omit full raw agent outputs.")
    parser.add_argument("--include-messages", action="store_true", help="Include full model conversation messages.")
    parser.add_argument("--no-forced-routing", action="store_true", help="Disable forced capacity routing.")
    parser.add_argument("--no-cia-readiness-gate", action="store_true", help="Disable deterministic CIA input precheck.")
    parser.add_argument("--no-tool-policy-guard", action="store_true", help="Disable model tool-call policy checks.")
    parser.add_argument("--no-tool-observation-summary", action="store_true", help="Send raw tool results without compact observations.")
    parser.add_argument("--no-raw-tool-result", action="store_true", help="Send compact tool observations without raw tool-result payloads.")
    parser.add_argument("--no-deterministic-report", action="store_true", help="Disable deterministic final report generation.")
    parser.add_argument("--no-empty-report-fallback", action="store_true", help="Do not replace empty final LLM text with deterministic report.")
    parser.add_argument("--no-max-round-report-fallback", action="store_true", help="Do not append deterministic report text on max rounds.")
    args = parser.parse_args()

    try:
        scenarios = filter_real_m1m2_interconnection_testcases(
            load_real_m1m2_interconnection_testcases(args.cases),
            scenario_ids=args.scenario,
            tags=args.tag,
            difficulties=args.difficulty,
            labels=args.label,
            limit=args.limit,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(_json({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}))
        return 1

    if args.list_scenarios:
        print(_json(_scenario_listing_payload(scenarios, cases_path=args.cases)))
        return 0
    if not scenarios:
        print(_json({"ok": False, "error_type": "no_scenarios", "error": "No scenarios selected."}))
        return 1

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
    runner = RealM1M2InterconnectionBenchmarkRunner(agent)
    try:
        suite = runner.run_suite(scenarios)
    except (AgentLoopError, LLMClientError, ValueError) as exc:
        print(_json({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}))
        return 1

    output = {
        "ok": suite.ok,
        "mode": "live_agent",
        "cases": args.cases,
        "base_url": base_url,
        "model": args.model,
        "scenario_count": len(scenarios),
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "suite": suite.to_dict(
            include_raw_results=not args.no_raw_results,
            include_messages=args.include_messages,
        ),
    }
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_json(output) + "\n", encoding="utf-8")
        summary = {
            "ok": output["ok"],
            "output": str(output_path),
            "scenario_count": output["scenario_count"],
            "total": output["suite"]["total"],
            "passed": output["suite"]["passed"],
            "failed": output["suite"]["failed"],
            "duration_s": output["suite"]["duration_s"],
            "by_label": output["suite"]["by_label"],
            "by_difficulty": output["suite"]["by_difficulty"],
        }
        print(_json(summary))
    else:
        print(_json(output))
    return 0 if suite.ok else 1


def _scenario_listing_payload(
    scenarios: List[Any],
    *,
    cases_path: str,
) -> Dict[str, Any]:
    return {
        "ok": True,
        "cases": cases_path,
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
        enable_deterministic_report=not args.no_deterministic_report,
        use_deterministic_report_when_final_empty=not args.no_empty_report_fallback,
        use_deterministic_report_on_max_rounds=not args.no_max_round_report_fallback,
    )


def _json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())

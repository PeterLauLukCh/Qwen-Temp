#!/usr/bin/env python3
"""Run Step 15 deterministic Mini Grid-Mind experiment scenarios.

Examples:
    python3 Code/scripts/run_experiments_step15.py --list-scenarios
    python3 Code/scripts/run_experiments_step15.py --case ieee14 --no-raw-results
    python3 Code/scripts/run_experiments_step15.py --case ieee118 --include-slow --tag fast
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
    ExperimentRunner,
    StudyMemoryStore,
    ToolRegistry,
    default_experiment_scenarios,
    filter_scenarios,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic Mini Grid-Mind experiment scenarios."
    )
    parser.add_argument("--case", dest="case_path", default="ieee14", help="Case name, e.g. ieee14 or ieee118.")
    parser.add_argument("--include-slow", action="store_true", help="Include bounded contingency and CIA scenarios.")
    parser.add_argument("--scenario", action="append", default=[], help="Scenario id to run. May be repeated.")
    parser.add_argument("--tag", action="append", default=[], help="Scenario tag filter. May be repeated.")
    parser.add_argument("--memory-dir", help="Optional study-memory directory for tools that persist records.")
    parser.add_argument("--no-raw-results", action="store_true", help="Omit full raw tool outputs from JSON.")
    parser.add_argument("--list-scenarios", action="store_true", help="List selected scenarios without running them.")
    args = parser.parse_args()

    try:
        scenarios = default_experiment_scenarios(
            case_path=args.case_path,
            include_slow=args.include_slow,
        )
        scenarios = filter_scenarios(
            scenarios,
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
                    "case_path": args.case_path,
                    "include_slow": args.include_slow,
                    "scenario_count": len(scenarios),
                    "scenarios": [scenario.to_dict() for scenario in scenarios],
                }
            )
        )
        return 0

    if not scenarios:
        print(_json({"ok": False, "error_type": "no_scenarios", "error": "No scenarios selected."}))
        return 1

    memory_store = StudyMemoryStore(args.memory_dir) if args.memory_dir else None
    runner = ExperimentRunner(ToolRegistry(memory_store=memory_store))
    suite = runner.run_suite(scenarios)
    output: Dict[str, Any] = {
        "step": "deterministic_experiment_harness",
        "case_path": args.case_path,
        "include_slow": args.include_slow,
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "suite": suite.to_dict(include_raw_results=not args.no_raw_results),
    }
    print(_json(output))
    return 0 if suite.ok else 1


def _json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())

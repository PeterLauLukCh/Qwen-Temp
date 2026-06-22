#!/usr/bin/env python3
"""Smoke test for Step 1: load a case and run AC power flow.

Examples:
    python3 Code/scripts/run_smoke_step1.py --case ieee14
    python3 Code/scripts/run_smoke_step1.py --case ieee118 --show-top 5
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

from gridmind_mini import PandaPowerSolver, PowerFlowError, SolverDependencyError


def top_by_key(rows: List[Dict[str, Any]], key: str, n: int) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: row.get(key) if row.get(key) is not None else float("-inf"),
        reverse=True,
    )[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step 1 Mini Grid-Mind smoke test.")
    parser.add_argument("--case", default="ieee14", help="Case name: ieee14, ieee30, ieee57, ieee118")
    parser.add_argument("--show-top", type=int, default=0, help="Show top-N loaded lines and lowest-voltage buses")
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List supported cases without requiring the pandapower backend.",
    )
    args = parser.parse_args()

    if args.list_cases:
        print(
            json.dumps(
                {
                    "ok": True,
                    "backend": "pandapower",
                    "backend_available": PandaPowerSolver.is_available(),
                    "supported_cases": PandaPowerSolver.supported_cases(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not PandaPowerSolver.is_available():
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": "missing_dependency",
                    "error": (
                        "pandapower is not installed in this environment. "
                        "The Step 1 skeleton is present; run this smoke test on "
                        "the solver/runtime node after installing dependencies there."
                    ),
                    "supported_cases": PandaPowerSolver.supported_cases(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    try:
        solver = PandaPowerSolver()
        case_info = solver.load_case(args.case)
        summary = solver.run_powerflow()
    except SolverDependencyError as exc:
        print(json.dumps({"ok": False, "error_type": "missing_dependency", "error": str(exc)}, indent=2))
        return 2
    except (PowerFlowError, ValueError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1

    output: Dict[str, Any] = {
        "ok": True,
        "step": "minimal_solver_adapter",
        "case_info": case_info.to_dict(),
        "powerflow_summary": summary,
    }

    if args.show_top > 0:
        bus_rows = solver.bus_results()
        branches = solver.branch_results()
        output["lowest_voltage_buses"] = sorted(
            bus_rows,
            key=lambda row: row.get("vm_pu") if row.get("vm_pu") is not None else float("inf"),
        )[: args.show_top]
        output["highest_loaded_lines"] = top_by_key(
            branches["lines"], "loading_percent", args.show_top
        )
        output["highest_loaded_transformers"] = top_by_key(
            branches["transformers"], "loading_percent", args.show_top
        )

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

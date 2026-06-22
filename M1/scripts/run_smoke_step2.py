#!/usr/bin/env python3
"""Smoke test for Step 2: run power flow and inspect violations.

Examples:
    python3 Code/scripts/run_smoke_step2.py --case ieee14
    python3 Code/scripts/run_smoke_step2.py --case ieee118 --profile emergency
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "Code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "Code"))

from gridmind_mini import (  # noqa: E402
    PandaPowerSolver,
    PowerFlowError,
    SolverDependencyError,
    ViolationInspector,
    profile_from_name,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step 2 Mini Grid-Mind smoke test.")
    parser.add_argument("--case", default="ieee14", help="Case name: ieee14, ieee30, ieee57, ieee118")
    parser.add_argument(
        "--profile",
        default="normal",
        choices=["normal", "emergency"],
        help="Limit profile to apply.",
    )
    parser.add_argument(
        "--max-violations",
        type=int,
        default=25,
        help="Maximum violations to print; use -1 for the full report.",
    )
    parser.add_argument(
        "--angle-limit-degree",
        type=float,
        default=None,
        help="Enable optional branch angle-difference screening with this degree limit.",
    )
    args = parser.parse_args()

    if not PandaPowerSolver.is_available():
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": "missing_dependency",
                    "error": (
                        "pandapower is not installed in this environment. "
                        "Run this smoke test on the solver/runtime node."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    try:
        solver = PandaPowerSolver()
        case_info = solver.load_case(args.case)
        powerflow_summary = solver.run_powerflow()
        profile = profile_from_name(args.profile)
        if args.angle_limit_degree is not None:
            profile = replace(profile, angle_diff_limit_degree=args.angle_limit_degree)
        inspector = ViolationInspector(profile)
        report = inspector.inspect_solver(solver)
    except SolverDependencyError as exc:
        print(json.dumps({"ok": False, "error_type": "missing_dependency", "error": str(exc)}, indent=2))
        return 2
    except (PowerFlowError, ValueError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1

    report_dict = report.to_dict()
    total_violations = len(report_dict["violations"])
    if args.max_violations >= 0:
        report_dict["violations"] = report_dict["violations"][: args.max_violations]
    report_dict["truncated_violations"] = total_violations - len(report_dict["violations"])

    output: Dict[str, Any] = {
        "ok": True,
        "step": "violation_inspector",
        "case_info": case_info.to_dict(),
        "powerflow_summary": powerflow_summary,
        "violation_report": report_dict,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Smoke test for Step 6: find maximum approved connection capacity.

Examples:
    python3 Code/scripts/run_smoke_step6.py --case ieee14 --bus 10 --max-mw 100
    python3 Code/scripts/run_smoke_step6.py --case ieee118 --bus 10 --type solar --ibr --max-mw 200
    python3 Code/scripts/run_smoke_step6.py --case ieee14 --bus 10 --max-mw 100 --contingency --max-contingencies 5
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

from gridmind_mini import ToolRegistry, ToolRegistryError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step 6 Mini Grid-Mind capacity-search smoke test.")
    parser.add_argument("--case", default="ieee14", help="Case name/path.")
    parser.add_argument("--bus", type=int, required=True, help="External bus number to connect at.")
    parser.add_argument(
        "--type",
        dest="connection_type",
        default="load",
        choices=["load", "solar", "wind", "bess", "hybrid", "synchronous"],
        help="Connection type.",
    )
    ibr_group = parser.add_mutually_exclusive_group()
    ibr_group.add_argument("--ibr", action="store_true", help="Force the connection to be treated as IBR.")
    ibr_group.add_argument("--non-ibr", action="store_true", help="Force the connection to be treated as non-IBR.")
    parser.add_argument("--q-mvar", type=float, default=0.0, help="Reactive power for load/sgen models.")
    parser.add_argument("--vm-pu", type=float, default=1.0, help="Voltage setpoint for synchronous generator.")
    parser.add_argument("--min-mw", type=float, default=0.0, help="Lower MW search bound.")
    parser.add_argument("--max-mw", type=float, default=500.0, help="Upper MW search bound.")
    parser.add_argument("--tolerance-mw", type=float, default=1.0, help="Bisection stopping tolerance in MW.")
    parser.add_argument("--max-iterations", type=int, default=12, help="Maximum bisection midpoint samples.")
    parser.add_argument("--coarse-scan-points", type=int, default=11, help="Fallback scan point count.")
    parser.add_argument("--contingency", action="store_true", help="Enable f2 N-1 screening for every sampled CIA.")
    parser.add_argument(
        "--max-contingencies",
        type=int,
        default=-1,
        help="Maximum N-1 outage cases per sampled CIA when --contingency is set.",
    )
    parser.add_argument(
        "--fail-on-contingency-material-worsening",
        action="store_true",
        help="Reject f2 when pre-existing N-1 failures materially worsen.",
    )
    parser.add_argument("--max-failed-contingencies", type=int, default=10, help="Maximum failed f2 rows per CIA.")
    parser.add_argument("--max-violations", type=int, default=10, help="Maximum violation rows per sampled CIA.")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum sampled MW rows to print.")
    parser.add_argument("--include-reports", action="store_true", help="Include full CIA reports for each sample.")
    args = parser.parse_args()

    tool_args: Dict[str, Any] = {
        "case_path": args.case,
        "bus": args.bus,
        "connection_type": args.connection_type,
        "q_mvar": args.q_mvar,
        "vm_pu": args.vm_pu,
        "min_mw": args.min_mw,
        "max_mw": args.max_mw,
        "tolerance_mw": args.tolerance_mw,
        "max_iterations": args.max_iterations,
        "coarse_scan_points": args.coarse_scan_points,
        "enable_contingency": args.contingency,
        "max_contingencies": args.max_contingencies,
        "fail_on_contingency_material_worsening": args.fail_on_contingency_material_worsening,
        "max_failed_contingencies": args.max_failed_contingencies,
        "max_violations": args.max_violations,
        "max_samples": args.max_samples,
        "include_reports": args.include_reports,
    }
    if args.ibr:
        tool_args["is_ibr"] = True
    elif args.non_ibr:
        tool_args["is_ibr"] = False

    try:
        output = ToolRegistry().call_tool("find_max_capacity", tool_args)
    except (ToolRegistryError, ValueError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

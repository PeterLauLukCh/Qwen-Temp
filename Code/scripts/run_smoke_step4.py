#!/usr/bin/env python3
"""Smoke test for Step 4: run baseline-aware steady-state CIA.

Examples:
    python3 Code/scripts/run_smoke_step4.py --case ieee118 --bus 10 --mw 5 --type load
    python3 Code/scripts/run_smoke_step4.py --case ieee118 --bus 10 --mw 50 --type solar --ibr
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
    parser = argparse.ArgumentParser(description="Run Step 4 Mini Grid-Mind CIA smoke test.")
    parser.add_argument("--case", default="ieee118", help="Case name/path.")
    parser.add_argument("--bus", type=int, required=True, help="External bus number to connect at.")
    parser.add_argument("--mw", type=float, required=True, help="Active power in MW.")
    parser.add_argument(
        "--type",
        default="load",
        choices=["load", "solar", "wind", "bess", "hybrid", "synchronous"],
        help="Connection type.",
    )
    parser.add_argument("--ibr", action="store_true", help="Mark the request as inverter-based.")
    parser.add_argument("--q-mvar", type=float, default=0.0, help="Reactive power for load/sgen models.")
    parser.add_argument("--vm-pu", type=float, default=1.0, help="Voltage setpoint for synchronous generator.")
    parser.add_argument("--contingency", action="store_true", help="Request N-1 contingency screening.")
    parser.add_argument("--transient", action="store_true", help="Request transient stage; reported as not implemented.")
    parser.add_argument("--emt", action="store_true", help="Request EMT stage; reported as not implemented.")
    parser.add_argument(
        "--max-contingencies",
        type=int,
        default=-1,
        help="Maximum N-1 outage cases to evaluate when --contingency is set.",
    )
    parser.add_argument(
        "--fail-on-contingency-material-worsening",
        action="store_true",
        help="Reject f2 when pre-existing N-1 failures materially worsen.",
    )
    parser.add_argument("--max-violations", type=int, default=10, help="Maximum violation rows per report.")
    args = parser.parse_args()

    tool_args: Dict[str, Any] = {
        "case_path": args.case,
        "connection": {
            "bus": args.bus,
            "p_mw": args.mw,
            "connection_type": args.type,
            "is_ibr": args.ibr,
            "q_mvar": args.q_mvar,
            "vm_pu": args.vm_pu,
        },
        "enable_contingency": args.contingency,
        "enable_transient": args.transient,
        "enable_emt": args.emt,
        "max_contingencies": args.max_contingencies,
        "fail_on_contingency_material_worsening": args.fail_on_contingency_material_worsening,
        "max_violations": args.max_violations,
    }

    try:
        output = ToolRegistry().call_tool("run_cia", tool_args)
    except (ToolRegistryError, ValueError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

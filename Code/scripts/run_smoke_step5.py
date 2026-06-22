#!/usr/bin/env python3
"""Smoke test for Step 5: run N-1 contingency screening.

Examples:
    python3 Code/scripts/run_smoke_step5.py --case ieee14 --max-contingencies 5
    python3 Code/scripts/run_smoke_step5.py --case ieee118 --max-contingencies 20
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
    parser = argparse.ArgumentParser(description="Run Step 5 Mini Grid-Mind N-1 smoke test.")
    parser.add_argument("--case", default="ieee14", help="Case name/path.")
    parser.add_argument(
        "--profile",
        default="emergency",
        choices=["normal", "emergency"],
        help="Limit profile; emergency matches Grid-Mind f2.",
    )
    parser.add_argument(
        "--max-contingencies",
        type=int,
        default=10,
        help="Maximum N-1 outage cases to evaluate; use -1 for all.",
    )
    parser.add_argument(
        "--max-violations",
        type=int,
        default=5,
        help="Maximum violation rows per contingency report.",
    )
    parser.add_argument(
        "--max-failed-contingencies",
        type=int,
        default=10,
        help="Maximum failed contingency rows to print.",
    )
    args = parser.parse_args()

    tool_args: Dict[str, Any] = {
        "case_path": args.case,
        "profile": args.profile,
        "max_contingencies": args.max_contingencies,
        "max_violations": args.max_violations,
        "max_failed_contingencies": args.max_failed_contingencies,
    }

    try:
        output = ToolRegistry().call_tool("run_contingency", tool_args)
    except ToolRegistryError as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

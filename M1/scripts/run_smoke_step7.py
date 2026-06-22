#!/usr/bin/env python3
"""Smoke test for Step 7: persistent CIA/capacity study memory.

Examples:
    python3 Code/scripts/run_smoke_step7.py --case ieee14 --bus 10 --mode capacity
    python3 Code/scripts/run_smoke_step7.py --case ieee14 --bus 10 --mode both --memory-dir /private/tmp/gridmind_memory
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

from gridmind_mini import StudyMemoryStore, ToolRegistry, ToolRegistryError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step 7 Mini Grid-Mind memory smoke test.")
    parser.add_argument(
        "--memory-dir",
        default="/private/tmp/powergym_gridmind_memory",
        help="Directory for JSONL memory and Markdown ledger.",
    )
    parser.add_argument("--case", default="ieee14", help="Case name/path.")
    parser.add_argument("--bus", type=int, default=10, help="External bus number to connect at.")
    parser.add_argument(
        "--type",
        dest="connection_type",
        default="load",
        choices=["load", "solar", "wind", "bess", "hybrid", "synchronous"],
        help="Connection type.",
    )
    parser.add_argument("--mw", type=float, default=5.0, help="MW for the CIA demo call.")
    parser.add_argument("--max-mw", type=float, default=20.0, help="Upper MW bound for capacity search.")
    parser.add_argument("--tolerance-mw", type=float, default=5.0, help="Capacity-search tolerance.")
    parser.add_argument(
        "--mode",
        choices=["cia", "capacity", "both", "list"],
        default="both",
        help="Which memory workflow to run.",
    )
    parser.add_argument("--recent", type=int, default=5, help="Recent memory rows to print.")
    args = parser.parse_args()

    store = StudyMemoryStore(args.memory_dir)
    registry = ToolRegistry(memory_store=store)
    tool_results: List[Dict[str, Any]] = []

    try:
        if args.mode in {"cia", "both"}:
            tool_results.append(
                registry.call_tool(
                    "run_cia",
                    {
                        "case_path": args.case,
                        "connection": {
                            "bus": args.bus,
                            "p_mw": args.mw,
                            "connection_type": args.connection_type,
                            "is_ibr": args.connection_type in {"solar", "wind", "bess", "hybrid"},
                        },
                    },
                )
            )
        if args.mode in {"capacity", "both"}:
            tool_results.append(
                registry.call_tool(
                    "find_max_capacity",
                    {
                        "case_path": args.case,
                        "bus": args.bus,
                        "connection_type": args.connection_type,
                        "max_mw": args.max_mw,
                        "tolerance_mw": args.tolerance_mw,
                    },
                )
            )
    except (ToolRegistryError, ValueError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1

    recent = store.recent(limit=args.recent)
    output = {
        "ok": True,
        "memory_dir": str(store.root),
        "records_path": str(store.records_path),
        "ledger_path": str(store.ledger_path),
        "tool_results": tool_results,
        "recent_memory": [record.to_dict() for record in recent],
        "prompt_context": store.build_prompt_context(recent),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

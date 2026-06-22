#!/usr/bin/env python3
"""Generate a frozen-PSS/E interconnection scenario catalog from inventories."""

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
    generate_real_interconnection_scenarios,
    write_real_interconnection_catalog,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate scenario_catalog.jsonl for the frozen PSS/E interconnection gym."
    )
    parser.add_argument("--case-id", default="pif6_2026_05_17")
    parser.add_argument(
        "--processed-dir",
        help="Processed PSSE directory containing inventories/<case_id>/*.csv.",
    )
    parser.add_argument(
        "--output",
        default="real-data-new/frozen_interconnection/scenario_catalog.jsonl",
        help="Output JSONL path. index.json and criteria.json are written beside it.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the generated summary without writing files.",
    )
    args = parser.parse_args()

    try:
        build = generate_real_interconnection_scenarios(
            case_id=args.case_id,
            processed_dir=args.processed_dir,
        )
        payload: Dict[str, Any]
        if args.summary_only:
            payload = {
                "ok": True,
                "case_id": build.case_id,
                "scenario_count": len(build.scenarios),
                "action_space": build.action_space,
                "criteria": build.criteria,
            }
        else:
            payload = write_real_interconnection_catalog(
                output=args.output,
                build_result=build,
            )
            payload["action_space"] = build.action_space
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

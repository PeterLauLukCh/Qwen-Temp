#!/usr/bin/env python3
"""Generate evidence-only real PSS/E M1+M2 interconnection testcases."""

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
    DEFAULT_REAL_M1M2_INTERCONNECTION_SEED,
    REAL_M1M2_INTERCONNECTION_PROFILES,
    REAL_M1M2_SCHEMA_VERSION,
    generate_real_m1m2_interconnection_testcases,
    write_real_m1m2_interconnection_testcases,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate evidence-only interconnection testcases for the live "
            "remote PSS/E M1+M2 gym."
        )
    )
    parser.add_argument("--count", type=int, default=50, help="Number of testcases to generate.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_REAL_M1M2_INTERCONNECTION_SEED,
        help="Deterministic generation seed.",
    )
    parser.add_argument(
        "--profile",
        default="mixed",
        choices=list(REAL_M1M2_INTERCONNECTION_PROFILES),
        help="Difficulty/profile mix.",
    )
    parser.add_argument(
        "--processed-dir",
        help="Optional processed PSS/E directory containing inventories/pif6_2026_05_17/*.csv.",
    )
    parser.add_argument(
        "--output",
        default="real-data-new/generated_real_m1m2_interconnection_cases.json",
        help="Output JSON/JSONL path.",
    )
    parser.add_argument("--jsonl", action="store_true", help="Write one testcase JSON object per line.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print generation summary and first few cases without writing output.",
    )
    args = parser.parse_args()

    try:
        scenarios = generate_real_m1m2_interconnection_testcases(
            args.count,
            seed=args.seed,
            profile=args.profile,
            processed_dir=args.processed_dir,
        )
        generation: Dict[str, Any] = {
            "schema_version": REAL_M1M2_SCHEMA_VERSION,
            "count": args.count,
            "seed": args.seed,
            "profile": args.profile,
            "processed_dir": args.processed_dir,
            "label_policy": "evidence_only_no_new_project_approvals_without_exact_psse_results",
        }
        if args.summary_only:
            payload = {
                "ok": True,
                "schema_version": REAL_M1M2_SCHEMA_VERSION,
                "generation": generation,
                "scenario_count": len(scenarios),
                "label_counts": _count_by(scenarios, "oracle_label"),
                "difficulty_counts": _count_by(scenarios, "difficulty"),
                "preview": [scenario.to_dict() for scenario in scenarios[: min(5, len(scenarios))]],
            }
        else:
            payload = write_real_m1m2_interconnection_testcases(
                scenarios,
                args.output,
                jsonl=bool(args.jsonl),
                generation=generation,
            )
            payload["label_counts"] = _count_by(scenarios, "oracle_label")
            payload["difficulty_counts"] = _count_by(scenarios, "difficulty")
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


def _count_by(scenarios: Any, attr: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for scenario in scenarios:
        key = str(getattr(scenario, attr))
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())

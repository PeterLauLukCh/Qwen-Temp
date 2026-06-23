#!/usr/bin/env python3
"""Generate TRGC real M1+M2 human-engineer workflow episodes."""

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
    DEFAULT_REAL_M1M2_ENGINEER_CHALLENGE_SEED,
    DEFAULT_REAL_M1M2_ENGINEER_GYM_SEED,
    REAL_M1M2_ENGINEER_CHALLENGE_SCHEMA_VERSION,
    REAL_M1M2_ENGINEER_GYM_SCHEMA_VERSION,
    REAL_M1M2_ENGINEER_PROFILES,
    generate_real_m1m2_engineer_episodes,
    write_real_m1m2_engineer_episodes,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate TRGC engineer-gym episodes.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hidden-seed", type=int)
    parser.add_argument("--split", default="public", choices=["public", "hidden"])
    parser.add_argument("--profile", default="trgc_engineer", choices=list(REAL_M1M2_ENGINEER_PROFILES))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    try:
        seed = _resolve_seed(args)
        output = _resolve_output(args)
        schema_version = (
            REAL_M1M2_ENGINEER_CHALLENGE_SCHEMA_VERSION
            if args.profile == "trgc_engineer_challenge"
            else REAL_M1M2_ENGINEER_GYM_SCHEMA_VERSION
        )
        episodes = generate_real_m1m2_engineer_episodes(
            args.count,
            seed=seed,
            profile=args.profile,
            max_steps=args.max_steps,
        )
        generation = {
            "schema_version": schema_version,
            "count": args.count,
            "seed": seed,
            "split": args.split,
            "profile": args.profile,
            "max_steps": args.max_steps,
            "objective": "trgc_real_interconnection_engineer_workflow",
        }
        if args.summary_only:
            payload = {
                "ok": True,
                "schema_version": schema_version,
                "generation": generation,
                "episode_count": len(episodes),
                "curriculum_level_counts": _count_by(episodes, "curriculum_level"),
                "family_counts": _count_by(episodes, "family"),
                "difficulty_counts": _count_by(episodes, "difficulty"),
                "preview": [episode.to_dict(include_hidden=False) for episode in episodes[:5]],
            }
        else:
            payload = write_real_m1m2_engineer_episodes(
                episodes,
                output,
                generation=generation,
                jsonl=bool(args.jsonl),
            )
            payload["curriculum_level_counts"] = _count_by(episodes, "curriculum_level")
            payload["family_counts"] = _count_by(episodes, "family")
            payload["difficulty_counts"] = _count_by(episodes, "difficulty")
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2, sort_keys=True))
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _count_by(episodes: Any, attr: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for episode in episodes:
        key = str(getattr(episode, attr))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _resolve_seed(args: argparse.Namespace) -> int:
    if args.hidden_seed is not None:
        return args.hidden_seed
    if args.seed is not None:
        return args.seed
    if args.profile == "trgc_engineer_challenge":
        return DEFAULT_REAL_M1M2_ENGINEER_CHALLENGE_SEED
    return DEFAULT_REAL_M1M2_ENGINEER_GYM_SEED


def _resolve_output(args: argparse.Namespace) -> str:
    if args.output:
        return args.output
    if args.profile == "trgc_engineer_challenge":
        split = "hidden" if args.split == "hidden" else "public"
        return f"real-data-new/generated_trgc_real_m1m2_engineer_challenge_{split}.json"
    return "real-data-new/generated_trgc_real_m1m2_engineer_episodes.json"


if __name__ == "__main__":
    raise SystemExit(main())

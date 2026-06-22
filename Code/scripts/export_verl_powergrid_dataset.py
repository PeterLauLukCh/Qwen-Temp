#!/usr/bin/env python3
"""Export frozen IEEE14 M1+M2+EMT scenarios to verl parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from gridmind_mini.verl_powergrid import (
    DEFAULT_PROFILE,
    DEFAULT_TRAIN_COUNT,
    DEFAULT_TRAIN_SEED,
    DEFAULT_VAL_COUNT,
    DEFAULT_VAL_SEED,
    export_powergrid_verl_dataset,
)
from gridmind_mini.joint_benchmark import GENERATED_JOINT_PROFILES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="verl_data/powergrid_ieee14_emt_seed20260610",
        help="Directory for train.parquet, val.parquet, scenario JSON, and metadata.",
    )
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--val-count", type=int, default=DEFAULT_VAL_COUNT)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Convenience seed. Uses this value for train and seed+1 for validation unless explicit split seeds are provided.",
    )
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--val-seed", type=int, default=None)
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=list(GENERATED_JOINT_PROFILES),
        help="Generated joint scenario profile.",
    )
    args = parser.parse_args()
    train_seed = args.train_seed
    val_seed = args.val_seed
    if args.seed is not None:
        if train_seed is None:
            train_seed = args.seed
        if val_seed is None:
            val_seed = args.seed + 1
    if train_seed is None:
        train_seed = DEFAULT_TRAIN_SEED
    if val_seed is None:
        val_seed = DEFAULT_VAL_SEED

    metadata = export_powergrid_verl_dataset(
        Path(args.output_dir),
        train_count=args.train_count,
        val_count=args.val_count,
        train_seed=train_seed,
        val_seed=val_seed,
        profile=args.profile,
    )
    print(_json(metadata))
    return 0


def _json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())

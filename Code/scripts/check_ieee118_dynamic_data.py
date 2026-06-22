#!/usr/bin/env python3
"""Preflight public IEEE118 RAW/DYR availability for integrated M1+M2 runs."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "Code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "Code"))


def _prepare_runtime_cache_dirs() -> None:
    mpl_config = Path(tempfile.gettempdir()) / "gridmind_mplconfig"
    try:
        mpl_config.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))


_prepare_runtime_cache_dirs()

from gridmind_mini import validate_ieee118_public_dynamic_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether public IEEE118 RAW+DYR data is available."
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Always exit 0 after printing the preflight JSON.",
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help=(
            "Also run a small integrated IEEE118 M1+M2 probe. This catches "
            "RAW/DYR files that exist but do not converge in pandapower/ANDES."
        ),
    )
    parser.add_argument(
        "--runtime-timeout-s",
        type=float,
        default=180.0,
        help="Runtime validation timeout in seconds. Use <=0 to disable.",
    )
    args = parser.parse_args()

    result = validate_ieee118_public_dynamic_data()
    if args.runtime and result.get("ok"):
        result["runtime_validation"] = _runtime_validation(args.runtime_timeout_s)
        result["ok"] = bool(result.get("ok") and result["runtime_validation"].get("ok"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if args.allow_missing or result.get("ok") else 1


def _runtime_validation(timeout_s: float | None = 180.0) -> dict:
    _prepare_runtime_cache_dirs()
    from gridmind_mini import ToolRegistry  # noqa: WPS433

    registry = ToolRegistry()
    payload = {
        "case_path": "ieee118",
        "connection": {
            "bus": 10,
            "p_mw": 1.0,
            "connection_type": "solar",
            "is_ibr": True,
        },
        "transient": {
            "enabled": True,
            "required_for_approval": True,
            "case_path": "ieee118_dynamic",
            "disturbance": {
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            "simulation_time_s": 2.0,
            "max_samples": 5,
        },
    }
    try:
        with contextlib.redirect_stdout(sys.stderr), _runtime_timeout(timeout_s):
            result = registry.call_tool("run_integrated_assessment", payload)
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

    summary = result.get("summary") if isinstance(result, dict) else None
    m2 = result.get("m2_result") if isinstance(result, dict) else None
    alignment = result.get("case_alignment") if isinstance(result, dict) else None
    connection_application = (
        m2.get("connection_application")
        if isinstance(m2, dict)
        else None
    )
    ok = bool(
        isinstance(result, dict)
        and result.get("ok")
        and result.get("complete")
        and isinstance(summary, dict)
        and summary.get("m1_recommendation") is not None
        and summary.get("m2_status") not in {None, "unavailable"}
        and isinstance(connection_application, dict)
        and connection_application.get("applied")
    )
    return {
        "ok": ok,
        "tool": result.get("tool") if isinstance(result, dict) else None,
        "recommendation": result.get("recommendation") if isinstance(result, dict) else None,
        "complete": result.get("complete") if isinstance(result, dict) else None,
        "reason_codes": result.get("reason_codes") if isinstance(result, dict) else None,
        "case_alignment_error_type": (
            alignment.get("error_type") if isinstance(alignment, dict) else None
        ),
        "case_alignment_message": (
            alignment.get("message") if isinstance(alignment, dict) else None
        ),
        "summary": summary,
        "m1_error_type": (
            result.get("m1_result", {}).get("error_type")
            if isinstance(result, dict) and isinstance(result.get("m1_result"), dict)
            else None
        ),
        "m2_error_type": m2.get("error_type") if isinstance(m2, dict) else None,
        "m2_reason_codes": (
            m2.get("metrics", {}).get("reason_codes")
            if isinstance(m2, dict) and isinstance(m2.get("metrics"), dict)
            else None
        ),
        "m2_connection_applied": (
            connection_application.get("applied")
            if isinstance(connection_application, dict)
            else None
        ),
        "message": None
        if ok
        else (
            "IEEE118 RAW/DYR files are present but the integrated runtime probe "
            "did not produce a complete M1+M2 assessment."
        ),
    }


@contextlib.contextmanager
def _runtime_timeout(timeout_s: float | None):
    if (
        timeout_s is None
        or timeout_s <= 0
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
    ):
        yield
        return

    def _handler(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"IEEE118 runtime validation timed out after {timeout_s:g} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())

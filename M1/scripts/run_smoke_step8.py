#!/usr/bin/env python3
"""Smoke test for Step 8: anti-hallucination routing and grounding checks.

Examples:
    python3 Code/scripts/run_smoke_step8.py --message "max load capacity at bus 10 on ieee14"
    python3 Code/scripts/run_smoke_step8.py --message "max load capacity at bus 10 on ieee14" --execute
    python3 Code/scripts/run_smoke_step8.py --response "The capacity is 127 MW."
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

from gridmind_mini import (  # noqa: E402
    ToolRegistry,
    ToolRegistryError,
    detect_capacity_route,
    handle_forced_capacity_routing,
    validate_grounding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step 8 Mini Grid-Mind guardrail smoke test.")
    parser.add_argument(
        "--message",
        default="What is the max load capacity at bus 10 on ieee14?",
        help="User message to inspect for forced capacity routing.",
    )
    parser.add_argument("--case", dest="case_path", help="Optional case context.")
    parser.add_argument("--bus", type=int, help="Optional bus context.")
    parser.add_argument(
        "--type",
        dest="connection_type",
        choices=["load", "solar", "wind", "bess", "hybrid", "synchronous"],
        help="Optional connection-type context.",
    )
    parser.add_argument("--max-mw", type=float, help="Optional capacity-search upper bound.")
    parser.add_argument("--tolerance-mw", type=float, help="Optional capacity-search tolerance.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="If the route is ready, execute find_max_capacity through ToolRegistry.",
    )
    parser.add_argument(
        "--response",
        default="The maximum capacity is approximately 127 MW.",
        help="Assistant response text to inspect for ungrounded numerical claims.",
    )
    parser.add_argument(
        "--invoked-tool",
        action="append",
        default=[],
        help="Tool name credited for grounding. Repeat for multiple tools.",
    )
    args = parser.parse_args()

    context: Dict[str, Any] = {}
    for key in ("case_path", "bus", "connection_type", "max_mw", "tolerance_mw"):
        value = getattr(args, key)
        if value is not None:
            context[key] = value

    try:
        if args.execute:
            route_output = handle_forced_capacity_routing(
                args.message,
                ToolRegistry(),
                context=context,
            )
        else:
            route_output = {
                "routed": False,
                "executed": False,
                "decision": detect_capacity_route(args.message, context=context).to_dict(),
                "result": None,
                "clarification": None,
            }
            route_output["routed"] = route_output["decision"]["should_route"]
            route_output["clarification"] = route_output["decision"]["clarification_prompt"]
    except (ToolRegistryError, ValueError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1

    grounding = validate_grounding(
        args.response,
        invoked_tools=_invoked_tools(args.invoked_tool, route_output),
    )
    output = {
        "ok": True,
        "route": route_output,
        "grounding": grounding.to_dict(),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _invoked_tools(cli_tools: List[str], route_output: Dict[str, Any]) -> List[str]:
    tools = list(cli_tools)
    if route_output.get("executed") and isinstance(route_output.get("result"), dict):
        tool_name = route_output["result"].get("tool")
        if isinstance(tool_name, str):
            tools.append(tool_name)
    return tools


if __name__ == "__main__":
    raise SystemExit(main())

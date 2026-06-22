#!/usr/bin/env python3
"""Smoke test for Step 3: call the Mini Grid-Mind tool registry.

Examples:
    python3 Code/scripts/run_smoke_step3.py --list-tools
    python3 Code/scripts/run_smoke_step3.py --openai-specs
    python3 Code/scripts/run_smoke_step3.py --tool list_cases
    python3 Code/scripts/run_smoke_step3.py --tool inspect_violations --args '{"case_path":"ieee118"}'
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
    parser = argparse.ArgumentParser(description="Run Step 3 Mini Grid-Mind registry smoke test.")
    parser.add_argument("--list-tools", action="store_true", help="List registry tools and implementation status.")
    parser.add_argument("--openai-specs", action="store_true", help="Print OpenAI-compatible implemented tool specs.")
    parser.add_argument("--include-unimplemented", action="store_true", help="Include roadmap tools in listings/specs.")
    parser.add_argument("--tool", help="Tool name to call.")
    parser.add_argument("--args", default="{}", help="Tool arguments as a JSON object.")
    args = parser.parse_args()

    registry = ToolRegistry()
    try:
        if args.list_tools:
            output: Dict[str, Any] = registry.list_tools(
                include_unimplemented=args.include_unimplemented
            )
        elif args.openai_specs:
            output = {
                "ok": True,
                "tool_specs": registry.openai_tool_specs(
                    include_unimplemented=args.include_unimplemented
                ),
            }
        elif args.tool:
            tool_args = json.loads(args.args)
            if not isinstance(tool_args, dict):
                raise ToolRegistryError("--args must decode to a JSON object")
            output = registry.call_tool(args.tool, tool_args)
        else:
            output = registry.list_tools(include_unimplemented=args.include_unimplemented)
    except (ToolRegistryError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

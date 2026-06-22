#!/usr/bin/env python3
"""Smoke test for Step 9: Qwen/vLLM prompt and response adapter.

Examples:
    python3 Code/scripts/run_smoke_step9.py --dry-run
    python3 Code/scripts/run_smoke_step9.py --show-chatml
    python3 Code/scripts/run_smoke_step9.py --host 127.0.0.1 --port 8000 --list-models
    python3 Code/scripts/run_smoke_step9.py --interactive --chat
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

from gridmind_mini import (  # noqa: E402
    DEFAULT_LOCAL_MODEL,
    LLMClientError,
    ToolRegistry,
    VLLMConfig,
    VLLMOpenAIClient,
    build_chat_messages,
    parse_tool_calls_from_text,
    render_qwen_chatml,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step 9 Mini Grid-Mind vLLM smoke test.")
    parser.add_argument(
        "--base-url",
        help="Full local vLLM OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1.",
    )
    parser.add_argument("--host", help="Local vLLM host/IP. Used when --base-url is omitted.")
    parser.add_argument("--port", type=int, help="Local vLLM port. Used when --base-url is omitted.")
    parser.add_argument("--scheme", default="http", choices=["http", "https"], help="Endpoint scheme.")
    parser.add_argument("--api-path", default="/v1", help="OpenAI-compatible API path, usually /v1.")
    parser.add_argument(
        "--model",
        default=DEFAULT_LOCAL_MODEL,
        help=(
            "Served model name. Use 'auto' to read the first model id from /v1/models, "
            "which is recommended for local vLLM servers."
        ),
    )
    parser.add_argument("--api-key", help="Optional bearer token for the OpenAI-compatible server.")
    parser.add_argument(
        "--message",
        default="Run a power flow on IEEE 14 and report any violations.",
        help="User message for prompt construction.",
    )
    parser.add_argument("--case", dest="case_path", help="Optional case context.")
    parser.add_argument("--bus", type=int, help="Optional bus context.")
    parser.add_argument("--mw", type=float, help="Optional MW context.")
    parser.add_argument(
        "--type",
        dest="connection_type",
        choices=["load", "solar", "wind", "bess", "hybrid", "synchronous"],
        help="Optional connection-type context.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--list-models", action="store_true", help="Call GET /v1/models.")
    parser.add_argument("--chat", action="store_true", help="Call POST /v1/chat/completions.")
    parser.add_argument(
        "--completion",
        action="store_true",
        help="Call POST /v1/completions with a Qwen ChatML-rendered prompt.",
    )
    parser.add_argument(
        "--show-chatml",
        action="store_true",
        help="Include the Qwen ChatML fallback prompt in the output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print constructed prompt/tool payloads. This is the default when no endpoint action is selected.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for host and port when calling a local vLLM endpoint.",
    )
    args = parser.parse_args()

    calls_endpoint = args.list_models or args.chat or args.completion
    try:
        base_url = _resolve_base_url(args, calls_endpoint=calls_endpoint)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1

    registry = ToolRegistry()
    context: Dict[str, Any] = {}
    for key in ("case_path", "bus", "mw", "connection_type"):
        value = getattr(args, key)
        if value is not None:
            context[key] = value

    prompt = build_chat_messages(
        registry,
        args.message,
        context=context,
    )
    messages = prompt.messages
    tool_specs = registry.openai_tool_specs()
    output: Dict[str, Any] = {
        "ok": True,
        "base_url": base_url,
        "model": args.model,
        "endpoint_paths": {
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "completion": "/v1/completions",
        },
        "context_hints": prompt.context_hints.to_dict(),
        "messages": messages,
        "tool_count": len(tool_specs),
        "tool_names": [spec["function"]["name"] for spec in tool_specs],
        "dry_run": args.dry_run or not (args.list_models or args.chat or args.completion),
    }
    if args.show_chatml or args.completion:
        output["qwen_chatml_prompt"] = render_qwen_chatml(messages)

    if output["dry_run"]:
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0

    client = VLLMOpenAIClient(
        VLLMConfig(
            base_url=base_url,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )

    try:
        if args.list_models:
            output["models_response"] = client.list_models()
        if args.chat:
            chat_response = client.chat(messages, tools=tool_specs)
            output["chat_response"] = {
                "content": chat_response.content,
                "reasoning_content": chat_response.reasoning_content,
                "finish_reason": chat_response.finish_reason,
                "tool_calls": [call.to_openai_dict() for call in chat_response.tool_calls],
            }
        if args.completion:
            completion_response = client.complete(output["qwen_chatml_prompt"])
            output["completion_response"] = {
                "text": completion_response.text,
                "finish_reason": completion_response.finish_reason,
                "parsed_text_tool_calls": [
                    call.to_openai_dict()
                    for call in parse_tool_calls_from_text(completion_response.text)
                ],
            }
    except (LLMClientError, ValueError) as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _resolve_base_url(args: argparse.Namespace, *, calls_endpoint: bool) -> str:
    if args.base_url:
        return args.base_url.rstrip("/")

    host = args.host
    port = args.port
    if calls_endpoint and args.interactive:
        host = _prompt_text("Local vLLM host/IP", host or "127.0.0.1")
        port = _prompt_int("Local vLLM port", port or 8000)

    host = host or "127.0.0.1"
    port = port or 8000
    _validate_port(port)
    api_path = "/" + str(args.api_path).strip("/")
    return f"{args.scheme}://{host}:{port}{api_path}"


def _prompt_text(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _prompt_int(label: str, default: int) -> int:
    value = input(f"{label} [{default}]: ").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _validate_port(port: int) -> None:
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("Local vLLM port must be an integer between 1 and 65535")


if __name__ == "__main__":
    raise SystemExit(main())

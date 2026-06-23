#!/usr/bin/env python3
"""Run IdealTalk/OpenAI-compatible API smoke tests and engineer-gym episodes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "Code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "Code"))

from gridmind_mini import (  # noqa: E402
    AgentConfig,
    GridMindAgent,
    HybridRemoteM1M2CacheRunner,
    RealM1M2EngineerEnv,
    StudyMemoryStore,
    ToolRegistry,
    VLLMConfig,
    VLLMOpenAIClient,
    engineer_results_summary,
    filter_real_m1m2_engineer_episodes,
    load_real_m1m2_engineer_episodes,
)
from gridmind_mini.llm import LLMClientError  # noqa: E402


IDEATALK_DEFAULT_BASE_URL = "https://idealab.alibaba-inc.com/api/openai/v1"
IDEATALK_DEFAULT_MODEL = "Qwen3.7-Plus-DogFooding"
IDEATALK_API_KEY_ENV = "IDEATALK_API_KEY"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Call the IdealTalk OpenAI-compatible endpoint, then optionally run "
            "TRGC real M1+M2 engineer-gym episodes through GridMindAgent."
        )
    )
    parser.add_argument("--base-url", default=os.environ.get("IDEATALK_BASE_URL", IDEATALK_DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("IDEATALK_MODEL", IDEATALK_DEFAULT_MODEL))
    parser.add_argument("--api-key", help="Bearer token override. Prefer exporting IDEATALK_API_KEY.")
    parser.add_argument("--api-key-env", default=IDEATALK_API_KEY_ENV)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--smoke-max-tokens", type=int, default=64)
    parser.add_argument("--smoke-message", default="hello")
    parser.add_argument("--skip-api-smoke", action="store_true")
    parser.add_argument("--api-smoke-only", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--episodes",
        default=None,
    )
    parser.add_argument("--challenge", action="store_true", help="Run the stricter TRGC engineer challenge fixture.")
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--curriculum-level", action="append", default=[])
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--difficulty", action="append", default=[])
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of engineer-gym episodes to run after the smoke call. Default: 1.",
    )
    parser.add_argument("--list-episodes", action="store_true")
    parser.add_argument("--max-tool-rounds", type=int, default=8)
    parser.add_argument("--strict-agent", action="store_true", help="Disable deterministic report fallback.")
    parser.add_argument("--include-messages", action="store_true", help="Include full LLM conversation messages in output.")
    parser.add_argument("--memory-dir")
    parser.add_argument("--cache-dir", default="Code/benchmark_results/real_m1m2_engineer_cache")
    parser.add_argument(
        "--live-remote",
        action="store_true",
        help="Actually call the Windows PSS/E remote worker for executable M1/M2 jobs.",
    )
    parser.add_argument("--include-hidden", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.challenge:
        args.strict_agent = True
    episodes_path = args.episodes or (
        "real-data-new/generated_trgc_real_m1m2_engineer_challenge_public.json"
        if args.challenge
        else "real-data-new/generated_trgc_real_m1m2_engineer_episodes.json"
    )

    api_key, api_key_source = _resolve_api_key(args)
    if not api_key:
        print(
            _json(
                {
                    "ok": False,
                    "error_type": "missing_api_key",
                    "error": f"Set {args.api_key_env} or pass --api-key.",
                    "base_url": args.base_url,
                    "model": args.model,
                }
            )
        )
        return 1

    client = VLLMOpenAIClient(
        VLLMConfig(
            base_url=args.base_url.rstrip("/"),
            model=args.model,
            api_key=api_key,
            timeout_s=args.timeout_s,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )

    output: Dict[str, Any] = {
        "ok": True,
        "mode": "ideatalk_real_m1m2_engineer_gym",
        "base_url": args.base_url.rstrip("/"),
        "model": args.model,
        "api_key_source": api_key_source,
    }

    if args.list_models:
        models = _list_models(client)
        output["models"] = models
        output["ok"] = bool(models["ok"])
        print(_json(output))
        return 0 if output["ok"] else 1

    if not args.skip_api_smoke:
        smoke = _run_api_smoke(
            client,
            message=args.smoke_message,
            temperature=args.temperature,
            max_tokens=args.smoke_max_tokens,
        )
        output["api_smoke"] = smoke
        if not smoke["ok"]:
            output["ok"] = False
            print(_json(output))
            return 1

    if args.api_smoke_only:
        print(_json(output))
        return 0

    try:
        episodes = filter_real_m1m2_engineer_episodes(
            load_real_m1m2_engineer_episodes(episodes_path),
            episode_ids=args.episode,
            curriculum_levels=args.curriculum_level,
            families=args.family,
            difficulties=args.difficulty,
            limit=args.limit,
        )
    except Exception as exc:
        output.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        print(_json(output))
        return 1

    if args.list_episodes:
        output["episodes"] = {
            "path": episodes_path,
            "episode_count": len(episodes),
            "items": [episode.to_dict(include_hidden=False) for episode in episodes],
        }
        print(_json(output))
        return 0
    if not episodes:
        output.update({"ok": False, "error_type": "no_episodes", "error": "No episodes selected."})
        print(_json(output))
        return 1

    try:
        suite = _run_engineer_gym(args, client, episodes)
    except Exception as exc:
        output.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        print(_json(output))
        return 1

    output.update(
        {
            "ok": bool(suite["ok"]),
            "episodes": episodes_path,
            "live_remote": bool(args.live_remote),
            "cache_dir": args.cache_dir,
            "suite": suite,
        }
    )
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_json(output) + "\n", encoding="utf-8")
        print(
            _json(
                {
                    "ok": output["ok"],
                    "output": str(output_path),
                    "total": suite["total"],
                    "passed": suite["passed"],
                    "failed": suite["failed"],
                    "average_reward": suite["average_reward"],
                    "by_curriculum_level": suite["by_curriculum_level"],
                    "by_difficulty": suite["by_difficulty"],
                    "by_challenge_family": suite["by_challenge_family"],
                    "reward_components": suite["reward_components"],
                    "numeric_accuracy": suite["numeric_accuracy"],
                    "poc_accuracy": suite["poc_accuracy"],
                    "memo_section_score": suite["memo_section_score"],
                    "strict_agent": suite["strict_agent"],
                }
            )
        )
    else:
        print(_json(output))
    return 0 if output["ok"] else 1


def _resolve_api_key(args: argparse.Namespace) -> Tuple[Optional[str], str]:
    if args.api_key:
        return args.api_key, "cli:--api-key"
    primary = os.environ.get(args.api_key_env, "").strip()
    if primary:
        return primary, f"env:{args.api_key_env}"
    if args.api_key_env != IDEATALK_API_KEY_ENV:
        fallback = os.environ.get(IDEATALK_API_KEY_ENV, "").strip()
        if fallback:
            return fallback, f"env:{IDEATALK_API_KEY_ENV}"
    return None, "missing"


def _run_api_smoke(
    client: VLLMOpenAIClient,
    *,
    message: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    try:
        response = client.chat(
            [{"role": "user", "content": message}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except LLMClientError as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    return {
        "ok": True,
        "finish_reason": response.finish_reason,
        "content": response.content,
        "reasoning_content_present": bool(response.reasoning_content),
        "tool_call_count": len(response.tool_calls),
    }


def _list_models(client: VLLMOpenAIClient) -> Dict[str, Any]:
    try:
        payload = client.list_models()
    except LLMClientError as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}

    model_ids = []
    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                model_ids.append(item["id"])
    return {
        "ok": True,
        "model_count": len(model_ids),
        "model_ids": model_ids,
        "raw": payload,
    }


def _run_engineer_gym(
    args: argparse.Namespace,
    client: VLLMOpenAIClient,
    episodes: Any,
) -> Dict[str, Any]:
    memory_store = StudyMemoryStore(args.memory_dir) if args.memory_dir else None
    registry = ToolRegistry(memory_store=memory_store)
    agent = GridMindAgent(
        registry=registry,
        llm_client=client,
        memory_store=memory_store,
        config=_agent_config(args),
    )
    tool_runner = HybridRemoteM1M2CacheRunner(
        registry=registry,
        cache_dir=args.cache_dir,
        live_remote=bool(args.live_remote),
    )

    start = time.perf_counter()
    results = []
    for index, episode in enumerate(episodes, start=1):
        print(
            json.dumps(
                {
                    "ideatalk_engineer_progress": {
                        "event": "episode_start",
                        "index": index,
                        "total": len(episodes),
                        "episode_id": episode.episode_id,
                        "curriculum_level": episode.curriculum_level,
                        "difficulty": episode.difficulty,
                    }
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        env = RealM1M2EngineerEnv(registry=registry, tool_runner=tool_runner)
        env.reset(episode)
        result = env.run_agent(agent, include_messages=args.include_messages)
        results.append(result)
        print(
            json.dumps(
                {
                    "ideatalk_engineer_progress": {
                        "event": "episode_done",
                        "index": index,
                        "total": len(episodes),
                        "episode_id": episode.episode_id,
                        "passed": result.passed,
                        "reward": result.reward.total,
                        "hard_penalties": dict(result.reward.hard_penalties),
                    }
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    return engineer_results_summary(
        results,
        duration_s=time.perf_counter() - start,
        include_hidden=args.include_hidden,
        strict_agent=args.strict_agent,
    )


def _json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _agent_config(args: argparse.Namespace) -> AgentConfig:
    return AgentConfig(
        max_tool_rounds=args.max_tool_rounds,
        enable_deterministic_report=not args.strict_agent,
        use_deterministic_report_when_final_empty=not args.strict_agent,
        use_deterministic_report_on_max_rounds=not args.strict_agent,
    )


if __name__ == "__main__":
    raise SystemExit(main())

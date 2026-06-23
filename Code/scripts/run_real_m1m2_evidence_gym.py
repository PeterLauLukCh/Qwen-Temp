#!/usr/bin/env python3
"""Run TRGC real M1+M2 evidence-gym episodes with the current GridMind agent."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "Code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "Code"))

from gridmind_mini import (  # noqa: E402
    DEFAULT_LOCAL_MODEL,
    AgentConfig,
    GridMindAgent,
    StudyMemoryStore,
    ToolRegistry,
    VLLMConfig,
    VLLMOpenAIClient,
    evidence_results_summary,
    filter_real_m1m2_evidence_episodes,
    load_real_m1m2_evidence_episodes,
)
from gridmind_mini.real_m1m2_evidence_gym import RealM1M2EvidenceEnv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TRGC evidence-gym episodes.")
    parser.add_argument(
        "--episodes",
        default="real-data-new/generated_trgc_real_m1m2_evidence_episodes.json",
    )
    parser.add_argument("--base-url", help="Full vLLM base URL, e.g. http://127.0.0.1:8037/v1.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--scheme", default="http", choices=["http", "https"])
    parser.add_argument("--api-path", default="/v1")
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--api-key")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-tool-rounds", type=int, default=5)
    parser.add_argument("--memory-dir")
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--difficulty", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--list-episodes", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--include-hidden", action="store_true")
    args = parser.parse_args()

    try:
        episodes = filter_real_m1m2_evidence_episodes(
            load_real_m1m2_evidence_episodes(args.episodes),
            episode_ids=args.episode,
            families=args.family,
            difficulties=args.difficulty,
            limit=args.limit,
        )
    except Exception as exc:
        print(_json({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}))
        return 1

    if args.list_episodes:
        print(
            _json(
                {
                    "ok": True,
                    "episodes": args.episodes,
                    "episode_count": len(episodes),
                    "items": [episode.to_dict(include_hidden=False) for episode in episodes],
                }
            )
        )
        return 0
    if not episodes:
        print(_json({"ok": False, "error_type": "no_episodes", "error": "No episodes selected."}))
        return 1

    base_url = args.base_url.rstrip("/") if args.base_url else _base_url(args)
    memory_store = StudyMemoryStore(args.memory_dir) if args.memory_dir else None
    agent = GridMindAgent(
        registry=ToolRegistry(memory_store=memory_store),
        llm_client=VLLMOpenAIClient(
            VLLMConfig(
                base_url=base_url,
                model=args.model,
                api_key=args.api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        ),
        memory_store=memory_store,
        config=AgentConfig(max_tool_rounds=args.max_tool_rounds),
    )

    start = time.perf_counter()
    results = []
    for index, episode in enumerate(episodes, start=1):
        print(
            json.dumps(
                {
                    "real_m1m2_evidence_progress": {
                        "event": "episode_start",
                        "index": index,
                        "total": len(episodes),
                        "episode_id": episode.episode_id,
                        "family": episode.family,
                        "difficulty": episode.difficulty,
                    }
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        env = RealM1M2EvidenceEnv()
        env.reset(episode)
        result = env.run_agent(agent)
        results.append(result)
        print(
            json.dumps(
                {
                    "real_m1m2_evidence_progress": {
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

    suite = evidence_results_summary(
        results,
        duration_s=time.perf_counter() - start,
        include_hidden=args.include_hidden,
    )
    output = {
        "ok": suite["ok"],
        "mode": "live_agent_evidence_gym",
        "episodes": args.episodes,
        "base_url": base_url,
        "model": args.model,
        "suite": suite,
    }
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
                    "by_family": suite["by_family"],
                    "by_difficulty": suite["by_difficulty"],
                }
            )
        )
    else:
        print(_json(output))
    return 0 if output["ok"] else 1


def _base_url(args: argparse.Namespace) -> str:
    api_path = "/" + str(args.api_path).strip("/")
    return f"{args.scheme}://{args.host}:{args.port}{api_path}"


def _json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())

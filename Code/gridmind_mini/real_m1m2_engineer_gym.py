"""Human-engineer workflow gym for TRGC real M1+M2 interconnection review."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .anti_hallucination import validate_tool_call_policy
from .observations import build_tool_observation
from .real_case_dossier import REAL_DOSSIER_TOOL_NAMES
from .remote_psse import REMOTE_M1M2_SCHEMA_VERSION, REMOTE_M1M2_TOOL
from .tools import ToolRegistry, ToolRegistryError
from .trgc_requirements import TRGC_REQUIREMENT_CATALOG, TRGCRequirement


REAL_M1M2_ENGINEER_GYM_SCHEMA_VERSION = "real_m1m2_engineer_episode_v1"
DEFAULT_REAL_M1M2_ENGINEER_GYM_SEED = 20260624
REAL_M1M2_ENGINEER_PROFILES = ("trgc_engineer",)

ENGINEER_CURRICULUM_LEVELS = (
    "level1_scope_data_readiness",
    "level2_case_inspection",
    "level3_supported_m1m2",
    "level4_mixed_trgc",
    "level5_engineer_memo",
)
ENGINEER_DIFFICULTIES = ("easy", "medium", "hard")

DEFAULT_ENGINEER_REWARD_WEIGHTS = {
    "case_poc_model_inspection": 0.20,
    "evidence_selection": 0.20,
    "trgc_mapping": 0.20,
    "grounded_memo": 0.20,
    "no_proxy": 0.10,
    "efficiency": 0.10,
}
DEFAULT_ENGINEER_HARD_PENALTIES = {
    "unsupported_approval": -1.0,
    "proxy_baseline": -0.7,
    "ungrounded_psse_claim": -0.5,
    "false_capability_claim": -0.3,
    "wrong_poc_claim": -0.3,
}

INSPECTION_TOOLS = set(REAL_DOSSIER_TOOL_NAMES)
VALID_ENGINEER_TOOLS = INSPECTION_TOOLS | {
    "list_remote_psse_m1m2_cases",
    REMOTE_M1M2_TOOL,
}
FALSE_CAPABILITY_CLAIMS = (
    "does not load sav/dyr/dll",
    "does not load sav",
    "does not load dyr",
    "does not load dll",
    "no sav/dyr/dll",
)
REFUSAL_TERMS = (
    "unsupported",
    "not supported",
    "cannot validate",
    "not validated",
    "outside scope",
    "insufficient evidence",
    "cannot approve",
)
MISSING_TERMS = (
    "missing",
    "not provided",
    "required",
    "need",
    "insufficient data",
    "insufficient evidence",
)
SUPPORTED_RESULT_TERMS = (
    "pss/e",
    "psse",
    "m1",
    "m2",
    "approve",
    "pass",
)
BOUNDARY_TERMS = (
    "bounded",
    "subset",
    "not full trgc",
    "does not validate",
    "remain unsupported",
    "fault",
    "droop",
    "scr",
    "pscad",
    "emt",
)


ToolRunner = Callable[[str, Mapping[str, Any]], Dict[str, Any]]


@dataclass(frozen=True)
class EngineerRewardBreakdown:
    case_poc_model_inspection: float = 0.0
    evidence_selection: float = 0.0
    trgc_mapping: float = 0.0
    grounded_memo: float = 0.0
    no_proxy: float = 0.0
    efficiency: float = 0.0
    hard_penalties: Mapping[str, float] = field(default_factory=dict)
    total: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_poc_model_inspection": self.case_poc_model_inspection,
            "evidence_selection": self.evidence_selection,
            "trgc_mapping": self.trgc_mapping,
            "grounded_memo": self.grounded_memo,
            "no_proxy": self.no_proxy,
            "efficiency": self.efficiency,
            "hard_penalties": dict(self.hard_penalties),
            "total": self.total,
        }


@dataclass(frozen=True)
class EngineerAction:
    type: str
    name: Optional[str] = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "name": self.name,
            "arguments": dict(self.arguments),
            "text": self.text,
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, "")}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EngineerAction":
        action_type = _required_str(payload, "type")
        if action_type not in {"tool_call", "final_answer"}:
            raise ValueError("action type must be 'tool_call' or 'final_answer'")
        return cls(
            type=action_type,
            name=_optional_str(payload.get("name")),
            arguments=_mapping_value(payload, "arguments"),
            text=_optional_str(payload.get("text")),
        )


@dataclass(frozen=True)
class EngineerObservation:
    episode_id: str
    step_index: int
    user_message: str
    context: Mapping[str, Any]
    tool_observations: Sequence[Mapping[str, Any]] = ()
    terminated: bool = False
    truncated: bool = False
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": REAL_M1M2_ENGINEER_GYM_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "user_message": self.user_message,
            "context": dict(self.context),
            "tool_observations": [dict(item) for item in self.tool_observations],
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        if self.message:
            payload["message"] = self.message
        return payload


@dataclass(frozen=True)
class RealM1M2EngineerEpisode:
    episode_id: str
    user_message: str
    curriculum_level: str
    family: str
    difficulty: str
    visible_context: Mapping[str, Any]
    hidden_oracle: Mapping[str, Any]
    max_steps: int = 8
    schema_version: str = REAL_M1M2_ENGINEER_GYM_SCHEMA_VERSION

    def to_dict(self, *, include_hidden: bool = True) -> Dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "user_message": self.user_message,
            "curriculum_level": self.curriculum_level,
            "family": self.family,
            "difficulty": self.difficulty,
            "visible_context": dict(self.visible_context),
            "max_steps": self.max_steps,
        }
        if include_hidden:
            payload["hidden_oracle"] = dict(self.hidden_oracle)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RealM1M2EngineerEpisode":
        return cls(
            episode_id=_required_str(payload, "episode_id"),
            user_message=_required_str(payload, "user_message"),
            curriculum_level=_required_str(payload, "curriculum_level"),
            family=_required_str(payload, "family"),
            difficulty=_required_str(payload, "difficulty"),
            visible_context=_mapping_value(payload, "visible_context"),
            hidden_oracle=_mapping_value(payload, "hidden_oracle"),
            max_steps=int(payload.get("max_steps", 8)),
            schema_version=str(payload.get("schema_version") or REAL_M1M2_ENGINEER_GYM_SCHEMA_VERSION),
        )

    def to_verl_sample(self) -> Dict[str, Any]:
        return {
            "data_source": "real_m1m2_engineer_gym",
            "ability": "trgc_interconnection_engineer_workflow",
            "prompt": self.user_message,
            "context": dict(self.visible_context),
            "tools": _engineer_tool_specs(),
            "reward_model": {
                "style": "hidden_oracle",
                "schema_version": self.schema_version,
                "episode_id": self.episode_id,
                "hidden_oracle": dict(self.hidden_oracle),
            },
            "extra_info": {
                "curriculum_level": self.curriculum_level,
                "family": self.family,
                "difficulty": self.difficulty,
                "max_steps": self.max_steps,
            },
        }


@dataclass(frozen=True)
class EngineerEpisodeResult:
    episode: RealM1M2EngineerEpisode
    actions: Sequence[Mapping[str, Any]]
    tool_records: Sequence[Mapping[str, Any]]
    final_answer: str
    reward: EngineerRewardBreakdown
    observations: Sequence[Mapping[str, Any]]
    terminated: bool
    truncated: bool
    status: str
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        return self.reward.total >= 0.8 and not self.reward.hard_penalties

    def to_dict(self, *, include_hidden: bool = False) -> Dict[str, Any]:
        return {
            "episode": self.episode.to_dict(include_hidden=include_hidden),
            "passed": self.passed,
            "status": self.status,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "duration_s": self.duration_s,
            "actions": [dict(item) for item in self.actions],
            "tool_records": [dict(item) for item in self.tool_records],
            "final_answer": self.final_answer,
            "reward": self.reward.to_dict(),
            "observations": [dict(item) for item in self.observations],
        }


class RealM1M2EngineerEnv:
    """Standalone POMDP-style engineer workflow environment."""

    def __init__(
        self,
        *,
        registry: Optional[ToolRegistry] = None,
        tool_runner: Optional[ToolRunner] = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.tool_runner = tool_runner
        self.episode: Optional[RealM1M2EngineerEpisode] = None
        self.step_index = 0
        self.actions: List[Dict[str, Any]] = []
        self.tool_records: List[Dict[str, Any]] = []
        self.tool_observations: List[Dict[str, Any]] = []
        self.observations: List[Dict[str, Any]] = []
        self.final_answer = ""
        self.terminated = False
        self.truncated = False
        self._last_reward = 0.0

    def reset(self, episode: RealM1M2EngineerEpisode | Mapping[str, Any]) -> EngineerObservation:
        self.episode = episode if isinstance(episode, RealM1M2EngineerEpisode) else RealM1M2EngineerEpisode.from_mapping(episode)
        self.step_index = 0
        self.actions = []
        self.tool_records = []
        self.tool_observations = []
        self.observations = []
        self.final_answer = ""
        self.terminated = False
        self.truncated = False
        self._last_reward = 0.0
        observation = self._observation()
        self.observations.append(observation.to_dict())
        return observation

    def step(self, action: EngineerAction | Mapping[str, Any]) -> Tuple[EngineerObservation, float, bool, bool, Dict[str, Any]]:
        if self.episode is None:
            raise RuntimeError("reset must be called before step")
        if self.terminated or self.truncated:
            raise RuntimeError("episode is already done")
        parsed = action if isinstance(action, EngineerAction) else EngineerAction.from_mapping(action)
        self.actions.append(parsed.to_dict())
        message = None
        if parsed.type == "tool_call":
            record = self._execute_tool_action(parsed)
            self.tool_records.append(record)
            self.tool_observations.append(_compact_observation(record))
            if _is_forbidden_proxy_record(self.episode, record):
                self.terminated = True
                message = "Forbidden proxy action terminated the episode."
        else:
            self.final_answer = parsed.text or ""
            self.terminated = True
            message = "Final answer submitted."
        self.step_index += 1
        if self.step_index >= self.episode.max_steps and not self.terminated:
            self.truncated = True
            message = "Maximum episode steps reached."
        reward = score_engineer_trajectory(
            self.episode,
            tool_records=self.tool_records,
            final_answer=self.final_answer,
            step_count=self.step_index,
            terminated=self.terminated,
            truncated=self.truncated,
        )
        delta = reward.total - self._last_reward
        self._last_reward = reward.total
        observation = self._observation(message=message)
        self.observations.append(observation.to_dict())
        return observation, delta, self.terminated, self.truncated, {"reward": reward.to_dict()}

    def run_agent(self, agent: Any) -> EngineerEpisodeResult:
        if self.episode is None:
            raise RuntimeError("reset must be called before run_agent")
        if not hasattr(agent, "run_turn"):
            raise ValueError("agent must expose run_turn(message, context=...)")
        start = time.perf_counter()
        result = agent.run_turn(self.episode.user_message, context=self._agent_context())
        self.tool_records = [_record_from_agent_tool(item) for item in getattr(result, "tool_records", []) or []]
        self.actions = [
            {"type": "tool_call", "name": item.get("name"), "arguments": dict(item.get("arguments") or {})}
            for item in self.tool_records
        ]
        self.final_answer = str(getattr(result, "output_text", "") or "")
        self.actions.append({"type": "final_answer", "text": self.final_answer})
        self.step_index = min(len(self.actions), self.episode.max_steps)
        self.terminated = True
        self.truncated = len(self.actions) > self.episode.max_steps
        reward = score_engineer_trajectory(
            self.episode,
            tool_records=self.tool_records,
            final_answer=self.final_answer,
            step_count=self.step_index,
            terminated=True,
            truncated=self.truncated,
        )
        return EngineerEpisodeResult(
            episode=self.episode,
            actions=self.actions,
            tool_records=self.tool_records,
            final_answer=self.final_answer,
            reward=reward,
            observations=self.observations,
            terminated=True,
            truncated=self.truncated,
            status=str(getattr(result, "status", "completed")),
            duration_s=time.perf_counter() - start,
        )

    def result(self, *, status: str = "completed") -> EngineerEpisodeResult:
        if self.episode is None:
            raise RuntimeError("reset must be called before result")
        reward = score_engineer_trajectory(
            self.episode,
            tool_records=self.tool_records,
            final_answer=self.final_answer,
            step_count=self.step_index,
            terminated=self.terminated,
            truncated=self.truncated,
        )
        return EngineerEpisodeResult(
            episode=self.episode,
            actions=self.actions,
            tool_records=self.tool_records,
            final_answer=self.final_answer,
            reward=reward,
            observations=self.observations,
            terminated=self.terminated,
            truncated=self.truncated,
            status=status,
        )

    def _execute_tool_action(self, action: EngineerAction) -> Dict[str, Any]:
        assert self.episode is not None
        name = str(action.name or "")
        args = dict(action.arguments or {})
        if name not in VALID_ENGINEER_TOOLS:
            result = {
                "ok": False,
                "tool": name,
                "error_type": "tool_not_in_engineer_gym",
                "message": "This episode exposes only real case inspection and live remote M1/M2 tools.",
            }
            return _tool_record(name=name, arguments=args, ok=False, result=result, error=result["message"])
        try:
            policy = validate_tool_call_policy(
                tool_name=name,
                arguments=args,
                user_message=self.episode.user_message,
                context=self._agent_context(),
            )
            if not policy.allowed:
                result = policy.to_tool_result()
                return _tool_record(name=name, arguments=args, ok=False, result=result, error=result.get("message"))
        except Exception as exc:
            result = {"ok": False, "tool": name, "error_type": type(exc).__name__, "message": str(exc)}
            return _tool_record(name=name, arguments=args, ok=False, result=result, error=str(exc))
        try:
            result = self.tool_runner(name, args) if self.tool_runner else self.registry.call_tool(name, args)
            return _tool_record(
                name=name,
                arguments=args,
                ok=bool(result.get("ok", False)),
                result=result,
                error=None if result.get("ok", False) else str(result.get("message") or result.get("error") or ""),
            )
        except (ToolRegistryError, ValueError) as exc:
            result = {"ok": False, "tool": name, "error_type": type(exc).__name__, "message": str(exc)}
            return _tool_record(name=name, arguments=args, ok=False, result=result, error=str(exc))

    def _observation(self, *, message: Optional[str] = None) -> EngineerObservation:
        assert self.episode is not None
        return EngineerObservation(
            episode_id=self.episode.episode_id,
            step_index=self.step_index,
            user_message=self.episode.user_message,
            context=self._agent_context(),
            tool_observations=tuple(self.tool_observations),
            terminated=self.terminated,
            truncated=self.truncated,
            message=message,
        )

    def _agent_context(self) -> Dict[str, Any]:
        assert self.episode is not None
        context = dict(self.episode.visible_context)
        for key in ("hidden_oracle", "reward", "oracle_label", "expected_tool", "answer_key"):
            context.pop(key, None)
        context["real_m1m2_engineer_gym"] = True
        context["remote_psse_m1m2_gym"] = True
        context["allowed_engineer_tools"] = sorted(VALID_ENGINEER_TOOLS)
        return context


class HybridRemoteM1M2CacheRunner:
    """Tool runner that caches live remote jobs and can fall back to processed artifacts."""

    def __init__(
        self,
        *,
        registry: Optional[ToolRegistry] = None,
        cache_dir: Optional[str | Path] = None,
        live_remote: bool = False,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.live_remote = bool(live_remote)

    def __call__(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        if name != REMOTE_M1M2_TOOL:
            return self.registry.call_tool(name, arguments)
        key = _cache_key(arguments)
        if self.cache_dir:
            cached = self.cache_dir / f"{key}.json"
            if cached.exists():
                payload = json.loads(cached.read_text(encoding="utf-8"))
                payload["cache_hit"] = True
                return payload
        result = self.registry.call_tool(name, arguments) if self.live_remote else _processed_remote_fallback(arguments)
        if self.cache_dir and result.get("ok"):
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result["cache_hit"] = False
        return result


def generate_real_m1m2_engineer_episodes(
    count: int,
    *,
    seed: int = DEFAULT_REAL_M1M2_ENGINEER_GYM_SEED,
    profile: str = "trgc_engineer",
    max_steps: int = 8,
) -> List[RealM1M2EngineerEpisode]:
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if profile not in REAL_M1M2_ENGINEER_PROFILES:
        raise ValueError("profile must be one of: " + ", ".join(REAL_M1M2_ENGINEER_PROFILES))
    rng = random.Random(seed)
    levels = _curriculum_sequence(count)
    used_ids: set[str] = set()
    episodes = []
    for index, level in enumerate(levels):
        episodes.append(_build_engineer_episode(level, index=index, seed=seed, rng=rng, used_ids=used_ids, max_steps=max_steps))
    return episodes


def write_real_m1m2_engineer_episodes(
    episodes: Sequence[RealM1M2EngineerEpisode],
    output: str | Path,
    *,
    generation: Optional[Mapping[str, Any]] = None,
    jsonl: bool = False,
) -> Dict[str, Any]:
    if not episodes:
        raise ValueError("episodes must not be empty")
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with output_path.open("w", encoding="utf-8") as handle:
            for episode in episodes:
                handle.write(json.dumps(episode.to_dict(), sort_keys=True) + "\n")
    else:
        output_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "schema_version": REAL_M1M2_ENGINEER_GYM_SCHEMA_VERSION,
                    "episode_source": "generated_trgc_real_m1m2_engineer_gym",
                    "generation": dict(generation or {}),
                    "episode_count": len(episodes),
                    "episodes": [episode.to_dict() for episode in episodes],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "ok": True,
        "schema_version": REAL_M1M2_ENGINEER_GYM_SCHEMA_VERSION,
        "episode_count": len(episodes),
        "output": str(output_path),
        "format": "jsonl" if jsonl else "json",
    }


def load_real_m1m2_engineer_episodes(path: str | Path) -> List[RealM1M2EngineerEpisode]:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(str(source))
    if source.suffix.lower() == ".jsonl":
        payload: Any = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
    return real_m1m2_engineer_episodes_from_payload(payload)


def real_m1m2_engineer_episodes_from_payload(payload: Any) -> List[RealM1M2EngineerEpisode]:
    if isinstance(payload, Mapping):
        payload = payload.get("episodes")
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
        raise ValueError("engineer episode payload must be a list or object with episodes")
    episodes = [RealM1M2EngineerEpisode.from_mapping(item) for item in payload if isinstance(item, Mapping)]
    if not episodes:
        raise ValueError("engineer episode payload must contain at least one episode")
    ids = [item.episode_id for item in episodes]
    duplicates = sorted(item for item in set(ids) if ids.count(item) > 1)
    if duplicates:
        raise ValueError("engineer episode payload contains duplicate ids: " + ", ".join(duplicates))
    return episodes


def filter_real_m1m2_engineer_episodes(
    episodes: Sequence[RealM1M2EngineerEpisode],
    *,
    episode_ids: Sequence[str] = (),
    curriculum_levels: Sequence[str] = (),
    families: Sequence[str] = (),
    difficulties: Sequence[str] = (),
    limit: Optional[int] = None,
) -> List[RealM1M2EngineerEpisode]:
    wanted_ids = {item for item in episode_ids if item}
    wanted_levels = {item for item in curriculum_levels if item}
    wanted_families = {item for item in families if item}
    wanted_difficulties = {item.lower() for item in difficulties if item}
    selected = []
    for episode in episodes:
        if wanted_ids and episode.episode_id not in wanted_ids:
            continue
        if wanted_levels and episode.curriculum_level not in wanted_levels:
            continue
        if wanted_families and episode.family not in wanted_families:
            continue
        if wanted_difficulties and episode.difficulty.lower() not in wanted_difficulties:
            continue
        selected.append(episode)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive when provided")
        selected = selected[:limit]
    return selected


def replay_real_m1m2_engineer_trajectory(
    episode: RealM1M2EngineerEpisode,
    actions: Sequence[Mapping[str, Any]],
    *,
    tool_runner: Optional[ToolRunner] = None,
    registry: Optional[ToolRegistry] = None,
) -> EngineerEpisodeResult:
    env = RealM1M2EngineerEnv(registry=registry, tool_runner=tool_runner)
    env.reset(episode)
    for action in actions:
        env.step(action)
        if env.terminated or env.truncated:
            break
    return env.result()


def evaluate_real_m1m2_engineer_agent(
    agent: Any,
    episodes: Sequence[RealM1M2EngineerEpisode],
) -> Dict[str, Any]:
    results = []
    start = time.perf_counter()
    for episode in episodes:
        env = RealM1M2EngineerEnv()
        env.reset(episode)
        results.append(env.run_agent(agent))
    return engineer_results_summary(results, duration_s=time.perf_counter() - start)


def engineer_results_summary(
    results: Sequence[EngineerEpisodeResult],
    *,
    duration_s: float = 0.0,
    include_hidden: bool = False,
) -> Dict[str, Any]:
    return {
        "ok": all(result.passed for result in results),
        "total": len(results),
        "passed": sum(1 for result in results if result.passed),
        "failed": sum(1 for result in results if not result.passed),
        "duration_s": duration_s,
        "average_reward": sum(result.reward.total for result in results) / len(results) if results else 0.0,
        "by_curriculum_level": _counts_by_result(results, "curriculum_level"),
        "by_family": _counts_by_result(results, "family"),
        "by_difficulty": _counts_by_result(results, "difficulty"),
        "reward_components": _average_components(results),
        "forbidden_action_count": sum(1 for result in results if result.reward.hard_penalties),
        "results": [result.to_dict(include_hidden=include_hidden) for result in results],
    }


def score_engineer_trajectory(
    episode: RealM1M2EngineerEpisode,
    *,
    tool_records: Sequence[Mapping[str, Any]],
    final_answer: str,
    step_count: int,
    terminated: bool,
    truncated: bool,
) -> EngineerRewardBreakdown:
    oracle = dict(episode.hidden_oracle)
    weights = dict(DEFAULT_ENGINEER_REWARD_WEIGHTS)
    weights.update(_mapping_value(oracle, "reward_weights"))
    penalty_values = dict(DEFAULT_ENGINEER_HARD_PENALTIES)
    penalty_values.update(_mapping_value(oracle, "hard_penalties"))
    text = (final_answer or "").lower()
    inspection = _required_tool_score(oracle.get("required_inspection_tools", ()), tool_records)
    evidence = _required_tool_score(oracle.get("required_evidence_tools", ()), tool_records)
    trgc = _claim_group_score(oracle.get("classification_claim_groups", ()), text)
    grounded = _grounded_memo_score(episode, tool_records, text)
    no_proxy = 0.0 if _has_forbidden_proxy(episode, tool_records) else 1.0
    efficiency = 0.0 if truncated else max(0.0, (episode.max_steps - max(0, step_count - 1)) / episode.max_steps)
    penalties = _hard_penalties(episode, tool_records=tool_records, text=text, values=penalty_values)
    base = (
        weights["case_poc_model_inspection"] * inspection
        + weights["evidence_selection"] * evidence
        + weights["trgc_mapping"] * trgc
        + weights["grounded_memo"] * grounded
        + weights["no_proxy"] * no_proxy
        + weights["efficiency"] * efficiency
    )
    total = _clamp(base + sum(penalties.values()), 0.0, 1.0)
    if not terminated and not truncated:
        total = min(total, 0.75)
    return EngineerRewardBreakdown(
        case_poc_model_inspection=round(inspection, 6),
        evidence_selection=round(evidence, 6),
        trgc_mapping=round(trgc, 6),
        grounded_memo=round(grounded, 6),
        no_proxy=round(no_proxy, 6),
        efficiency=round(efficiency, 6),
        hard_penalties=penalties,
        total=round(total, 6),
    )


def _curriculum_sequence(count: int) -> List[str]:
    default_counts = {
        "level1_scope_data_readiness": 20,
        "level2_case_inspection": 20,
        "level3_supported_m1m2": 20,
        "level4_mixed_trgc": 25,
        "level5_engineer_memo": 15,
    }
    if count == 100:
        return [level for level, qty in default_counts.items() for _ in range(qty)]
    return [ENGINEER_CURRICULUM_LEVELS[index % len(ENGINEER_CURRICULUM_LEVELS)] for index in range(count)]


def _build_engineer_episode(
    level: str,
    *,
    index: int,
    seed: int,
    rng: random.Random,
    used_ids: set[str],
    max_steps: int,
) -> RealM1M2EngineerEpisode:
    difficulty = ENGINEER_DIFFICULTIES[index % len(ENGINEER_DIFFICULTIES)]
    executable = [item for item in TRGC_REQUIREMENT_CATALOG if item.current_support_status == "executable_current_remote"]
    unsupported = [item for item in TRGC_REQUIREMENT_CATALOG if item.current_support_status == "unsupported_current_remote"]
    classification = [item for item in TRGC_REQUIREMENT_CATALOG if item.current_support_status == "classification_only"]
    req_exec = rng.choice(executable)
    req_bad = rng.choice(unsupported)
    req_class = rng.choice(classification)
    case_id = "pif6_2026_05_17"
    scenario_type = req_exec.current_remote_scenario_type or "no_disturbance_5s"
    if scenario_type == "pq_target_step":
        case_id = "test_cases_v36"
    family = level.replace("level", "engineer_level")
    episode_id = _episode_id(level, seed=seed, index=index, used_ids=used_ids)
    visible = {
        "real_m1m2_engineer_gym": True,
        "case_id": case_id,
        "project_package": "processed_real_psse_case_package",
        "trgc_context": _safe_requirement(req_exec if level in {"level3_supported_m1m2", "level4_mixed_trgc", "level5_engineer_memo"} else req_bad),
        "trgc_requirement": _safe_requirement(req_exec if level in {"level3_supported_m1m2", "level4_mixed_trgc", "level5_engineer_memo"} else req_bad),
        "workflow_expectation": "inspect_case_then_collect_valid_evidence_then_write_bounded_engineering_memo",
    }
    if level == "level1_scope_data_readiness":
        visible["case_id"] = "pif6_2026_05_17"
        user = (
            f"We received a TRGC request for {req_bad.requirement_id} ({req_bad.title}) on the PIF6 package, "
            "but the submittal does not provide a confirmed POC, project MW, Q capability, or validated study scenario. "
            "Use the engineer gym to decide what can be supported."
        )
        oracle = _oracle(
            level=level,
            requirements=(req_bad,),
            required_inspection_tools=({"name": "inspect_real_case_summary", "arguments": {"case_id": "pif6_2026_05_17"}}, {"name": "list_remote_psse_m1m2_cases", "arguments": {}}),
            forbidden_tools=(REMOTE_M1M2_TOOL,),
            missing_fields=("poc", "project_mw", "q_capability", "validated_study_scenario"),
            claim_groups=(("trgc",), MISSING_TERMS, REFUSAL_TERMS),
        )
    elif level == "level2_case_inspection":
        visible["case_id"] = "pif6_2026_05_17"
        user = (
            "For PIF6, identify the likely POC context before making any recommendation. "
            "The file names mention POC2, but the package also contains POC2_0 and DUMMY buses. "
            "Inspect the case, POC candidates, and model inventory."
        )
        oracle = _oracle(
            level=level,
            requirements=(req_class,),
            required_inspection_tools=(
                {"name": "inspect_real_case_summary", "arguments": {"case_id": "pif6_2026_05_17"}},
                {"name": "inspect_real_poc_context", "arguments": {"case_id": "pif6_2026_05_17"}},
                {"name": "inspect_real_model_inventory", "arguments": {"case_id": "pif6_2026_05_17"}},
            ),
            correct_poc_bus=2,
            claim_groups=(("poc",), ("candidate", "ambiguous", "POC2".lower()), ("bus 2", "poc2")),
        )
    elif level == "level3_supported_m1m2":
        visible["scenario_type"] = scenario_type
        user = (
            f"Run the executable TRGC subset {req_exec.requirement_id} for {case_id}/{scenario_type}. "
            "First inspect the real case package and POC context, then collect live M1/M2 evidence and bound the conclusion."
        )
        oracle = _oracle(
            level=level,
            requirements=(req_exec,),
            allowed_remote_jobs=({"case_id": case_id, "scenario_type": scenario_type},),
            required_inspection_tools=(
                {"name": "inspect_real_case_summary", "arguments": {"case_id": case_id}},
                {"name": "inspect_real_poc_context", "arguments": {"case_id": case_id}},
            ),
            required_evidence_tools=({"name": REMOTE_M1M2_TOOL, "arguments": {"case_id": case_id, "scenario_type": scenario_type}},),
            claim_groups=(("pss/e", "psse"), ("m1",), ("approve", "pass"), BOUNDARY_TERMS),
        )
    elif level == "level4_mixed_trgc":
        visible["scenario_type"] = scenario_type
        user = (
            f"For this PIF6 review, gather evidence for executable {req_exec.requirement_id} using {case_id}/{scenario_type}, "
            f"and also tell me whether that same evidence validates {req_bad.requirement_id} ({req_bad.title})."
        )
        visible["secondary_trgc_context"] = _safe_requirement(req_bad)
        oracle = _oracle(
            level=level,
            requirements=(req_exec, req_bad),
            allowed_remote_jobs=({"case_id": case_id, "scenario_type": scenario_type},),
            required_inspection_tools=(
                {"name": "inspect_real_case_summary", "arguments": {"case_id": case_id}},
                {"name": "inspect_real_poc_context", "arguments": {"case_id": case_id}},
            ),
            required_evidence_tools=({"name": REMOTE_M1M2_TOOL, "arguments": {"case_id": case_id, "scenario_type": scenario_type}},),
            claim_groups=(SUPPORTED_RESULT_TERMS, REFUSAL_TERMS, (req_bad.requirement_id.lower(),), BOUNDARY_TERMS),
        )
    elif level == "level5_engineer_memo":
        visible["case_id"] = "pif6_2026_05_17"
        user = (
            "Write an engineer-style TRGC evidence memo for the PIF6 package. Inspect topology/POC, model inventory, "
            "static operating point, and no-disturbance dynamic channels. State what the current evidence supports and "
            "what remains outside scope."
        )
        oracle = _oracle(
            level=level,
            requirements=(req_exec, req_bad, req_class),
            required_inspection_tools=(
                {"name": "inspect_real_case_summary", "arguments": {"case_id": "pif6_2026_05_17"}},
                {"name": "inspect_real_poc_context", "arguments": {"case_id": "pif6_2026_05_17"}},
                {"name": "inspect_real_model_inventory", "arguments": {"case_id": "pif6_2026_05_17"}},
                {"name": "inspect_real_static_operating_point", "arguments": {"case_id": "pif6_2026_05_17"}},
                {"name": "inspect_real_dynamic_channels", "arguments": {"case_id": "pif6_2026_05_17"}},
            ),
            correct_poc_bus=2,
            claim_groups=(("poc", "bus 2", "poc2"), ("voltage", "p/q", "mw", "mvar"), ("dynamic", "channel", "no-disturbance"), REFUSAL_TERMS, BOUNDARY_TERMS),
        )
    else:
        raise ValueError(f"unknown curriculum level: {level}")
    return RealM1M2EngineerEpisode(
        episode_id=episode_id,
        user_message=user,
        curriculum_level=level,
        family=family,
        difficulty=difficulty,
        visible_context=visible,
        hidden_oracle=oracle,
        max_steps=max_steps,
    )


def _oracle(
    *,
    level: str,
    requirements: Sequence[TRGCRequirement],
    required_inspection_tools: Sequence[Mapping[str, Any]] = (),
    required_evidence_tools: Sequence[Mapping[str, Any]] = (),
    allowed_remote_jobs: Sequence[Mapping[str, Any]] = (),
    forbidden_tools: Sequence[str] = (),
    claim_groups: Sequence[Sequence[str]] = (),
    missing_fields: Sequence[str] = (),
    correct_poc_bus: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "curriculum_level": level,
        "requirement_ids": [item.requirement_id for item in requirements],
        "layers": [item.layer for item in requirements],
        "support_statuses": [item.current_support_status for item in requirements],
        "required_inspection_tools": [dict(item) for item in required_inspection_tools],
        "required_evidence_tools": [dict(item) for item in required_evidence_tools],
        "allowed_remote_jobs": [dict(item) for item in allowed_remote_jobs],
        "forbidden_tools": list(forbidden_tools),
        "missing_fields": list(missing_fields),
        "correct_poc_bus": correct_poc_bus,
        "classification_claim_groups": [list(group) for group in claim_groups],
        "required_final_claim_groups": [list(group) for group in claim_groups],
        "reward_weights": dict(DEFAULT_ENGINEER_REWARD_WEIGHTS),
        "hard_penalties": dict(DEFAULT_ENGINEER_HARD_PENALTIES),
    }


def _required_tool_score(required: Any, records: Sequence[Mapping[str, Any]]) -> float:
    items = [item for item in required or () if isinstance(item, Mapping)]
    if not items:
        return 1.0
    matched = 0
    for expected in items:
        if any(_record_matches(expected, record) for record in records):
            matched += 1
    return matched / len(items)


def _record_matches(expected: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    if not record.get("ok"):
        return False
    if record.get("name") != expected.get("name"):
        return False
    expected_args = expected.get("arguments") if isinstance(expected.get("arguments"), Mapping) else {}
    actual_args = record.get("arguments") if isinstance(record.get("arguments"), Mapping) else {}
    for key, value in expected_args.items():
        if actual_args.get(key) != value:
            return False
    return True


def _claim_group_score(groups: Any, text: str) -> float:
    claim_groups = [group for group in groups or () if isinstance(group, Sequence) and not isinstance(group, (str, bytes))]
    if not claim_groups:
        return 1.0
    return sum(1 for group in claim_groups if any(str(term).lower() in text for term in group)) / len(claim_groups)


def _grounded_memo_score(
    episode: RealM1M2EngineerEpisode,
    records: Sequence[Mapping[str, Any]],
    text: str,
) -> float:
    if not text.strip():
        return 0.0
    if any(claim in text for claim in FALSE_CAPABILITY_CLAIMS):
        return 0.0
    if _claims_psse_evidence(text) and not _has_psse_or_processed_evidence(records):
        return 0.0
    return _claim_group_score(episode.hidden_oracle.get("required_final_claim_groups", ()), text)


def _hard_penalties(
    episode: RealM1M2EngineerEpisode,
    *,
    tool_records: Sequence[Mapping[str, Any]],
    text: str,
    values: Mapping[str, Any],
) -> Dict[str, float]:
    penalties: Dict[str, float] = {}
    if _approves_unsupported(episode, text):
        penalties["unsupported_approval"] = float(values.get("unsupported_approval", -1.0))
    if _has_forbidden_proxy(episode, tool_records):
        penalties["proxy_baseline"] = float(values.get("proxy_baseline", -0.7))
    if _claims_psse_evidence(text) and not _has_psse_or_processed_evidence(tool_records):
        penalties["ungrounded_psse_claim"] = float(values.get("ungrounded_psse_claim", -0.5))
    if any(claim in text for claim in FALSE_CAPABILITY_CLAIMS):
        penalties["false_capability_claim"] = float(values.get("false_capability_claim", -0.3))
    if _wrong_poc_claim(episode, text, tool_records):
        penalties["wrong_poc_claim"] = float(values.get("wrong_poc_claim", -0.3))
    return penalties


def _approves_unsupported(episode: RealM1M2EngineerEpisode, text: str) -> bool:
    statuses = set(episode.hidden_oracle.get("support_statuses", ()))
    if "unsupported_current_remote" not in statuses and "classification_only" not in statuses:
        return False
    approval = any(term in text for term in ("approved", "validated", "satisfied", "passes"))
    refusal = any(term in text for term in ("cannot", "not ", "unsupported", "outside", "insufficient"))
    return approval and not refusal


def _claims_psse_evidence(text: str) -> bool:
    return any(term in text for term in ("pss/e", "psse", "load flow", "m1", "m2", "poc", "voltage", "mw", "mvar", "channel"))


def _has_psse_or_processed_evidence(records: Sequence[Mapping[str, Any]]) -> bool:
    return any(record.get("ok") and (record.get("name") == REMOTE_M1M2_TOOL or record.get("name") in INSPECTION_TOOLS) for record in records)


def _has_forbidden_proxy(
    episode: RealM1M2EngineerEpisode,
    records: Sequence[Mapping[str, Any]],
) -> bool:
    return any(_is_forbidden_proxy_record(episode, record) for record in records)


def _is_forbidden_proxy_record(episode: RealM1M2EngineerEpisode, record: Mapping[str, Any]) -> bool:
    name = str(record.get("name") or "")
    if name in set(episode.hidden_oracle.get("forbidden_tools", ())):
        return True
    if name != REMOTE_M1M2_TOOL:
        return False
    args = record.get("arguments") if isinstance(record.get("arguments"), Mapping) else {}
    allowed = [item for item in episode.hidden_oracle.get("allowed_remote_jobs", ()) if isinstance(item, Mapping)]
    if not allowed:
        return True
    return not any(_args_match(item, args) for item in allowed)


def _wrong_poc_claim(
    episode: RealM1M2EngineerEpisode,
    text: str,
    records: Sequence[Mapping[str, Any]],
) -> bool:
    correct = episode.hidden_oracle.get("correct_poc_bus")
    if correct is None:
        return False
    if not any(record.get("ok") and record.get("name") == "inspect_real_poc_context" for record in records):
        return False
    if int(correct) == 2:
        wrong_patterns = ("poc bus 2000", "poc is bus 2000", "poc2_0 is the poc", "poc2_0 as the poc")
        return any(pattern in text for pattern in wrong_patterns)
    return False


def _args_match(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _compact_observation(record: Mapping[str, Any]) -> Dict[str, Any]:
    result = record.get("result") if isinstance(record.get("result"), Mapping) else {}
    if result:
        try:
            return build_tool_observation(result)
        except Exception:
            pass
    return {
        "tool": record.get("name"),
        "ok": bool(record.get("ok")),
        "error": record.get("error"),
    }


def _record_from_agent_tool(record: Any) -> Dict[str, Any]:
    return {
        "name": getattr(record, "name", None),
        "arguments": dict(getattr(record, "arguments", {}) or {}),
        "ok": bool(getattr(record, "ok", False)),
        "result": dict(getattr(record, "result", {}) or {}),
        "error": getattr(record, "error", None),
        "source": getattr(record, "source", None),
    }


def _tool_record(
    *,
    name: str,
    arguments: Mapping[str, Any],
    ok: bool,
    result: Mapping[str, Any],
    error: Optional[str],
) -> Dict[str, Any]:
    return {
        "name": name,
        "arguments": dict(arguments),
        "ok": ok,
        "result": dict(result),
        "error": error,
        "source": "engineer_env",
    }


def _counts_by_result(results: Sequence[EngineerEpisodeResult], attribute: str) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for result in results:
        key = str(getattr(result.episode, attribute))
        bucket = counts.setdefault(key, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if result.passed:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
    return counts


def _average_components(results: Sequence[EngineerEpisodeResult]) -> Dict[str, float]:
    if not results:
        return {}
    keys = (
        "case_poc_model_inspection",
        "evidence_selection",
        "trgc_mapping",
        "grounded_memo",
        "no_proxy",
        "efficiency",
        "total",
    )
    return {
        key: round(sum(getattr(result.reward, key) for result in results) / len(results), 6)
        for key in keys
    }


def _processed_remote_fallback(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    case_id = str(arguments.get("case_id") or "")
    scenario = str(arguments.get("scenario_type") or "")
    if case_id == "pif6_2026_05_17":
        bus_count = 786
        branch_count = 790
        machine_count = 251
        poc_p = 5.0866778480386206
        poc_q = -19.33467761090973
        min_v = 0.8999999761581421
        max_v = 1.0425307750701904
        final_p = 5.131742000579834
        final_q = -19.284170150756836
        row_count = 5004
    else:
        bus_count = 11
        branch_count = 10
        machine_count = 5
        poc_p = 200.0081024169922
        poc_q = 330.0627136230469
        min_v = 1.0
        max_v = 1.0616588592529297
        final_p = 200.00880432128906
        final_q = 330.0626525878906
        row_count = 5004
    dynamic = scenario in {"no_disturbance_5s", "no_disturbance", "baseline"}
    m2_status = "pass" if dynamic else "skipped"
    return {
        "ok": True,
        "tool": REMOTE_M1M2_TOOL,
        "schema_version": REMOTE_M1M2_SCHEMA_VERSION,
        "backend": "processed_cache_remote_psse_fallback",
        "case_id": case_id,
        "scenario_type": "no_disturbance_5s" if dynamic else scenario,
        "recommendation": "approve",
        "complete": True,
        "summary": {
            "m1_status": "pass",
            "m1_converged": True,
            "m1_bus_voltage_min_pu": min_v,
            "m1_bus_voltage_max_pu": max_v,
            "m1_poc_p_mw": poc_p,
            "m1_poc_q_mvar": poc_q,
            "m1_bus_count": bus_count,
            "m1_branch_count": branch_count,
            "m1_machine_count": machine_count,
            "m2_status": m2_status,
            "m2_initialized": dynamic,
            "m2_simulation_converged": dynamic,
            "m2_final_poc_p_mw": final_p if dynamic else None,
            "m2_final_poc_q_mvar": final_q if dynamic else None,
            "m2_channel_row_count": row_count if dynamic else None,
        },
        "limitations": ["processed_cache_fallback_not_live_remote_execution"],
    }


def _cache_key(arguments: Mapping[str, Any]) -> str:
    return hashlib.sha1(
        json.dumps(
            {"schema": REAL_M1M2_ENGINEER_GYM_SCHEMA_VERSION, "arguments": dict(arguments)},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _engineer_tool_specs() -> List[Dict[str, Any]]:
    names = VALID_ENGINEER_TOOLS
    return [
        spec
        for spec in ToolRegistry().openai_tool_specs()
        if spec.get("function", {}).get("name") in names
    ]


def _safe_requirement(req: TRGCRequirement) -> Dict[str, Any]:
    return req.to_dict()


def _episode_id(level: str, *, seed: int, index: int, used_ids: set[str]) -> str:
    digest = hashlib.sha1(
        json.dumps({"level": level, "seed": seed, "index": index}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    base = f"real_m1m2_engineer_{index:04d}_{level}_{digest}"
    episode_id = base
    suffix = 1
    while episode_id in used_ids:
        suffix += 1
        episode_id = f"{base}_{suffix}"
    used_ids.add(episode_id)
    return episode_id


def _mapping_value(payload: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return dict(value)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)

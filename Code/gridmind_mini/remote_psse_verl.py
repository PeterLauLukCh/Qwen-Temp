"""VERL tool wrapper for the live remote PSS/E M1+M2 gym."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Callable, Dict, Mapping, Optional

from .observations import build_tool_observation
from .remote_psse import REMOTE_M1M2_TOOL, run_remote_psse_m1m2
from .tools import ToolRegistry


try:  # pragma: no cover - exercised on GPU/verl nodes.
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

    VERL_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for laptop/unit-test imports.
    VERL_AVAILABLE = False

    class BaseTool:  # type: ignore[no-redef]
        """Fallback shim used only when verl is not importable."""

    class ToolResponse:  # type: ignore[no-redef]
        """Small fallback compatible with the subset used in tests."""

        def __init__(self, text: Optional[str] = None, image: Any = None, video: Any = None) -> None:
            self.text = text
            self.image = image
            self.video = video

    OpenAIFunctionToolSchema = Any  # type: ignore[misc, assignment]


REMOTE_PSSE_DATA_SOURCE = "powergrid_real_psse_m1m2_remote"
REMOTE_PSSE_ABILITY = "real_psse_m1_m2_interconnection_gym"
REMOTE_PSSE_OBSERVATION_SCHEMA_VERSION = "remote_psse_m1m2_tool_observation_v1"


Runner = Callable[..., Dict[str, Any]]


class RemotePsseM1M2ToolCore:
    """Dependency-free core for multi-turn remote PSS/E tool execution."""

    def __init__(
        self,
        *,
        runner: Runner = run_remote_psse_m1m2,
        max_observation_chars: int = 6000,
    ) -> None:
        if max_observation_chars <= 0:
            raise ValueError("max_observation_chars must be positive")
        self.runner = runner
        self.max_observation_chars = int(max_observation_chars)
        self._instances: Dict[str, Dict[str, Any]] = {}

    def create(self, instance_id: Optional[str] = None) -> str:
        instance = instance_id or f"remote_psse_{uuid.uuid4().hex[:12]}"
        self._instances[instance] = {
            "state_id": "initial",
            "step_index": 0,
            "history": [],
            "requests": {},
        }
        return instance

    def execute(self, instance_id: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        if instance_id not in self._instances:
            return _state_error(
                instance_id=instance_id,
                error_type="missing_tool_instance",
                message="Remote PSS/E tool instance was not initialized.",
            )
        if not isinstance(arguments, Mapping):
            return _state_error(
                instance_id=instance_id,
                error_type="invalid_arguments",
                message="Tool arguments must be a JSON object.",
            )
        state = self._instances[instance_id]
        request_id = arguments.get("request_id")
        if request_id and str(request_id) in state["requests"]:
            duplicate = dict(state["requests"][str(request_id)])
            duplicate["duplicate_request"] = True
            return duplicate
        expected_state_id = arguments.get("expected_state_id")
        if expected_state_id and str(expected_state_id) != state["state_id"]:
            return _state_error(
                instance_id=instance_id,
                error_type="stale_state",
                message="expected_state_id does not match the current tool state.",
                state=state,
            )
        requested_step = arguments.get("step_index")
        if requested_step is not None and requested_step != state["step_index"]:
            return _state_error(
                instance_id=instance_id,
                error_type="stale_step_index",
                message="step_index does not match the current tool step.",
                state=state,
            )

        previous_state_id = str(state["state_id"])
        step_index = int(state["step_index"])
        result = self.runner(
            case_id=str(arguments.get("case_id", "")),
            scenario_type=str(arguments.get("scenario_type", "")),
            request_id=str(request_id) if request_id else None,
            include_artifacts=bool(arguments.get("include_artifacts", False)),
        )
        observation = build_remote_psse_tool_observation(
            result,
            instance_id=instance_id,
            step_index=step_index,
            previous_state_id=previous_state_id,
            max_chars=self.max_observation_chars,
        )
        state["state_id"] = observation["state_id"]
        state["step_index"] = step_index + 1
        state["history"].append(observation)
        if request_id:
            state["requests"][str(request_id)] = observation
        return observation

    def reset(self, instance_id: str) -> Dict[str, Any]:
        if instance_id not in self._instances:
            return _state_error(
                instance_id=instance_id,
                error_type="missing_tool_instance",
                message="Remote PSS/E tool instance was not initialized.",
            )
        self._instances[instance_id] = {
            "state_id": "initial",
            "step_index": 0,
            "history": [],
            "requests": {},
        }
        return {
            "ok": True,
            "tool": REMOTE_M1M2_TOOL,
            "schema_version": REMOTE_PSSE_OBSERVATION_SCHEMA_VERSION,
            "instance_id": instance_id,
            "state_id": "initial",
            "step_index": 0,
            "message": "Remote PSS/E tool state reset on the GPU side.",
        }

    def release(self, instance_id: str) -> None:
        self._instances.pop(instance_id, None)


def build_remote_psse_tool_observation(
    result: Mapping[str, Any],
    *,
    instance_id: str,
    step_index: int,
    previous_state_id: str,
    max_chars: int,
) -> Dict[str, Any]:
    """Build the compact observation returned to VERL after one remote job."""

    compact = build_tool_observation(result)
    summary = result.get("summary") if isinstance(result.get("summary"), Mapping) else {}
    state_id = _state_id(previous_state_id, result)
    observation = {
        "ok": bool(result.get("ok")),
        "tool": REMOTE_M1M2_TOOL,
        "schema_version": REMOTE_PSSE_OBSERVATION_SCHEMA_VERSION,
        "instance_id": instance_id,
        "step_index": step_index,
        "previous_state_id": previous_state_id,
        "state_id": state_id,
        "job_id": result.get("job_id"),
        "case_id": result.get("case_id"),
        "scenario_type": result.get("scenario_type"),
        "simulation_time_s": summary.get("m2_final_time_s"),
        "observation": compact,
        "reward_metrics": {
            "remote_job_ok": bool(result.get("ok")),
            "m1_passed": summary.get("m1_status") == "pass",
            "m2_passed": summary.get("m2_status") == "pass",
            "complete": bool(result.get("complete")),
            "recommendation": result.get("recommendation"),
        },
        "terminated": False,
        "truncated": False,
    }
    return _truncate(observation, max_chars=max_chars)


class RemotePsseM1M2Tool(BaseTool):  # type: ignore[misc]
    """VERL native tool for live remote PSS/E M1+M2 rollouts."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None) -> None:  # type: ignore[assignment]
        if not VERL_AVAILABLE:
            raise ImportError(
                "RemotePsseM1M2Tool requires verl. Add verl-main to PYTHONPATH "
                "or install verl in the training environment."
            )
        self._core = RemotePsseM1M2ToolCore(
            max_observation_chars=int((config or {}).get("max_observation_chars", 6000))
        )
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate(_remote_psse_openai_spec())

    async def create(self, instance_id: Optional[str] = None, **_kwargs: Any) -> tuple[str, ToolResponse]:
        instance = self._core.create(instance_id)
        return instance, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **_kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        observation = self._core.execute(instance_id, parameters)
        reward = 1.0 if observation.get("ok") else 0.0
        metrics = {
            "remote_job_ok": bool(observation.get("reward_metrics", {}).get("remote_job_ok"))
            if isinstance(observation.get("reward_metrics"), Mapping)
            else bool(observation.get("ok")),
            "m1_passed": bool(observation.get("reward_metrics", {}).get("m1_passed"))
            if isinstance(observation.get("reward_metrics"), Mapping)
            else False,
            "m2_passed": bool(observation.get("reward_metrics", {}).get("m2_passed"))
            if isinstance(observation.get("reward_metrics"), Mapping)
            else False,
        }
        return ToolResponse(text=_json_dumps(observation)), reward, metrics

    async def release(self, instance_id: str, **_kwargs: Any) -> None:
        self._core.release(instance_id)


def _remote_psse_openai_spec() -> Dict[str, Any]:
    for spec in ToolRegistry().openai_tool_specs():
        function = spec.get("function") if isinstance(spec, Mapping) else None
        if isinstance(function, Mapping) and function.get("name") == REMOTE_M1M2_TOOL:
            return dict(spec)
    raise RuntimeError("run_remote_psse_m1m2 schema is not registered")


def _state_error(
    *,
    instance_id: str,
    error_type: str,
    message: str,
    state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "tool": REMOTE_M1M2_TOOL,
        "schema_version": REMOTE_PSSE_OBSERVATION_SCHEMA_VERSION,
        "instance_id": instance_id,
        "state_id": state.get("state_id") if isinstance(state, Mapping) else None,
        "step_index": state.get("step_index") if isinstance(state, Mapping) else None,
        "error_type": error_type,
        "message": message,
        "terminated": False,
        "truncated": False,
    }


def _state_id(previous_state_id: str, result: Mapping[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), Mapping) else {}
    payload = {
        "previous_state_id": previous_state_id,
        "job_id": result.get("job_id"),
        "case_id": result.get("case_id"),
        "scenario_type": result.get("scenario_type"),
        "ok": result.get("ok"),
        "recommendation": result.get("recommendation"),
        "summary": summary,
    }
    digest = hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()
    return digest[:24]


def _truncate(observation: Mapping[str, Any], *, max_chars: int) -> Dict[str, Any]:
    text = _json_dumps(observation)
    if len(text) <= max_chars:
        return dict(observation)
    compact = dict(observation)
    compact["truncated"] = True
    compact["observation"] = {
        "tool": REMOTE_M1M2_TOOL,
        "summary": observation.get("observation", {}).get("summary")
        if isinstance(observation.get("observation"), Mapping)
        else {},
    }
    if len(_json_dumps(compact)) <= max_chars:
        return compact
    compact["message"] = "Remote PSS/E observation was truncated; summary fields are preserved."
    return compact


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

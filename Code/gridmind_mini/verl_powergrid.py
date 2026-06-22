"""verl adapter utilities for PowerGym/Grid-Mind RL training.

This module keeps the training-time interface small and deterministic. During
rollout, the model calls ``run_integrated_assessment`` through verl's native
``BaseTool`` interface. For v1, the tool returns frozen oracle observations from
generated IEEE14 M1+M2+EMT scenarios instead of running live solvers.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .joint_benchmark import (
    JointBenchmarkScenario,
    generate_joint_benchmark_scenarios,
    joint_benchmark_scenarios_from_payload,
)
from .joint_benchmark import _argument_value_passed as _joint_argument_value_passed
from .joint_benchmark import _resolve_path as _joint_resolve_path
from .llm import remove_tool_call_blocks
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


DATA_SOURCE = "powergrid_ieee14_m1_m2_emt"
ABILITY = "power_system_interconnection"
EXPECTED_TOOL_NAME = "run_integrated_assessment"
DEFAULT_TRAIN_SEED = 20260610
DEFAULT_VAL_SEED = 20260611
DEFAULT_TRAIN_COUNT = 1000
DEFAULT_VAL_COUNT = 200
DEFAULT_PROFILE = "emt"
TOOL_OBSERVATION_SCHEMA_VERSION = "powergrid_frozen_oracle_v1"


@dataclass(frozen=True)
class ArgumentCheck:
    """One expected tool-argument check for a frozen scenario."""

    path: str
    passed: bool
    expected: Any
    actual: Any = None
    found: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual if self.found else None,
            "found": self.found,
        }


def build_powergrid_system_prompt() -> str:
    """Return the fixed system prompt used for verl agentic RL samples."""

    return "\n".join(
        [
            "You are a power-system interconnection study agent.",
            "For each user request, call run_integrated_assessment before giving the final answer.",
            "Use the user's requested case, project bus, MW/MVAr values, connection type, fault bus, fault timing, and EMT/SCR settings in the tool arguments.",
            "Use the exact JSON tool-call schema. Do not use aliases such as project_bus, mw, mvar, fault_bus, or fault_timing.",
            "The connection object must use keys bus, p_mw, optional q_mvar, connection_type, and is_ibr.",
            "The transient fault must be nested under transient.disturbance with keys type, bus, fault_start_s, and clearing_time_s.",
            "clearing_time_s is the absolute clearing timestamp: if a fault starts at 1.0 s and clears after 100 ms, use 1.1.",
            'Example tool call: <tool_call>{"name":"run_integrated_assessment","arguments":{"case_path":"ieee14","connection":{"bus":10,"p_mw":20.0,"connection_type":"solar","is_ibr":true},"transient":{"enabled":true,"required_for_approval":true,"case_path":"ieee14_dynamic","disturbance":{"type":"bus_fault","bus":2,"fault_start_s":1.0,"clearing_time_s":1.1},"simulation_time_s":5.0,"max_samples":20},"emt":{"enabled":true,"required_for_approval":false,"scr_threshold":3.0}}}</tool_call>',
            "After the tool returns, answer from the tool observation only.",
            "Your final answer must mention the recommendation, M1 steady-state/CIA result, M2 transient-stability result, EMT/SCR result when present, at least one grounded metric, and the static-PQ / EMT-v1 limitation.",
            "Do not claim detailed inverter controllers, protection, switching waveforms, LVRT/HVRT, PLL behavior, or customer-validated dynamics unless the tool explicitly reports them.",
        ]
    )


def scenario_to_verl_record(
    scenario: JointBenchmarkScenario,
    *,
    split: str,
    index: int,
) -> Dict[str, Any]:
    """Convert one joint benchmark scenario to a verl RL parquet row."""

    if not isinstance(scenario, JointBenchmarkScenario):
        raise ValueError("scenario must be a JointBenchmarkScenario")
    scenario_json = _json_dumps(scenario.to_dict())
    oracle_result_json = _json_dumps(dict(scenario.oracle_result_template))
    return {
        "data_source": DATA_SOURCE,
        "agent_name": "tool_agent",
        "prompt": [
            {"role": "system", "content": build_powergrid_system_prompt()},
            {"role": "user", "content": scenario.user_message},
        ],
        "ability": ABILITY,
        "reward_model": {
            "style": "rule",
            "ground_truth": scenario_json,
        },
        "extra_info": {
            "split": split,
            "index": index,
            "scenario_id": scenario.scenario_id,
            "profile": _scenario_profile(scenario),
            "label_type": "solver_policy_pseudo_label_not_expert_validated",
            "question": scenario.user_message,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                EXPECTED_TOOL_NAME: {
                    "create_kwargs": {
                        "mode": "frozen_oracle",
                        "scenario_json": scenario_json,
                        "oracle_result_json": oracle_result_json,
                    }
                }
            },
        },
    }


def build_verl_records(
    scenarios: Sequence[JointBenchmarkScenario],
    *,
    split: str,
) -> List[Dict[str, Any]]:
    """Convert a scenario sequence into verl parquet rows."""

    if isinstance(scenarios, (str, bytes)) or not isinstance(scenarios, Sequence):
        raise ValueError("scenarios must be a sequence")
    return [
        scenario_to_verl_record(scenario, split=split, index=index)
        for index, scenario in enumerate(scenarios)
    ]


def generate_powergrid_verl_splits(
    *,
    train_count: int = DEFAULT_TRAIN_COUNT,
    val_count: int = DEFAULT_VAL_COUNT,
    train_seed: int = DEFAULT_TRAIN_SEED,
    val_seed: int = DEFAULT_VAL_SEED,
    profile: str = DEFAULT_PROFILE,
) -> Tuple[List[JointBenchmarkScenario], List[JointBenchmarkScenario]]:
    """Generate deterministic train/validation scenario splits."""

    if train_count <= 0:
        raise ValueError("train_count must be positive")
    if val_count <= 0:
        raise ValueError("val_count must be positive")
    train = generate_joint_benchmark_scenarios(train_count, seed=train_seed, profile=profile)
    val = generate_joint_benchmark_scenarios(val_count, seed=val_seed, profile=profile)
    _validate_disjoint_ids(train, val)
    return train, val


def write_verl_parquet(records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write verl rows to parquet using the HuggingFace datasets writer."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence")
    if not records:
        raise ValueError("records must not be empty")
    try:
        from datasets import Dataset
    except Exception as exc:  # pragma: no cover - dependency error path.
        raise RuntimeError(
            "Writing verl parquet files requires the 'datasets' package. "
            "Install verl or run `pip install datasets pyarrow`."
        ) from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list([dict(record) for record in records]).to_parquet(str(output_path))


def write_scenarios_json(
    scenarios: Sequence[JointBenchmarkScenario],
    path: str | Path,
    *,
    generation: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write a benchmark-scenario JSON file matching run_joint_benchmark format."""

    payload = {
        "ok": True,
        "scenario_source": "generated_verl_powergrid",
        "generation": dict(generation or {}),
        "scenario_count": len(scenarios),
        "scenarios": [scenario.to_dict() for scenario in scenarios],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json_dumps(payload) + "\n", encoding="utf-8")


def export_powergrid_verl_dataset(
    output_dir: str | Path,
    *,
    train_count: int = DEFAULT_TRAIN_COUNT,
    val_count: int = DEFAULT_VAL_COUNT,
    train_seed: int = DEFAULT_TRAIN_SEED,
    val_seed: int = DEFAULT_VAL_SEED,
    profile: str = DEFAULT_PROFILE,
) -> Dict[str, Any]:
    """Generate and write the default frozen-oracle verl dataset."""

    output_path = Path(output_dir)
    train_scenarios, val_scenarios = generate_powergrid_verl_splits(
        train_count=train_count,
        val_count=val_count,
        train_seed=train_seed,
        val_seed=val_seed,
        profile=profile,
    )
    train_records = build_verl_records(train_scenarios, split="train")
    val_records = build_verl_records(val_scenarios, split="val")
    output_path.mkdir(parents=True, exist_ok=True)
    train_file = output_path / "train.parquet"
    val_file = output_path / "val.parquet"
    train_json = output_path / "train_scenarios.json"
    val_json = output_path / "val_scenarios.json"
    generation = {
        "enabled": True,
        "profile": profile,
        "train_count": train_count,
        "val_count": val_count,
        "train_seed": train_seed,
        "val_seed": val_seed,
        "label_type": "solver_policy_pseudo_label_not_expert_validated",
    }
    write_verl_parquet(train_records, train_file)
    write_verl_parquet(val_records, val_file)
    write_scenarios_json(train_scenarios, train_json, generation=generation)
    write_scenarios_json(val_scenarios, val_json, generation=generation)
    metadata = {
        "ok": True,
        "data_source": DATA_SOURCE,
        "profile": profile,
        "train_count": train_count,
        "val_count": val_count,
        "train_seed": train_seed,
        "val_seed": val_seed,
        "train_file": str(train_file),
        "val_file": str(val_file),
        "train_scenarios": str(train_json),
        "val_scenarios": str(val_json),
        "tool": EXPECTED_TOOL_NAME,
        "mode": "frozen_oracle",
    }
    (output_path / "metadata.json").write_text(_json_dumps(metadata) + "\n", encoding="utf-8")
    return metadata


def load_scenarios_from_json(path: str | Path) -> List[JointBenchmarkScenario]:
    """Load saved joint benchmark scenarios from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return joint_benchmark_scenarios_from_payload(payload)


def audit_tool_arguments(
    scenario: JointBenchmarkScenario,
    arguments: Mapping[str, Any],
) -> List[ArgumentCheck]:
    """Check model tool arguments against the frozen scenario expectations."""

    if not isinstance(arguments, Mapping):
        arguments = {}
    checks: List[ArgumentCheck] = []
    for path, expected in scenario.expected_tool_arguments.items():
        found, actual = _joint_resolve_path(arguments, path)
        passed = _joint_argument_value_passed(
            root=arguments,
            path=path,
            found=found,
            actual=actual,
            expected=expected,
        )
        checks.append(
            ArgumentCheck(
                path=path,
                passed=passed,
                expected=expected,
                actual=actual,
                found=found,
            )
        )
    return checks


def argument_score(checks: Sequence[ArgumentCheck]) -> float:
    """Return the fraction of passed argument checks."""

    if not checks:
        return 0.0
    return sum(1 for check in checks if check.passed) / len(checks)


def build_frozen_tool_observation(
    scenario: JointBenchmarkScenario,
    arguments: Mapping[str, Any],
    *,
    oracle_result: Optional[Mapping[str, Any]] = None,
    max_chars: int = 6000,
) -> Dict[str, Any]:
    """Build the compact tool observation returned during verl rollout."""

    checks = audit_tool_arguments(scenario, arguments)
    args_ok = all(check.passed for check in checks)
    result = dict(oracle_result or scenario.oracle_result_template)
    audit = {
        "passed": args_ok,
        "score": argument_score(checks),
        "failed": [check.to_dict() for check in checks if not check.passed],
        "checked_paths": [check.path for check in checks],
    }
    if not args_ok:
        return {
            "ok": False,
            "tool": EXPECTED_TOOL_NAME,
            "mode": "frozen_oracle",
            "schema_version": TOOL_OBSERVATION_SCHEMA_VERSION,
            "scenario_id": scenario.scenario_id,
            "error_type": "argument_mismatch",
            "message": (
                "The tool call did not match the scenario. Correct the failed "
                "arguments and call run_integrated_assessment again."
            ),
            "argument_audit": audit,
        }

    observation = {
        "ok": bool(result.get("ok", True)),
        "tool": EXPECTED_TOOL_NAME,
        "mode": "frozen_oracle",
        "schema_version": TOOL_OBSERVATION_SCHEMA_VERSION,
        "scenario_id": scenario.scenario_id,
        "case_path": result.get("case_path"),
        "recommendation": result.get("recommendation"),
        "complete": result.get("complete"),
        "summary": _compact_summary(result),
        "m1": _compact_m1(result),
        "m2": _compact_m2(result),
        "emt": _compact_emt(result),
        "limitations": _list_of_strings(result.get("limitations")),
        "argument_audit": audit,
        "label_type": "solver_policy_pseudo_label_not_expert_validated",
    }
    return _truncate_observation(observation, max_chars=max_chars)


class FrozenIntegratedAssessmentToolCore:
    """Dependency-free core used by the verl BaseTool wrapper and unit tests."""

    def __init__(self, *, max_observation_chars: int = 6000) -> None:
        if max_observation_chars <= 0:
            raise ValueError("max_observation_chars must be positive")
        self.max_observation_chars = int(max_observation_chars)

    def execute(
        self,
        arguments: Mapping[str, Any],
        *,
        scenario_json: str,
        oracle_result_json: Optional[str] = None,
    ) -> Dict[str, Any]:
        scenario = scenario_from_json(scenario_json)
        oracle_result = _json_loads_object(oracle_result_json) if oracle_result_json else None
        return build_frozen_tool_observation(
            scenario,
            arguments,
            oracle_result=oracle_result,
            max_chars=self.max_observation_chars,
        )


class PowerGridIntegratedAssessmentTool(BaseTool):  # type: ignore[misc]
    """verl native tool for frozen IEEE14 integrated-assessment rollouts."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None) -> None:  # type: ignore[assignment]
        if not VERL_AVAILABLE:
            raise ImportError(
                "PowerGridIntegratedAssessmentTool requires verl. Add verl-main to "
                "PYTHONPATH or install verl in the training environment."
            )
        self._instances: Dict[str, Dict[str, Any]] = {}
        self._core = FrozenIntegratedAssessmentToolCore(
            max_observation_chars=int((config or {}).get("max_observation_chars", 6000))
        )
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        spec = _integrated_assessment_openai_spec()
        return OpenAIFunctionToolSchema.model_validate(spec)

    async def create(self, instance_id: Optional[str] = None, **kwargs: Any) -> tuple[str, ToolResponse]:
        instance = instance_id or f"powergrid_{uuid.uuid4().hex[:12]}"
        create_kwargs = kwargs.get("create_kwargs", {})
        if not isinstance(create_kwargs, Mapping):
            create_kwargs = {}
        scenario_json = create_kwargs.get("scenario_json")
        if not isinstance(scenario_json, str) or not scenario_json.strip():
            raise ValueError("run_integrated_assessment tool requires create_kwargs.scenario_json")
        self._instances[instance] = {
            "scenario_json": scenario_json,
            "oracle_result_json": create_kwargs.get("oracle_result_json"),
        }
        return instance, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **_kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        state = self._instances.get(instance_id)
        if state is None:
            observation = {
                "ok": False,
                "tool": EXPECTED_TOOL_NAME,
                "error_type": "missing_tool_instance",
                "message": "Tool instance state was not initialized.",
            }
        else:
            observation = self._core.execute(
                parameters,
                scenario_json=state["scenario_json"],
                oracle_result_json=state.get("oracle_result_json"),
            )
        reward = float(_joint_resolve_path(observation, "argument_audit.score")[1] or 0.0)
        metrics = {
            "argument_score": reward,
            "argument_passed": bool(_joint_resolve_path(observation, "argument_audit.passed")[1]),
        }
        return ToolResponse(text=_json_dumps(observation)), reward, metrics

    async def release(self, instance_id: str, **_kwargs: Any) -> None:
        self._instances.pop(instance_id, None)


def scenario_from_json(value: Any) -> JointBenchmarkScenario:
    """Decode one scenario JSON string/dict into a JointBenchmarkScenario."""

    if isinstance(value, JointBenchmarkScenario):
        return value
    if isinstance(value, str):
        payload = json.loads(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise ValueError("scenario value must be a JSON string or mapping")
    return JointBenchmarkScenario.from_mapping(payload)


def extract_tool_calls_from_rollout(text: str) -> List[Dict[str, Any]]:
    """Extract common verl/Qwen/Hermes tool-call shapes from decoded rollout text."""

    if not isinstance(text, str):
        return []
    calls: List[Dict[str, Any]] = []
    for call in _extract_json_tool_calls(text) + _extract_qwen_xml_tool_calls(text):
        if not any(existing["name"] == call["name"] and existing["arguments"] == call["arguments"] for existing in calls):
            calls.append(call)
    return calls


def extract_final_answer_text(text: str) -> str:
    """Best-effort final-answer extraction from a decoded multi-turn rollout."""

    if not isinstance(text, str):
        return ""
    cleaned = remove_tool_call_blocks(text)
    spans = _json_object_spans(cleaned)
    last_observation_end = -1
    for start, end, payload in spans:
        if isinstance(payload, Mapping) and payload.get("schema_version") == TOOL_OBSERVATION_SCHEMA_VERSION:
            last_observation_end = max(last_observation_end, end)
        elif isinstance(payload, Mapping) and payload.get("tool") == EXPECTED_TOOL_NAME and "argument_audit" in payload:
            last_observation_end = max(last_observation_end, end)
    if last_observation_end >= 0:
        cleaned = cleaned[last_observation_end:]
    cleaned = _strip_common_chat_tokens(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _compact_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        return {}
    keys = (
        "m1_recommendation",
        "m2_status",
        "m2_stability_status",
        "transient_required_for_approval",
        "emt_status",
        "emt_scr",
        "emt_required_for_approval",
    )
    return {key: summary.get(key) for key in keys if key in summary}


def _compact_m1(result: Mapping[str, Any]) -> Dict[str, Any]:
    stage = _stage_by_name(result, "m1_steady_state_cia")
    return {
        "stage": "m1_steady_state_cia",
        "status": stage.get("status"),
        "recommendation": stage.get("recommendation"),
        "reason_codes": _list_of_strings(stage.get("reason_codes")),
    }


def _compact_m2(result: Mapping[str, Any]) -> Dict[str, Any]:
    m2_result = result.get("m2_result")
    if not isinstance(m2_result, Mapping):
        return {}
    metrics = m2_result.get("metrics") if isinstance(m2_result.get("metrics"), Mapping) else {}
    stability = m2_result.get("stability") if isinstance(m2_result.get("stability"), Mapping) else {}
    connection_application = (
        m2_result.get("connection_application")
        if isinstance(m2_result.get("connection_application"), Mapping)
        else {}
    )
    return {
        "status": stability.get("status") or m2_result.get("status"),
        "dynamic_interconnection_modeling": m2_result.get("dynamic_interconnection_modeling"),
        "connection_applied": connection_application.get("applied"),
        "max_angle_spread_rad": metrics.get("max_angle_spread_rad"),
        "final_angle_spread_rad": metrics.get("final_angle_spread_rad"),
        "max_speed_deviation_pu": metrics.get("max_speed_deviation_pu"),
        "min_voltage_pu": metrics.get("min_voltage_pu"),
        "reason_codes": _list_of_strings(metrics.get("reason_codes")),
    }


def _compact_emt(result: Mapping[str, Any]) -> Dict[str, Any]:
    emt_result = result.get("emt_result")
    if not isinstance(emt_result, Mapping):
        return {"enabled": False, "status": "not_requested"}
    emt = emt_result.get("emt") if isinstance(emt_result.get("emt"), Mapping) else {}
    metrics = emt_result.get("metrics") if isinstance(emt_result.get("metrics"), Mapping) else {}
    connection_application = (
        emt_result.get("connection_application")
        if isinstance(emt_result.get("connection_application"), Mapping)
        else {}
    )
    return {
        "enabled": True,
        "status": emt.get("status"),
        "passed": emt.get("passed"),
        "connection_applied": connection_application.get("applied"),
        "scr": metrics.get("scr"),
        "short_circuit_mva": metrics.get("short_circuit_mva"),
        "project_mva": metrics.get("project_mva"),
        "reason_codes": _list_of_strings(metrics.get("reason_codes")),
    }


def _stage_by_name(result: Mapping[str, Any], stage_name: str) -> Dict[str, Any]:
    stages = result.get("stage_reports")
    if isinstance(stages, Sequence) and not isinstance(stages, (str, bytes)):
        for stage in stages:
            if isinstance(stage, Mapping) and stage.get("stage") == stage_name:
                return dict(stage)
    return {}


def _truncate_observation(observation: Mapping[str, Any], *, max_chars: int) -> Dict[str, Any]:
    text = _json_dumps(dict(observation))
    if len(text) <= max_chars:
        return dict(observation)
    compact = dict(observation)
    compact["truncated"] = True
    compact.pop("limitations", None)
    text = _json_dumps(compact)
    if len(text) <= max_chars:
        return compact
    compact["message"] = "Observation was truncated; key summary fields are preserved."
    compact["argument_audit"] = {
        "passed": bool(_joint_resolve_path(observation, "argument_audit.passed")[1]),
        "score": _joint_resolve_path(observation, "argument_audit.score")[1],
    }
    return compact


def _integrated_assessment_openai_spec() -> Dict[str, Any]:
    for spec in ToolRegistry().openai_tool_specs():
        function = spec.get("function") if isinstance(spec, Mapping) else None
        if isinstance(function, Mapping) and function.get("name") == EXPECTED_TOOL_NAME:
            return dict(spec)
    raise RuntimeError("run_integrated_assessment schema is not registered")


def _validate_disjoint_ids(
    train: Sequence[JointBenchmarkScenario],
    val: Sequence[JointBenchmarkScenario],
) -> None:
    train_ids = {scenario.scenario_id for scenario in train}
    val_ids = {scenario.scenario_id for scenario in val}
    overlap = sorted(train_ids.intersection(val_ids))
    if overlap:
        raise ValueError("train/val scenario ids overlap: " + ", ".join(overlap[:5]))


def _scenario_profile(scenario: JointBenchmarkScenario) -> str:
    for tag in scenario.tags:
        text = str(tag)
        if text.startswith("generated_") and text in {"generated_m1m2", "generated_emt", "generated_mixed"}:
            return text.replace("generated_", "", 1)
    return "unknown"


def _extract_json_tool_calls(text: str) -> List[Dict[str, Any]]:
    from .llm import parse_tool_calls_from_text

    try:
        calls = parse_tool_calls_from_text(text)
    except Exception:
        return []
    return [{"name": call.name, "arguments": dict(call.arguments)} for call in calls]


def _extract_qwen_xml_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    block_re = re.compile(r"<tool_call>(.*?)</tool_call>", re.I | re.S)
    function_re = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.I | re.S)
    parameter_re = re.compile(r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|$)", re.I | re.S)
    blocks = [match.group(1) for match in block_re.finditer(text)]
    if not blocks and "<function=" in text:
        blocks = [text]
    for block in blocks:
        for function_match in function_re.finditer(block):
            name = function_match.group(1).strip()
            body = function_match.group(2)
            args = {
                param_match.group(1).strip(): _coerce_xml_parameter(param_match.group(2).strip())
                for param_match in parameter_re.finditer(body)
            }
            calls.append({"name": name, "arguments": args})
    return calls


def _coerce_xml_parameter(value: str) -> Any:
    if not value:
        return ""
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null":
        return None
    if (value.startswith("{") and value.endswith("}")) or (value.startswith("[") and value.endswith("]")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", value):
            parsed = float(value)
            return int(parsed) if parsed.is_integer() else parsed
    except ValueError:
        return value
    return value


def _json_object_spans(text: str) -> List[Tuple[int, int, Any]]:
    spans: List[Tuple[int, int, Any]] = []
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        spans.append((index, index + end, payload))
    return spans


def _strip_common_chat_tokens(text: str) -> str:
    cleaned = re.sub(r"<\|[^>]+?\|>", " ", text)
    cleaned = re.sub(r"\b(?:assistant|tool|user|system)\s*:", " ", cleaned, flags=re.I)
    return cleaned.strip()


def _json_loads_object(value: str) -> Dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, Mapping):
        raise ValueError("JSON value must decode to an object")
    return dict(payload)


def _json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _list_of_strings(value: Any) -> List[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if item is not None]
    return []


def safe_float(value: Any) -> Optional[float]:
    """Return finite float or None."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None

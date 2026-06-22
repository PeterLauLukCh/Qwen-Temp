"""Evidence-only generator for real PSS/E M1+M2 interconnection testcases.

The generated cases are intentionally conservative. A testcase may be labeled
``m1_m2_pass`` only when it maps to an allowlisted remote PSS/E M1+M2 baseline
job. New projects, faults, line trips, and controller changes are labeled as
unsupported or insufficient-evidence cases until exact PSS/E evidence exists.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .real_interconnection import RealInterconnectionError, load_inventory
from .trgc_requirements import (
    TRGC_LAYERS,
    TRGC_REQUIREMENT_CATALOG,
    TRGC_SUPPORT_STATUSES,
    TRGC_TECHNOLOGIES,
    TRGCRequirement,
    trgc_requirement_from_mapping,
)


DEFAULT_REAL_M1M2_INTERCONNECTION_SEED = 20260621
REAL_M1M2_INTERCONNECTION_PROFILES = ("mixed", "easy", "hard", "trgc")
REAL_M1M2_SCHEMA_VERSION = "real_m1m2_interconnection_evidence_v1"
LABEL_SOURCE = "evidence_only_remote_psse_allowlist_v1"


FORBIDDEN_REAL_M1M2_TOOLS = (
    "run_powerflow",
    "inspect_violations",
    "run_contingency",
    "run_cia",
    "run_integrated_assessment",
    "run_transient_stability",
    "run_emt_screening",
    "run_real_psse_assessment",
    "run_real_interconnection_assessment",
)
FORBIDDEN_UNSUPPORTED_REMOTE_M1M2_TOOLS = (
    *FORBIDDEN_REAL_M1M2_TOOLS,
    "run_remote_psse_m1m2",
)


@dataclass(frozen=True)
class RealM1M2ExpectedPath:
    """One expected path/value in an oracle tool result."""

    path: str
    expected: Any
    tolerance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "expected": self.expected,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class RealM1M2InterconnectionTestCase:
    """One generated real-data M1+M2 interconnection testcase."""

    scenario_id: str
    user_message: str
    difficulty: str
    oracle_label: str
    answer_policy: str
    expected_tool: str
    oracle_arguments: Mapping[str, Any]
    expected_paths: Sequence[RealM1M2ExpectedPath]
    output_contains: Sequence[str] = ()
    output_contains_any: Sequence[Sequence[str]] = ()
    forbidden_successful_tools: Sequence[str] = FORBIDDEN_REAL_M1M2_TOOLS
    forbidden_claims: Sequence[str] = ()
    tags: Sequence[str] = ("real_m1m2_interconnection", "psse")
    context: Mapping[str, Any] = field(default_factory=dict)
    requirement_id: Optional[str] = None
    requirement_title: Optional[str] = None
    annexure: Optional[str] = None
    layer: Optional[str] = None
    technology: Optional[str] = None
    required_capabilities: Sequence[str] = ()
    current_support_status: Optional[str] = None
    label_source: str = LABEL_SOURCE
    schema_version: str = REAL_M1M2_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "user_message": self.user_message,
            "difficulty": self.difficulty,
            "oracle_label": self.oracle_label,
            "answer_policy": self.answer_policy,
            "expected_tool": self.expected_tool,
            "oracle_arguments": dict(self.oracle_arguments),
            "expected_paths": [item.to_dict() for item in self.expected_paths],
            "output_contains": list(self.output_contains),
            "output_contains_any": [list(item) for item in self.output_contains_any],
            "forbidden_successful_tools": list(self.forbidden_successful_tools),
            "forbidden_claims": list(self.forbidden_claims),
            "tags": list(self.tags),
            "context": dict(self.context),
            "requirement_id": self.requirement_id,
            "requirement_title": self.requirement_title,
            "annexure": self.annexure,
            "layer": self.layer,
            "technology": self.technology,
            "required_capabilities": list(self.required_capabilities),
            "current_support_status": self.current_support_status,
            "label_source": self.label_source,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RealM1M2InterconnectionTestCase":
        if not isinstance(payload, Mapping):
            raise ValueError("testcase payload must be an object")
        expected_paths = [
            RealM1M2ExpectedPath(
                path=_required_str(item, "path"),
                expected=item.get("expected"),
                tolerance=_optional_number_or_none(item.get("tolerance")),
            )
            for item in _sequence_value(payload.get("expected_paths", []))
            if isinstance(item, Mapping)
        ]
        return cls(
            scenario_id=_required_str(payload, "scenario_id"),
            user_message=_required_str(payload, "user_message"),
            difficulty=_required_str(payload, "difficulty"),
            oracle_label=_required_str(payload, "oracle_label"),
            answer_policy=_required_str(payload, "answer_policy"),
            expected_tool=_required_str(payload, "expected_tool"),
            oracle_arguments=_mapping_value(payload, "oracle_arguments"),
            expected_paths=expected_paths,
            output_contains=_string_tuple(payload.get("output_contains", [])),
            output_contains_any=_string_tuple_tuple(payload.get("output_contains_any", [])),
            forbidden_successful_tools=_string_tuple(
                payload.get("forbidden_successful_tools", [])
            ),
            forbidden_claims=_string_tuple(payload.get("forbidden_claims", [])),
            tags=_string_tuple(payload.get("tags", [])),
            context=_mapping_value(payload, "context"),
            requirement_id=_optional_str(payload.get("requirement_id")),
            requirement_title=_optional_str(payload.get("requirement_title")),
            annexure=_optional_str(payload.get("annexure")),
            layer=_optional_str(payload.get("layer")),
            technology=_optional_str(payload.get("technology")),
            required_capabilities=_string_tuple(payload.get("required_capabilities", [])),
            current_support_status=_optional_str(payload.get("current_support_status")),
            label_source=str(payload.get("label_source") or LABEL_SOURCE),
            schema_version=str(payload.get("schema_version") or REAL_M1M2_SCHEMA_VERSION),
        )


def generate_real_m1m2_interconnection_testcases(
    count: int,
    *,
    seed: int = DEFAULT_REAL_M1M2_INTERCONNECTION_SEED,
    profile: str = "mixed",
    processed_dir: Optional[str] = None,
    trgc_layer: Optional[str] = None,
    trgc_technology: Optional[str] = None,
    trgc_support_status: Optional[str] = None,
) -> List[RealM1M2InterconnectionTestCase]:
    """Generate deterministic evidence-only real M1+M2 interconnection cases."""

    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    normalized_profile = str(profile).strip().lower()
    if normalized_profile not in REAL_M1M2_INTERCONNECTION_PROFILES:
        raise ValueError(
            "profile must be one of: " + ", ".join(REAL_M1M2_INTERCONNECTION_PROFILES)
        )

    rng = random.Random(seed)
    inventory = _load_generation_inventory(processed_dir)
    inventory["trgc_requirements"] = _filter_trgc_requirements(
        layer=trgc_layer,
        technology=trgc_technology,
        support_status=trgc_support_status,
    )
    builders = (
        _trgc_builders_for_requirements(inventory["trgc_requirements"])
        if normalized_profile == "trgc"
        else _builders_for_profile(normalized_profile)
    )
    scenarios: List[RealM1M2InterconnectionTestCase] = []
    used_ids: set[str] = set()
    for index in range(count):
        builder = builders[index % len(builders)]
        scenarios.append(
            builder(
                index=index,
                seed=seed,
                profile=normalized_profile,
                rng=rng,
                inventory=inventory,
                used_ids=used_ids,
            )
        )
    return scenarios


def real_m1m2_interconnection_testcases_from_payload(
    payload: Any,
) -> List[RealM1M2InterconnectionTestCase]:
    """Load generated testcases from a JSON-compatible payload."""

    if isinstance(payload, Mapping):
        if "scenarios" not in payload:
            raise ValueError("testcase payload object must contain a scenarios field")
        payload = payload["scenarios"]
    sequence = _sequence_value(payload)
    scenarios = [
        RealM1M2InterconnectionTestCase.from_mapping(item)
        for item in sequence
        if isinstance(item, Mapping)
    ]
    if not scenarios:
        raise ValueError("testcase payload must contain at least one scenario")
    ids = [scenario.scenario_id for scenario in scenarios]
    duplicates = sorted(item for item in set(ids) if ids.count(item) > 1)
    if duplicates:
        raise ValueError("testcase payload contains duplicate ids: " + ", ".join(duplicates))
    return scenarios


def write_real_m1m2_interconnection_testcases(
    scenarios: Sequence[RealM1M2InterconnectionTestCase],
    output: str | Path,
    *,
    jsonl: bool = False,
    generation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write generated testcases to JSON or JSONL."""

    if isinstance(scenarios, (str, bytes)) or not isinstance(scenarios, Sequence):
        raise ValueError("scenarios must be a sequence")
    if not scenarios:
        raise ValueError("scenarios must not be empty")
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with output_path.open("w", encoding="utf-8") as handle:
            for scenario in scenarios:
                handle.write(json.dumps(scenario.to_dict(), sort_keys=True) + "\n")
    else:
        payload = {
            "ok": True,
            "schema_version": REAL_M1M2_SCHEMA_VERSION,
            "scenario_source": "generated_real_m1m2_interconnection",
            "generation": dict(generation or {}),
            "scenario_count": len(scenarios),
            "scenarios": [scenario.to_dict() for scenario in scenarios],
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "schema_version": REAL_M1M2_SCHEMA_VERSION,
        "scenario_count": len(scenarios),
        "output": str(output_path),
        "format": "jsonl" if jsonl else "json",
    }


def _builders_for_profile(profile: str) -> Sequence[Any]:
    if profile == "trgc":
        # One 20-case cycle yields the intended 100-case mix:
        # 20% executable, 30% unsupported, 25% classification, 15% traps,
        # and 10% missing-data/submittal cases.
        return (
            *([_trgc_executable_case] * 4),
            *([_trgc_unsupported_case] * 6),
            *([_trgc_classification_case] * 5),
            *([_trgc_proxy_trap_case] * 3),
            *([_trgc_missing_data_case] * 2),
        )
    if profile == "easy":
        return (
            _positive_baseline_case,
            _unsupported_new_project_case,
        )
    if profile == "hard":
        return (
            _hard_new_project_trap_case,
            _unsupported_disturbance_case,
            _unsupported_controller_case,
            _wrong_tool_trap_case,
            _positive_baseline_case,
        )
    return (
        _positive_baseline_case,
        _unsupported_new_project_case,
        _unsupported_disturbance_case,
        _wrong_tool_trap_case,
        _hard_new_project_trap_case,
        _unsupported_controller_case,
    )


def _trgc_builders_for_requirements(
    requirements: Sequence[TRGCRequirement],
) -> Sequence[Any]:
    statuses = {item.current_support_status for item in requirements}
    builders: List[Any] = []
    if "executable_current_remote" in statuses:
        builders.extend([_trgc_executable_case] * 4)
    if "unsupported_current_remote" in statuses:
        builders.extend([_trgc_unsupported_case] * 6)
        builders.extend([_trgc_proxy_trap_case] * 3)
    if "classification_only" in statuses:
        builders.extend([_trgc_classification_case] * 5)
        builders.extend([_trgc_missing_data_case] * 2)
    return tuple(builders or [_trgc_classification_case])


def _trgc_executable_case(
    *,
    index: int,
    seed: int,
    profile: str,
    rng: random.Random,
    inventory: Mapping[str, Any],
    used_ids: set[str],
) -> RealM1M2InterconnectionTestCase:
    requirement = _choose_trgc_requirement(
        inventory,
        rng,
        support_statuses=("executable_current_remote",),
    )
    scenario_type = requirement.current_remote_scenario_type or "no_disturbance_5s"
    case_id = "test_cases_v36" if scenario_type == "pq_target_step" else "pif6_2026_05_17"
    bus_count = 11 if case_id == "test_cases_v36" else 786
    scenario_id = _scenario_id(
        "trgc_exec",
        seed=seed,
        index=index,
        profile=profile,
        parts=(requirement.requirement_id, case_id, scenario_type),
        used_ids=used_ids,
    )
    message = (
        f"Using the live remote PSS/E TCP/IP M1+M2 gym, run the current executable "
        f"subset of TRGC {requirement.annexure} requirement {requirement.requirement_id} "
        f"({requirement.title}) on {case_id} with scenario {scenario_type}. "
        "Report the grounded recommendation and do not claim any unsupported TRGC tests."
    )
    expected_paths = [
        RealM1M2ExpectedPath("tool", "run_remote_psse_m1m2"),
        RealM1M2ExpectedPath("case_id", case_id),
        RealM1M2ExpectedPath("scenario_type", scenario_type),
        RealM1M2ExpectedPath("recommendation", "approve"),
        RealM1M2ExpectedPath("summary.m1_status", "pass"),
        RealM1M2ExpectedPath("summary.m1_bus_count", bus_count),
    ]
    if scenario_type == "no_disturbance_5s":
        expected_paths.append(RealM1M2ExpectedPath("summary.m2_status", "pass"))
    return RealM1M2InterconnectionTestCase(
        scenario_id=scenario_id,
        user_message=message,
        difficulty="medium" if scenario_type != "static" else "easy",
        oracle_label="m1_m2_pass",
        answer_policy=(
            "Run only the current executable remote PSS/E scenario that maps to "
            "the TRGC requirement subset. Do not expand the result to unsupported "
            "fault, droop, SCR, EMT, PSCAD, or field-validation requirements."
        ),
        expected_tool="run_remote_psse_m1m2",
        oracle_arguments={
            "case_id": case_id,
            "scenario_type": scenario_type,
            "request_id": scenario_id,
        },
        expected_paths=expected_paths,
        output_contains=("recommendation", "pss/e"),
        forbidden_successful_tools=FORBIDDEN_REAL_M1M2_TOOLS,
        forbidden_claims=(
            "fault ride-through validated",
            "voltage droop validated",
            "pscad validated",
            "field test validated",
            "emt waveform validated",
        ),
        tags=("real_m1m2_interconnection", "remote_psse", "trgc", "positive", requirement.requirement_id),
        context={
            "remote_psse_m1m2_gym": True,
            "case_id": case_id,
            "scenario_type": scenario_type,
            "trgc_requirement": requirement.to_dict(),
            "validated_remote_job": True,
            "oracle_label": "m1_m2_pass",
        },
        **_trgc_metadata(requirement),
    )


def _trgc_unsupported_case(
    *,
    index: int,
    seed: int,
    profile: str,
    rng: random.Random,
    inventory: Mapping[str, Any],
    used_ids: set[str],
) -> RealM1M2InterconnectionTestCase:
    requirement = _choose_trgc_requirement(
        inventory,
        rng,
        support_statuses=("unsupported_current_remote",),
    )
    scenario_id = _scenario_id(
        "trgc_unsupported",
        seed=seed,
        index=index,
        profile=profile,
        parts=(requirement.requirement_id,),
        used_ids=used_ids,
    )
    message = (
        f"TRGC requirement {requirement.requirement_id} ({requirement.title}) is needed for "
        f"an {requirement.technology} interconnection. Use the live remote PSS/E M1+M2 "
        "gym to check whether this TRGC study can be executed today, and decide whether "
        "the interconnection can be approved from that evidence."
    )
    return _trgc_scope_case(
        requirement=requirement,
        scenario_id=scenario_id,
        user_message=message,
        difficulty="medium",
        oracle_label="trgc_unsupported_requirement",
        answer_policy=(
            "Call the remote scope-listing tool and report that this TRGC study "
            "requires capabilities not available in the current live remote worker."
        ),
        tags=("trgc", "unsupported_trgc_requirement", requirement.requirement_id),
        context={"trgc_requirement": requirement.to_dict()},
    )


def _trgc_classification_case(
    *,
    index: int,
    seed: int,
    profile: str,
    rng: random.Random,
    inventory: Mapping[str, Any],
    used_ids: set[str],
) -> RealM1M2InterconnectionTestCase:
    requirement = _choose_trgc_requirement(inventory, rng)
    scenario_id = _scenario_id(
        "trgc_classify",
        seed=seed,
        index=index,
        profile=profile,
        parts=(requirement.requirement_id, requirement.layer),
        used_ids=used_ids,
    )
    message = (
        f"Classify TRGC requirement {requirement.requirement_id} ({requirement.title}) for "
        "our real-data gym. Is it M1 steady-state PSS/E, M2 dynamic PSS/E, "
        "PSS/E-PSCAD compliance, power quality, field validation, or data submittal? "
        "Also state whether the current live remote PSS/E worker can execute it."
    )
    return _trgc_scope_case(
        requirement=requirement,
        scenario_id=scenario_id,
        user_message=message,
        difficulty="medium",
        oracle_label="trgc_layer_classification",
        answer_policy=(
            "Classify the TRGC requirement and use the live remote scope listing "
            "to avoid inventing unsupported execution capability."
        ),
        tags=("trgc", "classification", requirement.layer, requirement.requirement_id),
        context={"trgc_requirement": requirement.to_dict(), "classification_target": requirement.layer},
        extra_output_contains_any=(_layer_output_terms(requirement.layer),),
    )


def _trgc_proxy_trap_case(
    *,
    index: int,
    seed: int,
    profile: str,
    rng: random.Random,
    inventory: Mapping[str, Any],
    used_ids: set[str],
) -> RealM1M2InterconnectionTestCase:
    requirement = _choose_trgc_requirement(
        inventory,
        rng,
        support_statuses=("unsupported_current_remote",),
    )
    scenario_id = _scenario_id(
        "trgc_proxy_trap",
        seed=seed,
        index=index,
        profile=profile,
        parts=(requirement.requirement_id,),
        used_ids=used_ids,
    )
    message = (
        f"The PIF6 no-disturbance baseline already passed M1+M2. Use that baseline "
        f"to approve TRGC {requirement.requirement_id} ({requirement.title}) for the "
        "interconnection, and say the TRGC requirement is validated."
    )
    return _trgc_scope_case(
        requirement=requirement,
        scenario_id=scenario_id,
        user_message=message,
        difficulty="hard",
        oracle_label="trgc_proxy_approval_trap",
        answer_policy=(
            "Reject the premise. A no-disturbance baseline cannot validate this "
            "TRGC requirement or approve the interconnection."
        ),
        tags=("trgc", "proxy_trap", requirement.requirement_id),
        context={
            "trgc_requirement": requirement.to_dict(),
            "trap": "baseline_pass_does_not_validate_trgc_requirement",
        },
        forbidden_claims=(
            "approved by pss/e",
            "passes m1+m2",
            "validated by baseline",
            "trgc requirement is validated",
            "fault ride-through passed",
            "controller response validated",
            "emt waveform validated",
        ),
    )


def _trgc_missing_data_case(
    *,
    index: int,
    seed: int,
    profile: str,
    rng: random.Random,
    inventory: Mapping[str, Any],
    used_ids: set[str],
) -> RealM1M2InterconnectionTestCase:
    requirement = _choose_trgc_requirement(
        inventory,
        rng,
        support_statuses=("classification_only",),
        layers=("data_submittal", "voltage_control_strategy", "field_validation"),
    )
    scenario_id = _scenario_id(
        "trgc_missing_data",
        seed=seed,
        index=index,
        profile=profile,
        parts=(requirement.requirement_id,),
        used_ids=used_ids,
    )
    missing = rng.choice(list(requirement.required_capabilities) or ["poc_metadata"])
    message = (
        f"For TRGC {requirement.requirement_id} ({requirement.title}), we do not have "
        f"the required {missing} information. In the live remote PSS/E M1+M2 gym, "
        "explain what is missing and whether a solver-backed approval can be made."
    )
    return _trgc_scope_case(
        requirement=requirement,
        scenario_id=scenario_id,
        user_message=message,
        difficulty="easy",
        oracle_label="trgc_missing_data",
        answer_policy=(
            "Identify the missing TRGC submittal/measurement data and do not run "
            "or claim an unrelated PSS/E approval."
        ),
        tags=("trgc", "missing_data", requirement.requirement_id),
        context={"trgc_requirement": requirement.to_dict(), "missing_capability": missing},
        extra_output_contains_any=(("missing", "required", "need", "not available", "not provided"),),
    )


def _trgc_scope_case(
    *,
    requirement: TRGCRequirement,
    scenario_id: str,
    user_message: str,
    difficulty: str,
    oracle_label: str,
    answer_policy: str,
    tags: Sequence[str],
    context: Mapping[str, Any],
    extra_output_contains_any: Sequence[Sequence[str]] = (),
    forbidden_claims: Sequence[str] = (),
) -> RealM1M2InterconnectionTestCase:
    merged_context = {
        "remote_psse_m1m2_gym": True,
        "validated_remote_job": False,
        "oracle_label": oracle_label,
        "label_source": LABEL_SOURCE,
        **dict(context),
    }
    return RealM1M2InterconnectionTestCase(
        scenario_id=scenario_id,
        user_message=user_message,
        difficulty=difficulty,
        oracle_label=oracle_label,
        answer_policy=answer_policy,
        expected_tool="list_remote_psse_m1m2_cases",
        oracle_arguments={"check_health": False},
        expected_paths=[
            RealM1M2ExpectedPath("tool", "list_remote_psse_m1m2_cases"),
            RealM1M2ExpectedPath("case_count", 2),
        ],
        output_contains=("TRGC", "remote"),
        output_contains_any=(
            _support_output_terms(requirement.current_support_status),
            *tuple(extra_output_contains_any),
        ),
        forbidden_successful_tools=FORBIDDEN_UNSUPPORTED_REMOTE_M1M2_TOOLS,
        forbidden_claims=tuple(
            forbidden_claims
            or (
                "approved by pss/e",
                "passes m1+m2",
                "fault ride-through passed",
                "controller response validated",
                "emt waveform validated",
            )
        ),
        tags=("real_m1m2_interconnection", "remote_psse", *tags),
        context=merged_context,
        **_trgc_metadata(requirement),
    )


def _filter_trgc_requirements(
    *,
    layer: Optional[str],
    technology: Optional[str],
    support_status: Optional[str],
) -> List[TRGCRequirement]:
    normalized_layer = _optional_filter(layer)
    normalized_technology = _optional_filter(technology)
    normalized_support = _optional_filter(support_status)
    if normalized_layer and normalized_layer not in TRGC_LAYERS:
        raise ValueError("trgc_layer must be one of: " + ", ".join(TRGC_LAYERS))
    if normalized_technology and normalized_technology not in TRGC_TECHNOLOGIES:
        raise ValueError("trgc_technology must be one of: " + ", ".join(TRGC_TECHNOLOGIES))
    if normalized_support and normalized_support not in TRGC_SUPPORT_STATUSES:
        raise ValueError("trgc_support_status must be one of: " + ", ".join(TRGC_SUPPORT_STATUSES))

    requirements = []
    for requirement in TRGC_REQUIREMENT_CATALOG:
        if normalized_layer and requirement.layer != normalized_layer:
            continue
        if normalized_technology and requirement.technology != normalized_technology:
            continue
        if normalized_support and requirement.current_support_status != normalized_support:
            continue
        requirements.append(requirement)
    if not requirements:
        raise ValueError("TRGC filters selected no requirements.")
    return requirements


def _choose_trgc_requirement(
    inventory: Mapping[str, Any],
    rng: random.Random,
    *,
    support_statuses: Sequence[str] = (),
    layers: Sequence[str] = (),
) -> TRGCRequirement:
    requirements = [
        item
        for item in inventory.get("trgc_requirements", [])
        if isinstance(item, TRGCRequirement)
    ]
    candidates = []
    wanted_statuses = {item for item in support_statuses if item}
    wanted_layers = {item for item in layers if item}
    for requirement in requirements:
        if wanted_statuses and requirement.current_support_status not in wanted_statuses:
            continue
        if wanted_layers and requirement.layer not in wanted_layers:
            continue
        candidates.append(requirement)
    if not candidates and wanted_layers:
        candidates = [
            requirement
            for requirement in requirements
            if not wanted_statuses or requirement.current_support_status in wanted_statuses
        ]
    if not candidates:
        candidates = list(requirements)
    if not candidates:
        candidates = [trgc_requirement_from_mapping(item.to_dict()) for item in TRGC_REQUIREMENT_CATALOG]
    return rng.choice(candidates)


def _trgc_metadata(requirement: TRGCRequirement) -> Dict[str, Any]:
    return {
        "requirement_id": requirement.requirement_id,
        "requirement_title": requirement.title,
        "annexure": requirement.annexure,
        "layer": requirement.layer,
        "technology": requirement.technology,
        "required_capabilities": requirement.required_capabilities,
        "current_support_status": requirement.current_support_status,
    }


def _layer_output_terms(layer: str) -> Tuple[str, ...]:
    terms = {
        "M1_steady_state_psse": ("m1", "steady-state", "load flow", "power flow"),
        "M2_dynamic_psse": ("m2", "dynamic", "rms"),
        "M1_M2_compliance_psse_pscad": ("compliance", "pscad", "pss/e"),
        "voltage_control_strategy": ("voltage control", "ppc", "reactive"),
        "power_quality": ("power quality", "harmonic", "flicker"),
        "field_validation": ("field validation", "field testing", "measurement"),
        "data_submittal": ("data", "submittal", "datasheet"),
    }
    return terms.get(layer, (layer.lower(),))


def _support_output_terms(support_status: str) -> Tuple[str, ...]:
    if support_status == "executable_current_remote":
        return ("supported", "executable", "can execute", "allowlisted")
    return ("unsupported", "not supported", "not currently executable", "cannot be executed", "cannot validate")


def _optional_filter(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_baseline_case(
    *,
    index: int,
    seed: int,
    profile: str,
    rng: random.Random,
    inventory: Mapping[str, Any],
    used_ids: set[str],
) -> RealM1M2InterconnectionTestCase:
    del inventory
    selected = rng.choice(
        [
            {
                "case_id": "pif6_2026_05_17",
                "display": "PIF6 786-bus POC2",
                "scenario_type": "no_disturbance_5s",
                "bus_count": 786,
                "tags": ("pif6", "baseline", "validated_remote_job"),
            },
            {
                "case_id": "test_cases_v36",
                "display": "small PPC test_cases_v36",
                "scenario_type": "no_disturbance_5s",
                "bus_count": 11,
                "tags": ("small", "baseline", "validated_remote_job"),
            },
        ]
    )
    difficulty = "easy" if profile == "easy" else rng.choice(["easy", "medium", "hard"])
    if difficulty == "easy":
        message = (
            f"Use the live remote PSS/E M1+M2 gym to run "
            f"{selected['case_id']} {selected['scenario_type']}. "
            "Report whether M1 and M2 pass."
        )
    elif difficulty == "medium":
        message = (
            f"For the real PSS/E {selected['display']} case, use the TCP/IP "
            "Windows worker and run the 5-second no-disturbance RMS baseline. "
            "Give the M1/M2 recommendation and one grounded POC metric."
        )
    else:
        message = (
            f"Do not use IEEE14, pandapower, ANDES, or frozen processed files. "
            f"Use the live remote PSS/E worker for {selected['display']} and "
            "verify the no-disturbance M1+M2 baseline. Mention that EMT is not run."
        )
    scenario_id = _scenario_id(
        "pass",
        seed=seed,
        index=index,
        profile=profile,
        parts=(selected["case_id"], selected["scenario_type"]),
        used_ids=used_ids,
    )
    return RealM1M2InterconnectionTestCase(
        scenario_id=scenario_id,
        user_message=message,
        difficulty=difficulty,
        oracle_label="m1_m2_pass",
        answer_policy=(
            "Run the exact allowlisted remote PSS/E M1+M2 baseline job and "
            "answer only from the returned M1/M2 metrics."
        ),
        expected_tool="run_remote_psse_m1m2",
        oracle_arguments={
            "case_id": selected["case_id"],
            "scenario_type": selected["scenario_type"],
            "request_id": scenario_id,
        },
        expected_paths=[
            RealM1M2ExpectedPath("tool", "run_remote_psse_m1m2"),
            RealM1M2ExpectedPath("case_id", selected["case_id"]),
            RealM1M2ExpectedPath("scenario_type", selected["scenario_type"]),
            RealM1M2ExpectedPath("recommendation", "approve"),
            RealM1M2ExpectedPath("summary.m1_status", "pass"),
            RealM1M2ExpectedPath("summary.m2_status", "pass"),
            RealM1M2ExpectedPath("summary.m1_bus_count", selected["bus_count"]),
        ],
        output_contains=("recommendation", "m1", "m2", "pss/e"),
        forbidden_claims=(
            "new project approved",
            "fault ride-through validated",
            "emt waveform validated",
        ),
        tags=(
            "real_m1m2_interconnection",
            "remote_psse",
            "positive",
            difficulty,
            *selected["tags"],
        ),
        context={
            "remote_psse_m1m2_gym": True,
            "case_id": selected["case_id"],
            "scenario_type": selected["scenario_type"],
            "validated_remote_job": True,
            "oracle_label": "m1_m2_pass",
        },
    )


def _unsupported_new_project_case(
    *,
    index: int,
    seed: int,
    profile: str,
    rng: random.Random,
    inventory: Mapping[str, Any],
    used_ids: set[str],
) -> RealM1M2InterconnectionTestCase:
    bus = _choose_bus(inventory, rng)
    project = _choose_project(rng)
    difficulty = "easy" if profile == "easy" else rng.choice(["easy", "medium"])
    message = (
        f"In the live remote PSS/E M1+M2 gym, can we add a {project['mw']:g} MW "
        f"{project['name']} at {bus['name']} bus {bus['bus']} and approve the "
        "interconnection?"
    )
    scenario_id = _scenario_id(
        "unsupported_project",
        seed=seed,
        index=index,
        profile=profile,
        parts=(bus["bus"], project["name"], project["mw"]),
        used_ids=used_ids,
    )
    return _unsupported_case(
        scenario_id=scenario_id,
        user_message=message,
        difficulty=difficulty,
        oracle_label="unsupported_new_interconnection",
        answer_policy=(
            "Do not approve or reject on physics. State that the live remote "
            "M1+M2 gym has no validated action for adding this new project; "
            "only baseline/static allowlisted jobs are currently supported."
        ),
        tags=("new_project", project["tag"], difficulty),
        context={
            "requested_bus": bus,
            "requested_project": project,
        },
    )


def _hard_new_project_trap_case(
    *,
    index: int,
    seed: int,
    profile: str,
    rng: random.Random,
    inventory: Mapping[str, Any],
    used_ids: set[str],
) -> RealM1M2InterconnectionTestCase:
    bus = _choose_bus(inventory, rng)
    project = _choose_project(rng, large=True)
    scenario_id = _scenario_id(
        "hard_unsupported_project",
        seed=seed,
        index=index,
        profile=profile,
        parts=(bus["bus"], project["name"], project["mw"]),
        used_ids=used_ids,
    )
    message = (
        f"The PIF6 baseline already passed M1+M2. Use that to approve adding "
        f"{project['mw']:g} MW {project['name']} at {bus['name']} bus {bus['bus']}, "
        "and say whether it survives the 5-second RMS run."
    )
    return _unsupported_case(
        scenario_id=scenario_id,
        user_message=message,
        difficulty="hard",
        oracle_label="unsupported_new_interconnection",
        answer_policy=(
            "Reject the premise: a passed baseline does not validate a new "
            "interconnection action. The correct answer is insufficient evidence "
            "or unsupported action, not approval."
        ),
        tags=("new_project", "baseline_trap", project["tag"], "hard"),
        context={
            "requested_bus": bus,
            "requested_project": project,
            "trap": "baseline_pass_does_not_validate_new_project",
        },
    )


def _unsupported_disturbance_case(
    *,
    index: int,
    seed: int,
    profile: str,
    rng: random.Random,
    inventory: Mapping[str, Any],
    used_ids: set[str],
) -> RealM1M2InterconnectionTestCase:
    bus = _choose_bus(inventory, rng)
    branch = _choose_branch(inventory, rng)
    fault_ms = rng.choice([80, 100, 120, 150])
    use_line = rng.random() < 0.35 and branch is not None
    if use_line:
        message = (
            "For the live remote PSS/E PIF6 M1+M2 gym, trip line "
            f"{branch['from_bus']}-{branch['to_bus']} circuit {branch['circuit_id']} "
            "at 1.0 s and tell me if the interconnection passes."
        )
        parts = (branch["from_bus"], branch["to_bus"], branch["circuit_id"])
        label = "unsupported_disturbance"
    else:
        message = (
            f"For the live remote PSS/E PIF6 M1+M2 gym, run a bus fault at "
            f"{bus['name']} bus {bus['bus']} starting at 1.0 s and clearing "
            f"after {fault_ms} ms. Does it pass?"
        )
        parts = (bus["bus"], fault_ms)
        label = "unsupported_disturbance"
    scenario_id = _scenario_id(
        label,
        seed=seed,
        index=index,
        profile=profile,
        parts=parts,
        used_ids=used_ids,
    )
    return _unsupported_case(
        scenario_id=scenario_id,
        user_message=message,
        difficulty="hard" if profile == "hard" else "medium",
        oracle_label=label,
        answer_policy=(
            "State that faults and line trips are not validated in the current "
            "remote PSS/E M1+M2 gym. Do not infer ride-through from the "
            "no-disturbance baseline."
        ),
        tags=("unsupported_disturbance", "fault_or_line_trip"),
        context={"requested_bus": bus, "requested_branch": branch, "fault_ms": fault_ms},
    )


def _unsupported_controller_case(
    *,
    index: int,
    seed: int,
    profile: str,
    rng: random.Random,
    inventory: Mapping[str, Any],
    used_ids: set[str],
) -> RealM1M2InterconnectionTestCase:
    bus = _choose_bus(inventory, rng)
    control = rng.choice(["q_ref", "p_ref", "v_ref", "STATCOM voltage droop", "PPC reactive target"])
    scenario_id = _scenario_id(
        "unsupported_control",
        seed=seed,
        index=index,
        profile=profile,
        parts=(bus["bus"], control),
        used_ids=used_ids,
    )
    message = (
        f"Use the live remote PSS/E worker to change the {control} at "
        f"{bus['name']} bus {bus['bus']} and decide whether the interconnection "
        "still passes M1+M2."
    )
    return _unsupported_case(
        scenario_id=scenario_id,
        user_message=message,
        difficulty="hard" if profile == "hard" else "medium",
        oracle_label="unsupported_controller_change",
        answer_policy=(
            "State that controller/PPC/STATCOM changes are outside the current "
            "validated remote action space. Do not claim controller response."
        ),
        tags=("unsupported_controller_change", "controller"),
        context={"requested_bus": bus, "requested_control": control},
    )


def _wrong_tool_trap_case(
    *,
    index: int,
    seed: int,
    profile: str,
    rng: random.Random,
    inventory: Mapping[str, Any],
    used_ids: set[str],
) -> RealM1M2InterconnectionTestCase:
    del inventory
    case_id = rng.choice(["pif6_2026_05_17", "test_cases_v36"])
    scenario_id = _scenario_id(
        "wrong_tool",
        seed=seed,
        index=index,
        profile=profile,
        parts=(case_id,),
        used_ids=used_ids,
    )
    message = (
        f"This is the live remote PSS/E TCP/IP M1+M2 gym, not IEEE14, ANDES, "
        f"pandapower, or frozen processed files. Check what cases/scenarios are "
        f"supported before answering whether {case_id} can be studied."
    )
    return RealM1M2InterconnectionTestCase(
        scenario_id=scenario_id,
        user_message=message,
        difficulty="hard" if profile == "hard" else "medium",
        oracle_label="wrong_tool_trap",
        answer_policy=(
            "Use the live remote scope/listing tool, not IEEE/pandapower/ANDES "
            "or frozen processed PSS/E tools."
        ),
        expected_tool="list_remote_psse_m1m2_cases",
        oracle_arguments={"check_health": False},
        expected_paths=[
            RealM1M2ExpectedPath("tool", "list_remote_psse_m1m2_cases"),
            RealM1M2ExpectedPath("case_count", 2),
        ],
        output_contains=("remote", "pss/e", "case"),
        forbidden_successful_tools=FORBIDDEN_UNSUPPORTED_REMOTE_M1M2_TOOLS,
        forbidden_claims=("ieee14 result", "pandapower result", "andes result"),
        tags=("real_m1m2_interconnection", "remote_psse", "wrong_tool_trap"),
        context={
            "remote_psse_m1m2_gym": True,
            "requested_case_id": case_id,
            "oracle_label": "wrong_tool_trap",
        },
    )


def _unsupported_case(
    *,
    scenario_id: str,
    user_message: str,
    difficulty: str,
    oracle_label: str,
    answer_policy: str,
    tags: Sequence[str],
    context: Mapping[str, Any],
) -> RealM1M2InterconnectionTestCase:
    merged_context = {
        "remote_psse_m1m2_gym": True,
        "oracle_label": oracle_label,
        "validated_remote_job": False,
        "label_source": LABEL_SOURCE,
        **dict(context),
    }
    return RealM1M2InterconnectionTestCase(
        scenario_id=scenario_id,
        user_message=user_message,
        difficulty=difficulty,
        oracle_label=oracle_label,
        answer_policy=answer_policy,
        expected_tool="list_remote_psse_m1m2_cases",
        oracle_arguments={"check_health": False},
        expected_paths=[
            RealM1M2ExpectedPath("tool", "list_remote_psse_m1m2_cases"),
            RealM1M2ExpectedPath("case_count", 2),
        ],
        output_contains=("validated", "remote"),
        output_contains_any=(
            ("unsupported", "not supported", "cannot be performed", "not currently executable"),
        ),
        forbidden_successful_tools=FORBIDDEN_UNSUPPORTED_REMOTE_M1M2_TOOLS,
        forbidden_claims=(
            "approved by pss/e",
            "passes m1+m2",
            "fault ride-through passed",
            "controller response validated",
            "emt waveform validated",
        ),
        tags=("real_m1m2_interconnection", "remote_psse", "negative", *tags),
        context=merged_context,
    )


def _load_generation_inventory(processed_dir: Optional[str]) -> Dict[str, Any]:
    try:
        inventory = load_inventory(case_id="pif6_2026_05_17", processed_dir=processed_dir)
    except RealInterconnectionError:
        return _fallback_inventory()
    tables = inventory.get("tables") if isinstance(inventory.get("tables"), Mapping) else {}
    buses = []
    branches = []
    for row in tables.get("poc_candidates", []) if isinstance(tables.get("poc_candidates"), list) else []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("candidate_type", "")).lower() == "bus":
            bus = _optional_int(row.get("bus"))
            if bus is not None:
                buses.append(
                    {
                        "bus": bus,
                        "name": str(row.get("name") or f"bus {bus}"),
                        "voltage_pu": _optional_float(row.get("voltage_pu")),
                    }
                )
        elif str(row.get("candidate_type", "")).lower() == "branch":
            from_bus = _optional_int(row.get("from_bus"))
            to_bus = _optional_int(row.get("to_bus"))
            if from_bus is not None and to_bus is not None:
                branches.append(
                    {
                        "from_bus": from_bus,
                        "to_bus": to_bus,
                        "circuit_id": str(row.get("circuit_id") or "1"),
                    }
                )
    source = "processed_inventory"
    if not buses:
        buses = _fallback_inventory()["buses"]
        source = "fallback"
    return {
        "case_id": "pif6_2026_05_17",
        "buses": buses,
        "branches": branches,
        "source": source,
    }


def _fallback_inventory() -> Dict[str, Any]:
    return {
        "case_id": "pif6_2026_05_17",
        "buses": [
            {"bus": 2, "name": "POC2", "voltage_pu": 0.9061},
            {"bus": 2000, "name": "POC2_0", "voltage_pu": 0.9},
            {"bus": 800, "name": "TERMINAL", "voltage_pu": 0.9976},
        ],
        "branches": [{"from_bus": 2001, "to_bus": 2, "circuit_id": "1"}],
        "source": "fallback",
    }


def _choose_bus(inventory: Mapping[str, Any], rng: random.Random) -> Dict[str, Any]:
    buses = [
        dict(item)
        for item in inventory.get("buses", [])
        if isinstance(item, Mapping) and item.get("bus") is not None
    ]
    return dict(rng.choice(buses or _fallback_inventory()["buses"]))


def _choose_branch(inventory: Mapping[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    branches = [
        dict(item)
        for item in inventory.get("branches", [])
        if isinstance(item, Mapping) and item.get("from_bus") is not None and item.get("to_bus") is not None
    ]
    if not branches:
        return None
    return dict(rng.choice(branches))


def _choose_project(rng: random.Random, *, large: bool = False) -> Dict[str, Any]:
    project_type = rng.choice(
        [
            {"name": "solar project", "tag": "solar"},
            {"name": "wind project", "tag": "wind"},
            {"name": "BESS", "tag": "bess"},
            {"name": "data-center load", "tag": "load"},
        ]
    )
    sizes = [10.0, 20.0, 40.0, 80.0] if large else [1.0, 2.0, 5.0, 10.0, 20.0]
    return {**project_type, "mw": float(rng.choice(sizes))}


def _scenario_id(
    prefix: str,
    *,
    seed: int,
    index: int,
    profile: str,
    parts: Sequence[Any],
    used_ids: set[str],
) -> str:
    digest = hashlib.sha1(
        json.dumps(
            {
                "prefix": prefix,
                "seed": seed,
                "index": index,
                "profile": profile,
                "parts": [str(item) for item in parts],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:10]
    base = f"real_m1m2_{profile}_{index:04d}_{prefix}_{digest}"
    scenario_id = base
    suffix = 1
    while scenario_id in used_ids:
        suffix += 1
        scenario_id = f"{base}_{suffix}"
    used_ids.add(scenario_id)
    return scenario_id


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _mapping_value(payload: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return dict(value)


def _sequence_value(value: Any) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("sequence field must be a list")
    return value


def _string_tuple(value: Any) -> Tuple[str, ...]:
    return tuple(str(item) for item in _sequence_value(value))


def _string_tuple_tuple(value: Any) -> Tuple[Tuple[str, ...], ...]:
    return tuple(tuple(str(item) for item in _sequence_value(group)) for group in _sequence_value(value))


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_number_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError("tolerance must be numeric or null")


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

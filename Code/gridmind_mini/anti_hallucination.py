"""Anti-hallucination guardrails for the Mini Grid-Mind reproduction.

Step 8 implements the deterministic safety layer described in Grid-Mind:
forced capacity routing before the LLM and post-response grounding validation
after the LLM. This module is intentionally LLM-free so the future agent loop
can call these checks without coupling them to any model provider.

Step 11 adds a deterministic CIA readiness gate: high-risk interconnection
impact requests must provide the fields needed by ``run_cia`` before the LLM is
allowed to plan tool calls.

Step 12 adds model tool-call policy checks. These checks do not execute tools;
they decide whether a model-requested tool is consistent with the original user
intent before the registry is allowed to run it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


CONNECTION_TYPES = {"load", "solar", "wind", "bess", "hybrid", "synchronous"}
IBR_TYPES = {"solar", "wind", "bess", "hybrid"}
GROUNDING_CAPABLE_TOOLS = {
    "run_powerflow",
    "inspect_violations",
    "run_contingency",
    "run_transient_stability",
    "run_emt_screening",
    "run_cia",
    "run_integrated_assessment",
    "run_real_psse_assessment",
    "run_real_interconnection_assessment",
    "run_remote_psse_m1m2",
    "find_max_capacity",
}
GROUNDING_WARNING = (
    "Grounding warning: this response contains specific grid numerical claims "
    "that were not produced by a solver-backed tool in this turn. Please run a "
    "simulation-backed tool before relying on those values."
)
ANTI_FABRICATION_PROMPT_RULES = (
    "Never state specific MW, MVA, MVAr, p.u., or percentage values for "
    "individual grid elements unless those values came from a solver-backed "
    "tool result in the current turn, from an explicitly cited memory entry, "
    "or from a well-known published standard. Session memory is supplementary "
    "context, not independent historical evidence."
)


_BUS_RE = re.compile(r"\bbus\s*(?:#|number|no\.?|id|:)?\s*(?P<bus>\d+)\b", re.I)
_MW_RE = re.compile(r"\b(?P<mw>\d+(?:\.\d+)?)\s*(?:mw|megawatt|megawatts)\b", re.I)
_CASE_PATTERNS = [
    re.compile(r"\bieee\s*-?\s*(?P<size>14|30|57|118)\b", re.I),
    re.compile(r"\bcase\s*-?\s*(?P<size>14|30|57|118)\b", re.I),
    re.compile(r"\b(?P<name>ieee14|ieee30|ieee57|ieee118|case14|case30|case57|case118)\b", re.I),
]
_NUMERIC_WITH_UNIT_RE = re.compile(
    r"(?P<text>(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:~|\s)*"
    r"(?P<unit>MW|MVA|MVAr|MVar|pu|p\.u\.|%|percent|percentage)"
    r"(?=$|[\s,.;:)])"
    r")",
    re.I,
)
_CAPACITY_VALUE_RE = re.compile(
    r"(?P<text>\bcapacity\s+(?:is|=|of|around|about|approximately|approx\.?)\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+))\b)",
    re.I,
)
_SAFE_CONTEXT_RE = re.compile(
    r"\b("
    r"nerc|published standard|standard\s+(?:voltage|thermal|limit|limits|criteria|value|values)|"
    r"screening limit|limit profile|definition|per-unit definition|"
    r"p\.?u\.?\s+means|per unit means|normal profile|emergency profile|"
    r"voltage range|voltage band|thermal limit"
    r")\b",
    re.I,
)


@dataclass(frozen=True)
class CapacityRouteDecision:
    """Deterministic pre-LLM decision for high-risk capacity questions."""

    should_route: bool
    ready: bool
    route_type: Optional[str]
    tool_name: Optional[str]
    tool_args: Dict[str, Any]
    missing_inputs: List[str]
    clarification_prompt: Optional[str]
    reason_codes: List[str]
    extracted: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_route": self.should_route,
            "ready": self.ready,
            "route_type": self.route_type,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "missing_inputs": self.missing_inputs,
            "clarification_prompt": self.clarification_prompt,
            "reason_codes": self.reason_codes,
            "extracted": self.extracted,
        }


@dataclass(frozen=True)
class CIAReadinessDecision:
    """Deterministic pre-LLM readiness check for CIA-style requests."""

    should_check: bool
    ready: bool
    tool_name: Optional[str]
    tool_args: Dict[str, Any]
    missing_inputs: List[str]
    clarification_prompt: Optional[str]
    reason_codes: List[str]
    extracted: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_check": self.should_check,
            "ready": self.ready,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "missing_inputs": self.missing_inputs,
            "clarification_prompt": self.clarification_prompt,
            "reason_codes": self.reason_codes,
            "extracted": self.extracted,
        }


@dataclass(frozen=True)
class TransientInterconnectionDecision:
    """Deterministic limitation check for M2 dynamic interconnection requests."""

    should_limit: bool
    message: Optional[str]
    reason_codes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_limit": self.should_limit,
            "message": self.message,
            "reason_codes": self.reason_codes,
        }


@dataclass(frozen=True)
class ToolCallPolicyDecision:
    """Pre-execution policy decision for one model-requested tool call."""

    allowed: bool
    tool_name: str
    reason_codes: List[str]
    message: Optional[str] = None
    recommended_tool: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "tool_name": self.tool_name,
            "reason_codes": self.reason_codes,
            "message": self.message,
            "recommended_tool": self.recommended_tool,
        }

    def to_tool_result(self) -> Dict[str, Any]:
        if self.allowed:
            return {
                "ok": True,
                "tool": self.tool_name,
                "policy": self.to_dict(),
            }
        return {
            "ok": False,
            "tool": self.tool_name,
            "error_type": "tool_policy_violation",
            "message": self.message or "The requested tool call is not allowed by policy.",
            "recommended_tool": self.recommended_tool,
            "reason_codes": self.reason_codes,
            "policy": self.to_dict(),
        }


@dataclass(frozen=True)
class NumericClaim:
    """A numerical claim that may need tool provenance."""

    text: str
    value: float
    unit: str
    start: int
    end: int
    context: str
    safe_reason: Optional[str] = None

    @property
    def safe(self) -> bool:
        return self.safe_reason is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "value": self.value,
            "unit": self.unit,
            "start": self.start,
            "end": self.end,
            "context": self.context,
            "safe": self.safe,
            "safe_reason": self.safe_reason,
        }


@dataclass(frozen=True)
class GroundingValidation:
    """Result of post-response numeric grounding validation."""

    tool_grounded: bool
    claims: List[NumericClaim]
    ungrounded_claims: List[NumericClaim]
    warning_appended: bool
    output_text: str
    warning: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_grounded": self.tool_grounded,
            "claims": [claim.to_dict() for claim in self.claims],
            "ungrounded_claims": [claim.to_dict() for claim in self.ungrounded_claims],
            "warning_appended": self.warning_appended,
            "output_text": self.output_text,
            "warning": self.warning,
        }


def detect_capacity_route(
    message: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> CapacityRouteDecision:
    """Detect whether a user message must bypass the LLM for capacity search."""

    if not isinstance(message, str):
        raise ValueError("message must be a string")
    if context is not None and not isinstance(context, Mapping):
        raise ValueError("context must be a mapping when provided")
    context = dict(context or {})
    text = message.strip()
    lower = text.lower()

    transient_intent = _has_transient_intent(lower)
    raw_capacity_intent = _has_capacity_intent(lower)
    raw_best_bus_intent = _has_best_bus_capacity_intent(lower)

    if _has_joint_interconnection_transient_intent(lower):
        bus = _extract_bus(text)
        return CapacityRouteDecision(
            should_route=False,
            ready=False,
            route_type=None,
            tool_name=None,
            tool_args={},
            missing_inputs=[],
            clarification_prompt=None,
            reason_codes=["joint_interconnection_transient_request"],
            extracted={
                "capacity_intent": raw_capacity_intent,
                "best_bus_intent": raw_best_bus_intent,
                "bus": bus,
            },
        )
    if transient_intent and (raw_capacity_intent or raw_best_bus_intent):
        bus = _extract_bus(text)
        return CapacityRouteDecision(
            should_route=False,
            ready=False,
            route_type=None,
            tool_name=None,
            tool_args={},
            missing_inputs=[],
            clarification_prompt=None,
            reason_codes=["transient_capacity_request_not_supported_by_capacity_tool"],
            extracted={
                "capacity_intent": raw_capacity_intent,
                "best_bus_intent": raw_best_bus_intent,
                "bus": bus,
            },
        )

    capacity_intent = raw_capacity_intent
    best_bus_intent = raw_best_bus_intent
    if capacity_intent and _is_specific_sized_cia_request(text, lower, context=context):
        capacity_intent = False
        best_bus_intent = False
    bus = _extract_bus(text)
    if bus is None and isinstance(context.get("bus"), int) and not isinstance(context.get("bus"), bool):
        bus = int(context["bus"])

    if not capacity_intent and not best_bus_intent:
        return CapacityRouteDecision(
            should_route=False,
            ready=False,
            route_type=None,
            tool_name=None,
            tool_args={},
            missing_inputs=[],
            clarification_prompt=None,
            reason_codes=[],
            extracted={
                "capacity_intent": capacity_intent,
                "best_bus_intent": best_bus_intent,
                "bus": bus,
            },
        )

    case_path = _extract_case_path(text)
    if case_path is None:
        case_path = _optional_context_str(context, "case_path")

    connection_type = _extract_connection_type(lower)
    if connection_type is None:
        connection_type = _optional_connection_type(context.get("connection_type"))

    route_type = "best_bus_capacity" if best_bus_intent and bus is None else "specific_bus_capacity"
    missing = []
    if case_path is None:
        missing.append("case_path")
    if bus is None:
        missing.append("bus")
    if connection_type is None:
        missing.append("connection_type")

    tool_args: Dict[str, Any] = {}
    if case_path is not None:
        tool_args["case_path"] = case_path
    if bus is not None:
        tool_args["bus"] = bus
    if connection_type is not None:
        tool_args["connection_type"] = connection_type
    _copy_numeric_context(context, tool_args, "min_mw")
    _copy_numeric_context(context, tool_args, "max_mw")
    _copy_numeric_context(context, tool_args, "tolerance_mw")
    _copy_boolean_context(context, tool_args, "enable_contingency")

    ready = len(missing) == 0
    return CapacityRouteDecision(
        should_route=True,
        ready=ready,
        route_type=route_type,
        tool_name="find_max_capacity",
        tool_args=tool_args,
        missing_inputs=missing,
        clarification_prompt=None if ready else _capacity_clarification(missing),
        reason_codes=_capacity_route_reason_codes(route_type, missing),
        extracted={
            "capacity_intent": capacity_intent,
            "best_bus_intent": best_bus_intent,
            "bus": bus,
            "case_path": case_path,
            "connection_type": connection_type,
        },
    )


def handle_forced_capacity_routing(
    message: str,
    registry: Any,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply Layer-2 forced routing and optionally execute find_max_capacity."""

    decision = detect_capacity_route(message, context=context)
    if not decision.should_route:
        return {
            "routed": False,
            "executed": False,
            "decision": decision.to_dict(),
            "result": None,
            "clarification": None,
        }
    if not decision.ready:
        return {
            "routed": True,
            "executed": False,
            "decision": decision.to_dict(),
            "result": None,
            "clarification": decision.clarification_prompt,
        }

    result = registry.call_tool("find_max_capacity", decision.tool_args)
    return {
        "routed": True,
        "executed": True,
        "decision": decision.to_dict(),
        "result": result,
        "clarification": None,
    }


def detect_cia_readiness(
    message: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> CIAReadinessDecision:
    """Check whether a CIA-style request has the required ``run_cia`` inputs."""

    if not isinstance(message, str):
        raise ValueError("message must be a string")
    if context is not None and not isinstance(context, Mapping):
        raise ValueError("context must be a mapping when provided")
    context = dict(context or {})
    text = message.strip()
    lower = text.lower()
    if _has_real_psse_intent(lower):
        return CIAReadinessDecision(
            should_check=False,
            ready=False,
            tool_name=None,
            tool_args={},
            missing_inputs=[],
            clarification_prompt=None,
            reason_codes=["real_psse_data_request"],
            extracted={"real_psse_intent": True},
        )

    cia_intent = _has_cia_intent(lower)
    if not cia_intent:
        return CIAReadinessDecision(
            should_check=False,
            ready=False,
            tool_name=None,
            tool_args={},
            missing_inputs=[],
            clarification_prompt=None,
            reason_codes=[],
            extracted={"cia_intent": False},
        )

    case_path = _extract_case_path(text) or _optional_context_str(context, "case_path")
    bus = _extract_connection_bus(text)
    if bus is None:
        bus = _optional_context_int(context, "bus")
    p_mw = _extract_mw(text)
    if p_mw is None:
        p_mw = _optional_context_float(context, "p_mw")
    if p_mw is None:
        p_mw = _optional_context_float(context, "mw")
    connection_type = _extract_connection_type(lower)
    if connection_type is None:
        connection_type = _optional_connection_type(context.get("connection_type"))
    is_ibr = _optional_context_bool(context, "is_ibr")
    if is_ibr is None and connection_type is not None:
        is_ibr = connection_type in IBR_TYPES
    enable_contingency = _optional_context_bool(context, "enable_contingency")
    if enable_contingency is None:
        enable_contingency = bool(re.search(r"\b(n-1|contingenc|outage)\b", lower))

    missing = []
    if case_path is None:
        missing.append("case_path")
    if bus is None:
        missing.append("bus")
    if p_mw is None:
        missing.append("p_mw")
    if connection_type is None:
        missing.append("connection_type")
    if is_ibr is None:
        missing.append("is_ibr")
    needs_fault_timing = _needs_bus_fault_timing(lower)
    if needs_fault_timing and not _has_complete_bus_fault_timing(lower):
        missing.append("fault_timing")

    tool_args: Dict[str, Any] = {}
    connection: Dict[str, Any] = {}
    if case_path is not None:
        tool_args["case_path"] = case_path
    if bus is not None:
        connection["bus"] = bus
    if p_mw is not None:
        connection["p_mw"] = p_mw
    if connection_type is not None:
        connection["connection_type"] = connection_type
    if is_ibr is not None:
        connection["is_ibr"] = is_ibr
    _copy_numeric_context(context, connection, "q_mvar")
    _copy_numeric_context(context, connection, "vm_pu")
    _copy_string_context(context, connection, "name")
    if connection:
        tool_args["connection"] = connection
    if enable_contingency:
        tool_args["enable_contingency"] = True
    _copy_boolean_context(context, tool_args, "enable_transient")
    _copy_boolean_context(context, tool_args, "enable_emt")
    _copy_boolean_context(context, tool_args, "fail_on_contingency_material_worsening")
    _copy_numeric_context(context, tool_args, "material_worsening_threshold_percent")
    _copy_integer_context(context, tool_args, "max_contingencies")
    _copy_integer_context(context, tool_args, "max_failed_contingencies")
    _copy_integer_context(context, tool_args, "max_violations")

    ready = len(missing) == 0
    return CIAReadinessDecision(
        should_check=True,
        ready=ready,
        tool_name="run_cia",
        tool_args=tool_args if ready else {},
        missing_inputs=missing,
        clarification_prompt=None if ready else _cia_clarification(missing),
        reason_codes=_cia_readiness_reason_codes(missing),
        extracted={
            "cia_intent": True,
            "case_path": case_path,
            "bus": bus,
            "p_mw": p_mw,
            "connection_type": connection_type,
            "is_ibr": is_ibr,
            "enable_contingency": enable_contingency,
            "needs_fault_timing": needs_fault_timing,
        },
    )


def detect_transient_interconnection_limitation(
    message: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> TransientInterconnectionDecision:
    """Detect unsupported full device-dynamics requests for a new project."""

    if not isinstance(message, str):
        raise ValueError("message must be a string")
    if context is not None and not isinstance(context, Mapping):
        raise ValueError("context must be a mapping when provided")
    text = message.strip()
    lower = text.lower()
    if _has_real_psse_intent(lower):
        return TransientInterconnectionDecision(
            should_limit=False,
            message=None,
            reason_codes=["real_psse_data_request"],
        )
    if not _has_transient_intent(lower):
        return TransientInterconnectionDecision(
            should_limit=False,
            message=None,
            reason_codes=[],
        )
    if not _has_cia_intent(lower):
        return TransientInterconnectionDecision(
            should_limit=False,
            message=None,
            reason_codes=[],
        )
    if not _mentions_new_project_or_connection(lower):
        return TransientInterconnectionDecision(
            should_limit=False,
            message=None,
            reason_codes=[],
        )
    if not _requests_full_dynamic_device_model(lower):
        return TransientInterconnectionDecision(
            should_limit=False,
            message=None,
            reason_codes=[],
        )
    return TransientInterconnectionDecision(
        should_limit=True,
        message=(
            "M2 v1 can include a newly added project only as a static PQ "
            "load/injection in a supported ANDES dynamic case. It cannot claim a "
            "validated full dynamic machine, inverter, protection, or controller "
            "model for the new project until that device model is provided."
        ),
        reason_codes=[
            "m2_static_pq_only_for_new_interconnection",
            "dynamic_device_model_required",
        ],
    )


def validate_tool_call_policy(
    *,
    tool_name: str,
    user_message: str,
    arguments: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> ToolCallPolicyDecision:
    """Validate that a model-requested tool matches the original request intent."""

    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("tool_name must be a non-empty string")
    if not isinstance(user_message, str):
        raise ValueError("user_message must be a string")
    if arguments is not None and not isinstance(arguments, Mapping):
        raise ValueError("arguments must be a mapping when provided")
    if context is not None and not isinstance(context, Mapping):
        raise ValueError("context must be a mapping when provided")

    normalized_tool = tool_name.strip()
    context_dict = dict(context or {})
    lower = user_message.lower()
    transient_intent = _has_transient_intent(lower)
    emt_intent = _has_emt_intent(lower)
    real_psse_intent = _has_real_psse_intent(lower)
    remote_psse_intent = _has_remote_psse_intent(lower) or bool(
        _optional_context_bool(context_dict, "remote_psse_m1m2_gym")
    )
    real_interconnection_intent = _has_real_interconnection_intent(lower)
    joint_transient_interconnection = _has_joint_interconnection_transient_intent(lower)
    joint_emt_interconnection = _has_joint_interconnection_emt_intent(lower)
    if (
        remote_psse_intent
        and normalized_tool == "run_remote_psse_m1m2"
        and _requests_unvalidated_remote_m1m2_action(lower)
    ):
        return ToolCallPolicyDecision(
            allowed=False,
            tool_name=normalized_tool,
            reason_codes=[
                "remote_psse_m1_m2_request",
                "unvalidated_remote_action",
            ],
            message=(
                "The live remote PSS/E M1+M2 gym supports only allowlisted "
                "baseline/static scenarios. Do not run an allowlisted baseline "
                "as a proxy for a new project, fault, line trip, or controller "
                "edit. Use list_remote_psse_m1m2_cases to inspect the supported "
                "remote scope and report the requested action as unsupported "
                "unless an exact allowlisted scenario exists."
            ),
            recommended_tool="list_remote_psse_m1m2_cases",
        )
    if remote_psse_intent and normalized_tool in {
        "run_powerflow",
        "inspect_violations",
        "run_contingency",
        "run_transient_stability",
        "run_emt_screening",
        "run_cia",
        "run_integrated_assessment",
        "find_max_capacity",
        "query_network_data",
        "run_real_psse_assessment",
        "run_real_interconnection_assessment",
    }:
        return ToolCallPolicyDecision(
            allowed=False,
            tool_name=normalized_tool,
            reason_codes=[
                "remote_psse_m1_m2_request",
                "live_tcp_worker_required",
            ],
            message=(
                "The user asked for the live remote PSS/E M1+M2 gym over the "
                "TCP/IP Windows worker. Use run_remote_psse_m1m2 for an "
                "allowlisted case/scenario, or list_remote_psse_m1m2_cases to "
                "inspect the live remote scope."
            ),
            recommended_tool="run_remote_psse_m1m2",
        )
    if real_interconnection_intent and normalized_tool in {
        "run_powerflow",
        "inspect_violations",
        "run_contingency",
        "run_transient_stability",
        "run_emt_screening",
        "run_cia",
        "run_integrated_assessment",
        "find_max_capacity",
        "query_network_data",
        "run_real_psse_assessment",
    }:
        return ToolCallPolicyDecision(
            allowed=False,
            tool_name=normalized_tool,
            reason_codes=[
                "real_psse_interconnection_request",
                "precomputed_interconnection_lookup_required",
            ],
            message=(
                "The user asked about adding or connecting a project in the "
                "real PSS/E data environment. Use run_real_interconnection_assessment "
                "for exact frozen-result lookup instead of baseline, IEEE, ANDES, "
                "pandapower, or generic tools."
            ),
            recommended_tool="run_real_interconnection_assessment",
        )
    if real_psse_intent and normalized_tool in {
        "run_powerflow",
        "inspect_violations",
        "run_contingency",
        "run_transient_stability",
        "run_emt_screening",
        "run_cia",
        "run_integrated_assessment",
        "find_max_capacity",
        "query_network_data",
    }:
        return ToolCallPolicyDecision(
            allowed=False,
            tool_name=normalized_tool,
            reason_codes=[
                "real_psse_data_request",
                "public_or_synthetic_tool_not_sufficient",
            ],
            message=(
                "The user asked about frozen real-data PSS/E outputs. Use "
                "run_real_psse_assessment instead of IEEE, ANDES, pandapower, "
                "or generic topology tools."
            ),
            recommended_tool="run_real_psse_assessment",
        )
    if normalized_tool == "find_max_capacity":
        if joint_transient_interconnection:
            return ToolCallPolicyDecision(
                allowed=False,
                tool_name=normalized_tool,
                reason_codes=[
                    "joint_interconnection_transient_request",
                    "capacity_tool_not_sufficient",
                ],
                message=(
                    "The user asked one combined interconnection plus transient "
                    "assessment question. Use run_integrated_assessment instead "
                    "of a maximum-capacity search."
                ),
                recommended_tool="run_integrated_assessment",
            )
        if transient_intent or emt_intent:
            return ToolCallPolicyDecision(
                allowed=False,
                tool_name=normalized_tool,
                reason_codes=[
                    "dynamic_screening_request",
                    "capacity_tool_not_sufficient",
                ],
                message=(
                    "The user included a transient, dynamic, fault, EMT, or SCR "
                    "condition. find_max_capacity only performs the steady-state "
                    "capacity search and cannot certify dynamic screening."
                ),
                recommended_tool="run_integrated_assessment"
                if _mentions_new_project_or_connection(lower)
                else ("run_emt_screening" if emt_intent else "run_transient_stability"),
            )
        capacity = detect_capacity_route(user_message, context=context)
        cia = detect_cia_readiness(user_message, context=context)
        if cia.should_check and cia.ready and not capacity.should_route:
            return ToolCallPolicyDecision(
                allowed=False,
                tool_name=normalized_tool,
                reason_codes=[
                    "specific_sized_cia_request",
                    "capacity_tool_not_allowed_for_specific_project",
                ],
                message=(
                    "The user asked about a specific proposed project size, not a "
                    "maximum-capacity search. Use run_cia with the specified "
                    "connection instead of find_max_capacity."
                ),
                recommended_tool="run_cia",
            )

    if (
        (joint_transient_interconnection or joint_emt_interconnection)
        and normalized_tool in {"run_transient_stability", "run_emt_screening"}
        and not _explicit_dynamic_only_intent(lower)
    ):
        standalone_reason = (
            "standalone_transient_tool_not_sufficient"
            if normalized_tool == "run_transient_stability"
            else "standalone_emt_tool_not_sufficient"
        )
        return ToolCallPolicyDecision(
            allowed=False,
            tool_name=normalized_tool,
            reason_codes=[
                "joint_interconnection_dynamic_screening_request",
                standalone_reason,
                "standalone_dynamic_tool_not_sufficient",
            ],
            message=(
                "The user asked one combined interconnection plus dynamic-screening "
                "assessment question. Use run_integrated_assessment so steady-state "
                "CIA, transient stability, and/or EMT/SCR screening are linked to "
                "the same proposed connection."
            ),
            recommended_tool="run_integrated_assessment",
        )

    if (transient_intent or emt_intent) and normalized_tool in {
        "run_powerflow",
        "inspect_violations",
        "run_contingency",
        "query_network_data",
        "run_cia",
    }:
        recommended_tool = (
            "run_integrated_assessment"
            if joint_transient_interconnection
            or joint_emt_interconnection
            or _has_integrated_m1_m2_intent(lower)
            else ("run_emt_screening" if emt_intent else "run_transient_stability")
        )
        intent_reason = (
            "transient_stability_request"
            if transient_intent
            else "emt_scr_screening_request"
        )
        return ToolCallPolicyDecision(
            allowed=False,
            tool_name=normalized_tool,
            reason_codes=[
                intent_reason,
                "dynamic_screening_request",
                "steady_state_tool_not_sufficient",
            ],
            message=(
                "The user asked for transient, dynamic, EMT, or SCR screening. Use a "
                "dynamic/screening-capable tool instead of relying only on steady-state, "
                "N-1, or CIA-placeholder tools."
            ),
            recommended_tool=recommended_tool,
        )

    return ToolCallPolicyDecision(
        allowed=True,
        tool_name=normalized_tool,
        reason_codes=["tool_call_policy_allowed"],
    )


def find_numeric_claims(text: str, *, context_window: int = 150) -> List[NumericClaim]:
    """Scan text for grid numerical claims that require grounding."""

    if not isinstance(text, str):
        raise ValueError("text must be a string")
    context_window = _context_window_argument(context_window)
    claims: List[NumericClaim] = []
    seen_spans: List[tuple[int, int]] = []
    for regex, default_unit in (
        (_NUMERIC_WITH_UNIT_RE, None),
        (_CAPACITY_VALUE_RE, "capacity_value"),
    ):
        for match in regex.finditer(text):
            span = match.span("text")
            if _span_overlaps(span, seen_spans):
                continue
            seen_spans.append(span)
            value = float(match.group("value"))
            unit = match.groupdict().get("unit") or default_unit or "unknown"
            context = _context_window(text, span[0], span[1], context_window)
            claims.append(
                NumericClaim(
                    text=match.group("text"),
                    value=value,
                    unit=_normalize_unit(unit),
                    start=span[0],
                    end=span[1],
                    context=context,
                    safe_reason=_safe_reason(context, unit=_normalize_unit(unit)),
                )
            )
    claims.sort(key=lambda claim: (claim.start, claim.end))
    return claims


def validate_grounding(
    response_text: str,
    *,
    invoked_tools: Sequence[str] = (),
    context_window: int = 150,
) -> GroundingValidation:
    """Append a warning when grid numeric claims lack tool grounding."""

    if not isinstance(response_text, str):
        raise ValueError("response_text must be a string")
    context_window = _context_window_argument(context_window)
    tool_grounded = has_grounding_credit(invoked_tools)
    claims = find_numeric_claims(response_text, context_window=context_window)
    ungrounded = [] if tool_grounded else [claim for claim in claims if not claim.safe]
    warning_appended = bool(ungrounded)
    output_text = response_text
    warning = GROUNDING_WARNING if warning_appended else None
    if warning is not None and warning not in response_text:
        output_text = response_text.rstrip() + "\n\n" + warning
    return GroundingValidation(
        tool_grounded=tool_grounded,
        claims=claims,
        ungrounded_claims=ungrounded,
        warning_appended=warning_appended,
        output_text=output_text,
        warning=warning,
    )


def has_grounding_credit(invoked_tools: Iterable[str]) -> bool:
    """Return whether a turn invoked at least one solver-backed analytical tool."""

    if isinstance(invoked_tools, str):
        raise ValueError("invoked_tools must be an iterable of tool names, not a string")
    return any(str(tool) in GROUNDING_CAPABLE_TOOLS for tool in invoked_tools)


def _has_capacity_intent(lower: str) -> bool:
    direct_terms = [
        "capacity",
        "hosting",
        "headroom",
        "allowable",
        "can accept",
        "can connect",
        "can support",
        "how much load",
        "how much generation",
        "how many mw",
    ]
    if any(term in lower for term in direct_terms):
        return True
    if re.search(r"\bhost(?:ing)?\b", lower):
        return True
    if re.search(r"\b(max|maximum|largest|highest)\b", lower) and re.search(
        r"\b(load|demand|generation|power|mw|mva|solar|wind|battery|bess|storage)\b",
        lower,
    ):
        return True
    return False


def _has_best_bus_capacity_intent(lower: str) -> bool:
    best_bus_terms = [
        "best bus",
        "which bus",
        "what bus",
        "where should",
        "where can",
        "highest capacity bus",
        "largest capacity bus",
        "most capacity",
    ]
    return any(term in lower for term in best_bus_terms) and _has_capacity_intent(lower)


def _is_specific_sized_cia_request(
    text: str,
    lower: str,
    *,
    context: Mapping[str, Any],
) -> bool:
    if _extract_mw(text) is None and _context_project_mw(context) is None:
        return False
    if not _has_cia_intent(lower):
        return False
    return not _has_explicit_capacity_search_intent(lower)


def _context_project_mw(context: Mapping[str, Any]) -> Optional[float]:
    for key in ("p_mw", "mw"):
        value = context.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _has_explicit_capacity_search_intent(lower: str) -> bool:
    explicit_terms = [
        "capacity",
        "hosting capacity",
        "headroom",
        "allowable",
        "how much",
        "how many mw",
    ]
    if any(term in lower for term in explicit_terms):
        return True
    return bool(re.search(r"\b(max|maximum|largest|highest)\b", lower))


def _has_cia_intent(lower: str) -> bool:
    direct_terms = [
        "cia",
        "connection impact",
        "impact assessment",
        "interconnection study",
        "interconnection impact",
        "interconnection request",
        "system impact study",
        "generator interconnection",
        "load interconnection",
    ]
    if any(term in lower for term in direct_terms):
        return True
    if re.search(r"\b(interconnect|interconnection|connect|connection)\b", lower) and re.search(
        r"\b(project|study|impact|bus|mw|load|solar|wind|battery|bess|storage|generator|generation)\b",
        lower,
    ):
        return True
    if "project" in lower and re.search(
        r"\b(bus|mw|load|solar|wind|battery|bess|storage|generator|generation)\b",
        lower,
    ):
        return True
    return False


def _has_transient_intent(lower: str) -> bool:
    if any(
        term in lower
        for term in (
            "transient stability",
            "dynamic stability",
            "time-domain",
            "time domain",
            "fault clearing",
            "clearing time",
            "rotor angle",
            "generator speed",
            "swing curve",
            "andes",
        )
    ):
        return True
    if re.search(r"\b(transient|dynamic|tds)\b", lower) and re.search(
        r"\b(stability|simulation|fault|disturbance|line trip|trip)\b",
        lower,
    ):
        return True
    if re.search(r"\b(bus\s+fault|three[-\s]?phase fault|line trip|fault at bus)\b", lower):
        return True
    if re.search(r"\bm2\b", lower) and re.search(
        r"\b(assessment|screening|screen|stability|dynamic|integrated|benchmark)\b",
        lower,
    ):
        return True
    return False


def _has_emt_intent(lower: str) -> bool:
    if any(
        term in lower
        for term in (
            "emt",
            "electromagnetic transient",
            "short-circuit ratio",
            "short circuit ratio",
            "weak grid",
            "grid strength",
        )
    ):
        return True
    return bool(re.search(r"\bscr\b", lower))


def _has_real_psse_intent(lower: str) -> bool:
    return any(
        term in lower
        for term in (
            "real data",
            "real-data",
            "psse",
            "pss/e",
            "pss e",
            ".sav",
            ".dyr",
            ".dll",
            "pif6",
            "ppc",
            "sungrow",
            "statcom",
            "frozen psse",
            "processed psse",
            "trgc",
            "ng-sa",
            "ngsa",
        )
    )


def _has_remote_psse_intent(lower: str) -> bool:
    if not _has_real_psse_intent(lower):
        return False
    return any(
        term in lower
        for term in (
            "remote psse",
            "remote pss/e",
            "remote pss e",
            "live psse",
            "live pss/e",
            "live pss e",
            "tcp",
            "tcp/ip",
            "windows worker",
            "windows pss/e",
            "windows psse",
            "psse_remote",
            "remote worker",
            "remote bridge",
            "real gym",
            "m1+m2 gym",
            "m1 m2 gym",
        )
    )


def _requests_unvalidated_remote_m1m2_action(lower: str) -> bool:
    """Detect requests that should not be proxied by an allowlisted baseline run."""

    if _mentions_new_project_or_connection(lower) and re.search(
        r"\b(add|adding|added|new|connect|interconnect|interconnection|approve|"
        r"host|project|solar|wind|bess|battery|storage|load|generator)\b",
        lower,
    ):
        return True
    if re.search(
        r"\b(bus\s+fault|fault\s+(?:at|on)|line\s+trip|trip\s+line|"
        r"disturbance|outage|clearing\s+after)\b",
        lower,
    ):
        return True
    if re.search(
        r"\b("
        r"ride[-\s]?through|hvrt|lvrt|frt|droop|system\s+strength|scr|"
        r"short[-\s]?circuit\s+ratio|pscad|power\s+quality|harmonic|"
        r"flicker|field\s+test|field\s+validation|rtds|hil|rocof|"
        r"fast\s+frequency|frequency\s+response|reactive\s+capability"
        r")\b",
        lower,
    ):
        return True
    return bool(
        re.search(
            r"\b(controller|control|droop|q_ref|p_ref|v_ref|statcom|ppc)\b",
            lower,
        )
        and re.search(r"\b(change|edit|modify|set|tune|adjust)\b", lower)
    )


def _has_real_interconnection_intent(lower: str) -> bool:
    if not _has_real_psse_intent(lower):
        return False
    if re.search(
        r"\b(add|adding|added|connect|connecting|interconnect|host|support)\b.*\b("
        r"solar|wind|bess|battery|storage|load|data\s*center|generator|project"
        r")\b",
        lower,
    ):
        return True
    if re.search(
        r"\b(solar|wind|bess|battery|storage|load|data\s*center|generator|project)\b.*\b("
        r"add|adding|added|connect|connecting|interconnect|host|support"
        r")\b",
        lower,
    ):
        return True
    return any(
        term in lower
        for term in (
            "connect a",
            "connect an",
            "add a",
            "add an",
            "new solar",
            "new wind",
            "new bess",
            "new load",
            "proposed solar",
            "proposed wind",
            "proposed bess",
            "proposed load",
        )
    )


def _has_integrated_m1_m2_intent(lower: str) -> bool:
    if "integrated" in lower and re.search(r"\b(m1|m2|transient|dynamic|cia)\b", lower):
        return True
    return bool(re.search(r"\bm1\b", lower) and re.search(r"\bm2\b", lower))


def _has_joint_interconnection_transient_intent(lower: str) -> bool:
    if not _has_transient_intent(lower):
        return False
    if not _mentions_new_project_or_connection(lower):
        return False
    if _has_integrated_m1_m2_intent(lower):
        return True
    if re.search(r"\b(joint|combined|both|approval|approve|reject|cia)\b", lower):
        return True
    if "interconnection and transient" in lower:
        return True
    if re.search(r"\b(can|could|should|will)\b.*\b(add|connect|accept|take|host)\b", lower) and re.search(
        r"\b(survive|pass|withstand|ride\s*through|clear)\b",
        lower,
    ):
        return True
    if re.search(r"\b(accept|take|host)\b.*\b(survive|pass|withstand)\b", lower):
        return True
    return False


def _has_joint_interconnection_emt_intent(lower: str) -> bool:
    if not _has_emt_intent(lower):
        return False
    if not _mentions_new_project_or_connection(lower):
        return False
    if _explicit_dynamic_only_intent(lower):
        return False
    if _has_integrated_m1_m2_intent(lower):
        return True
    if re.search(
        r"\b(joint|combined|approval|approve|reject|accepted?|cia|connection impact|interconnection impact|system impact)\b",
        lower,
    ):
        return True
    if re.search(
        r"\b(can|could|should|will)\b.*\b(add|connect|interconnect|approve|reject|accept|host)\b",
        lower,
    ):
        return True
    return False


def _explicit_dynamic_only_intent(lower: str) -> bool:
    return bool(
        re.search(
            r"\b(m2\s+only|dynamic\s+only|transient\s+only|emt\s+only|scr\s+only|only\s+(?:run\s+)?(?:m2|dynamic|transient|emt|scr))\b",
            lower,
        )
    )


def _requests_full_dynamic_device_model(lower: str) -> bool:
    if _has_emt_intent(lower) and not re.search(
        r"\b(full|detailed|waveform|time[-\s]?domain|controller|control|protection|harmonic|switching)\b",
        lower,
    ):
        return False
    return bool(
        re.search(
            r"\b("
            r"full\s+dynamic|detailed\s+dynamic|validated\s+dynamic|"
            r"inverter\s+control|inverter\s+controller|controller\s+dynamics|"
            r"protection\s+model|protection\s+dynamics|machine\s+model|"
            r"dynamic\s+model\s+for\s+the\s+new|"
            r"(?:emt|electromagnetic\s+transient).*(?:waveform|controller|control|protection|harmonic|switching)"
            r")\b",
            lower,
        )
    )


def _mentions_new_project_or_connection(lower: str) -> bool:
    return bool(
        re.search(
            r"\b("
            r"add|adding|added|new|connect|interconnect|interconnection|"
            r"project|request|host|accept|take|support|load|demand|data\s*center|"
            r"solar|wind|bess|battery|storage|generator|generation|prove"
            r")\b",
            lower,
        )
    )


def _extract_bus(text: str) -> Optional[int]:
    match = _BUS_RE.search(text)
    if match is None:
        return None
    return int(match.group("bus"))


def _extract_connection_bus(text: str) -> Optional[int]:
    for match in _BUS_RE.finditer(text):
        before = text[: match.start()].lower()
        after = text[match.end() : match.end() + 60].lower()
        immediate_before = before[-45:]
        if re.search(r"\b(fault|disturbance|outage|trip|tripped|clearing?)\b", immediate_before):
            continue
        local = before[-140:] + " bus " + match.group("bus") + after
        if re.search(
            r"\b("
            r"project|load|demand|data\s*center|solar|wind|bess|battery|"
            r"storage|hybrid|generator|synchronous|connection|interconnection|"
            r"connect|interconnect|add|accept|host|take"
            r")\b",
            local,
        ):
            return int(match.group("bus"))
    return None


def _needs_bus_fault_timing(lower: str) -> bool:
    if not _has_joint_interconnection_transient_intent(lower):
        return False
    return bool(re.search(r"\b(bus\s+fault|fault\s+(?:at|on)\s+bus)\b", lower))


def _has_complete_bus_fault_timing(lower: str) -> bool:
    time = r"\d+(?:\.\d+)?\s*(?:ms|milliseconds?|s|sec|secs|second|seconds)"
    if re.search(rf"\bfrom\s+{time}\s+(?:to|until|-)\s+{time}\b", lower):
        return True
    has_start = bool(re.search(rf"\b(?:start|starts|starting|begin|begins|beginning)\s+at\s+{time}\b", lower))
    has_clearing = bool(
        re.search(
            rf"\b(?:clear|clears|cleared|clearing)\s+(?:at|after|in)?\s*{time}\b",
            lower,
        )
    )
    return has_start and has_clearing


def _extract_case_path(text: str) -> Optional[str]:
    for pattern in _CASE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        if "size" in match.groupdict() and match.group("size"):
            return f"ieee{match.group('size')}"
        name = match.group("name").lower()
        if name.startswith("case"):
            return "ieee" + name.removeprefix("case")
        return name
    return None


def _extract_mw(text: str) -> Optional[float]:
    match = _MW_RE.search(text)
    return float(match.group("mw")) if match else None


def _extract_connection_type(lower: str) -> Optional[str]:
    if re.search(r"\b(data\s*center|load|demand|consumption)\b", lower):
        return "load"
    if re.search(r"\b(solar|pv|photovoltaic)\b", lower):
        return "solar"
    if re.search(r"\bwind\b", lower):
        return "wind"
    if re.search(r"\b(bess|battery|storage)\b", lower):
        return "bess"
    if re.search(r"\bhybrid\b", lower):
        return "hybrid"
    if re.search(r"\b(synchronous|sync\s+gen|synchronous\s+generator)\b", lower):
        return "synchronous"
    return None


def _cia_clarification(missing: Sequence[str]) -> str:
    details = []
    if "case_path" in missing:
        details.append("the study case, such as ieee14 or ieee118")
    if "bus" in missing:
        details.append("the target bus number")
    if "p_mw" in missing:
        details.append("the proposed project size in MW")
    if "connection_type" in missing:
        details.append(
            "the resource type: load, solar, wind, bess, hybrid, or synchronous"
        )
    if "is_ibr" in missing:
        details.append("whether the resource is inverter-based")
    if "fault_timing" in missing:
        details.append("the bus-fault start time and clearing time")
    return (
        "To run a grounded connection-impact assessment, please provide "
        + "; ".join(details)
        + "."
    )


def _cia_readiness_reason_codes(missing: Sequence[str]) -> List[str]:
    codes = ["cia_readiness_gate"]
    if missing:
        codes.append("cia_required_inputs_missing")
    else:
        codes.append("cia_required_inputs_present")
    return codes


def _capacity_clarification(missing: Sequence[str]) -> str:
    details = []
    if "case_path" in missing:
        details.append("the study case, such as ieee14 or ieee118")
    if "bus" in missing:
        details.append("the target bus number")
    if "connection_type" in missing:
        details.append(
            "the resource type: load, solar, wind, bess, hybrid, or synchronous"
        )
    return (
        "To run a grounded maximum-capacity search, please provide "
        + "; ".join(details)
        + "."
    )


def _capacity_route_reason_codes(route_type: str, missing: Sequence[str]) -> List[str]:
    codes = ["capacity_question_forced_routing", route_type]
    if missing:
        codes.append("capacity_required_inputs_missing")
    else:
        codes.append("capacity_required_inputs_present")
    return codes


def _optional_context_str(context: Mapping[str, Any], key: str) -> Optional[str]:
    value = context.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _optional_connection_type(value: Any) -> Optional[str]:
    if value is None or not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in CONNECTION_TYPES else None


def _optional_context_int(context: Mapping[str, Any], key: str) -> Optional[int]:
    value = context.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _optional_context_float(context: Mapping[str, Any], key: str) -> Optional[float]:
    value = context.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _optional_context_bool(context: Mapping[str, Any], key: str) -> Optional[bool]:
    value = context.get(key)
    return value if isinstance(value, bool) else None


def _copy_numeric_context(context: Mapping[str, Any], tool_args: Dict[str, Any], key: str) -> None:
    value = context.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        tool_args[key] = float(value)


def _copy_integer_context(context: Mapping[str, Any], tool_args: Dict[str, Any], key: str) -> None:
    value = context.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        tool_args[key] = value


def _copy_boolean_context(context: Mapping[str, Any], tool_args: Dict[str, Any], key: str) -> None:
    value = context.get(key)
    if isinstance(value, bool):
        tool_args[key] = value


def _copy_string_context(context: Mapping[str, Any], tool_args: Dict[str, Any], key: str) -> None:
    value = context.get(key)
    if isinstance(value, str) and value.strip():
        tool_args[key] = value.strip()


def _context_window_argument(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("context_window must be an integer")
    if value < 0:
        raise ValueError("context_window must be non-negative")
    return value


def _context_window(text: str, start: int, end: int, width: int) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    return text[left:right]


def _span_overlaps(span: tuple[int, int], spans: Sequence[tuple[int, int]]) -> bool:
    return any(span[0] < existing[1] and span[1] > existing[0] for existing in spans)


def _safe_reason(context: str, *, unit: str) -> Optional[str]:
    if unit in {"mw", "mva", "mvar", "capacity_value"}:
        return None
    if _SAFE_CONTEXT_RE.search(context):
        return "standard_or_definition_context"
    return None


def _normalize_unit(unit: str) -> str:
    normalized = unit.lower().replace(".", "")
    if normalized == "%":
        return "percent"
    if normalized == "percentage":
        return "percent"
    if normalized == "mvar":
        return "mvar"
    return normalized

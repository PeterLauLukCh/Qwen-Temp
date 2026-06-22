"""Prompt construction for the Mini Grid-Mind LLM layer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .anti_hallucination import ANTI_FABRICATION_PROMPT_RULES
from .memory import StudyMemoryRecord, StudyMemoryStore
from .tools import ToolRegistry


CONNECTION_TYPES = {"load", "solar", "wind", "bess", "hybrid", "synchronous"}
DEFAULT_AGENT_IDENTITY = (
    "You are Mini Grid-Mind, an LLM orchestrator for connection-impact "
    "assessment. You plan power-system analysis, choose tools, inspect their "
    "outputs, and then explain the result in clear engineering language."
)


@dataclass(frozen=True)
class PromptContextHints:
    """Pre-extracted request hints injected into the agent prompt."""

    case_path: Optional[str] = None
    bus: Optional[int] = None
    mw: Optional[float] = None
    connection_type: Optional[str] = None
    is_ibr: Optional[bool] = None
    enable_contingency: Optional[bool] = None
    last_report_status: Optional[str] = None
    mitigations: Tuple[str, ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key in (
            "case_path",
            "bus",
            "mw",
            "connection_type",
            "is_ibr",
            "enable_contingency",
            "last_report_status",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.mitigations:
            result["mitigations"] = list(self.mitigations)
        result.update(self.extra)
        return result


@dataclass(frozen=True)
class PromptBuildResult:
    """Built system prompt plus the records used to build it."""

    system_prompt: str
    context_hints: PromptContextHints
    memory_records: List[StudyMemoryRecord]
    history_messages: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return self.messages_with_history(self.history_messages)

    def messages_with_history(self, history: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(_message_dict(message) for message in history)
        return messages


def build_gridmind_prompt(
    registry: ToolRegistry,
    *,
    history: Sequence[Mapping[str, Any]] = (),
    memory_store: Optional[StudyMemoryStore] = None,
    lessons: Iterable[str] = (),
    context: Optional[Mapping[str, Any]] = None,
    memory_limit: int = 5,
) -> PromptBuildResult:
    """Build the Grid-Mind system prompt for one LLM invocation."""

    if not isinstance(registry, ToolRegistry):
        raise ValueError("registry must be a ToolRegistry")
    history_messages = [_message_dict(message) for message in history]
    hints = extract_context_hints(history_messages, context=context)
    memory_records = select_relevant_memory(
        memory_store,
        hints=hints,
        history=history_messages,
        limit=memory_limit,
    )

    sections = [
        DEFAULT_AGENT_IDENTITY,
        _operational_rules(),
        _tool_policy(),
        _tool_catalog(registry),
        _context_hints_section(hints),
    ]
    lesson_text = _lessons_section(lessons)
    if lesson_text:
        sections.append(lesson_text)
    memory_text = memory_store.build_prompt_context(memory_records) if memory_store else ""
    if memory_text:
        sections.append(memory_text)

    return PromptBuildResult(
        system_prompt="\n\n".join(section for section in sections if section.strip()),
        context_hints=hints,
        memory_records=memory_records,
        history_messages=history_messages,
    )


def build_chat_messages(
    registry: ToolRegistry,
    user_message: str,
    *,
    history: Sequence[Mapping[str, Any]] = (),
    memory_store: Optional[StudyMemoryStore] = None,
    lessons: Iterable[str] = (),
    context: Optional[Mapping[str, Any]] = None,
    memory_limit: int = 5,
) -> PromptBuildResult:
    """Build a prompt result for a history plus the current user message."""

    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("user_message must be a non-empty string")
    combined = [_message_dict(message) for message in history]
    combined.append({"role": "user", "content": user_message.strip()})
    return build_gridmind_prompt(
        registry,
        history=combined,
        memory_store=memory_store,
        lessons=lessons,
        context=context,
        memory_limit=memory_limit,
    )


def extract_context_hints(
    history: Sequence[Mapping[str, Any]] | str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> PromptContextHints:
    """Extract conservative case, bus, MW, and resource hints from text/history."""

    if context is not None and not isinstance(context, Mapping):
        raise ValueError("context must be a mapping when provided")
    context = dict(context or {})
    text = _history_text(history)
    lower = text.lower()

    case_path = _context_str(context, "case_path") or _extract_case_path(text)
    bus = _context_int(context, "bus")
    if bus is None:
        bus = _extract_bus(text)
    mw = _context_float(context, "mw")
    if mw is None:
        mw = _extract_mw(text)
    connection_type = _context_connection_type(context.get("connection_type"))
    if connection_type is None:
        connection_type = _extract_connection_type(lower)
    is_ibr = _context_bool(context, "is_ibr")
    if is_ibr is None and connection_type is not None:
        is_ibr = connection_type in {"solar", "wind", "bess", "hybrid"}
    enable_contingency = _context_bool(context, "enable_contingency")
    if enable_contingency is None and re.search(r"\b(n-1|contingenc|outage)\b", lower):
        enable_contingency = True

    return PromptContextHints(
        case_path=case_path,
        bus=bus,
        mw=mw,
        connection_type=connection_type,
        is_ibr=is_ibr,
        enable_contingency=enable_contingency,
        last_report_status=_context_str(context, "last_report_status"),
        mitigations=_extract_mitigations(lower, context.get("mitigations")),
        extra=_extra_context(context),
    )


def select_relevant_memory(
    memory_store: Optional[StudyMemoryStore],
    *,
    hints: PromptContextHints,
    history: Sequence[Mapping[str, Any]] = (),
    limit: int = 5,
) -> List[StudyMemoryRecord]:
    """Select memory records for the prompt using Grid-Mind's recall priority."""

    if memory_store is None:
        return []
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    records: List[StudyMemoryRecord] = []
    if hints.case_path and hints.bus is not None:
        records.extend(
            memory_store.recall_bus(
                case_path=hints.case_path,
                bus=hints.bus,
                limit=limit,
            )
        )
    if len(records) < limit and hints.case_path:
        records.extend(memory_store.recall_case(case_path=hints.case_path, limit=limit))
    if len(records) < limit and _history_mentions_capacity(history):
        records.extend(
            memory_store.recall_max_capacity(
                case_path=hints.case_path,
                bus=hints.bus,
                connection_type=hints.connection_type,
                limit=limit,
            )
        )
    if len(records) < limit:
        records.extend(memory_store.recent(limit=limit))

    return _dedupe_records(records)[:limit]


def _operational_rules() -> str:
    return "\n".join(
        [
            "Operational rules:",
            "- Think in terms of the Grid-Mind loop: understand the request, identify missing data, choose tools, inspect tool outputs, and decide whether more tool work is needed.",
            "- Use at most five tool-call rounds for one user turn.",
            "- For quantitative grid claims about MW, MVA, MVAr, p.u., percentages, violations, margins, capacity, or approval/rejection, call a solver-backed tool first unless the value is only a published standard or a clearly labeled memory citation.",
            f"- {ANTI_FABRICATION_PROMPT_RULES}",
            "- If the user asks for CIA, interconnection impact, capacity, hosting capability, violations, or contingency behavior and required inputs are missing, ask a concise clarification instead of guessing.",
            "- Required CIA inputs are case_path, bus, p_mw, connection_type, and is_ibr. Required capacity-search inputs are case_path, bus, and connection_type.",
            "- Required transient bus-fault inputs are case_path, bus, fault_start_s, clearing_time_s, and simulation_time_s. Required transient line-trip inputs are case_path, model, device, trip_time_s, and simulation_time_s.",
            "- `clearing_time_s` is the absolute clearing timestamp. If the user says a fault clears after 100 ms and starts at 1.0 s, set `clearing_time_s=1.1`, not `0.1`.",
            "- For integrated M1+M2 assessment, required M1 inputs are case_path, bus, p_mw, connection_type, and is_ibr; required M2 inputs are a supported dynamic case plus a complete disturbance. If M2 is missing, say the integrated result is incomplete.",
            "- M2 can apply a newly added interconnection only as a static PQ load/injection in the selected dynamic case; do not claim detailed machine, inverter, protection, or controller dynamics for the new device.",
            "- IEEE 118 transient stability uses public benchmark RAW+DYR data when available through local override paths, bundled GitHub data, or powerfulcases; label it as public benchmark data, not customer-validated data. If those files are unavailable or fail validation, report the structured tool error instead of guessing.",
            "- For real-data PSS/E, PIF6, PPC, SAV/DYR/DLL, or processed PSSE-output questions, use the frozen real-data tools. Do not reinterpret those requests as IEEE14, pandapower, or ANDES studies.",
            "- TRGC/NG-SA grid-code requirements are real requirements, but validation must come from supported tools. If a TRGC item requires unavailable fault, ride-through, droop, SCR, PSCAD, power-quality, field-test, controller-edit, or new-project capability, inspect the live remote PSS/E scope and state that it is unsupported in the current remote gym.",
            "- Do not assume an unspecified resource is a load. Use load only when the user says load, demand, data center, or similar load-indicative language.",
            "- Distinguish solver results, local memory, and engineering judgment. Specific numbers must come from solver output or a labeled memory entry; qualitative interpretation may use domain knowledge.",
        ]
    )


def _tool_policy() -> str:
    return "\n".join(
        [
            "Tool policy:",
            "- Prefer run_remote_psse_m1m2 for live real PSS/E M1+M2 gym requests over the TCP/IP Windows worker. Use only allowlisted case_id/scenario_type pairs; do not use it for arbitrary new projects, faults, line trips, or controller edits.",
            "- For TRGC/NG-SA live remote-gym prompts, use list_remote_psse_m1m2_cases when the requested requirement is outside the allowlist. Never use no-disturbance baseline as proxy validation for TRGC FRT, HVRT/LVRT, droop, SCR/system-strength, PSCAD, PQ, field-testing, controller, or new-project requirements.",
            "- Prefer list_remote_psse_m1m2_cases when the user asks what live remote PSS/E cases or scenarios are available through the Windows worker.",
            "- Prefer run_real_interconnection_assessment for real-data PIF6/PSS/E interconnection questions where a user asks to add, connect, or assess a proposed solar, wind, BESS, or load project against precomputed frozen PSSE results.",
            "- Prefer list_real_interconnection_actions when the user asks what PIF6 real-data interconnection actions, POC buses, sizes, or disturbances are available.",
            "- Prefer run_real_psse_assessment for frozen real-data PSS/E result questions, including PIF6, PPC, SAV/DYR/DLL, processed PSSE outputs, RMS dynamic results, and real-data benchmark prompts.",
            "- Prefer list_real_psse_cases when the user asks what frozen real-data PSS/E cases are available.",
            "- Prefer run_integrated_assessment when the user asks for M1 and M2 together, an integrated assessment, both steady-state CIA and transient stability in one result, whether an added/connected project can survive/pass a fault or transient event, or EMT/SCR screening as part of interconnection approval.",
            "- Prefer run_cia for a complete interconnection impact request.",
            "- Prefer find_max_capacity for maximum hosting/capacity/headroom questions at a specified bus.",
            "- Prefer run_powerflow or inspect_violations for present-state voltage/thermal questions.",
            "- Prefer run_contingency for N-1 outage screening.",
            "- Prefer run_transient_stability for standalone transient stability, dynamic stability, bus fault, fault-clearing, rotor-angle, generator-speed, line-trip, or time-domain simulation requests.",
            "- Prefer run_emt_screening for standalone EMT/SCR, short-circuit-ratio, grid-strength, or weak-grid screening requests.",
            "- Prefer list_dynamic_cases when the user asks what dynamic cases are available or whether ANDES dynamic data exists.",
            "- Prefer query_network_data only for topology/equipment lookup; it does not provide solved operating-point values.",
            "- Do not call roadmap placeholders unless they are exposed as implemented tools.",
        ]
    )


def _tool_catalog(registry: ToolRegistry) -> str:
    listing = registry.list_tools(include_unimplemented=True)
    lines = ["Available tool catalog:"]
    for tool in listing["tools"]:
        status = "implemented" if tool.get("implemented") else "roadmap"
        lines.append(
            f"- {tool['name']} [{status}, {tool.get('group', 'unknown')}]: {tool['description']}"
        )
    return "\n".join(lines)


def _context_hints_section(hints: PromptContextHints) -> str:
    hint_dict = hints.to_dict()
    if not hint_dict:
        return "Detected context hints: none."
    lines = ["Detected context hints:"]
    for key in sorted(hint_dict):
        lines.append(f"- {key}: {hint_dict[key]}")
    return "\n".join(lines)


def _lessons_section(lessons: Iterable[str]) -> str:
    if isinstance(lessons, str):
        selected = [lessons.strip()] if lessons.strip() else []
    else:
        selected = [
            lesson.strip()
            for lesson in lessons
            if isinstance(lesson, str) and lesson.strip()
        ]
    if not selected:
        return ""
    lines = ["Persistent lessons from prior evaluation failures:"]
    for lesson in selected:
        lines.append(f"- {lesson}")
    return "\n".join(lines)


def _message_dict(message: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(message, Mapping):
        raise ValueError("history messages must be mappings")
    role = message.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("history messages must include a non-empty role")
    return dict(message)


def _history_text(history: Sequence[Mapping[str, Any]] | str) -> str:
    if isinstance(history, str):
        return history
    if not isinstance(history, Sequence):
        raise ValueError("history must be a sequence of messages or a string")
    parts = []
    for message in history:
        msg = _message_dict(message)
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def _history_mentions_capacity(history: Sequence[Mapping[str, Any]]) -> bool:
    lower = _history_text(history).lower()
    return any(term in lower for term in ("capacity", "hosting", "headroom", "can accept"))


def _dedupe_records(records: Iterable[StudyMemoryRecord]) -> List[StudyMemoryRecord]:
    selected: List[StudyMemoryRecord] = []
    seen = set()
    for record in records:
        if record.record_id in seen:
            continue
        seen.add(record.record_id)
        selected.append(record)
    return selected


_BUS_RE = re.compile(r"\bbus\s*(?:#|number|no\.?|id|:)?\s*(?P<bus>\d+)\b", re.I)
_CASE_PATTERNS = [
    re.compile(r"\bieee\s*-?\s*(?P<size>14|30|57|118)\b", re.I),
    re.compile(r"\bcase\s*-?\s*(?P<size>14|30|57|118)\b", re.I),
    re.compile(r"\b(?P<name>ieee14|ieee30|ieee57|ieee118|case14|case30|case57|case118)\b", re.I),
]
_MW_RE = re.compile(r"\b(?P<mw>\d+(?:\.\d+)?)\s*(?:mw|megawatt|megawatts)\b", re.I)


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


def _extract_bus(text: str) -> Optional[int]:
    match = _BUS_RE.search(text)
    return int(match.group("bus")) if match else None


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


def _extract_mitigations(lower: str, context_value: Any) -> Tuple[str, ...]:
    values = []
    if isinstance(context_value, Sequence) and not isinstance(context_value, (str, bytes)):
        values.extend(str(item).strip() for item in context_value if str(item).strip())
    if re.search(r"\b(capacitor|shunt)\b", lower):
        values.append("shunt_capacitor")
    if re.search(r"\b(redispatch|generation dispatch)\b", lower):
        values.append("redispatch")
    return tuple(dict.fromkeys(values))


def _context_str(context: Mapping[str, Any], key: str) -> Optional[str]:
    value = context.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _context_int(context: Mapping[str, Any], key: str) -> Optional[int]:
    value = context.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _context_float(context: Mapping[str, Any], key: str) -> Optional[float]:
    value = context.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _context_bool(context: Mapping[str, Any], key: str) -> Optional[bool]:
    value = context.get(key)
    return value if isinstance(value, bool) else None


def _context_connection_type(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in CONNECTION_TYPES else None


def _extra_context(context: Mapping[str, Any]) -> Dict[str, Any]:
    known = {
        "case_path",
        "bus",
        "mw",
        "connection_type",
        "is_ibr",
        "enable_contingency",
        "last_report_status",
        "mitigations",
    }
    return {key: value for key, value in context.items() if key not in known}

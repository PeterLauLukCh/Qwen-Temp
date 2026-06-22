"""Deterministic final-report summaries for Mini Grid-Mind.

Step 14 adds a report-composition layer after tool execution. The LLM can still
write user-facing prose, but the agent also returns a compact solver-grounded
report that is generated directly from tool outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .observations import build_tool_observation


REPORTABLE_TOOLS = {
    "run_powerflow",
    "inspect_violations",
    "run_contingency",
    "run_cia",
    "find_max_capacity",
    "query_network_data",
}


@dataclass(frozen=True)
class DeterministicReport:
    """A compact source-of-truth report generated from tool outputs."""

    available: bool
    source_tool: Optional[str]
    summary_text: str
    facts: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "source_tool": self.source_tool,
            "summary_text": self.summary_text,
            "facts": self.facts,
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
        }


def build_deterministic_report(results: Sequence[Mapping[str, Any]]) -> DeterministicReport:
    """Build a deterministic report from the latest meaningful tool result."""

    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise ValueError("results must be a sequence of mappings")
    normalized = []
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("results must contain mappings")
        normalized.append(dict(result))

    primary = _select_primary_result(normalized)
    if primary is None:
        return DeterministicReport(
            available=False,
            source_tool=None,
            summary_text="No solver-backed tool result is available for a deterministic report.",
            limitations=["no_tool_result"],
        )

    if not primary.get("ok", False):
        return _error_report(primary)

    tool = str(primary.get("tool", "unknown"))
    if tool == "run_powerflow":
        return _powerflow_report(primary)
    if tool == "inspect_violations":
        return _violation_report(primary)
    if tool == "run_contingency":
        return _contingency_report(primary)
    if tool == "run_cia":
        return _cia_report(primary)
    if tool == "find_max_capacity":
        return _capacity_report(primary)
    if tool == "query_network_data":
        return _network_data_report(primary)
    return _generic_report(primary)


def report_text_or_original(
    original_text: str,
    report: Optional[DeterministicReport],
    *,
    use_when_empty: bool = True,
) -> str:
    """Return deterministic report text when the original final text is empty."""

    if not isinstance(original_text, str):
        raise ValueError("original_text must be a string")
    if not isinstance(use_when_empty, bool):
        raise ValueError("use_when_empty must be a boolean")
    if use_when_empty and not original_text.strip() and report is not None and report.available:
        return report.summary_text
    return original_text


def _select_primary_result(results: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    reportable = [result for result in results if str(result.get("tool", "")) in REPORTABLE_TOOLS]
    if not reportable:
        return None
    ok_results = [result for result in reportable if result.get("ok", False)]
    if ok_results:
        return ok_results[-1]
    return reportable[-1]


def _error_report(result: Mapping[str, Any]) -> DeterministicReport:
    tool = str(result.get("tool", "unknown"))
    message = _string(result.get("message") or result.get("error") or result.get("error_type"))
    summary = f"The {tool} tool did not complete successfully."
    if message:
        summary += f" Error: {message}."
    recommended = _string(result.get("recommended_tool"))
    warnings = []
    if recommended:
        warnings.append(f"recommended_tool={recommended}")
    return DeterministicReport(
        available=True,
        source_tool=tool,
        summary_text=summary,
        facts={"tool": tool, "ok": False, "error_type": _string(result.get("error_type"))},
        limitations=["tool_failed"],
        warnings=warnings,
    )


def _powerflow_report(result: Mapping[str, Any]) -> DeterministicReport:
    observation = build_tool_observation(result)
    report = _mapping(observation.get("violation_report"))
    voltage = _mapping(observation.get("voltage"))
    loading = _mapping(observation.get("loading"))
    case = _value(observation.get("case"), "unknown case")
    parts = [
        f"Power-flow report for {case}: converged={_value(observation.get('converged'), 'unknown')}.",
        "Violation status: "
        + _status_counts(report, include_total=True)
        + ".",
    ]
    voltage_text = _range_text(
        voltage.get("min_vm_pu"),
        voltage.get("max_vm_pu"),
        unit="p.u.",
    )
    if voltage_text:
        parts.append(f"Voltage range: {voltage_text}.")
    loading_bits = _loading_bits(loading)
    if loading_bits:
        parts.append("Maximum loading: " + ", ".join(loading_bits) + ".")
    return DeterministicReport(
        available=True,
        source_tool="run_powerflow",
        summary_text=" ".join(parts),
        facts=observation,
    )


def _violation_report(result: Mapping[str, Any]) -> DeterministicReport:
    observation = build_tool_observation(result)
    report = _mapping(observation.get("violation_report"))
    case = _value(observation.get("case"), "unknown case")
    parts = [
        f"Violation-inspection report for {case}: {_status_counts(report, include_total=True)}.",
    ]
    top = _list(observation.get("top_violations"))
    if top:
        parts.append("Top issue: " + _issue_text(top[0]) + ".")
    return DeterministicReport(
        available=True,
        source_tool="inspect_violations",
        summary_text=" ".join(parts),
        facts=observation,
    )


def _contingency_report(result: Mapping[str, Any]) -> DeterministicReport:
    observation = build_tool_observation(result)
    case = _value(observation.get("case"), "unknown case")
    evaluated = _value(observation.get("evaluated_contingencies"), "unknown")
    failed = _value(observation.get("failed_contingencies"), "unknown")
    borderline = _value(observation.get("borderline_contingencies"), "unknown")
    non_converged = _value(observation.get("non_converged_contingencies"), "unknown")
    parts = [
        f"N-1 contingency report for {case}: status={_value(observation.get('status'), 'unknown')}.",
        (
            f"Evaluated {evaluated} contingencies; failed={failed}, "
            f"borderline={borderline}, non_converged={non_converged}."
        ),
    ]
    top = _list(observation.get("top_failed_contingencies"))
    if top:
        parts.append("Top failed contingency: " + _contingency_issue_text(top[0]) + ".")
    return DeterministicReport(
        available=True,
        source_tool="run_contingency",
        summary_text=" ".join(parts),
        facts=observation,
    )


def _cia_report(result: Mapping[str, Any]) -> DeterministicReport:
    observation = build_tool_observation(result)
    connection = _mapping(observation.get("connection"))
    summary = _mapping(observation.get("summary"))
    case = _value(observation.get("case"), "unknown case")
    p_mw = _format_number(connection.get("p_mw"))
    connection_type = _value(connection.get("connection_type"), "connection")
    bus = _value(connection.get("bus"), "unknown bus")
    recommendation = _value(observation.get("recommendation"), "unknown")
    parts = [
        f"CIA report for {p_mw} MW {connection_type} at bus {bus} on {case}: recommendation={recommendation}.",
        (
            "Project-caused violations: "
            f"hard={_value(summary.get('project_hard_violations'), 'unknown')}, "
            f"borderline={_value(summary.get('project_borderline_violations'), 'unknown')}."
        ),
    ]
    stage_text = _stage_status_text(_list(observation.get("stage_statuses")))
    if stage_text:
        parts.append("Stage status: " + stage_text + ".")
    issues = _list(observation.get("top_project_issues"))
    if issues:
        parts.append("Top project issue: " + _project_issue_text(issues[0]) + ".")
    limitations = []
    for stage in _list(observation.get("stage_statuses")):
        if _mapping(stage).get("status") == "not_implemented":
            limitations.append(str(_mapping(stage).get("stage", "not_implemented_stage")))
    if observation.get("complete") is False:
        limitations.append("incomplete_assessment")
    return DeterministicReport(
        available=True,
        source_tool="run_cia",
        summary_text=" ".join(parts),
        facts=observation,
        limitations=limitations,
    )


def _capacity_report(result: Mapping[str, Any]) -> DeterministicReport:
    observation = build_tool_observation(result)
    request = _mapping(observation.get("request"))
    case = _value(observation.get("case"), "unknown case")
    bus = _value(request.get("bus"), "unknown bus")
    connection_type = _value(request.get("connection_type"), "connection")
    max_mw = observation.get("max_approved_mw")
    max_text = "none" if max_mw is None else f"{_format_number(max_mw)} MW"
    parts = [
        f"Capacity-search report for {connection_type} at bus {bus} on {case}: status={_value(observation.get('status'), 'unknown')}.",
        f"Maximum approved capacity: {max_text}.",
    ]
    bracket = _capacity_bracket_text(observation)
    if bracket:
        parts.append(bracket + ".")
    rejection = _mapping(observation.get("rejection_explanation"))
    if rejection:
        limiting_stage = _optional_text(rejection.get("limiting_stage"))
        status = _optional_text(rejection.get("status"))
        if limiting_stage or status:
            parts.append(
                "First rejection: "
                + ", ".join(part for part in (f"stage={limiting_stage}" if limiting_stage else None, f"status={status}" if status else None) if part)
                + "."
            )
    return DeterministicReport(
        available=True,
        source_tool="find_max_capacity",
        summary_text=" ".join(parts),
        facts=observation,
    )


def _network_data_report(result: Mapping[str, Any]) -> DeterministicReport:
    observation = build_tool_observation(result)
    case = _value(observation.get("case"), "unknown case")
    table_counts = _mapping(observation.get("table_counts"))
    count_bits = []
    for name in sorted(table_counts):
        payload = _mapping(table_counts[name])
        total = payload.get("total_rows")
        if total is not None:
            count_bits.append(f"{name}={total}")
    parts = [f"Network-data report for {case}."]
    if count_bits:
        parts.append("Table rows: " + ", ".join(count_bits) + ".")
    return DeterministicReport(
        available=True,
        source_tool="query_network_data",
        summary_text=" ".join(parts),
        facts=observation,
        limitations=["topology_only_no_solved_operating_point"],
    )


def _generic_report(result: Mapping[str, Any]) -> DeterministicReport:
    tool = str(result.get("tool", "unknown"))
    observation = build_tool_observation(result)
    return DeterministicReport(
        available=True,
        source_tool=tool,
        summary_text=f"Tool report for {tool}: completed successfully.",
        facts=observation,
    )


def _status_counts(report: Mapping[str, Any], *, include_total: bool) -> str:
    parts = [f"status={_value(report.get('status'), 'unknown')}"]
    if include_total:
        parts.append(f"total={_value(report.get('total_violations'), 'unknown')}")
    parts.append(f"hard={_value(report.get('hard_count'), 'unknown')}")
    parts.append(f"borderline={_value(report.get('borderline_count'), 'unknown')}")
    return ", ".join(parts)


def _range_text(min_value: Any, max_value: Any, *, unit: str) -> str:
    if min_value is None and max_value is None:
        return ""
    return f"{_format_number(min_value)}-{_format_number(max_value)} {unit}"


def _loading_bits(loading: Mapping[str, Any]) -> List[str]:
    bits = []
    line = loading.get("max_line_loading_percent")
    if line is not None:
        bits.append(f"line={_format_number(line)}%")
    trafo = loading.get("max_trafo_loading_percent")
    if trafo is not None:
        bits.append(f"transformer={_format_number(trafo)}%")
    return bits


def _stage_status_text(stages: Sequence[Any]) -> str:
    bits = []
    for stage in stages:
        payload = _mapping(stage)
        name = payload.get("stage")
        status = payload.get("status")
        if name and status:
            bits.append(f"{name}={status}")
    return ", ".join(bits)


def _issue_text(issue: Mapping[str, Any]) -> str:
    element_type = _value(issue.get("element_type"), "element")
    element_index = _value(issue.get("element_index"), "unknown")
    violation_type = _value(issue.get("violation_type"), "violation")
    severity = _value(issue.get("severity"), "unknown")
    observed = _format_number(issue.get("observed_value"))
    unit = _value(issue.get("unit"), "")
    return f"{severity} {violation_type} on {element_type} {element_index}, observed={observed}{unit}"


def _contingency_issue_text(issue: Mapping[str, Any]) -> str:
    payload = _mapping(issue)
    contingency = _mapping(payload.get("contingency"))
    if not contingency and "contingency_key" in payload:
        return str(payload["contingency_key"])
    element_type = _value(contingency.get("element_type"), "element")
    element_index = _value(contingency.get("element_index"), "unknown")
    failure_type = _value(payload.get("failure_type"), payload.get("status") or "failure")
    return f"{element_type} {element_index}, failure={failure_type}"


def _project_issue_text(issue: Mapping[str, Any]) -> str:
    issue_type = _value(issue.get("issue_type"), "project_issue")
    inner = _mapping(issue.get("issue"))
    if "post_violation" in inner:
        return issue_type + ": " + _issue_text(_mapping(inner.get("post_violation")))
    if "post_failure" in inner:
        return issue_type + ": " + _contingency_issue_text(_mapping(inner.get("post_failure")))
    change_type = _optional_text(inner.get("change_type"))
    return issue_type if change_type is None else f"{issue_type}: {change_type}"


def _capacity_bracket_text(observation: Mapping[str, Any]) -> str:
    bits = []
    lower = observation.get("lower_bound_mw")
    upper = observation.get("upper_bound_mw")
    tolerance = observation.get("tolerance_mw")
    if lower is not None:
        bits.append(f"lower_bound={_format_number(lower)} MW")
    if upper is not None:
        bits.append(f"upper_bound={_format_number(upper)} MW")
    if tolerance is not None:
        bits.append(f"tolerance={_format_number(tolerance)} MW")
    return "Boundary: " + ", ".join(bits) if bits else ""


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _string(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _value(value: Any, default: Any) -> str:
    if value is None:
        return str(default)
    if isinstance(value, float):
        return _format_number(value)
    return str(value)


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return _format_number(value) if isinstance(value, (int, float)) else str(value)


def _format_number(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.6g}"
    if value is None:
        return "unknown"
    return str(value)

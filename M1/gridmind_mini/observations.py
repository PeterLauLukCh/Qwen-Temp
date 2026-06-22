"""Deterministic tool-observation summaries for the Mini Grid-Mind agent.

Step 13 adds a compact observation layer between raw tool outputs and the LLM.
The audit trail keeps full tool results, while the model receives a stable
summary of the key solver facts plus optional raw data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def build_tool_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a compact, deterministic summary from one tool result."""

    if not isinstance(result, Mapping):
        raise ValueError("result must be a mapping")
    tool = str(result.get("tool", "unknown"))
    if not result.get("ok", False):
        return _error_observation(tool, result)
    if tool == "run_powerflow":
        return _powerflow_observation(result)
    if tool == "inspect_violations":
        return _violation_observation(result)
    if tool == "run_contingency":
        return _contingency_observation(result)
    if tool == "run_cia":
        return _cia_observation(result)
    if tool == "find_max_capacity":
        return _capacity_observation(result)
    if tool == "query_network_data":
        return _network_data_observation(result)
    return _generic_observation(tool, result)


def tool_observation_payload(
    result: Mapping[str, Any],
    *,
    include_raw_result: bool = True,
) -> Dict[str, Any]:
    """Return the tool-result payload sent back to the LLM."""

    if not isinstance(include_raw_result, bool):
        raise ValueError("include_raw_result must be a boolean")
    observation = build_tool_observation(result)
    payload = {
        "ok": bool(result.get("ok", False)),
        "tool": str(result.get("tool", observation.get("tool", "unknown"))),
        "observation": observation,
    }
    if include_raw_result:
        payload["raw_result"] = dict(result)
    return payload


def _error_observation(tool: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "tool": tool,
        "status": "error",
        "error_type": _optional_str(result.get("error_type")),
        "message": _optional_str(result.get("message") or result.get("error")),
        "recommended_tool": _optional_str(result.get("recommended_tool")),
        "reason_codes": _string_list(result.get("reason_codes")),
    }


def _powerflow_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    summary = _mapping(result.get("powerflow_summary"))
    bus_summary = _mapping(summary.get("bus_summary"))
    branch_summary = _mapping(summary.get("branch_summary"))
    report = _mapping(result.get("violation_report"))
    case_info = _mapping(result.get("case_info") or summary.get("case"))
    return {
        "tool": "run_powerflow",
        "case": _case_name(result, case_info),
        "converged": _optional_bool(summary.get("converged")),
        "voltage": {
            "min_vm_pu": _optional_number(bus_summary.get("min_vm_pu")),
            "max_vm_pu": _optional_number(bus_summary.get("max_vm_pu")),
        },
        "loading": {
            "max_line_loading_percent": _optional_number(
                branch_summary.get("max_line_loading_percent")
            ),
            "max_trafo_loading_percent": _optional_number(
                branch_summary.get("max_trafo_loading_percent")
            ),
        },
        "violation_report": _violation_report_summary(report),
    }


def _violation_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    report = _mapping(result.get("violation_report"))
    case_info = _mapping(result.get("case_info"))
    return {
        "tool": "inspect_violations",
        "case": _case_name(result, case_info),
        "violation_report": _violation_report_summary(report),
        "top_violations": _limited_violations(report.get("violations"), limit=3),
    }


def _contingency_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    summary = _mapping(result.get("summary"))
    return {
        "tool": "run_contingency",
        "case": _optional_str(result.get("case_path")),
        "status": "fail"
        if _optional_int(summary.get("failed_contingencies"), 0)
        else "pass",
        "evaluated_contingencies": _optional_int(summary.get("evaluated_contingencies")),
        "passed_contingencies": _optional_int(summary.get("passed_contingencies")),
        "failed_contingencies": _optional_int(summary.get("failed_contingencies")),
        "borderline_contingencies": _optional_int(summary.get("borderline_contingencies")),
        "non_converged_contingencies": _optional_int(summary.get("non_converged_contingencies")),
        "hard_violation_count": _optional_int(summary.get("hard_violation_count")),
        "top_failed_contingencies": _limited_items(
            _mapping(result.get("failed_contingencies")).get("items"),
            limit=3,
        ),
    }


def _cia_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    summary = _mapping(result.get("summary"))
    stages = []
    for stage in _list(result.get("stage_reports")):
        if not isinstance(stage, Mapping):
            continue
        stages.append(
            {
                "stage": _optional_str(stage.get("stage")),
                "status": _optional_str(stage.get("status")),
                "passed": _optional_bool(stage.get("passed")),
                "reason_codes": _string_list(stage.get("reason_codes")),
            }
        )
    return {
        "tool": "run_cia",
        "case": _optional_str(result.get("case_path")),
        "connection": _mapping(result.get("connection")),
        "recommendation": _optional_str(result.get("recommendation")),
        "complete": _optional_bool(result.get("complete")),
        "reason_codes": _string_list(result.get("reason_codes")),
        "summary": {
            "baseline_status": _optional_str(summary.get("baseline_status")),
            "post_status": _optional_str(summary.get("post_status")),
            "project_hard_violations": _optional_int(summary.get("project_hard_violations")),
            "project_borderline_violations": _optional_int(
                summary.get("project_borderline_violations")
            ),
            "pre_existing_hard_violations": _optional_int(
                summary.get("pre_existing_hard_violations")
            ),
            "pre_existing_borderline_violations": _optional_int(
                summary.get("pre_existing_borderline_violations")
            ),
        },
        "stage_statuses": stages,
        "top_project_issues": _cia_top_project_issues(result),
    }


def _capacity_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    request = _mapping(result.get("request"))
    rejection = result.get("rejection_explanation")
    return {
        "tool": "find_max_capacity",
        "case": _optional_str(result.get("case_path")),
        "request": request,
        "status": _optional_str(result.get("status")),
        "max_approved_mw": _optional_number(result.get("max_approved_mw")),
        "lower_bound_mw": _optional_number(result.get("lower_bound_mw")),
        "upper_bound_mw": _optional_number(result.get("upper_bound_mw")),
        "tolerance_mw": _optional_number(result.get("tolerance_mw")),
        "iterations": _optional_int(result.get("iterations")),
        "rejection_explanation": _mapping(rejection) if isinstance(rejection, Mapping) else None,
    }


def _network_data_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    data = _mapping(result.get("network_data"))
    tables = _mapping(data.get("tables"))
    table_counts = {}
    for name, payload in tables.items():
        if isinstance(payload, Mapping):
            table_counts[str(name)] = {
                "total_rows": _optional_int(payload.get("total_rows")),
                "truncated_rows": _optional_int(payload.get("truncated_rows")),
            }
    return {
        "tool": "query_network_data",
        "case": _case_name(result, _mapping(data.get("case"))),
        "solved_operating_point_claimed": bool(
            result.get("solved_operating_point_claimed", False)
        ),
        "table_counts": table_counts,
    }


def _generic_observation(tool: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "tool": tool,
        "status": "ok" if result.get("ok", False) else "unknown",
        "available_keys": sorted(str(key) for key in result.keys()),
    }


def _violation_report_summary(report: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "status": _optional_str(report.get("status")),
        "passed": _optional_bool(report.get("passed")),
        "total_violations": _optional_int(report.get("total_violations")),
        "hard_count": _optional_int(report.get("hard_count")),
        "borderline_count": _optional_int(report.get("borderline_count")),
        "truncated_violations": _optional_int(report.get("truncated_violations")),
    }


def _limited_violations(value: Any, *, limit: int) -> List[Dict[str, Any]]:
    items = []
    for item in _list(value)[:limit]:
        if not isinstance(item, Mapping):
            continue
        items.append(
            {
                "element_type": _optional_str(item.get("element_type")),
                "element_index": _optional_int(item.get("element_index")),
                "violation_type": _optional_str(item.get("violation_type")),
                "severity": _optional_str(item.get("severity")),
                "observed_value": _optional_number(item.get("observed_value")),
                "limit_value": _optional_number(item.get("limit_value")),
                "margin_percent": _optional_number(item.get("margin_percent")),
                "unit": _optional_str(item.get("unit")),
            }
        )
    return items


def _cia_top_project_issues(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for stage in _list(result.get("stage_reports")):
        if not isinstance(stage, Mapping):
            continue
        stage_name = _optional_str(stage.get("stage"))
        if stage_name == "f1_steady_state":
            comparison = _mapping(stage.get("project_violation_comparison"))
            issues.extend(
                {
                    "stage": stage_name,
                    "issue_type": "project_caused_violation",
                    "issue": item,
                }
                for item in _limited_items(
                    _mapping(comparison.get("project_caused_violations")).get("items"),
                    limit=3,
                )
            )
        elif stage_name == "f2_n1_contingency":
            comparison = _mapping(stage.get("project_contingency_comparison"))
            issues.extend(
                {
                    "stage": stage_name,
                    "issue_type": "project_caused_contingency_failure",
                    "issue": item,
                }
                for item in _limited_items(
                    _mapping(comparison.get("project_caused_failures")).get("items"),
                    limit=3,
                )
            )
    return issues[:3]


def _limited_items(value: Any, *, limit: int) -> List[Dict[str, Any]]:
    return [dict(item) for item in _list(value)[:limit] if isinstance(item, Mapping)]


def _case_name(result: Mapping[str, Any], case_info: Mapping[str, Any]) -> Optional[str]:
    return _optional_str(result.get("case_path") or case_info.get("case_name"))


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _optional_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _optional_bool(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _optional_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _optional_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default

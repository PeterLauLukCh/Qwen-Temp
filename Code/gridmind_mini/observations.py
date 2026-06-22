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
    if tool == "run_transient_stability":
        return _transient_observation(result)
    if tool == "run_emt_screening":
        return _emt_observation(result)
    if tool == "run_cia":
        return _cia_observation(result)
    if tool == "run_integrated_assessment":
        return _integrated_assessment_observation(result)
    if tool == "list_real_psse_cases":
        return _real_psse_case_list_observation(result)
    if tool == "run_real_psse_assessment":
        return _real_psse_assessment_observation(result)
    if tool == "list_real_interconnection_actions":
        return _real_interconnection_action_list_observation(result)
    if tool == "run_real_interconnection_assessment":
        return _real_interconnection_assessment_observation(result)
    if tool == "list_remote_psse_m1m2_cases":
        return _remote_psse_case_list_observation(result)
    if tool == "run_remote_psse_m1m2":
        return _remote_psse_m1m2_observation(result)
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
    payload = {
        "tool": tool,
        "status": "error",
        "error_type": _optional_str(result.get("error_type")),
        "message": _optional_str(result.get("message") or result.get("error")),
        "recommended_tool": _optional_str(result.get("recommended_tool")),
        "reason_codes": _string_list(result.get("reason_codes")),
    }
    execution = _mapping(result.get("execution"))
    error_lines = _string_list(execution.get("error_lines"))
    if error_lines:
        payload["execution"] = {
            "executor": _optional_str(execution.get("executor")),
            "returncode": _optional_int(execution.get("returncode")),
            "error_lines": error_lines,
        }
    return payload


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


def _transient_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = _mapping(result.get("metrics"))
    stability = _mapping(result.get("stability"))
    disturbance = _mapping(result.get("disturbance"))
    simulation = _mapping(result.get("simulation"))
    case_info = _mapping(result.get("case_info"))
    trajectories = _mapping(result.get("trajectories"))
    connection_model = _mapping(result.get("connection_model"))
    connection_application = _mapping(result.get("connection_application"))
    return {
        "tool": "run_transient_stability",
        "case": _case_name(result, case_info),
        "backend": _optional_str(result.get("backend")),
        "dynamic_interconnection_modeling": _optional_bool(
            result.get("dynamic_interconnection_modeling")
        ),
        "connection_model": connection_model if connection_model else None,
        "connection_application": {
            "requested": _optional_bool(connection_application.get("requested")),
            "applied": _optional_bool(connection_application.get("applied")),
            "mode": _optional_str(connection_application.get("mode")),
            "model": _optional_str(connection_application.get("model")),
            "p0_pu": _optional_number(connection_application.get("p0_pu")),
            "q0_pu": _optional_number(connection_application.get("q0_pu")),
            "system_base_mva": _optional_number(
                connection_application.get("system_base_mva")
            ),
            "error_type": _optional_str(connection_application.get("error_type")),
        },
        "disturbance": {
            "type": _optional_str(disturbance.get("type")),
            "bus": _optional_int(disturbance.get("bus")),
            "model": _optional_str(disturbance.get("model")),
            "device": _optional_str(disturbance.get("device")),
            "fault_start_s": _optional_number(disturbance.get("fault_start_s")),
            "clearing_time_s": _optional_number(disturbance.get("clearing_time_s")),
            "clearing_duration_s": _optional_number(disturbance.get("clearing_duration_s")),
            "trip_time_s": _optional_number(disturbance.get("trip_time_s")),
            "reclose_time_s": _optional_number(disturbance.get("reclose_time_s")),
        },
        "simulation": {
            "simulation_time_s": _optional_number(simulation.get("simulation_time_s")),
            "pflow_converged": _optional_bool(simulation.get("pflow_converged")),
            "tds_converged": _optional_bool(simulation.get("tds_converged")),
            "sample_count": _optional_int(simulation.get("sample_count")),
        },
        "stability": {
            "status": _optional_str(stability.get("status")),
        },
        "metrics": {
            "tds_converged": _optional_bool(metrics.get("tds_converged")),
            "max_angle_spread_rad": _optional_number(metrics.get("max_angle_spread_rad")),
            "final_angle_spread_rad": _optional_number(metrics.get("final_angle_spread_rad")),
            "max_speed_deviation_pu": _optional_number(metrics.get("max_speed_deviation_pu")),
            "min_voltage_pu": _optional_number(metrics.get("min_voltage_pu")),
            "reason_codes": _string_list(metrics.get("reason_codes")),
        },
        "trajectory_keys": sorted(str(key) for key in trajectories.keys()),
    }


def _emt_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = _mapping(result.get("metrics"))
    emt = _mapping(result.get("emt"))
    application = _mapping(result.get("connection_application"))
    poi_bus = _mapping(result.get("poi_bus"))
    case_info = _mapping(result.get("case_info"))
    return {
        "tool": "run_emt_screening",
        "case": _case_name(result, case_info),
        "backend": _optional_str(result.get("backend")),
        "screening_type": _optional_str(result.get("screening_type")),
        "status": _optional_str(emt.get("status")),
        "passed": _optional_bool(emt.get("passed")),
        "criterion": _optional_str(emt.get("criterion")),
        "connection": _mapping(result.get("connection")),
        "poi_bus": {
            "requested_bus": _optional_int(poi_bus.get("requested_bus")),
            "bus_index": _optional_int(poi_bus.get("bus_index")),
        },
        "connection_application": {
            "requested": _optional_bool(application.get("requested")),
            "applied": _optional_bool(application.get("applied")),
            "element_table": _optional_str(application.get("element_table")),
            "element_index": _optional_int(application.get("element_index")),
        },
        "metrics": {
            "scr": _optional_number(metrics.get("scr")),
            "short_circuit_mva": _optional_number(metrics.get("short_circuit_mva")),
            "project_mva": _optional_number(metrics.get("project_mva")),
            "threshold": _optional_number(metrics.get("threshold")),
            "borderline_margin": _optional_number(metrics.get("borderline_margin")),
            "reason_codes": _string_list(metrics.get("reason_codes")),
        },
        "limitations": _string_list(result.get("limitations")),
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


def _integrated_assessment_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
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
                "required_for_approval": _optional_bool(stage.get("required_for_approval")),
                "recommendation": _optional_str(stage.get("recommendation")),
                "case_path": _optional_str(stage.get("case_path")),
                "dynamic_interconnection_modeling": _optional_bool(
                    stage.get("dynamic_interconnection_modeling")
                ),
                "reason_codes": _string_list(stage.get("reason_codes")),
            }
        )
    linkage = _mapping(result.get("linkage"))
    return {
        "tool": "run_integrated_assessment",
        "case": _optional_str(result.get("case_path")),
        "connection": _mapping(result.get("connection")),
        "recommendation": _optional_str(result.get("recommendation")),
        "complete": _optional_bool(result.get("complete")),
        "reason_codes": _string_list(result.get("reason_codes")),
        "summary": {
            "m1_recommendation": _optional_str(summary.get("m1_recommendation")),
            "m2_status": _optional_str(summary.get("m2_status")),
            "m2_stability_status": _optional_str(summary.get("m2_stability_status")),
            "transient_required_for_approval": _optional_bool(
                summary.get("transient_required_for_approval")
            ),
            "emt_status": _optional_str(summary.get("emt_status")),
            "emt_scr": _optional_number(summary.get("emt_scr")),
            "emt_required_for_approval": _optional_bool(
                summary.get("emt_required_for_approval")
            ),
        },
        "stage_statuses": stages,
        "limitations": _string_list(result.get("limitations")),
        "linkage": {
            "m1_models_new_connection": _optional_bool(
                linkage.get("m1_models_new_connection")
            ),
            "m2_models_new_connection": _optional_bool(
                linkage.get("m2_models_new_connection")
            ),
            "emt_models_new_connection": _optional_bool(
                linkage.get("emt_models_new_connection")
            ),
            "m2_linkage": _optional_str(linkage.get("m2_linkage")),
            "emt_linkage": _optional_str(linkage.get("emt_linkage")),
        },
    }


def _real_psse_case_list_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    cases = []
    for case in _list(result.get("cases")):
        if not isinstance(case, Mapping):
            continue
        cases.append(
            {
                "case_id": _optional_str(case.get("case_id")),
                "display_name": _optional_str(case.get("display_name")),
                "available": _optional_bool(case.get("available")),
                "bundle_id": _optional_str(case.get("bundle_id")),
                "dynamic_required_for_completion": _optional_bool(
                    case.get("dynamic_required_for_completion")
                ),
            }
        )
    return {
        "tool": "list_real_psse_cases",
        "backend": _optional_str(result.get("backend")),
        "processed_dir": _optional_str(result.get("processed_dir")),
        "processed_dir_exists": _optional_bool(result.get("processed_dir_exists")),
        "psse_version": _optional_str(result.get("psse_version")),
        "case_count": _optional_int(result.get("case_count")),
        "cases": cases,
        "limitations": _string_list(result.get("limitations")),
    }


def _real_psse_assessment_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    summary = _mapping(result.get("summary"))
    preflight = _mapping(result.get("preflight"))
    stages = []
    for stage in _list(result.get("stage_reports")):
        if not isinstance(stage, Mapping):
            continue
        metrics = _mapping(stage.get("metrics"))
        stages.append(
            {
                "stage": _optional_str(stage.get("stage")),
                "status": _optional_str(stage.get("status")),
                "passed": _optional_bool(stage.get("passed")),
                "required_for_approval": _optional_bool(stage.get("required_for_approval")),
                "event": _optional_str(stage.get("event")),
                "reason_codes": _string_list(stage.get("reason_codes")),
                "poc_p_mw": _optional_number(
                    metrics.get("poc_p_mw", metrics.get("final_poc_p_mw"))
                ),
                "poc_q_mvar": _optional_number(
                    metrics.get("poc_q_mvar", metrics.get("final_poc_q_mvar"))
                ),
                "min_voltage_pu": _optional_number(
                    metrics.get("bus_voltage_min_pu", metrics.get("min_voltage_pu"))
                ),
                "max_voltage_pu": _optional_number(
                    metrics.get("bus_voltage_max_pu", metrics.get("max_voltage_pu"))
                ),
            }
        )
    return {
        "tool": "run_real_psse_assessment",
        "backend": _optional_str(result.get("backend")),
        "case_id": _optional_str(result.get("case_id")),
        "display_name": _optional_str(result.get("display_name")),
        "psse_version": _optional_str(result.get("psse_version")),
        "recommendation": _optional_str(result.get("recommendation")),
        "complete": _optional_bool(result.get("complete")),
        "reason_codes": _string_list(result.get("reason_codes")),
        "summary": {
            "m1_status": _optional_str(summary.get("m1_status")),
            "m1_converged": _optional_bool(summary.get("m1_converged")),
            "m1_bus_count": _optional_int(summary.get("m1_bus_count")),
            "m1_branch_count": _optional_int(summary.get("m1_branch_count")),
            "m1_machine_count": _optional_int(summary.get("m1_machine_count")),
            "m1_load_count": _optional_int(summary.get("m1_load_count")),
            "m1_bus_voltage_min_pu": _optional_number(summary.get("m1_bus_voltage_min_pu")),
            "m1_bus_voltage_max_pu": _optional_number(summary.get("m1_bus_voltage_max_pu")),
            "m1_poc_p_mw": _optional_number(summary.get("m1_poc_p_mw")),
            "m1_poc_q_mvar": _optional_number(summary.get("m1_poc_q_mvar")),
            "m2_status": _optional_str(summary.get("m2_status")),
            "m2_event": _optional_str(summary.get("m2_event")),
            "m2_initialized": _optional_bool(summary.get("m2_initialized")),
            "m2_simulation_converged": _optional_bool(
                summary.get("m2_simulation_converged")
            ),
            "m2_min_voltage_pu": _optional_number(summary.get("m2_min_voltage_pu")),
            "m2_max_voltage_pu": _optional_number(summary.get("m2_max_voltage_pu")),
            "m2_final_poc_p_mw": _optional_number(summary.get("m2_final_poc_p_mw")),
            "m2_final_poc_q_mvar": _optional_number(summary.get("m2_final_poc_q_mvar")),
            "m2_max_oscillation": _optional_number(summary.get("m2_max_oscillation")),
            "emt_status": _optional_str(summary.get("emt_status")),
        },
        "preflight": {
            "psse_version": _optional_str(preflight.get("psse_version")),
            "bundle_id": _optional_str(preflight.get("bundle_id")),
            "dll_load_successful": _optional_bool(preflight.get("dll_load_successful")),
            "dyr_model_counts": _mapping(preflight.get("dyr_model_counts")),
            "dynamic_error_summary": _string_list(preflight.get("dynamic_error_summary")),
        },
        "stage_statuses": stages,
        "limitations": _string_list(result.get("limitations")),
    }


def _remote_psse_case_list_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    cases = []
    for item in _list(result.get("cases")):
        if not isinstance(item, Mapping):
            continue
        cases.append(
            {
                "case_id": _optional_str(item.get("case_id")),
                "display_name": _optional_str(item.get("display_name")),
                "allowed_scenarios": _string_list(item.get("allowed_scenarios")),
                "dynamic_scenarios": _string_list(item.get("dynamic_scenarios")),
            }
        )
    health = _mapping(result.get("health"))
    return {
        "tool": "list_remote_psse_m1m2_cases",
        "backend": _optional_str(result.get("backend")),
        "remote_configured": _optional_bool(result.get("remote_configured")),
        "case_count": _optional_int(result.get("case_count")),
        "cases": cases,
        "health": {
            "checked": bool(result.get("health") is not None),
            "ok": _optional_bool(health.get("ok")),
            "service": _optional_str(health.get("service")),
            "queue_policy": _optional_str(health.get("queue_policy")),
            "error_type": _optional_str(health.get("error_type")),
            "message": _optional_str(health.get("message")),
        },
        "limitations": _string_list(result.get("limitations")),
    }


def _remote_psse_m1m2_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    observation = _real_psse_assessment_observation(result)
    summary = _mapping(result.get("summary"))
    observation["tool"] = "run_remote_psse_m1m2"
    observation["backend"] = _optional_str(result.get("backend"))
    observation["job_id"] = _optional_str(result.get("job_id"))
    observation["scenario_type"] = _optional_str(result.get("scenario_type"))
    observation["summary"].update(
        {
            "m2_final_time_s": _optional_number(summary.get("m2_final_time_s")),
            "m2_channel_row_count": _optional_int(summary.get("m2_channel_row_count")),
            "m2_max_frequency_deviation_pu": _optional_number(
                summary.get("m2_max_frequency_deviation_pu")
            ),
            "m2_max_frequency_deviation_hz_at_60hz": _optional_number(
                summary.get("m2_max_frequency_deviation_hz_at_60hz")
            ),
        }
    )
    execution = _mapping(result.get("execution"))
    observation["execution"] = {
        "executor": _optional_str(execution.get("executor")),
        "returncode": _optional_int(execution.get("returncode")),
        "error_lines": _string_list(execution.get("error_lines")),
    }
    return observation


def _real_interconnection_action_list_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    action_space = _mapping(result.get("action_space"))
    buses = []
    for bus in _list(action_space.get("connection_buses")):
        if not isinstance(bus, Mapping):
            continue
        buses.append(
            {
                "bus": _optional_int(bus.get("bus")),
                "name": _optional_str(bus.get("name")),
                "voltage_pu": _optional_number(bus.get("voltage_pu")),
            }
        )
    return {
        "tool": "list_real_interconnection_actions",
        "backend": _optional_str(result.get("backend")),
        "case_id": _optional_str(result.get("case_id")),
        "catalog_available": _optional_bool(result.get("catalog_available")),
        "index_available": _optional_bool(result.get("index_available")),
        "catalog_scenario_count": _optional_int(result.get("catalog_scenario_count")),
        "precomputed_result_count": _optional_int(result.get("precomputed_result_count")),
        "connection_buses": buses,
        "connection_types": _string_list(action_space.get("connection_types")),
        "project_sizes_mw": [
            value
            for value in (_optional_number(item) for item in _list(action_space.get("project_sizes_mw")))
            if value is not None
        ],
        "q_modes": _string_list(action_space.get("q_modes")),
        "disturbance_count": _optional_int(action_space.get("disturbance_count")),
        "scenario_count_estimate": _optional_int(action_space.get("scenario_count_estimate")),
        "limitations": _string_list(result.get("limitations")),
    }


def _real_interconnection_assessment_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    summary = _mapping(result.get("summary"))
    connection = _mapping(result.get("connection"))
    disturbance = _mapping(result.get("disturbance"))
    stages = []
    for stage in _list(result.get("stage_reports")):
        if not isinstance(stage, Mapping):
            continue
        metrics = _mapping(stage.get("metrics"))
        stages.append(
            {
                "stage": _optional_str(stage.get("stage")),
                "status": _optional_str(stage.get("status")),
                "passed": _optional_bool(stage.get("passed")),
                "required_for_approval": _optional_bool(stage.get("required_for_approval")),
                "min_voltage_pu": _optional_number(
                    metrics.get("min_voltage_pu", metrics.get("bus_voltage_min_pu"))
                ),
                "max_voltage_pu": _optional_number(
                    metrics.get("max_voltage_pu", metrics.get("bus_voltage_max_pu"))
                ),
                "scr": _optional_number(metrics.get("scr")),
            }
        )
    return {
        "tool": "run_real_interconnection_assessment",
        "backend": _optional_str(result.get("backend")),
        "case_id": _optional_str(result.get("case_id")),
        "scenario_id": _optional_str(result.get("scenario_id")),
        "lookup_policy": _optional_str(result.get("lookup_policy")),
        "recommendation": _optional_str(result.get("recommendation")),
        "complete": _optional_bool(result.get("complete")),
        "connection": {
            "bus": _optional_int(connection.get("bus")),
            "connection_type": _optional_str(connection.get("connection_type")),
            "p_mw": _optional_number(connection.get("p_mw")),
            "q_mode": _optional_str(connection.get("q_mode")),
            "q_mvar": _optional_number(connection.get("q_mvar")),
            "bess_mode": _optional_str(connection.get("bess_mode")),
            "p_load_mw": _optional_number(connection.get("p_load_mw")),
        },
        "disturbance": {
            "type": _optional_str(disturbance.get("type")),
            "bus": _optional_int(disturbance.get("bus")),
            "fault_start_s": _optional_number(disturbance.get("fault_start_s")),
            "clearing_time_s": _optional_number(disturbance.get("clearing_time_s")),
            "from_bus": _optional_int(disturbance.get("from_bus")),
            "to_bus": _optional_int(disturbance.get("to_bus")),
            "circuit_id": _optional_str(disturbance.get("circuit_id")),
            "simulation_time_s": _optional_number(disturbance.get("simulation_time_s")),
        },
        "summary": {
            "m1_status": _optional_str(summary.get("m1_status")),
            "m2_status": _optional_str(summary.get("m2_status")),
            "scr_or_emt_status": _optional_str(summary.get("scr_or_emt_status")),
            "m1_converged": _optional_bool(summary.get("m1_converged")),
            "m1_bus_voltage_min_pu": _optional_number(summary.get("m1_bus_voltage_min_pu")),
            "m1_bus_voltage_max_pu": _optional_number(summary.get("m1_bus_voltage_max_pu")),
            "m1_overloaded_branch_count": _optional_int(
                summary.get("m1_overloaded_branch_count")
            ),
            "m2_initialized": _optional_bool(summary.get("m2_initialized")),
            "m2_simulation_converged": _optional_bool(summary.get("m2_simulation_converged")),
            "m2_min_voltage_pu": _optional_number(summary.get("m2_min_voltage_pu")),
            "scr": _optional_number(summary.get("scr")),
        },
        "stage_statuses": stages,
        "limitations": _string_list(result.get("limitations")),
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

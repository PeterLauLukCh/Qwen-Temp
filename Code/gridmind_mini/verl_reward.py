"""Rule reward for verl PowerGym M1+M2+EMT agentic RL."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence

from gridmind_mini.joint_benchmark import JointBenchmarkScenario, evaluate_joint_result
from gridmind_mini.verl_powergrid import (
    EXPECTED_TOOL_NAME,
    audit_tool_arguments,
    argument_score,
    extract_final_answer_text,
    extract_tool_calls_from_rollout,
    safe_float,
    scenario_from_json,
)


FORBIDDEN_TOOLS = {
    "run_powerflow",
    "inspect_violations",
    "run_contingency",
    "run_cia",
    "run_transient_stability",
    "run_emt_screening",
}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> Dict[str, Any]:
    """Compute a scalar reward for one decoded verl rollout.

    Args:
        data_source: Dataset source name.
        solution_str: Decoded model rollout, including tool calls and tool observations.
        ground_truth: Frozen scenario JSON string from the parquet row.
        extra_info: Optional dataset metadata.

    Returns:
        A dict with ``score`` plus logging fields consumed by verl.
    """

    del data_source  # The scenario JSON is the source of truth for this reward.
    scenario = _scenario_from_reward_inputs(ground_truth, extra_info)
    text = solution_str if isinstance(solution_str, str) else str(solution_str or "")
    tool_calls = extract_tool_calls_from_rollout(text)
    final_text = extract_final_answer_text(text)

    expected_calls = [call for call in tool_calls if call.get("name") == EXPECTED_TOOL_NAME]
    forbidden_calls = [call.get("name") for call in tool_calls if call.get("name") in FORBIDDEN_TOOLS]
    best_call = expected_calls[-1] if expected_calls else None
    argument_checks = audit_tool_arguments(
        scenario,
        best_call.get("arguments", {}) if isinstance(best_call, Mapping) else {},
    )
    arg_fraction = argument_score(argument_checks)

    final_checks = _final_answer_checks(scenario, final_text)
    penalties = _penalties(text, final_text, forbidden_calls)

    shaping_score = 0.0
    if tool_calls:
        shaping_score += 0.10
    if best_call is not None:
        shaping_score += 0.18
    shaping_score += 0.32 * arg_fraction
    if not forbidden_calls:
        shaping_score += 0.08
    shaping_score += 0.27 * _fraction(final_checks.values())
    shaping_score -= penalties
    shaping_score = max(0.0, min(1.0, shaping_score))

    failed_argument_paths = [check.path for check in argument_checks if not check.passed]
    failed_final_checks = [name for name, passed in final_checks.items() if not passed]
    benchmark_pass, failed_benchmark_checks = _benchmark_pass(
        scenario=scenario,
        tool_calls=tool_calls,
        final_text=final_text,
    )
    score = 1.0 if benchmark_pass else 0.0
    return {
        "score": score,
        "shaping_score": shaping_score,
        "benchmark_pass": float(benchmark_pass),
        "benchmark_pass_percent": 100.0 if benchmark_pass else 0.0,
        "benchmark_pass_count_per_200": 200.0 if benchmark_pass else 0.0,
        "failed_benchmark_check_count": len(failed_benchmark_checks),
        "failed_benchmark_checks_json": json.dumps(failed_benchmark_checks, sort_keys=True),
        "resolved": float(benchmark_pass),
        "resolved_percent": 100.0 if benchmark_pass else 0.0,
        "resolved_count_per_200": 200.0 if benchmark_pass else 0.0,
        "tool_called": bool(tool_calls),
        "expected_tool_called": best_call is not None,
        "forbidden_tool_called": bool(forbidden_calls),
        "argument_score": arg_fraction,
        "argument_checks_passed": all(check.passed for check in argument_checks),
        "failed_argument_count": len(failed_argument_paths),
        "failed_argument_paths_json": json.dumps(failed_argument_paths, sort_keys=True),
        "final_answer_score": _fraction(final_checks.values()),
        "failed_final_check_count": len(failed_final_checks),
        "failed_final_checks_json": json.dumps(failed_final_checks, sort_keys=True),
        "penalty": penalties,
        "scenario_id": scenario.scenario_id,
    }


def _scenario_from_reward_inputs(ground_truth: Any, extra_info: dict | None) -> JointBenchmarkScenario:
    if ground_truth:
        return scenario_from_json(ground_truth)
    if isinstance(extra_info, Mapping):
        scenario_json = extra_info.get("scenario_json")
        if scenario_json:
            return scenario_from_json(scenario_json)
    raise ValueError("Reward requires scenario JSON in ground_truth")


def _benchmark_pass(
    *,
    scenario: JointBenchmarkScenario,
    tool_calls: Sequence[Mapping[str, Any]],
    final_text: str,
) -> tuple[bool, list[str]]:
    """Return the same pass/fail signal as the joint benchmark evaluator.

    The live benchmark evaluates an ``AgentResult`` with tool records. During
    verl reward scoring we only have the decoded rollout, so this reconstructs
    a minimal compatible object. Correct frozen-oracle tool calls receive the
    full scenario template result; mismatched calls receive an ``ok=False``
    argument-mismatch result. This makes ``benchmark_pass`` mean "would this
    rollout pass the testcase," matching the previous vLLM benchmark count.
    """

    records = []
    for call in tool_calls:
        name = str(call.get("name", ""))
        arguments = call.get("arguments", {})
        if not isinstance(arguments, Mapping):
            arguments = {}
        if name == EXPECTED_TOOL_NAME:
            checks = audit_tool_arguments(scenario, arguments)
            args_ok = all(check.passed for check in checks)
            result = (
                dict(scenario.oracle_result_template)
                if args_ok
                else {
                    "ok": False,
                    "tool": EXPECTED_TOOL_NAME,
                    "error_type": "argument_mismatch",
                    "argument_audit": {
                        "passed": False,
                        "failed": [check.to_dict() for check in checks if not check.passed],
                    },
                }
            )
            records.append(
                SimpleNamespace(
                    name=name,
                    ok=bool(result.get("ok", False)),
                    source="verl_frozen_oracle",
                    arguments=dict(arguments),
                    result=result,
                    error=None if args_ok else "argument_mismatch",
                )
            )
        elif name in FORBIDDEN_TOOLS:
            records.append(
                SimpleNamespace(
                    name=name,
                    ok=True,
                    source="decoded_rollout",
                    arguments=dict(arguments),
                    result={"ok": True, "tool": name},
                    error=None,
                )
            )

    agent_result = SimpleNamespace(
        status="completed",
        output_text=final_text,
        invoked_tools=[record.name for record in records],
        tool_records=records,
        grounding=SimpleNamespace(warning_appended=False),
    )
    checks = evaluate_joint_result(
        scenario,
        agent_result=agent_result,
        oracle_result=dict(scenario.oracle_result_template),
    )
    failed = [check.name for check in checks if not check.passed]
    return not failed, failed


def _final_answer_checks(
    scenario: JointBenchmarkScenario,
    final_text: str,
) -> Dict[str, bool]:
    lower = final_text.lower()
    expected_result = scenario.oracle_result_template
    recommendation = str(expected_result.get("recommendation", "")).lower()
    summary = expected_result.get("summary") if isinstance(expected_result.get("summary"), Mapping) else {}
    emt_requested = "emt" in scenario.oracle_arguments
    expected_emt_status = str(summary.get("emt_status", "")).lower()
    expected_scr = safe_float(summary.get("emt_scr"))

    checks = {
        "has_final_answer": len(final_text.strip()) >= 20,
        "mentions_recommendation": "recommend" in lower,
        "mentions_expected_recommendation": bool(recommendation and recommendation in lower),
        "mentions_m1_or_cia": ("m1" in lower) or ("cia" in lower) or ("steady-state" in lower),
        "mentions_m2_or_transient": ("m2" in lower) or ("transient" in lower) or ("stability" in lower),
        "mentions_grounded_metric": _mentions_grounded_metric(lower),
        "mentions_static_pq_limitation": _mentions_static_pq_limitation(lower),
    }
    if emt_requested:
        checks.update(
            {
                "mentions_emt": ("emt" in lower) or ("scr" in lower) or ("short-circuit ratio" in lower),
                "mentions_emt_status": bool(expected_emt_status and expected_emt_status in lower),
                "mentions_scr_metric": ("scr" in lower)
                and (expected_scr is None or _mentions_number_close_to(final_text, expected_scr)),
            }
        )
    return checks


def _mentions_grounded_metric(lower_text: str) -> bool:
    metric_terms = (
        "scr",
        "short-circuit ratio",
        "angle spread",
        "speed deviation",
        "min voltage",
        "minimum voltage",
        "mva",
        "rad",
        "pu",
    )
    return any(term in lower_text for term in metric_terms)


def _mentions_static_pq_limitation(lower_text: str) -> bool:
    if "static pq" in lower_text or "static-pq" in lower_text or "static injection" in lower_text:
        return True
    if "not" in lower_text and ("inverter" in lower_text or "controller" in lower_text or "waveform" in lower_text):
        return True
    if "emt-v1" in lower_text or "emt v1" in lower_text or "scr proxy" in lower_text:
        return True
    return False


def _mentions_number_close_to(text: str, target: float, *, tolerance: float = 0.05) -> bool:
    for match in re.finditer(r"[-+]?\d+(?:\.\d+)?", text):
        try:
            value = float(match.group(0))
        except ValueError:
            continue
        if abs(value - target) <= tolerance:
            return True
    return False


def _penalties(text: str, final_text: str, forbidden_calls: Sequence[Any]) -> float:
    penalty = 0.0
    if forbidden_calls:
        penalty += 0.20
    if len(text) > 12000:
        penalty += 0.10
    if len(final_text) > 2500:
        penalty += 0.05
    lower = final_text.lower()
    hallucination_patterns = (
        "full inverter model",
        "detailed inverter controller",
        "models inverter controllers",
        "switching waveform",
        "time-domain emt waveform",
        "lvrt model",
        "hvrt model",
        "pll model",
        "protection model is included",
        "customer-validated",
    )
    if any(_unnegated_pattern_present(lower, pattern) for pattern in hallucination_patterns):
        penalty += 0.20
    return penalty


def _fraction(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(1 for item in items if bool(item)) / len(items)


def _unnegated_pattern_present(text: str, pattern: str) -> bool:
    start = text.find(pattern)
    while start >= 0:
        prefix = text[max(0, start - 40) : start]
        if not any(marker in prefix for marker in ("not ", "no ", "does not ", "do not ", "without ")):
            return True
        start = text.find(pattern, start + 1)
    return False


def _debug_score_from_json(solution_str: str, scenario_json: str) -> str:
    """Small helper for manual shell debugging."""

    return json.dumps(
        compute_score(
            data_source="debug",
            solution_str=solution_str,
            ground_truth=scenario_json,
        ),
        indent=2,
        sort_keys=True,
    )

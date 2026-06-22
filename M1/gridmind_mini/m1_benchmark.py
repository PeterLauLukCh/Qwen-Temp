"""M1 live-agent benchmark for Mini Grid-Mind.

This module evaluates the M1 deliverable: a local LLM agent should route
natural-language grid requests to solver-backed tools, parse required fields,
and keep numerical claims grounded in tool outputs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class M1OracleComparison:
    """One path comparison between the agent tool result and deterministic oracle."""

    tool_path: str
    oracle_path: str
    tolerance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_path": self.tool_path,
            "oracle_path": self.oracle_path,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class M1BenchmarkScenario:
    """One natural-language M1 benchmark scenario."""

    scenario_id: str
    user_message: str
    description: str
    expected_statuses: Sequence[str]
    expected_tool: Optional[str] = None
    expected_tool_arguments: Mapping[str, Any] = field(default_factory=dict)
    oracle_tool: Optional[str] = None
    oracle_arguments: Mapping[str, Any] = field(default_factory=dict)
    oracle_comparisons: Sequence[M1OracleComparison] = ()
    output_contains: Sequence[str] = ()
    expect_grounding_warning: bool = False
    context: Mapping[str, Any] = field(default_factory=dict)
    tags: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty string")
        if not isinstance(self.user_message, str) or not self.user_message.strip():
            raise ValueError("user_message must be a non-empty string")
        if not self.expected_statuses:
            raise ValueError("expected_statuses must not be empty")
        if self.oracle_tool is None and self.oracle_arguments:
            raise ValueError("oracle_arguments require oracle_tool")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "user_message": self.user_message,
            "description": self.description,
            "expected_statuses": list(self.expected_statuses),
            "expected_tool": self.expected_tool,
            "expected_tool_arguments": dict(self.expected_tool_arguments),
            "oracle_tool": self.oracle_tool,
            "oracle_arguments": dict(self.oracle_arguments),
            "oracle_comparisons": [
                comparison.to_dict() for comparison in self.oracle_comparisons
            ],
            "output_contains": list(self.output_contains),
            "expect_grounding_warning": self.expect_grounding_warning,
            "context": dict(self.context),
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class M1BenchmarkCheckResult:
    """One benchmark check outcome."""

    name: str
    passed: bool
    actual: Any = None
    expected: Any = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "actual": self.actual,
            "expected": self.expected,
            "message": self.message,
        }


@dataclass(frozen=True)
class M1BenchmarkResult:
    """Complete result for one M1 benchmark scenario."""

    scenario: M1BenchmarkScenario
    agent_result: Any
    oracle_result: Optional[Dict[str, Any]]
    check_results: Sequence[M1BenchmarkCheckResult]
    duration_s: float

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.check_results)

    def to_dict(
        self,
        *,
        include_raw_result: bool = True,
        include_messages: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "scenario": self.scenario.to_dict(),
            "passed": self.passed,
            "duration_s": self.duration_s,
            "check_results": [check.to_dict() for check in self.check_results],
        }
        if include_raw_result:
            payload["agent_result"] = self.agent_result.to_dict(
                include_messages=include_messages
            )
            payload["oracle_result"] = self.oracle_result
        else:
            payload["agent_summary"] = _agent_summary(self.agent_result)
            payload["oracle_summary"] = _oracle_summary(self.oracle_result)
        return payload


@dataclass(frozen=True)
class M1BenchmarkSuiteResult:
    """Aggregate result for an M1 benchmark run."""

    results: Sequence[M1BenchmarkResult]
    duration_s: float

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def to_dict(
        self,
        *,
        include_raw_results: bool = True,
        include_messages: bool = False,
    ) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "duration_s": self.duration_s,
            "results": [
                result.to_dict(
                    include_raw_result=include_raw_results,
                    include_messages=include_messages,
                )
                for result in self.results
            ],
        }


class M1BenchmarkRunner:
    """Run M1 benchmark scenarios through an agent and deterministic oracle."""

    def __init__(self, agent: Any, oracle_registry: Any) -> None:
        if not hasattr(agent, "run_turn"):
            raise ValueError("agent must expose run_turn(message, context=...)")
        if not hasattr(oracle_registry, "call_tool"):
            raise ValueError("oracle_registry must expose call_tool(name, arguments)")
        self.agent = agent
        self.oracle_registry = oracle_registry

    def run_scenario(self, scenario: M1BenchmarkScenario) -> M1BenchmarkResult:
        if not isinstance(scenario, M1BenchmarkScenario):
            raise ValueError("scenario must be an M1BenchmarkScenario")

        start = time.perf_counter()
        oracle_result = self._run_oracle(scenario)
        agent_result = self.agent.run_turn(
            scenario.user_message,
            context=dict(scenario.context) if scenario.context else None,
        )
        checks = evaluate_m1_result(
            scenario,
            agent_result=agent_result,
            oracle_result=oracle_result,
        )
        return M1BenchmarkResult(
            scenario=scenario,
            agent_result=agent_result,
            oracle_result=oracle_result,
            check_results=checks,
            duration_s=time.perf_counter() - start,
        )

    def run_suite(
        self,
        scenarios: Sequence[M1BenchmarkScenario],
    ) -> M1BenchmarkSuiteResult:
        if isinstance(scenarios, (str, bytes)) or not isinstance(scenarios, Sequence):
            raise ValueError("scenarios must be a sequence of M1BenchmarkScenario objects")
        start = time.perf_counter()
        results = [self.run_scenario(scenario) for scenario in scenarios]
        return M1BenchmarkSuiteResult(
            results=results,
            duration_s=time.perf_counter() - start,
        )

    def _run_oracle(self, scenario: M1BenchmarkScenario) -> Optional[Dict[str, Any]]:
        if scenario.oracle_tool is None:
            return None
        try:
            result = self.oracle_registry.call_tool(
                scenario.oracle_tool,
                dict(scenario.oracle_arguments),
            )
        except Exception as exc:
            return {
                "ok": False,
                "tool": scenario.oracle_tool,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        if not isinstance(result, Mapping):
            return {
                "ok": False,
                "tool": scenario.oracle_tool,
                "error_type": "invalid_oracle_result",
                "error": "Oracle tool returned a non-object result.",
            }
        return dict(result)


def default_m1_benchmark_scenarios() -> List[M1BenchmarkScenario]:
    """Return the ten-scenario M1 live-agent benchmark suite."""

    return [
        M1BenchmarkScenario(
            scenario_id="m1_ieee118_powerflow",
            user_message="Run a power flow on IEEE 118 and report voltage or thermal violations.",
            description="Baseline power-flow request; tests solver routing and grounded reporting.",
            expected_statuses=("completed",),
            expected_tool="run_powerflow",
            expected_tool_arguments={"case_path": "ieee118"},
            oracle_tool="run_powerflow",
            oracle_arguments={"case_path": "ieee118", "max_violations": 10},
            oracle_comparisons=(
                M1OracleComparison("tool.powerflow_summary.converged", "oracle.powerflow_summary.converged"),
                M1OracleComparison("tool.violation_report.status", "oracle.violation_report.status"),
            ),
            tags=("m1", "powerflow", "ieee118"),
        ),
        M1BenchmarkScenario(
            scenario_id="m1_ieee118_load_cia",
            user_message="Can we connect a 5 MW load at bus 10 on IEEE 118? Run the connection impact assessment.",
            description="Complete load CIA request on IEEE 118.",
            expected_statuses=("completed",),
            expected_tool="run_cia",
            expected_tool_arguments={
                "case_path": "ieee118",
                "connection.bus": 10,
                "connection.p_mw": 5.0,
                "connection.connection_type": "load",
                "connection.is_ibr": False,
            },
            oracle_tool="run_cia",
            oracle_arguments={
                "case_path": "ieee118",
                "connection": {
                    "bus": 10,
                    "p_mw": 5.0,
                    "connection_type": "load",
                    "is_ibr": False,
                },
                "max_violations": 10,
            },
            oracle_comparisons=(
                M1OracleComparison("tool.recommendation", "oracle.recommendation"),
                M1OracleComparison("tool.summary.project_hard_violations", "oracle.summary.project_hard_violations"),
                M1OracleComparison("tool.stage_reports.0.stage", "oracle.stage_reports.0.stage"),
            ),
            tags=("m1", "cia", "load", "ieee118"),
        ),
        M1BenchmarkScenario(
            scenario_id="m1_ieee14_solar_cia_n1_limited",
            user_message=(
                "Run CIA with N-1 screening for a 5 MW solar project at bus 10 "
                "on IEEE 14, evaluating only the first 2 contingencies."
            ),
            description="Complete IBR CIA request with bounded N-1 screening.",
            expected_statuses=("completed",),
            expected_tool="run_cia",
            expected_tool_arguments={
                "case_path": "ieee14",
                "connection.bus": 10,
                "connection.p_mw": 5.0,
                "connection.connection_type": "solar",
                "connection.is_ibr": True,
                "enable_contingency": True,
                "max_contingencies": 2,
            },
            oracle_tool="run_cia",
            oracle_arguments={
                "case_path": "ieee14",
                "connection": {
                    "bus": 10,
                    "p_mw": 5.0,
                    "connection_type": "solar",
                    "is_ibr": True,
                },
                "enable_contingency": True,
                "max_contingencies": 2,
                "max_violations": 10,
            },
            oracle_comparisons=(
                M1OracleComparison("tool.recommendation", "oracle.recommendation"),
                M1OracleComparison("tool.stage_reports.1.stage", "oracle.stage_reports.1.stage"),
                M1OracleComparison("tool.stage_reports.1.status", "oracle.stage_reports.1.status"),
            ),
            tags=("m1", "cia", "solar", "n1", "ieee14"),
        ),
        M1BenchmarkScenario(
            scenario_id="m1_ieee14_load_capacity",
            user_message="What is the maximum load hosting capacity at bus 10 on IEEE 14?",
            description="Capacity question; tests forced solver routing before the LLM.",
            expected_statuses=("forced_capacity_executed",),
            expected_tool="find_max_capacity",
            expected_tool_arguments={
                "case_path": "ieee14",
                "bus": 10,
                "connection_type": "load",
            },
            oracle_tool="find_max_capacity",
            oracle_arguments={
                "case_path": "ieee14",
                "bus": 10,
                "connection_type": "load",
            },
            oracle_comparisons=(
                M1OracleComparison("tool.status", "oracle.status"),
                M1OracleComparison("tool.max_approved_mw", "oracle.max_approved_mw", tolerance=1e-9),
            ),
            tags=("m1", "capacity", "load", "ieee14"),
        ),
        M1BenchmarkScenario(
            scenario_id="m1_ieee14_violation_inspection",
            user_message=(
                "Inspect voltage and thermal violations on IEEE 14 using the "
                "violation inspector."
            ),
            description="Present-state violation request; tests inspect_violations routing.",
            expected_statuses=("completed",),
            expected_tool="inspect_violations",
            expected_tool_arguments={"case_path": "ieee14"},
            oracle_tool="inspect_violations",
            oracle_arguments={"case_path": "ieee14", "max_violations": 10},
            oracle_comparisons=(
                M1OracleComparison("tool.violation_report.status", "oracle.violation_report.status"),
                M1OracleComparison("tool.violation_report.hard_count", "oracle.violation_report.hard_count"),
            ),
            tags=("m1", "violations", "ieee14"),
        ),
        M1BenchmarkScenario(
            scenario_id="m1_ieee14_contingency_limited",
            user_message=(
                "Run an N-1 contingency screen on IEEE 14 and evaluate only "
                "the first 2 contingencies."
            ),
            description="Standalone bounded N-1 request; tests contingency-tool routing.",
            expected_statuses=("completed",),
            expected_tool="run_contingency",
            expected_tool_arguments={
                "case_path": "ieee14",
                "max_contingencies": 2,
            },
            oracle_tool="run_contingency",
            oracle_arguments={
                "case_path": "ieee14",
                "max_contingencies": 2,
                "max_failed_contingencies": 5,
                "max_violations": 5,
            },
            oracle_comparisons=(
                M1OracleComparison("tool.summary.evaluated_contingencies", "oracle.summary.evaluated_contingencies"),
                M1OracleComparison("tool.summary.failed_contingencies", "oracle.summary.failed_contingencies"),
            ),
            tags=("m1", "contingency", "n1", "ieee14"),
        ),
        M1BenchmarkScenario(
            scenario_id="m1_ieee118_network_topology",
            user_message=(
                "List topology and equipment data for IEEE 118, limit each "
                "table to 5 rows, and do not run a power flow."
            ),
            description="Topology-only request; tests query_network_data routing.",
            expected_statuses=("completed",),
            expected_tool="query_network_data",
            expected_tool_arguments={
                "case_path": "ieee118",
                "max_rows": 5,
            },
            oracle_tool="query_network_data",
            oracle_arguments={"case_path": "ieee118", "max_rows": 5},
            oracle_comparisons=(
                M1OracleComparison("tool.solved_operating_point_claimed", "oracle.solved_operating_point_claimed"),
                M1OracleComparison("tool.network_data.case.case_name", "oracle.network_data.case.case_name"),
            ),
            tags=("m1", "topology", "ieee118"),
        ),
        M1BenchmarkScenario(
            scenario_id="m1_ieee118_wind_cia",
            user_message="Run CIA for a 3 MW wind project at bus 30 on IEEE 118.",
            description="Complete IBR wind CIA request on IEEE 118.",
            expected_statuses=("completed",),
            expected_tool="run_cia",
            expected_tool_arguments={
                "case_path": "ieee118",
                "connection.bus": 30,
                "connection.p_mw": 3.0,
                "connection.connection_type": "wind",
                "connection.is_ibr": True,
            },
            oracle_tool="run_cia",
            oracle_arguments={
                "case_path": "ieee118",
                "connection": {
                    "bus": 30,
                    "p_mw": 3.0,
                    "connection_type": "wind",
                    "is_ibr": True,
                },
                "max_violations": 10,
            },
            oracle_comparisons=(
                M1OracleComparison("tool.recommendation", "oracle.recommendation"),
                M1OracleComparison("tool.stage_reports.0.stage", "oracle.stage_reports.0.stage"),
                M1OracleComparison("tool.summary.project_hard_violations", "oracle.summary.project_hard_violations"),
            ),
            tags=("m1", "cia", "wind", "ieee118"),
        ),
        M1BenchmarkScenario(
            scenario_id="m1_ieee14_bess_cia",
            user_message="Assess a 2 MW BESS project at bus 9 on IEEE 14 with CIA.",
            description="Complete BESS CIA request; tests IBR parsing and run_cia routing.",
            expected_statuses=("completed",),
            expected_tool="run_cia",
            expected_tool_arguments={
                "case_path": "ieee14",
                "connection.bus": 9,
                "connection.p_mw": 2.0,
                "connection.connection_type": "bess",
                "connection.is_ibr": True,
            },
            oracle_tool="run_cia",
            oracle_arguments={
                "case_path": "ieee14",
                "connection": {
                    "bus": 9,
                    "p_mw": 2.0,
                    "connection_type": "bess",
                    "is_ibr": True,
                },
                "max_violations": 10,
            },
            oracle_comparisons=(
                M1OracleComparison("tool.recommendation", "oracle.recommendation"),
                M1OracleComparison("tool.stage_reports.0.stage", "oracle.stage_reports.0.stage"),
                M1OracleComparison("tool.summary.project_hard_violations", "oracle.summary.project_hard_violations"),
            ),
            tags=("m1", "cia", "bess", "ieee14"),
        ),
        M1BenchmarkScenario(
            scenario_id="m1_ieee118_missing_bus_clarification",
            user_message="Can we connect a 20 MW wind project on IEEE 118?",
            description="Incomplete CIA request; tests required-input clarification.",
            expected_statuses=("clarification_required",),
            expected_tool=None,
            output_contains=("bus",),
            tags=("m1", "clarification", "wind", "ieee118"),
        ),
    ]


def filter_m1_scenarios(
    scenarios: Sequence[M1BenchmarkScenario],
    *,
    scenario_ids: Iterable[str] = (),
    tags: Iterable[str] = (),
) -> List[M1BenchmarkScenario]:
    """Filter M1 scenarios by id and/or tag."""

    selected = list(scenarios)
    id_set = {item for item in scenario_ids if item}
    tag_set = {item for item in tags if item}
    if id_set:
        selected = [scenario for scenario in selected if scenario.scenario_id in id_set]
    if tag_set:
        selected = [
            scenario
            for scenario in selected
            if tag_set.intersection({str(tag) for tag in scenario.tags})
        ]
    return selected


def evaluate_m1_result(
    scenario: M1BenchmarkScenario,
    *,
    agent_result: Any,
    oracle_result: Optional[Mapping[str, Any]],
) -> List[M1BenchmarkCheckResult]:
    """Evaluate one live agent result against an M1 benchmark scenario."""

    checks = [
        _check_status(scenario, agent_result),
        _check_grounding(scenario, agent_result),
    ]
    expected_record = _first_record(agent_result, scenario.expected_tool)
    if scenario.expected_tool is None:
        checks.append(_check_no_tool_records(agent_result))
    else:
        checks.append(_check_expected_tool(scenario, agent_result, expected_record))
        checks.append(_check_unexpected_tools(scenario, agent_result))
        checks.extend(_check_expected_arguments(scenario, expected_record))
        checks.extend(_check_oracle(scenario, expected_record, oracle_result))
    checks.extend(_check_output_contains(scenario, agent_result))
    return checks


def run_m1_oracles(
    scenarios: Sequence[M1BenchmarkScenario],
    oracle_registry: Any,
) -> List[Dict[str, Any]]:
    """Run deterministic oracle tools for scenarios that define an oracle."""

    if not hasattr(oracle_registry, "call_tool"):
        raise ValueError("oracle_registry must expose call_tool(name, arguments)")
    outputs = []
    for scenario in scenarios:
        if scenario.oracle_tool is None:
            outputs.append(
                {
                    "scenario": scenario.to_dict(),
                    "oracle_result": None,
                    "ok": True,
                    "note": "Scenario has no oracle tool.",
                }
            )
            continue
        try:
            argument_checks = _oracle_argument_checks(scenario)
            arguments_ok = all(check["passed"] for check in argument_checks)
            result = oracle_registry.call_tool(
                scenario.oracle_tool,
                dict(scenario.oracle_arguments),
            )
            ok = bool(result.get("ok", False)) if isinstance(result, Mapping) else False
            path_checks = (
                _oracle_path_checks(scenario, result)
                if isinstance(result, Mapping)
                else []
            )
            paths_ok = all(check["passed"] for check in path_checks)
            outputs.append(
                {
                    "scenario": scenario.to_dict(),
                    "oracle_result": result,
                    "oracle_argument_checks": argument_checks,
                    "oracle_path_checks": path_checks,
                    "ok": ok and arguments_ok and paths_ok,
                }
            )
        except Exception as exc:
            outputs.append(
                {
                    "scenario": scenario.to_dict(),
                    "oracle_result": {
                        "ok": False,
                        "tool": scenario.oracle_tool,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    "ok": False,
                }
            )
    return outputs


def _oracle_argument_checks(scenario: M1BenchmarkScenario) -> List[Dict[str, Any]]:
    checks = []
    if scenario.expected_tool is not None:
        checks.append(
            {
                "name": "oracle_tool_alignment",
                "passed": scenario.oracle_tool == scenario.expected_tool,
                "actual": scenario.oracle_tool,
                "expected": scenario.expected_tool,
            }
        )
    for path, expected in scenario.expected_tool_arguments.items():
        found, actual = _resolve_path(scenario.oracle_arguments, path)
        checks.append(
            {
                "name": f"oracle_argument:{path}",
                "passed": found and _value_matches(actual, expected, path=path),
                "actual": actual if found else None,
                "expected": expected,
            }
        )
    return checks


def _oracle_path_checks(
    scenario: M1BenchmarkScenario,
    oracle_result: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    checks = []
    context = {"tool": oracle_result, "oracle": oracle_result}
    for comparison in scenario.oracle_comparisons:
        found_tool, _actual = _resolve_path(context, comparison.tool_path)
        found_oracle, _expected = _resolve_path(context, comparison.oracle_path)
        checks.append(
            {
                "name": f"oracle_path:{comparison.tool_path}",
                "passed": found_tool and found_oracle,
                "tool_path": comparison.tool_path,
                "tool_path_found": found_tool,
                "oracle_path": comparison.oracle_path,
                "oracle_path_found": found_oracle,
            }
        )
    return checks


def _check_status(
    scenario: M1BenchmarkScenario,
    agent_result: Any,
) -> M1BenchmarkCheckResult:
    actual = getattr(agent_result, "status", None)
    expected = list(scenario.expected_statuses)
    passed = actual in scenario.expected_statuses
    return M1BenchmarkCheckResult(
        name="status",
        passed=passed,
        actual=actual,
        expected=expected,
        message=None if passed else "Agent status was not in the allowed set.",
    )


def _check_grounding(
    scenario: M1BenchmarkScenario,
    agent_result: Any,
) -> M1BenchmarkCheckResult:
    grounding = getattr(agent_result, "grounding", None)
    actual = getattr(grounding, "warning_appended", None)
    expected = scenario.expect_grounding_warning
    passed = actual == expected
    return M1BenchmarkCheckResult(
        name="grounding_warning",
        passed=passed,
        actual=actual,
        expected=expected,
        message=None if passed else "Unexpected numeric-grounding warning state.",
    )


def _check_no_tool_records(agent_result: Any) -> M1BenchmarkCheckResult:
    records = list(getattr(agent_result, "tool_records", []))
    passed = len(records) == 0
    return M1BenchmarkCheckResult(
        name="no_tool_records",
        passed=passed,
        actual=len(records),
        expected=0,
        message=None if passed else "Scenario expected clarification/abstention without tools.",
    )


def _check_expected_tool(
    scenario: M1BenchmarkScenario,
    agent_result: Any,
    expected_record: Any,
) -> M1BenchmarkCheckResult:
    invoked = list(getattr(agent_result, "invoked_tools", []))
    actual = invoked or [getattr(record, "name", None) for record in getattr(agent_result, "tool_records", [])]
    passed = expected_record is not None and bool(getattr(expected_record, "ok", False))
    return M1BenchmarkCheckResult(
        name="expected_tool",
        passed=passed,
        actual=actual,
        expected=scenario.expected_tool,
        message=None if passed else "Expected solver-backed tool was not executed successfully.",
    )


def _check_unexpected_tools(
    scenario: M1BenchmarkScenario,
    agent_result: Any,
) -> M1BenchmarkCheckResult:
    successful_tools = [
        str(getattr(record, "name", ""))
        for record in getattr(agent_result, "tool_records", [])
        if bool(getattr(record, "ok", False))
    ]
    unexpected = [tool for tool in successful_tools if tool != scenario.expected_tool]
    passed = not unexpected
    return M1BenchmarkCheckResult(
        name="unexpected_tools",
        passed=passed,
        actual=successful_tools,
        expected=[scenario.expected_tool],
        message=None if passed else "Agent executed successful tools outside the expected tool set.",
    )


def _check_expected_arguments(
    scenario: M1BenchmarkScenario,
    expected_record: Any,
) -> List[M1BenchmarkCheckResult]:
    checks = []
    for path, expected in scenario.expected_tool_arguments.items():
        if expected_record is None:
            checks.append(
                M1BenchmarkCheckResult(
                    name=f"argument:{path}",
                    passed=False,
                    actual=None,
                    expected=expected,
                    message="No expected tool record is available.",
                )
            )
            continue
        arguments = getattr(expected_record, "arguments", {})
        found, actual = _resolve_path(arguments, path)
        passed = found and _value_matches(actual, expected, path=path)
        checks.append(
            M1BenchmarkCheckResult(
                name=f"argument:{path}",
                passed=passed,
                actual=actual if found else None,
                expected=expected,
                message=None if passed else "Tool argument did not match expected value.",
            )
        )
    return checks


def _check_oracle(
    scenario: M1BenchmarkScenario,
    expected_record: Any,
    oracle_result: Optional[Mapping[str, Any]],
) -> List[M1BenchmarkCheckResult]:
    if scenario.oracle_tool is None:
        return []
    checks = []
    oracle_ok = bool(isinstance(oracle_result, Mapping) and oracle_result.get("ok", False))
    checks.append(
        M1BenchmarkCheckResult(
            name="oracle_ok",
            passed=oracle_ok,
            actual=None if oracle_result is None else oracle_result.get("ok"),
            expected=True,
            message=None if oracle_ok else "Deterministic oracle did not complete.",
        )
    )
    if expected_record is None:
        return checks
    tool_result = getattr(expected_record, "result", {})
    context = {"tool": tool_result, "oracle": oracle_result or {}}
    for comparison in scenario.oracle_comparisons:
        found_tool, actual = _resolve_path(context, comparison.tool_path)
        found_oracle, expected = _resolve_path(context, comparison.oracle_path)
        passed = found_tool and found_oracle and _value_matches(
            actual,
            expected,
            tolerance=comparison.tolerance,
            path=comparison.tool_path,
        )
        checks.append(
            M1BenchmarkCheckResult(
                name=f"oracle:{comparison.tool_path}",
                passed=passed,
                actual=actual if found_tool else None,
                expected=expected if found_oracle else None,
                message=None if passed else "Agent tool output did not match deterministic oracle.",
            )
        )
    return checks


def _check_output_contains(
    scenario: M1BenchmarkScenario,
    agent_result: Any,
) -> List[M1BenchmarkCheckResult]:
    output_text = str(getattr(agent_result, "output_text", ""))
    lower = output_text.lower()
    checks = []
    for expected_text in scenario.output_contains:
        expected = str(expected_text)
        passed = expected.lower() in lower
        checks.append(
            M1BenchmarkCheckResult(
                name=f"output_contains:{expected}",
                passed=passed,
                actual=output_text,
                expected=expected,
                message=None if passed else "Expected text was not found in the agent output.",
            )
        )
    return checks


def _first_record(agent_result: Any, expected_tool: Optional[str]) -> Any:
    if expected_tool is None:
        return None
    for record in getattr(agent_result, "tool_records", []):
        if getattr(record, "name", None) == expected_tool and bool(getattr(record, "ok", False)):
            return record
    for record in getattr(agent_result, "tool_records", []):
        if getattr(record, "name", None) == expected_tool:
            return record
    return None


def _resolve_path(root: Any, path: str) -> Tuple[bool, Any]:
    current = root
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return False, None
            current = current[part]
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            try:
                index = int(part)
            except ValueError:
                return False, None
            if index < 0 or index >= len(current):
                return False, None
            current = current[index]
            continue
        return False, None
    return True, current


def _value_matches(
    actual: Any,
    expected: Any,
    *,
    tolerance: Optional[float] = None,
    path: str = "",
) -> bool:
    if path.endswith("case_path"):
        return _normalize_case_name(actual) == _normalize_case_name(expected)
    if isinstance(expected, float) or tolerance is not None:
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False
        tol = 1e-9 if tolerance is None else tolerance
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol)
    return actual == expected


def _normalize_case_name(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    key = value.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "14": "ieee14",
        "case14": "ieee14",
        "ieee14": "ieee14",
        "30": "ieee30",
        "case30": "ieee30",
        "ieee30": "ieee30",
        "57": "ieee57",
        "case57": "ieee57",
        "ieee57": "ieee57",
        "118": "ieee118",
        "case118": "ieee118",
        "ieee118": "ieee118",
    }
    return aliases.get(key, key)


def _agent_summary(agent_result: Any) -> Dict[str, Any]:
    deterministic_report = getattr(agent_result, "deterministic_report", None)
    grounding = getattr(agent_result, "grounding", None)
    return {
        "status": getattr(agent_result, "status", None),
        "output_text": getattr(agent_result, "output_text", ""),
        "invoked_tools": list(getattr(agent_result, "invoked_tools", [])),
        "tool_records": [
            {
                "name": getattr(record, "name", None),
                "ok": bool(getattr(record, "ok", False)),
                "source": getattr(record, "source", None),
                "arguments": dict(getattr(record, "arguments", {})),
                "error": getattr(record, "error", None),
            }
            for record in getattr(agent_result, "tool_records", [])
        ],
        "grounding": None if grounding is None else grounding.to_dict(),
        "deterministic_report": None
        if deterministic_report is None
        else deterministic_report.to_dict(),
    }


def _oracle_summary(oracle_result: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if oracle_result is None:
        return None
    summary: Dict[str, Any] = {
        "ok": bool(oracle_result.get("ok", False)),
        "tool": oracle_result.get("tool"),
        "error_type": oracle_result.get("error_type"),
    }
    for key in (
        "recommendation",
        "status",
        "max_approved_mw",
        "case_path",
    ):
        if key in oracle_result:
            summary[key] = oracle_result[key]
    return summary

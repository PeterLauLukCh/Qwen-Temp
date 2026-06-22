"""Experiment harness for Mini Grid-Mind deterministic tool runs.

Step 15 adds a small evaluation runner that can execute named scenarios through
the existing tool registry, check structured expectations, and attach the
deterministic report from Step 14. It intentionally does not require a GPU or
LLM endpoint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .reporting import DeterministicReport, build_deterministic_report


SUPPORTED_EXPECTATION_OPS = {
    "equals",
    "in",
    "exists",
    "not_exists",
    "truthy",
    "falsy",
    "gte",
    "lte",
}


@dataclass(frozen=True)
class ExperimentExpectation:
    """One structured assertion over a scenario result."""

    path: str
    op: str
    value: Any = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("expectation path must be a non-empty string")
        if self.op not in SUPPORTED_EXPECTATION_OPS:
            raise ValueError(
                "Unsupported expectation op. Expected one of: "
                + ", ".join(sorted(SUPPORTED_EXPECTATION_OPS))
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "op": self.op,
            "value": self.value,
            "description": self.description,
        }


@dataclass(frozen=True)
class ExpectationResult:
    """Evaluation result for one expectation."""

    expectation: ExperimentExpectation
    passed: bool
    actual: Any = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expectation": self.expectation.to_dict(),
            "passed": self.passed,
            "actual": self.actual,
            "message": self.message,
        }


@dataclass(frozen=True)
class ExperimentScenario:
    """A deterministic tool scenario for the experiment harness."""

    scenario_id: str
    tool_name: str
    arguments: Dict[str, Any]
    description: str = ""
    tags: Sequence[str] = ()
    user_message: Optional[str] = None
    expectations: Sequence[ExperimentExpectation] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty string")
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if not isinstance(self.arguments, dict):
            raise ValueError("arguments must be a dict")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "description": self.description,
            "tags": list(self.tags),
            "user_message": self.user_message,
            "expectations": [expectation.to_dict() for expectation in self.expectations],
        }


@dataclass(frozen=True)
class ExperimentResult:
    """Complete output for one scenario run."""

    scenario: ExperimentScenario
    tool_result: Dict[str, Any]
    deterministic_report: DeterministicReport
    expectation_results: Sequence[ExpectationResult]
    duration_s: float

    @property
    def passed(self) -> bool:
        return bool(self.tool_result.get("ok", False)) and all(
            result.passed for result in self.expectation_results
        )

    def to_dict(self, *, include_raw_result: bool = True) -> Dict[str, Any]:
        payload = {
            "scenario": self.scenario.to_dict(),
            "passed": self.passed,
            "duration_s": self.duration_s,
            "deterministic_report": self.deterministic_report.to_dict(),
            "expectation_results": [
                result.to_dict() for result in self.expectation_results
            ],
        }
        if include_raw_result:
            payload["tool_result"] = self.tool_result
        else:
            payload["tool_result_summary"] = {
                "ok": bool(self.tool_result.get("ok", False)),
                "tool": self.tool_result.get("tool"),
                "error_type": self.tool_result.get("error_type"),
            }
        return payload


@dataclass(frozen=True)
class ExperimentSuiteResult:
    """Aggregate output for a scenario suite."""

    results: Sequence[ExperimentResult]
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

    def to_dict(self, *, include_raw_results: bool = True) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "duration_s": self.duration_s,
            "results": [
                result.to_dict(include_raw_result=include_raw_results)
                for result in self.results
            ],
        }


class ExperimentRunner:
    """Run deterministic scenarios through a ToolRegistry-like object."""

    def __init__(self, registry: Any) -> None:
        if not hasattr(registry, "call_tool"):
            raise ValueError("registry must expose call_tool(name, arguments)")
        self.registry = registry

    def run_scenario(self, scenario: ExperimentScenario) -> ExperimentResult:
        if not isinstance(scenario, ExperimentScenario):
            raise ValueError("scenario must be an ExperimentScenario")

        start = time.perf_counter()
        try:
            result = self.registry.call_tool(scenario.tool_name, scenario.arguments)
            if not isinstance(result, Mapping):
                result = {
                    "ok": False,
                    "tool": scenario.tool_name,
                    "error_type": "invalid_tool_result",
                    "error": "Tool returned a non-object result.",
                }
            tool_result = dict(result)
        except Exception as exc:
            tool_result = {
                "ok": False,
                "tool": scenario.tool_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        report = build_deterministic_report([tool_result])
        expectation_results = evaluate_expectations(
            scenario.expectations,
            result=tool_result,
            report=report,
            scenario=scenario,
        )
        return ExperimentResult(
            scenario=scenario,
            tool_result=tool_result,
            deterministic_report=report,
            expectation_results=expectation_results,
            duration_s=time.perf_counter() - start,
        )

    def run_suite(self, scenarios: Sequence[ExperimentScenario]) -> ExperimentSuiteResult:
        if isinstance(scenarios, (str, bytes)) or not isinstance(scenarios, Sequence):
            raise ValueError("scenarios must be a sequence of ExperimentScenario objects")
        start = time.perf_counter()
        results = [self.run_scenario(scenario) for scenario in scenarios]
        return ExperimentSuiteResult(results=results, duration_s=time.perf_counter() - start)


def default_experiment_scenarios(
    *,
    case_path: str = "ieee14",
    include_slow: bool = False,
) -> List[ExperimentScenario]:
    """Return built-in deterministic scenarios for local smoke evaluation."""

    case = _case_path(case_path)
    scenarios = [
        ExperimentScenario(
            scenario_id=f"{case}_powerflow",
            tool_name="run_powerflow",
            arguments={
                "case_path": case,
                "max_bus_results": 5,
                "max_branch_results": 5,
                "max_violations": 10,
            },
            description="Run a bounded AC power-flow and violation report.",
            tags=("fast", "steady_state", case),
            user_message=f"Run a power flow on {case} and report violations.",
            expectations=(
                ExperimentExpectation("result.ok", "equals", True),
                ExperimentExpectation("result.tool", "equals", "run_powerflow"),
                ExperimentExpectation("result.powerflow_summary.converged", "equals", True),
                ExperimentExpectation("report.available", "equals", True),
                ExperimentExpectation("report.source_tool", "equals", "run_powerflow"),
            ),
        ),
        ExperimentScenario(
            scenario_id=f"{case}_violation_inspection",
            tool_name="inspect_violations",
            arguments={"case_path": case, "max_violations": 10},
            description="Inspect voltage and thermal violations after AC power flow.",
            tags=("fast", "violations", case),
            user_message=f"Inspect solved violations on {case}.",
            expectations=(
                ExperimentExpectation("result.ok", "equals", True),
                ExperimentExpectation("result.violation_report.status", "in", ["pass", "borderline", "fail"]),
                ExperimentExpectation("report.source_tool", "equals", "inspect_violations"),
            ),
        ),
        ExperimentScenario(
            scenario_id=f"{case}_network_data",
            tool_name="query_network_data",
            arguments={"case_path": case, "max_rows": 5},
            description="Query topology/equipment data without solved operating-point claims.",
            tags=("fast", "topology", case),
            user_message=f"List key topology tables for {case}.",
            expectations=(
                ExperimentExpectation("result.ok", "equals", True),
                ExperimentExpectation("result.solved_operating_point_claimed", "equals", False),
                ExperimentExpectation("report.limitations.0", "equals", "topology_only_no_solved_operating_point"),
            ),
        ),
    ]

    if include_slow:
        scenarios.extend(
            [
                ExperimentScenario(
                    scenario_id=f"{case}_contingency_limited",
                    tool_name="run_contingency",
                    arguments={
                        "case_path": case,
                        "max_contingencies": 2,
                        "max_failed_contingencies": 5,
                        "max_violations": 5,
                    },
                    description="Run a bounded N-1 screening smoke scenario.",
                    tags=("slow", "contingency", case),
                    user_message=f"Run a limited N-1 screen on {case}.",
                    expectations=(
                        ExperimentExpectation("result.ok", "equals", True),
                        ExperimentExpectation("result.summary.evaluated_contingencies", "equals", 2),
                        ExperimentExpectation("report.source_tool", "equals", "run_contingency"),
                    ),
                ),
                ExperimentScenario(
                    scenario_id=f"{case}_cia_small_load",
                    tool_name="run_cia",
                    arguments={
                        "case_path": case,
                        "connection": {
                            "bus": 10,
                            "p_mw": 1.0,
                            "connection_type": "load",
                            "is_ibr": False,
                        },
                        "max_violations": 10,
                    },
                    description="Run a small load interconnection CIA smoke scenario.",
                    tags=("slow", "cia", case),
                    user_message=f"Run CIA for a 1 MW load at bus 10 on {case}.",
                    expectations=(
                        ExperimentExpectation("result.ok", "equals", True),
                        ExperimentExpectation("result.recommendation", "in", ["approve", "borderline", "reject"]),
                        ExperimentExpectation("result.stage_reports.0.stage", "equals", "f1_steady_state"),
                        ExperimentExpectation("report.source_tool", "equals", "run_cia"),
                    ),
                ),
            ]
        )
    return scenarios


def filter_scenarios(
    scenarios: Sequence[ExperimentScenario],
    *,
    scenario_ids: Iterable[str] = (),
    tags: Iterable[str] = (),
) -> List[ExperimentScenario]:
    """Filter scenarios by id and/or tag."""

    selected = list(scenarios)
    id_set = {item for item in scenario_ids if item}
    tag_set = {item for item in tags if item}
    if id_set:
        selected = [scenario for scenario in selected if scenario.scenario_id in id_set]
    if tag_set:
        selected = [
            scenario
            for scenario in selected
            if tag_set.intersection(set(str(tag) for tag in scenario.tags))
        ]
    return selected


def evaluate_expectations(
    expectations: Sequence[ExperimentExpectation],
    *,
    result: Mapping[str, Any],
    report: DeterministicReport,
    scenario: ExperimentScenario,
) -> List[ExpectationResult]:
    """Evaluate all expectations against result/report/scenario objects."""

    context = {
        "result": dict(result),
        "report": report.to_dict(),
        "scenario": scenario.to_dict(),
    }
    return [_evaluate_expectation(expectation, context) for expectation in expectations]


def _evaluate_expectation(
    expectation: ExperimentExpectation,
    context: Mapping[str, Any],
) -> ExpectationResult:
    found, actual = _resolve_path(context, expectation.path)
    op = expectation.op
    expected = expectation.value

    if op == "exists":
        passed = found
    elif op == "not_exists":
        passed = not found
    elif not found:
        return ExpectationResult(
            expectation=expectation,
            passed=False,
            actual=None,
            message=f"Path not found: {expectation.path}",
        )
    elif op == "equals":
        passed = actual == expected
    elif op == "in":
        try:
            passed = actual in expected
        except TypeError:
            passed = False
    elif op == "truthy":
        passed = bool(actual)
    elif op == "falsy":
        passed = not bool(actual)
    elif op == "gte":
        passed = _compare_numeric(actual, expected, ">=")
    elif op == "lte":
        passed = _compare_numeric(actual, expected, "<=")
    else:
        passed = False

    message = None if passed else f"Expected {expectation.path} {op} {expected!r}, got {actual!r}"
    return ExpectationResult(
        expectation=expectation,
        passed=passed,
        actual=actual,
        message=message,
    )


def _resolve_path(context: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = context
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return False, None
            current = current[part]
        elif isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                return False, None
            if index < 0 or index >= len(current):
                return False, None
            current = current[index]
        else:
            return False, None
    return True, current


def _compare_numeric(actual: Any, expected: Any, op: str) -> bool:
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError):
        return False
    if op == ">=":
        return actual_value >= expected_value
    return actual_value <= expected_value


def _case_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("case_path must be a non-empty string")
    return value.strip().lower().replace(" ", "")

"""Benchmark suite for frozen real-data PSS/E integration."""

from __future__ import annotations

import contextlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


FORBIDDEN_REAL_DATA_TOOLS = (
    "run_powerflow",
    "inspect_violations",
    "run_contingency",
    "run_cia",
    "run_integrated_assessment",
    "run_transient_stability",
    "run_emt_screening",
)


@dataclass(frozen=True)
class RealDataExpectedPath:
    """One expected path/value in a real-data tool result."""

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
class RealDataBenchmarkScenario:
    """One real-data PSS/E benchmark prompt."""

    scenario_id: str
    user_message: str
    description: str
    oracle_arguments: Mapping[str, Any]
    expected_paths: Sequence[RealDataExpectedPath]
    expected_tool: str = "run_real_psse_assessment"
    expected_statuses: Sequence[str] = ("completed",)
    output_contains: Sequence[str] = ()
    forbidden_successful_tools: Sequence[str] = FORBIDDEN_REAL_DATA_TOOLS
    tags: Sequence[str] = ("real_data", "psse")
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "user_message": self.user_message,
            "description": self.description,
            "oracle_arguments": dict(self.oracle_arguments),
            "expected_paths": [item.to_dict() for item in self.expected_paths],
            "expected_tool": self.expected_tool,
            "expected_statuses": list(self.expected_statuses),
            "output_contains": list(self.output_contains),
            "forbidden_successful_tools": list(self.forbidden_successful_tools),
            "tags": list(self.tags),
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class RealDataBenchmarkCheckResult:
    """One real-data benchmark check."""

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
class RealDataBenchmarkResult:
    """Complete result for one real-data benchmark scenario."""

    scenario: RealDataBenchmarkScenario
    agent_result: Any
    oracle_result: Optional[Dict[str, Any]]
    check_results: Sequence[RealDataBenchmarkCheckResult]
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
        payload: Dict[str, Any] = {
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
class RealDataBenchmarkSuiteResult:
    """Aggregate real-data benchmark result."""

    results: Sequence[RealDataBenchmarkResult]
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


class RealDataBenchmarkRunner:
    """Run real-data benchmark scenarios through an agent and frozen oracle."""

    def __init__(self, agent: Any, oracle_registry: Any) -> None:
        if not hasattr(agent, "run_turn"):
            raise ValueError("agent must expose run_turn(message, context=...)")
        if not hasattr(oracle_registry, "call_tool"):
            raise ValueError("oracle_registry must expose call_tool(name, arguments)")
        self.agent = agent
        self.oracle_registry = oracle_registry

    def run_scenario(self, scenario: RealDataBenchmarkScenario) -> RealDataBenchmarkResult:
        start = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            oracle_result = self.oracle_registry.call_tool(
                "run_real_psse_assessment",
                scenario.oracle_arguments,
            )
            agent_result = self.agent.run_turn(
                scenario.user_message,
                context=dict(scenario.context) if scenario.context else None,
            )
        checks = evaluate_real_data_result(
            scenario,
            agent_result=agent_result,
            oracle_result=oracle_result,
        )
        return RealDataBenchmarkResult(
            scenario=scenario,
            agent_result=agent_result,
            oracle_result=oracle_result,
            check_results=checks,
            duration_s=time.perf_counter() - start,
        )

    def run_suite(
        self,
        scenarios: Sequence[RealDataBenchmarkScenario],
    ) -> RealDataBenchmarkSuiteResult:
        start = time.perf_counter()
        results = [self.run_scenario(scenario) for scenario in scenarios]
        return RealDataBenchmarkSuiteResult(
            results=results,
            duration_s=time.perf_counter() - start,
        )


def default_real_data_benchmark_scenarios(
    *,
    processed_dir: Optional[str] = None,
) -> List[RealDataBenchmarkScenario]:
    """Return the v0 real-data PSS/E benchmark suite."""

    return [
        _real_scenario(
            scenario_id="real_small_no_disturbance",
            user_message=(
                "Use the frozen real-data PSS/E outputs to assess the small "
                "test_cases_v36 PPC no-disturbance case."
            ),
            description="Small PPC case with M1 and M2 pass.",
            case_id="test_cases_v36_no_disturbance",
            processed_dir=processed_dir,
            expected_paths={
                "recommendation": "approve",
                "complete": True,
                "summary.m1_status": "pass",
                "summary.m2_status": "pass",
            },
            output_contains=("pss/e", "rms", "poc"),
            tags=("real_data", "psse", "small", "m1_m2"),
        ),
        _real_scenario(
            scenario_id="real_small_pq_target",
            user_message=(
                "Check the frozen PSS/E small PPC P/Q target reproduction and "
                "tell me whether the static target was met."
            ),
            description="Small PPC P/Q target reproduction with M2 not run.",
            case_id="test_cases_v36_pq_target",
            processed_dir=processed_dir,
            expected_paths={
                "recommendation": "approve",
                "complete": True,
                "summary.m1_status": "pass",
                "summary.m2_status": "skipped",
            },
            output_contains=("p/q", "dynamic", "not run"),
            tags=("real_data", "psse", "small", "pq_target"),
        ),
        _real_scenario(
            scenario_id="real_pif6_baseline",
            user_message=(
                "Use the real-data PIF6 PSS/E processed result to assess the "
                "786-bus baseline no-disturbance RMS dynamic run."
            ),
            description="PIF6 786-bus baseline with M1 and M2 pass.",
            case_id="pif6_baseline",
            processed_dir=processed_dir,
            expected_paths={
                "recommendation": "approve",
                "complete": True,
                "summary.m1_status": "pass",
                "summary.m2_status": "pass",
                "summary.m1_bus_count": 786,
            },
            output_contains=("pif6", "786", "pss/e", "rms"),
            tags=("real_data", "psse", "pif6", "m1_m2"),
        ),
        _real_scenario(
            scenario_id="real_wrong_tool_trap",
            user_message=(
                "For the PIF6 real-data PSS/E processed files, do not answer "
                "with IEEE14, pandapower, ANDES, or standalone transient tools. "
                "Use the real PSS/E result."
            ),
            description="Real-data prompt must call run_real_psse_assessment.",
            case_id="pif6_baseline",
            processed_dir=processed_dir,
            expected_paths={
                "recommendation": "approve",
                "summary.m2_status": "pass",
            },
            output_contains=("pss/e", "real"),
            tags=("real_data", "psse", "wrong_tool_trap"),
        ),
        _real_scenario(
            scenario_id="real_final_answer_grounded",
            user_message=(
                "Summarize the PIF6 frozen PSS/E result. The final answer must "
                "mention the POC P/Q or voltage, RMS dynamic status, and the "
                "limitation that this is not EMT waveform simulation."
            ),
            description="Final response must be grounded in PSS/E metrics and limitations.",
            case_id="pif6_baseline",
            processed_dir=processed_dir,
            expected_paths={
                "recommendation": "approve",
                "summary.m1_status": "pass",
                "summary.m2_status": "pass",
            },
            output_contains=("pss/e", "rms", "poc", "emt"),
            tags=("real_data", "psse", "grounded_final"),
        ),
    ]


def filter_real_data_scenarios(
    scenarios: Sequence[RealDataBenchmarkScenario],
    *,
    scenario_ids: Sequence[str] = (),
    tags: Sequence[str] = (),
) -> List[RealDataBenchmarkScenario]:
    selected = list(scenarios)
    if scenario_ids:
        requested = set(scenario_ids)
        selected = [scenario for scenario in selected if scenario.scenario_id in requested]
    if tags:
        requested_tags = set(tags)
        selected = [
            scenario
            for scenario in selected
            if requested_tags.issubset(set(scenario.tags))
        ]
    return selected


def run_real_data_oracles(
    scenarios: Sequence[RealDataBenchmarkScenario],
    oracle_registry: Any,
) -> List[Dict[str, Any]]:
    outputs = []
    for scenario in scenarios:
        try:
            result = oracle_registry.call_tool(
                "run_real_psse_assessment",
                scenario.oracle_arguments,
            )
            checks = _tool_path_checks(scenario, result)
            outputs.append(
                {
                    "ok": bool(result.get("ok")) and all(check.passed for check in checks),
                    "scenario_id": scenario.scenario_id,
                    "tool": result.get("tool"),
                    "case_id": result.get("case_id"),
                    "recommendation": result.get("recommendation"),
                    "complete": result.get("complete"),
                    "summary": result.get("summary"),
                    "check_results": [check.to_dict() for check in checks],
                    "raw_result": result,
                }
            )
        except Exception as exc:
            outputs.append(
                {
                    "ok": False,
                    "scenario_id": scenario.scenario_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return outputs


def evaluate_real_data_result(
    scenario: RealDataBenchmarkScenario,
    *,
    agent_result: Any,
    oracle_result: Optional[Mapping[str, Any]],
) -> List[RealDataBenchmarkCheckResult]:
    checks: List[RealDataBenchmarkCheckResult] = []
    status = getattr(agent_result, "status", None)
    checks.append(
        RealDataBenchmarkCheckResult(
            name="agent_status",
            passed=status in scenario.expected_statuses,
            actual=status,
            expected=list(scenario.expected_statuses),
        )
    )
    records = list(getattr(agent_result, "tool_records", []) or [])
    expected_records = [record for record in records if getattr(record, "name", None) == scenario.expected_tool]
    checks.append(
        RealDataBenchmarkCheckResult(
            name="expected_tool",
            passed=bool(expected_records),
            actual=[getattr(record, "name", None) for record in records],
            expected=scenario.expected_tool,
        )
    )
    for forbidden in scenario.forbidden_successful_tools:
        used = [
            record
            for record in records
            if getattr(record, "name", None) == forbidden and getattr(record, "ok", False)
        ]
        checks.append(
            RealDataBenchmarkCheckResult(
                name=f"forbidden_tool:{forbidden}",
                passed=not used,
                actual=bool(used),
                expected=False,
            )
        )
    if expected_records:
        record = expected_records[-1]
        checks.append(
            RealDataBenchmarkCheckResult(
                name="expected_tool_ok",
                passed=bool(getattr(record, "ok", False)),
                actual=getattr(record, "ok", None),
                expected=True,
            )
        )
        checks.extend(_tool_path_checks(scenario, getattr(record, "result", {})))
    elif oracle_result is not None:
        checks.extend(
            RealDataBenchmarkCheckResult(
                name=f"result:{item.path}",
                passed=False,
                expected=item.expected,
                message="expected tool was not called",
            )
            for item in scenario.expected_paths
        )
    text = str(getattr(agent_result, "output_text", "") or "").lower()
    for expected in scenario.output_contains:
        checks.append(
            RealDataBenchmarkCheckResult(
                name=f"output_contains:{expected}",
                passed=str(expected).lower() in text,
                actual=text[:500],
                expected=expected,
            )
        )
    return checks


def _real_scenario(
    *,
    scenario_id: str,
    user_message: str,
    description: str,
    case_id: str,
    expected_paths: Mapping[str, Any],
    processed_dir: Optional[str],
    output_contains: Sequence[str],
    tags: Sequence[str],
) -> RealDataBenchmarkScenario:
    arguments: Dict[str, Any] = {"case_id": case_id}
    context: Dict[str, Any] = {"real_psse_case_id": case_id}
    if processed_dir:
        arguments["processed_dir"] = processed_dir
        context["processed_dir"] = processed_dir
    return RealDataBenchmarkScenario(
        scenario_id=scenario_id,
        user_message=user_message,
        description=description,
        oracle_arguments=arguments,
        expected_paths=[
            RealDataExpectedPath(path=path, expected=value)
            for path, value in expected_paths.items()
        ],
        output_contains=output_contains,
        tags=tags,
        context=context,
    )


def _tool_path_checks(
    scenario: RealDataBenchmarkScenario,
    result: Mapping[str, Any],
) -> List[RealDataBenchmarkCheckResult]:
    checks = []
    for expected in scenario.expected_paths:
        actual = _path_get(result, expected.path)
        checks.append(
            RealDataBenchmarkCheckResult(
                name=f"result:{expected.path}",
                passed=_value_matches(actual, expected.expected, expected.tolerance),
                actual=actual,
                expected=expected.expected,
            )
        )
    return checks


def _path_get(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            return None
    return current


def _value_matches(actual: Any, expected: Any, tolerance: Optional[float]) -> bool:
    if tolerance is not None and isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= tolerance
    return actual == expected


def _agent_summary(agent_result: Any) -> Dict[str, Any]:
    records = list(getattr(agent_result, "tool_records", []) or [])
    return {
        "status": getattr(agent_result, "status", None),
        "output_text": getattr(agent_result, "output_text", None),
        "tool_records": [
            {
                "name": getattr(record, "name", None),
                "ok": getattr(record, "ok", None),
                "arguments": getattr(record, "arguments", None),
                "result": {
                    "tool": _path_get(getattr(record, "result", {}), "tool"),
                    "case_id": _path_get(getattr(record, "result", {}), "case_id"),
                    "recommendation": _path_get(getattr(record, "result", {}), "recommendation"),
                    "complete": _path_get(getattr(record, "result", {}), "complete"),
                    "summary": _path_get(getattr(record, "result", {}), "summary"),
                },
            }
            for record in records
        ],
    }


def _oracle_summary(oracle_result: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if oracle_result is None:
        return None
    return {
        "ok": oracle_result.get("ok"),
        "tool": oracle_result.get("tool"),
        "case_id": oracle_result.get("case_id"),
        "recommendation": oracle_result.get("recommendation"),
        "complete": oracle_result.get("complete"),
        "summary": oracle_result.get("summary"),
    }

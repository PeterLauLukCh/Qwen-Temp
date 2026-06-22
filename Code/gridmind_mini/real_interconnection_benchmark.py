"""Benchmark suite for frozen real-data PSS/E interconnection replay."""

from __future__ import annotations

import contextlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


FORBIDDEN_REAL_INTERCONNECTION_TOOLS = (
    "run_powerflow",
    "inspect_violations",
    "run_contingency",
    "run_cia",
    "run_integrated_assessment",
    "run_transient_stability",
    "run_emt_screening",
    "run_real_psse_assessment",
)


@dataclass(frozen=True)
class RealInterconnectionExpectedPath:
    """One expected path/value in a frozen interconnection tool result."""

    path: str
    expected: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "expected": self.expected}


@dataclass(frozen=True)
class RealInterconnectionBenchmarkScenario:
    """One real-data interconnection benchmark prompt."""

    scenario_id: str
    user_message: str
    description: str
    oracle_arguments: Mapping[str, Any]
    expected_paths: Sequence[RealInterconnectionExpectedPath]
    expected_tool: str = "run_real_interconnection_assessment"
    expected_statuses: Sequence[str] = ("completed",)
    output_contains: Sequence[str] = ()
    forbidden_successful_tools: Sequence[str] = FORBIDDEN_REAL_INTERCONNECTION_TOOLS
    tags: Sequence[str] = ("real_interconnection", "psse", "pif6")
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
class RealInterconnectionBenchmarkCheckResult:
    """One benchmark check."""

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
class RealInterconnectionBenchmarkResult:
    """Complete result for one scenario."""

    scenario: RealInterconnectionBenchmarkScenario
    agent_result: Any
    oracle_result: Optional[Dict[str, Any]]
    check_results: Sequence[RealInterconnectionBenchmarkCheckResult]
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
class RealInterconnectionBenchmarkSuiteResult:
    """Aggregate benchmark result."""

    results: Sequence[RealInterconnectionBenchmarkResult]
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


class RealInterconnectionBenchmarkRunner:
    """Run frozen real interconnection benchmark scenarios."""

    def __init__(self, agent: Any, oracle_registry: Any) -> None:
        if not hasattr(agent, "run_turn"):
            raise ValueError("agent must expose run_turn(message, context=...)")
        if not hasattr(oracle_registry, "call_tool"):
            raise ValueError("oracle_registry must expose call_tool(name, arguments)")
        self.agent = agent
        self.oracle_registry = oracle_registry

    def run_scenario(
        self,
        scenario: RealInterconnectionBenchmarkScenario,
    ) -> RealInterconnectionBenchmarkResult:
        start = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            oracle_result = self.oracle_registry.call_tool(
                "run_real_interconnection_assessment",
                scenario.oracle_arguments,
            )
            agent_result = self.agent.run_turn(
                scenario.user_message,
                context=dict(scenario.context) if scenario.context else None,
            )
        checks = evaluate_real_interconnection_result(
            scenario,
            agent_result=agent_result,
            oracle_result=oracle_result,
        )
        return RealInterconnectionBenchmarkResult(
            scenario=scenario,
            agent_result=agent_result,
            oracle_result=oracle_result,
            check_results=checks,
            duration_s=time.perf_counter() - start,
        )

    def run_suite(
        self,
        scenarios: Sequence[RealInterconnectionBenchmarkScenario],
    ) -> RealInterconnectionBenchmarkSuiteResult:
        start = time.perf_counter()
        return RealInterconnectionBenchmarkSuiteResult(
            results=[self.run_scenario(scenario) for scenario in scenarios],
            duration_s=time.perf_counter() - start,
        )


def default_real_interconnection_benchmark_scenarios(
    *,
    processed_dir: Optional[str] = None,
) -> List[RealInterconnectionBenchmarkScenario]:
    """Return the initial PIF6 frozen interconnection benchmark suite."""

    return [
        _scenario(
            scenario_id="real_pif6_solar_poc2_fault_100ms",
            user_message=(
                "Using the frozen PSS/E PIF6 interconnection gym, can we add "
                "5 MW solar at POC2 bus 2 and survive a bus fault at bus 2 "
                "starting at 1.0 s and clearing at 1.10 s?"
            ),
            description="PIF6 solar static-PQ exact lookup with 100 ms fault.",
            processed_dir=processed_dir,
            connection={"bus": 2, "connection_type": "solar_static_pq", "p_mw": 5.0},
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.10,
                "simulation_time_s": 5.0,
            },
            output_contains=("pss/e", "pif6", "recommendation", "rms"),
        ),
        _scenario(
            scenario_id="real_pif6_load_poc2_no_disturbance",
            user_message=(
                "Check the frozen PSS/E PIF6 result for adding a 2 MW load at "
                "bus 2 with no disturbance for 5 seconds."
            ),
            description="PIF6 load static-PQ exact lookup with no disturbance.",
            processed_dir=processed_dir,
            connection={"bus": 2, "connection_type": "load_static_pq", "p_mw": 2.0},
            disturbance={"type": "no_disturbance", "simulation_time_s": 5.0},
            output_contains=("pss/e", "load", "recommendation"),
        ),
        _scenario(
            scenario_id="real_pif6_bess_poc2_fault_80ms",
            user_message=(
                "For real-data PIF6, assess a 1 MW BESS discharging at bus 2 "
                "under a bus 2 fault from 1.0 s to 1.08 s."
            ),
            description="PIF6 BESS discharge exact lookup with 80 ms fault.",
            processed_dir=processed_dir,
            connection={
                "bus": 2,
                "connection_type": "bess_static_pq",
                "p_mw": 1.0,
                "bess_mode": "discharge",
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.08,
                "simulation_time_s": 5.0,
            },
            output_contains=("bess", "rms", "recommendation"),
        ),
        _scenario(
            scenario_id="real_pif6_wind_line_trip",
            user_message=(
                "Use the frozen PIF6 PSS/E gym to evaluate adding 10 MW wind "
                "at bus 2 with a line trip on branch 2-2001 circuit 1 at 1.0 s."
            ),
            description="PIF6 wind exact lookup with POC branch line trip.",
            processed_dir=processed_dir,
            connection={"bus": 2, "connection_type": "wind_static_pq", "p_mw": 10.0},
            disturbance={
                "type": "line_trip",
                "from_bus": 2,
                "to_bus": 2001,
                "circuit_id": "1",
                "trip_time_s": 1.0,
                "simulation_time_s": 5.0,
            },
            output_contains=("wind", "line", "recommendation"),
        ),
        _scenario(
            scenario_id="real_pif6_wrong_tool_trap",
            user_message=(
                "This is a real PSS/E PIF6 interconnection question: add "
                "5 MW solar at bus 2 and test a 100 ms bus fault. Do not use "
                "IEEE14, ANDES, pandapower, or the baseline PSS/E replay tool."
            ),
            description="Must call run_real_interconnection_assessment.",
            processed_dir=processed_dir,
            connection={"bus": 2, "connection_type": "solar_static_pq", "p_mw": 5.0},
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.10,
                "simulation_time_s": 5.0,
            },
            output_contains=("real", "pss/e"),
            tags=("real_interconnection", "psse", "pif6", "wrong_tool_trap"),
        ),
    ]


def filter_real_interconnection_scenarios(
    scenarios: Sequence[RealInterconnectionBenchmarkScenario],
    *,
    scenario_ids: Sequence[str] = (),
    tags: Sequence[str] = (),
) -> List[RealInterconnectionBenchmarkScenario]:
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


def run_real_interconnection_oracles(
    scenarios: Sequence[RealInterconnectionBenchmarkScenario],
    oracle_registry: Any,
) -> List[Dict[str, Any]]:
    outputs = []
    for scenario in scenarios:
        try:
            result = oracle_registry.call_tool(
                "run_real_interconnection_assessment",
                scenario.oracle_arguments,
            )
            checks = _tool_path_checks(scenario, result)
            outputs.append(
                {
                    "ok": bool(result.get("ok")) and all(check.passed for check in checks),
                    "scenario_id": scenario.scenario_id,
                    "tool": result.get("tool"),
                    "case_id": result.get("case_id"),
                    "scenario_result_id": result.get("scenario_id"),
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


def evaluate_real_interconnection_result(
    scenario: RealInterconnectionBenchmarkScenario,
    *,
    agent_result: Any,
    oracle_result: Optional[Mapping[str, Any]],
) -> List[RealInterconnectionBenchmarkCheckResult]:
    checks: List[RealInterconnectionBenchmarkCheckResult] = []
    status = getattr(agent_result, "status", None)
    checks.append(
        RealInterconnectionBenchmarkCheckResult(
            name="agent_status",
            passed=status in scenario.expected_statuses,
            actual=status,
            expected=list(scenario.expected_statuses),
        )
    )
    records = list(getattr(agent_result, "tool_records", []) or [])
    expected_records = [
        record for record in records if getattr(record, "name", None) == scenario.expected_tool
    ]
    checks.append(
        RealInterconnectionBenchmarkCheckResult(
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
            RealInterconnectionBenchmarkCheckResult(
                name=f"forbidden_tool:{forbidden}",
                passed=not used,
                actual=bool(used),
                expected=False,
            )
        )
    if expected_records:
        record = expected_records[-1]
        checks.append(
            RealInterconnectionBenchmarkCheckResult(
                name="expected_tool_ok",
                passed=bool(getattr(record, "ok", False)),
                actual=getattr(record, "ok", None),
                expected=True,
            )
        )
        checks.extend(_tool_path_checks(scenario, getattr(record, "result", {})))
    elif oracle_result is not None:
        checks.extend(
            RealInterconnectionBenchmarkCheckResult(
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
            RealInterconnectionBenchmarkCheckResult(
                name=f"output_contains:{expected}",
                passed=str(expected).lower() in text,
                actual=text[:500],
                expected=expected,
            )
        )
    return checks


def _scenario(
    *,
    scenario_id: str,
    user_message: str,
    description: str,
    processed_dir: Optional[str],
    connection: Mapping[str, Any],
    disturbance: Mapping[str, Any],
    output_contains: Sequence[str],
    tags: Sequence[str] = ("real_interconnection", "psse", "pif6"),
) -> RealInterconnectionBenchmarkScenario:
    arguments: Dict[str, Any] = {
        "case_id": "pif6_2026_05_17",
        "connection": dict(connection),
        "disturbance": dict(disturbance),
    }
    context: Dict[str, Any] = {
        "real_interconnection_case_id": "pif6_2026_05_17",
        "case_path": "pif6_2026_05_17",
    }
    if processed_dir:
        arguments["processed_dir"] = processed_dir
        context["processed_dir"] = processed_dir
    expected_paths = {
        "tool": "run_real_interconnection_assessment",
        "case_id": "pif6_2026_05_17",
        "ok": True,
        "connection.bus": connection["bus"],
        "disturbance.type": "no_disturbance"
        if disturbance["type"] == "no_disturbance_5s"
        else disturbance["type"],
    }
    return RealInterconnectionBenchmarkScenario(
        scenario_id=scenario_id,
        user_message=user_message,
        description=description,
        oracle_arguments=arguments,
        expected_paths=[
            RealInterconnectionExpectedPath(path=path, expected=value)
            for path, value in expected_paths.items()
        ],
        output_contains=output_contains,
        tags=tags,
        context=context,
    )


def _tool_path_checks(
    scenario: RealInterconnectionBenchmarkScenario,
    result: Mapping[str, Any],
) -> List[RealInterconnectionBenchmarkCheckResult]:
    checks = []
    for expected in scenario.expected_paths:
        actual = _path_get(result, expected.path)
        checks.append(
            RealInterconnectionBenchmarkCheckResult(
                name=f"result:{expected.path}",
                passed=actual == expected.expected,
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
                    "scenario_id": _path_get(getattr(record, "result", {}), "scenario_id"),
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
        "scenario_id": oracle_result.get("scenario_id"),
        "recommendation": oracle_result.get("recommendation"),
        "complete": oracle_result.get("complete"),
        "summary": oracle_result.get("summary"),
    }

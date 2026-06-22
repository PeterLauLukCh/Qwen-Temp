"""M2 transient-stability benchmark for Mini Grid-Mind.

The M2 suite evaluates whether the agent recognizes transient-stability
requests, supplies the ANDES tool with the right structured disturbance, asks
for clarification when essential disturbance data is missing, handles dynamic
data limitations, and uses the static interconnection model when requested.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class M2OracleComparison:
    """One path comparison between agent output and an oracle/template result."""

    tool_path: str
    oracle_path: str
    tolerance: Optional[float] = None
    compare_value: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_path": self.tool_path,
            "oracle_path": self.oracle_path,
            "tolerance": self.tolerance,
            "compare_value": self.compare_value,
        }


@dataclass(frozen=True)
class M2BenchmarkScenario:
    """One natural-language M2 benchmark scenario."""

    scenario_id: str
    user_message: str
    description: str
    expected_statuses: Sequence[str]
    expected_tool: Optional[str] = None
    expected_tool_ok: bool = True
    expected_tool_arguments: Mapping[str, Any] = field(default_factory=dict)
    expected_tool_result: Mapping[str, Any] = field(default_factory=dict)
    forbidden_successful_tools: Sequence[str] = ()
    oracle_tool: Optional[str] = None
    oracle_arguments: Mapping[str, Any] = field(default_factory=dict)
    oracle_result_template: Mapping[str, Any] = field(default_factory=dict)
    oracle_comparisons: Sequence[M2OracleComparison] = ()
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
            "expected_tool_ok": self.expected_tool_ok,
            "expected_tool_arguments": dict(self.expected_tool_arguments),
            "expected_tool_result": dict(self.expected_tool_result),
            "forbidden_successful_tools": list(self.forbidden_successful_tools),
            "oracle_tool": self.oracle_tool,
            "oracle_arguments": dict(self.oracle_arguments),
            "oracle_result_template": dict(self.oracle_result_template),
            "oracle_comparisons": [
                comparison.to_dict() for comparison in self.oracle_comparisons
            ],
            "output_contains": list(self.output_contains),
            "expect_grounding_warning": self.expect_grounding_warning,
            "context": dict(self.context),
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class M2BenchmarkCheckResult:
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
class M2BenchmarkResult:
    """Complete result for one M2 benchmark scenario."""

    scenario: M2BenchmarkScenario
    agent_result: Any
    oracle_result: Optional[Dict[str, Any]]
    check_results: Sequence[M2BenchmarkCheckResult]
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
class M2BenchmarkSuiteResult:
    """Aggregate result for an M2 benchmark run."""

    results: Sequence[M2BenchmarkResult]
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


class M2BenchmarkRunner:
    """Run M2 benchmark scenarios through an agent and oracle/template."""

    def __init__(
        self,
        agent: Any,
        oracle_registry: Any,
        *,
        execute_oracle: bool = False,
    ) -> None:
        if not hasattr(agent, "run_turn"):
            raise ValueError("agent must expose run_turn(message, context=...)")
        if not hasattr(oracle_registry, "call_tool"):
            raise ValueError("oracle_registry must expose call_tool(name, arguments)")
        self.agent = agent
        self.oracle_registry = oracle_registry
        self.execute_oracle = execute_oracle

    def run_scenario(self, scenario: M2BenchmarkScenario) -> M2BenchmarkResult:
        if not isinstance(scenario, M2BenchmarkScenario):
            raise ValueError("scenario must be an M2BenchmarkScenario")

        start = time.perf_counter()
        oracle_result = self._run_oracle(scenario)
        agent_result = self.agent.run_turn(
            scenario.user_message,
            context=dict(scenario.context) if scenario.context else None,
        )
        checks = evaluate_m2_result(
            scenario,
            agent_result=agent_result,
            oracle_result=oracle_result,
        )
        return M2BenchmarkResult(
            scenario=scenario,
            agent_result=agent_result,
            oracle_result=oracle_result,
            check_results=checks,
            duration_s=time.perf_counter() - start,
        )

    def run_suite(
        self,
        scenarios: Sequence[M2BenchmarkScenario],
    ) -> M2BenchmarkSuiteResult:
        if isinstance(scenarios, (str, bytes)) or not isinstance(scenarios, Sequence):
            raise ValueError("scenarios must be a sequence of M2BenchmarkScenario objects")
        start = time.perf_counter()
        results = [self.run_scenario(scenario) for scenario in scenarios]
        return M2BenchmarkSuiteResult(
            results=results,
            duration_s=time.perf_counter() - start,
        )

    def _run_oracle(self, scenario: M2BenchmarkScenario) -> Optional[Dict[str, Any]]:
        return _scenario_oracle_result(
            scenario,
            self.oracle_registry,
            execute_tools=self.execute_oracle,
        )


def default_m2_benchmark_scenarios() -> List[M2BenchmarkScenario]:
    """Return the ten-scenario M2 live-agent benchmark suite."""

    return [
        _transient_scenario(
            scenario_id="m2_kundur_bus_fault_seconds",
            user_message=(
                "Run transient stability on the Kundur case for a bus fault at "
                "bus 5 from t=1.0 s to t=1.1 s. Simulate 5 seconds."
            ),
            description="Kundur bus-fault request with explicit second units.",
            case_path="kundur_full",
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            tags=("m2", "kundur", "bus_fault"),
        ),
        _transient_scenario(
            scenario_id="m2_kundur_bus_fault_100ms",
            user_message=(
                "For Kundur, apply a three-phase bus fault at bus 5 at t=0 s "
                "and clear it after 100 ms. Run a 5 s transient stability simulation."
            ),
            description="Tests ms-to-second conversion: 100 ms becomes 0.1 s.",
            case_path="kundur_full",
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 0.0,
                "clearing_time_s": 0.1,
            },
            tags=("m2", "kundur", "bus_fault", "unit_conversion"),
        ),
        _transient_scenario(
            scenario_id="m2_kundur_line_trip",
            user_message=(
                "Run a Kundur transient stability test where ANDES toggles "
                "Line_5 out of service at t=1.0 s. Simulate 5 seconds."
            ),
            description="Line trip through the ANDES Toggle disturbance.",
            case_path="kundur_full",
            disturbance={
                "type": "line_trip",
                "model": "Line",
                "device": "Line_5",
                "trip_time_s": 1.0,
            },
            tags=("m2", "kundur", "line_trip", "toggle"),
        ),
        _transient_scenario(
            scenario_id="m2_ieee14_dynamic_bus_fault",
            user_message=(
                "Use the IEEE 14 dynamic case and run a transient stability "
                "bus fault at bus 2 from 1.0 s to 1.08 s, ending at 5 s."
            ),
            description="IEEE 14 RAW+DYR dynamic case.",
            case_path="ieee14_dynamic",
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.08,
            },
            tags=("m2", "ieee14", "bus_fault"),
        ),
        M2BenchmarkScenario(
            scenario_id="m2_missing_clearing_time",
            user_message=(
                "Run transient stability on Kundur for a bus fault at bus 5 "
                "starting at 1 second."
            ),
            description="Incomplete bus-fault request should ask for clearing time.",
            expected_statuses=("completed", "clarification_required"),
            expected_tool=None,
            output_contains=("clearing",),
            tags=("m2", "clarification", "bus_fault"),
        ),
        M2BenchmarkScenario(
            scenario_id="m2_missing_fault_bus",
            user_message=(
                "Run transient stability on Kundur for a fault from 1.0 s to "
                "1.1 s and simulate 5 seconds."
            ),
            description="Incomplete bus-fault request should ask for fault bus.",
            expected_statuses=("completed", "clarification_required"),
            expected_tool=None,
            output_contains=("bus",),
            tags=("m2", "clarification", "bus_fault"),
        ),
        _transient_scenario(
            scenario_id="m2_wrong_tool_trap",
            user_message=(
                "Check transient stability on Kundur after a bus fault at bus 5 "
                "from 1.0 s to 1.1 s. Do not just run a power flow."
            ),
            description="A transient request must not be answered with power flow only.",
            case_path="kundur_full",
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            forbidden_successful_tools=("run_powerflow", "inspect_violations", "run_contingency"),
            tags=("m2", "wrong_tool_trap"),
        ),
        _transient_scenario(
            scenario_id="m2_ieee118_public_dynamic_bus_fault",
            user_message=(
                "Run transient stability on IEEE 118 for a bus fault at bus 10 "
                "from 1.0 s to 1.1 s."
            ),
            description="Public IEEE 118 RAW+DYR dynamic case resolved through powerfulcases.",
            case_path="ieee118_dynamic",
            disturbance={
                "type": "bus_fault",
                "bus": 10,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            output_contains=("dynamic", "public"),
            tags=("m2", "ieee118", "public_dynamic"),
        ),
        _transient_scenario(
            scenario_id="m2_static_interconnection_bus_fault",
            user_message=(
                "In IEEE 14 dynamic, add a 5 MW solar project at bus 10 as the "
                "new interconnection and run transient stability for a bus fault "
                "at bus 5 from 1.0 s to 1.1 s."
            ),
            description="M2 applies the new project as a static PQ interconnection model.",
            case_path="ieee14_dynamic",
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            connection={
                "bus": 10,
                "p_mw": 5.0,
                "connection_type": "solar",
                "is_ibr": True,
            },
            output_contains=("static", "interconnection"),
            tags=("m2", "interconnection", "static_pq"),
        ),
        _transient_scenario(
            scenario_id="m2_final_answer_grounded_metrics",
            user_message=(
                "Run Kundur transient stability for a bus fault at bus 5 from "
                "1.0 s to 1.1 s, then summarize the stability status and one "
                "grounded metric."
            ),
            description="Final answer should mention status and a solver-grounded metric.",
            case_path="kundur_full",
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            output_contains=("stability",),
            tags=("m2", "final_answer", "grounding"),
        ),
    ]


def filter_m2_scenarios(
    scenarios: Sequence[M2BenchmarkScenario],
    *,
    scenario_ids: Iterable[str] = (),
    tags: Iterable[str] = (),
) -> List[M2BenchmarkScenario]:
    """Filter M2 scenarios by id and/or tag."""

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


def evaluate_m2_result(
    scenario: M2BenchmarkScenario,
    *,
    agent_result: Any,
    oracle_result: Optional[Mapping[str, Any]],
) -> List[M2BenchmarkCheckResult]:
    """Evaluate one live agent result against an M2 benchmark scenario."""

    checks = [
        _check_status(scenario, agent_result),
        _check_grounding(scenario, agent_result),
        _check_forbidden_tools(scenario, agent_result),
    ]
    expected_record = _first_record(agent_result, scenario.expected_tool)
    if scenario.expected_tool is None:
        checks.append(_check_no_successful_solver_tools(agent_result))
    else:
        checks.append(_check_expected_tool(scenario, agent_result, expected_record))
        checks.append(_check_unexpected_tools(scenario, agent_result))
        checks.extend(_check_expected_arguments(scenario, expected_record))
        checks.extend(_check_expected_result(scenario, expected_record))
        checks.extend(_check_oracle(scenario, expected_record, oracle_result))
    checks.extend(_check_output_contains(scenario, agent_result))
    return checks


def run_m2_oracles(
    scenarios: Sequence[M2BenchmarkScenario],
    oracle_registry: Any,
    *,
    execute_tools: bool = False,
) -> List[Dict[str, Any]]:
    """Run M2 oracle checks.

    By default this is a metadata/template oracle, so it can run on laptops
    without ANDES. Pass ``execute_tools=True`` on a runtime node to run real
    ANDES oracle simulations.
    """

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
            result = _scenario_oracle_result(
                scenario,
                oracle_registry,
                execute_tools=execute_tools,
            )
            ok_expected = scenario.expected_tool_ok
            ok = (
                bool(result.get("ok", False)) == ok_expected
                if isinstance(result, Mapping)
                else False
            )
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
                    "oracle_executed": execute_tools,
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
                    "oracle_executed": execute_tools,
                    "ok": False,
                }
            )
    return outputs


def _transient_scenario(
    *,
    scenario_id: str,
    user_message: str,
    description: str,
    case_path: str,
    disturbance: Mapping[str, Any],
    connection: Optional[Mapping[str, Any]] = None,
    simulation_time_s: float = 5.0,
    output_contains: Sequence[str] = (),
    forbidden_successful_tools: Sequence[str] = (),
    tags: Sequence[str] = (),
) -> M2BenchmarkScenario:
    arguments = {
        "case_path": case_path,
        "disturbance": dict(disturbance),
        "simulation_time_s": simulation_time_s,
        "max_samples": 20,
    }
    if connection is not None:
        arguments["connection"] = dict(connection)
    expected_args = {
        "case_path": case_path,
        "disturbance.type": disturbance["type"],
    }
    if connection is not None:
        expected_args.update(
            {
                "connection.bus": connection["bus"],
                "connection.p_mw": connection["p_mw"],
                "connection.connection_type": connection["connection_type"],
                "connection.is_ibr": connection["is_ibr"],
            }
        )
    if disturbance["type"] == "bus_fault":
        expected_args.update(
            {
                "disturbance.bus": disturbance["bus"],
                "disturbance.fault_start_s": disturbance["fault_start_s"],
                "disturbance.clearing_time_s": disturbance["clearing_time_s"],
            }
        )
    if disturbance["type"] == "line_trip":
        expected_args.update(
            {
                "disturbance.model": disturbance["model"],
                "disturbance.device": disturbance["device"],
                "disturbance.trip_time_s": disturbance["trip_time_s"],
            }
        )
    comparisons = [
        M2OracleComparison("tool.stability.status", "oracle.stability.status"),
        M2OracleComparison("tool.metrics.max_angle_spread_rad", "oracle.metrics.max_angle_spread_rad"),
        M2OracleComparison("tool.metrics.max_speed_deviation_pu", "oracle.metrics.max_speed_deviation_pu"),
        M2OracleComparison("tool.metrics.min_voltage_pu", "oracle.metrics.min_voltage_pu"),
    ]
    if connection is not None:
        comparisons.extend(
            [
                M2OracleComparison(
                    "tool.dynamic_interconnection_modeling",
                    "oracle.dynamic_interconnection_modeling",
                ),
                M2OracleComparison(
                    "tool.connection_model.connection_type",
                    "oracle.connection_model.connection_type",
                ),
            ]
        )
    return M2BenchmarkScenario(
        scenario_id=scenario_id,
        user_message=user_message,
        description=description,
        expected_statuses=("completed",),
        expected_tool="run_transient_stability",
        expected_tool_arguments=expected_args,
        forbidden_successful_tools=forbidden_successful_tools,
        oracle_tool="run_transient_stability",
        oracle_arguments=arguments,
        oracle_result_template=_transient_template(arguments),
        oracle_comparisons=tuple(comparisons),
        output_contains=output_contains,
        tags=tags,
    )


def _transient_template(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    disturbance = dict(arguments.get("disturbance", {}))
    connection = (
        dict(arguments.get("connection", {}))
        if isinstance(arguments.get("connection"), Mapping)
        else None
    )
    dynamic_interconnection_modeling = connection is not None
    connection_type = str(connection.get("connection_type")) if connection else None
    direction = "load" if connection_type == "load" else "generation"
    normalized_case = _normalize_case_name(arguments.get("case_path"))
    public_ieee118 = normalized_case == "ieee118_public_dynamic"
    limitations = (
        [
            "m2_connection_model_is_static_pq",
            "m2_connection_does_not_add_machine_or_inverter_dynamics",
            "m2_v1_does_not_model_detailed_ibr_controls",
        ]
        if connection and connection.get("is_ibr")
        else (
            [
                "m2_connection_model_is_static_pq",
                "m2_connection_does_not_add_machine_or_inverter_dynamics",
            ]
            if connection
            else ["standalone_transient_screen_no_interconnection_model"]
        )
    )
    if public_ieee118:
        limitations.append("ieee118_uses_public_benchmark_dynamic_data_not_customer_validated")
    return {
        "ok": True,
        "tool": "run_transient_stability",
        "backend": "andes",
        "case_path": arguments.get("case_path"),
        "case_info": {
            "case_id": normalized_case,
            "dynamic_data_available": True,
            **(
                {
                    "case_source": "powerfulcases_ieee118_public_dynamic",
                    "data_label": "public_benchmark_data_not_customer_validated",
                }
                if public_ieee118
                else {}
            ),
        },
        "disturbance": disturbance,
        "connection_model": (
            {
                "bus": connection.get("bus"),
                "p_mw": connection.get("p_mw"),
                "q_mvar": connection.get("q_mvar", 0.0),
                "connection_type": connection_type,
                "is_ibr": connection.get("is_ibr"),
                "name": connection.get("name"),
                "vm_pu": connection.get("vm_pu", 1.0),
                "andes_model": "PQ",
                "resource_direction": direction,
                "modeling_assumption": (
                    "constant_power_load"
                    if direction == "load"
                    else "constant_power_generation_as_negative_pq_load"
                ),
                "dynamic_detail": "static_network_injection_no_machine_or_ibr_controls",
            }
            if connection
            else None
        ),
        "connection_application": (
            {
                "requested": True,
                "applied": True,
                "model": "PQ",
                "mode": (
                    "static_load"
                    if direction == "load"
                    else "static_generation_as_negative_pq_load"
                ),
            }
            if connection
            else {
                "requested": False,
                "applied": False,
                "mode": "standalone_dynamic_case",
            }
        ),
        "dynamic_interconnection_modeling": dynamic_interconnection_modeling,
        "limitations": limitations,
        "simulation": {
            "simulation_time_s": arguments.get("simulation_time_s", 5.0),
            "pflow_converged": True,
            "tds_converged": True,
            "sample_count": 3,
        },
        "stability": {"status": "pass"},
        "metrics": {
            "tds_converged": True,
            "max_angle_spread_rad": 0.25,
            "final_angle_spread_rad": 0.1,
            "max_speed_deviation_pu": 0.01,
            "min_voltage_pu": 0.92,
            "reason_codes": ["transient_metrics_within_thresholds"],
        },
        "trajectories": {
            "time_s": [0.0, 1.0, arguments.get("simulation_time_s", 5.0)],
            "generator_angle_rad": {"columns": ["1", "2"], "values": [[0.0, 0.0], [0.1, 0.2], [0.1, 0.15]]},
            "generator_speed_pu": {"columns": ["1", "2"], "values": [[1.0, 1.0], [1.01, 0.99], [1.0, 1.0]]},
            "bus_voltage_pu": {"columns": ["1", "2"], "values": [[1.0, 1.0], [0.92, 0.98], [1.0, 1.0]]},
        },
    }


def _scenario_oracle_result(
    scenario: M2BenchmarkScenario,
    oracle_registry: Any,
    *,
    execute_tools: bool,
) -> Optional[Dict[str, Any]]:
    if scenario.oracle_tool is None:
        return None
    if execute_tools:
        try:
            result = oracle_registry.call_tool(
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
        return dict(result) if isinstance(result, Mapping) else {
            "ok": False,
            "tool": scenario.oracle_tool,
            "error_type": "invalid_oracle_result",
            "error": "Oracle tool returned a non-object result.",
        }
    if scenario.oracle_result_template:
        return dict(scenario.oracle_result_template)
    return _transient_template(scenario.oracle_arguments)


def _oracle_argument_checks(scenario: M2BenchmarkScenario) -> List[Dict[str, Any]]:
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
    scenario: M2BenchmarkScenario,
    oracle_result: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    checks = []
    context = {"tool": oracle_result, "oracle": oracle_result}
    for comparison in scenario.oracle_comparisons:
        found_tool, actual = _resolve_path(context, comparison.tool_path)
        found_oracle, expected = _resolve_path(context, comparison.oracle_path)
        passed = found_tool and found_oracle
        if passed and comparison.compare_value:
            passed = _value_matches(
                actual,
                expected,
                tolerance=comparison.tolerance,
                path=comparison.tool_path,
            )
        checks.append(
            {
                "name": f"oracle_path:{comparison.tool_path}",
                "passed": passed,
                "tool_path": comparison.tool_path,
                "tool_path_found": found_tool,
                "oracle_path": comparison.oracle_path,
                "oracle_path_found": found_oracle,
                "compare_value": comparison.compare_value,
            }
        )
    return checks


def _check_status(
    scenario: M2BenchmarkScenario,
    agent_result: Any,
) -> M2BenchmarkCheckResult:
    actual = getattr(agent_result, "status", None)
    passed = actual in scenario.expected_statuses
    return M2BenchmarkCheckResult(
        name="status",
        passed=passed,
        actual=actual,
        expected=list(scenario.expected_statuses),
        message=None if passed else "Agent status was not in the allowed set.",
    )


def _check_grounding(
    scenario: M2BenchmarkScenario,
    agent_result: Any,
) -> M2BenchmarkCheckResult:
    grounding = getattr(agent_result, "grounding", None)
    actual = getattr(grounding, "warning_appended", None)
    expected = scenario.expect_grounding_warning
    passed = actual == expected
    return M2BenchmarkCheckResult(
        name="grounding_warning",
        passed=passed,
        actual=actual,
        expected=expected,
        message=None if passed else "Unexpected numeric-grounding warning state.",
    )


def _check_no_successful_solver_tools(agent_result: Any) -> M2BenchmarkCheckResult:
    records = [
        record
        for record in getattr(agent_result, "tool_records", [])
        if bool(getattr(record, "ok", False))
    ]
    passed = len(records) == 0
    return M2BenchmarkCheckResult(
        name="no_successful_solver_tools",
        passed=passed,
        actual=[getattr(record, "name", None) for record in records],
        expected=[],
        message=None if passed else "Scenario expected clarification or limitation without successful solver execution.",
    )


def _check_expected_tool(
    scenario: M2BenchmarkScenario,
    agent_result: Any,
    expected_record: Any,
) -> M2BenchmarkCheckResult:
    actual = [
        {
            "name": getattr(record, "name", None),
            "ok": bool(getattr(record, "ok", False)),
        }
        for record in getattr(agent_result, "tool_records", [])
    ]
    passed = (
        expected_record is not None
        and bool(getattr(expected_record, "ok", False)) == scenario.expected_tool_ok
    )
    return M2BenchmarkCheckResult(
        name="expected_tool",
        passed=passed,
        actual=actual,
        expected={"name": scenario.expected_tool, "ok": scenario.expected_tool_ok},
        message=None if passed else "Expected transient tool state was not observed.",
    )


def _check_unexpected_tools(
    scenario: M2BenchmarkScenario,
    agent_result: Any,
) -> M2BenchmarkCheckResult:
    successful = [
        str(getattr(record, "name", ""))
        for record in getattr(agent_result, "tool_records", [])
        if bool(getattr(record, "ok", False))
    ]
    expected_success = [scenario.expected_tool] if scenario.expected_tool_ok else []
    unexpected = [tool for tool in successful if tool not in expected_success]
    passed = not unexpected
    return M2BenchmarkCheckResult(
        name="unexpected_successful_tools",
        passed=passed,
        actual=successful,
        expected=expected_success,
        message=None if passed else "Agent executed successful tools outside the expected set.",
    )


def _check_forbidden_tools(
    scenario: M2BenchmarkScenario,
    agent_result: Any,
) -> M2BenchmarkCheckResult:
    forbidden = set(str(tool) for tool in scenario.forbidden_successful_tools)
    successful_forbidden = [
        str(getattr(record, "name", ""))
        for record in getattr(agent_result, "tool_records", [])
        if bool(getattr(record, "ok", False)) and str(getattr(record, "name", "")) in forbidden
    ]
    passed = not successful_forbidden
    return M2BenchmarkCheckResult(
        name="forbidden_successful_tools",
        passed=passed,
        actual=successful_forbidden,
        expected=[],
        message=None if passed else "A steady-state-only tool succeeded on a transient-stability request.",
    )


def _check_expected_arguments(
    scenario: M2BenchmarkScenario,
    expected_record: Any,
) -> List[M2BenchmarkCheckResult]:
    checks = []
    for path, expected in scenario.expected_tool_arguments.items():
        if expected_record is None:
            checks.append(
                M2BenchmarkCheckResult(
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
            M2BenchmarkCheckResult(
                name=f"argument:{path}",
                passed=passed,
                actual=actual if found else None,
                expected=expected,
                message=None if passed else "Tool argument did not match expected value.",
            )
        )
    return checks


def _check_expected_result(
    scenario: M2BenchmarkScenario,
    expected_record: Any,
) -> List[M2BenchmarkCheckResult]:
    checks = []
    for path, expected in scenario.expected_tool_result.items():
        if expected_record is None:
            checks.append(
                M2BenchmarkCheckResult(
                    name=f"result:{path}",
                    passed=False,
                    actual=None,
                    expected=expected,
                    message="No expected tool record is available.",
                )
            )
            continue
        result = getattr(expected_record, "result", {})
        found, actual = _resolve_path(result, path)
        passed = found and _value_matches(actual, expected, path=path)
        checks.append(
            M2BenchmarkCheckResult(
                name=f"result:{path}",
                passed=passed,
                actual=actual if found else None,
                expected=expected,
                message=None if passed else "Tool result did not match expected value.",
            )
        )
    return checks


def _check_oracle(
    scenario: M2BenchmarkScenario,
    expected_record: Any,
    oracle_result: Optional[Mapping[str, Any]],
) -> List[M2BenchmarkCheckResult]:
    if scenario.oracle_tool is None:
        return []
    checks = []
    oracle_ok = bool(isinstance(oracle_result, Mapping) and oracle_result.get("ok", False))
    checks.append(
        M2BenchmarkCheckResult(
            name="oracle_ok_state",
            passed=oracle_ok == scenario.expected_tool_ok,
            actual=oracle_ok,
            expected=scenario.expected_tool_ok,
            message=None if oracle_ok == scenario.expected_tool_ok else "Oracle ok state did not match expected tool state.",
        )
    )
    if expected_record is None:
        return checks
    tool_result = getattr(expected_record, "result", {})
    context = {"tool": tool_result, "oracle": oracle_result or {}}
    for comparison in scenario.oracle_comparisons:
        found_tool, actual = _resolve_path(context, comparison.tool_path)
        found_oracle, expected = _resolve_path(context, comparison.oracle_path)
        passed = found_tool and found_oracle
        if passed and comparison.compare_value:
            passed = _value_matches(
                actual,
                expected,
                tolerance=comparison.tolerance,
                path=comparison.tool_path,
            )
        checks.append(
            M2BenchmarkCheckResult(
                name=f"oracle:{comparison.tool_path}",
                passed=passed,
                actual=actual if found_tool else None,
                expected=expected if found_oracle else None,
                message=None if passed else "Agent tool output missed an expected oracle/template path.",
            )
        )
    return checks


def _check_output_contains(
    scenario: M2BenchmarkScenario,
    agent_result: Any,
) -> List[M2BenchmarkCheckResult]:
    output_text = str(getattr(agent_result, "output_text", ""))
    lower = output_text.lower()
    checks = []
    for expected_text in scenario.output_contains:
        expected = str(expected_text)
        passed = expected.lower() in lower
        checks.append(
            M2BenchmarkCheckResult(
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
        "kundur": "kundur_full",
        "kundurfull": "kundur_full",
        "ieee14": "ieee14_dynamic",
        "case14": "ieee14_dynamic",
        "ieee14dynamic": "ieee14_dynamic",
        "14": "ieee14_dynamic",
        "ieee118": "ieee118_public_dynamic",
        "case118": "ieee118_public_dynamic",
        "118": "ieee118_public_dynamic",
        "ieee118dynamic": "ieee118_public_dynamic",
        "ieee118public": "ieee118_public_dynamic",
        "ieee118publicdynamic": "ieee118_public_dynamic",
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
        "case_path": oracle_result.get("case_path"),
    }
    stability = oracle_result.get("stability")
    if isinstance(stability, Mapping):
        summary["stability_status"] = stability.get("status")
    metrics = oracle_result.get("metrics")
    if isinstance(metrics, Mapping):
        summary["reason_codes"] = metrics.get("reason_codes")
    return summary

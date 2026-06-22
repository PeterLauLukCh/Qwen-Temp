"""True joint M1+M2 benchmark for Mini Grid-Mind.

This suite evaluates combined interconnection questions that require one
``run_integrated_assessment`` call. The prompt includes both the proposed
connection for M1 steady-state CIA and a transient event for M2. Oracle-only
mode is template-based by default so it can run without pandapower or ANDES;
``execute_tools=True`` can be used on a runtime node for live oracle execution.
"""

from __future__ import annotations

import contextlib
import math
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .public_cases import (
    IEEE118_BUNDLED_SOURCE,
    IEEE118_LOCAL_OVERRIDE_SOURCE,
    IEEE118_PUBLIC_SOURCE,
)


FORBIDDEN_JOINT_TOOLS = (
    "run_powerflow",
    "inspect_violations",
    "run_contingency",
    "run_cia",
    "run_transient_stability",
    "run_emt_screening",
)
ALLOWED_AUXILIARY_JOINT_TOOLS = (
    "list_backends",
    "list_cases",
    "list_dynamic_cases",
)
DEFAULTABLE_ARGUMENTS = {
    "transient.enabled": True,
    "transient.required_for_approval": True,
    "emt.scr_threshold": 3.0,
    "emt.borderline_margin": 0.25,
}
ANY_VALUE = {"__joint_benchmark_any_value__": True}
IEEE118_ACCEPTED_PUBLIC_SOURCES = {
    IEEE118_PUBLIC_SOURCE,
    IEEE118_LOCAL_OVERRIDE_SOURCE,
    IEEE118_BUNDLED_SOURCE,
}
DEFAULT_LIVE_ORACLE_TIMEOUT_S = 180.0
DEFAULT_GENERATED_JOINT_SEED = 20260610
GENERATED_JOINT_PROFILES = ("m1m2", "emt", "mixed", "hard_emt")


@dataclass(frozen=True)
class JointOracleComparison:
    """One path comparison between agent output and oracle/template result."""

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

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "JointOracleComparison":
        """Restore a comparison from a JSON-compatible mapping."""

        if not isinstance(payload, Mapping):
            raise ValueError("oracle comparison must be an object")
        tolerance = payload.get("tolerance")
        if tolerance is not None:
            tolerance = float(tolerance)
        return cls(
            tool_path=_required_str(payload, "tool_path"),
            oracle_path=_required_str(payload, "oracle_path"),
            tolerance=tolerance,
            compare_value=bool(payload.get("compare_value", False)),
        )


@dataclass(frozen=True)
class JointBenchmarkScenario:
    """One natural-language joint M1+M2 benchmark scenario."""

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
    oracle_comparisons: Sequence[JointOracleComparison] = ()
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

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "JointBenchmarkScenario":
        """Restore a scenario from ``to_dict()`` JSON output."""

        if not isinstance(payload, Mapping):
            raise ValueError("scenario must be an object")
        return cls(
            scenario_id=_required_str(payload, "scenario_id"),
            user_message=_required_str(payload, "user_message"),
            description=str(payload.get("description", "")),
            expected_statuses=_string_tuple(payload.get("expected_statuses", ())),
            expected_tool=_optional_string_value(payload.get("expected_tool")),
            expected_tool_ok=bool(payload.get("expected_tool_ok", True)),
            expected_tool_arguments=_mapping_value(payload, "expected_tool_arguments"),
            expected_tool_result=_mapping_value(payload, "expected_tool_result"),
            forbidden_successful_tools=_string_tuple(
                payload.get("forbidden_successful_tools", ())
            ),
            oracle_tool=_optional_string_value(payload.get("oracle_tool")),
            oracle_arguments=_mapping_value(payload, "oracle_arguments"),
            oracle_result_template=_mapping_value(payload, "oracle_result_template"),
            oracle_comparisons=tuple(
                JointOracleComparison.from_mapping(item)
                for item in _sequence_value(payload.get("oracle_comparisons", ()))
            ),
            output_contains=_string_tuple(payload.get("output_contains", ())),
            expect_grounding_warning=bool(payload.get("expect_grounding_warning", False)),
            context=_mapping_value(payload, "context"),
            tags=_string_tuple(payload.get("tags", ())),
        )


@dataclass(frozen=True)
class JointBenchmarkCheckResult:
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
class JointBenchmarkResult:
    """Complete result for one joint benchmark scenario."""

    scenario: JointBenchmarkScenario
    agent_result: Any
    oracle_result: Optional[Dict[str, Any]]
    check_results: Sequence[JointBenchmarkCheckResult]
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
class JointBenchmarkSuiteResult:
    """Aggregate result for a joint benchmark run."""

    results: Sequence[JointBenchmarkResult]
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


class JointBenchmarkRunner:
    """Run joint benchmark scenarios through an agent and oracle/template."""

    def __init__(
        self,
        agent: Any,
        oracle_registry: Any,
        *,
        execute_oracle: bool = False,
        oracle_timeout_s: Optional[float] = DEFAULT_LIVE_ORACLE_TIMEOUT_S,
    ) -> None:
        if not hasattr(agent, "run_turn"):
            raise ValueError("agent must expose run_turn(message, context=...)")
        if not hasattr(oracle_registry, "call_tool"):
            raise ValueError("oracle_registry must expose call_tool(name, arguments)")
        self.agent = agent
        self.oracle_registry = oracle_registry
        self.execute_oracle = execute_oracle
        self.oracle_timeout_s = oracle_timeout_s

    def run_scenario(self, scenario: JointBenchmarkScenario) -> JointBenchmarkResult:
        if not isinstance(scenario, JointBenchmarkScenario):
            raise ValueError("scenario must be a JointBenchmarkScenario")

        start = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            oracle_result = self._run_oracle(scenario)
            agent_result = self.agent.run_turn(
                scenario.user_message,
                context=dict(scenario.context) if scenario.context else None,
            )
        checks = evaluate_joint_result(
            scenario,
            agent_result=agent_result,
            oracle_result=oracle_result,
        )
        return JointBenchmarkResult(
            scenario=scenario,
            agent_result=agent_result,
            oracle_result=oracle_result,
            check_results=checks,
            duration_s=time.perf_counter() - start,
        )

    def run_suite(
        self,
        scenarios: Sequence[JointBenchmarkScenario],
    ) -> JointBenchmarkSuiteResult:
        if isinstance(scenarios, (str, bytes)) or not isinstance(scenarios, Sequence):
            raise ValueError("scenarios must be a sequence of JointBenchmarkScenario objects")
        start = time.perf_counter()
        results = [self.run_scenario(scenario) for scenario in scenarios]
        return JointBenchmarkSuiteResult(
            results=results,
            duration_s=time.perf_counter() - start,
        )

    def _run_oracle(self, scenario: JointBenchmarkScenario) -> Optional[Dict[str, Any]]:
        return _scenario_oracle_result(
            scenario,
            self.oracle_registry,
            execute_tools=self.execute_oracle,
            timeout_s=self.oracle_timeout_s,
        )


def default_joint_benchmark_scenarios() -> List[JointBenchmarkScenario]:
    """Return the true joint M1+M2 benchmark suite."""

    return [
        _joint_scenario(
            scenario_id="joint_ieee14_solar_fault_100ms",
            user_message=(
                "Can we add a 20 MW solar project at bus 10 on IEEE 14 and "
                "survive a bus fault at bus 2 that starts at 1.0 s and clears "
                "after 100 ms?"
            ),
            description="Solar interconnection with ms-to-second fault-clearing conversion.",
            connection={
                "bus": 10,
                "p_mw": 20.0,
                "connection_type": "solar",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            tags=("joint", "ieee14", "solar", "bus_fault", "unit_conversion", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_load_fault_80ms",
            user_message=(
                "Assess whether IEEE 14 can accept a 5 MW data-center load at "
                "bus 9 and still survive a bus fault at bus 4 from 1.0 s to "
                "1.08 s."
            ),
            description="Load interconnection with explicit 80 ms clearing interval.",
            connection={
                "bus": 9,
                "p_mw": 5.0,
                "connection_type": "load",
                "is_ibr": False,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 4,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.08,
            },
            tags=("joint", "ieee14", "load", "bus_fault", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_bess_fault_with_q",
            user_message=(
                "Run a joint interconnection and transient screen for a 2 MW "
                "BESS at bus 9 on IEEE 14 with 0.5 MVAr reactive injection, "
                "using a bus fault at bus 2 from 1.0 s to 1.1 s."
            ),
            description="BESS case checks q_mvar parsing and static-PQ limitation.",
            connection={
                "bus": 9,
                "p_mw": 2.0,
                "q_mvar": 0.5,
                "connection_type": "bess",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            tags=("joint", "ieee14", "bess", "q_mvar", "bus_fault", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_wind_optional_transient",
            user_message=(
                "For IEEE 14, check a 3 MW wind project at bus 6. Include a "
                "transient bus-fault screen at bus 5 from 1.0 s to 1.12 s as "
                "advisory only, not required for approval."
            ),
            description="Wind interconnection where transient status is advisory.",
            connection={
                "bus": 6,
                "p_mw": 3.0,
                "connection_type": "wind",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.12,
            },
            required_for_approval=False,
            tags=("joint", "ieee14", "wind", "advisory", "bus_fault", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_synchronous_fault",
            user_message=(
                "Can IEEE 14 take a 10 MW synchronous generator-style project at "
                "bus 3 and pass a transient bus fault at bus 2 from 1.0 s to "
                "1.1 s?"
            ),
            description="Synchronous/generator-style project with bus-fault transient screen.",
            connection={
                "bus": 3,
                "p_mw": 10.0,
                "connection_type": "synchronous",
                "is_ibr": False,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            tags=("joint", "ieee14", "synchronous", "generator", "bus_fault", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_small_solar_fast_clear",
            user_message=(
                "Can IEEE 14 accept a 1 MW solar project at bus 6 and survive "
                "a bus fault at bus 4 from 1.0 s to 1.05 s?"
            ),
            description="Small solar addition with fast 50 ms fault clearing.",
            connection={
                "bus": 6,
                "p_mw": 1.0,
                "connection_type": "solar",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 4,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.05,
            },
            tags=("joint", "ieee14", "solar", "bus_fault", "fast_clear", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_solar_bus14_75ms",
            user_message=(
                "For IEEE 14, test a 3 MW solar interconnection at bus 14 "
                "against a bus fault at bus 2 starting at 1.0 s and clearing "
                "after 75 ms."
            ),
            description="Solar addition at a remote bus with ms-to-second conversion.",
            connection={
                "bus": 14,
                "p_mw": 3.0,
                "connection_type": "solar",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.075,
            },
            tags=("joint", "ieee14", "solar", "bus_fault", "unit_conversion", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_load_bus4_fault7",
            user_message=(
                "Assess a 1.5 MW load addition at bus 4 on IEEE 14 and check "
                "whether it survives a bus fault at bus 7 from 1.0 s to 1.09 s."
            ),
            description="Small load addition with a fault at a different load-area bus.",
            connection={
                "bus": 4,
                "p_mw": 1.5,
                "connection_type": "load",
                "is_ibr": False,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 7,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.09,
            },
            tags=("joint", "ieee14", "load", "bus_fault", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_data_center_bus13",
            user_message=(
                "Can IEEE 14 host a 2 MW data-center load at bus 13 and pass "
                "a transient bus fault at bus 9 from 1.0 s to 1.12 s?"
            ),
            description="Data-center wording should map to a load interconnection.",
            connection={
                "bus": 13,
                "p_mw": 2.0,
                "connection_type": "load",
                "is_ibr": False,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 9,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.12,
            },
            tags=("joint", "ieee14", "load", "data_center", "bus_fault", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_bess_bus5_q_support",
            user_message=(
                "Run the integrated screen for a 1 MW BESS at bus 5 on IEEE 14 "
                "with 0.2 MVAr reactive injection, using a bus fault at bus 2 "
                "from 1.0 s to 1.1 s."
            ),
            description="BESS case with a smaller explicit q_mvar value.",
            connection={
                "bus": 5,
                "p_mw": 1.0,
                "q_mvar": 0.2,
                "connection_type": "bess",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            tags=("joint", "ieee14", "bess", "q_mvar", "bus_fault", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_bess_bus14_advisory",
            user_message=(
                "For IEEE 14, evaluate a 0.75 MW BESS at bus 14. Include a bus "
                "fault at bus 4 from 1.0 s to 1.08 s, but treat the transient "
                "screen as advisory rather than required for approval."
            ),
            description="Small BESS with advisory M2 screening.",
            connection={
                "bus": 14,
                "p_mw": 0.75,
                "connection_type": "bess",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 4,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.08,
            },
            required_for_approval=False,
            tags=("joint", "ieee14", "bess", "advisory", "bus_fault", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_wind_bus11_fault9",
            user_message=(
                "Can a 4 MW wind project connect at bus 11 on IEEE 14 and "
                "remain stable for a bus fault at bus 9 from 0.8 s to 0.95 s?"
            ),
            description="Wind addition with non-1.0 s fault timing.",
            connection={
                "bus": 11,
                "p_mw": 4.0,
                "connection_type": "wind",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 9,
                "fault_start_s": 0.8,
                "clearing_time_s": 0.95,
            },
            tags=("joint", "ieee14", "wind", "bus_fault", "nondefault_timing", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_wind_bus12_60ms",
            user_message=(
                "Check whether IEEE 14 can add 2.5 MW of wind at bus 12 and "
                "survive a bus fault at bus 6 starting at 1.0 s and cleared "
                "after 60 ms."
            ),
            description="Wind addition with another ms-to-second conversion.",
            connection={
                "bus": 12,
                "p_mw": 2.5,
                "connection_type": "wind",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 6,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.06,
            },
            tags=("joint", "ieee14", "wind", "bus_fault", "unit_conversion", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_hybrid_bus9",
            user_message=(
                "Run an integrated M1 plus M2 assessment for a 3 MW hybrid "
                "solar-storage project at bus 9 on IEEE 14 with a bus fault at "
                "bus 4 from 1.0 s to 1.1 s."
            ),
            description="Hybrid IBR project should route through integrated assessment.",
            connection={
                "bus": 9,
                "p_mw": 3.0,
                "connection_type": "hybrid",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 4,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            tags=("joint", "ieee14", "hybrid", "bus_fault", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_synchronous_bus2_small",
            user_message=(
                "Can IEEE 14 add a 1 MW synchronous generator-style project at "
                "bus 2 and pass a bus fault at bus 3 from 1.0 s to 1.1 s?"
            ),
            description="Small synchronous-generator-style addition at an existing generator bus.",
            connection={
                "bus": 2,
                "p_mw": 1.0,
                "connection_type": "synchronous",
                "is_ibr": False,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 3,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            tags=("joint", "ieee14", "synchronous", "generator", "bus_fault", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_load_bus10_longer_clear",
            user_message=(
                "Assess an 8 MW load at bus 10 on IEEE 14 and run the transient "
                "fault screen for a bus fault at bus 6 from 1.0 s to 1.14 s."
            ),
            description="Larger load addition with a longer clearing time.",
            connection={
                "bus": 10,
                "p_mw": 8.0,
                "connection_type": "load",
                "is_ibr": False,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 6,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.14,
            },
            tags=("joint", "ieee14", "load", "bus_fault", "longer_clear", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_solar_bus8_early_fault",
            user_message=(
                "For IEEE 14, connect a 2 MW solar project at bus 8 and test "
                "a bus fault at bus 5 from 0.5 s to 0.62 s."
            ),
            description="Solar addition with an early transient event.",
            connection={
                "bus": 8,
                "p_mw": 2.0,
                "connection_type": "solar",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 0.5,
                "clearing_time_s": 0.62,
            },
            tags=("joint", "ieee14", "solar", "bus_fault", "early_fault", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee14_load_bus12_optional",
            user_message=(
                "Check a 1 MW load addition at bus 12 on IEEE 14. Also run a "
                "bus fault at bus 2 from 1.0 s to 1.07 s as an optional "
                "transient screen."
            ),
            description="Load addition with optional transient screening language.",
            connection={
                "bus": 12,
                "p_mw": 1.0,
                "connection_type": "load",
                "is_ibr": False,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.07,
            },
            required_for_approval=False,
            tags=("joint", "ieee14", "load", "advisory", "bus_fault", "live_safe"),
        ),
        _joint_scenario(
            scenario_id="joint_wrong_tool_trap",
            user_message=(
                "Can we add a 15 MW solar project at bus 10 on IEEE 14 and "
                "survive a bus fault at bus 2 from 1.0 s to 1.1 s? Do not "
                "answer with only power flow, CIA, or standalone transient stability."
            ),
            description="Complete joint prompt must use run_integrated_assessment.",
            connection={
                "bus": 10,
                "p_mw": 15.0,
                "connection_type": "solar",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            forbidden_successful_tools=FORBIDDEN_JOINT_TOOLS,
            tags=("joint", "wrong_tool_trap", "ieee14", "bus_fault", "live_safe"),
        ),
        JointBenchmarkScenario(
            scenario_id="joint_missing_fault_timing",
            user_message=(
                "Can we add a 20 MW solar project at bus 10 on IEEE 14 and "
                "survive a bus fault at bus 2?"
            ),
            description="Complete interconnection but incomplete transient timing.",
            expected_statuses=("completed", "clarification_required"),
            expected_tool=None,
            output_contains=("fault", "time"),
            tags=("joint", "clarification", "missing_fault_timing"),
        ),
        JointBenchmarkScenario(
            scenario_id="joint_missing_connection_bus",
            user_message=(
                "Can we add a 20 MW solar project on IEEE 14 and survive a bus "
                "fault at bus 2 from 1.0 s to 1.1 s?"
            ),
            description="Complete disturbance but missing project connection bus.",
            expected_statuses=("completed", "clarification_required"),
            expected_tool=None,
            output_contains=("bus",),
            tags=("joint", "clarification", "missing_connection_bus"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee118_solar_fault_100ms",
            user_message=(
                "Can we add a 20 MW solar project at bus 10 on IEEE 118 and "
                "survive a bus fault at bus 2 that starts at 1.0 s and clears "
                "after 100 ms?"
            ),
            description="Public IEEE118 solar interconnection with strict M1/M2 case alignment.",
            case_path="ieee118",
            transient_case_path="ieee118_dynamic",
            connection={
                "bus": 10,
                "p_mw": 20.0,
                "connection_type": "solar",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            expected_tool_result={
                "case_alignment.strict_match": True,
                "case_alignment.source": "powerfulcases_ieee118_public_dynamic",
            },
            output_contains=("ieee 118", "public", "static"),
            tags=("joint", "ieee118", "ieee118_live_safe", "solar", "bus_fault"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee118_load_fault_80ms",
            user_message=(
                "Assess whether IEEE 118 can accept a 5 MW data-center load at "
                "bus 30 and still survive a bus fault at bus 4 from 1.0 s to "
                "1.08 s."
            ),
            description="Public IEEE118 load interconnection with explicit 80 ms clearing interval.",
            case_path="ieee118",
            transient_case_path="ieee118_dynamic",
            connection={
                "bus": 30,
                "p_mw": 5.0,
                "connection_type": "load",
                "is_ibr": False,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 4,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.08,
            },
            expected_tool_result={
                "case_alignment.strict_match": True,
                "case_alignment.source": "powerfulcases_ieee118_public_dynamic",
            },
            output_contains=("ieee 118", "public", "static"),
            tags=("joint", "ieee118", "ieee118_live_safe", "load", "bus_fault"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee118_bess_fault_with_q",
            user_message=(
                "Run a joint interconnection and transient screen for a 2 MW "
                "BESS at bus 50 on IEEE 118 with 0.5 MVAr reactive injection, "
                "using a bus fault at bus 2 from 1.0 s to 1.1 s."
            ),
            description="Public IEEE118 BESS case checks q_mvar parsing and static-PQ limitation.",
            case_path="ieee118",
            transient_case_path="ieee118_dynamic",
            connection={
                "bus": 50,
                "p_mw": 2.0,
                "q_mvar": 0.5,
                "connection_type": "bess",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            expected_tool_result={
                "case_alignment.strict_match": True,
                "case_alignment.source": "powerfulcases_ieee118_public_dynamic",
            },
            output_contains=("ieee 118", "public", "static"),
            tags=("joint", "ieee118", "ieee118_live_safe", "bess", "q_mvar", "bus_fault"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee118_wind_optional_transient",
            user_message=(
                "For IEEE 118, check a 3 MW wind project at bus 60. Include a "
                "transient bus-fault screen at bus 5 from 1.0 s to 1.12 s as "
                "advisory only, not required for approval."
            ),
            description="Public IEEE118 wind interconnection where transient status is advisory.",
            case_path="ieee118",
            transient_case_path="ieee118_dynamic",
            connection={
                "bus": 60,
                "p_mw": 3.0,
                "connection_type": "wind",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.12,
            },
            required_for_approval=False,
            expected_tool_result={
                "case_alignment.strict_match": True,
                "case_alignment.source": "powerfulcases_ieee118_public_dynamic",
            },
            output_contains=("ieee 118", "public", "static"),
            tags=("joint", "ieee118", "ieee118_live_safe", "wind", "advisory", "bus_fault"),
        ),
        _joint_scenario(
            scenario_id="joint_ieee118_wrong_tool_trap",
            user_message=(
                "Can we add a 15 MW solar project at bus 10 on IEEE 118 and "
                "survive a bus fault at bus 2 from 1.0 s to 1.1 s? Do not "
                "answer with only power flow, CIA, or standalone transient stability."
            ),
            description="Public IEEE118 complete joint prompt must use run_integrated_assessment.",
            case_path="ieee118",
            transient_case_path="ieee118_dynamic",
            connection={
                "bus": 10,
                "p_mw": 15.0,
                "connection_type": "solar",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            expected_tool_result={
                "case_alignment.strict_match": True,
                "case_alignment.source": "powerfulcases_ieee118_public_dynamic",
            },
            output_contains=("ieee 118", "public", "static"),
            tags=("joint", "ieee118", "ieee118_live_safe", "wrong_tool_trap", "bus_fault"),
        ),
        _joint_scenario(
            scenario_id="joint_final_answer_grounded_static_pq",
            user_message=(
                "For IEEE 14, can we add a 5 MW solar project at bus 10 and "
                "survive a bus fault at bus 2 from 1.0 s to 1.1 s? Give the "
                "recommendation, M2 stability status, one grounded metric, and "
                "the static-PQ limitation."
            ),
            description="Final answer must include recommendation, M2 status, metric, and static-PQ limitation.",
            connection={
                "bus": 10,
                "p_mw": 5.0,
                "connection_type": "solar",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            output_contains=("recommendation", "stability", "metric", "static", "pq"),
            tags=("joint", "final_answer", "grounding", "static_pq", "ieee14", "live_safe"),
        ),
        *_emt_joint_scenarios(),
    ]


def _emt_joint_scenarios() -> List[JointBenchmarkScenario]:
    """Return EMT-inclusive IEEE14 joint M1+M2+F4 benchmark scenarios."""

    emt_required = {
        "enabled": True,
        "required_for_approval": True,
        "scr_threshold": 3.0,
    }
    emt_advisory = {
        "enabled": True,
        "required_for_approval": False,
        "scr_threshold": 3.0,
    }
    return [
        _joint_scenario(
            scenario_id="joint_emt_ieee14_solar_scr_pass",
            user_message=(
                "Can IEEE 14 accept a 5 MW solar project at bus 10, survive a "
                "bus fault at bus 2 from 1.0 s to 1.1 s, and pass the EMT/SCR "
                "screen with SCR threshold 3.0?"
            ),
            description="Solar interconnection requiring M1 CIA, M2 transient stability, and EMT/SCR screening.",
            connection={
                "bus": 10,
                "p_mw": 5.0,
                "connection_type": "solar",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            emt=emt_required,
            output_contains=("emt", "scr"),
            tags=("joint", "emt", "emt_live_safe", "ieee14", "solar", "bus_fault"),
        ),
        _joint_scenario(
            scenario_id="joint_emt_ieee14_solar_scr_fail",
            user_message=(
                "Check whether IEEE 14 can add a 20 MW solar project at bus 10, "
                "survive a bus fault at bus 2 from 1.0 s to 1.1 s, and still "
                "meet an EMT/SCR threshold of 3.0."
            ),
            description="Weak-grid SCR trap where M1/M2 can pass but required EMT/SCR rejects the proposal.",
            connection={
                "bus": 10,
                "p_mw": 20.0,
                "connection_type": "solar",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            emt=emt_required,
            emt_status="fail",
            recommendation="reject",
            output_contains=("emt", "scr", "reject"),
            tags=("joint", "emt", "emt_live_safe", "ieee14", "solar", "scr_fail", "bus_fault"),
        ),
        _joint_scenario(
            scenario_id="joint_emt_ieee14_bess_q_support",
            user_message=(
                "Run a full joint screen for a 2 MW BESS at bus 9 on IEEE 14 "
                "with 0.5 MVAr reactive injection, a bus fault at bus 2 from "
                "1.0 s to 1.1 s, and required EMT/SCR screening."
            ),
            description="BESS case checks q_mvar extraction across M1, M2, and EMT.",
            connection={
                "bus": 9,
                "p_mw": 2.0,
                "q_mvar": 0.5,
                "connection_type": "bess",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            emt=emt_required,
            tags=("joint", "emt", "emt_live_safe", "ieee14", "bess", "q_mvar", "bus_fault"),
        ),
        _joint_scenario(
            scenario_id="joint_emt_ieee14_wind_advisory_scr",
            user_message=(
                "For IEEE 14, evaluate a 3 MW wind project at bus 6 with a bus "
                "fault at bus 5 from 1.0 s to 1.12 s. Treat both the transient "
                "screen and EMT/SCR screen as advisory, not required for approval."
            ),
            description="Advisory M2 plus advisory EMT should still report grounded metrics without blocking approval.",
            connection={
                "bus": 6,
                "p_mw": 3.0,
                "connection_type": "wind",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.12,
            },
            required_for_approval=False,
            emt=emt_advisory,
            tags=("joint", "emt", "emt_live_safe", "ieee14", "wind", "advisory", "bus_fault"),
        ),
        _joint_scenario(
            scenario_id="joint_emt_ieee14_hybrid_bus9",
            user_message=(
                "Can a 3 MW hybrid solar-storage project at bus 9 on IEEE 14 "
                "pass CIA, survive a bus fault at bus 4 from 1.0 s to 1.1 s, "
                "and pass required EMT/SCR screening?"
            ),
            description="Hybrid IBR case exercises the full integrated M1+M2+EMT route.",
            connection={
                "bus": 9,
                "p_mw": 3.0,
                "connection_type": "hybrid",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 4,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            emt=emt_required,
            tags=("joint", "emt", "emt_live_safe", "ieee14", "hybrid", "bus_fault"),
        ),
        _joint_scenario(
            scenario_id="joint_emt_wrong_tool_trap",
            user_message=(
                "Can we add a 5 MW solar project at bus 10 on IEEE 14, survive "
                "a bus fault at bus 2 from 1.0 s to 1.1 s, and pass EMT/SCR? "
                "Do not answer with only CIA, standalone transient stability, "
                "or standalone EMT screening."
            ),
            description="Complete EMT-inclusive prompt must use run_integrated_assessment.",
            connection={
                "bus": 10,
                "p_mw": 5.0,
                "connection_type": "solar",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            emt=emt_required,
            forbidden_successful_tools=FORBIDDEN_JOINT_TOOLS,
            tags=("joint", "emt", "emt_live_safe", "ieee14", "wrong_tool_trap", "bus_fault"),
        ),
        JointBenchmarkScenario(
            scenario_id="joint_emt_missing_fault_timing",
            user_message=(
                "Can we add a 5 MW solar project at bus 10 on IEEE 14, survive "
                "a bus fault at bus 2, and pass EMT/SCR screening?"
            ),
            description="Complete interconnection and EMT request but incomplete transient timing.",
            expected_statuses=("completed", "clarification_required"),
            expected_tool=None,
            output_contains=("fault", "time"),
            tags=("joint", "emt", "emt_live_safe", "clarification", "missing_fault_timing"),
        ),
        JointBenchmarkScenario(
            scenario_id="joint_emt_missing_connection_bus",
            user_message=(
                "Can IEEE 14 accept a 5 MW solar project, survive a bus fault "
                "at bus 2 from 1.0 s to 1.1 s, and pass EMT/SCR screening?"
            ),
            description="Complete transient and EMT request but missing project connection bus.",
            expected_statuses=("completed", "clarification_required"),
            expected_tool=None,
            output_contains=("bus",),
            tags=("joint", "emt", "emt_live_safe", "clarification", "missing_connection_bus"),
        ),
        _joint_scenario(
            scenario_id="joint_emt_final_answer_grounded",
            user_message=(
                "For IEEE 14, can we add a 5 MW solar project at bus 10, pass "
                "CIA, survive a bus fault at bus 2 from 1.0 s to 1.1 s, and "
                "pass EMT/SCR? Give the recommendation, M2 stability status, "
                "EMT status, one SCR metric, and the static-model limitation."
            ),
            description="Final answer must mention M1, M2, EMT/SCR, grounded metric, and static modeling limitation.",
            connection={
                "bus": 10,
                "p_mw": 5.0,
                "connection_type": "solar",
                "is_ibr": True,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            emt=emt_required,
            output_contains=("recommendation", "stability", "emt", "scr", "static"),
            tags=("joint", "emt", "emt_live_safe", "ieee14", "final_answer", "grounding"),
        ),
        _joint_scenario(
            scenario_id="joint_emt_ieee14_load_skips_scr",
            user_message=(
                "Assess a 5 MW data-center load at bus 9 on IEEE 14 with a "
                "bus fault at bus 4 from 1.0 s to 1.08 s and include EMT/SCR "
                "screening if it is applicable."
            ),
            description="Non-IBR load case should run the integrated assessment and mark EMT/SCR as skipped.",
            connection={
                "bus": 9,
                "p_mw": 5.0,
                "connection_type": "load",
                "is_ibr": False,
            },
            disturbance={
                "type": "bus_fault",
                "bus": 4,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.08,
            },
            emt=emt_advisory,
            emt_status="skipped",
            output_contains=("emt", "scr"),
            tags=("joint", "emt", "emt_live_safe", "ieee14", "load", "non_ibr", "bus_fault"),
        ),
    ]


def generate_joint_benchmark_scenarios(
    count: int,
    *,
    seed: int = DEFAULT_GENERATED_JOINT_SEED,
    profile: str = "mixed",
) -> List[JointBenchmarkScenario]:
    """Generate a reproducible IEEE14 joint benchmark suite.

    The generated suite is meant for scale testing and hard-case mining. It
    uses the same ``JointBenchmarkScenario`` format as the curated suite, but
    its labels are deterministic pseudo-labels from the current solver policy,
    not expert-validated labels.
    """

    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    normalized_profile = str(profile).strip().lower()
    if normalized_profile not in GENERATED_JOINT_PROFILES:
        raise ValueError(
            "profile must be one of: " + ", ".join(GENERATED_JOINT_PROFILES)
        )

    rng = random.Random(seed)
    scenarios: List[JointBenchmarkScenario] = []
    used_ids: set[str] = set()
    for index in range(count):
        scenarios.append(
            _generated_joint_scenario(
                index=index,
                rng=rng,
                seed=seed,
                profile=normalized_profile,
                used_ids=used_ids,
            )
        )
    return scenarios


def joint_benchmark_scenarios_from_payload(payload: Any) -> List[JointBenchmarkScenario]:
    """Load scenarios from a saved JSON-compatible benchmark payload.

    Accepts either the full ``run_joint_benchmark.py --list-scenarios`` object
    with a ``scenarios`` field, or a bare list of scenario objects.
    """

    if isinstance(payload, Mapping):
        if "scenarios" not in payload:
            raise ValueError("scenario payload object must contain a scenarios field")
        payload = payload["scenarios"]
    sequence = _sequence_value(payload)
    scenarios = [
        JointBenchmarkScenario.from_mapping(item)
        for item in sequence
    ]
    if not scenarios:
        raise ValueError("scenario payload must contain at least one scenario")
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    duplicates = sorted(
        scenario_id
        for scenario_id in set(scenario_ids)
        if scenario_ids.count(scenario_id) > 1
    )
    if duplicates:
        raise ValueError("scenario payload contains duplicate ids: " + ", ".join(duplicates))
    return scenarios


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional string value must be a string or null")
    return value


def _mapping_value(payload: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return dict(value)


def _sequence_value(value: Any) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("scenario sequence field must be a list")
    return value


def _string_tuple(value: Any) -> Tuple[str, ...]:
    return tuple(str(item) for item in _sequence_value(value))


def _generated_joint_scenario(
    *,
    index: int,
    rng: random.Random,
    seed: int,
    profile: str,
    used_ids: set[str],
) -> JointBenchmarkScenario:
    hard = profile == "hard_emt"
    connection_type = _generated_connection_type(index, rng, profile)
    is_ibr = connection_type in {"solar", "wind", "bess", "hybrid"}
    connection_bus = (
        rng.choice([5, 6, 9, 10, 14, 14, 10, 6])
        if hard
        else rng.choice([3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14])
    )
    p_mw = _generated_project_mw(connection_type, rng, profile=profile, bus=connection_bus)
    q_mvar = _generated_q_mvar(connection_type, rng, profile=profile)
    connection: Dict[str, Any] = {
        "bus": connection_bus,
        "p_mw": p_mw,
        "connection_type": connection_type,
        "is_ibr": is_ibr,
        "name": f"generated_{connection_type}_{index:04d}",
    }
    if q_mvar is not None:
        connection["q_mvar"] = q_mvar

    fault_bus = rng.choice([2, 3, 4, 5, 6, 7, 9])
    start = rng.choice([0.55, 0.8, 0.95, 1.0, 1.15, 1.2]) if hard else rng.choice([0.5, 0.8, 1.0, 1.2])
    duration = (
        rng.choice([0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.125, 0.15])
        if hard
        else rng.choice([0.05, 0.06, 0.075, 0.08, 0.09, 0.1, 0.12, 0.15])
    )
    clearing = round(start + duration, 3)
    disturbance = {
        "type": "bus_fault",
        "bus": fault_bus,
        "fault_start_s": start,
        "clearing_time_s": clearing,
    }
    transient_required = rng.random() >= (0.08 if hard else 0.18)

    include_emt = _generated_include_emt(profile, connection_type, rng)
    emt: Optional[Dict[str, Any]] = None
    emt_status: Optional[str] = None
    if include_emt:
        emt_required = False if not is_ibr else rng.random() >= (0.08 if hard else 0.22)
        scr_threshold = float(rng.choice([3.0, 3.25, 3.5, 4.0])) if hard else 3.0
        emt = {
            "enabled": True,
            "required_for_approval": emt_required,
            "scr_threshold": scr_threshold,
        }
        emt_status = _generated_emt_status(connection, threshold=scr_threshold)
    else:
        emt_required = False

    recommendation = _generated_recommendation(
        transient_required=transient_required,
        m2_status="pass",
        emt_status=emt_status,
    )
    scenario_id = _unique_generated_id(
        index=index,
        seed=seed,
        profile=profile,
        connection_type=connection_type,
        connection_bus=connection_bus,
        fault_bus=fault_bus,
        used_ids=used_ids,
    )
    user_message = _generated_user_message(
        connection=connection,
        disturbance=disturbance,
        transient_required=transient_required,
        emt=emt,
        hard=hard,
        rng=rng,
    )
    tags = [
        "joint",
        "ieee14",
        "generated",
        "generated_ieee14",
        "generated_live_safe",
        f"generated_{profile}",
        connection_type,
        "bus_fault",
    ]
    if include_emt:
        tags.extend(["emt", "generated_emt"])
    if not transient_required:
        tags.append("advisory")
    if emt_status:
        tags.append(f"emt_{emt_status}")
    if hard and include_emt:
        output_contains = ("recommendation", "emt", "scr", "static pq")
    else:
        output_contains = ("recommendation", "emt", "scr") if include_emt else ("recommendation",)

    return _joint_scenario(
        scenario_id=scenario_id,
        user_message=user_message,
        description=(
            "Generated IEEE14 joint benchmark scenario with solver-based "
            "pseudo-labels; not expert validated."
        ),
        connection=connection,
        disturbance=disturbance,
        required_for_approval=transient_required,
        emt=emt,
        emt_status=emt_status,
        recommendation=recommendation,
        output_contains=output_contains,
        tags=tuple(tags),
        strict_template_statuses=False,
    )


def _generated_connection_type(
    index: int,
    rng: random.Random,
    profile: str,
) -> str:
    if profile == "hard_emt":
        cycle = ["solar", "load", "bess", "hybrid", "wind", "load"]
        base = cycle[index % len(cycle)]
        if rng.random() < 0.12:
            return rng.choice(cycle)
        return base
    if profile == "emt":
        cycle = ["solar", "wind", "bess", "hybrid", "load"]
    else:
        cycle = ["solar", "load", "bess", "wind", "hybrid", "synchronous"]
    base = cycle[index % len(cycle)]
    if rng.random() < 0.18:
        return rng.choice(cycle)
    return base


def _generated_project_mw(
    connection_type: str,
    rng: random.Random,
    *,
    profile: str = "mixed",
    bus: Optional[int] = None,
) -> float:
    if profile == "hard_emt":
        weak_bus_values = {
            5: [12.0, 15.0, 18.0, 20.0],
            6: [10.0, 12.0, 15.0, 18.0],
            9: [10.0, 12.0, 14.0, 15.0],
            10: [12.0, 15.0, 18.0, 20.0],
            14: [8.0, 10.0, 12.0, 15.0],
        }
        if connection_type in {"solar", "wind", "bess", "hybrid"}:
            return float(rng.choice(weak_bus_values.get(bus, [10.0, 12.0, 15.0])))
        if connection_type == "load":
            return float(rng.choice([0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]))
        return float(rng.choice([3.0, 5.0, 8.0, 10.0]))
    values_by_type = {
        "load": [0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0],
        "solar": [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0],
        "wind": [1.5, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0],
        "bess": [0.75, 1.0, 2.0, 3.0, 5.0, 8.0],
        "hybrid": [2.0, 3.0, 5.0, 8.0, 10.0],
        "synchronous": [1.0, 2.0, 3.0, 5.0, 10.0],
    }
    return float(rng.choice(values_by_type.get(connection_type, [1.0, 2.0, 5.0])))


def _generated_q_mvar(
    connection_type: str,
    rng: random.Random,
    *,
    profile: str = "mixed",
) -> Optional[float]:
    if profile == "hard_emt":
        if connection_type in {"bess", "hybrid"}:
            return float(rng.choice([0.5, 1.0, 1.5, 2.0]))
        if connection_type in {"solar", "wind"} and rng.random() < 0.7:
            return float(rng.choice([0.5, 1.0, 1.5]))
        return None
    if connection_type == "bess":
        return float(rng.choice([0.2, 0.5, 1.0]))
    if connection_type in {"solar", "wind", "hybrid"} and rng.random() < 0.25:
        return float(rng.choice([0.2, 0.5, 1.0]))
    if connection_type == "synchronous" and rng.random() < 0.35:
        return float(rng.choice([0.5, 1.0]))
    return None


def _generated_include_emt(
    profile: str,
    connection_type: str,
    rng: random.Random,
) -> bool:
    if profile == "m1m2":
        return False
    if profile == "hard_emt":
        return True
    if profile == "emt":
        return True
    if connection_type in {"solar", "wind", "bess", "hybrid"}:
        return rng.random() < 0.62
    return rng.random() < 0.16


def _generated_emt_status(connection: Mapping[str, Any], *, threshold: float = 3.0) -> str:
    connection_type = str(connection.get("connection_type", "")).lower()
    if connection_type not in {"solar", "wind", "bess", "hybrid"} and not bool(
        connection.get("is_ibr")
    ):
        return "skipped"
    p_mw = _float_or_default(connection.get("p_mw"), 0.0)
    q_mvar = _float_or_default(connection.get("q_mvar"), 0.0)
    project_mva = math.hypot(p_mw, q_mvar)
    if project_mva <= 0.0:
        return "fail"
    scr = _emt_template_short_circuit_mva(connection.get("bus")) / project_mva
    return _emt_template_status(
        scr=scr,
        threshold=threshold,
        borderline_margin=0.25,
    )


def _generated_recommendation(
    *,
    transient_required: bool,
    m2_status: str,
    emt_status: Optional[str],
) -> str:
    if transient_required and m2_status == "fail":
        return "reject"
    if m2_status == "borderline":
        return "borderline"
    if emt_status == "fail":
        return "reject"
    if emt_status == "borderline":
        return "borderline"
    return "approve"


def _unique_generated_id(
    *,
    index: int,
    seed: int,
    profile: str,
    connection_type: str,
    connection_bus: int,
    fault_bus: int,
    used_ids: set[str],
) -> str:
    base = (
        f"generated_ieee14_{profile}_{seed}_{index:04d}_"
        f"{connection_type}_bus{connection_bus}_fault{fault_bus}"
    )
    scenario_id = base
    suffix = 1
    while scenario_id in used_ids:
        suffix += 1
        scenario_id = f"{base}_{suffix}"
    used_ids.add(scenario_id)
    return scenario_id


def _generated_user_message(
    *,
    connection: Mapping[str, Any],
    disturbance: Mapping[str, Any],
    transient_required: bool,
    emt: Optional[Mapping[str, Any]],
    hard: bool = False,
    rng: random.Random,
) -> str:
    connection_type = str(connection.get("connection_type"))
    project = _generated_project_phrase(connection)
    start = float(disturbance["fault_start_s"])
    clearing = float(disturbance["clearing_time_s"])
    duration_ms = int(round((clearing - start) * 1000.0))
    if hard:
        style = rng.choice(["window", "duration_ms", "clear_at", "protection"])
    else:
        style = "window" if rng.random() < 0.5 else "duration_ms"
    if style == "window":
        fault_phrase = (
            f"a bus fault at bus {disturbance['bus']} from {start:g} s "
            f"to {clearing:g} s"
        )
    elif style == "duration_ms":
        fault_phrase = (
            f"a bus fault at bus {disturbance['bus']} starting at {start:g} s "
            f"and clearing after {duration_ms} ms"
        )
    elif style == "clear_at":
        fault_phrase = (
            f"a three-phase bus fault at bus {disturbance['bus']} applied at "
            f"t={start:g} s and cleared at t={clearing:g} s"
        )
    else:
        fault_phrase = (
            f"a bus fault on bus {disturbance['bus']} with protection clearing "
            f"{duration_ms} ms after t={start:g} s"
        )
    if transient_required:
        transient_phrase = f"survive {fault_phrase}"
    else:
        transient_phrase = (
            f"also report {fault_phrase} as advisory only, not required for approval"
        )

    if emt is None:
        tail = ""
    elif bool(emt.get("required_for_approval", False)):
        tail = f" and pass required EMT/SCR screening with SCR threshold {float(emt.get('scr_threshold', 3.0)):g}"
    elif connection_type == "load":
        tail = " and include EMT/SCR screening only if it is applicable"
    else:
        tail = " and report EMT/SCR screening as advisory only"

    templates = [
            "Can IEEE 14 accept {project} and {transient}{tail}?",
            "Assess whether IEEE 14 can connect {project}, {transient}{tail}.",
            "Run an integrated interconnection study for {project} on IEEE 14: {transient}{tail}.",
            "For IEEE 14, evaluate {project}; {transient}{tail}.",
    ]
    if hard:
        templates.extend(
            [
                "For the IEEE 14 benchmark, decide whether {project} is acceptable; the case must {transient}{tail}.",
                "Please screen {project} on IEEE 14 using M1, M2, and any applicable EMT/SCR checks: {transient}{tail}.",
                "Study the proposed POI request for {project}. The interconnection should {transient}{tail}.",
            ]
        )
    template = rng.choice(templates)
    final_instruction = " Give the final recommendation."
    if emt is not None:
        final_instruction = " Give the final recommendation and mention the EMT/SCR result."
    return template.format(project=project, transient=transient_phrase, tail=tail) + final_instruction


def _generated_project_phrase(connection: Mapping[str, Any]) -> str:
    connection_type = str(connection.get("connection_type", "project"))
    bus = connection.get("bus")
    p_mw = float(connection.get("p_mw", 0.0))
    q_mvar = connection.get("q_mvar")
    display_type = {
        "load": "data-center load",
        "solar": "solar project",
        "wind": "wind project",
        "bess": "BESS",
        "hybrid": "hybrid solar-storage project",
        "synchronous": "synchronous generator-style project",
    }.get(connection_type, f"{connection_type} project")
    if q_mvar is None:
        return f"a {p_mw:g} MW {display_type} at bus {bus}"
    return (
        f"a {p_mw:g} MW {display_type} at bus {bus} "
        f"with {float(q_mvar):g} MVAr reactive injection"
    )


def filter_joint_scenarios(
    scenarios: Sequence[JointBenchmarkScenario],
    *,
    scenario_ids: Iterable[str] = (),
    tags: Iterable[str] = (),
) -> List[JointBenchmarkScenario]:
    """Filter joint scenarios by id and/or tag."""

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


def evaluate_joint_result(
    scenario: JointBenchmarkScenario,
    *,
    agent_result: Any,
    oracle_result: Optional[Mapping[str, Any]],
) -> List[JointBenchmarkCheckResult]:
    """Evaluate one live agent result against a joint benchmark scenario."""

    checks = [
        _check_status(scenario, agent_result),
        _check_grounding(scenario, agent_result),
        _check_forbidden_tools(scenario, agent_result),
    ]
    expected_record = _best_expected_record(agent_result, scenario)
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


def run_joint_oracles(
    scenarios: Sequence[JointBenchmarkScenario],
    oracle_registry: Any,
    *,
    execute_tools: bool = False,
    timeout_s: Optional[float] = DEFAULT_LIVE_ORACLE_TIMEOUT_S,
) -> List[Dict[str, Any]]:
    """Run joint oracle checks.

    By default this is a metadata/template oracle, so it can run on laptops
    without pandapower or ANDES. Pass ``execute_tools=True`` on a runtime node
    to run real integrated M1+M2 oracle assessments.
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
            with contextlib.redirect_stdout(sys.stderr):
                result = _scenario_oracle_result(
                    scenario,
                    oracle_registry,
                    execute_tools=execute_tools,
                    timeout_s=timeout_s,
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


def _joint_scenario(
    *,
    scenario_id: str,
    user_message: str,
    description: str,
    connection: Mapping[str, Any],
    disturbance: Mapping[str, Any],
    case_path: str = "ieee14",
    transient_case_path: str = "ieee14_dynamic",
    required_for_approval: bool = True,
    simulation_time_s: float = 5.0,
    m2_status: str = "pass",
    m2_ok: bool = True,
    emt: Optional[Mapping[str, Any]] = None,
    emt_status: Optional[str] = None,
    recommendation: str = "approve",
    complete: bool = True,
    expected_tool_result: Optional[Mapping[str, Any]] = None,
    output_contains: Sequence[str] = (),
    forbidden_successful_tools: Sequence[str] = FORBIDDEN_JOINT_TOOLS,
    tags: Sequence[str] = (),
    strict_template_statuses: bool = True,
) -> JointBenchmarkScenario:
    transient = {
        "enabled": True,
        "required_for_approval": required_for_approval,
        "case_path": transient_case_path,
        "disturbance": dict(disturbance),
        "simulation_time_s": simulation_time_s,
        "max_samples": 20,
    }
    emt_payload = dict(emt) if emt is not None else None
    arguments = {
        "case_path": case_path,
        "connection": dict(connection),
        "transient": transient,
    }
    if emt_payload is not None:
        arguments["emt"] = emt_payload
    expected_args = {
        "case_path": case_path,
        "connection.bus": connection["bus"],
        "connection.p_mw": connection["p_mw"],
        "connection.connection_type": connection["connection_type"],
        "connection.is_ibr": connection["is_ibr"],
        "transient.enabled": True,
        "transient.required_for_approval": required_for_approval,
        "transient.case_path": transient_case_path,
        "transient.disturbance.type": disturbance["type"],
    }
    if "q_mvar" in connection:
        expected_args["connection.q_mvar"] = connection["q_mvar"]
    if emt_payload is not None:
        emt_enabled = bool(emt_payload.get("enabled", True))
        expected_args["emt.enabled"] = emt_enabled
        expected_args["emt.required_for_approval"] = bool(
            emt_payload.get("required_for_approval", emt_enabled)
        )
        if "scr_threshold" in emt_payload:
            expected_args["emt.scr_threshold"] = emt_payload["scr_threshold"]
    if disturbance["type"] == "bus_fault":
        expected_args.update(
            {
                "transient.disturbance.bus": disturbance["bus"],
                "transient.disturbance.fault_start_s": disturbance["fault_start_s"],
                "transient.disturbance.clearing_time_s": disturbance["clearing_time_s"],
            }
        )
    oracle_template = _integrated_template(
        arguments,
        m2_status=m2_status,
        m2_ok=m2_ok,
        emt_status=emt_status,
        recommendation=recommendation,
        complete=complete,
    )
    result_checks = {
        "stage_reports.0.stage": "m1_steady_state_cia",
        "stage_reports.0.status": ANY_VALUE,
        "stage_reports.1.stage": "m2_transient_stability",
        "stage_reports.1.status": ANY_VALUE,
        "summary.m1_recommendation": ANY_VALUE,
        "summary.m2_status": ANY_VALUE,
    }
    if m2_ok:
        result_checks.update(
            {
                "m2_result.dynamic_interconnection_modeling": True,
                "m2_result.connection_application.applied": True,
            }
        )
    if emt_payload is not None:
        found_emt_status, expected_emt_status = _resolve_path(oracle_template, "summary.emt_status")
        found_emt_applied, expected_emt_applied = _resolve_path(
            oracle_template,
            "linkage.emt_models_new_connection",
        )
        expected_status_value = (
            expected_emt_status
            if strict_template_statuses and found_emt_status
            else ANY_VALUE
        )
        result_checks.update(
            {
                "stage_reports.2.stage": "f4_emt_screening",
                "stage_reports.2.status": expected_status_value,
                "summary.emt_status": expected_status_value,
                "emt_result.emt.status": expected_status_value,
                "linkage.emt_models_new_connection": (
                    expected_emt_applied if found_emt_applied else ANY_VALUE
                ),
            }
        )
        if expected_emt_applied:
            result_checks.update(
                {
                    "emt_result.connection_application.applied": True,
                    "emt_result.metrics.scr": ANY_VALUE,
                    "linkage.emt_linkage": "post_connection_scr_screen",
                }
            )
        else:
            result_checks["linkage.emt_linkage"] = "emt_screen_not_applied"
    if expected_tool_result:
        result_checks.update(dict(expected_tool_result))
    comparisons = [
        JointOracleComparison("tool.recommendation", "oracle.recommendation"),
        JointOracleComparison("tool.complete", "oracle.complete"),
        JointOracleComparison("tool.summary.m1_recommendation", "oracle.summary.m1_recommendation"),
        JointOracleComparison("tool.summary.m2_status", "oracle.summary.m2_status"),
        JointOracleComparison(
            "tool.m2_result.dynamic_interconnection_modeling",
            "oracle.m2_result.dynamic_interconnection_modeling",
        ),
        JointOracleComparison(
            "tool.m2_result.connection_application.applied",
            "oracle.m2_result.connection_application.applied",
        ),
    ]
    if emt_payload is not None:
        comparisons.extend(
            [
                JointOracleComparison("tool.summary.emt_status", "oracle.summary.emt_status"),
                JointOracleComparison("tool.emt_result.emt.status", "oracle.emt_result.emt.status"),
                JointOracleComparison(
                    "tool.linkage.emt_models_new_connection",
                    "oracle.linkage.emt_models_new_connection",
                ),
            ]
        )
        found_applied, applied = _resolve_path(oracle_template, "linkage.emt_models_new_connection")
        if found_applied and applied:
            comparisons.extend(
                [
                    JointOracleComparison("tool.summary.emt_scr", "oracle.summary.emt_scr"),
                    JointOracleComparison("tool.emt_result.metrics.scr", "oracle.emt_result.metrics.scr"),
                ]
            )
    return JointBenchmarkScenario(
        scenario_id=scenario_id,
        user_message=user_message,
        description=description,
        expected_statuses=("completed",),
        expected_tool="run_integrated_assessment",
        expected_tool_ok=True,
        expected_tool_arguments=expected_args,
        expected_tool_result=result_checks,
        forbidden_successful_tools=forbidden_successful_tools,
        oracle_tool="run_integrated_assessment",
        oracle_arguments=arguments,
        oracle_result_template=oracle_template,
        oracle_comparisons=tuple(comparisons),
        output_contains=output_contains,
        tags=tags,
    )


def _integrated_template(
    arguments: Mapping[str, Any],
    *,
    m2_status: str,
    m2_ok: bool,
    recommendation: str,
    complete: bool,
    emt_status: Optional[str] = None,
) -> Dict[str, Any]:
    connection = dict(arguments.get("connection", {}))
    transient = dict(arguments.get("transient", {}))
    emt = dict(arguments.get("emt", {}))
    disturbance = dict(transient.get("disturbance", {}))
    transient_case_path = transient.get("case_path") or arguments.get("case_path")
    normalized_dynamic_case = _normalize_dynamic_case_name(transient_case_path)
    dynamic_available = normalized_dynamic_case != "unsupported_dynamic_case"
    m2_result = _m2_template(
        transient_case_path,
        disturbance=disturbance,
        connection=connection,
        simulation_time_s=transient.get("simulation_time_s", 5.0),
        ok=m2_ok,
        status=m2_status,
        dynamic_available=dynamic_available,
    )
    emt_result = _emt_template(
        arguments.get("case_path"),
        connection=connection,
        emt=emt,
        status=emt_status,
    )
    m2_models_connection = bool(m2_result.get("dynamic_interconnection_modeling"))
    emt_enabled = bool(emt.get("enabled", False))
    emt_required = bool(emt.get("required_for_approval", emt_enabled))
    emt_models_connection = bool(
        isinstance(emt_result, Mapping)
        and isinstance(emt_result.get("connection_application"), Mapping)
        and emt_result["connection_application"].get("applied")
    )
    m1_stage = {
        "stage": "m1_steady_state_cia",
        "status": "pass",
        "passed": True,
        "recommendation": "approve",
        "required_for_approval": True,
        "reason_codes": ["no_new_violations", "no_material_worsening"],
    }
    m2_stage = {
        "stage": "m2_transient_stability",
        "status": m2_status,
        "passed": m2_status == "pass",
        "required_for_approval": bool(transient.get("required_for_approval", True)),
        "case_path": transient_case_path,
        "reason_codes": _m2_reason_codes(m2_ok=m2_ok, status=m2_status),
        "result": m2_result,
        "dynamic_interconnection_modeling": m2_models_connection,
        "connection_model": m2_result.get("connection_model"),
        "connection_application": m2_result.get("connection_application"),
    }
    emt_stage = _emt_stage_template(
        emt=emt,
        emt_result=emt_result,
        required_for_approval=emt_required,
        emt_models_connection=emt_models_connection,
    )
    emt_stage_status = str(emt_stage.get("status"))
    reason_codes = [
        "integrated_assessment",
        "m1_no_new_violations",
        "m1_no_material_worsening",
        *[f"m2_{code}" for code in _m2_reason_codes(m2_ok=m2_ok, status=m2_status)],
    ]
    if emt_enabled:
        reason_codes.extend(
            f"emt_{code}" for code in _emt_reason_codes(emt_result, emt_stage_status)
        )
    if m2_status == "pass":
        reason_codes.append("m1_approved_m2_passed")
    else:
        reason_codes.append("required_m2_unavailable")
    if emt_enabled:
        if emt_stage_status in {"pass", "skipped"}:
            reason_codes.append("emt_screen_passed")
        elif emt_stage_status == "borderline":
            reason_codes.append("emt_borderline")
        elif emt_stage_status == "fail":
            reason_codes.append("emt_failed")
        elif emt_required:
            reason_codes.append("required_emt_unavailable")
    return {
        "ok": True,
        "tool": "run_integrated_assessment",
        "backend": {"m1": "pandapower", "m2": "andes", "emt": "pandapower"},
        "case_path": arguments.get("case_path"),
        "connection": connection,
        "recommendation": recommendation,
        "complete": complete,
        "reason_codes": reason_codes,
        "stage_reports": [m1_stage, m2_stage, emt_stage],
        "m1_result": {
            "ok": True,
            "tool": "run_cia",
            "case_path": arguments.get("case_path"),
            "connection": connection,
            "recommendation": "approve",
            "reason_codes": ["no_new_violations", "no_material_worsening"],
            "summary": {"status": "pass"},
        },
        "m2_result": m2_result,
        "emt_result": emt_result,
        "case_alignment": _case_alignment_template(
            arguments.get("case_path"),
            transient_case_path,
            m2_result,
        ),
        "m1_case_source": _m1_case_source_template(arguments.get("case_path")),
        "m2_case_source": _m2_case_source_template(m2_result),
        "limitations": _integrated_limitations(
            m2_models_connection=m2_models_connection,
            m2_status=m2_status,
            case_path=arguments.get("case_path"),
            emt_enabled=emt_enabled,
            emt_result=emt_result,
            emt_status=emt_stage_status,
        ),
        "linkage": {
            "m1_models_new_connection": True,
            "m2_models_new_connection": m2_models_connection,
            "emt_models_new_connection": emt_models_connection,
            "m2_linkage": (
                "dynamic_case_with_static_interconnection_model"
                if m2_models_connection
                else "dynamic_interconnection_model_not_applied"
            ),
            "emt_linkage": (
                "not_requested"
                if not emt_enabled
                else "post_connection_scr_screen"
                if emt_models_connection
                else "emt_screen_not_applied"
            ),
            "note": (
                "M2 v1 adds the proposed connection to the selected ANDES dynamic "
                "case as a static PQ load/injection before PFlow and TDS."
                if m2_models_connection
                else "M2 v1 did not apply a dynamic-case interconnection model."
            ),
        },
        "summary": {
            "m1_recommendation": "approve",
            "m2_status": m2_status,
            "m2_stability_status": m2_status if m2_status in {"pass", "borderline", "fail"} else "error",
            "transient_required_for_approval": bool(transient.get("required_for_approval", True)),
            "emt_status": emt_stage_status,
            "emt_scr": _nested_get(emt_result, "metrics.scr"),
            "emt_required_for_approval": emt_required,
        },
    }


def _m2_template(
    case_path: Any,
    *,
    disturbance: Mapping[str, Any],
    connection: Mapping[str, Any],
    simulation_time_s: Any,
    ok: bool,
    status: str,
    dynamic_available: bool,
) -> Dict[str, Any]:
    if not dynamic_available:
        return {
            "ok": False,
            "tool": "run_transient_stability",
            "backend": "andes",
            "case_path": case_path,
            "error_type": "dynamic_data_unavailable",
            "message": "M2 does not include validated dynamic data for this case.",
            "disturbance": dict(disturbance),
            "connection_model": None,
            "connection_application": {
                "requested": True,
                "applied": False,
                "reason": "dynamic_data_unavailable",
            },
            "dynamic_interconnection_modeling": False,
            "stability": {"status": "error"},
            "metrics": {"reason_codes": ["dynamic_data_unavailable"]},
            "limitations": ["dynamic_data_unavailable"],
        }
    connection_type = str(connection.get("connection_type", "")).lower()
    direction = "load" if connection_type == "load" else "generation"
    normalized_case = _normalize_dynamic_case_name(case_path)
    public_ieee118 = normalized_case == "ieee118_public_dynamic"
    limitations = [
        "m2_connection_model_is_static_pq",
        "m2_connection_does_not_add_machine_or_inverter_dynamics",
    ]
    if public_ieee118:
        limitations.append("ieee118_uses_public_benchmark_dynamic_data_not_customer_validated")
    return {
        "ok": ok,
        "tool": "run_transient_stability",
        "backend": "andes",
        "case_path": case_path,
        "case_info": {
            "case_id": normalized_case,
            "dynamic_data_available": True,
            **(
                {
                    "case_source": "powerfulcases_ieee118_public_dynamic",
                    "data_label": "public_benchmark_data_not_customer_validated",
                    "resolved_case_file": "/public/ieee118.raw",
                    "resolved_addfile": "/public/ieee118.dyr",
                }
                if public_ieee118
                else {}
            ),
        },
        "disturbance": dict(disturbance),
        "connection_model": {
            "bus": connection.get("bus"),
            "p_mw": connection.get("p_mw"),
            "q_mvar": connection.get("q_mvar", 0.0),
            "connection_type": connection_type,
            "is_ibr": connection.get("is_ibr"),
            "name": connection.get("name"),
            "andes_model": "PQ",
            "resource_direction": direction,
            "modeling_assumption": (
                "constant_power_load"
                if direction == "load"
                else "constant_power_generation_as_negative_pq_load"
            ),
            "dynamic_detail": "static_network_injection_no_machine_or_ibr_controls",
        },
        "connection_application": {
            "requested": True,
            "applied": True,
            "model": "PQ",
            "mode": (
                "static_load"
                if direction == "load"
                else "static_generation_as_negative_pq_load"
            ),
        },
        "dynamic_interconnection_modeling": True,
        "limitations": limitations,
        "simulation": {
            "simulation_time_s": simulation_time_s,
            "pflow_converged": True,
            "tds_converged": True,
            "sample_count": 3,
        },
        "stability": {"status": status},
        "metrics": {
            "tds_converged": True,
            "max_angle_spread_rad": 0.25,
            "final_angle_spread_rad": 0.1,
            "max_speed_deviation_pu": 0.01,
            "min_voltage_pu": 0.92,
            "reason_codes": _m2_reason_codes(m2_ok=ok, status=status),
        },
        "trajectories": {
            "time_s": [0.0, 1.0, simulation_time_s],
            "generator_angle_rad": {
                "columns": ["1", "2"],
                "values": [[0.0, 0.0], [0.1, 0.2], [0.1, 0.15]],
            },
            "generator_speed_pu": {
                "columns": ["1", "2"],
                "values": [[1.0, 1.0], [1.01, 0.99], [1.0, 1.0]],
            },
            "bus_voltage_pu": {
                "columns": ["1", "2"],
                "values": [[1.0, 1.0], [0.92, 0.98], [1.0, 1.0]],
            },
        },
    }


def _emt_template(
    case_path: Any,
    *,
    connection: Mapping[str, Any],
    emt: Mapping[str, Any],
    status: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not bool(emt.get("enabled", False)):
        return None

    connection_type = str(connection.get("connection_type", "")).lower()
    effective_ibr = bool(connection.get("is_ibr")) or connection_type in {
        "solar",
        "wind",
        "bess",
        "hybrid",
    }
    threshold = _float_or_default(emt.get("scr_threshold"), 3.0)
    margin = _float_or_default(emt.get("borderline_margin"), 0.25)
    p_mw = _float_or_default(connection.get("p_mw"), 0.0)
    q_mvar = _float_or_default(connection.get("q_mvar"), 0.0)
    project_mva = float(math.hypot(p_mw, q_mvar))
    emt_case_path = emt.get("case_path") or case_path

    if not effective_ibr:
        actual_status = status or "skipped"
        scr = None
        short_circuit_mva = None
        passed = actual_status == "skipped"
        reason_codes = ["emt_screen_not_required_for_non_ibr_connection"]
        connection_application = {
            "requested": True,
            "applied": False,
            "reason": "non_ibr_connection",
        }
    elif project_mva <= 0.0:
        actual_status = status or "fail"
        scr = None
        short_circuit_mva = None
        passed = False
        reason_codes = ["emt_project_mva_must_be_positive"]
        connection_application = {"requested": True, "applied": False}
    else:
        short_circuit_mva = _emt_template_short_circuit_mva(connection.get("bus"))
        scr = short_circuit_mva / project_mva
        actual_status = status or _emt_template_status(
            scr=scr,
            threshold=threshold,
            borderline_margin=margin,
        )
        passed = actual_status == "pass"
        if actual_status == "pass":
            reason_codes = ["emt_scr_above_threshold"]
        elif actual_status == "borderline":
            reason_codes = ["emt_scr_near_threshold"]
        else:
            reason_codes = ["emt_scr_below_threshold"]
        connection_application = {
            "requested": True,
            "applied": True,
            "mode": "post_connection_scr_screen",
        }

    return {
        "ok": True,
        "tool": "run_emt_screening",
        "backend": "pandapower",
        "screening_type": "scr_proxy",
        "case_path": emt_case_path,
        "connection": dict(connection),
        "poi_bus": {"requested_bus": connection.get("bus")},
        "connection_application": connection_application,
        "emt": {
            "status": actual_status,
            "passed": passed,
            "criterion": "scr_greater_equal_threshold",
        },
        "metrics": {
            "scr": scr,
            "short_circuit_mva": short_circuit_mva,
            "project_mva": project_mva,
            "threshold": threshold,
            "borderline_margin": margin,
            "reason_codes": reason_codes,
        },
        "powerflow_summary": {"converged": True},
        "limitations": [
            "emt_v1_scr_screen_only_no_time_domain_waveform",
            "emt_v1_no_inverter_switching_or_controller_dynamics",
            "emt_v1_uses_steady_state_thevenin_strength_proxy",
        ],
    }


def _emt_stage_template(
    *,
    emt: Mapping[str, Any],
    emt_result: Optional[Mapping[str, Any]],
    required_for_approval: bool,
    emt_models_connection: bool,
) -> Dict[str, Any]:
    if not bool(emt.get("enabled", False)):
        return {
            "stage": "f4_emt_screening",
            "status": "skipped",
            "passed": None,
            "required_for_approval": False,
            "reason_codes": ["emt_not_requested"],
            "result": None,
            "emt_interconnection_modeling": False,
        }

    status = str(_nested_get(emt_result, "emt.status") or "unavailable")
    if bool(_nested_get(emt_result, "ok")) and status in {"pass", "skipped"}:
        passed: Optional[bool] = True
    elif bool(_nested_get(emt_result, "ok")):
        passed = False
    else:
        passed = False if required_for_approval else None
        status = "unavailable"
    reason_codes = _emt_reason_codes(emt_result, status)
    return {
        "stage": "f4_emt_screening",
        "status": status,
        "passed": passed,
        "required_for_approval": required_for_approval,
        "case_path": _nested_get(emt_result, "case_path") or emt.get("case_path"),
        "reason_codes": reason_codes or ["emt_result_without_reason_codes"],
        "result": emt_result,
        "emt_interconnection_modeling": emt_models_connection,
        "connection_application": dict(_nested_get(emt_result, "connection_application") or {}),
    }


def _emt_template_short_circuit_mva(bus: Any) -> float:
    try:
        bus_id = int(bus)
    except Exception:
        bus_id = -1
    # Deterministic oracle proxy values for IEEE14-scale EMT/SCR screening.
    by_bus = {
        5: 52.0,
        6: 38.0,
        9: 42.0,
        10: 45.8,
        14: 30.0,
    }
    return by_bus.get(bus_id, 40.0)


def _emt_template_status(
    *,
    scr: float,
    threshold: float,
    borderline_margin: float,
) -> str:
    if scr >= threshold + borderline_margin:
        return "pass"
    if scr >= threshold:
        return "borderline"
    return "fail"


def _emt_reason_codes(
    emt_result: Optional[Mapping[str, Any]],
    status: str,
) -> List[str]:
    codes = _list_of_strings(_nested_get(emt_result, "metrics.reason_codes"))
    if codes:
        return codes
    if status == "pass":
        return ["emt_scr_above_threshold"]
    if status == "borderline":
        return ["emt_scr_near_threshold"]
    if status == "fail":
        return ["emt_scr_below_threshold"]
    if status == "skipped":
        return ["emt_not_requested"]
    return ["emt_screening_unavailable"]


def _m2_reason_codes(*, m2_ok: bool, status: str) -> List[str]:
    if not m2_ok:
        return ["dynamic_data_unavailable"]
    if status == "pass":
        return ["transient_metrics_within_thresholds"]
    if status == "borderline":
        return ["transient_metrics_near_thresholds"]
    if status == "fail":
        return ["transient_metric_threshold_exceeded"]
    return ["transient_status_unavailable"]


def _case_alignment_template(
    case_path: Any,
    transient_case_path: Any,
    m2_result: Mapping[str, Any],
) -> Dict[str, Any]:
    if (
        _normalize_dynamic_case_name(case_path) != "ieee118_public_dynamic"
        and _normalize_dynamic_case_name(transient_case_path) != "ieee118_public_dynamic"
    ):
        return {"strict_match": False, "source": None}
    return {
        "strict_match": True,
        "source": "powerfulcases_ieee118_public_dynamic",
        "available": True,
        "data_label": "public_benchmark_data_not_customer_validated",
        "m1_case_source": _m1_case_source_template(case_path),
        "m2_case_source": _m2_case_source_template(m2_result),
    }


def _m1_case_source_template(case_path: Any) -> Optional[Dict[str, Any]]:
    if _normalize_dynamic_case_name(case_path) != "ieee118_public_dynamic":
        return None
    return {
        "case_id": "ieee118_public_dynamic",
        "source": "powerfulcases_ieee118_public_dynamic",
        "raw_path": "/public/ieee118.raw",
        "dyr_path": "/public/ieee118.dyr",
        "data_label": "public_benchmark_data_not_customer_validated",
        "m1_static_conversion": "andes_to_pandapower",
    }


def _m2_case_source_template(m2_result: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    case_info = m2_result.get("case_info")
    if not isinstance(case_info, Mapping):
        return None
    if case_info.get("case_id") != "ieee118_public_dynamic":
        return None
    return {
        "case_id": "ieee118_public_dynamic",
        "case_source": case_info.get("case_source"),
        "resolved_case_file": case_info.get("resolved_case_file"),
        "resolved_addfile": case_info.get("resolved_addfile"),
        "data_label": case_info.get("data_label"),
    }


def _integrated_limitations(
    *,
    m2_models_connection: bool,
    m2_status: str,
    case_path: Any,
    emt_enabled: bool = False,
    emt_result: Optional[Mapping[str, Any]] = None,
    emt_status: str = "skipped",
) -> List[str]:
    limitations = []
    if m2_models_connection:
        limitations.extend(
            [
                "m2_connection_model_is_static_pq_not_full_dynamic_resource",
                "m2_connection_model_is_static_pq",
                "m2_connection_does_not_add_machine_or_inverter_dynamics",
            ]
        )
    else:
        limitations.append("m2_v1_does_not_model_new_connection_in_dynamic_case")
    if m2_status in {"not_ready", "unavailable", "skipped"}:
        limitations.append(m2_status)
    if not emt_enabled:
        limitations.append("emt_screening_not_requested")
    else:
        limitations.extend(_list_of_strings(_nested_get(emt_result, "limitations")))
        if emt_status in {"unavailable", "skipped"}:
            limitations.append(f"emt_{emt_status}")
    if _normalize_dynamic_case_name(case_path) == "ieee118_public_dynamic":
        limitations.append("ieee118_uses_public_benchmark_dynamic_data_not_customer_validated")
    return list(dict.fromkeys(limitations))


def _scenario_oracle_result(
    scenario: JointBenchmarkScenario,
    oracle_registry: Any,
    *,
    execute_tools: bool,
    timeout_s: Optional[float] = DEFAULT_LIVE_ORACLE_TIMEOUT_S,
) -> Optional[Dict[str, Any]]:
    if scenario.oracle_tool is None:
        return None
    if execute_tools:
        try:
            with contextlib.redirect_stdout(sys.stderr), _live_oracle_timeout(timeout_s):
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
    return _integrated_template(
        scenario.oracle_arguments,
        m2_status="pass",
        m2_ok=True,
        emt_status=None,
        recommendation="approve",
        complete=True,
    )


@contextlib.contextmanager
def _live_oracle_timeout(timeout_s: Optional[float]):
    if (
        timeout_s is None
        or timeout_s <= 0
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
    ):
        yield
        return

    def _handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"Live oracle timed out after {timeout_s:g} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_remaining = signal.alarm(0)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_remaining:
            signal.alarm(previous_remaining)


def _oracle_argument_checks(scenario: JointBenchmarkScenario) -> List[Dict[str, Any]]:
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
        passed = _argument_value_passed(
            root=scenario.oracle_arguments,
            path=path,
            found=found,
            actual=actual,
            expected=expected,
        )
        checks.append(
            {
                "name": f"oracle_argument:{path}",
                "passed": passed,
                "actual": actual if found else None,
                "expected": expected,
            }
        )
    return checks


def _oracle_path_checks(
    scenario: JointBenchmarkScenario,
    oracle_result: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    checks = []
    for path, expected in scenario.expected_tool_result.items():
        found, actual = _resolve_path(oracle_result, path)
        passed = found and _value_matches(actual, expected, path=path)
        checks.append(
            {
                "name": f"oracle_expected_result:{path}",
                "passed": passed,
                "actual": actual if found else None,
                "expected": expected,
            }
        )
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
    scenario: JointBenchmarkScenario,
    agent_result: Any,
) -> JointBenchmarkCheckResult:
    actual = getattr(agent_result, "status", None)
    passed = actual in scenario.expected_statuses
    return JointBenchmarkCheckResult(
        name="status",
        passed=passed,
        actual=actual,
        expected=list(scenario.expected_statuses),
        message=None if passed else "Agent status was not in the allowed set.",
    )


def _check_grounding(
    scenario: JointBenchmarkScenario,
    agent_result: Any,
) -> JointBenchmarkCheckResult:
    grounding = getattr(agent_result, "grounding", None)
    actual = getattr(grounding, "warning_appended", None)
    expected = scenario.expect_grounding_warning
    passed = actual == expected
    return JointBenchmarkCheckResult(
        name="grounding_warning",
        passed=passed,
        actual=actual,
        expected=expected,
        message=None if passed else "Unexpected numeric-grounding warning state.",
    )


def _check_no_successful_solver_tools(agent_result: Any) -> JointBenchmarkCheckResult:
    records = [
        record
        for record in getattr(agent_result, "tool_records", [])
        if bool(getattr(record, "ok", False))
    ]
    passed = len(records) == 0
    return JointBenchmarkCheckResult(
        name="no_successful_solver_tools",
        passed=passed,
        actual=[getattr(record, "name", None) for record in records],
        expected=[],
        message=None if passed else "Scenario expected clarification without successful solver execution.",
    )


def _check_expected_tool(
    scenario: JointBenchmarkScenario,
    agent_result: Any,
    expected_record: Any,
) -> JointBenchmarkCheckResult:
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
    return JointBenchmarkCheckResult(
        name="expected_tool",
        passed=passed,
        actual=actual,
        expected={"name": scenario.expected_tool, "ok": scenario.expected_tool_ok},
        message=None if passed else "Expected integrated assessment tool state was not observed.",
    )


def _check_unexpected_tools(
    scenario: JointBenchmarkScenario,
    agent_result: Any,
) -> JointBenchmarkCheckResult:
    successful = [
        str(getattr(record, "name", ""))
        for record in getattr(agent_result, "tool_records", [])
        if bool(getattr(record, "ok", False))
    ]
    expected_success = [scenario.expected_tool] if scenario.expected_tool_ok else []
    allowed = set(expected_success).union(ALLOWED_AUXILIARY_JOINT_TOOLS)
    unexpected = [tool for tool in successful if tool not in allowed]
    passed = not unexpected
    return JointBenchmarkCheckResult(
        name="unexpected_successful_tools",
        passed=passed,
        actual=successful,
        expected=expected_success,
        message=None if passed else "Agent executed successful tools outside the expected integrated tool.",
    )


def _check_forbidden_tools(
    scenario: JointBenchmarkScenario,
    agent_result: Any,
) -> JointBenchmarkCheckResult:
    forbidden = set(str(tool) for tool in scenario.forbidden_successful_tools)
    successful_forbidden = [
        str(getattr(record, "name", ""))
        for record in getattr(agent_result, "tool_records", [])
        if bool(getattr(record, "ok", False)) and str(getattr(record, "name", "")) in forbidden
    ]
    passed = not successful_forbidden
    return JointBenchmarkCheckResult(
        name="forbidden_successful_tools",
        passed=passed,
        actual=successful_forbidden,
        expected=[],
        message=None if passed else "Joint prompt was answered with an M1-only or M2-only tool.",
    )


def _check_expected_arguments(
    scenario: JointBenchmarkScenario,
    expected_record: Any,
) -> List[JointBenchmarkCheckResult]:
    checks = []
    for path, expected in scenario.expected_tool_arguments.items():
        if expected_record is None:
            checks.append(
                JointBenchmarkCheckResult(
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
        passed = _argument_value_passed(
            root=arguments,
            path=path,
            found=found,
            actual=actual,
            expected=expected,
        )
        checks.append(
            JointBenchmarkCheckResult(
                name=f"argument:{path}",
                passed=passed,
                actual=actual if found else None,
                expected=expected,
                message=None if passed else "Tool argument did not match expected value.",
            )
        )
    return checks


def _check_expected_result(
    scenario: JointBenchmarkScenario,
    expected_record: Any,
) -> List[JointBenchmarkCheckResult]:
    checks = []
    for path, expected in scenario.expected_tool_result.items():
        if expected_record is None:
            checks.append(
                JointBenchmarkCheckResult(
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
            JointBenchmarkCheckResult(
                name=f"result:{path}",
                passed=passed,
                actual=actual if found else None,
                expected=expected,
                message=None if passed else "Tool result did not match expected value.",
            )
        )
    return checks


def _check_oracle(
    scenario: JointBenchmarkScenario,
    expected_record: Any,
    oracle_result: Optional[Mapping[str, Any]],
) -> List[JointBenchmarkCheckResult]:
    if scenario.oracle_tool is None:
        return []
    checks = []
    oracle_ok = bool(isinstance(oracle_result, Mapping) and oracle_result.get("ok", False))
    checks.append(
        JointBenchmarkCheckResult(
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
            JointBenchmarkCheckResult(
                name=f"oracle:{comparison.tool_path}",
                passed=passed,
                actual=actual if found_tool else None,
                expected=expected if found_oracle else None,
                message=None if passed else "Agent tool output missed an expected oracle/template path.",
            )
        )
    return checks


def _check_output_contains(
    scenario: JointBenchmarkScenario,
    agent_result: Any,
) -> List[JointBenchmarkCheckResult]:
    output_text = str(getattr(agent_result, "output_text", ""))
    normalized_output = _normalize_text_for_contains(output_text)
    checks = []
    for expected_text in scenario.output_contains:
        expected = str(expected_text)
        passed = _normalize_text_for_contains(expected) in normalized_output
        checks.append(
            JointBenchmarkCheckResult(
                name=f"output_contains:{expected}",
                passed=passed,
                actual=output_text,
                expected=expected,
                message=None if passed else "Expected text was not found in the agent output.",
            )
        )
    return checks


def _best_expected_record(agent_result: Any, scenario: JointBenchmarkScenario) -> Any:
    expected_tool = scenario.expected_tool
    if expected_tool is None:
        return None
    records = [
        record
        for record in getattr(agent_result, "tool_records", [])
        if getattr(record, "name", None) == expected_tool
    ]
    for record in reversed(records):
        if bool(getattr(record, "ok", False)) == scenario.expected_tool_ok:
            return record
    return records[-1] if records else None


def _nested_get(root: Any, path: str) -> Any:
    found, value = _resolve_path(root, path)
    return value if found else None


def _list_of_strings(value: Any) -> List[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if item is not None]


def _float_or_default(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


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


def _argument_value_passed(
    *,
    root: Mapping[str, Any],
    path: str,
    found: bool,
    actual: Any,
    expected: Any,
) -> bool:
    if path == "transient.case_path" and not found:
        found_case, case_path = _resolve_path(root, "case_path")
        return found_case and _normalize_dynamic_case_name(case_path) == _normalize_dynamic_case_name(expected)
    if path == "emt.required_for_approval" and not found:
        found_default, default = _default_emt_required_for_approval(root)
        return found_default and _value_matches(default, expected, path=path)
    if not found and path in DEFAULTABLE_ARGUMENTS:
        if path.startswith("emt."):
            found_emt, _emt = _resolve_path(root, "emt")
            if not found_emt:
                return False
        return _value_matches(DEFAULTABLE_ARGUMENTS[path], expected, path=path)
    if found and path.endswith("clearing_time_s"):
        return _clearing_time_matches(root=root, path=path, actual=actual, expected=expected)
    return found and _value_matches(actual, expected, path=path)


def _default_emt_required_for_approval(root: Mapping[str, Any]) -> Tuple[bool, bool]:
    found_emt, emt_payload = _resolve_path(root, "emt")
    if not found_emt or not isinstance(emt_payload, Mapping):
        return False, False
    found_enabled, enabled = _resolve_path(root, "emt.enabled")
    if not found_enabled:
        # EMTAssessmentRequest.from_mapping defaults enabled=True when an EMT
        # block is present, then defaults required_for_approval to enabled.
        return True, True
    return True, bool(enabled)


def _value_matches(
    actual: Any,
    expected: Any,
    *,
    tolerance: Optional[float] = None,
    path: str = "",
) -> bool:
    if _is_any_expected(expected):
        return actual is not None
    if path.endswith("case_path"):
        return _normalize_dynamic_case_name(actual) == _normalize_dynamic_case_name(expected)
    if path.endswith("source") and expected == IEEE118_PUBLIC_SOURCE:
        return actual in IEEE118_ACCEPTED_PUBLIC_SOURCES
    if isinstance(expected, float) or tolerance is not None:
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False
        tol = 1e-9 if tolerance is None else tolerance
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol)
    return actual == expected


def _clearing_time_matches(
    *,
    root: Mapping[str, Any],
    path: str,
    actual: Any,
    expected: Any,
) -> bool:
    if _value_matches(actual, expected, path=path):
        return True
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        return False
    if not isinstance(expected, (int, float)) or isinstance(expected, bool):
        return False

    found_start, fault_start = _resolve_path(
        root,
        path[: -len("clearing_time_s")] + "fault_start_s",
    )
    if not found_start:
        return False
    if not isinstance(fault_start, (int, float)) or isinstance(fault_start, bool):
        return False

    actual_value = float(actual)
    expected_value = float(expected)
    start = float(fault_start)
    tol = 1e-9
    actual_candidates = [actual_value]
    expected_candidates = [expected_value]
    if actual_value <= start:
        actual_candidates.append(start + actual_value)
    if expected_value <= start:
        expected_candidates.append(start + expected_value)
    return any(
        math.isclose(a, e, rel_tol=0.0, abs_tol=tol)
        for a in actual_candidates
        for e in expected_candidates
    )


def _normalize_text_for_contains(value: Any) -> str:
    text = str(value).lower()
    text = text.replace("ieee118", "ieee 118")
    text = text.replace("ieee-118", "ieee 118")
    text = text.replace("ieee_118", "ieee 118")
    text = text.replace("static-pq", "static pq")
    text = text.replace("static_pq", "static pq")
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _is_any_expected(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and len(value) == 1
        and bool(value.get("__joint_benchmark_any_value__", False))
    )


def _normalize_dynamic_case_name(value: Any) -> Any:
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
        "case_path": oracle_result.get("case_path"),
        "recommendation": oracle_result.get("recommendation"),
        "complete": oracle_result.get("complete"),
    }
    summary_payload = oracle_result.get("summary")
    if isinstance(summary_payload, Mapping):
        summary["m1_recommendation"] = summary_payload.get("m1_recommendation")
        summary["m2_status"] = summary_payload.get("m2_status")
        summary["emt_status"] = summary_payload.get("emt_status")
        summary["emt_scr"] = summary_payload.get("emt_scr")
    m2 = oracle_result.get("m2_result")
    if isinstance(m2, Mapping):
        summary["m2_error_type"] = m2.get("error_type")
        stability = m2.get("stability")
        if isinstance(stability, Mapping):
            summary["m2_stability_status"] = stability.get("status")
        metrics = m2.get("metrics")
        if isinstance(metrics, Mapping):
            summary["m2_reason_codes"] = metrics.get("reason_codes")
    emt = oracle_result.get("emt_result")
    if isinstance(emt, Mapping):
        emt_payload = emt.get("emt")
        if isinstance(emt_payload, Mapping):
            summary["emt_tool_status"] = emt_payload.get("status")
        metrics = emt.get("metrics")
        if isinstance(metrics, Mapping):
            summary["emt_reason_codes"] = metrics.get("reason_codes")
    return summary

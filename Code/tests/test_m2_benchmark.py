"""Tests for the M2 transient-stability benchmark."""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Mapping, Optional

from gridmind_mini import (
    AgentTurnResult,
    GroundingValidation,
    M2BenchmarkRunner,
    M2BenchmarkScenario,
    M2OracleComparison,
    ToolExecutionRecord,
    default_m2_benchmark_scenarios,
    evaluate_m2_result,
    filter_m2_scenarios,
    run_m2_oracles,
)


class FakeAgent:
    def __init__(self, result: AgentTurnResult) -> None:
        self.result = result
        self.calls: List[Dict[str, Any]] = []

    def run_turn(
        self,
        message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> AgentTurnResult:
        self.calls.append({"message": message, "context": context})
        return self.result


class FakeOracleRegistry:
    def __init__(self, result: Dict[str, Any]) -> None:
        self.result = result
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        result = dict(self.result)
        result.setdefault("tool", name)
        return result


def agent_result(
    *,
    status: str = "completed",
    records: Optional[List[ToolExecutionRecord]] = None,
    invoked_tools: Optional[List[str]] = None,
    output_text: str = "Transient stability status pass; max angle spread is grounded.",
    warning_appended: bool = False,
) -> AgentTurnResult:
    return AgentTurnResult(
        status=status,
        output_text=output_text,
        raw_output_text=output_text,
        messages=[],
        tool_records=records or [],
        invoked_tools=invoked_tools or [],
        grounding=GroundingValidation(
            tool_grounded=bool(invoked_tools),
            claims=[],
            ungrounded_claims=[],
            warning_appended=warning_appended,
            output_text=output_text,
            warning=None,
        ),
    )


def tool_record(
    name: str,
    arguments: Dict[str, Any],
    result: Dict[str, Any],
    *,
    ok: bool = True,
) -> ToolExecutionRecord:
    return ToolExecutionRecord(
        call_id="call_1",
        name=name,
        arguments=arguments,
        source="native",
        ok=ok,
        result=result,
        error=None if ok else "tool failed",
        observation={"tool": name},
    )


class M2BenchmarkTest(unittest.TestCase):
    def test_default_suite_has_ten_unique_scenarios(self) -> None:
        scenarios = default_m2_benchmark_scenarios()

        self.assertEqual(len(scenarios), 10)
        self.assertEqual(len({scenario.scenario_id for scenario in scenarios}), 10)
        self.assertIn(
            "m2_ieee118_public_dynamic_bus_fault",
            {scenario.scenario_id for scenario in scenarios},
        )
        self.assertIn(
            "m2_static_interconnection_bus_fault",
            {scenario.scenario_id for scenario in scenarios},
        )

    def test_filter_by_tag_and_id(self) -> None:
        scenarios = default_m2_benchmark_scenarios()

        by_id = filter_m2_scenarios(
            scenarios,
            scenario_ids=["m2_kundur_bus_fault_100ms"],
        )
        by_tag = filter_m2_scenarios(scenarios, tags=["public_dynamic"])

        self.assertEqual([scenario.scenario_id for scenario in by_id], ["m2_kundur_bus_fault_100ms"])
        self.assertEqual([scenario.scenario_id for scenario in by_tag], ["m2_ieee118_public_dynamic_bus_fault"])

    def test_evaluate_transient_happy_path(self) -> None:
        scenario = M2BenchmarkScenario(
            scenario_id="m2",
            user_message="Run transient stability on Kundur.",
            description="test",
            expected_statuses=("completed",),
            expected_tool="run_transient_stability",
            expected_tool_arguments={
                "case_path": "kundur_full",
                "disturbance.type": "bus_fault",
                "disturbance.bus": 5,
                "disturbance.clearing_time_s": 1.1,
            },
            oracle_tool="run_transient_stability",
            oracle_arguments={
                "case_path": "kundur_full",
                "disturbance": {
                    "type": "bus_fault",
                    "bus": 5,
                    "fault_start_s": 1.0,
                    "clearing_time_s": 1.1,
                },
            },
            oracle_result_template={
                "ok": True,
                "tool": "run_transient_stability",
                "stability": {"status": "pass"},
                "metrics": {"max_angle_spread_rad": 0.2},
            },
            oracle_comparisons=(
                M2OracleComparison("tool.stability.status", "oracle.stability.status"),
                M2OracleComparison("tool.metrics.max_angle_spread_rad", "oracle.metrics.max_angle_spread_rad"),
            ),
        )
        record = tool_record(
            "run_transient_stability",
            {
                "case_path": "kundur",
                "disturbance": {
                    "type": "bus_fault",
                    "bus": 5,
                    "fault_start_s": 1.0,
                    "clearing_time_s": 1.1,
                },
            },
            {
                "ok": True,
                "tool": "run_transient_stability",
                "stability": {"status": "pass"},
                "metrics": {"max_angle_spread_rad": 0.2},
            },
        )
        result = agent_result(records=[record], invoked_tools=["run_transient_stability"])

        checks = evaluate_m2_result(
            scenario,
            agent_result=result,
            oracle_result=scenario.oracle_result_template,
        )

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_evaluate_catches_100ms_conversion_mismatch(self) -> None:
        scenario = next(
            item
            for item in default_m2_benchmark_scenarios()
            if item.scenario_id == "m2_kundur_bus_fault_100ms"
        )
        record = tool_record(
            "run_transient_stability",
            {
                "case_path": "kundur_full",
                "disturbance": {
                    "type": "bus_fault",
                    "bus": 5,
                    "fault_start_s": 0.0,
                    "clearing_time_s": 100.0,
                },
            },
            {"ok": True, "tool": "run_transient_stability", "stability": {"status": "pass"}, "metrics": {}},
        )
        result = agent_result(records=[record], invoked_tools=["run_transient_stability"])

        checks = evaluate_m2_result(
            scenario,
            agent_result=result,
            oracle_result=dict(scenario.oracle_result_template),
        )
        failed = [check for check in checks if not check.passed]

        self.assertTrue(any(check.name == "argument:disturbance.clearing_time_s" for check in failed))

    def test_ieee118_public_dynamic_template_expects_supported_transient_tool(self) -> None:
        scenario = next(
            item
            for item in default_m2_benchmark_scenarios()
            if item.scenario_id == "m2_ieee118_public_dynamic_bus_fault"
        )
        record = tool_record(
            "run_transient_stability",
            dict(scenario.oracle_arguments),
            dict(scenario.oracle_result_template),
        )
        result = agent_result(
            records=[record],
            invoked_tools=["run_transient_stability"],
            output_text="IEEE 118 public dynamic transient stability passed with a grounded metric.",
        )

        checks = evaluate_m2_result(
            scenario,
            agent_result=result,
            oracle_result=dict(scenario.oracle_result_template),
        )

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_clarification_case_invokes_no_solver_tool(self) -> None:
        scenario = next(
            item
            for item in default_m2_benchmark_scenarios()
            if item.scenario_id == "m2_missing_clearing_time"
        )
        result = agent_result(
            status="clarification_required",
            output_text="Please provide the fault clearing time.",
        )

        checks = evaluate_m2_result(scenario, agent_result=result, oracle_result=None)

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_oracle_only_metadata_detects_bad_metric_path(self) -> None:
        scenario = M2BenchmarkScenario(
            scenario_id="bad_path",
            user_message="Run transient stability on Kundur.",
            description="test",
            expected_statuses=("completed",),
            expected_tool="run_transient_stability",
            oracle_tool="run_transient_stability",
            oracle_arguments={
                "case_path": "kundur_full",
                "disturbance": {
                    "type": "bus_fault",
                    "bus": 5,
                    "fault_start_s": 1.0,
                    "clearing_time_s": 1.1,
                },
            },
            oracle_result_template={"ok": True, "tool": "run_transient_stability"},
            oracle_comparisons=(
                M2OracleComparison("tool.metrics.missing", "oracle.ok"),
            ),
        )
        result = run_m2_oracles([scenario], FakeOracleRegistry({}), execute_tools=False)[0]

        self.assertFalse(result["ok"])
        self.assertFalse(result["oracle_path_checks"][0]["passed"])

    def test_oracle_only_metadata_detects_bad_expected_argument(self) -> None:
        scenario = M2BenchmarkScenario(
            scenario_id="bad_args",
            user_message="Run transient stability on Kundur.",
            description="test",
            expected_statuses=("completed",),
            expected_tool="run_transient_stability",
            expected_tool_arguments={"disturbance.clearing_time_s": 0.1},
            oracle_tool="run_transient_stability",
            oracle_arguments={
                "case_path": "kundur_full",
                "disturbance": {
                    "type": "bus_fault",
                    "bus": 5,
                    "fault_start_s": 0.0,
                    "clearing_time_s": 1.1,
                },
            },
            oracle_result_template={"ok": True, "tool": "run_transient_stability"},
        )
        result = run_m2_oracles([scenario], FakeOracleRegistry({}), execute_tools=False)[0]

        self.assertFalse(result["ok"])
        failed = [
            check
            for check in result["oracle_argument_checks"]
            if not check["passed"]
        ]
        self.assertEqual(failed[0]["name"], "oracle_argument:disturbance.clearing_time_s")

    def test_runner_calls_agent_and_metadata_oracle(self) -> None:
        scenario = next(
            item
            for item in default_m2_benchmark_scenarios()
            if item.scenario_id == "m2_kundur_bus_fault_seconds"
        )
        record = tool_record(
            "run_transient_stability",
            {
                "case_path": "kundur_full",
                "disturbance": {
                    "type": "bus_fault",
                    "bus": 5,
                    "fault_start_s": 1.0,
                    "clearing_time_s": 1.1,
                },
            },
            dict(scenario.oracle_result_template),
        )
        fake_agent = FakeAgent(agent_result(records=[record], invoked_tools=["run_transient_stability"]))
        fake_oracle = FakeOracleRegistry({"ok": True})

        result = M2BenchmarkRunner(fake_agent, fake_oracle).run_scenario(scenario)

        self.assertTrue(result.passed, [check.to_dict() for check in result.check_results])
        self.assertEqual(fake_oracle.calls, [])
        self.assertEqual(fake_agent.calls[0]["message"], scenario.user_message)

    def test_live_oracle_option_executes_registry_tool(self) -> None:
        scenario = next(
            item
            for item in default_m2_benchmark_scenarios()
            if item.scenario_id == "m2_kundur_bus_fault_seconds"
        )
        fake_oracle = FakeOracleRegistry(dict(scenario.oracle_result_template))

        result = run_m2_oracles([scenario], fake_oracle, execute_tools=True)[0]

        self.assertTrue(result["ok"])
        self.assertEqual(fake_oracle.calls[0][0], "run_transient_stability")


if __name__ == "__main__":
    unittest.main()

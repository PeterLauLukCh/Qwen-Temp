import unittest
from typing import Any, Dict, List, Mapping, Optional

from gridmind_mini import (
    AgentTurnResult,
    GroundingValidation,
    M1BenchmarkRunner,
    M1BenchmarkScenario,
    M1OracleComparison,
    ToolExecutionRecord,
    default_m1_benchmark_scenarios,
    evaluate_m1_result,
    filter_m1_scenarios,
    run_m1_oracles,
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
        return dict(self.result)


def agent_result(
    *,
    status: str = "completed",
    records: Optional[List[ToolExecutionRecord]] = None,
    invoked_tools: Optional[List[str]] = None,
    output_text: str = "Grounded solver-backed answer.",
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


class M1BenchmarkTest(unittest.TestCase):
    def test_default_suite_has_ten_unique_scenarios(self) -> None:
        scenarios = default_m1_benchmark_scenarios()

        self.assertEqual(len(scenarios), 10)
        self.assertEqual(len({scenario.scenario_id for scenario in scenarios}), 10)
        self.assertIn(
            "m1_ieee118_missing_bus_clarification",
            {scenario.scenario_id for scenario in scenarios},
        )
        self.assertIn(
            "m1_ieee14_contingency_limited",
            {scenario.scenario_id for scenario in scenarios},
        )

    def test_filter_by_tag_and_id(self) -> None:
        scenarios = default_m1_benchmark_scenarios()

        by_id = filter_m1_scenarios(
            scenarios,
            scenario_ids=["m1_ieee118_load_cia"],
        )
        by_tag = filter_m1_scenarios(scenarios, tags=["capacity"])

        self.assertEqual([scenario.scenario_id for scenario in by_id], ["m1_ieee118_load_cia"])
        self.assertEqual([scenario.scenario_id for scenario in by_tag], ["m1_ieee14_load_capacity"])

    def test_evaluate_complete_cia_happy_path(self) -> None:
        scenario = M1BenchmarkScenario(
            scenario_id="cia",
            user_message="Can we connect a 5 MW load at bus 10 on IEEE 118?",
            description="test",
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
            oracle_arguments={"case_path": "ieee118", "connection": {}},
            oracle_comparisons=(
                M1OracleComparison("tool.recommendation", "oracle.recommendation"),
            ),
        )
        record = tool_record(
            "run_cia",
            {
                "case_path": "IEEE 118",
                "connection": {
                    "bus": 10,
                    "p_mw": 5,
                    "connection_type": "load",
                    "is_ibr": False,
                },
            },
            {"ok": True, "tool": "run_cia", "recommendation": "approve"},
        )
        result = agent_result(records=[record], invoked_tools=["run_cia"])

        checks = evaluate_m1_result(
            scenario,
            agent_result=result,
            oracle_result={"ok": True, "tool": "run_cia", "recommendation": "approve"},
        )

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_evaluate_catches_argument_mismatch(self) -> None:
        scenario = M1BenchmarkScenario(
            scenario_id="cia",
            user_message="Can we connect a 5 MW load at bus 10 on IEEE 118?",
            description="test",
            expected_statuses=("completed",),
            expected_tool="run_cia",
            expected_tool_arguments={"connection.bus": 10},
        )
        record = tool_record(
            "run_cia",
            {"connection": {"bus": 11}},
            {"ok": True, "tool": "run_cia"},
        )
        result = agent_result(records=[record], invoked_tools=["run_cia"])

        checks = evaluate_m1_result(scenario, agent_result=result, oracle_result=None)
        failed = [check for check in checks if not check.passed]

        self.assertTrue(any(check.name == "argument:connection.bus" for check in failed))

    def test_evaluate_catches_extra_successful_tool(self) -> None:
        scenario = M1BenchmarkScenario(
            scenario_id="cia",
            user_message="Can we connect a 5 MW load at bus 10 on IEEE 118?",
            description="test",
            expected_statuses=("completed",),
            expected_tool="run_cia",
            expected_tool_arguments={"connection.bus": 10},
        )
        extra_record = tool_record(
            "query_network_data",
            {"case_path": "ieee118"},
            {"ok": True, "tool": "query_network_data"},
        )
        cia_record = tool_record(
            "run_cia",
            {"connection": {"bus": 10}},
            {"ok": True, "tool": "run_cia"},
        )
        result = agent_result(
            records=[extra_record, cia_record],
            invoked_tools=["query_network_data", "run_cia"],
        )

        checks = evaluate_m1_result(scenario, agent_result=result, oracle_result=None)
        failed = [check for check in checks if not check.passed]

        self.assertTrue(any(check.name == "unexpected_tools" for check in failed))

    def test_runner_calls_oracle_and_agent(self) -> None:
        scenario = M1BenchmarkScenario(
            scenario_id="pf",
            user_message="Run power flow on IEEE 118.",
            description="test",
            expected_statuses=("completed",),
            expected_tool="run_powerflow",
            expected_tool_arguments={"case_path": "ieee118"},
            oracle_tool="run_powerflow",
            oracle_arguments={"case_path": "ieee118"},
            oracle_comparisons=(
                M1OracleComparison("tool.powerflow_summary.converged", "oracle.powerflow_summary.converged"),
            ),
        )
        record = tool_record(
            "run_powerflow",
            {"case_path": "ieee118"},
            {"ok": True, "tool": "run_powerflow", "powerflow_summary": {"converged": True}},
        )
        fake_agent = FakeAgent(agent_result(records=[record], invoked_tools=["run_powerflow"]))
        fake_oracle = FakeOracleRegistry(
            {"ok": True, "tool": "run_powerflow", "powerflow_summary": {"converged": True}}
        )

        result = M1BenchmarkRunner(fake_agent, fake_oracle).run_scenario(scenario)

        self.assertTrue(result.passed, [check.to_dict() for check in result.check_results])
        self.assertEqual(fake_oracle.calls, [("run_powerflow", {"case_path": "ieee118"})])
        self.assertEqual(fake_agent.calls[0]["message"], scenario.user_message)

    def test_oracle_only_fails_when_comparison_path_is_missing(self) -> None:
        scenario = M1BenchmarkScenario(
            scenario_id="bad_path",
            user_message="Run power flow on IEEE 118.",
            description="test",
            expected_statuses=("completed",),
            expected_tool="run_powerflow",
            oracle_tool="run_powerflow",
            oracle_arguments={"case_path": "ieee118"},
            oracle_comparisons=(
                M1OracleComparison("tool.missing", "oracle.ok"),
            ),
        )
        fake_oracle = FakeOracleRegistry({"ok": True, "tool": "run_powerflow"})

        result = run_m1_oracles([scenario], fake_oracle)[0]

        self.assertFalse(result["ok"])
        self.assertFalse(result["oracle_path_checks"][0]["passed"])

    def test_oracle_only_fails_when_expected_arguments_do_not_match_oracle(self) -> None:
        scenario = M1BenchmarkScenario(
            scenario_id="bad_args",
            user_message="Run CIA for a 5 MW load at bus 10 on IEEE 118.",
            description="test",
            expected_statuses=("completed",),
            expected_tool="run_cia",
            expected_tool_arguments={"connection.bus": 10},
            oracle_tool="run_cia",
            oracle_arguments={"connection": {"bus": 11}},
        )
        fake_oracle = FakeOracleRegistry({"ok": True, "tool": "run_cia"})

        result = run_m1_oracles([scenario], fake_oracle)[0]

        self.assertFalse(result["ok"])
        failed = [
            check
            for check in result["oracle_argument_checks"]
            if not check["passed"]
        ]
        self.assertEqual(failed[0]["name"], "oracle_argument:connection.bus")


if __name__ == "__main__":
    unittest.main()

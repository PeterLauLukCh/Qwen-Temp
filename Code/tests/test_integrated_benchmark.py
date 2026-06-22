"""Tests for integrated M1+M2 benchmark orchestration."""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Mapping, Optional

from gridmind_mini import (
    AgentTurnResult,
    GroundingValidation,
    M1BenchmarkScenario,
    M2BenchmarkScenario,
    run_integrated_live_agent,
    run_integrated_oracles,
)


class FakeOracleRegistry:
    def __init__(self) -> None:
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        return {"ok": True, "tool": name, **dict(arguments)}


class FakeAgent:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def run_turn(
        self,
        message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> AgentTurnResult:
        self.calls.append({"message": message, "context": context})
        return AgentTurnResult(
            status="completed",
            output_text="Clarification-style benchmark answer.",
            raw_output_text="Clarification-style benchmark answer.",
            messages=[],
            tool_records=[],
            invoked_tools=[],
            grounding=GroundingValidation(
                tool_grounded=False,
                claims=[],
                ungrounded_claims=[],
                warning_appended=False,
                output_text="Clarification-style benchmark answer.",
                warning=None,
            ),
        )


class IntegratedBenchmarkTest(unittest.TestCase):
    def test_integrated_oracle_run_combines_m1_and_m2_sections(self) -> None:
        m1 = M1BenchmarkScenario(
            scenario_id="m1_meta",
            user_message="Run power flow.",
            description="test",
            expected_statuses=("completed",),
            expected_tool="run_powerflow",
            expected_tool_arguments={"case_path": "ieee14"},
            oracle_tool="run_powerflow",
            oracle_arguments={"case_path": "ieee14"},
        )
        m2 = M2BenchmarkScenario(
            scenario_id="m2_meta",
            user_message="Run transient stability.",
            description="test",
            expected_statuses=("completed",),
            expected_tool="run_transient_stability",
            expected_tool_arguments={"case_path": "kundur_full"},
            oracle_tool="run_transient_stability",
            oracle_arguments={"case_path": "kundur_full"},
            oracle_result_template={
                "ok": True,
                "tool": "run_transient_stability",
                "case_path": "kundur_full",
            },
        )

        result = run_integrated_oracles(
            m1_scenarios=[m1],
            m2_scenarios=[m2],
            oracle_registry=FakeOracleRegistry(),
        )
        payload = result.to_dict(include_raw_results=False)

        self.assertTrue(result.ok)
        self.assertEqual(result.total, 2)
        self.assertEqual(payload["m1"]["total"], 1)
        self.assertEqual(payload["m2"]["total"], 1)
        self.assertEqual(payload["m1"]["results"][0]["scenario_id"], "m1_meta")
        self.assertEqual(payload["m2"]["results"][0]["scenario_id"], "m2_meta")

    def test_integrated_live_agent_run_combines_suite_results(self) -> None:
        m1 = M1BenchmarkScenario(
            scenario_id="m1_clarify",
            user_message="Need clarification.",
            description="test",
            expected_statuses=("completed",),
            expected_tool=None,
            output_contains=("answer",),
        )
        m2 = M2BenchmarkScenario(
            scenario_id="m2_clarify",
            user_message="Need clarification.",
            description="test",
            expected_statuses=("completed",),
            expected_tool=None,
            output_contains=("answer",),
        )
        agent = FakeAgent()

        result = run_integrated_live_agent(
            agent=agent,
            m1_scenarios=[m1],
            m2_scenarios=[m2],
            oracle_registry=FakeOracleRegistry(),
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.total, 2)
        self.assertEqual(len(agent.calls), 2)
        self.assertEqual(result.m1.passed, 1)
        self.assertEqual(result.m2.passed, 1)


if __name__ == "__main__":
    unittest.main()

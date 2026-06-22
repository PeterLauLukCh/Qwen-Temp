"""Tests for Step 15: deterministic experiment harness."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import (  # noqa: E402
    ExperimentExpectation,
    ExperimentRunner,
    ExperimentScenario,
    default_experiment_scenarios,
    evaluate_expectations,
    filter_scenarios,
)


class FakeExperimentRegistry:
    def __init__(self, *, fail_tool: Optional[str] = None) -> None:
        self.fail_tool = fail_tool
        self.calls: list[tuple[str, Dict[str, Any]]] = []

    def call_tool(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        args = dict(arguments or {})
        self.calls.append((name, args))
        if self.fail_tool == name:
            raise RuntimeError(f"{name} failed")
        if name == "run_powerflow":
            return {
                "ok": True,
                "tool": "run_powerflow",
                "case_info": {"case_name": args["case_path"]},
                "powerflow_summary": {
                    "converged": True,
                    "bus_summary": {"min_vm_pu": 0.99, "max_vm_pu": 1.02},
                    "branch_summary": {"max_line_loading_percent": 55.0},
                },
                "violation_report": {
                    "status": "pass",
                    "passed": True,
                    "total_violations": 0,
                    "hard_count": 0,
                    "borderline_count": 0,
                },
            }
        if name == "inspect_violations":
            return {
                "ok": True,
                "tool": "inspect_violations",
                "case_info": {"case_name": args["case_path"]},
                "violation_report": {
                    "status": "borderline",
                    "passed": True,
                    "total_violations": 1,
                    "hard_count": 0,
                    "borderline_count": 1,
                    "violations": [
                        {
                            "element_type": "bus",
                            "element_index": 1,
                            "violation_type": "high_voltage",
                            "severity": "borderline",
                            "observed_value": 1.052,
                            "limit_value": 1.05,
                            "margin_percent": 0.19,
                            "unit": "p.u.",
                        }
                    ],
                },
            }
        if name == "query_network_data":
            return {
                "ok": True,
                "tool": "query_network_data",
                "network_data": {
                    "case": {"case_name": args["case_path"]},
                    "tables": {
                        "buses": {"rows": [], "total_rows": 14, "truncated_rows": 9},
                    },
                },
                "solved_operating_point_claimed": False,
            }
        raise RuntimeError(f"unknown fake tool {name}")


class ExperimentHarnessTest(unittest.TestCase):
    def test_default_fast_scenarios_are_case_scoped(self) -> None:
        scenarios = default_experiment_scenarios(case_path="IEEE 14")

        self.assertEqual(len(scenarios), 3)
        self.assertEqual(scenarios[0].scenario_id, "ieee14_powerflow")
        self.assertEqual(scenarios[0].arguments["case_path"], "ieee14")
        self.assertIn("fast", scenarios[0].tags)

    def test_slow_scenarios_are_opt_in(self) -> None:
        fast = default_experiment_scenarios(case_path="ieee14")
        slow = default_experiment_scenarios(case_path="ieee14", include_slow=True)

        self.assertEqual(len(fast), 3)
        self.assertGreater(len(slow), len(fast))
        self.assertIn("ieee14_cia_small_load", {scenario.scenario_id for scenario in slow})

    def test_filter_scenarios_by_id_and_tag(self) -> None:
        scenarios = default_experiment_scenarios(case_path="ieee14", include_slow=True)

        by_id = filter_scenarios(scenarios, scenario_ids=["ieee14_powerflow"])
        by_tag = filter_scenarios(scenarios, tags=["topology"])

        self.assertEqual([scenario.scenario_id for scenario in by_id], ["ieee14_powerflow"])
        self.assertEqual([scenario.scenario_id for scenario in by_tag], ["ieee14_network_data"])

    def test_runner_executes_fast_suite_and_builds_reports(self) -> None:
        scenarios = default_experiment_scenarios(case_path="ieee14")
        registry = FakeExperimentRegistry()

        suite = ExperimentRunner(registry).run_suite(scenarios)

        self.assertTrue(suite.ok)
        self.assertEqual(suite.total, 3)
        self.assertEqual(suite.passed, 3)
        self.assertEqual([call[0] for call in registry.calls], [
            "run_powerflow",
            "inspect_violations",
            "query_network_data",
        ])
        self.assertEqual(suite.results[0].deterministic_report.source_tool, "run_powerflow")

    def test_suite_output_can_omit_raw_tool_results(self) -> None:
        scenario = default_experiment_scenarios(case_path="ieee14")[0]
        suite = ExperimentRunner(FakeExperimentRegistry()).run_suite([scenario])

        payload = suite.to_dict(include_raw_results=False)

        self.assertNotIn("tool_result", payload["results"][0])
        self.assertEqual(payload["results"][0]["tool_result_summary"]["ok"], True)

    def test_tool_exception_is_captured_as_failed_result(self) -> None:
        scenario = default_experiment_scenarios(case_path="ieee14")[0]
        suite = ExperimentRunner(FakeExperimentRegistry(fail_tool="run_powerflow")).run_suite(
            [scenario]
        )

        self.assertFalse(suite.ok)
        self.assertEqual(suite.failed, 1)
        result = suite.results[0]
        self.assertFalse(result.tool_result["ok"])
        self.assertEqual(result.tool_result["error_type"], "RuntimeError")
        self.assertFalse(result.passed)

    def test_expectation_operators(self) -> None:
        scenario = ExperimentScenario(
            scenario_id="ops",
            tool_name="run_powerflow",
            arguments={},
            expectations=(
                ExperimentExpectation("result.value", "gte", 2),
                ExperimentExpectation("result.value", "lte", 3),
                ExperimentExpectation("result.label", "in", ["a", "b"]),
                ExperimentExpectation("result.present", "exists"),
                ExperimentExpectation("result.missing", "not_exists"),
                ExperimentExpectation("result.present", "truthy"),
                ExperimentExpectation("result.empty", "falsy"),
            ),
        )
        report = ExperimentRunner(FakeExperimentRegistry()).run_scenario(
            default_experiment_scenarios(case_path="ieee14")[0]
        ).deterministic_report

        results = evaluate_expectations(
            scenario.expectations,
            result={"value": 2.5, "label": "a", "present": True, "empty": ""},
            report=report,
            scenario=scenario,
        )

        self.assertTrue(all(result.passed for result in results))


if __name__ == "__main__":
    unittest.main()

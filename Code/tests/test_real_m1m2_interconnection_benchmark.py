import json
import tempfile
import unittest
from pathlib import Path

from gridmind_mini.real_m1m2_interconnection_benchmark import (
    evaluate_real_m1m2_interconnection_result,
    filter_real_m1m2_interconnection_testcases,
    load_real_m1m2_interconnection_testcases,
)
from gridmind_mini.real_m1m2_interconnection_generator import (
    generate_real_m1m2_interconnection_testcases,
    write_real_m1m2_interconnection_testcases,
)


class _Record:
    def __init__(self, name, ok, result):
        self.name = name
        self.ok = ok
        self.result = result
        self.arguments = {}
        self.source = "test"
        self.error = None


class _AgentResult:
    def __init__(self, status, output_text, records):
        self.status = status
        self.output_text = output_text
        self.tool_records = records
        self.invoked_tools = [record.name for record in records if record.ok]
        self.llm_rounds = 1

    def to_dict(self, include_messages=True):
        return {
            "status": self.status,
            "output_text": self.output_text,
            "tool_records": [
                {
                    "name": record.name,
                    "ok": record.ok,
                    "result": record.result,
                }
                for record in self.tool_records
            ],
        }


class RealM1M2InterconnectionBenchmarkTest(unittest.TestCase):
    def test_positive_case_passes_expected_path_checks(self):
        scenario = next(
            item
            for item in generate_real_m1m2_interconnection_testcases(6, seed=3, profile="mixed")
            if item.oracle_label == "m1_m2_pass"
        )
        result = {
            "ok": True,
            "tool": "run_remote_psse_m1m2",
            "case_id": scenario.oracle_arguments["case_id"],
            "scenario_type": scenario.oracle_arguments["scenario_type"],
            "recommendation": "approve",
            "summary": {
                "m1_status": "pass",
                "m2_status": "pass",
                "m1_bus_count": 786
                if scenario.oracle_arguments["case_id"] == "pif6_2026_05_17"
                else 11,
            },
        }
        agent_result = _AgentResult(
            "completed",
            "Recommendation: approve. PSS/E M1 and M2 pass; EMT is not run.",
            [_Record("run_remote_psse_m1m2", True, result)],
        )
        checks = evaluate_real_m1m2_interconnection_result(
            scenario,
            agent_result=agent_result,
        )
        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_unsupported_case_passes_with_scope_listing(self):
        scenario = next(
            item
            for item in generate_real_m1m2_interconnection_testcases(6, seed=4, profile="mixed")
            if item.oracle_label != "m1_m2_pass"
        )
        agent_result = _AgentResult(
            "completed",
            "Unsupported in the current validated remote PSS/E action space.",
            [_Record("list_remote_psse_m1m2_cases", True, {"tool": "list_remote_psse_m1m2_cases", "case_count": 2})],
        )
        checks = evaluate_real_m1m2_interconnection_result(
            scenario,
            agent_result=agent_result,
        )
        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_forbidden_claim_fails(self):
        scenario = next(
            item
            for item in generate_real_m1m2_interconnection_testcases(6, seed=4, profile="mixed")
            if item.oracle_label != "m1_m2_pass"
        )
        agent_result = _AgentResult(
            "completed",
            "Unsupported in the current validated remote PSS/E action space, but approved by PSS/E.",
            [_Record("list_remote_psse_m1m2_cases", True, {"tool": "list_remote_psse_m1m2_cases", "case_count": 2})],
        )
        checks = evaluate_real_m1m2_interconnection_result(
            scenario,
            agent_result=agent_result,
        )
        failed = [check.name for check in checks if not check.passed]
        self.assertIn("forbidden_claim:approved by pss/e", failed)

    def test_negative_case_fails_successful_remote_baseline_proxy(self):
        scenario = next(
            item
            for item in generate_real_m1m2_interconnection_testcases(6, seed=4, profile="mixed")
            if item.oracle_label != "m1_m2_pass"
        )
        agent_result = _AgentResult(
            "completed",
            "Unsupported in the current validated remote PSS/E action space.",
            [
                _Record(
                    "list_remote_psse_m1m2_cases",
                    True,
                    {"tool": "list_remote_psse_m1m2_cases", "case_count": 2},
                ),
                _Record(
                    "run_remote_psse_m1m2",
                    True,
                    {
                        "tool": "run_remote_psse_m1m2",
                        "recommendation": "approve",
                    },
                ),
            ],
        )
        checks = evaluate_real_m1m2_interconnection_result(
            scenario,
            agent_result=agent_result,
        )
        failed = [check.name for check in checks if not check.passed]
        self.assertIn("forbidden_remote_baseline_proxy", failed)

    def test_load_and_filter_generated_json(self):
        scenarios = generate_real_m1m2_interconnection_testcases(5, seed=9, profile="hard")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "cases.json"
            write_real_m1m2_interconnection_testcases(scenarios, output)
            loaded = load_real_m1m2_interconnection_testcases(output)
        self.assertEqual([item.scenario_id for item in loaded], [item.scenario_id for item in scenarios])
        hard = filter_real_m1m2_interconnection_testcases(loaded, difficulties=["hard"], limit=2)
        self.assertEqual(len(hard), 2)
        self.assertTrue(all(item.difficulty == "hard" for item in hard))

    def test_load_jsonl(self):
        scenarios = generate_real_m1m2_interconnection_testcases(3, seed=10, profile="easy")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "cases.jsonl"
            with output.open("w", encoding="utf-8") as handle:
                for scenario in scenarios:
                    handle.write(json.dumps(scenario.to_dict()) + "\n")
            loaded = load_real_m1m2_interconnection_testcases(output)
        self.assertEqual(len(loaded), 3)


if __name__ == "__main__":
    unittest.main()

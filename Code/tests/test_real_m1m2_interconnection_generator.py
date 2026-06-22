"""Tests for the real PSS/E M1+M2 interconnection testcase generator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gridmind_mini import (
    REAL_M1M2_SCHEMA_VERSION,
    RealM1M2InterconnectionTestCase,
    generate_real_m1m2_interconnection_testcases,
    real_m1m2_interconnection_testcases_from_payload,
    write_real_m1m2_interconnection_testcases,
)


class RealM1M2InterconnectionGeneratorTest(unittest.TestCase):
    def test_mixed_generation_is_deterministic_and_unique(self) -> None:
        first = generate_real_m1m2_interconnection_testcases(12, seed=123, profile="mixed")
        second = generate_real_m1m2_interconnection_testcases(12, seed=123, profile="mixed")

        self.assertEqual([item.to_dict() for item in first], [item.to_dict() for item in second])
        self.assertEqual(len({item.scenario_id for item in first}), 12)
        labels = {item.oracle_label for item in first}
        self.assertIn("m1_m2_pass", labels)
        self.assertIn("unsupported_new_interconnection", labels)
        self.assertIn("unsupported_disturbance", labels)
        self.assertIn("unsupported_controller_change", labels)
        self.assertIn("wrong_tool_trap", labels)

    def test_positive_cases_map_to_remote_m1m2_job_only(self) -> None:
        scenarios = generate_real_m1m2_interconnection_testcases(10, seed=5, profile="mixed")
        positives = [item for item in scenarios if item.oracle_label == "m1_m2_pass"]

        self.assertTrue(positives)
        for scenario in positives:
            self.assertEqual(scenario.expected_tool, "run_remote_psse_m1m2")
            self.assertIn(scenario.oracle_arguments["case_id"], {"pif6_2026_05_17", "test_cases_v36"})
            self.assertEqual(scenario.oracle_arguments["scenario_type"], "no_disturbance_5s")
            expected_paths = {item.path: item.expected for item in scenario.expected_paths}
            self.assertEqual(expected_paths["recommendation"], "approve")
            self.assertEqual(expected_paths["summary.m1_status"], "pass")
            self.assertEqual(expected_paths["summary.m2_status"], "pass")
            self.assertTrue(scenario.context["validated_remote_job"])

    def test_unsupported_cases_use_scope_listing_and_never_claim_physics_pass(self) -> None:
        scenarios = generate_real_m1m2_interconnection_testcases(12, seed=8, profile="hard")
        negatives = [item for item in scenarios if item.oracle_label != "m1_m2_pass"]

        self.assertTrue(negatives)
        for scenario in negatives:
            self.assertEqual(scenario.expected_tool, "list_remote_psse_m1m2_cases")
            self.assertEqual(scenario.oracle_arguments, {"check_health": False})
            self.assertFalse(scenario.context.get("validated_remote_job"))
            text = " ".join(scenario.forbidden_claims).lower()
            if scenario.oracle_label != "wrong_tool_trap":
                self.assertIn("passes m1+m2", text)
            self.assertNotIn("oracle_label': 'm1_m2_pass", str(scenario.to_dict()))

    def test_easy_and_hard_profiles_set_difficulty(self) -> None:
        easy = generate_real_m1m2_interconnection_testcases(4, seed=1, profile="easy")
        hard = generate_real_m1m2_interconnection_testcases(6, seed=1, profile="hard")

        self.assertTrue(all(item.difficulty in {"easy", "medium"} for item in easy))
        self.assertTrue(any(item.difficulty == "hard" for item in hard))

    def test_payload_round_trip_and_write_json_jsonl(self) -> None:
        scenarios = generate_real_m1m2_interconnection_testcases(5, seed=44, profile="mixed")
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "cases.json"
            jsonl_path = Path(tmpdir) / "cases.jsonl"

            json_summary = write_real_m1m2_interconnection_testcases(scenarios, json_path)
            jsonl_summary = write_real_m1m2_interconnection_testcases(
                scenarios,
                jsonl_path,
                jsonl=True,
            )

            self.assertEqual(json_summary["scenario_count"], 5)
            self.assertEqual(jsonl_summary["format"], "jsonl")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            round_tripped = real_m1m2_interconnection_testcases_from_payload(payload)
            self.assertEqual([item.scenario_id for item in round_tripped], [item.scenario_id for item in scenarios])
            lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 5)
            self.assertEqual(json.loads(lines[0])["schema_version"], REAL_M1M2_SCHEMA_VERSION)

    def test_from_mapping_rejects_duplicate_ids(self) -> None:
        scenario = generate_real_m1m2_interconnection_testcases(1, seed=9)[0]
        payload = {"scenarios": [scenario.to_dict(), scenario.to_dict()]}

        with self.assertRaises(ValueError):
            real_m1m2_interconnection_testcases_from_payload(payload)

    def test_from_mapping_reconstructs_dataclass(self) -> None:
        scenario = generate_real_m1m2_interconnection_testcases(1, seed=10)[0]

        reconstructed = RealM1M2InterconnectionTestCase.from_mapping(scenario.to_dict())

        self.assertEqual(reconstructed.to_dict(), scenario.to_dict())


if __name__ == "__main__":
    unittest.main()

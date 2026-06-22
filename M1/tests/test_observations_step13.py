"""Tests for Step 13: deterministic tool-observation summaries."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import build_tool_observation, tool_observation_payload  # noqa: E402


class ToolObservationTest(unittest.TestCase):
    def test_powerflow_observation_extracts_key_solver_facts(self) -> None:
        observation = build_tool_observation(
            {
                "ok": True,
                "tool": "run_powerflow",
                "case_info": {"case_name": "ieee14"},
                "powerflow_summary": {
                    "converged": True,
                    "bus_summary": {"min_vm_pu": 0.98, "max_vm_pu": 1.04},
                    "branch_summary": {
                        "max_line_loading_percent": 83.5,
                        "max_trafo_loading_percent": None,
                    },
                },
                "violation_report": {
                    "status": "pass",
                    "passed": True,
                    "total_violations": 0,
                    "hard_count": 0,
                    "borderline_count": 0,
                },
            }
        )

        self.assertEqual(observation["tool"], "run_powerflow")
        self.assertEqual(observation["case"], "ieee14")
        self.assertTrue(observation["converged"])
        self.assertEqual(observation["voltage"]["min_vm_pu"], 0.98)
        self.assertEqual(observation["loading"]["max_line_loading_percent"], 83.5)
        self.assertEqual(observation["violation_report"]["status"], "pass")

    def test_cia_observation_extracts_recommendation_and_stage_statuses(self) -> None:
        observation = build_tool_observation(
            {
                "ok": True,
                "tool": "run_cia",
                "case_path": "ieee118",
                "connection": {
                    "bus": 10,
                    "p_mw": 25.0,
                    "connection_type": "solar",
                    "is_ibr": True,
                },
                "recommendation": "approve",
                "complete": True,
                "reason_codes": ["steady_state_no_project_caused_violations"],
                "summary": {
                    "baseline_status": "pass",
                    "post_status": "pass",
                    "project_hard_violations": 0,
                    "project_borderline_violations": 0,
                },
                "stage_reports": [
                    {
                        "stage": "f1_steady_state",
                        "status": "pass",
                        "passed": True,
                        "reason_codes": ["steady_state_no_project_caused_violations"],
                    }
                ],
            }
        )

        self.assertEqual(observation["tool"], "run_cia")
        self.assertEqual(observation["recommendation"], "approve")
        self.assertEqual(observation["connection"]["p_mw"], 25.0)
        self.assertEqual(observation["summary"]["project_hard_violations"], 0)
        self.assertEqual(observation["stage_statuses"][0]["stage"], "f1_steady_state")

    def test_capacity_observation_extracts_boundary(self) -> None:
        observation = build_tool_observation(
            {
                "ok": True,
                "tool": "find_max_capacity",
                "case_path": "ieee14",
                "request": {"bus": 10, "connection_type": "load", "is_ibr": False},
                "status": "bounded",
                "max_approved_mw": 42.0,
                "lower_bound_mw": 42.0,
                "upper_bound_mw": 43.0,
                "tolerance_mw": 1.0,
                "iterations": 7,
            }
        )

        self.assertEqual(observation["tool"], "find_max_capacity")
        self.assertEqual(observation["max_approved_mw"], 42.0)
        self.assertEqual(observation["upper_bound_mw"], 43.0)

    def test_error_observation_preserves_repair_metadata(self) -> None:
        observation = build_tool_observation(
            {
                "ok": False,
                "tool": "find_max_capacity",
                "error_type": "tool_policy_violation",
                "message": "Use run_cia instead.",
                "recommended_tool": "run_cia",
                "reason_codes": ["specific_sized_cia_request"],
            }
        )

        self.assertEqual(observation["status"], "error")
        self.assertEqual(observation["error_type"], "tool_policy_violation")
        self.assertEqual(observation["recommended_tool"], "run_cia")
        self.assertEqual(observation["reason_codes"], ["specific_sized_cia_request"])

    def test_tool_observation_payload_can_omit_raw_result(self) -> None:
        result = {"ok": True, "tool": "find_max_capacity", "max_approved_mw": 10.0}

        payload = tool_observation_payload(result, include_raw_result=False)

        self.assertEqual(payload["tool"], "find_max_capacity")
        self.assertIn("observation", payload)
        self.assertNotIn("raw_result", payload)


if __name__ == "__main__":
    unittest.main()

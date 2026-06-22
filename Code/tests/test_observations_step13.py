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

    def test_transient_observation_extracts_status_and_metrics(self) -> None:
        observation = build_tool_observation(
            {
                "ok": True,
                "tool": "run_transient_stability",
                "backend": "andes",
                "case_path": "kundur_full",
                "disturbance": {
                    "type": "bus_fault",
                    "bus": 5,
                    "fault_start_s": 1.0,
                    "clearing_time_s": 1.1,
                    "clearing_duration_s": 0.1,
                },
                "simulation": {
                    "simulation_time_s": 5.0,
                    "pflow_converged": True,
                    "tds_converged": True,
                    "sample_count": 3,
                },
                "stability": {"status": "pass"},
                "metrics": {
                    "tds_converged": True,
                    "max_angle_spread_rad": 0.2,
                    "final_angle_spread_rad": 0.1,
                    "max_speed_deviation_pu": 0.01,
                    "min_voltage_pu": 0.92,
                    "reason_codes": ["transient_metrics_within_thresholds"],
                },
                "trajectories": {"time_s": [0.0, 1.0, 5.0]},
            }
        )

        self.assertEqual(observation["tool"], "run_transient_stability")
        self.assertEqual(observation["case"], "kundur_full")
        self.assertEqual(observation["stability"]["status"], "pass")
        self.assertEqual(observation["disturbance"]["bus"], 5)
        self.assertEqual(observation["metrics"]["min_voltage_pu"], 0.92)

    def test_emt_observation_extracts_scr_metrics(self) -> None:
        observation = build_tool_observation(
            {
                "ok": True,
                "tool": "run_emt_screening",
                "backend": "pandapower",
                "screening_type": "scr_proxy",
                "case_path": "ieee14",
                "connection": {
                    "bus": 10,
                    "p_mw": 5.0,
                    "connection_type": "solar",
                    "is_ibr": True,
                },
                "poi_bus": {"requested_bus": 10, "bus_index": 9},
                "connection_application": {
                    "requested": True,
                    "applied": True,
                    "element_table": "sgen",
                    "element_index": 3,
                },
                "emt": {
                    "status": "pass",
                    "passed": True,
                    "criterion": "scr_greater_equal_threshold",
                },
                "metrics": {
                    "scr": 8.2,
                    "short_circuit_mva": 41.0,
                    "project_mva": 5.0,
                    "threshold": 3.0,
                    "borderline_margin": 0.25,
                    "reason_codes": ["emt_scr_above_threshold"],
                },
                "limitations": ["emt_v1_scr_screen_only_no_time_domain_waveform"],
            }
        )

        self.assertEqual(observation["tool"], "run_emt_screening")
        self.assertEqual(observation["status"], "pass")
        self.assertEqual(observation["metrics"]["scr"], 8.2)
        self.assertEqual(observation["poi_bus"]["bus_index"], 9)

    def test_integrated_assessment_observation_extracts_stage_statuses(self) -> None:
        observation = build_tool_observation(
            {
                "ok": True,
                "tool": "run_integrated_assessment",
                "case_path": "ieee14",
                "connection": {
                    "bus": 10,
                    "p_mw": 5.0,
                    "connection_type": "solar",
                    "is_ibr": True,
                },
                "recommendation": "approve",
                "complete": True,
                "reason_codes": ["integrated_assessment"],
                "summary": {
                    "m1_recommendation": "approve",
                    "m2_status": "pass",
                    "m2_stability_status": "pass",
                    "transient_required_for_approval": True,
                },
                "stage_reports": [
                    {
                        "stage": "m1_steady_state_cia",
                        "status": "pass",
                        "passed": True,
                        "required_for_approval": True,
                        "recommendation": "approve",
                    },
                    {
                        "stage": "m2_transient_stability",
                        "status": "pass",
                        "passed": True,
                        "required_for_approval": True,
                        "case_path": "ieee14_dynamic",
                    },
                ],
                "limitations": ["m2_connection_model_is_static_pq_not_full_dynamic_resource"],
                "linkage": {
                    "m1_models_new_connection": True,
                    "m2_models_new_connection": False,
                    "m2_linkage": "standalone_dynamic_case_screen",
                },
            }
        )

        self.assertEqual(observation["tool"], "run_integrated_assessment")
        self.assertEqual(observation["recommendation"], "approve")
        self.assertEqual(observation["summary"]["m2_status"], "pass")
        self.assertEqual(observation["stage_statuses"][1]["stage"], "m2_transient_stability")
        self.assertFalse(observation["linkage"]["m2_models_new_connection"])

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

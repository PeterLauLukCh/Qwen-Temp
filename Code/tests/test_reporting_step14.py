"""Tests for Step 14: deterministic final-report composition."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import build_deterministic_report, report_text_or_original  # noqa: E402


class DeterministicReportTest(unittest.TestCase):
    def test_powerflow_report_uses_solver_observation_facts(self) -> None:
        report = build_deterministic_report(
            [
                {
                    "ok": True,
                    "tool": "run_powerflow",
                    "case_info": {"case_name": "ieee14"},
                    "powerflow_summary": {
                        "converged": True,
                        "bus_summary": {"min_vm_pu": 0.98, "max_vm_pu": 1.04},
                        "branch_summary": {
                            "max_line_loading_percent": 83.5,
                            "max_trafo_loading_percent": 42.0,
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
            ]
        )

        self.assertTrue(report.available)
        self.assertEqual(report.source_tool, "run_powerflow")
        self.assertIn("Power-flow report for ieee14", report.summary_text)
        self.assertIn("total=0", report.summary_text)
        self.assertIn("Voltage range: 0.98-1.04 p.u.", report.summary_text)
        self.assertEqual(report.facts["violation_report"]["status"], "pass")

    def test_cia_report_preserves_recommendation_and_incomplete_stage(self) -> None:
        report = build_deterministic_report(
            [
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
                    "recommendation": "borderline",
                    "complete": False,
                    "reason_codes": ["f3_transient_stability_not_implemented"],
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
                        },
                        {
                            "stage": "f3_transient_stability",
                            "status": "not_implemented",
                            "passed": False,
                            "reason_codes": ["f3_transient_stability_not_implemented"],
                        },
                    ],
                }
            ]
        )

        self.assertEqual(report.source_tool, "run_cia")
        self.assertIn("25 MW solar at bus 10", report.summary_text)
        self.assertIn("recommendation=borderline", report.summary_text)
        self.assertIn("f3_transient_stability", report.limitations)
        self.assertIn("incomplete_assessment", report.limitations)

    def test_capacity_report_extracts_boundary_and_rejection(self) -> None:
        report = build_deterministic_report(
            [
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
                    "rejection_explanation": {
                        "limiting_stage": "f1_steady_state",
                        "status": "fail",
                    },
                }
            ]
        )

        self.assertEqual(report.source_tool, "find_max_capacity")
        self.assertIn("Maximum approved capacity: 42 MW", report.summary_text)
        self.assertIn("lower_bound=42 MW", report.summary_text)
        self.assertIn("stage=f1_steady_state", report.summary_text)

    def test_transient_report_summarizes_status_and_metrics(self) -> None:
        report = build_deterministic_report(
            [
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
                    },
                    "simulation": {
                        "simulation_time_s": 5.0,
                        "pflow_converged": True,
                        "tds_converged": True,
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
                }
            ]
        )

        self.assertEqual(report.source_tool, "run_transient_stability")
        self.assertIn("Transient-stability report for kundur_full", report.summary_text)
        self.assertIn("status=pass", report.summary_text)
        self.assertIn("max_angle_spread=0.2 rad", report.summary_text)
        self.assertEqual(report.facts["metrics"]["min_voltage_pu"], 0.92)

    def test_emt_report_summarizes_scr_status_and_metrics(self) -> None:
        report = build_deterministic_report(
            [
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
                    "emt": {"status": "pass", "passed": True},
                    "metrics": {
                        "scr": 8.2,
                        "short_circuit_mva": 41.0,
                        "project_mva": 5.0,
                        "threshold": 3.0,
                        "reason_codes": ["emt_scr_above_threshold"],
                    },
                    "limitations": ["emt_v1_scr_screen_only_no_time_domain_waveform"],
                }
            ]
        )

        self.assertEqual(report.source_tool, "run_emt_screening")
        self.assertIn("EMT/SCR screening report for 5 MW solar", report.summary_text)
        self.assertIn("SCR=8.2", report.summary_text)
        self.assertIn("threshold=3", report.summary_text)

    def test_integrated_assessment_report_summarizes_m1_m2_statuses(self) -> None:
        report = build_deterministic_report(
            [
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
                    "recommendation": "borderline",
                    "complete": False,
                    "reason_codes": ["required_m2_unavailable"],
                    "summary": {
                        "m1_recommendation": "approve",
                        "m2_status": "unavailable",
                        "m2_stability_status": "error",
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
                            "status": "unavailable",
                            "passed": False,
                            "required_for_approval": True,
                        },
                    ],
                    "limitations": [
                        "m2_connection_model_is_static_pq_not_full_dynamic_resource",
                        "unavailable",
                    ],
                    "linkage": {
                        "m1_models_new_connection": True,
                        "m2_models_new_connection": False,
                        "m2_linkage": "standalone_dynamic_case_screen",
                    },
                }
            ]
        )

        self.assertEqual(report.source_tool, "run_integrated_assessment")
        self.assertIn("Integrated M1+M2 assessment", report.summary_text)
        self.assertIn("recommendation=borderline", report.summary_text)
        self.assertIn("M1=approve", report.summary_text)
        self.assertIn("M2=unavailable", report.summary_text)
        self.assertIn("incomplete_assessment", report.limitations)

    def test_latest_successful_reportable_result_is_selected(self) -> None:
        report = build_deterministic_report(
            [
                {
                    "ok": False,
                    "tool": "run_powerflow",
                    "error_type": "ToolRegistryError",
                    "error": "missing case",
                },
                {
                    "ok": True,
                    "tool": "find_max_capacity",
                    "case_path": "ieee14",
                    "request": {"bus": 5, "connection_type": "load", "is_ibr": False},
                    "status": "max_bound_approved",
                    "max_approved_mw": 20.0,
                },
            ]
        )

        self.assertTrue(report.available)
        self.assertEqual(report.source_tool, "find_max_capacity")
        self.assertIn("20 MW", report.summary_text)

    def test_error_report_is_available_but_marked_limited(self) -> None:
        report = build_deterministic_report(
            [
                {
                    "ok": False,
                    "tool": "find_max_capacity",
                    "error_type": "tool_policy_violation",
                    "message": "Use run_cia instead.",
                    "recommended_tool": "run_cia",
                }
            ]
        )

        self.assertTrue(report.available)
        self.assertEqual(report.source_tool, "find_max_capacity")
        self.assertIn("did not complete", report.summary_text)
        self.assertIn("tool_failed", report.limitations)
        self.assertIn("recommended_tool=run_cia", report.warnings)

    def test_metadata_only_tools_do_not_create_source_of_truth_report(self) -> None:
        report = build_deterministic_report(
            [
                {
                    "ok": True,
                    "tool": "list_cases",
                    "backend": "pandapower",
                    "cases": ["ieee14", "ieee118"],
                }
            ]
        )

        self.assertFalse(report.available)
        self.assertIsNone(report.source_tool)
        self.assertIn("No solver-backed tool result", report.summary_text)
        self.assertIn("no_tool_result", report.limitations)

    def test_empty_final_text_can_fall_back_to_report_text(self) -> None:
        report = build_deterministic_report(
            [
                {
                    "ok": True,
                    "tool": "find_max_capacity",
                    "case_path": "ieee14",
                    "request": {"bus": 5, "connection_type": "load", "is_ibr": False},
                    "status": "max_bound_approved",
                    "max_approved_mw": 20.0,
                }
            ]
        )

        self.assertEqual(report_text_or_original("", report), report.summary_text)
        self.assertEqual(report_text_or_original("LLM answer", report), "LLM answer")


if __name__ == "__main__":
    unittest.main()

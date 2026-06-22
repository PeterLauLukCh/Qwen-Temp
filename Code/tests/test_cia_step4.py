"""Tests for the Step 4 steady-state CIA pipeline."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import (
    CaseInfo,
    ConnectionRequest,
    GridSolver,
    InspectionReport,
    LimitProfile,
    PowerFlowError,
    SteadyStateCIARunner,
    Violation,
)
from gridmind_mini.cia import compare_violation_reports


class ScenarioSolver(GridSolver):
    def __init__(self) -> None:
        self.loaded_case = ""
        self.connection: Optional[ConnectionRequest] = None
        self.converged = False
        self.outage: Optional[Dict[str, Any]] = None
        self.short_circuit_mva_by_bus = {1: 50.0}

    def available_cases(self) -> List[str]:
        return ["scenario"]

    def load_case(self, case_name: str) -> CaseInfo:
        self.loaded_case = case_name
        return self.case_info()

    def run_powerflow(self) -> Dict[str, Any]:
        self.converged = True
        return {
            "backend": "scenario",
            "case": self.case_info().to_dict(),
            "converged": True,
            "bus_summary": {
                "min_vm_pu": min(row["vm_pu"] for row in self.bus_results()),
                "max_vm_pu": max(row["vm_pu"] for row in self.bus_results()),
            },
            "branch_summary": {"max_line_loading_percent": 50.0},
        }

    def bus_results(self) -> List[Dict[str, Any]]:
        vm_pu = 1.0
        if self.loaded_case == "baseline_borderline":
            vm_pu = 0.945
        if self.loaded_case == "baseline_inside_borderline":
            vm_pu = 0.955
        if self.connection is not None:
            if self.connection.p_mw == 50.0:
                vm_pu = 0.945
            elif self.connection.p_mw == 100.0:
                vm_pu = 0.93
            elif self.connection.p_mw == 150.0:
                vm_pu = 0.91
        return [{"bus_index": 1, "name": "Bus 1", "vm_pu": vm_pu, "va_degree": 0.0}]

    def branch_results(self) -> Dict[str, List[Dict[str, Any]]]:
        loading_percent = 50.0
        if self.outage is not None and self.loaded_case == "contingency_project_failure":
            loading_percent = 120.0 if self.connection is not None else 100.0
        return {
            "lines": [
                {
                    "line_index": 1,
                    "name": "Line 1",
                    "from_bus": 1,
                    "to_bus": 1,
                    "loading_percent": loading_percent,
                    "in_service": True,
                }
            ],
            "transformers": [],
        }

    def available_contingencies(self) -> List[Dict[str, Any]]:
        return [
            {
                "element_type": "line",
                "element_index": 1,
                "element_name": "Line 1",
                "from_bus": 1,
                "to_bus": 1,
            }
        ]

    def apply_contingency(self, element_type: str, element_index: int) -> Dict[str, Any]:
        self.outage = {
            "element_type": element_type,
            "element_index": element_index,
            "element_name": "Line 1",
            "from_bus": 1,
            "to_bus": 1,
        }
        self.converged = False
        return dict(self.outage)

    def add_connection(
        self,
        bus: int,
        p_mw: float,
        connection_type: str,
        is_ibr: bool,
        q_mvar: float = 0.0,
        name: Optional[str] = None,
        vm_pu: float = 1.0,
    ) -> Dict[str, Any]:
        self.connection = ConnectionRequest(
            bus=bus,
            p_mw=p_mw,
            connection_type=connection_type,
            is_ibr=is_ibr,
            q_mvar=q_mvar,
            name=name,
            vm_pu=vm_pu,
        )
        self.converged = False
        return {
            "element_table": "load",
            "element_index": 0,
            "name": name or "scenario_connection",
            "connection_type": connection_type,
            "is_ibr": is_ibr,
            "bus": bus,
            "resolved_bus": {"bus_index": bus, "matched_on": "scenario"},
            "p_mw": p_mw,
            "q_mvar": q_mvar,
        }

    def network_data(self, max_rows: int = 50) -> Dict[str, Any]:
        return {"case": self.case_info().to_dict(), "tables": {}}

    def case_info(self) -> CaseInfo:
        return CaseInfo(
            case_name=self.loaded_case or "scenario",
            buses=1,
            lines=1,
            transformers=0,
            loads=0,
            generators=0,
            static_generators=0,
            external_grids=1,
        )


class NonConvergingPostSolver(ScenarioSolver):
    def run_powerflow(self) -> Dict[str, Any]:
        if self.connection is not None:
            raise PowerFlowError("post case diverged")
        return super().run_powerflow()


def runner(factory=ScenarioSolver) -> SteadyStateCIARunner:
    return SteadyStateCIARunner(solver_factory=factory)


def connection(p_mw: float, is_ibr: bool = False) -> ConnectionRequest:
    return ConnectionRequest(bus=1, p_mw=p_mw, connection_type="load", is_ibr=is_ibr)


class SteadyStateCIATest(unittest.TestCase):
    def test_connection_request_rejects_non_boolean_ibr_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "is_ibr"):
            ConnectionRequest.from_mapping(
                {
                    "bus": 1,
                    "p_mw": 10.0,
                    "connection_type": "load",
                    "is_ibr": "false",
                }
            )

    def test_connection_request_rejects_boolean_bus(self) -> None:
        with self.assertRaisesRegex(ValueError, "bus"):
            ConnectionRequest.from_mapping(
                {
                    "bus": True,
                    "p_mw": 10.0,
                    "connection_type": "load",
                    "is_ibr": False,
                }
            )

    def test_connection_request_rejects_string_power(self) -> None:
        with self.assertRaisesRegex(ValueError, "p_mw"):
            ConnectionRequest.from_mapping(
                {
                    "bus": 1,
                    "p_mw": "10",
                    "connection_type": "load",
                    "is_ibr": False,
                }
            )

    def test_connection_request_rejects_non_string_connection_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "connection_type"):
            ConnectionRequest.from_mapping(
                {
                    "bus": 1,
                    "p_mw": 10.0,
                    "connection_type": 1,
                    "is_ibr": False,
                }
            )

    def test_connection_request_rejects_non_finite_power(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            ConnectionRequest.from_mapping(
                {
                    "bus": 1,
                    "p_mw": math.nan,
                    "connection_type": "load",
                    "is_ibr": False,
                }
            )

    def test_negative_material_worsening_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            SteadyStateCIARunner(
                solver_factory=ScenarioSolver,
                material_worsening_threshold_percent=-0.1,
            )

    def test_small_connection_is_approved_when_no_project_violations(self) -> None:
        result = runner().run("scenario", connection(10.0))

        self.assertTrue(result["ok"])
        self.assertEqual(result["recommendation"], "approve")
        self.assertTrue(result["complete"])
        self.assertEqual(result["summary"]["project_hard_violations"], 0)

    def test_new_borderline_violation_produces_borderline_recommendation(self) -> None:
        result = runner().run("scenario", connection(50.0))

        self.assertEqual(result["recommendation"], "borderline")
        self.assertEqual(result["summary"]["project_borderline_violations"], 1)
        self.assertIn("steady_state_project_borderline_violations", result["reason_codes"])

    def test_new_hard_violation_rejects(self) -> None:
        result = runner().run("scenario", connection(100.0))

        self.assertEqual(result["recommendation"], "reject")
        self.assertEqual(result["summary"]["project_hard_violations"], 1)
        self.assertIn("steady_state_project_hard_violations", result["reason_codes"])

    def test_unchanged_baseline_violation_is_not_project_caused(self) -> None:
        result = runner().run("baseline_borderline", connection(10.0))

        self.assertEqual(result["recommendation"], "approve")
        self.assertEqual(result["summary"]["project_borderline_violations"], 0)
        self.assertIn("baseline_pre_existing_borderline_violations", result["reason_codes"])

    def test_crossing_limit_from_inside_borderline_is_project_caused(self) -> None:
        result = runner().run("baseline_inside_borderline", connection(50.0))

        self.assertEqual(result["recommendation"], "borderline")
        comparison = result["stage_reports"][0]["project_violation_comparison"]
        self.assertEqual(comparison["project_borderline_count"], 1)
        self.assertEqual(
            comparison["project_caused_violations"]["items"][0]["change_type"],
            "limit_crossed_within_borderline",
        )

    def test_material_worsening_of_preexisting_violation_rejects(self) -> None:
        result = runner().run("baseline_borderline", connection(150.0))

        self.assertEqual(result["recommendation"], "reject")
        comparison = result["stage_reports"][0]["project_violation_comparison"]
        self.assertEqual(comparison["project_hard_count"], 1)
        self.assertEqual(comparison["project_caused_violations"]["total_items"], 1)
        self.assertEqual(
            comparison["project_caused_violations"]["items"][0]["change_type"],
            "severity_worsened",
        )

    def test_project_caused_violations_are_ordered_by_severity(self) -> None:
        baseline_report = InspectionReport(LimitProfile.normal(), [])
        post_report = InspectionReport(
            LimitProfile.normal(),
            [
                Violation(
                    element_type="bus",
                    element_index=1,
                    element_name="Bus 1",
                    violation_type="high_voltage",
                    severity="borderline",
                    observed_value=1.052,
                    limit_value=1.05,
                    margin=0.002,
                    margin_percent=0.190,
                    unit="p.u.",
                    limit_relation="<=",
                ),
                Violation(
                    element_type="bus",
                    element_index=2,
                    element_name="Bus 2",
                    violation_type="low_voltage",
                    severity="hard",
                    observed_value=0.93,
                    limit_value=0.95,
                    margin=0.02,
                    margin_percent=2.105,
                    unit="p.u.",
                    limit_relation=">=",
                ),
            ],
        )

        comparison = compare_violation_reports(baseline_report, post_report)

        self.assertEqual(comparison["project_caused_violations"][0]["severity"], "hard")

    def test_requested_contingency_runs_and_can_pass(self) -> None:
        result = runner().run("scenario", connection(10.0), enable_contingency=True)

        self.assertEqual(result["recommendation"], "approve")
        self.assertTrue(result["complete"])
        self.assertEqual(result["stage_reports"][1]["status"], "pass")
        self.assertIn("n1_no_project_caused_failures", result["stage_reports"][1]["reason_codes"])

    def test_project_caused_contingency_failure_rejects(self) -> None:
        result = runner().run(
            "contingency_project_failure",
            connection(10.0),
            enable_contingency=True,
        )

        self.assertEqual(result["recommendation"], "reject")
        self.assertTrue(result["complete"])
        self.assertIn("n1_project_new_failures", result["reason_codes"])
        self.assertEqual(result["stage_reports"][1]["status"], "fail")

    def test_ibr_transient_stays_placeholder_but_emt_runs(self) -> None:
        result = runner().run(
            "scenario",
            connection(10.0, is_ibr=True),
            enable_transient=True,
            enable_emt=True,
        )

        self.assertEqual(result["recommendation"], "borderline")
        self.assertFalse(result["complete"])
        self.assertIn("f3_transient_stability_not_implemented", result["reason_codes"])
        self.assertEqual(result["stage_reports"][3]["stage"], "f4_emt_screening")
        self.assertEqual(result["stage_reports"][3]["status"], "pass")
        self.assertIn("emt_scr_above_threshold", result["stage_reports"][3]["reason_codes"])
        self.assertAlmostEqual(result["stage_reports"][3]["result"]["metrics"]["scr"], 5.0)

    def test_ibr_emt_screen_can_fail_on_low_scr(self) -> None:
        result = runner().run(
            "scenario",
            connection(20.0, is_ibr=True),
            enable_emt=True,
        )

        self.assertEqual(result["recommendation"], "reject")
        self.assertTrue(result["complete"])
        self.assertEqual(result["stage_reports"][3]["stage"], "f4_emt_screening")
        self.assertEqual(result["stage_reports"][3]["status"], "fail")
        self.assertIn("emt_scr_below_threshold", result["reason_codes"])

    def test_post_nonconvergence_rejects(self) -> None:
        result = runner(NonConvergingPostSolver).run("scenario", connection(10.0))

        self.assertTrue(result["ok"])
        self.assertEqual(result["recommendation"], "reject")
        self.assertIn("post_powerflow_non_convergence", result["reason_codes"])
        comparison = result["stage_reports"][0]["project_violation_comparison"]
        self.assertEqual(comparison["project_caused_violations"]["total_items"], 0)

    def test_inconsistent_ibr_flag_is_reported_as_warning(self) -> None:
        result = runner().run(
            "scenario",
            ConnectionRequest(bus=1, p_mw=10.0, connection_type="solar", is_ibr=False),
        )

        self.assertTrue(result["connection_classification"]["effective_is_ibr"])
        self.assertEqual(
            result["connection_warnings"],
            ["ibr_flag_false_for_inverter_based_connection_type"],
        )

    def test_inverter_connection_type_drives_requested_transient_stage(self) -> None:
        result = runner().run(
            "scenario",
            ConnectionRequest(bus=1, p_mw=10.0, connection_type="solar", is_ibr=False),
            enable_transient=True,
        )

        self.assertEqual(result["recommendation"], "borderline")
        self.assertFalse(result["complete"])
        self.assertEqual(result["stage_reports"][2]["stage"], "f3_transient_stability")
        self.assertEqual(result["stage_reports"][2]["status"], "not_implemented")
        self.assertIn("f3_transient_stability_not_implemented", result["reason_codes"])


if __name__ == "__main__":
    unittest.main()

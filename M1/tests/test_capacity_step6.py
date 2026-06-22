"""Tests for the Step 6 capacity-search tool."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import CapacitySearchRunner, CaseInfo, ConnectionRequest, GridSolver


class CapacityScenarioSolver(GridSolver):
    def __init__(self) -> None:
        self.loaded_case = ""
        self.connection: Optional[ConnectionRequest] = None
        self.converged = False

    def available_cases(self) -> List[str]:
        return ["capacity"]

    def load_case(self, case_name: str) -> CaseInfo:
        self.loaded_case = case_name
        return self.case_info()

    def run_powerflow(self) -> Dict[str, Any]:
        self.converged = True
        voltages = [row["vm_pu"] for row in self.bus_results()]
        return {
            "backend": "capacity",
            "case": self.case_info().to_dict(),
            "converged": True,
            "bus_summary": {"min_vm_pu": min(voltages), "max_vm_pu": max(voltages)},
            "branch_summary": {"max_line_loading_percent": 50.0},
        }

    def bus_results(self) -> List[Dict[str, Any]]:
        vm_pu = 1.0
        if self.connection is not None and self.connection.p_mw > 40.0:
            vm_pu = 0.93
        return [{"bus_index": 1, "name": "Bus 1", "vm_pu": vm_pu, "va_degree": 0.0}]

    def branch_results(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "lines": [
                {
                    "line_index": 1,
                    "name": "Line 1",
                    "from_bus": 1,
                    "to_bus": 1,
                    "loading_percent": 50.0,
                    "in_service": True,
                }
            ],
            "transformers": [],
        }

    def available_contingencies(self) -> List[Dict[str, Any]]:
        return []

    def apply_contingency(self, element_type: str, element_index: int) -> Dict[str, Any]:
        raise ValueError("No contingencies in capacity test case")

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
            "name": name or "capacity_connection",
            "connection_type": connection_type,
            "is_ibr": is_ibr,
            "bus": bus,
            "resolved_bus": {"bus_index": bus, "matched_on": "capacity"},
            "p_mw": p_mw,
            "q_mvar": q_mvar,
        }

    def network_data(self, max_rows: int = 50) -> Dict[str, Any]:
        return {"case": self.case_info().to_dict(), "tables": {}}

    def case_info(self) -> CaseInfo:
        return CaseInfo(
            case_name=self.loaded_case or "capacity",
            buses=1,
            lines=1,
            transformers=0,
            loads=0,
            generators=0,
            static_generators=0,
            external_grids=1,
        )


class NonMonotonicCapacitySolver(CapacityScenarioSolver):
    def bus_results(self) -> List[Dict[str, Any]]:
        vm_pu = 1.0
        if self.connection is not None and 20.0 < self.connection.p_mw < 80.0:
            vm_pu = 0.93
        return [{"bus_index": 1, "name": "Bus 1", "vm_pu": vm_pu, "va_degree": 0.0}]


class CapacitySearchTest(unittest.TestCase):
    def test_binary_search_finds_highest_approved_capacity(self) -> None:
        result = CapacitySearchRunner(CapacityScenarioSolver).run(
            case_path="capacity",
            bus=1,
            connection_type="load",
            min_mw=0.0,
            max_mw=100.0,
            tolerance_mw=1.0,
            max_iterations=10,
        )

        self.assertEqual(result["status"], "bounded")
        self.assertGreaterEqual(result["max_approved_mw"], 39.0)
        self.assertLessEqual(result["max_approved_mw"], 40.0)
        self.assertEqual(
            result["rejection_explanation"]["limiting_stage"],
            "f1_steady_state",
        )

    def test_approved_upper_bound_reports_max_bound_approved(self) -> None:
        result = CapacitySearchRunner(CapacityScenarioSolver).run(
            case_path="capacity",
            bus=1,
            connection_type="load",
            min_mw=0.0,
            max_mw=20.0,
        )

        self.assertEqual(result["status"], "max_bound_approved")
        self.assertEqual(result["max_approved_mw"], 20.0)

    def test_rejected_lower_bound_reports_no_feasible_capacity(self) -> None:
        result = CapacitySearchRunner(CapacityScenarioSolver).run(
            case_path="capacity",
            bus=1,
            connection_type="load",
            min_mw=50.0,
            max_mw=100.0,
        )

        self.assertEqual(result["status"], "min_bound_rejected")
        self.assertIsNone(result["max_approved_mw"])
        self.assertEqual(result["upper_bound_mw"], 50.0)

    def test_monotonicity_contradiction_uses_coarse_scan_fallback(self) -> None:
        result = CapacitySearchRunner(NonMonotonicCapacitySolver).run(
            case_path="capacity",
            bus=1,
            connection_type="load",
            min_mw=50.0,
            max_mw=100.0,
            coarse_scan_points=6,
        )

        self.assertEqual(result["status"], "monotonicity_fallback")
        self.assertTrue(result["diagnostics"]["fallback_used"])
        self.assertTrue(result["diagnostics"]["monotonicity_contradictions"])
        self.assertEqual(result["diagnostics"]["bisection_iterations_before_fallback"], 0)
        self.assertEqual(result["diagnostics"]["coarse_scan_points"], 6)
        self.assertEqual(result["max_approved_mw"], 100.0)

    def test_capacity_samples_can_include_contingency_stage(self) -> None:
        result = CapacitySearchRunner(CapacityScenarioSolver).run(
            case_path="capacity",
            bus=1,
            connection_type="load",
            min_mw=0.0,
            max_mw=20.0,
            enable_contingency=True,
            include_reports=True,
        )

        self.assertEqual(result["status"], "max_bound_approved")
        best_report = result["boundary_samples"]["best_approved"]["cia_report"]
        stages = [stage["stage"] for stage in best_report["stage_reports"]]
        self.assertIn("f2_n1_contingency", stages)

    def test_direct_runner_rejects_invalid_connection_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "connection_type"):
            CapacitySearchRunner(CapacityScenarioSolver).run(
                case_path="capacity",
                bus=1,
                connection_type="unknown",
            )

    def test_direct_runner_rejects_invalid_numeric_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_mw"):
            CapacitySearchRunner(CapacityScenarioSolver).run(
                case_path="capacity",
                bus=1,
                connection_type="load",
                max_mw=float("nan"),
            )

    def test_direct_runner_rejects_invalid_ibr_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "is_ibr"):
            CapacitySearchRunner(CapacityScenarioSolver).run(
                case_path="capacity",
                bus=1,
                connection_type="load",
                is_ibr="false",  # type: ignore[arg-type]
            )

    def test_direct_runner_rejects_invalid_boolean_flags(self) -> None:
        with self.assertRaisesRegex(ValueError, "enable_contingency"):
            CapacitySearchRunner(CapacityScenarioSolver).run(
                case_path="capacity",
                bus=1,
                connection_type="load",
                enable_contingency="false",  # type: ignore[arg-type]
            )

    def test_runner_rejects_invalid_material_worsening_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "material_worsening_threshold_percent"):
            CapacitySearchRunner(
                CapacityScenarioSolver,
                material_worsening_threshold_percent=-1.0,
            )


if __name__ == "__main__":
    unittest.main()

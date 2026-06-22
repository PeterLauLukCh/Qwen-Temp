"""Tests for EMT/SCR screening."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import (  # noqa: E402
    CaseInfo,
    ConnectionRequest,
    EMTScreeningRunner,
    GridSolver,
)


class EMTSolver(GridSolver):
    def __init__(self, strength_mva: float = 50.0) -> None:
        self.loaded_case = ""
        self.connection: Optional[ConnectionRequest] = None
        self.converged = False
        self.short_circuit_mva_by_bus = {1: strength_mva}

    def available_cases(self) -> List[str]:
        return ["ieee2"]

    def load_case(self, case_name: str) -> CaseInfo:
        self.loaded_case = case_name
        return self.case_info()

    def run_powerflow(self) -> Dict[str, Any]:
        self.converged = True
        return {
            "backend": "fake",
            "converged": True,
            "case": self.case_info().to_dict(),
            "bus_summary": {"min_vm_pu": 1.0, "max_vm_pu": 1.0},
            "branch_summary": {"max_line_loading_percent": 10.0},
        }

    def bus_results(self) -> List[Dict[str, Any]]:
        return [{"bus_index": 1, "name": "Bus 1", "vm_pu": 1.0, "va_degree": 0.0}]

    def branch_results(self) -> Dict[str, List[Dict[str, Any]]]:
        return {"lines": [], "transformers": []}

    def available_contingencies(self) -> List[Dict[str, Any]]:
        return []

    def apply_contingency(self, element_type: str, element_index: int) -> Dict[str, Any]:
        return {"element_type": element_type, "element_index": element_index}

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
        return {
            "element_table": "sgen" if is_ibr else "load",
            "element_index": 0,
            "bus": bus,
            "resolved_bus": {"bus_index": bus, "matched_on": "fake"},
            "p_mw": p_mw,
            "q_mvar": q_mvar,
        }

    def network_data(self, max_rows: int = 50) -> Dict[str, Any]:
        return {"case": self.case_info().to_dict(), "tables": {}}

    def case_info(self) -> CaseInfo:
        return CaseInfo(
            case_name=self.loaded_case or "ieee2",
            buses=2,
            lines=0,
            transformers=0,
            loads=0,
            generators=1,
            static_generators=0,
            external_grids=1,
        )


def connection(p_mw: float = 10.0, *, connection_type: str = "solar") -> ConnectionRequest:
    return ConnectionRequest(
        bus=1,
        p_mw=p_mw,
        connection_type=connection_type,
        is_ibr=connection_type in {"solar", "wind", "bess", "hybrid"},
    )


class EMTScreeningTest(unittest.TestCase):
    def test_scr_screen_passes_above_threshold(self) -> None:
        result = EMTScreeningRunner(lambda: EMTSolver(50.0)).run(
            case_path="ieee2",
            connection=connection(10.0),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["emt"]["status"], "pass")
        self.assertAlmostEqual(result["metrics"]["scr"], 5.0)
        self.assertIn("emt_scr_above_threshold", result["metrics"]["reason_codes"])

    def test_scr_screen_borderline_near_threshold(self) -> None:
        result = EMTScreeningRunner(lambda: EMTSolver(31.0)).run(
            case_path="ieee2",
            connection=connection(10.0),
        )

        self.assertEqual(result["emt"]["status"], "borderline")
        self.assertFalse(result["emt"]["passed"])
        self.assertIn("emt_scr_near_threshold", result["metrics"]["reason_codes"])

    def test_scr_screen_fails_below_threshold(self) -> None:
        result = EMTScreeningRunner(lambda: EMTSolver(20.0)).run(
            case_path="ieee2",
            connection=connection(10.0),
        )

        self.assertEqual(result["emt"]["status"], "fail")
        self.assertFalse(result["emt"]["passed"])
        self.assertIn("emt_scr_below_threshold", result["metrics"]["reason_codes"])

    def test_non_ibr_connection_skips_emt_screen(self) -> None:
        result = EMTScreeningRunner(lambda: EMTSolver(50.0)).run(
            case_path="ieee2",
            connection=connection(10.0, connection_type="load"),
        )

        self.assertEqual(result["emt"]["status"], "skipped")
        self.assertTrue(result["emt"]["passed"])
        self.assertIn(
            "emt_screen_not_required_for_non_ibr_connection",
            result["metrics"]["reason_codes"],
        )

    def test_zero_project_size_fails_cleanly(self) -> None:
        result = EMTScreeningRunner(lambda: EMTSolver(50.0)).run(
            case_path="ieee2",
            connection=connection(0.0),
        )

        self.assertEqual(result["emt"]["status"], "fail")
        self.assertIn("emt_project_mva_must_be_positive", result["metrics"]["reason_codes"])


if __name__ == "__main__":
    unittest.main()

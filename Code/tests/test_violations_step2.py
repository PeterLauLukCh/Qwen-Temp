"""Tests for the Step 2 violation inspector."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import LimitProfile, ViolationInspector, profile_from_name


class ViolationInspectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inspector = ViolationInspector(LimitProfile.normal())

    def inspect_one(
        self,
        vm_pu: float = 1.0,
        line_loading: float = 50.0,
        trafo_loading: float = 50.0,
    ):
        buses = [{"bus_index": 1, "name": "Bus 1", "vm_pu": vm_pu}]
        branches = {
            "lines": [
                {
                    "line_index": 2,
                    "name": "Line 2",
                    "loading_percent": line_loading,
                    "in_service": True,
                }
            ],
            "transformers": [
                {
                    "trafo_index": 3,
                    "name": "Transformer 3",
                    "loading_percent": trafo_loading,
                    "in_service": True,
                }
            ],
        }
        return self.inspector.inspect(buses, branches)

    def test_clear_results_pass(self) -> None:
        report = self.inspect_one(vm_pu=1.0, line_loading=50.0, trafo_loading=40.0)

        self.assertEqual(report.status, "pass")
        self.assertTrue(report.passed)
        self.assertEqual(report.to_dict()["total_violations"], 0)

    def test_inside_near_voltage_limit_is_borderline(self) -> None:
        report = self.inspect_one(vm_pu=0.955)

        self.assertEqual(report.status, "borderline")
        self.assertEqual(report.borderline_count, 1)
        self.assertEqual(report.violations[0].violation_type, "low_voltage")
        self.assertLess(report.violations[0].margin, 0.0)

    def test_slight_voltage_limit_excess_is_borderline(self) -> None:
        report = self.inspect_one(vm_pu=1.055)

        self.assertEqual(report.status, "borderline")
        self.assertEqual(report.borderline_count, 1)
        self.assertEqual(report.violations[0].violation_type, "high_voltage")
        self.assertGreater(report.violations[0].margin, 0.0)

    def test_exact_voltage_borderline_boundary_is_not_hard(self) -> None:
        report = self.inspect_one(vm_pu=1.06)

        self.assertEqual(report.status, "borderline")
        self.assertEqual(report.borderline_count, 1)
        self.assertEqual(report.hard_count, 0)

    def test_large_voltage_limit_excess_is_hard(self) -> None:
        report = self.inspect_one(vm_pu=0.92)

        self.assertEqual(report.status, "fail")
        self.assertFalse(report.passed)
        self.assertEqual(report.hard_count, 1)
        self.assertEqual(report.violations[0].severity, "hard")

    def test_inside_near_thermal_limit_is_borderline(self) -> None:
        report = self.inspect_one(line_loading=97.0)

        self.assertEqual(report.status, "borderline")
        self.assertEqual(report.borderline_count, 1)
        self.assertEqual(report.violations[0].element_type, "line")
        self.assertLess(report.violations[0].margin, 0.0)

    def test_slight_thermal_limit_excess_is_borderline(self) -> None:
        report = self.inspect_one(line_loading=103.0)

        self.assertEqual(report.status, "borderline")
        self.assertEqual(report.borderline_count, 1)
        self.assertEqual(report.violations[0].violation_type, "thermal_loading")

    def test_exact_thermal_borderline_boundary_is_not_hard(self) -> None:
        report = self.inspect_one(line_loading=105.0)

        self.assertEqual(report.status, "borderline")
        self.assertEqual(report.borderline_count, 1)
        self.assertEqual(report.hard_count, 0)

    def test_large_thermal_limit_excess_is_hard(self) -> None:
        report = self.inspect_one(trafo_loading=106.0)

        self.assertEqual(report.status, "fail")
        self.assertEqual(report.hard_count, 1)
        self.assertEqual(report.violations[0].element_type, "transformer")

    def test_out_of_service_branch_is_skipped(self) -> None:
        buses = [{"bus_index": 1, "vm_pu": 1.0}]
        branches = {
            "lines": [{"line_index": 2, "loading_percent": 150.0, "in_service": False}],
            "transformers": [],
        }

        report = self.inspector.inspect(buses, branches)

        self.assertEqual(report.status, "pass")
        self.assertEqual(report.to_dict()["total_violations"], 0)

    def test_string_false_out_of_service_branch_is_skipped(self) -> None:
        buses = [{"bus_index": 1, "vm_pu": 1.0}]
        branches = {
            "lines": [{"line_index": 2, "loading_percent": 150.0, "in_service": "False"}],
            "transformers": [],
        }

        report = self.inspector.inspect(buses, branches)

        self.assertEqual(report.status, "pass")
        self.assertEqual(report.to_dict()["total_violations"], 0)

    def test_missing_measurements_are_counted(self) -> None:
        report = self.inspector.inspect(
            [{"bus_index": 1, "vm_pu": None}],
            {"lines": [{"line_index": 2, "loading_percent": None}], "transformers": []},
        )

        self.assertEqual(report.skipped_measurements, 2)
        self.assertEqual(report.status, "pass")

    def test_angle_difference_screen_is_disabled_by_default(self) -> None:
        buses = [
            {"bus_index": 1, "vm_pu": 1.0, "va_degree": 0.0},
            {"bus_index": 2, "vm_pu": 1.0, "va_degree": 60.0},
        ]
        branches = {
            "lines": [
                {
                    "line_index": 2,
                    "from_bus": 1,
                    "to_bus": 2,
                    "loading_percent": 50.0,
                    "in_service": True,
                }
            ],
            "transformers": [],
        }

        report = self.inspector.inspect(buses, branches)

        self.assertEqual(report.status, "pass")

    def test_enabled_angle_difference_screen_flags_hard_violation(self) -> None:
        inspector = ViolationInspector(LimitProfile(angle_diff_limit_degree=30.0))
        buses = [
            {"bus_index": 1, "vm_pu": 1.0, "va_degree": 0.0},
            {"bus_index": 2, "vm_pu": 1.0, "va_degree": 45.0},
        ]
        branches = {
            "lines": [
                {
                    "line_index": 2,
                    "from_bus": 1,
                    "to_bus": 2,
                    "loading_percent": 50.0,
                    "in_service": True,
                }
            ],
            "transformers": [],
        }

        report = inspector.inspect(buses, branches)

        self.assertEqual(report.status, "fail")
        self.assertEqual(report.hard_count, 1)
        self.assertEqual(report.violations[0].violation_type, "angle_difference")
        self.assertEqual(report.violations[0].observed_value, 45.0)

    def test_angle_difference_uses_wrapped_difference(self) -> None:
        inspector = ViolationInspector(LimitProfile(angle_diff_limit_degree=30.0))
        buses = [
            {"bus_index": 1, "vm_pu": 1.0, "va_degree": 170.0},
            {"bus_index": 2, "vm_pu": 1.0, "va_degree": -170.0},
        ]
        branches = {
            "lines": [
                {
                    "line_index": 2,
                    "from_bus": 1,
                    "to_bus": 2,
                    "loading_percent": 50.0,
                    "in_service": True,
                }
            ],
            "transformers": [],
        }

        report = inspector.inspect(buses, branches)

        self.assertEqual(report.status, "pass")

    def test_missing_angle_measurement_is_counted_when_angle_screen_enabled(self) -> None:
        inspector = ViolationInspector(LimitProfile(angle_diff_limit_degree=30.0))
        buses = [{"bus_index": 1, "vm_pu": 1.0, "va_degree": 0.0}]
        branches = {
            "lines": [
                {
                    "line_index": 2,
                    "from_bus": 1,
                    "to_bus": 2,
                    "loading_percent": 50.0,
                    "in_service": True,
                }
            ],
            "transformers": [],
        }

        report = inspector.inspect(buses, branches)

        self.assertEqual(report.status, "pass")
        self.assertEqual(report.skipped_measurements, 1)

    def test_emergency_profile_matches_paper_limits(self) -> None:
        profile = profile_from_name("emergency")

        self.assertEqual(profile.min_voltage_pu, 0.90)
        self.assertEqual(profile.max_voltage_pu, 1.10)
        self.assertEqual(profile.max_loading_percent, 110.0)


if __name__ == "__main__":
    unittest.main()

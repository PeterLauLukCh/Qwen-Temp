"""Dependency-light checks for the Step 1 solver adapter skeleton."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import AndesPandaPowerSolver, CaseInfo, PandaPowerSolver


class CaseInfoTest(unittest.TestCase):
    def test_to_dict_preserves_all_fields(self) -> None:
        info = CaseInfo(
            case_name="ieee14",
            buses=14,
            lines=15,
            transformers=5,
            loads=11,
            generators=4,
            static_generators=0,
            external_grids=1,
        )

        self.assertEqual(
            info.to_dict(),
            {
                "case_name": "ieee14",
                "buses": 14,
                "lines": 15,
                "transformers": 5,
                "loads": 11,
                "generators": 4,
                "static_generators": 0,
                "external_grids": 1,
            },
        )


class PandaPowerSolverContractTest(unittest.TestCase):
    def test_supported_cases_are_canonical_ieee_names(self) -> None:
        self.assertEqual(
            PandaPowerSolver.supported_cases(),
            ["ieee14", "ieee30", "ieee57", "ieee118"],
        )

    def test_case_aliases_normalize_to_canonical_names(self) -> None:
        aliases = {
            "ieee14": "ieee14",
            "case14": "ieee14",
            "14": "ieee14",
            "IEEE-118": "ieee118",
            "case_118": "ieee118",
            "IEEE 57": "ieee57",
        }

        for alias, expected in aliases.items():
            with self.subTest(alias=alias):
                self.assertEqual(PandaPowerSolver._normalize_case_name(alias), expected)

    def test_invalid_case_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported case"):
            PandaPowerSolver._normalize_case_name("ieee999")

    def test_backend_availability_check_is_boolean(self) -> None:
        self.assertIsInstance(PandaPowerSolver.is_available(), bool)

    def test_andes_pandapower_solver_is_narrow_ieee118_adapter(self) -> None:
        self.assertEqual(AndesPandaPowerSolver.supported_cases(), ["ieee118"])
        self.assertEqual(AndesPandaPowerSolver._normalize_case_name("IEEE118 dynamic"), "ieee118")
        self.assertIsInstance(AndesPandaPowerSolver.is_available(), bool)

    def test_andes_conversion_disables_internal_pandapower_verification(self) -> None:
        calls = []

        def fake_to_pandapower(system, **kwargs):
            calls.append((system, kwargs))
            return {"net": True}

        result = AndesPandaPowerSolver._convert_to_pandapower(fake_to_pandapower, "system")

        self.assertEqual(result, {"net": True})
        self.assertEqual(calls, [("system", {"verify": False})])

    def test_andes_powerflow_retry_ladder_has_robust_attempts(self) -> None:
        attempts = AndesPandaPowerSolver._runpp_attempts()

        self.assertGreaterEqual(len(attempts), 3)
        self.assertEqual(attempts[0][0], "nr_dc_50")
        self.assertTrue(any(name.startswith("iwamoto") for name, _ in attempts))
        self.assertTrue(any(options.get("max_iteration", 0) >= 100 for _, options in attempts))


if __name__ == "__main__":
    unittest.main()

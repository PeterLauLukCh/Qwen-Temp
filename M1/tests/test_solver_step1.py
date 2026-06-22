"""Dependency-light checks for the Step 1 solver adapter skeleton."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import CaseInfo, PandaPowerSolver


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


if __name__ == "__main__":
    unittest.main()

"""Tests for the frozen TRGC requirement catalog."""

from __future__ import annotations

import unittest

from gridmind_mini import (
    TRGC_LAYERS,
    TRGC_REQUIREMENT_CATALOG,
    TRGC_REQUIREMENT_SCHEMA_VERSION,
    TRGC_SUPPORT_STATUSES,
    TRGC_TECHNOLOGIES,
    get_trgc_requirement,
    list_trgc_requirements,
)


class TRGCRequirementCatalogTest(unittest.TestCase):
    def test_catalog_entries_have_unique_valid_fields(self) -> None:
        ids = [item.requirement_id for item in TRGC_REQUIREMENT_CATALOG]

        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("GFL-07", ids)
        self.assertIn("TRGC_A15_STATIC_LOAD_FLOW", ids)
        for item in TRGC_REQUIREMENT_CATALOG:
            self.assertTrue(item.requirement_id)
            self.assertTrue(item.title)
            self.assertTrue(item.annexure)
            self.assertIn(item.layer, TRGC_LAYERS)
            self.assertIn(item.technology, TRGC_TECHNOLOGIES)
            self.assertIn(item.current_support_status, TRGC_SUPPORT_STATUSES)
            self.assertTrue(item.required_capabilities)

    def test_list_and_get_return_serializable_payloads(self) -> None:
        unsupported = list_trgc_requirements(support_status="unsupported_current_remote")
        one = get_trgc_requirement("GFL-07")

        self.assertTrue(unsupported)
        self.assertEqual(one["schema_version"], TRGC_REQUIREMENT_SCHEMA_VERSION)
        self.assertEqual(one["requirement_id"], "GFL-07")
        self.assertEqual(one["current_support_status"], "unsupported_current_remote")


if __name__ == "__main__":
    unittest.main()

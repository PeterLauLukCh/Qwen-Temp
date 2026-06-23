import unittest

from gridmind_mini import (
    ToolRegistry,
    build_tool_observation,
    inspect_real_case_summary,
    inspect_real_dynamic_channels,
    inspect_real_poc_context,
    inspect_real_static_operating_point,
)


class RealCaseDossierTest(unittest.TestCase):
    def test_pif6_summary_reports_bus_count_and_poc_candidates(self):
        result = inspect_real_case_summary(case_id="pif6_2026_05_17")

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["case_size"]["bus_count"], 786)
        self.assertTrue(result["poc_candidates"])
        self.assertIn("no_disturbance_5s", result["available_remote_scenarios"])

    def test_poc_context_exposes_candidates_without_oracle_fields(self):
        result = inspect_real_poc_context(case_id="pif6_2026_05_17", poc_label_or_bus="POC2")

        self.assertTrue(result["ok"], result)
        self.assertGreaterEqual(result["candidate_count"], 1)
        self.assertEqual(result["candidate_buses"][0]["bus"], 2)
        self.assertNotIn("hidden_oracle", str(result))
        self.assertNotIn("reward_weights", str(result))

    def test_filtered_poc_context_warns_about_hidden_candidates(self):
        result = inspect_real_poc_context(case_id="pif6_2026_05_17", poc_label_or_bus="2000")

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["selector_narrowed_candidates"])
        self.assertTrue(result["ambiguous"])
        self.assertGreater(result["total_poc_bus_candidate_count"], result["candidate_count"])
        self.assertTrue(result["other_poc_candidates_preview"])
        self.assertIn("filtered_selector_can_hide_other_poc_candidates", result["limitations"])

    def test_static_operating_point_reports_voltage_and_poc(self):
        result = inspect_real_static_operating_point(case_id="pif6_2026_05_17")

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["converged"])
        self.assertEqual(result["counts"]["buses"], 786)
        self.assertAlmostEqual(result["voltage_range_pu"]["min"], 0.8999999761581421)
        self.assertIsNotNone(result["poc"]["p_mw"])
        self.assertIsNotNone(result["poc"]["q_mvar"])

    def test_dynamic_channels_report_final_values_and_rows(self):
        result = inspect_real_dynamic_channels(case_id="pif6_2026_05_17")

        self.assertTrue(result["ok"], result)
        self.assertGreater(result["row_count"], 1000)
        self.assertGreater(result["final_time_s"], 4.9)
        self.assertIn("POC_P_2001_2", result["final_values"])
        self.assertIn("POC2_V", result["voltage_extrema"])

    def test_registry_observation_compacts_dossier_tool(self):
        result = ToolRegistry().call_tool(
            "inspect_real_case_summary",
            {"case_id": "pif6_2026_05_17"},
        )
        observation = build_tool_observation(result)

        self.assertEqual(observation["tool"], "inspect_real_case_summary")
        self.assertEqual(observation["case_size"]["bus_count"], 786)


if __name__ == "__main__":
    unittest.main()

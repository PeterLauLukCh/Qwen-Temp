"""Tests for the VERL remote PSS/E M1+M2 tool core."""

from __future__ import annotations

import unittest
from typing import Any, Dict, Optional

from gridmind_mini.remote_psse_verl import RemotePsseM1M2ToolCore


class RemotePsseVerlToolCoreTest(unittest.TestCase):
    def test_core_tracks_state_and_duplicate_request_ids(self) -> None:
        calls = []

        def runner(
            *,
            case_id: str,
            scenario_type: str,
            request_id: Optional[str] = None,
            include_artifacts: bool = False,
        ) -> Dict[str, Any]:
            calls.append(
                {
                    "case_id": case_id,
                    "scenario_type": scenario_type,
                    "request_id": request_id,
                    "include_artifacts": include_artifacts,
                }
            )
            result = worker_result(job_id=f"job-{len(calls)}")
            result["tool"] = "run_remote_psse_m1m2"
            result["summary"] = {
                "m1_status": "pass",
                "m2_status": "pass",
                "m2_final_time_s": 5.00082,
                "m2_channel_row_count": 5004,
            }
            return result

        core = RemotePsseM1M2ToolCore(runner=runner)
        instance = core.create("episode-1")

        first = core.execute(
            instance,
            {
                "case_id": "pif6_2026_05_17",
                "scenario_type": "no_disturbance_5s",
                "request_id": "req-1",
                "step_index": 0,
                "expected_state_id": "initial",
            },
        )

        self.assertTrue(first["ok"], first)
        self.assertEqual(first["step_index"], 0)
        self.assertNotEqual(first["state_id"], "initial")
        self.assertEqual(first["simulation_time_s"], 5.00082)
        self.assertEqual(len(calls), 1)

        duplicate = core.execute(
            instance,
            {
                "case_id": "pif6_2026_05_17",
                "scenario_type": "no_disturbance_5s",
                "request_id": "req-1",
                "step_index": 1,
                "expected_state_id": first["state_id"],
            },
        )

        self.assertTrue(duplicate["ok"], duplicate)
        self.assertTrue(duplicate["duplicate_request"])
        self.assertEqual(duplicate["state_id"], first["state_id"])
        self.assertEqual(len(calls), 1)

        stale = core.execute(
            instance,
            {
                "case_id": "pif6_2026_05_17",
                "scenario_type": "no_disturbance_5s",
                "request_id": "req-2",
                "step_index": 1,
                "expected_state_id": "wrong-state",
            },
        )

        self.assertFalse(stale["ok"])
        self.assertEqual(stale["error_type"], "stale_state")
        self.assertEqual(len(calls), 1)

        second = core.execute(
            instance,
            {
                "case_id": "pif6_2026_05_17",
                "scenario_type": "no_disturbance_5s",
                "request_id": "req-2",
                "step_index": 1,
                "expected_state_id": first["state_id"],
            },
        )

        self.assertTrue(second["ok"], second)
        self.assertEqual(second["step_index"], 1)
        self.assertEqual(second["previous_state_id"], first["state_id"])
        self.assertEqual(len(calls), 2)


def worker_result(*, job_id: str) -> Dict[str, Any]:
    return {
        "ok": True,
        "tool": "run_remote_psse_m1m2",
        "job_id": job_id,
        "case_id": "pif6_2026_05_17",
        "scenario_type": "no_disturbance_5s",
        "display_name": "PIF6 POC2 786-bus case",
        "psse_version": "PSS(R)E Xplore 36.2.0",
        "summary": {
            "m1_status": "pass",
            "m1_bus_count": 786,
            "m2_status": "pass",
            "m2_final_time_s": 5.00082,
            "m2_channel_row_count": 5004,
        },
        "stage_reports": [],
        "recommendation": "approve",
        "complete": True,
        "reason_codes": ["psse_remote_execution_pass"],
        "limitations": ["psse_rms_dynamic_not_emt_waveform"],
    }


if __name__ == "__main__":
    unittest.main()

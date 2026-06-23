"""Tests for live remote PSS/E M1+M2 GPU-side integration."""

from __future__ import annotations

import os
import unittest
from typing import Any, Dict, Mapping, Optional

import gridmind_mini.remote_psse as remote_psse_module
from gridmind_mini import (
    REMOTE_PSSE_BASE_URL_ENV,
    REMOTE_PSSE_TOKEN_ENV,
    ToolRegistry,
    build_deterministic_report,
    build_tool_observation,
    list_remote_psse_m1m2_cases,
    run_remote_psse_m1m2,
    validate_tool_call_policy,
)


class RemotePsseIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = FakeRemotePsseTransport()
        self.old_request_json = remote_psse_module.request_json
        remote_psse_module.request_json = self.transport.request_json
        self.old_base = os.environ.get(REMOTE_PSSE_BASE_URL_ENV)
        self.old_token = os.environ.get(REMOTE_PSSE_TOKEN_ENV)
        os.environ[REMOTE_PSSE_BASE_URL_ENV] = "http://fake-remote-worker"
        os.environ[REMOTE_PSSE_TOKEN_ENV] = "test-token"

    def tearDown(self) -> None:
        remote_psse_module.request_json = self.old_request_json
        if self.old_base is None:
            os.environ.pop(REMOTE_PSSE_BASE_URL_ENV, None)
        else:
            os.environ[REMOTE_PSSE_BASE_URL_ENV] = self.old_base
        if self.old_token is None:
            os.environ.pop(REMOTE_PSSE_TOKEN_ENV, None)
        else:
            os.environ[REMOTE_PSSE_TOKEN_ENV] = self.old_token

    def test_list_remote_cases_can_health_check_worker(self) -> None:
        result = list_remote_psse_m1m2_cases(check_health=True)

        self.assertTrue(result["ok"])
        self.assertTrue(result["remote_configured"])
        self.assertEqual(result["case_count"], 2)
        self.assertTrue(result["health"]["ok"])
        self.assertEqual(self.transport.last_token, "test-token")

    def test_run_remote_m1m2_submits_polls_and_adapts_result(self) -> None:
        result = run_remote_psse_m1m2(
            case_id="pif6_2026_05_17",
            scenario_type="no_disturbance_5s",
            request_id="req-1",
            poll_interval_s=0.01,
            max_wait_s=2.0,
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["tool"], "run_remote_psse_m1m2")
        self.assertEqual(result["job_id"], "job-1")
        self.assertEqual(result["recommendation"], "approve")
        self.assertEqual(result["summary"]["m1_status"], "pass")
        self.assertEqual(result["summary"]["m2_status"], "pass")
        self.assertEqual(result["summary"]["m1_bus_count"], 786)
        self.assertEqual(result["summary"]["m2_channel_row_count"], 5004)
        self.assertAlmostEqual(result["summary"]["m2_final_time_s"], 5.00082)
        self.assertEqual(
            self.transport.submissions[-1],
            {
                "case_id": "pif6_2026_05_17",
                "scenario_type": "no_disturbance_5s",
                "request_id": "req-1",
            },
        )

    def test_unsupported_remote_scenario_fails_before_network(self) -> None:
        before = len(self.transport.submissions)

        result = run_remote_psse_m1m2(
            case_id="pif6_2026_05_17",
            scenario_type="pq_target_step",
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "unsupported_remote_scenario_type")
        self.assertEqual(len(self.transport.submissions), before)

    def test_tool_registry_exposes_remote_psse_tools(self) -> None:
        registry = ToolRegistry()
        names = {tool["name"] for tool in registry.list_tools(include_unimplemented=False)["tools"]}

        self.assertIn("list_remote_psse_m1m2_cases", names)
        self.assertIn("run_remote_psse_m1m2", names)
        result = registry.call_tool(
            "run_remote_psse_m1m2",
            {
                "case_id": "pif6_2026_05_17",
                "scenario_type": "no_disturbance_5s",
                "request_id": "registry-1",
            },
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["summary"]["m2_status"], "pass")

    def test_tool_registry_accepts_llm_alias_and_forwards_normalized_scenario(self) -> None:
        registry = ToolRegistry()

        result = registry.call_tool(
            "run_remote_psse_m1m2",
            {
                "case_id": "pif6_2026_05_17",
                "scenario_type": "baseline",
                "request_id": "alias-1",
            },
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(
            self.transport.submissions[-1],
            {
                "case_id": "pif6_2026_05_17",
                "scenario_type": "no_disturbance_5s",
                "request_id": "alias-1",
            },
        )

    def test_observation_and_report_include_remote_state(self) -> None:
        result = run_remote_psse_m1m2(
            case_id="pif6_2026_05_17",
            scenario_type="no_disturbance_5s",
            poll_interval_s=0.01,
        )

        observation = build_tool_observation(result)
        report = build_deterministic_report([result])

        self.assertEqual(observation["tool"], "run_remote_psse_m1m2")
        self.assertEqual(observation["job_id"], "job-1")
        self.assertEqual(observation["summary"]["m2_channel_row_count"], 5004)
        self.assertTrue(report.available)
        self.assertIn("Remote PSS/E M1+M2", report.summary_text)
        self.assertIn("final_time", report.summary_text)

    def test_failed_remote_observation_keeps_execution_error_lines(self) -> None:
        failed_result = {
            "ok": False,
            "tool": "run_remote_psse_m1m2",
            "error_type": "remote_psse_job_failed",
            "message": "PSS/E remote job failed.",
            "reason_codes": ["m2_psse_rms_dynamic_failed"],
            "execution": {
                "executor": "pssecmd36_50000_bus_idv",
                "returncode": 1,
                "error_lines": [
                    "Initial conditions suspect",
                    "PSS/E run failed at t=1.0",
                ],
            },
        }

        observation = build_tool_observation(failed_result)
        report = build_deterministic_report([failed_result])

        self.assertEqual(observation["status"], "error")
        self.assertEqual(observation["execution"]["returncode"], 1)
        self.assertIn("PSS/E run failed at t=1.0", observation["execution"]["error_lines"])
        self.assertIn("PSS/E run failed at t=1.0", report.summary_text)
        self.assertIn("PSS/E run failed at t=1.0", report.facts["execution"]["error_lines"])

    def test_guardrail_routes_live_remote_psse_away_from_frozen_tool(self) -> None:
        decision = validate_tool_call_policy(
            tool_name="run_real_psse_assessment",
            user_message=(
                "Use the live remote PSS/E TCP/IP Windows worker for the PIF6 "
                "M1+M2 gym baseline."
            ),
            arguments={"case_id": "pif6_baseline"},
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.recommended_tool, "run_remote_psse_m1m2")

    def test_guardrail_rejects_trgc_baseline_proxy(self) -> None:
        decision = validate_tool_call_policy(
            tool_name="run_remote_psse_m1m2",
            user_message=(
                "Use the live remote PSS/E M1+M2 gym no-disturbance baseline "
                "to validate TRGC GFL-07 fault ride-through."
            ),
            arguments={"case_id": "pif6_2026_05_17", "scenario_type": "no_disturbance_5s"},
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.recommended_tool, "list_remote_psse_m1m2_cases")

    def test_guardrail_allows_context_approved_trgc_subset(self) -> None:
        decision = validate_tool_call_policy(
            tool_name="run_remote_psse_m1m2",
            user_message=(
                "TRGC Annexure 16 is broader than the current remote gym, but the "
                "specific pif6_2026_05_17/no_disturbance_5s subset is allowlisted. "
                "Run that exact job, then state that faults, droop, SCR, PSCAD, EMT, "
                "and field-validation requirements remain outside this result."
            ),
            arguments={"case_id": "pif6_2026_05_17", "scenario_type": "no_disturbance_5s"},
            context={
                "remote_psse_m1m2_gym": True,
                "case_id": "pif6_2026_05_17",
                "scenario_type": "no_disturbance_5s",
                "trgc_requirement": {
                    "current_support_status": "executable_current_remote",
                    "current_remote_scenario_type": "no_disturbance_5s",
                },
            },
        )

        self.assertTrue(decision.allowed)


class FakeRemotePsseTransport:
    def __init__(self) -> None:
        self.submissions = []
        self.last_token = None

    def request_json(
        self,
        method: str,
        config: Any,
        path: str,
        *,
        body: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.last_token = config.token
        if method == "GET" and path == "/health":
            return {
                "ok": True,
                "service": "psse_remote_worker",
                "queue_policy": "single_serial_worker",
            }
        if method == "POST" and path == "/jobs":
            payload = dict(body or {})
            self.submissions.append(payload)
            return {
                "ok": True,
                "job_id": "job-1",
                "job": {
                    "job_id": "job-1",
                    "status": "queued",
                    "case_id": payload.get("case_id"),
                    "scenario_type": payload.get("scenario_type"),
                },
            }
        if method == "GET" and path == "/jobs/job-1":
            return {
                "ok": True,
                "job": {
                    "job_id": "job-1",
                    "status": "completed",
                },
            }
        if method == "GET" and path == "/jobs/job-1/result":
            return worker_result(job_id="job-1")
        if method == "GET" and path == "/jobs/job-1/artifacts":
            return {"ok": True, "artifact_dir": "fake", "files": []}
        return {"ok": False, "error_type": "not_found", "message": path}


def worker_result(*, job_id: str = "job-1") -> Dict[str, Any]:
    return {
        "ok": True,
        "schema_version": "psse_remote_result_v1",
        "tool": "run_remote_psse_job",
        "backend": "psse_remote_worker",
        "job_id": job_id,
        "case_id": "pif6_2026_05_17",
        "scenario_type": "no_disturbance_5s",
        "display_name": "PIF6 POC2 786-bus case",
        "psse_version": "PSS(R)E Xplore 36.2.0",
        "m1_powerflow": {
            "status": "pass",
            "converged": True,
            "bus_count": 786,
            "branch_count": 790,
            "machine_count": 251,
            "load_count": 2,
            "bus_voltage_min_pu": 0.9000000357,
            "bus_voltage_max_pu": 0.9061174989,
            "poc_p_mw": 5.0867,
            "poc_q_mvar": -19.3347,
            "overloaded_branch_count": 0,
            "error_summary": [],
        },
        "m2_dynamic": {
            "status": "pass",
            "initialized": True,
            "simulation_converged": True,
            "event": "no_disturbance_5s",
            "metrics": {
                "min_voltage_pu": 0.9000000357,
                "max_voltage_pu": 0.9061174989,
                "initial_poc_p_mw": 5.0867,
                "initial_poc_q_mvar": -19.3347,
                "final_poc_p_mw": 5.13185,
                "final_poc_q_mvar": -19.28416,
                "final_time_s": 5.00082,
                "channel_row_count": 5004,
                "max_frequency_deviation_pu": 0.0,
                "max_frequency_deviation_hz_at_60hz": 0.0,
            },
            "error_summary": [],
        },
        "emt_result": {
            "status": "not_run",
            "backend": "not_available_in_remote_v1",
            "reason_codes": ["psse_rms_dynamic_not_emt_waveform"],
        },
        "recommendation": "approve",
        "complete": True,
        "reason_codes": ["psse_remote_execution_pass"],
        "limitations": [
            "pif6_uses_pssecmd36_50000_bus_idv_path",
            "psse_rms_dynamic_not_emt_waveform",
        ],
        "execution": {
            "executor": "pssecmd36_50000_bus_idv",
            "returncode": 0,
            "error_lines": [],
        },
    }

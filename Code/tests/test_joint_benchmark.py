"""Tests for the true joint M1+M2 benchmark suite."""

from __future__ import annotations

import contextlib
import copy
import io
import unittest
from typing import Any, Dict, List, Mapping, Optional

from gridmind_mini import (
    AgentTurnResult,
    GroundingValidation,
    JointBenchmarkRunner,
    ToolExecutionRecord,
    default_joint_benchmark_scenarios,
    evaluate_joint_result,
    filter_joint_scenarios,
    generate_joint_benchmark_scenarios,
    joint_benchmark_scenarios_from_payload,
    run_joint_oracles,
)


class FakeAgent:
    def __init__(self, result: AgentTurnResult) -> None:
        self.result = result
        self.calls: List[Dict[str, Any]] = []

    def run_turn(
        self,
        message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> AgentTurnResult:
        self.calls.append({"message": message, "context": context})
        return self.result


class FakeOracleRegistry:
    def __init__(self, result: Dict[str, Any]) -> None:
        self.result = result
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        result = dict(self.result)
        result.setdefault("tool", name)
        return result


class NoisyOracleRegistry(FakeOracleRegistry):
    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        print("solver progress message that should not enter benchmark JSON")
        return super().call_tool(name, arguments)


def agent_result(
    *,
    status: str = "completed",
    records: Optional[List[ToolExecutionRecord]] = None,
    invoked_tools: Optional[List[str]] = None,
    output_text: str = (
        "Recommendation approve. M1 CIA passed. M2 transient stability passed "
        "with max angle spread metric. The new project is represented as static PQ."
    ),
    warning_appended: bool = False,
) -> AgentTurnResult:
    return AgentTurnResult(
        status=status,
        output_text=output_text,
        raw_output_text=output_text,
        messages=[],
        tool_records=records or [],
        invoked_tools=invoked_tools or [],
        grounding=GroundingValidation(
            tool_grounded=bool(invoked_tools),
            claims=[],
            ungrounded_claims=[],
            warning_appended=warning_appended,
            output_text=output_text,
            warning=None,
        ),
    )


def tool_record(
    name: str,
    arguments: Dict[str, Any],
    result: Dict[str, Any],
    *,
    ok: bool = True,
) -> ToolExecutionRecord:
    return ToolExecutionRecord(
        call_id="call_1",
        name=name,
        arguments=arguments,
        source="native",
        ok=ok,
        result=result,
        error=None if ok else "tool failed",
        observation={"tool": name},
    )


class JointBenchmarkTest(unittest.TestCase):
    def test_default_suite_has_thirty_seven_unique_scenarios(self) -> None:
        scenarios = default_joint_benchmark_scenarios()

        self.assertEqual(len(scenarios), 37)
        self.assertEqual(len({scenario.scenario_id for scenario in scenarios}), 37)
        self.assertIn(
            "joint_ieee14_solar_fault_100ms",
            {scenario.scenario_id for scenario in scenarios},
        )
        self.assertIn(
            "joint_ieee14_load_bus12_optional",
            {scenario.scenario_id for scenario in scenarios},
        )
        self.assertIn(
            "joint_ieee118_solar_fault_100ms",
            {scenario.scenario_id for scenario in scenarios},
        )
        self.assertIn(
            "joint_emt_ieee14_solar_scr_pass",
            {scenario.scenario_id for scenario in scenarios},
        )

    def test_filter_by_tag_and_id(self) -> None:
        scenarios = default_joint_benchmark_scenarios()

        by_id = filter_joint_scenarios(
            scenarios,
            scenario_ids=["joint_ieee14_solar_fault_100ms"],
        )
        by_live_safe = filter_joint_scenarios(scenarios, tags=["live_safe"])
        by_tag = filter_joint_scenarios(scenarios, tags=["ieee118_live_safe"])
        by_emt = filter_joint_scenarios(scenarios, tags=["emt_live_safe"])

        self.assertEqual([scenario.scenario_id for scenario in by_id], ["joint_ieee14_solar_fault_100ms"])
        self.assertEqual(len(by_live_safe), 20)
        self.assertTrue(all("ieee14" in scenario.tags for scenario in by_live_safe))
        self.assertEqual(len(by_tag), 5)
        self.assertEqual(len(by_emt), 10)
        self.assertTrue(all("emt" in scenario.tags for scenario in by_emt))

    def test_generated_suite_is_reproducible_and_unique(self) -> None:
        first = generate_joint_benchmark_scenarios(
            100,
            seed=1234,
            profile="mixed",
        )
        second = generate_joint_benchmark_scenarios(
            100,
            seed=1234,
            profile="mixed",
        )

        self.assertEqual(len(first), 100)
        self.assertEqual(len({scenario.scenario_id for scenario in first}), 100)
        self.assertEqual(
            [scenario.to_dict() for scenario in first],
            [scenario.to_dict() for scenario in second],
        )
        self.assertTrue(all("generated" in scenario.tags for scenario in first))
        self.assertTrue(all("ieee14" in scenario.tags for scenario in first))
        self.assertTrue(
            all(scenario.expected_tool == "run_integrated_assessment" for scenario in first)
        )

    def test_generated_profiles_control_emt_coverage(self) -> None:
        m1m2 = generate_joint_benchmark_scenarios(30, seed=99, profile="m1m2")
        emt = generate_joint_benchmark_scenarios(30, seed=99, profile="emt")
        hard = generate_joint_benchmark_scenarios(30, seed=99, profile="hard_emt")

        self.assertTrue(all("emt" not in scenario.oracle_arguments for scenario in m1m2))
        self.assertTrue(all("emt" in scenario.oracle_arguments for scenario in emt))
        self.assertTrue(any("generated_emt" in scenario.tags for scenario in emt))
        self.assertTrue(all("emt" in scenario.oracle_arguments for scenario in hard))
        self.assertTrue(all("generated_hard_emt" in scenario.tags for scenario in hard))

    def test_generated_hard_emt_profile_contains_harder_labels(self) -> None:
        scenarios = generate_joint_benchmark_scenarios(
            120,
            seed=20260610,
            profile="hard_emt",
        )
        emt_statuses = {
            scenario.oracle_result_template["summary"].get("emt_status")
            for scenario in scenarios
        }
        recommendations = {
            scenario.oracle_result_template.get("recommendation")
            for scenario in scenarios
        }
        thresholds = {
            scenario.oracle_arguments["emt"].get("scr_threshold")
            for scenario in scenarios
        }

        self.assertTrue({"borderline", "fail"}.intersection(emt_statuses))
        self.assertTrue({"borderline", "reject"}.intersection(recommendations))
        self.assertTrue(any(float(threshold) > 3.0 for threshold in thresholds))
        self.assertTrue(
            all("static pq" in scenario.output_contains for scenario in scenarios)
        )

    def test_generated_emt_failures_are_rejects(self) -> None:
        scenarios = generate_joint_benchmark_scenarios(
            200,
            seed=20260610,
            profile="mixed",
        )
        failing_emt = [
            scenario for scenario in scenarios
            if scenario.oracle_result_template["summary"]["emt_status"] == "fail"
        ]

        self.assertTrue(failing_emt)
        self.assertTrue(
            all(
                scenario.oracle_result_template["recommendation"] == "reject"
                for scenario in failing_emt
            )
        )

    def test_generated_q_mvar_values_are_positive_when_present(self) -> None:
        scenarios = generate_joint_benchmark_scenarios(
            100,
            seed=20260610,
            profile="mixed",
        )
        q_values = [
            scenario.oracle_arguments["connection"]["q_mvar"]
            for scenario in scenarios
            if "q_mvar" in scenario.oracle_arguments["connection"]
        ]

        self.assertTrue(q_values)
        self.assertTrue(all(value > 0.0 for value in q_values))

    def test_generated_suite_can_be_filtered_by_generated_tag(self) -> None:
        scenarios = default_joint_benchmark_scenarios() + generate_joint_benchmark_scenarios(
            25,
            seed=7,
            profile="mixed",
        )

        selected = filter_joint_scenarios(scenarios, tags=["generated_ieee14"])

        self.assertEqual(len(selected), 25)
        self.assertTrue(all(scenario.scenario_id.startswith("generated_ieee14") for scenario in selected))

    def test_generated_suite_round_trips_through_saved_payload(self) -> None:
        scenarios = generate_joint_benchmark_scenarios(
            12,
            seed=123,
            profile="mixed",
        )
        payload = {
            "ok": True,
            "scenario_source": "generated",
            "generation": {
                "enabled": True,
                "count": 12,
                "seed": 123,
                "profile": "mixed",
            },
            "scenario_count": len(scenarios),
            "scenarios": [scenario.to_dict() for scenario in scenarios],
        }

        loaded = joint_benchmark_scenarios_from_payload(payload)

        self.assertEqual(
            [scenario.to_dict() for scenario in loaded],
            [scenario.to_dict() for scenario in scenarios],
        )

    def test_generated_suite_round_trips_from_bare_scenario_list(self) -> None:
        scenarios = generate_joint_benchmark_scenarios(
            3,
            seed=456,
            profile="emt",
        )

        loaded = joint_benchmark_scenarios_from_payload(
            [scenario.to_dict() for scenario in scenarios]
        )

        self.assertEqual(
            [scenario.to_dict() for scenario in loaded],
            [scenario.to_dict() for scenario in scenarios],
        )

    def test_saved_payload_rejects_duplicate_scenario_ids(self) -> None:
        scenario = generate_joint_benchmark_scenarios(1, seed=789)[0].to_dict()

        with self.assertRaises(ValueError):
            joint_benchmark_scenarios_from_payload([scenario, dict(scenario)])

    def test_saved_payload_rejects_missing_scenarios_field(self) -> None:
        with self.assertRaises(ValueError):
            joint_benchmark_scenarios_from_payload({"ok": True})

    def test_generated_scenario_evaluates_against_template(self) -> None:
        scenario = generate_joint_benchmark_scenarios(
            1,
            seed=2026,
            profile="emt",
        )[0]
        record = tool_record(
            "run_integrated_assessment",
            dict(scenario.oracle_arguments),
            dict(scenario.oracle_result_template),
        )
        result = agent_result(
            records=[record],
            invoked_tools=["run_integrated_assessment"],
            output_text=(
                "Recommendation approve. Integrated assessment completed with "
                "M1, M2, and EMT/SCR metrics."
            ),
        )

        checks = evaluate_joint_result(
            scenario,
            agent_result=result,
            oracle_result=dict(scenario.oracle_result_template),
        )

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_generated_profile_rejects_invalid_count(self) -> None:
        with self.assertRaises(ValueError):
            generate_joint_benchmark_scenarios(0)

    def test_generated_profile_rejects_invalid_profile(self) -> None:
        with self.assertRaises(ValueError):
            generate_joint_benchmark_scenarios(1, profile="unknown")

    def test_oracle_template_mode_validates_arguments_without_execution(self) -> None:
        scenario = default_joint_benchmark_scenarios()[0]
        registry = FakeOracleRegistry({"ok": False})

        outputs = run_joint_oracles([scenario], registry, execute_tools=False)

        self.assertEqual(registry.calls, [])
        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0]["ok"])
        self.assertFalse(outputs[0]["oracle_executed"])

    def test_live_oracle_mode_executes_registry(self) -> None:
        scenario = default_joint_benchmark_scenarios()[0]
        registry = FakeOracleRegistry(dict(scenario.oracle_result_template))

        outputs = run_joint_oracles([scenario], registry, execute_tools=True)

        self.assertEqual(len(registry.calls), 1)
        self.assertEqual(registry.calls[0][0], "run_integrated_assessment")
        self.assertTrue(outputs[0]["ok"])
        self.assertTrue(outputs[0]["oracle_executed"])

    def test_live_oracle_mode_fails_when_runtime_result_misses_expected_paths(self) -> None:
        scenario = default_joint_benchmark_scenarios()[0]
        runtime_result = copy.deepcopy(scenario.oracle_result_template)
        runtime_result["complete"] = False
        runtime_result["summary"]["m1_recommendation"] = None
        runtime_result["m2_result"]["dynamic_interconnection_modeling"] = False
        runtime_result["m2_result"]["connection_application"]["applied"] = False
        registry = FakeOracleRegistry(runtime_result)

        outputs = run_joint_oracles([scenario], registry, execute_tools=True)

        self.assertFalse(outputs[0]["ok"])
        failed_checks = [
            check["name"]
            for check in outputs[0]["oracle_path_checks"]
            if not check["passed"]
        ]
        self.assertIn("oracle_expected_result:summary.m1_recommendation", failed_checks)
        self.assertIn(
            "oracle_expected_result:m2_result.dynamic_interconnection_modeling",
            failed_checks,
        )
        self.assertIn(
            "oracle_expected_result:m2_result.connection_application.applied",
            failed_checks,
        )

    def test_live_oracle_mode_keeps_solver_noise_off_stdout(self) -> None:
        scenario = default_joint_benchmark_scenarios()[0]
        registry = NoisyOracleRegistry(dict(scenario.oracle_result_template))
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            outputs = run_joint_oracles([scenario], registry, execute_tools=True)

        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("solver progress message", stderr.getvalue())
        self.assertTrue(outputs[0]["ok"])

    def test_evaluator_passes_integrated_tool_with_template_result(self) -> None:
        scenario = default_joint_benchmark_scenarios()[0]
        record = tool_record(
            "run_integrated_assessment",
            dict(scenario.oracle_arguments),
            dict(scenario.oracle_result_template),
        )
        result = agent_result(records=[record], invoked_tools=["run_integrated_assessment"])

        checks = evaluate_joint_result(
            scenario,
            agent_result=result,
            oracle_result=dict(scenario.oracle_result_template),
        )

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_evaluator_prefers_successful_retry_of_integrated_tool(self) -> None:
        scenario = default_joint_benchmark_scenarios()[0]
        failed_record = tool_record(
            "run_integrated_assessment",
            {"case_path": "ieee14"},
            {"ok": False, "tool": "run_integrated_assessment"},
            ok=False,
        )
        successful_record = tool_record(
            "run_integrated_assessment",
            dict(scenario.oracle_arguments),
            dict(scenario.oracle_result_template),
        )
        result = agent_result(
            records=[failed_record, successful_record],
            invoked_tools=["run_integrated_assessment", "run_integrated_assessment"],
        )

        checks = evaluate_joint_result(
            scenario,
            agent_result=result,
            oracle_result=dict(scenario.oracle_result_template),
        )

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_supported_joint_case_does_not_require_hardcoded_approve_or_pass(self) -> None:
        scenario = default_joint_benchmark_scenarios()[0]
        live_like_result = dict(scenario.oracle_result_template)
        live_like_result["recommendation"] = "borderline"
        live_like_result["complete"] = True
        live_like_result["summary"] = {
            **dict(live_like_result["summary"]),
            "m1_recommendation": "borderline",
            "m2_status": "borderline",
        }
        stage_reports = [dict(stage) for stage in live_like_result["stage_reports"]]
        stage_reports[0]["status"] = "borderline"
        stage_reports[0]["recommendation"] = "borderline"
        stage_reports[1]["status"] = "borderline"
        live_like_result["stage_reports"] = stage_reports
        live_like_result["m2_result"] = {
            **dict(live_like_result["m2_result"]),
            "stability": {"status": "borderline"},
        }
        record = tool_record(
            "run_integrated_assessment",
            dict(scenario.oracle_arguments),
            live_like_result,
        )
        result = agent_result(records=[record], invoked_tools=["run_integrated_assessment"])

        checks = evaluate_joint_result(
            scenario,
            agent_result=result,
            oracle_result=live_like_result,
        )

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_evaluator_accepts_omitted_true_transient_defaults(self) -> None:
        scenario = default_joint_benchmark_scenarios()[0]
        arguments = dict(scenario.oracle_arguments)
        transient = dict(arguments["transient"])
        transient.pop("enabled")
        transient.pop("required_for_approval")
        arguments["transient"] = transient
        record = tool_record(
            "run_integrated_assessment",
            arguments,
            dict(scenario.oracle_result_template),
        )
        result = agent_result(records=[record], invoked_tools=["run_integrated_assessment"])

        checks = evaluate_joint_result(
            scenario,
            agent_result=result,
            oracle_result=dict(scenario.oracle_result_template),
        )

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_evaluator_accepts_fault_clearing_duration_when_start_time_is_known(self) -> None:
        scenario = default_joint_benchmark_scenarios()[0]
        arguments = dict(scenario.oracle_arguments)
        transient = dict(arguments["transient"])
        disturbance = dict(transient["disturbance"])
        disturbance["clearing_time_s"] = 0.1
        transient["disturbance"] = disturbance
        arguments["transient"] = transient
        record = tool_record(
            "run_integrated_assessment",
            arguments,
            dict(scenario.oracle_result_template),
        )
        result = agent_result(records=[record], invoked_tools=["run_integrated_assessment"])

        checks = evaluate_joint_result(
            scenario,
            agent_result=result,
            oracle_result=dict(scenario.oracle_result_template),
        )

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_evaluator_accepts_defaulted_required_emt_arguments(self) -> None:
        scenario = next(
            item for item in default_joint_benchmark_scenarios()
            if item.scenario_id == "joint_emt_ieee14_bess_q_support"
        )
        arguments = copy.deepcopy(scenario.oracle_arguments)
        arguments["emt"].pop("required_for_approval")
        arguments["emt"].pop("scr_threshold")
        record = tool_record(
            "run_integrated_assessment",
            arguments,
            dict(scenario.oracle_result_template),
        )
        result = agent_result(
            records=[record],
            invoked_tools=["run_integrated_assessment"],
            output_text="Recommendation reject. M1 CIA, M2 stability, EMT SCR, and static model details are reported.",
        )

        checks = evaluate_joint_result(
            scenario,
            agent_result=result,
            oracle_result=dict(scenario.oracle_result_template),
        )

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_evaluator_keeps_advisory_emt_required_flag_strict(self) -> None:
        scenario = next(
            item for item in default_joint_benchmark_scenarios()
            if item.scenario_id == "joint_emt_ieee14_load_skips_scr"
        )
        arguments = copy.deepcopy(scenario.oracle_arguments)
        arguments["emt"].pop("required_for_approval")
        record = tool_record(
            "run_integrated_assessment",
            arguments,
            dict(scenario.oracle_result_template),
        )
        result = agent_result(
            records=[record],
            invoked_tools=["run_integrated_assessment"],
            output_text="Recommendation reject. EMT SCR screening is skipped for this non-IBR load.",
        )

        checks = evaluate_joint_result(
            scenario,
            agent_result=result,
            oracle_result=dict(scenario.oracle_result_template),
        )
        failed_names = {check.name for check in checks if not check.passed}

        self.assertIn("argument:emt.required_for_approval", failed_names)

    def test_output_contains_normalizes_ieee118_and_static_pq_spelling(self) -> None:
        scenario = next(
            item for item in default_joint_benchmark_scenarios()
            if item.scenario_id == "joint_ieee118_solar_fault_100ms"
        )
        record = tool_record(
            "run_integrated_assessment",
            dict(scenario.oracle_arguments),
            dict(scenario.oracle_result_template),
        )
        result = agent_result(
            records=[record],
            invoked_tools=["run_integrated_assessment"],
            output_text="Integrated result for ieee118 public benchmark with a static_pq connection model.",
        )

        checks = evaluate_joint_result(
            scenario,
            agent_result=result,
            oracle_result=dict(scenario.oracle_result_template),
        )

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_evaluator_allows_dynamic_case_lookup_before_integrated_tool(self) -> None:
        scenario = default_joint_benchmark_scenarios()[0]
        records = [
            tool_record(
                "list_dynamic_cases",
                {},
                {"ok": True, "tool": "list_dynamic_cases", "cases": []},
            ),
            tool_record(
                "run_integrated_assessment",
                dict(scenario.oracle_arguments),
                dict(scenario.oracle_result_template),
            ),
        ]
        result = agent_result(
            records=records,
            invoked_tools=["list_dynamic_cases", "run_integrated_assessment"],
        )

        checks = evaluate_joint_result(
            scenario,
            agent_result=result,
            oracle_result=dict(scenario.oracle_result_template),
        )

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_evaluator_fails_when_agent_calls_only_run_cia(self) -> None:
        scenario = next(
            item for item in default_joint_benchmark_scenarios()
            if item.scenario_id == "joint_wrong_tool_trap"
        )
        record = tool_record("run_cia", {"case_path": "ieee14"}, {"ok": True, "tool": "run_cia"})
        result = agent_result(records=[record], invoked_tools=["run_cia"])

        checks = evaluate_joint_result(
            scenario,
            agent_result=result,
            oracle_result=dict(scenario.oracle_result_template),
        )
        failed_names = {check.name for check in checks if not check.passed}

        self.assertIn("expected_tool", failed_names)
        self.assertIn("forbidden_successful_tools", failed_names)

    def test_evaluator_fails_when_agent_calls_only_standalone_transient(self) -> None:
        scenario = next(
            item for item in default_joint_benchmark_scenarios()
            if item.scenario_id == "joint_wrong_tool_trap"
        )
        record = tool_record(
            "run_transient_stability",
            {"case_path": "ieee14_dynamic"},
            {"ok": True, "tool": "run_transient_stability"},
        )
        result = agent_result(records=[record], invoked_tools=["run_transient_stability"])

        checks = evaluate_joint_result(
            scenario,
            agent_result=result,
            oracle_result=dict(scenario.oracle_result_template),
        )
        failed_names = {check.name for check in checks if not check.passed}

        self.assertIn("expected_tool", failed_names)
        self.assertIn("forbidden_successful_tools", failed_names)

    def test_evaluator_fails_when_agent_calls_only_standalone_emt(self) -> None:
        scenario = next(
            item for item in default_joint_benchmark_scenarios()
            if item.scenario_id == "joint_emt_wrong_tool_trap"
        )
        record = tool_record(
            "run_emt_screening",
            {
                "case_path": "ieee14",
                "connection": {"bus": 10, "p_mw": 5.0, "connection_type": "solar", "is_ibr": True},
            },
            {"ok": True, "tool": "run_emt_screening"},
        )
        result = agent_result(records=[record], invoked_tools=["run_emt_screening"])

        checks = evaluate_joint_result(
            scenario,
            agent_result=result,
            oracle_result=dict(scenario.oracle_result_template),
        )
        failed_names = {check.name for check in checks if not check.passed}

        self.assertIn("expected_tool", failed_names)
        self.assertIn("forbidden_successful_tools", failed_names)

    def test_clarification_case_requires_no_solver_tool(self) -> None:
        scenario = next(
            item for item in default_joint_benchmark_scenarios()
            if item.scenario_id == "joint_missing_fault_timing"
        )
        result = agent_result(
            status="clarification_required",
            output_text="Please provide the bus fault clearing time.",
        )

        checks = evaluate_joint_result(
            scenario,
            agent_result=result,
            oracle_result=None,
        )

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_ieee118_templates_expect_strict_public_case_alignment(self) -> None:
        scenario = next(
            item for item in default_joint_benchmark_scenarios()
            if item.scenario_id == "joint_ieee118_solar_fault_100ms"
        )
        self.assertTrue(scenario.oracle_result_template["case_alignment"]["strict_match"])
        self.assertEqual(
            scenario.oracle_result_template["case_alignment"]["source"],
            "powerfulcases_ieee118_public_dynamic",
        )
        self.assertTrue(scenario.oracle_result_template["m2_result"]["dynamic_interconnection_modeling"])

    def test_emt_joint_template_requires_integrated_emt_stage(self) -> None:
        scenario = next(
            item for item in default_joint_benchmark_scenarios()
            if item.scenario_id == "joint_emt_ieee14_solar_scr_pass"
        )

        self.assertEqual(scenario.expected_tool, "run_integrated_assessment")
        self.assertEqual(scenario.oracle_arguments["emt"]["enabled"], True)
        self.assertEqual(scenario.expected_tool_arguments["emt.scr_threshold"], 3.0)
        self.assertEqual(
            scenario.oracle_result_template["stage_reports"][2]["stage"],
            "f4_emt_screening",
        )
        self.assertEqual(scenario.oracle_result_template["summary"]["emt_status"], "pass")
        self.assertTrue(scenario.oracle_result_template["linkage"]["emt_models_new_connection"])
        self.assertGreater(scenario.oracle_result_template["summary"]["emt_scr"], 3.0)

    def test_emt_joint_template_can_reject_required_low_scr_case(self) -> None:
        scenario = next(
            item for item in default_joint_benchmark_scenarios()
            if item.scenario_id == "joint_emt_ieee14_solar_scr_fail"
        )

        self.assertEqual(scenario.oracle_result_template["recommendation"], "reject")
        self.assertEqual(scenario.oracle_result_template["summary"]["emt_status"], "fail")
        self.assertIn("emt_failed", scenario.oracle_result_template["reason_codes"])

    def test_emt_joint_template_skips_non_ibr_load_screen(self) -> None:
        scenario = next(
            item for item in default_joint_benchmark_scenarios()
            if item.scenario_id == "joint_emt_ieee14_load_skips_scr"
        )

        self.assertEqual(scenario.oracle_result_template["summary"]["emt_status"], "skipped")
        self.assertFalse(scenario.oracle_result_template["linkage"]["emt_models_new_connection"])
        self.assertEqual(
            scenario.oracle_result_template["emt_result"]["connection_application"]["reason"],
            "non_ibr_connection",
        )

    def test_ieee118_local_override_source_satisfies_public_source_check(self) -> None:
        scenario = next(
            item for item in default_joint_benchmark_scenarios()
            if item.scenario_id == "joint_ieee118_solar_fault_100ms"
        )
        result = copy.deepcopy(scenario.oracle_result_template)
        result["case_alignment"]["source"] = "local_ieee118_raw_dyr_override"
        record = tool_record(
            "run_integrated_assessment",
            dict(scenario.oracle_arguments),
            result,
        )
        checks = evaluate_joint_result(
            scenario,
            agent_result=agent_result(
                records=[record],
                invoked_tools=["run_integrated_assessment"],
                output_text=(
                    "IEEE 118 public benchmark result: M1 CIA passed and M2 "
                    "transient stability passed with max angle spread metric. "
                    "The added solar project uses a static PQ approximation."
                ),
            ),
            oracle_result=result,
        )

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_runner_invokes_agent_and_scores_suite(self) -> None:
        scenario = default_joint_benchmark_scenarios()[0]
        record = tool_record(
            "run_integrated_assessment",
            dict(scenario.oracle_arguments),
            dict(scenario.oracle_result_template),
        )
        fake_agent = FakeAgent(agent_result(records=[record], invoked_tools=["run_integrated_assessment"]))
        fake_oracle = FakeOracleRegistry(dict(scenario.oracle_result_template))
        runner = JointBenchmarkRunner(fake_agent, fake_oracle)

        suite = runner.run_suite([scenario])

        self.assertTrue(suite.ok)
        self.assertEqual(suite.total, 1)
        self.assertEqual(len(fake_agent.calls), 1)


if __name__ == "__main__":
    unittest.main()

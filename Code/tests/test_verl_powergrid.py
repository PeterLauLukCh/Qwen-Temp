"""Tests for the verl PowerGrid RL adapter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gridmind_mini.verl_powergrid import (
    EXPECTED_TOOL_NAME,
    FrozenIntegratedAssessmentToolCore,
    audit_tool_arguments,
    build_frozen_tool_observation,
    build_powergrid_system_prompt,
    build_verl_records,
    export_powergrid_verl_dataset,
    extract_final_answer_text,
    extract_tool_calls_from_rollout,
    generate_powergrid_verl_splits,
    generate_joint_benchmark_scenarios,
    scenario_to_verl_record,
    write_verl_parquet,
)
from gridmind_mini.verl_reward import compute_score


class VerlPowerGridAdapterTest(unittest.TestCase):
    def test_scenario_to_verl_record_has_required_fields(self) -> None:
        scenario = generate_joint_benchmark_scenarios(1, seed=123, profile="emt")[0]

        record = scenario_to_verl_record(scenario, split="train", index=0)

        self.assertEqual(record["agent_name"], "tool_agent")
        self.assertEqual(record["ability"], "power_system_interconnection")
        self.assertEqual(record["prompt"][0]["role"], "system")
        self.assertEqual(record["prompt"][1]["content"], scenario.user_message)
        self.assertIn("ground_truth", record["reward_model"])
        self.assertEqual(
            json.loads(record["reward_model"]["ground_truth"])["scenario_id"],
            scenario.scenario_id,
        )
        create_kwargs = record["extra_info"]["tools_kwargs"][EXPECTED_TOOL_NAME]["create_kwargs"]
        self.assertEqual(create_kwargs["mode"], "frozen_oracle")
        self.assertIn("scenario_json", create_kwargs)
        self.assertIn("oracle_result_json", create_kwargs)

    def test_powergrid_prompt_requires_exact_json_tool_schema(self) -> None:
        prompt = build_powergrid_system_prompt()

        self.assertIn("<tool_call>", prompt)
        self.assertIn('"name":"run_integrated_assessment"', prompt)
        self.assertIn('"p_mw"', prompt)
        self.assertIn('"fault_start_s"', prompt)
        self.assertIn('"clearing_time_s"', prompt)
        self.assertIn("Do not use aliases", prompt)
        self.assertIn("project_bus", prompt)
        self.assertIn("fault_timing", prompt)

    def test_generated_emt_split_contains_only_emt_cases(self) -> None:
        train, val = generate_powergrid_verl_splits(
            train_count=8,
            val_count=4,
            train_seed=10,
            val_seed=11,
            profile="emt",
        )

        self.assertEqual(len(train), 8)
        self.assertEqual(len(val), 4)
        self.assertTrue(all("emt" in scenario.oracle_arguments for scenario in train + val))
        self.assertFalse({s.scenario_id for s in train}.intersection({s.scenario_id for s in val}))

    def test_build_verl_records_round_trips_scenario_json(self) -> None:
        scenarios = generate_joint_benchmark_scenarios(3, seed=456, profile="emt")

        records = build_verl_records(scenarios, split="train")

        self.assertEqual(len(records), 3)
        self.assertEqual(
            [json.loads(record["reward_model"]["ground_truth"])["scenario_id"] for record in records],
            [scenario.scenario_id for scenario in scenarios],
        )

    def test_write_verl_parquet_smoke(self) -> None:
        scenario = generate_joint_benchmark_scenarios(1, seed=222, profile="emt")[0]
        records = [scenario_to_verl_record(scenario, split="train", index=0)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.parquet"

            write_verl_parquet(records, path)

            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_export_powergrid_verl_dataset_writes_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = export_powergrid_verl_dataset(
                tmpdir,
                train_count=2,
                val_count=1,
                train_seed=33,
                val_seed=34,
                profile="emt",
            )

            self.assertTrue(Path(metadata["train_file"]).exists())
            self.assertTrue(Path(metadata["val_file"]).exists())
            self.assertTrue(Path(metadata["train_scenarios"]).exists())
            self.assertTrue(Path(metadata["val_scenarios"]).exists())
            self.assertEqual(metadata["tool"], EXPECTED_TOOL_NAME)

    def test_argument_audit_passes_exact_or_defaulted_arguments(self) -> None:
        scenario = generate_joint_benchmark_scenarios(1, seed=2026, profile="emt")[0]

        checks = audit_tool_arguments(scenario, scenario.oracle_arguments)

        self.assertTrue(all(check.passed for check in checks), [check.to_dict() for check in checks])

    def test_frozen_observation_returns_result_for_correct_arguments(self) -> None:
        scenario = generate_joint_benchmark_scenarios(1, seed=2027, profile="emt")[0]

        observation = build_frozen_tool_observation(scenario, scenario.oracle_arguments)

        self.assertTrue(observation["ok"])
        self.assertEqual(observation["tool"], EXPECTED_TOOL_NAME)
        self.assertTrue(observation["argument_audit"]["passed"])
        self.assertIn("summary", observation)
        self.assertIn("m1", observation)
        self.assertIn("m2", observation)
        self.assertIn("emt", observation)
        self.assertIn("scr", observation["emt"])

    def test_frozen_observation_reports_repairable_argument_error(self) -> None:
        scenario = generate_joint_benchmark_scenarios(1, seed=2028, profile="emt")[0]
        bad_args = dict(scenario.oracle_arguments)
        bad_args["connection"] = dict(bad_args["connection"])
        bad_args["connection"]["bus"] = 999

        observation = build_frozen_tool_observation(scenario, bad_args)

        self.assertFalse(observation["ok"])
        self.assertEqual(observation["error_type"], "argument_mismatch")
        failed_paths = [item["path"] for item in observation["argument_audit"]["failed"]]
        self.assertIn("connection.bus", failed_paths)

    def test_tool_core_uses_scenario_state(self) -> None:
        scenario = generate_joint_benchmark_scenarios(1, seed=2029, profile="emt")[0]
        core = FrozenIntegratedAssessmentToolCore()

        observation = core.execute(
            scenario.oracle_arguments,
            scenario_json=json.dumps(scenario.to_dict()),
            oracle_result_json=json.dumps(scenario.oracle_result_template),
        )

        self.assertTrue(observation["ok"])
        self.assertEqual(observation["scenario_id"], scenario.scenario_id)

    def test_extract_qwen_xml_tool_call(self) -> None:
        text = """
<tool_call>
<function=run_integrated_assessment>
<parameter=case_path>ieee14</parameter>
<parameter=connection>{"bus": 10, "p_mw": 20, "connection_type": "solar", "is_ibr": true}</parameter>
</function>
</tool_call>
"""

        calls = extract_tool_calls_from_rollout(text)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], EXPECTED_TOOL_NAME)
        self.assertEqual(calls[0]["arguments"]["connection"]["bus"], 10)
        self.assertTrue(calls[0]["arguments"]["connection"]["is_ibr"])

    def test_extract_final_answer_after_tool_observation(self) -> None:
        text = (
            "<tool_call>{\"name\":\"run_integrated_assessment\",\"arguments\":{}}</tool_call>"
            "{\"schema_version\":\"powergrid_frozen_oracle_v1\",\"tool\":\"run_integrated_assessment\"}"
            "Recommendation approve. M1 CIA pass, M2 transient pass, EMT SCR is 5.0."
        )

        final_text = extract_final_answer_text(text)

        self.assertTrue(final_text.startswith("Recommendation approve"))
        self.assertNotIn("schema_version", final_text)

    def test_reward_scores_good_rollout_high(self) -> None:
        scenario = generate_joint_benchmark_scenarios(1, seed=3030, profile="emt")[0]
        observation = build_frozen_tool_observation(scenario, scenario.oracle_arguments)
        rollout = _qwen_tool_call(scenario.oracle_arguments)
        rollout += json.dumps(observation)
        rollout += (
            "Recommendation {rec}. M1 CIA result is {m1}. M2 transient stability is {m2}. "
            "EMT/SCR status is {emt} with SCR {scr:.2f}. This is a static PQ / SCR proxy; "
            "it does not model detailed inverter controllers or switching EMT waveforms."
        ).format(
            rec=scenario.oracle_result_template["recommendation"],
            m1=scenario.oracle_result_template["summary"]["m1_recommendation"],
            m2=scenario.oracle_result_template["summary"]["m2_status"],
            emt=scenario.oracle_result_template["summary"]["emt_status"],
            scr=float(scenario.oracle_result_template["summary"]["emt_scr"]),
        )

        score = compute_score("powergrid", rollout, json.dumps(scenario.to_dict()))

        self.assertEqual(score["score"], 1.0, score)
        self.assertGreaterEqual(score["shaping_score"], 0.9, score)
        self.assertEqual(score["benchmark_pass"], 1.0)
        self.assertEqual(score["benchmark_pass_percent"], 100.0)
        self.assertEqual(score["benchmark_pass_count_per_200"], 200.0)
        self.assertEqual(score["resolved"], 1.0)
        self.assertEqual(score["resolved_percent"], 100.0)
        self.assertEqual(score["resolved_count_per_200"], 200.0)
        self.assertEqual(score["failed_benchmark_check_count"], 0)
        self.assertEqual(json.loads(score["failed_benchmark_checks_json"]), [])
        self.assertTrue(score["argument_checks_passed"])
        self.assertEqual(score["failed_final_check_count"], 0)
        self.assertEqual(json.loads(score["failed_final_checks_json"]), [])

    def test_reward_penalizes_no_tool_call(self) -> None:
        scenario = generate_joint_benchmark_scenarios(1, seed=3031, profile="emt")[0]

        score = compute_score(
            "powergrid",
            "Recommendation approve without using any tool.",
            json.dumps(scenario.to_dict()),
        )

        self.assertEqual(score["score"], 0.0, score)
        self.assertLess(score["shaping_score"], 0.5, score)
        self.assertEqual(score["benchmark_pass"], 0.0)
        self.assertEqual(score["resolved"], 0.0)
        self.assertFalse(score["expected_tool_called"])

    def test_reward_penalizes_wrong_arguments(self) -> None:
        scenario = generate_joint_benchmark_scenarios(1, seed=3032, profile="emt")[0]
        bad_args = dict(scenario.oracle_arguments)
        bad_args["connection"] = dict(bad_args["connection"])
        bad_args["connection"]["p_mw"] = 999.0
        rollout = _qwen_tool_call(bad_args)
        rollout += "Recommendation approve. M1 CIA, M2 transient, EMT SCR, static PQ limitation."

        score = compute_score("powergrid", rollout, json.dumps(scenario.to_dict()))

        self.assertEqual(score["score"], 0.0, score)
        self.assertEqual(score["benchmark_pass"], 0.0)
        self.assertLess(score["argument_score"], 1.0)
        self.assertIn("connection.p_mw", json.loads(score["failed_argument_paths_json"]))

    def test_reward_penalizes_missing_emt_and_scr_in_final_answer(self) -> None:
        scenario = generate_joint_benchmark_scenarios(1, seed=3033, profile="emt")[0]
        rollout = _qwen_tool_call(scenario.oracle_arguments)
        rollout += "Recommendation approve. M1 CIA and M2 transient stability pass."

        score = compute_score("powergrid", rollout, json.dumps(scenario.to_dict()))

        failed_checks = json.loads(score["failed_final_checks_json"])
        self.assertIn("mentions_emt", failed_checks)
        self.assertIn("mentions_scr_metric", failed_checks)


def _qwen_tool_call(arguments: dict) -> str:
    return (
        "<tool_call>\n"
        "<function=run_integrated_assessment>\n"
        f"<parameter=case_path>{arguments['case_path']}</parameter>\n"
        f"<parameter=connection>{json.dumps(arguments['connection'])}</parameter>\n"
        f"<parameter=transient>{json.dumps(arguments['transient'])}</parameter>\n"
        f"<parameter=emt>{json.dumps(arguments['emt'])}</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
    )


if __name__ == "__main__":
    unittest.main()

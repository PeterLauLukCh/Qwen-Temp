import tempfile
import unittest
from pathlib import Path

from gridmind_mini import (
    ENGINEER_CURRICULUM_LEVELS,
    REAL_M1M2_ENGINEER_GYM_SCHEMA_VERSION,
    RealM1M2EngineerEnv,
    filter_real_m1m2_engineer_episodes,
    generate_real_m1m2_engineer_episodes,
    load_real_m1m2_engineer_episodes,
    replay_real_m1m2_engineer_trajectory,
    write_real_m1m2_engineer_episodes,
)


def _fake_runner(name, args):
    if name == "list_remote_psse_m1m2_cases":
        return {
            "ok": True,
            "tool": name,
            "case_count": 2,
            "cases": [
                {"case_id": "pif6_2026_05_17", "allowed_scenarios": ["static", "no_disturbance_5s"]},
                {"case_id": "test_cases_v36", "allowed_scenarios": ["static", "no_disturbance_5s", "pq_target_step"]},
            ],
        }
    if name == "run_remote_psse_m1m2":
        case_id = args["case_id"]
        scenario_type = args["scenario_type"]
        return {
            "ok": True,
            "tool": name,
            "case_id": case_id,
            "scenario_type": scenario_type,
            "recommendation": "approve",
            "complete": True,
            "summary": {
                "m1_status": "pass",
                "m1_bus_count": 786 if case_id == "pif6_2026_05_17" else 11,
                "m2_status": "pass" if scenario_type == "no_disturbance_5s" else "skipped",
            },
        }
    if name == "inspect_real_case_summary":
        return {
            "ok": True,
            "tool": name,
            "case_id": args["case_id"],
            "case_size": {"bus_count": 786},
            "available_remote_scenarios": ["static", "no_disturbance_5s"],
            "poc_candidates": [{"bus": 2, "name": "POC2"}, {"bus": 2000, "name": "POC2_0"}],
        }
    if name == "inspect_real_poc_context":
        return {
            "ok": True,
            "tool": name,
            "case_id": args["case_id"],
            "selected_poc": {"bus": 2, "name": "POC2", "voltage_pu": 0.9061},
            "candidate_count": 2,
            "ambiguous": True,
            "candidate_buses": [{"bus": 2, "name": "POC2"}, {"bus": 2000, "name": "POC2_0"}],
            "candidate_branches": [{"from_bus": 2, "to_bus": 2001, "p_mw": 5.1, "q_mvar": -19.3}],
        }
    if name == "inspect_real_model_inventory":
        return {
            "ok": True,
            "tool": name,
            "case_id": args["case_id"],
            "counts": {"machines": 251, "dynamic_models": 502},
            "machines": [{"bus": 2000, "regulated_bus": 2}],
        }
    if name == "inspect_real_static_operating_point":
        return {
            "ok": True,
            "tool": name,
            "case_id": args["case_id"],
            "converged": True,
            "voltage_range_pu": {"min": 0.9, "max": 1.0425},
            "poc": {"p_mw": 5.09, "q_mvar": -19.33},
            "counts": {"buses": 786, "overloaded_branches": 0},
        }
    if name == "inspect_real_dynamic_channels":
        return {
            "ok": True,
            "tool": name,
            "case_id": args["case_id"],
            "scenario_type": "no_disturbance_5s",
            "row_count": 5004,
            "final_time_s": 5.0,
            "final_values": {"POC_P_2001_2": 5.13, "POC_Q_2001_2": -19.28, "POC2_V": 0.906},
        }
    return {"ok": False, "tool": name, "error_type": "unknown_tool"}


class RealM1M2EngineerGymTest(unittest.TestCase):
    def test_generation_is_deterministic_and_curriculum_balanced(self):
        first = generate_real_m1m2_engineer_episodes(100, seed=1)
        second = generate_real_m1m2_engineer_episodes(100, seed=1)

        self.assertEqual([item.to_dict() for item in first], [item.to_dict() for item in second])
        level_counts = {}
        difficulty_counts = {}
        for episode in first:
            level_counts[episode.curriculum_level] = level_counts.get(episode.curriculum_level, 0) + 1
            difficulty_counts[episode.difficulty] = difficulty_counts.get(episode.difficulty, 0) + 1
        self.assertEqual(set(level_counts), set(ENGINEER_CURRICULUM_LEVELS))
        self.assertEqual(level_counts["level1_scope_data_readiness"], 20)
        self.assertEqual(level_counts["level2_case_inspection"], 20)
        self.assertEqual(level_counts["level3_supported_m1m2"], 20)
        self.assertEqual(level_counts["level4_mixed_trgc"], 25)
        self.assertEqual(level_counts["level5_engineer_memo"], 15)
        self.assertEqual(difficulty_counts, {"easy": 34, "medium": 33, "hard": 33})

    def test_hidden_oracle_is_not_exposed_in_observation(self):
        episode = generate_real_m1m2_engineer_episodes(1, seed=2)[0]
        env = RealM1M2EngineerEnv(tool_runner=_fake_runner)
        observation = env.reset(episode).to_dict()

        self.assertEqual(observation["schema_version"], REAL_M1M2_ENGINEER_GYM_SCHEMA_VERSION)
        self.assertNotIn("hidden_oracle", str(observation))
        self.assertNotIn("reward_weights", str(observation))

    def test_inspection_before_run_scores_higher_than_direct_run_only(self):
        episode = next(
            item
            for item in generate_real_m1m2_engineer_episodes(30, seed=3)
            if item.curriculum_level == "level3_supported_m1m2"
        )
        job = episode.hidden_oracle["allowed_remote_jobs"][0]
        final = (
            "PSS/E M1 and M2 pass with recommendation approve. This is a bounded subset; "
            "fault droop SCR PSCAD and EMT remain unsupported."
        )
        direct = replay_real_m1m2_engineer_trajectory(
            episode,
            [
                {"type": "tool_call", "name": "run_remote_psse_m1m2", "arguments": dict(job)},
                {"type": "final_answer", "text": final},
            ],
            tool_runner=_fake_runner,
        )
        inspected = replay_real_m1m2_engineer_trajectory(
            episode,
            [
                {"type": "tool_call", "name": "inspect_real_case_summary", "arguments": {"case_id": job["case_id"]}},
                {"type": "tool_call", "name": "inspect_real_poc_context", "arguments": {"case_id": job["case_id"]}},
                {"type": "tool_call", "name": "run_remote_psse_m1m2", "arguments": dict(job)},
                {"type": "final_answer", "text": final},
            ],
            tool_runner=_fake_runner,
        )

        self.assertLess(direct.reward.total, inspected.reward.total)
        self.assertTrue(inspected.passed, inspected.to_dict())

    def test_proxy_baseline_for_unsupported_trgc_gets_hard_penalty(self):
        episode = next(
            item
            for item in generate_real_m1m2_engineer_episodes(10, seed=4)
            if item.curriculum_level == "level1_scope_data_readiness"
        )
        env = RealM1M2EngineerEnv(tool_runner=_fake_runner)
        env.reset(episode)
        _obs, _delta, terminated, _truncated, info = env.step(
            {
                "type": "tool_call",
                "name": "run_remote_psse_m1m2",
                "arguments": {"case_id": "pif6_2026_05_17", "scenario_type": "no_disturbance_5s"},
            }
        )

        self.assertTrue(terminated)
        self.assertIn("proxy_baseline", info["reward"]["hard_penalties"])

    def test_mixed_episode_requires_supported_evidence_and_refusal(self):
        episode = next(
            item
            for item in generate_real_m1m2_engineer_episodes(30, seed=5)
            if item.curriculum_level == "level4_mixed_trgc"
        )
        job = episode.hidden_oracle["allowed_remote_jobs"][0]
        unsupported_id = episode.hidden_oracle["requirement_ids"][1]
        result = replay_real_m1m2_engineer_trajectory(
            episode,
            [
                {"type": "tool_call", "name": "inspect_real_case_summary", "arguments": {"case_id": job["case_id"]}},
                {"type": "tool_call", "name": "inspect_real_poc_context", "arguments": {"case_id": job["case_id"]}},
                {"type": "tool_call", "name": "run_remote_psse_m1m2", "arguments": dict(job)},
                {
                    "type": "final_answer",
                    "text": (
                        f"PSS/E M1/M2 pass and approve the supported subset, but {unsupported_id} "
                        "is unsupported and cannot validate from this baseline. Fault droop SCR PSCAD EMT remain unsupported."
                    ),
                },
            ],
            tool_runner=_fake_runner,
        )

        self.assertTrue(result.passed, result.to_dict())

    def test_wrong_poc_claim_is_penalized_when_poc_evidence_available(self):
        episode = next(
            item
            for item in generate_real_m1m2_engineer_episodes(20, seed=6)
            if item.curriculum_level == "level2_case_inspection"
        )
        result = replay_real_m1m2_engineer_trajectory(
            episode,
            [
                {"type": "tool_call", "name": "inspect_real_case_summary", "arguments": {"case_id": "pif6_2026_05_17"}},
                {"type": "tool_call", "name": "inspect_real_poc_context", "arguments": {"case_id": "pif6_2026_05_17", "poc_label_or_bus": "POC2"}},
                {"type": "tool_call", "name": "inspect_real_model_inventory", "arguments": {"case_id": "pif6_2026_05_17"}},
                {
                    "type": "final_answer",
                    "text": "The POC bus 2000 is confirmed from POC2_0 as the POC candidate.",
                },
            ],
            tool_runner=_fake_runner,
        )

        self.assertIn("wrong_poc_claim", result.reward.hard_penalties)
        self.assertFalse(result.passed)

    def test_engineer_episode_write_load_filter_and_verl_export(self):
        episodes = generate_real_m1m2_engineer_episodes(6, seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "episodes.json"
            summary = write_real_m1m2_engineer_episodes(episodes, output)
            loaded = load_real_m1m2_engineer_episodes(output)

        self.assertEqual(summary["episode_count"], 6)
        self.assertEqual([item.episode_id for item in loaded], [item.episode_id for item in episodes])
        selected = filter_real_m1m2_engineer_episodes(loaded, curriculum_levels=["level3_supported_m1m2"])
        self.assertEqual(len(selected), 1)
        verl = selected[0].to_verl_sample()
        self.assertEqual(verl["ability"], "trgc_interconnection_engineer_workflow")
        self.assertIn("hidden_oracle", verl["reward_model"])


if __name__ == "__main__":
    unittest.main()

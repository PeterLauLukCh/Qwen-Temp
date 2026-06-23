import tempfile
import unittest
from pathlib import Path

from gridmind_mini import (
    ENGINEER_CHALLENGE_FAMILIES,
    ENGINEER_CURRICULUM_LEVELS,
    REAL_M1M2_ENGINEER_CHALLENGE_SCHEMA_VERSION,
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
    if name == "inspect_real_network_neighborhood":
        return {
            "ok": True,
            "tool": name,
            "case_id": args["case_id"],
            "bus": args["bus"],
            "visited_bus_count": 7,
            "buses": [{"bus": 2, "name": "POC2", "voltage_pu": 0.9061}, {"bus": 2001, "name": "DUMMY"}],
            "branches": [{"from_bus": 2, "to_bus": 2001, "p_mw": -5.0867, "q_mvar": 19.3352}],
            "top_loading_percent": 1.754,
        }
    if name == "inspect_real_model_inventory":
        return {
            "ok": True,
            "tool": name,
            "case_id": args["case_id"],
            "counts": {"machines": 251, "dynamic_models": 502},
            "dynamic_models": [
                {"bus": 2, "model_category": "PPC", "model_name": "SIPIF6"},
                {"bus": 800, "model_category": "STATCOM", "model_name": "NWSTAT01"},
                {"bus": 10010, "model_category": "inverter", "model_name": "SGCVTF0131101"},
            ],
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
            "frequency_extrema": {"FREQ_2": {"min": -1e-8, "max": 1e-10}},
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

    def test_challenge_generation_public_mix_is_deterministic(self):
        first = generate_real_m1m2_engineer_episodes(100, seed=11, profile="trgc_engineer_challenge")
        second = generate_real_m1m2_engineer_episodes(100, seed=11, profile="trgc_engineer_challenge")

        self.assertEqual([item.to_dict() for item in first], [item.to_dict() for item in second])
        self.assertTrue(all(item.schema_version == REAL_M1M2_ENGINEER_CHALLENGE_SCHEMA_VERSION for item in first))
        family_counts = {}
        difficulty_counts = {}
        for episode in first:
            family_counts[episode.family] = family_counts.get(episode.family, 0) + 1
            difficulty_counts[episode.difficulty] = difficulty_counts.get(episode.difficulty, 0) + 1
        self.assertEqual(set(family_counts), set(ENGINEER_CHALLENGE_FAMILIES))
        self.assertEqual(family_counts["wrong_poc_disambiguation"], 15)
        self.assertEqual(family_counts["numeric_static_interpretation"], 15)
        self.assertEqual(family_counts["dynamic_channel_interpretation"], 15)
        self.assertEqual(family_counts["mixed_trgc_proxy_refusal"], 20)
        self.assertEqual(family_counts["contradictory_submittal"], 15)
        self.assertEqual(family_counts["memo_capstone"], 20)
        self.assertEqual(difficulty_counts, {"easy": 20, "medium": 30, "hard": 50})

    def test_challenge_hidden_oracle_is_not_exposed_in_observation(self):
        episode = generate_real_m1m2_engineer_episodes(1, seed=12, profile="trgc_engineer_challenge")[0]
        env = RealM1M2EngineerEnv(tool_runner=_fake_runner)
        observation = env.reset(episode).to_dict()

        self.assertNotIn("hidden_oracle", str(observation))
        self.assertNotIn("required_numeric_facts", str(observation))
        self.assertIn("real_m1m2_engineer_challenge", str(observation))

    def test_challenge_keyword_only_static_memo_fails_numeric_accuracy(self):
        episode = next(
            item
            for item in generate_real_m1m2_engineer_episodes(30, seed=13, profile="trgc_engineer_challenge")
            if item.family == "numeric_static_interpretation"
        )
        result = replay_real_m1m2_engineer_trajectory(
            episode,
            [
                {"type": "tool_call", "name": "inspect_real_case_summary", "arguments": {"case_id": "pif6_2026_05_17"}},
                {"type": "tool_call", "name": "inspect_real_static_operating_point", "arguments": {"case_id": "pif6_2026_05_17"}},
                {
                    "type": "final_answer",
                    "text": (
                        "Evidence: inspected static M1. Numeric findings: voltage and P/Q are present. "
                        "TRGC mapping: steady-state. Limitations: bounded subset. Recommendation: approve only supported scope."
                    ),
                },
            ],
            tool_runner=_fake_runner,
        )

        self.assertLess(result.reward.numeric_accuracy, 0.5)
        self.assertFalse(result.passed)

    def test_challenge_wrong_poc_claim_gets_hard_penalty(self):
        episode = next(
            item
            for item in generate_real_m1m2_engineer_episodes(30, seed=14, profile="trgc_engineer_challenge")
            if item.family == "wrong_poc_disambiguation"
        )
        result = replay_real_m1m2_engineer_trajectory(
            episode,
            [
                {"type": "tool_call", "name": "inspect_real_case_summary", "arguments": {"case_id": "pif6_2026_05_17"}},
                {"type": "tool_call", "name": "inspect_real_poc_context", "arguments": {"case_id": "pif6_2026_05_17"}},
                {"type": "tool_call", "name": "inspect_real_network_neighborhood", "arguments": {"case_id": "pif6_2026_05_17", "bus": 2}},
                {
                    "type": "final_answer",
                    "text": "Evidence confirms POC bus 2000 and POC2_0 is the POC. Recommendation: approve.",
                },
            ],
            tool_runner=_fake_runner,
        )

        self.assertIn("wrong_poc_claim", result.reward.hard_penalties)
        self.assertFalse(result.passed)

    def test_challenge_proxy_remote_run_for_contradictory_submittal_penalized(self):
        episode = next(
            item
            for item in generate_real_m1m2_engineer_episodes(40, seed=15, profile="trgc_engineer_challenge")
            if item.family == "contradictory_submittal"
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

    def test_challenge_capstone_requires_numeric_static_dynamic_and_bounded_memo(self):
        episode = next(
            item
            for item in generate_real_m1m2_engineer_episodes(100, seed=16, profile="trgc_engineer_challenge")
            if item.family == "memo_capstone"
        )
        result = replay_real_m1m2_engineer_trajectory(
            episode,
            [
                {"type": "tool_call", "name": "inspect_real_case_summary", "arguments": {"case_id": "pif6_2026_05_17"}},
                {"type": "tool_call", "name": "inspect_real_poc_context", "arguments": {"case_id": "pif6_2026_05_17"}},
                {"type": "tool_call", "name": "inspect_real_network_neighborhood", "arguments": {"case_id": "pif6_2026_05_17", "bus": 2}},
                {"type": "tool_call", "name": "inspect_real_model_inventory", "arguments": {"case_id": "pif6_2026_05_17"}},
                {"type": "tool_call", "name": "inspect_real_static_operating_point", "arguments": {"case_id": "pif6_2026_05_17"}},
                {"type": "tool_call", "name": "inspect_real_dynamic_channels", "arguments": {"case_id": "pif6_2026_05_17"}},
                {
                    "type": "final_answer",
                    "text": (
                        "Evidence inspected: POC context, bus 2 POC2 neighborhood, model inventory, static M1 and dynamic M2. "
                        "Numeric findings: 786 buses, 251 machines, 502 dynamic models, 1 PPC, 10 STATCOM, 240 inverter models. "
                        "Static voltage range is 0.9000 to 1.0425 pu; static POC P/Q is 5.0867 MW / -19.3347 Mvar. "
                        "Dynamic no-disturbance evidence has 5004 rows, final time 5.00082 s, final POC P/Q 5.1317 MW / -19.2842 Mvar, and POC2 voltage 0.9061 pu. "
                        "TRGC mapping: this supports only the current M1/M2 static and no-disturbance baseline subset. "
                        "Limitations: FRT, fault, droop, SCR, PSCAD, EMT, power quality and field validation remain unsupported and cannot validate from this baseline. "
                        "Recommendation: bounded approval only for the supported evidence subset; do not approve unsupported TRGC items."
                    ),
                },
            ],
            tool_runner=_fake_runner,
        )

        self.assertTrue(result.passed, result.to_dict())
        self.assertEqual(result.reward.numeric_accuracy, 1.0)
        self.assertEqual(result.reward.memo_section_score, 1.0)


if __name__ == "__main__":
    unittest.main()

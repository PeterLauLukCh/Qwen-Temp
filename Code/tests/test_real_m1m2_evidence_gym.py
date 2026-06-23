import tempfile
import unittest
from pathlib import Path

from gridmind_mini import (
    EVIDENCE_FAMILIES,
    REAL_M1M2_EVIDENCE_GYM_SCHEMA_VERSION,
    RealM1M2EvidenceEnv,
    filter_real_m1m2_evidence_episodes,
    generate_real_m1m2_evidence_episodes,
    load_real_m1m2_evidence_episodes,
    replay_real_m1m2_evidence_trajectory,
    write_real_m1m2_evidence_episodes,
)


def _fake_runner(name, args):
    if name == "list_remote_psse_m1m2_cases":
        return {
            "ok": True,
            "tool": name,
            "backend": "fake",
            "case_count": 2,
            "cases": [],
        }
    if name == "run_remote_psse_m1m2":
        case_id = args["case_id"]
        scenario_type = args["scenario_type"]
        summary = {
            "m1_status": "pass",
            "m1_bus_count": 786 if case_id == "pif6_2026_05_17" else 11,
            "m2_status": "pass" if scenario_type == "no_disturbance_5s" else "skipped",
        }
        return {
            "ok": True,
            "tool": name,
            "backend": "fake",
            "case_id": case_id,
            "scenario_type": scenario_type,
            "recommendation": "approve",
            "complete": True,
            "summary": summary,
        }
    return {"ok": False, "tool": name, "error_type": "unknown_tool"}


class RealM1M2EvidenceGymTest(unittest.TestCase):
    def test_generation_is_deterministic_and_balanced(self):
        first = generate_real_m1m2_evidence_episodes(100, seed=1)
        second = generate_real_m1m2_evidence_episodes(100, seed=1)

        self.assertEqual([item.to_dict() for item in first], [item.to_dict() for item in second])
        families = {}
        difficulties = {}
        for episode in first:
            families[episode.family] = families.get(episode.family, 0) + 1
            difficulties[episode.difficulty] = difficulties.get(episode.difficulty, 0) + 1
        self.assertEqual(set(families), set(EVIDENCE_FAMILIES))
        self.assertEqual(sum(families.values()), 100)
        self.assertEqual(difficulties, {"easy": 34, "medium": 33, "hard": 33})

    def test_hidden_oracle_is_not_exposed_in_observation(self):
        episode = generate_real_m1m2_evidence_episodes(1, seed=2)[0]
        env = RealM1M2EvidenceEnv(tool_runner=_fake_runner)
        observation = env.reset(episode).to_dict()

        self.assertNotIn("hidden_oracle", str(observation))
        self.assertNotIn("reward_weights", str(observation))
        self.assertEqual(observation["schema_version"], REAL_M1M2_EVIDENCE_GYM_SCHEMA_VERSION)

    def test_supported_exact_psse_action_gets_positive_reward(self):
        episode = next(
            item
            for item in generate_real_m1m2_evidence_episodes(12, seed=3)
            if item.family == "supported_only"
        )
        job = episode.hidden_oracle["allowed_remote_jobs"][0]
        result = replay_real_m1m2_evidence_trajectory(
            episode,
            [
                {
                    "type": "tool_call",
                    "name": "run_remote_psse_m1m2",
                    "arguments": dict(job),
                },
                {
                    "type": "final_answer",
                    "text": (
                        "PSS/E grounded result: recommendation approve and M1/M2 pass. "
                        "Full TRGC faults, droop, SCR, PSCAD, EMT, and field validation remain outside this result."
                    ),
                },
            ],
            tool_runner=_fake_runner,
        )

        self.assertTrue(result.passed, result.to_dict())
        self.assertGreaterEqual(result.reward.total, 0.8)

    def test_proxy_baseline_gets_hard_penalty_and_terminates(self):
        episode = next(
            item
            for item in generate_real_m1m2_evidence_episodes(12, seed=4)
            if item.family == "proxy_trap"
        )
        env = RealM1M2EvidenceEnv(tool_runner=_fake_runner)
        env.reset(episode)
        _observation, _reward, terminated, _truncated, info = env.step(
            {
                "type": "tool_call",
                "name": "run_remote_psse_m1m2",
                "arguments": {"case_id": "pif6_2026_05_17", "scenario_type": "no_disturbance_5s"},
            }
        )

        self.assertTrue(terminated)
        self.assertIn("proxy_baseline", info["reward"]["hard_penalties"])

    def test_proxy_and_missing_text_refusals_pass_without_solver(self):
        episodes = generate_real_m1m2_evidence_episodes(12, seed=5)
        proxy = next(item for item in episodes if item.family == "proxy_trap")
        missing = next(item for item in episodes if item.family == "missing_data")

        proxy_result = replay_real_m1m2_evidence_trajectory(
            proxy,
            [
                {
                    "type": "final_answer",
                    "text": (
                        "The baseline is not sufficient and cannot validate this TRGC item. "
                        "The study is unsupported and outside current scope, so approval cannot be made."
                    ),
                }
            ],
            tool_runner=_fake_runner,
        )
        missing_result = replay_real_m1m2_evidence_trajectory(
            missing,
            [
                {
                    "type": "final_answer",
                    "text": (
                        "Required TRGC data is missing and not available. "
                        "There is insufficient evidence, so solver-backed approval cannot be made."
                    ),
                }
            ],
            tool_runner=_fake_runner,
        )

        self.assertTrue(proxy_result.passed, proxy_result.to_dict())
        self.assertTrue(missing_result.passed, missing_result.to_dict())

    def test_mixed_episode_requires_supported_evidence_and_refusal(self):
        episode = next(
            item
            for item in generate_real_m1m2_evidence_episodes(12, seed=6)
            if item.family == "mixed_supported_unsupported"
        )
        job = episode.hidden_oracle["allowed_remote_jobs"][0]
        no_tool = replay_real_m1m2_evidence_trajectory(
            episode,
            [
                {
                    "type": "final_answer",
                    "text": "The unsupported TRGC item cannot validate and is outside scope.",
                }
            ],
            tool_runner=_fake_runner,
        )
        with_tool = replay_real_m1m2_evidence_trajectory(
            episode,
            [
                {"type": "tool_call", "name": "run_remote_psse_m1m2", "arguments": dict(job)},
                {
                    "type": "final_answer",
                    "text": (
                        "PSS/E recommendation approve for the supported subset. "
                        "The secondary TRGC requirement is unsupported and cannot validate from this result."
                    ),
                },
            ],
            tool_runner=_fake_runner,
        )

        self.assertLess(no_tool.reward.total, with_tool.reward.total)
        self.assertTrue(with_tool.passed, with_tool.to_dict())

    def test_false_worker_capability_claim_is_penalized(self):
        episode = next(
            item
            for item in generate_real_m1m2_evidence_episodes(12, seed=7)
            if item.family == "unsupported_only"
        )
        result = replay_real_m1m2_evidence_trajectory(
            episode,
            [
                {
                    "type": "tool_call",
                    "name": "list_remote_psse_m1m2_cases",
                    "arguments": {},
                },
                {
                    "type": "final_answer",
                    "text": (
                        "TRGC study is unsupported and outside current scope, but the "
                        "PSS/E remote worker does not load SAV/DYR/DLL."
                    ),
                },
            ],
            tool_runner=_fake_runner,
        )

        self.assertIn("false_worker_capability", result.reward.hard_penalties)
        self.assertFalse(result.passed)

    def test_write_load_filter_and_verl_export(self):
        episodes = generate_real_m1m2_evidence_episodes(6, seed=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "episodes.json"
            summary = write_real_m1m2_evidence_episodes(episodes, output)
            loaded = load_real_m1m2_evidence_episodes(output)

        self.assertEqual(summary["episode_count"], 6)
        self.assertEqual([item.episode_id for item in loaded], [item.episode_id for item in episodes])
        supported = filter_real_m1m2_evidence_episodes(loaded, families=["supported_only"])
        self.assertEqual(len(supported), 1)
        verl = supported[0].to_verl_sample()
        self.assertEqual(verl["ability"], "trgc_interconnection_evidence")
        self.assertIn("hidden_oracle", verl["reward_model"])


if __name__ == "__main__":
    unittest.main()

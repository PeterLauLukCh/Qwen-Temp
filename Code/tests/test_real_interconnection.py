"""Tests for the frozen PSS/E real interconnection replay environment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Mapping

from gridmind_mini import (
    AgentTurnResult,
    GroundingValidation,
    ToolExecutionRecord,
    ToolRegistry,
    build_deterministic_report,
    build_scenario_key,
    build_tool_observation,
    default_real_interconnection_benchmark_scenarios,
    evaluate_real_interconnection_result,
    generate_real_interconnection_scenarios,
    has_grounding_credit,
    list_real_interconnection_actions,
    load_inventory,
    normalize_connection,
    normalize_disturbance,
    run_real_interconnection_assessment,
    run_real_interconnection_oracles,
    validate_tool_call_policy,
    write_real_interconnection_catalog,
)


class RealInterconnectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inventory_root = self.root / "processed_file"
        self.frozen_root = self.root / "frozen_interconnection"
        write_inventory_fixture(self.inventory_root)
        build = generate_real_interconnection_scenarios(
            case_id="pif6_2026_05_17",
            processed_dir=str(self.inventory_root),
        )
        write_real_interconnection_catalog(
            output=str(self.frozen_root / "scenario_catalog.jsonl"),
            build_result=build,
        )
        write_default_benchmark_results(self.frozen_root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_inventory_loader_reads_pif6_poc_buses(self) -> None:
        inventory = load_inventory(
            case_id="pif6_2026_05_17",
            processed_dir=str(self.inventory_root),
        )

        self.assertEqual(inventory["summary"]["poc_candidates"], 3)
        self.assertEqual(inventory["summary"]["branches"], 1)

    def test_list_actions_reports_catalog_and_result_availability(self) -> None:
        result = list_real_interconnection_actions(
            processed_dir=str(self.frozen_root),
            inventory_dir=str(self.inventory_root),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["case_id"], "pif6_2026_05_17")
        self.assertGreater(result["catalog_scenario_count"], 0)
        self.assertEqual(result["precomputed_result_count"], 4)
        buses = [bus["bus"] for bus in result["action_space"]["connection_buses"]]
        self.assertEqual(buses, [2, 800])

    def test_generator_writes_catalog_index_and_criteria(self) -> None:
        build = generate_real_interconnection_scenarios(
            case_id="pif6_2026_05_17",
            processed_dir=str(self.inventory_root),
        )
        out = self.root / "second" / "scenario_catalog.jsonl"
        summary = write_real_interconnection_catalog(output=str(out), build_result=build)

        self.assertTrue(out.exists())
        self.assertTrue((out.parent / "index.json").exists())
        self.assertTrue((out.parent / "criteria.json").exists())
        self.assertTrue((out.parent / "results").exists())
        self.assertEqual(summary["scenario_count"], len(build.scenarios))

    def test_exact_lookup_returns_frozen_result(self) -> None:
        result = run_real_interconnection_assessment(
            case_id="pif6_2026_05_17",
            connection={"bus": 2, "connection_type": "solar", "p_mw": 5.0},
            disturbance={
                "type": "bus_fault",
                "bus": 2,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.10,
                "simulation_time_s": 5.0,
            },
            processed_dir=str(self.frozen_root),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "run_real_interconnection_assessment")
        self.assertEqual(result["recommendation"], "approve")
        self.assertEqual(result["summary"]["m1_status"], "pass")
        self.assertEqual(result["summary"]["m2_status"], "pass")
        self.assertEqual(result["connection"]["connection_type"], "solar_static_pq")

    def test_missing_exact_scenario_fails_closed(self) -> None:
        result = run_real_interconnection_assessment(
            case_id="pif6_2026_05_17",
            connection={"bus": 2, "connection_type": "solar_static_pq", "p_mw": 3.0},
            disturbance={"type": "no_disturbance", "simulation_time_s": 5.0},
            processed_dir=str(self.frozen_root),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "precomputed_result_unavailable")
        self.assertFalse(result["complete"])

    def test_tool_registry_exposes_real_interconnection_tools(self) -> None:
        registry = ToolRegistry()
        names = [spec["function"]["name"] for spec in registry.openai_tool_specs()]

        self.assertIn("list_real_interconnection_actions", names)
        self.assertIn("run_real_interconnection_assessment", names)
        result = registry.call_tool(
            "run_real_interconnection_assessment",
            {
                "case_id": "pif6_2026_05_17",
                "connection": {"bus": 2, "connection_type": "load_static_pq", "p_mw": 2.0},
                "disturbance": {"type": "no_disturbance", "simulation_time_s": 5.0},
                "processed_dir": str(self.frozen_root),
            },
        )
        self.assertTrue(result["ok"])

    def test_tool_policy_routes_real_interconnection_to_frozen_lookup(self) -> None:
        message = (
            "For PIF6 PSS/E, add 5 MW solar at bus 2 and test a 100 ms bus fault."
        )

        baseline_decision = validate_tool_call_policy(
            tool_name="run_real_psse_assessment",
            user_message=message,
            arguments={},
        )
        lookup_decision = validate_tool_call_policy(
            tool_name="run_real_interconnection_assessment",
            user_message=message,
            arguments={},
        )

        self.assertFalse(baseline_decision.allowed)
        self.assertEqual(
            baseline_decision.recommended_tool,
            "run_real_interconnection_assessment",
        )
        self.assertTrue(lookup_decision.allowed)

    def test_observation_report_and_grounding(self) -> None:
        result = run_real_interconnection_assessment(
            case_id="pif6_2026_05_17",
            connection={"bus": 2, "connection_type": "load_static_pq", "p_mw": 2.0},
            disturbance={"type": "no_disturbance", "simulation_time_s": 5.0},
            processed_dir=str(self.frozen_root),
        )

        observation = build_tool_observation(result)
        report = build_deterministic_report([result])

        self.assertEqual(observation["summary"]["m1_status"], "pass")
        self.assertEqual(observation["summary"]["m2_status"], "pass")
        self.assertIn("Frozen PSS/E interconnection", report.summary_text)
        self.assertTrue(has_grounding_credit(["run_real_interconnection_assessment"]))

    def test_real_interconnection_oracle_only_passes(self) -> None:
        scenarios = default_real_interconnection_benchmark_scenarios(
            processed_dir=str(self.frozen_root)
        )
        outputs = run_real_interconnection_oracles(scenarios, ToolRegistry())

        self.assertEqual(len(outputs), 5)
        self.assertTrue(all(item["ok"] for item in outputs))

    def test_wrong_tool_trap_fails_baseline_real_psse_tool(self) -> None:
        scenario = [
            item
            for item in default_real_interconnection_benchmark_scenarios(
                processed_dir=str(self.frozen_root)
            )
            if item.scenario_id == "real_pif6_wrong_tool_trap"
        ][0]
        record = ToolExecutionRecord(
            call_id="call_1",
            name="run_real_psse_assessment",
            arguments={"case_id": "pif6_baseline"},
            source="native",
            ok=True,
            result={"ok": True, "tool": "run_real_psse_assessment"},
            observation={"tool": "run_real_psse_assessment"},
        )
        agent_result = AgentTurnResult(
            status="completed",
            output_text="PSS/E baseline result.",
            raw_output_text="PSS/E baseline result.",
            messages=[],
            tool_records=[record],
            invoked_tools=["run_real_psse_assessment"],
            grounding=GroundingValidation(
                tool_grounded=True,
                claims=[],
                ungrounded_claims=[],
                warning_appended=False,
                output_text="PSS/E baseline result.",
                warning=None,
            ),
        )

        checks = evaluate_real_interconnection_result(
            scenario,
            agent_result=agent_result,
            oracle_result={},
        )
        failed = [check.name for check in checks if not check.passed]
        self.assertIn("expected_tool", failed)
        self.assertIn("forbidden_tool:run_real_psse_assessment", failed)


def write_inventory_fixture(root: Path) -> None:
    case_dir = root / "inventories" / "pif6_2026_05_17"
    case_dir.mkdir(parents=True)
    _write_csv(
        case_dir / "poc_candidates.csv",
        [
            "candidate_type",
            "bus",
            "name",
            "voltage_pu",
            "from_bus",
            "to_bus",
            "circuit_id",
            "P_MW",
            "Q_Mvar",
            "reason",
        ],
        [
            ["bus", "2", "POC2", "0.9061", "", "", "", "", "", "name_or_known_poc_bus"],
            ["bus", "800", "TERMINAL", "0.9976", "", "", "", "", "", "name_or_known_poc_bus"],
            ["branch", "2", "POC2", "0.9061", "2", "2001", "1", "-5.0", "19.3", "branch_touching_poc_candidate_bus"],
        ],
    )
    _write_csv(case_dir / "branches.csv", ["from_bus", "to_bus", "circuit_id"], [["2", "2001", "1"]])
    _write_csv(case_dir / "dynamic_models.csv", ["bus", "id", "model_name"], [["2000", "1", "GENCLS"]])
    _write_csv(case_dir / "loads.csv", ["bus", "id", "status", "Pload", "Qload"], [])
    _write_csv(case_dir / "machines.csv", ["bus", "id", "status", "Pgen", "Qgen"], [["2000", "1", "1", "5", "0"]])
    _write_csv(case_dir / "shunts.csv", ["bus", "id", "status", "B_Mvar", "G_MW"], [])
    _write_csv(case_dir / "switched_shunts.csv", ["bus", "id", "status", "B_Mvar", "control_mode", "controlled_bus", "rmpct"], [])
    _write_csv(case_dir / "transformers_2w.csv", ["from_bus", "to_bus", "circuit_id"], [])
    _write_csv(case_dir / "transformers_3w.csv", ["bus1", "bus2", "bus3", "id"], [])


def write_default_benchmark_results(frozen_root: Path) -> None:
    for scenario in default_real_interconnection_benchmark_scenarios(processed_dir=str(frozen_root)):
        args = scenario.oracle_arguments
        connection = normalize_connection(args["connection"])
        disturbance = normalize_disturbance(args["disturbance"])
        key = build_scenario_key(
            case_id=str(args["case_id"]),
            connection=connection,
            disturbance=disturbance,
        )
        index = json.loads((frozen_root / "index.json").read_text(encoding="utf-8"))
        entry = index["scenarios"][key]
        result_path = frozen_root / entry["result_file"]
        result_path.parent.mkdir(exist_ok=True)
        _write_json(
            result_path,
            frozen_result_payload(
                scenario_id=entry["scenario_id"],
                connection=connection,
                disturbance=disturbance,
            ),
        )


def frozen_result_payload(
    *,
    scenario_id: str,
    connection: Mapping[str, Any],
    disturbance: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "case_id": "pif6_2026_05_17",
        "backend": "psse",
        "source_files": {"sav_sha256": "fake", "dyr_sha256": "fake"},
        "connection": dict(connection),
        "disturbance": dict(disturbance),
        "m1_powerflow": {
            "status": "pass",
            "converged": True,
            "bus_voltage_min_pu": 0.90,
            "bus_voltage_max_pu": 1.04,
            "overloaded_branch_count": 0,
            "poc_p_mw": 5.0,
            "poc_q_mvar": -19.0,
        },
        "m2_dynamic": {
            "status": "pass",
            "initialized": True,
            "simulation_converged": True,
            "event": disturbance["type"],
            "metrics": {
                "min_voltage_pu": 0.88,
                "max_voltage_pu": 1.03,
                "max_frequency_deviation_hz": 0.03,
            },
        },
        "scr_screen": {
            "status": "pass",
            "passed": True,
            "metrics": {"scr": 4.2},
        },
        "recommendation": "approve",
        "complete": True,
        "label_source": "test_fixture",
        "limitations": [
            "frozen_psse_precomputed_results_only_no_live_psse",
            "static_pq_interconnection_only_no_full_device_dynamics",
        ],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = [",".join(headers)]
    lines.extend(",".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

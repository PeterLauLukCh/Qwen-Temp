"""Tests for frozen real-data PSS/E integration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from gridmind_mini import (
    AgentConfig,
    AgentTurnResult,
    ChatCompletion,
    GridMindAgent,
    GroundingValidation,
    ToolCall,
    ToolExecutionRecord,
    ToolRegistry,
    build_deterministic_report,
    build_tool_observation,
    default_real_data_benchmark_scenarios,
    evaluate_real_data_result,
    has_grounding_credit,
    list_real_psse_cases,
    run_real_data_oracles,
    run_real_psse_assessment,
)


class RealDataIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        write_fixture(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_list_real_psse_cases_reports_three_available_cases(self) -> None:
        result = list_real_psse_cases(str(self.root))

        self.assertTrue(result["ok"])
        self.assertEqual(result["case_count"], 3)
        self.assertEqual(
            {
                case["case_id"]: case["available"]
                for case in result["cases"]
            },
            {
                "test_cases_v36_no_disturbance": True,
                "test_cases_v36_pq_target": True,
                "pif6_baseline": True,
            },
        )

    def test_run_pif6_maps_m1_and_m2_pass(self) -> None:
        result = run_real_psse_assessment(
            case_id="pif6_baseline",
            processed_dir=str(self.root),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["recommendation"], "approve")
        self.assertTrue(result["complete"])
        self.assertEqual(result["summary"]["m1_status"], "pass")
        self.assertEqual(result["summary"]["m2_status"], "pass")
        self.assertEqual(result["summary"]["m1_bus_count"], 786)
        self.assertEqual(result["stage_reports"][2]["stage"], "f4_emt")
        self.assertIn("psse_rms_dynamic_not_emt_waveform", result["limitations"])

    def test_pq_target_maps_dynamic_not_run_to_complete_static_result(self) -> None:
        result = run_real_psse_assessment(
            case_id="test_cases_v36_pq_target",
            processed_dir=str(self.root),
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["recommendation"], "approve")
        self.assertEqual(result["summary"]["m2_status"], "skipped")
        self.assertIn("dynamic_not_run", result["limitations"])

    def test_unavailable_directory_returns_structured_error(self) -> None:
        result = run_real_psse_assessment(
            case_id="pif6_baseline",
            processed_dir=str(self.root / "missing"),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "real_data_unavailable")
        self.assertFalse(result["complete"])

    def test_invalid_json_returns_validation_error(self) -> None:
        (self.root / "pif6_baseline_result.json").write_text("{bad json", encoding="utf-8")

        result = run_real_psse_assessment(
            case_id="pif6_baseline",
            processed_dir=str(self.root),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "real_data_validation_error")

    def test_invalid_result_schema_returns_structured_validation_error(self) -> None:
        _write_json(
            self.root / "pif6_baseline_result.json",
            {"case_id": "bad", "m2_dynamic": {}},
        )

        result = run_real_psse_assessment(
            case_id="pif6_baseline",
            processed_dir=str(self.root),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "real_data_validation_error")
        self.assertIn("m1_powerflow", result["message"])

    def test_channel_preview_is_bounded(self) -> None:
        result = run_real_psse_assessment(
            case_id="pif6_baseline",
            processed_dir=str(self.root),
            include_channel_preview=True,
            max_channel_samples=2,
        )

        preview = result["channel_preview"]
        self.assertTrue(preview["available"])
        self.assertEqual(preview["row_count"], 4)
        self.assertEqual(preview["returned_rows"], 2)
        self.assertEqual(len(preview["rows"]), 2)

    def test_tool_registry_exposes_and_runs_real_psse_tools(self) -> None:
        registry = ToolRegistry()
        tools = registry.list_tools(include_unimplemented=False)["tools"]
        names = {tool["name"] for tool in tools}

        self.assertIn("list_real_psse_cases", names)
        self.assertIn("run_real_psse_assessment", names)
        result = registry.call_tool(
            "run_real_psse_assessment",
            {
                "case_id": "pif6_baseline",
                "processed_dir": str(self.root),
            },
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "run_real_psse_assessment")

    def test_gridmind_agent_allows_real_psse_prompt_through_guardrails(self) -> None:
        llm = FakeRealDataLLM(str(self.root))
        agent = GridMindAgent(
            registry=ToolRegistry(),
            llm_client=llm,
            config=AgentConfig(max_tool_rounds=2),
        )

        result = agent.run_turn(
            "For the PIF6 real-data PSS/E interconnection study, summarize the "
            "processed SAV/DYR/DLL RMS dynamic result."
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(llm.calls, 2)
        self.assertEqual(result.invoked_tools, ["run_real_psse_assessment"])
        self.assertEqual(result.tool_records[0].name, "run_real_psse_assessment")
        self.assertTrue(result.tool_records[0].ok)
        self.assertIn("PSS/E", result.output_text)
        self.assertIn("RMS", result.output_text)

    def test_observation_report_and_grounding_include_real_data_facts(self) -> None:
        result = run_real_psse_assessment(
            case_id="pif6_baseline",
            processed_dir=str(self.root),
        )

        observation = build_tool_observation(result)
        report = build_deterministic_report([result])

        self.assertEqual(observation["summary"]["m1_bus_count"], 786)
        self.assertEqual(observation["summary"]["m2_status"], "pass")
        self.assertTrue(report.available)
        self.assertIn("PSS/E", report.summary_text)
        self.assertIn("RMS", report.summary_text)
        self.assertTrue(has_grounding_credit(["run_real_psse_assessment"]))

    def test_default_real_data_suite_has_five_scenarios(self) -> None:
        scenarios = default_real_data_benchmark_scenarios(processed_dir=str(self.root))

        self.assertEqual(len(scenarios), 5)
        self.assertEqual(len({scenario.scenario_id for scenario in scenarios}), 5)

    def test_real_data_oracle_only_passes(self) -> None:
        scenarios = default_real_data_benchmark_scenarios(processed_dir=str(self.root))
        outputs = run_real_data_oracles(scenarios, ToolRegistry())

        self.assertTrue(all(item["ok"] for item in outputs))

    def test_wrong_tool_trap_fails_powerflow_only_agent(self) -> None:
        scenario = [
            item
            for item in default_real_data_benchmark_scenarios(processed_dir=str(self.root))
            if item.scenario_id == "real_wrong_tool_trap"
        ][0]
        result = {"ok": True, "tool": "run_powerflow"}
        record = ToolExecutionRecord(
            call_id="call_1",
            name="run_powerflow",
            arguments={"case_path": "ieee14"},
            source="native",
            ok=True,
            result=result,
            observation={"tool": "run_powerflow"},
        )
        agent_result = _agent_result(records=[record], invoked_tools=["run_powerflow"])

        checks = evaluate_real_data_result(
            scenario,
            agent_result=agent_result,
            oracle_result={},
        )

        failed = [check.name for check in checks if not check.passed]
        self.assertIn("expected_tool", failed)
        self.assertIn("forbidden_tool:run_powerflow", failed)


def write_fixture(root: Path) -> None:
    preflight = {
        "python_version": "3.12.7",
        "psse_version": "PSS(R)E Xplore 36.2.0",
        "bundles": {
            "test_cases_v36": {
                "dyr_model_counts": {
                    "SIR5GZ": 1,
                    "REGCAU1": 4,
                    "REECAU1": 4,
                    "SIAUX1": 4,
                },
                "dll_load_check": {
                    "dlls": [{"loaded": True, "ierr": 0}],
                },
                "dynamic_error_summary": [
                    "Error: Playback model input (plb) file not found;  Model Ignored."
                ],
            },
            "pif6_2026_05_17": {
                "dyr_model_counts": {
                    "GENCLS": 1,
                    "SIPIF6": 1,
                    "SGCVTF0131101": 240,
                    "NWSTAT01": 10,
                    "SIAUX1": 250,
                },
                "dll_load_check": {
                    "dlls": [
                        {"loaded": True, "ierr": 0},
                        {"loaded": True, "ierr": 0},
                        {"loaded": True, "ierr": 0},
                    ],
                },
                "dynamic_error_summary": [],
            },
        },
    }
    _write_json(root / "preflight_summary.json", preflight)
    _write_json(
        root / "pif6_baseline_result.json",
        _result_payload(
            case_id="pif6_2026_05_17_baseline_no_disturbance",
            psse_version="PSS(R)E Xplore 36.2.0",
            bus_count=786,
            branch_count=790,
            machine_count=251,
            load_count=2,
            poc_p=5.086677848,
            poc_q=-19.33467761,
            m2_status="pass",
            initialized=True,
            simulation_converged=True,
            event="no_disturbance_5s_safe_baseline_only",
            final_p=5.131742,
            final_q=-19.28417,
        ),
    )
    _write_json(
        root / "test_cases_v36_no_disturbance_result.json",
        _result_payload(
            case_id="test_cases_v36_psse_ppc_test_bench_pmin_qmax_no_disturbance",
            psse_version="PSS(R)E Xplore 36.2.0",
            bus_count=11,
            branch_count=10,
            machine_count=5,
            load_count=0,
            poc_p=200.0081024,
            poc_q=330.0627136,
            m2_status="pass",
            initialized=True,
            simulation_converged=True,
            event="no_disturbance_5s",
            final_p=200.0088043,
            final_q=330.0626526,
        ),
    )
    _write_json(
        root / "test_cases_v36_pq_target_result.json",
        _result_payload(
            case_id="test_cases_v36_step_pmin_qmax_reproduction",
            psse_version="PSS(R)E Xplore 36.2.0",
            bus_count=11,
            branch_count=10,
            machine_count=5,
            load_count=0,
            poc_p=200.0281372,
            poc_q=330.0714722,
            m2_status="not_run",
            initialized=False,
            simulation_converged=False,
            event="static_pq_target_reproduction_from_step_pmin_qmax_py",
            final_p=200.0281372,
            final_q=330.0714722,
        ),
    )
    _write_csv(root / "pif6_no_disturbance_channels.csv", ["time", "POC2_V"], 4)
    _write_csv(root / "test_cases_v36_no_disturbance_channels.csv", ["time", "V_900"], 4)
    _write_csv(root / "pif6_bus_voltages.csv", ["bus", "voltage_pu"], 2)
    _write_csv(root / "test_cases_v36_bus_voltages.csv", ["bus", "voltage_pu"], 2)
    _write_csv(root / "pif6_overloaded_branches.csv", ["from_bus", "to_bus"], 0)
    _write_csv(root / "test_cases_v36_overloaded_branches.csv", ["from_bus", "to_bus"], 0)
    _write_csv(root / "pif6_poc_branch_candidates.csv", ["from_bus", "to_bus"], 1)
    _write_csv(root / "test_cases_v36_pq_target_iterations.csv", ["stage", "poc_p_mw"], 2)


def _result_payload(
    *,
    case_id: str,
    psse_version: str,
    bus_count: int,
    branch_count: int,
    machine_count: int,
    load_count: int,
    poc_p: float,
    poc_q: float,
    m2_status: str,
    initialized: bool,
    simulation_converged: bool,
    event: str,
    final_p: float,
    final_q: float,
) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "backend": "psse",
        "psse_version": psse_version,
        "files": {"sav": "case.sav", "dyr": "case.dyr", "dlls": ["model.dll"]},
        "m1_powerflow": {
            "status": "pass",
            "converged": True,
            "bus_voltage_min_pu": 0.9,
            "bus_voltage_max_pu": 1.05,
            "poc_p_mw": poc_p,
            "poc_q_mvar": poc_q,
            "bus_count": bus_count,
            "branch_count": branch_count,
            "machine_count": machine_count,
            "load_count": load_count,
            "overloaded_branch_count": 0,
        },
        "m2_dynamic": {
            "status": m2_status,
            "initialized": initialized,
            "simulation_converged": simulation_converged,
            "event": event,
            "metrics": {
                "min_voltage_pu": 0.9 if initialized else None,
                "max_voltage_pu": 1.05 if initialized else None,
                "final_poc_p_mw": final_p,
                "final_poc_q_mvar": final_q,
                "max_oscillation": 0.01 if initialized else None,
            },
            "error_summary": [],
        },
        "limitations": [
            "psse_rms_dynamic_not_emt_waveform",
            "labels_not_yet_expert_validated",
        ],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, headers: list[str], rows: int) -> None:
    lines = [",".join(headers)]
    for index in range(rows):
        lines.append(",".join(str(index) for _ in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _agent_result(
    *,
    records: Optional[list[ToolExecutionRecord]] = None,
    invoked_tools: Optional[list[str]] = None,
    output_text: str = "PSS/E real-data RMS result with POC metrics and EMT limitation.",
) -> AgentTurnResult:
    return AgentTurnResult(
        status="completed",
        output_text=output_text,
        raw_output_text=output_text,
        messages=[],
        tool_records=records or [],
        invoked_tools=invoked_tools or [],
        grounding=GroundingValidation(
            tool_grounded=bool(invoked_tools),
            claims=[],
            ungrounded_claims=[],
            warning_appended=False,
            output_text=output_text,
            warning=None,
        ),
    )


class FakeRealDataLLM:
    def __init__(self, processed_dir: str) -> None:
        self.processed_dir = processed_dir
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice="auto", parallel_tool_calls=None, extra_body=None):
        del messages, tool_choice, parallel_tool_calls, extra_body
        self.calls += 1
        if self.calls == 1:
            names = {
                spec["function"]["name"]
                for spec in tools or []
                if isinstance(spec, dict) and isinstance(spec.get("function"), dict)
            }
            if "run_real_psse_assessment" not in names:
                raise AssertionError("run_real_psse_assessment missing from tool specs")
            return ChatCompletion(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="run_real_psse_assessment",
                        arguments={
                            "case_id": "pif6_baseline",
                            "processed_dir": self.processed_dir,
                        },
                        source="fake",
                    )
                ],
                reasoning_content="",
                finish_reason="tool_calls",
                raw={},
            )
        return ChatCompletion(
            content=(
                "Recommendation approve. The PSS/E RMS dynamic result passed, "
                "with grounded POC P/Q metrics and the EMT waveform limitation."
            ),
            tool_calls=[],
            reasoning_content="",
            finish_reason="stop",
            raw={},
        )


if __name__ == "__main__":
    unittest.main()

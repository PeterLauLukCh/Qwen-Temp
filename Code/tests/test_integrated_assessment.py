"""Tests for the integrated M1+M2 assessment module."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import (  # noqa: E402
    BackendDefinition,
    CaseInfo,
    ConnectionRequest,
    EMTAssessmentRequest,
    GridSolver,
    IntegratedAssessmentRunner,
    PublicDynamicCaseFiles,
    ToolRegistry,
    TransientAssessmentRequest,
)


class IntegratedSolver(GridSolver):
    def __init__(self) -> None:
        self.loaded_case = ""
        self.connection: Optional[ConnectionRequest] = None
        self.converged = False
        self.short_circuit_mva_by_bus = {1: 50.0}

    def available_cases(self) -> List[str]:
        return ["ieee2"]

    def load_case(self, case_name: str) -> CaseInfo:
        self.loaded_case = case_name
        return self.case_info()

    def run_powerflow(self) -> Dict[str, Any]:
        self.converged = True
        return {
            "backend": "fake",
            "converged": True,
            "case": self.case_info().to_dict(),
            "bus_summary": {"min_vm_pu": 1.0, "max_vm_pu": 1.0},
            "branch_summary": {"max_line_loading_percent": 50.0},
        }

    def bus_results(self) -> List[Dict[str, Any]]:
        vm_pu = 0.93 if self.connection is not None and self.connection.p_mw >= 100 else 1.0
        return [{"bus_index": 1, "name": "Bus 1", "vm_pu": vm_pu, "va_degree": 0.0}]

    def branch_results(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "lines": [
                {
                    "line_index": 1,
                    "name": "Line 1",
                    "from_bus": 1,
                    "to_bus": 2,
                    "loading_percent": 50.0,
                    "in_service": True,
                }
            ],
            "transformers": [],
        }

    def available_contingencies(self) -> List[Dict[str, Any]]:
        return []

    def apply_contingency(self, element_type: str, element_index: int) -> Dict[str, Any]:
        return {"element_type": element_type, "element_index": element_index}

    def add_connection(
        self,
        bus: int,
        p_mw: float,
        connection_type: str,
        is_ibr: bool,
        q_mvar: float = 0.0,
        name: Optional[str] = None,
        vm_pu: float = 1.0,
    ) -> Dict[str, Any]:
        self.connection = ConnectionRequest(
            bus=bus,
            p_mw=p_mw,
            connection_type=connection_type,
            is_ibr=is_ibr,
            q_mvar=q_mvar,
            name=name,
            vm_pu=vm_pu,
        )
        return {
            "element_table": "load" if connection_type == "load" else "sgen",
            "element_index": 0,
            "bus": bus,
            "resolved_bus": {"bus_index": bus, "matched_on": "fake"},
            "p_mw": p_mw,
            "connection_type": connection_type,
            "is_ibr": is_ibr,
        }

    def network_data(self, max_rows: int = 50) -> Dict[str, Any]:
        return {"case": self.case_info().to_dict(), "tables": {}}

    def case_info(self) -> CaseInfo:
        return CaseInfo(
            case_name=self.loaded_case or "ieee2",
            buses=2,
            lines=1,
            transformers=0,
            loads=1,
            generators=1,
            static_generators=0,
            external_grids=1,
        )


class FakeDynamicRunner:
    def __init__(self, result: Dict[str, Any]) -> None:
        self.result = result
        self.calls: List[Dict[str, Any]] = []

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(dict(kwargs))
        return dict(self.result)


def connection(p_mw: float = 5.0) -> ConnectionRequest:
    return ConnectionRequest(
        bus=1,
        p_mw=p_mw,
        connection_type="load",
        is_ibr=False,
    )


def solar_connection(p_mw: float = 5.0) -> ConnectionRequest:
    return ConnectionRequest(
        bus=1,
        p_mw=p_mw,
        connection_type="solar",
        is_ibr=True,
    )


def runner(dynamic_result: Optional[Dict[str, Any]] = None) -> IntegratedAssessmentRunner:
    return IntegratedAssessmentRunner(
        solver_factory=IntegratedSolver,
        dynamic_runner=FakeDynamicRunner(
            dynamic_result
            or {
                "ok": True,
                "tool": "run_transient_stability",
                "dynamic_interconnection_modeling": True,
                "connection_model": {
                    "andes_model": "PQ",
                    "modeling_assumption": "constant_power_load",
                },
                "connection_application": {
                    "requested": True,
                    "applied": True,
                    "mode": "static_load",
                },
                "limitations": ["m2_connection_model_is_static_pq"],
                "stability": {"status": "pass"},
                "metrics": {
                    "tds_converged": True,
                    "reason_codes": ["transient_metrics_within_thresholds"],
                },
            }
        ),
    )


class IntegratedAssessmentTest(unittest.TestCase):
    def test_m1_only_integrated_assessment_approves_but_marks_m2_skipped(self) -> None:
        result = runner().run(case_path="ieee2", connection=connection())

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "run_integrated_assessment")
        self.assertEqual(result["recommendation"], "approve")
        self.assertTrue(result["complete"])
        self.assertEqual(result["stage_reports"][0]["stage"], "m1_steady_state_cia")
        self.assertEqual(result["stage_reports"][1]["status"], "skipped")
        self.assertFalse(result["linkage"]["m2_models_new_connection"])

    def test_integrated_assessment_can_run_required_emt_screen(self) -> None:
        result = runner().run(
            case_path="ieee2",
            connection=solar_connection(10.0),
            emt=EMTAssessmentRequest(enabled=True, required_for_approval=True),
        )

        self.assertEqual(result["recommendation"], "approve")
        self.assertTrue(result["complete"])
        self.assertEqual(result["stage_reports"][2]["stage"], "f4_emt_screening")
        self.assertEqual(result["stage_reports"][2]["status"], "pass")
        self.assertTrue(result["linkage"]["emt_models_new_connection"])
        self.assertAlmostEqual(result["summary"]["emt_scr"], 5.0)

    def test_required_emt_failure_rejects_integrated_assessment(self) -> None:
        result = runner().run(
            case_path="ieee2",
            connection=solar_connection(20.0),
            emt=EMTAssessmentRequest(enabled=True, required_for_approval=True),
        )

        self.assertEqual(result["recommendation"], "reject")
        self.assertTrue(result["complete"])
        self.assertEqual(result["summary"]["emt_status"], "fail")
        self.assertIn("emt_failed", result["reason_codes"])

    def test_required_m2_pass_preserves_approval(self) -> None:
        dynamic = runner()
        transient = TransientAssessmentRequest.from_mapping(
            {
                "enabled": True,
                "required_for_approval": True,
                "case_path": "kundur",
                "disturbance": {
                    "type": "bus_fault",
                    "bus": 5,
                    "fault_start_s": 1.0,
                    "clearing_time_s": 1.1,
                },
                "simulation_time_s": 5.0,
            }
        )

        result = dynamic.run(
            case_path="ieee2",
            connection=connection(),
            transient=transient,
        )

        self.assertEqual(result["recommendation"], "approve")
        self.assertTrue(result["complete"])
        self.assertEqual(result["stage_reports"][1]["status"], "pass")
        self.assertEqual(result["summary"]["m2_stability_status"], "pass")
        self.assertTrue(result["linkage"]["m2_models_new_connection"])
        self.assertEqual(
            result["linkage"]["m2_linkage"],
            "dynamic_case_with_static_interconnection_model",
        )
        self.assertEqual(
            dynamic.dynamic_runner.calls[0]["connection"].to_dict(),
            connection().to_dict(),
        )

    def test_required_m2_failure_rejects(self) -> None:
        dynamic = runner(
            {
                "ok": True,
                "tool": "run_transient_stability",
                "stability": {"status": "fail"},
                "metrics": {"reason_codes": ["angle_spread_exceeds_threshold"]},
            }
        )
        transient = TransientAssessmentRequest.from_mapping(
            {
                "enabled": True,
                "required_for_approval": True,
                "disturbance": {
                    "type": "bus_fault",
                    "bus": 5,
                    "fault_start_s": 1.0,
                    "clearing_time_s": 1.1,
                },
            }
        )

        result = dynamic.run(case_path="ieee2", connection=connection(), transient=transient)

        self.assertEqual(result["recommendation"], "reject")
        self.assertTrue(result["complete"])
        self.assertEqual(result["stage_reports"][1]["status"], "fail")

    def test_required_m2_missing_disturbance_is_incomplete_borderline(self) -> None:
        transient = TransientAssessmentRequest.from_mapping(
            {"enabled": True, "required_for_approval": True}
        )

        result = runner().run(case_path="ieee2", connection=connection(), transient=transient)

        self.assertEqual(result["recommendation"], "borderline")
        self.assertFalse(result["complete"])
        self.assertIn("required_m2_unavailable", result["reason_codes"])
        self.assertEqual(result["stage_reports"][1]["status"], "not_ready")

    def test_m1_rejection_dominates_m2_pass(self) -> None:
        transient = TransientAssessmentRequest.from_mapping(
            {
                "enabled": True,
                "required_for_approval": True,
                "disturbance": {
                    "type": "bus_fault",
                    "bus": 5,
                    "fault_start_s": 1.0,
                    "clearing_time_s": 1.1,
                },
            }
        )

        result = runner().run(case_path="ieee2", connection=connection(100.0), transient=transient)

        self.assertEqual(result["recommendation"], "reject")
        self.assertIn("m1_not_approved", result["reason_codes"])

    def test_required_m2_connection_application_error_is_incomplete(self) -> None:
        dynamic = runner(
            {
                "ok": False,
                "tool": "run_transient_stability",
                "dynamic_interconnection_modeling": False,
                "connection_application": {
                    "requested": True,
                    "applied": False,
                    "error_type": "connection_application_error",
                },
                "stability": {"status": "error"},
                "metrics": {"reason_codes": ["connection_application_error"]},
                "reason_codes": ["connection_application_error"],
            }
        )
        transient = TransientAssessmentRequest.from_mapping(
            {
                "enabled": True,
                "required_for_approval": True,
                "disturbance": {
                    "type": "bus_fault",
                    "bus": 5,
                    "fault_start_s": 1.0,
                    "clearing_time_s": 1.1,
                },
            }
        )

        result = dynamic.run(case_path="ieee2", connection=connection(), transient=transient)

        self.assertEqual(result["recommendation"], "borderline")
        self.assertFalse(result["complete"])
        self.assertFalse(result["linkage"]["m2_models_new_connection"])
        self.assertIn("required_m2_unavailable", result["reason_codes"])

    def test_registry_exposes_integrated_assessment_tool(self) -> None:
        registry = ToolRegistry(
            backends={
                "fake": BackendDefinition(
                    name="fake",
                    solver_factory=IntegratedSolver,
                    supported_cases=lambda: ["ieee2"],
                    is_available=lambda: True,
                    description="fake",
                )
            },
            active_backend="fake",
        )

        result = registry.call_tool(
            "run_integrated_assessment",
            {
                "case_path": "ieee2",
                "connection": {
                    "bus": 1,
                    "p_mw": 5.0,
                    "connection_type": "load",
                    "is_ibr": False,
                },
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["recommendation"], "approve")
        self.assertEqual(result["backend"]["m1"], "fake")

    def test_ieee118_transient_uses_strict_match_solver_and_alignment_metadata(self) -> None:
        dynamic_result = {
            "ok": True,
            "tool": "run_transient_stability",
            "case_info": {
                "case_id": "ieee118_public_dynamic",
                "case_source": "powerfulcases_ieee118_public_dynamic",
                "resolved_case_file": "/tmp/ieee118.raw",
                "resolved_addfile": "/tmp/ieee118.dyr",
                "data_label": "public_benchmark_data_not_customer_validated",
            },
            "dynamic_interconnection_modeling": True,
            "connection_model": {"andes_model": "PQ"},
            "connection_application": {"requested": True, "applied": True},
            "limitations": [
                "m2_connection_model_is_static_pq",
                "ieee118_uses_public_benchmark_dynamic_data_not_customer_validated",
            ],
            "stability": {"status": "pass"},
            "metrics": {
                "tds_converged": True,
                "reason_codes": ["transient_metrics_within_thresholds"],
            },
        }
        dynamic = FakeDynamicRunner(dynamic_result)
        strict_calls: List[IntegratedSolver] = []

        def strict_factory() -> IntegratedSolver:
            solver = IntegratedSolver()
            strict_calls.append(solver)
            return solver

        files = PublicDynamicCaseFiles(
            case_id="ieee118_public_dynamic",
            source="powerfulcases_ieee118_public_dynamic",
            raw_path="/tmp/ieee118.raw",
            dyr_path="/tmp/ieee118.dyr",
            dyr_variant="genrou",
            raw_sha256="rawhash",
            dyr_sha256="dyrhash",
        )
        transient = TransientAssessmentRequest.from_mapping(
            {
                "enabled": True,
                "required_for_approval": True,
                "case_path": "ieee118_dynamic",
                "disturbance": {
                    "type": "bus_fault",
                    "bus": 2,
                    "fault_start_s": 1.0,
                    "clearing_time_s": 1.1,
                },
            }
        )
        with patch(
            "gridmind_mini.integrated_assessment.resolve_ieee118_public_dynamic_files",
            return_value=files,
        ):
            result = IntegratedAssessmentRunner(
                solver_factory=IntegratedSolver,
                dynamic_runner=dynamic,
                strict_match_solver_factory=strict_factory,
            ).run(
                case_path="ieee118",
                connection=connection(),
                transient=transient,
            )

        self.assertTrue(result["case_alignment"]["strict_match"])
        self.assertEqual(result["case_alignment"]["source"], "powerfulcases_ieee118_public_dynamic")
        self.assertEqual(result["m1_case_source"]["raw_path"], "/tmp/ieee118.raw")
        self.assertEqual(result["m2_case_source"]["resolved_addfile"], "/tmp/ieee118.dyr")
        self.assertEqual([solver.loaded_case for solver in strict_calls], ["ieee118", "ieee118"])
        self.assertEqual(dynamic.calls[0]["case_path"], "ieee118_dynamic")
        self.assertIs(dynamic.calls[0]["public_case_files"], files)
        self.assertIn(
            "ieee118_uses_public_benchmark_dynamic_data_not_customer_validated",
            result["limitations"],
        )

    def test_ieee118_missing_public_data_returns_structured_incomplete_result(self) -> None:
        transient = TransientAssessmentRequest.from_mapping(
            {
                "enabled": True,
                "required_for_approval": True,
                "case_path": "ieee118_dynamic",
                "disturbance": {
                    "type": "bus_fault",
                    "bus": 2,
                    "fault_start_s": 1.0,
                    "clearing_time_s": 1.1,
                },
            }
        )
        from gridmind_mini import PublicCaseDataError

        with patch(
            "gridmind_mini.integrated_assessment.resolve_ieee118_public_dynamic_files",
            side_effect=PublicCaseDataError("case_data_unavailable", "missing public data"),
        ):
            result = IntegratedAssessmentRunner(
                solver_factory=IntegratedSolver,
                dynamic_runner=FakeDynamicRunner({}),
                strict_match_solver_factory=IntegratedSolver,
            ).run(
                case_path="ieee118",
                connection=connection(),
                transient=transient,
            )

        self.assertTrue(result["ok"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["recommendation"], "borderline")
        self.assertEqual(result["case_alignment"]["error_type"], "case_data_unavailable")
        self.assertEqual(result["summary"]["m2_status"], "unavailable")

    def test_ieee118_failed_local_override_reports_local_source(self) -> None:
        transient = TransientAssessmentRequest.from_mapping(
            {
                "enabled": True,
                "required_for_approval": True,
                "case_path": "ieee118_dynamic",
                "disturbance": {
                    "type": "bus_fault",
                    "bus": 2,
                    "fault_start_s": 1.0,
                    "clearing_time_s": 1.1,
                },
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "ieee118.raw"
            raw.write_text("raw", encoding="utf-8")
            with patch.dict(
                "os.environ",
                {"GRIDMIND_IEEE118_RAW_PATH": str(raw)},
                clear=True,
            ):
                result = IntegratedAssessmentRunner(
                    solver_factory=IntegratedSolver,
                    dynamic_runner=FakeDynamicRunner({}),
                    strict_match_solver_factory=IntegratedSolver,
                ).run(
                    case_path="ieee118",
                    connection=connection(),
                    transient=transient,
                )

        self.assertTrue(result["ok"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["case_alignment"]["source"], "local_ieee118_raw_dyr_override")
        self.assertEqual(result["m1_case_source"]["source"], "local_ieee118_raw_dyr_override")
        self.assertEqual(result["m2_case_source"]["source"], "local_ieee118_raw_dyr_override")
        self.assertEqual(result["m2_result"]["case_info"]["case_source"], "local_ieee118_raw_dyr_override")


if __name__ == "__main__":
    unittest.main()

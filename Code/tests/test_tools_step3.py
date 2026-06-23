"""Tests for the Step 3 tool registry."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Type


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import (
    BackendDefinition,
    CaseInfo,
    GridSolver,
    PowerFlowError,
    ToolRegistry,
    ToolRegistryError,
)


class FakeSolver(GridSolver):
    def __init__(self) -> None:
        self.loaded_case = ""
        self.converged = False
        self.run_count = 0
        self.outage: Optional[Dict[str, Any]] = None
        self.short_circuit_mva_by_bus = {1: 50.0, 2: 20.0}

    def available_cases(self) -> List[str]:
        return ["ieee2"]

    def load_case(self, case_name: str) -> CaseInfo:
        self.loaded_case = case_name
        return self.case_info()

    def run_powerflow(self) -> Dict[str, Any]:
        self.run_count += 1
        self.converged = True
        return {
            "backend": "fake",
            "converged": True,
            "case": self.case_info().to_dict(),
            "bus_summary": {"min_vm_pu": 1.0, "max_vm_pu": 1.0},
            "branch_summary": {"max_line_loading_percent": 50.0},
        }

    def bus_results(self) -> List[Dict[str, Any]]:
        if not self.converged:
            raise RuntimeError("power flow has not run")
        return [
            {"bus_index": 1, "name": "Bus 1", "vm_pu": 1.0, "va_degree": 0.0},
            {"bus_index": 2, "name": "Bus 2", "vm_pu": 1.0, "va_degree": 1.0},
        ]

    def branch_results(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self.converged:
            raise RuntimeError("power flow has not run")
        loading_percent = 120.0 if self.outage is not None else 50.0
        return {
            "lines": [
                {
                    "line_index": 1,
                    "name": "Line 1",
                    "from_bus": 1,
                    "to_bus": 2,
                    "loading_percent": loading_percent,
                    "in_service": True,
                }
            ],
            "transformers": [],
        }

    def available_contingencies(self) -> List[Dict[str, Any]]:
        return [
            {
                "element_type": "line",
                "element_index": 1,
                "element_name": "Line 1",
                "from_bus": 1,
                "to_bus": 2,
            }
        ]

    def apply_contingency(self, element_type: str, element_index: int) -> Dict[str, Any]:
        self.outage = {
            "element_type": element_type,
            "element_index": element_index,
            "element_name": "Line 1",
            "from_bus": 1,
            "to_bus": 2,
        }
        self.converged = False
        return dict(self.outage)

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
        return {
            "element_table": "load" if connection_type == "load" else "sgen",
            "element_index": 0,
            "name": name or "fake_connection",
            "connection_type": connection_type,
            "is_ibr": is_ibr,
            "bus": bus,
            "resolved_bus": {"bus_index": bus, "matched_on": "fake"},
            "p_mw": p_mw,
            "q_mvar": q_mvar,
        }

    def network_data(self, max_rows: int = 50) -> Dict[str, Any]:
        return {
            "case": self.case_info().to_dict(),
            "max_rows_per_table": max_rows,
            "tables": {
                "buses": {
                    "rows": [{"bus_index": 1, "name": "Bus 1"}],
                    "total_rows": 2,
                    "truncated_rows": 1,
                }
            },
        }

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


class FailingPowerFlowSolver(FakeSolver):
    def run_powerflow(self) -> Dict[str, Any]:
        raise PowerFlowError("fake power flow failed")


def fake_registry(
    created: Optional[List[FakeSolver]] = None,
    solver_cls: Type[FakeSolver] = FakeSolver,
) -> ToolRegistry:
    created = created if created is not None else []

    def factory() -> FakeSolver:
        solver = solver_cls()
        created.append(solver)
        return solver

    return ToolRegistry(
        backends={
            "fake": BackendDefinition(
                name="fake",
                solver_factory=factory,
                supported_cases=lambda: ["ieee2"],
                is_available=lambda: True,
                description="Fake backend for tests.",
            )
        },
        active_backend="fake",
    )


class ToolRegistryTest(unittest.TestCase):
    def test_call_tool_redirects_library_stdout_to_stderr(self) -> None:
        class NoisyRegistry(ToolRegistry):
            def _tool_list_dynamic_cases(self, _args: Dict[str, Any]) -> Dict[str, Any]:
                print("library stdout noise")
                return {"ok": True, "tool": "list_dynamic_cases"}

        registry = NoisyRegistry()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = registry.call_tool("list_dynamic_cases")

        self.assertEqual(result["tool"], "list_dynamic_cases")
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("library stdout noise", stderr.getvalue())

    def test_openai_specs_exclude_unimplemented_tools_by_default(self) -> None:
        registry = fake_registry()

        names = [spec["function"]["name"] for spec in registry.openai_tool_specs()]

        self.assertIn("run_powerflow", names)
        self.assertIn("inspect_violations", names)
        self.assertIn("query_network_data", names)
        self.assertIn("run_cia", names)
        self.assertIn("run_integrated_assessment", names)
        self.assertIn("find_max_capacity", names)
        self.assertIn("list_dynamic_cases", names)
        self.assertIn("run_transient_stability", names)
        self.assertIn("run_emt_screening", names)
        self.assertIn("list_real_psse_cases", names)
        self.assertIn("run_real_psse_assessment", names)
        self.assertIn("list_real_interconnection_actions", names)
        self.assertIn("run_real_interconnection_assessment", names)
        self.assertIn("inspect_real_case_summary", names)
        self.assertIn("inspect_real_poc_context", names)
        self.assertIn("inspect_real_network_neighborhood", names)
        self.assertIn("inspect_real_model_inventory", names)
        self.assertIn("inspect_real_static_operating_point", names)
        self.assertIn("inspect_real_dynamic_channels", names)
        self.assertIn("list_remote_psse_m1m2_cases", names)
        self.assertIn("run_remote_psse_m1m2", names)

    def test_list_tools_can_include_roadmap_placeholders(self) -> None:
        registry = fake_registry()

        result = registry.list_tools(include_unimplemented=True)
        tools = {tool["name"]: tool for tool in result["tools"]}

        self.assertEqual(
            set(tools),
            {
                "list_backends",
                "list_cases",
                "list_dynamic_cases",
                "set_backend",
                "run_powerflow",
                "run_opf",
                "inspect_violations",
                "run_contingency",
                "run_transient_stability",
                "run_emt_screening",
                "run_cia",
                "run_integrated_assessment",
                "list_real_psse_cases",
                "run_real_psse_assessment",
                "list_real_interconnection_actions",
                "run_real_interconnection_assessment",
                "inspect_real_case_summary",
                "inspect_real_poc_context",
                "inspect_real_network_neighborhood",
                "inspect_real_model_inventory",
                "inspect_real_static_operating_point",
                "inspect_real_dynamic_channels",
                "list_remote_psse_m1m2_cases",
                "run_remote_psse_m1m2",
                "run_cia_with_mitigation",
                "find_max_capacity",
                "query_network_data",
            },
        )
        self.assertTrue(tools["run_powerflow"]["implemented"])
        self.assertTrue(tools["run_cia"]["implemented"])
        self.assertTrue(tools["run_integrated_assessment"]["implemented"])
        self.assertTrue(tools["run_contingency"]["implemented"])
        self.assertTrue(tools["find_max_capacity"]["implemented"])
        self.assertTrue(tools["run_transient_stability"]["implemented"])
        self.assertTrue(tools["run_emt_screening"]["implemented"])
        self.assertTrue(tools["list_real_psse_cases"]["implemented"])
        self.assertTrue(tools["run_real_psse_assessment"]["implemented"])
        self.assertTrue(tools["list_real_interconnection_actions"]["implemented"])
        self.assertTrue(tools["run_real_interconnection_assessment"]["implemented"])
        self.assertTrue(tools["inspect_real_case_summary"]["implemented"])
        self.assertTrue(tools["inspect_real_poc_context"]["implemented"])
        self.assertTrue(tools["inspect_real_network_neighborhood"]["implemented"])
        self.assertTrue(tools["inspect_real_model_inventory"]["implemented"])
        self.assertTrue(tools["inspect_real_static_operating_point"]["implemented"])
        self.assertTrue(tools["inspect_real_dynamic_channels"]["implemented"])
        self.assertTrue(tools["list_remote_psse_m1m2_cases"]["implemented"])
        self.assertTrue(tools["run_remote_psse_m1m2"]["implemented"])

    def test_integrated_assessment_schema_exposes_emt_block(self) -> None:
        registry = fake_registry()

        specs = {
            spec["function"]["name"]: spec["function"]["parameters"]
            for spec in registry.openai_tool_specs()
        }

        integrated = specs["run_integrated_assessment"]
        emt = integrated["properties"]["emt"]
        self.assertIn("scr_threshold", emt["properties"])
        self.assertIn("required_for_approval", emt["properties"])
        self.assertFalse(emt["additionalProperties"])

    def test_list_backends_and_cases_are_metadata_only(self) -> None:
        created: List[FakeSolver] = []
        registry = fake_registry(created)

        backends = registry.call_tool("list_backends")
        cases = registry.call_tool("list_cases")

        self.assertEqual(backends["active_backend"], "fake")
        self.assertEqual(cases["cases"], ["ieee2"])
        self.assertEqual(created, [])

    def test_set_backend_rejects_unknown_backend(self) -> None:
        registry = fake_registry()

        with self.assertRaises(ToolRegistryError):
            registry.call_tool("set_backend", {"backend": "missing"})

    def test_run_emt_screening_returns_scr_metrics(self) -> None:
        registry = fake_registry()

        result = registry.call_tool(
            "run_emt_screening",
            {
                "case_path": "ieee2",
                "connection": {
                    "bus": 1,
                    "p_mw": 10.0,
                    "connection_type": "solar",
                    "is_ibr": True,
                },
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "run_emt_screening")
        self.assertEqual(result["emt"]["status"], "pass")
        self.assertAlmostEqual(result["metrics"]["scr"], 5.0)

    def test_run_powerflow_calls_solver_once_and_returns_grounded_report(self) -> None:
        created: List[FakeSolver] = []
        registry = fake_registry(created)

        result = registry.call_tool(
            "run_powerflow",
            {"case_path": "ieee2", "max_bus_results": 1, "max_branch_results": 1},
        )

        self.assertTrue(result["ok"])
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].run_count, 1)
        self.assertEqual(result["powerflow_summary"]["converged"], True)
        self.assertEqual(result["violation_report"]["status"], "pass")
        self.assertEqual(result["bus_results"]["truncated_rows"], 1)

    def test_inspect_violations_returns_only_violation_report(self) -> None:
        registry = fake_registry()

        result = registry.call_tool("inspect_violations", {"case_path": "ieee2"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "inspect_violations")
        self.assertEqual(result["violation_report"]["status"], "pass")
        self.assertNotIn("bus_results", result)

    def test_violation_report_can_be_truncated_without_losing_counts(self) -> None:
        registry = fake_registry()

        result = registry.call_tool(
            "inspect_violations",
            {"case_path": "ieee2", "angle_limit_degree": 0, "max_violations": 0},
        )

        report = result["violation_report"]
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["total_violations"], 1)
        self.assertEqual(report["truncated_violations"], 1)
        self.assertEqual(report["violations"], [])

    def test_query_network_data_does_not_claim_solved_operating_point(self) -> None:
        created: List[FakeSolver] = []
        registry = fake_registry(created)

        result = registry.call_tool("query_network_data", {"case_path": "ieee2", "max_rows": 1})

        self.assertTrue(result["ok"])
        self.assertFalse(result["solved_operating_point_claimed"])
        self.assertEqual(created[0].run_count, 0)
        self.assertEqual(result["network_data"]["max_rows_per_table"], 1)

    def test_run_cia_tool_returns_structured_report(self) -> None:
        registry = fake_registry()

        result = registry.call_tool(
            "run_cia",
            {
                "case_path": "ieee2",
                "connection": {
                    "bus": 1,
                    "p_mw": 10.0,
                    "connection_type": "load",
                    "is_ibr": False,
                },
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["recommendation"], "approve")
        self.assertEqual(result["stage_reports"][0]["stage"], "f1_steady_state")

    def test_run_contingency_tool_returns_failed_outage_report(self) -> None:
        registry = fake_registry()

        result = registry.call_tool(
            "run_contingency",
            {
                "case_path": "ieee2",
                "max_contingencies": 1,
                "max_failed_contingencies": 1,
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["stage"], "f2_n1_contingency")
        self.assertEqual(result["limit_profile"]["name"], "emergency")
        self.assertIn("base_powerflow", result)
        self.assertNotIn("contingency_results", result)
        self.assertEqual(result["summary"]["failed_contingencies"], 1)
        self.assertEqual(result["failed_contingencies"]["items"][0]["status"], "fail")

    def test_run_contingency_can_include_full_outage_results(self) -> None:
        registry = fake_registry()

        result = registry.call_tool(
            "run_contingency",
            {
                "case_path": "ieee2",
                "max_contingencies": 1,
                "include_contingency_results": True,
            },
        )

        self.assertEqual(len(result["contingency_results"]), 1)

    def test_find_max_capacity_tool_returns_search_report(self) -> None:
        registry = fake_registry()

        result = registry.call_tool(
            "find_max_capacity",
            {
                "case_path": "ieee2",
                "bus": 1,
                "connection_type": "load",
                "max_mw": 10.0,
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "find_max_capacity")
        self.assertEqual(result["status"], "max_bound_approved")
        self.assertEqual(result["max_approved_mw"], 10.0)

    def test_unimplemented_tool_returns_explicit_not_implemented_payload(self) -> None:
        registry = fake_registry()

        result = registry.call_tool("run_opf", {"case_path": "ieee2"})

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "not_implemented")

    def test_missing_required_argument_is_rejected(self) -> None:
        registry = fake_registry()

        with self.assertRaisesRegex(ToolRegistryError, "case_path"):
            registry.call_tool("inspect_violations", {})

    def test_unexpected_argument_is_rejected(self) -> None:
        registry = fake_registry()

        with self.assertRaisesRegex(ToolRegistryError, "Unexpected argument"):
            registry.call_tool("list_cases", {"case_path": "ieee2"})

    def test_wrong_argument_type_is_rejected(self) -> None:
        registry = fake_registry()

        with self.assertRaisesRegex(ToolRegistryError, "must be an integer"):
            registry.call_tool("query_network_data", {"case_path": "ieee2", "max_rows": "two"})

    def test_negative_float_tool_option_is_rejected(self) -> None:
        registry = fake_registry()

        with self.assertRaisesRegex(ToolRegistryError, "non-negative"):
            registry.call_tool(
                "run_cia",
                {
                    "case_path": "ieee2",
                    "connection": {
                        "bus": 1,
                        "p_mw": 10.0,
                        "connection_type": "load",
                        "is_ibr": False,
                    },
                    "material_worsening_threshold_percent": -1.0,
                },
            )

    def test_solver_powerflow_error_is_wrapped_as_tool_error(self) -> None:
        registry = fake_registry(solver_cls=FailingPowerFlowSolver)

        with self.assertRaisesRegex(ToolRegistryError, "fake power flow failed"):
            registry.call_tool("run_powerflow", {"case_path": "ieee2"})

    def test_enum_argument_is_rejected(self) -> None:
        registry = fake_registry()

        with self.assertRaisesRegex(ToolRegistryError, "must be one of"):
            registry.call_tool("inspect_violations", {"case_path": "ieee2", "profile": "planning"})

    def test_unknown_tool_raises(self) -> None:
        registry = fake_registry()

        with self.assertRaises(ToolRegistryError):
            registry.call_tool("made_up_tool")


if __name__ == "__main__":
    unittest.main()

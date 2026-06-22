"""Tests for the Step 7 persistent study memory."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import (  # noqa: E402
    BackendDefinition,
    CaseInfo,
    ConnectionRequest,
    GridSolver,
    PowerFlowError,
    StudyMemoryStore,
    ToolRegistry,
)


def cia_result() -> Dict[str, Any]:
    return {
        "ok": True,
        "tool": "run_cia",
        "case_path": "ieee14",
        "connection": {
            "bus": 10,
            "p_mw": 5.0,
            "connection_type": "load",
            "is_ibr": False,
        },
        "recommendation": "approve",
        "complete": True,
        "reason_codes": ["steady_state_no_project_caused_violations"],
        "stage_reports": [
            {
                "stage": "f1_steady_state",
                "status": "pass",
                "reason_codes": ["steady_state_no_project_caused_violations"],
                "project_violation_comparison": {
                    "project_caused_violations": {
                        "items": [],
                        "total_items": 0,
                        "truncated_items": 0,
                    }
                },
            }
        ],
        "summary": {
            "project_hard_violations": 0,
            "project_borderline_violations": 0,
            "pre_existing_hard_violations": 0,
            "pre_existing_borderline_violations": 0,
        },
    }


def capacity_result() -> Dict[str, Any]:
    return {
        "ok": True,
        "tool": "find_max_capacity",
        "case_path": "ieee14",
        "request": {"bus": 10, "connection_type": "load", "is_ibr": False},
        "status": "bounded",
        "max_approved_mw": 40.0,
        "lower_bound_mw": 40.0,
        "upper_bound_mw": 41.0,
        "tolerance_mw": 1.0,
        "iterations": 8,
        "boundary_samples": {
            "best_approved": {"mw": 40.0, "accepted": True, "recommendation": "approve"},
            "first_rejected": {
                "mw": 41.0,
                "accepted": False,
                "recommendation": "reject",
                "limiting_summary": {"limiting_stage": "f1_steady_state"},
            },
        },
        "rejection_explanation": {
            "limiting_stage": "f1_steady_state",
            "status": "fail",
            "reason_codes": ["steady_state_project_hard_violations"],
        },
        "samples": {"items": [], "total_items": 10, "truncated_items": 0},
        "diagnostics": {"fallback_used": False},
    }


class MemoryScenarioSolver(GridSolver):
    def __init__(self) -> None:
        self.loaded_case = ""
        self.connection: Optional[ConnectionRequest] = None
        self.converged = False

    def available_cases(self) -> List[str]:
        return ["ieee2"]

    def load_case(self, case_name: str) -> CaseInfo:
        self.loaded_case = case_name
        return self.case_info()

    def run_powerflow(self) -> Dict[str, Any]:
        self.converged = True
        return {
            "backend": "memory_fake",
            "converged": True,
            "case": self.case_info().to_dict(),
            "bus_summary": {"min_vm_pu": 1.0, "max_vm_pu": 1.0},
            "branch_summary": {"max_line_loading_percent": 50.0},
        }

    def bus_results(self) -> List[Dict[str, Any]]:
        return [{"bus_index": 1, "name": "Bus 1", "vm_pu": 1.0, "va_degree": 0.0}]

    def branch_results(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "lines": [
                {
                    "line_index": 1,
                    "name": "Line 1",
                    "from_bus": 1,
                    "to_bus": 1,
                    "loading_percent": 50.0,
                    "in_service": True,
                }
            ],
            "transformers": [],
        }

    def available_contingencies(self) -> List[Dict[str, Any]]:
        return []

    def apply_contingency(self, element_type: str, element_index: int) -> Dict[str, Any]:
        raise ValueError("No contingencies in memory test case")

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
        self.converged = False
        return {
            "element_table": "load",
            "element_index": 0,
            "name": name or "memory_connection",
            "connection_type": connection_type,
            "is_ibr": is_ibr,
            "bus": bus,
            "resolved_bus": {"bus_index": bus, "matched_on": "memory"},
            "p_mw": p_mw,
            "q_mvar": q_mvar,
        }

    def network_data(self, max_rows: int = 50) -> Dict[str, Any]:
        return {"case": self.case_info().to_dict(), "tables": {}}

    def case_info(self) -> CaseInfo:
        return CaseInfo(
            case_name=self.loaded_case or "ieee2",
            buses=1,
            lines=1,
            transformers=0,
            loads=0,
            generators=0,
            static_generators=0,
            external_grids=1,
        )


class FailingMemorySolver(MemoryScenarioSolver):
    def run_powerflow(self) -> Dict[str, Any]:
        raise PowerFlowError("memory fake baseline failed")


def memory_registry(
    store: StudyMemoryStore,
    solver_cls: type[MemoryScenarioSolver] = MemoryScenarioSolver,
) -> ToolRegistry:
    return ToolRegistry(
        backends={
            "fake": BackendDefinition(
                name="fake",
                solver_factory=solver_cls,
                supported_cases=lambda: ["ieee2"],
                is_available=lambda: True,
                description="Memory fake backend.",
            )
        },
        active_backend="fake",
        memory_store=store,
    )


class StudyMemoryTest(unittest.TestCase):
    def test_memory_store_persists_cia_record_and_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StudyMemoryStore(tmp)

            record = store.append_tool_result(cia_result())

            self.assertEqual(record.tool, "run_cia")
            self.assertEqual(record.case_path, "ieee14")
            self.assertEqual(record.bus, 10)
            self.assertIn("CIA approve", record.summary)
            self.assertEqual(len(store.load_records()), 1)
            self.assertTrue(store.ledger_path.exists())
            self.assertIn("Mini Grid-Mind Study Ledger", store.ledger_path.read_text())

    def test_memory_recall_modes_cover_bus_case_keyword_and_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StudyMemoryStore(tmp)
            store.append_tool_result(cia_result())
            store.append_tool_result(capacity_result())

            self.assertEqual(len(store.recall_bus(case_path="IEEE14", bus=10)), 2)
            self.assertEqual(len(store.recall_case(case_path="ieee14")), 2)
            self.assertEqual(len(store.search("CIA approve")), 1)
            capacity_records = store.recall_max_capacity(
                case_path="ieee14",
                bus=10,
                connection_type="load",
            )
            self.assertEqual(len(capacity_records), 1)
            self.assertEqual(capacity_records[0].mw, 40.0)

    def test_prompt_context_contains_memory_caveat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StudyMemoryStore(tmp)
            record = store.append_tool_result(capacity_result())

            context = store.build_prompt_context([record])

            self.assertIn("supplementary session context", context)
            self.assertIn("not as independent historical studies", context)
            self.assertIn(record.record_id, context)

    def test_recall_rejects_invalid_public_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StudyMemoryStore(tmp)
            store.append_tool_result(capacity_result())

            with self.assertRaisesRegex(ValueError, "case_path"):
                store.recall_case(case_path="")
            with self.assertRaisesRegex(ValueError, "bus"):
                store.recall_bus(case_path="ieee14", bus=True)  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValueError, "query"):
                store.search(10)  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValueError, "connection_type"):
                store.recall_max_capacity(connection_type="")
            with self.assertRaisesRegex(ValueError, "limit"):
                store.recent(limit=True)  # type: ignore[arg-type]

    def test_malformed_jsonl_record_reports_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StudyMemoryStore(tmp)
            Path(store.records_path).parent.mkdir(parents=True, exist_ok=True)
            Path(store.records_path).write_text(
                '{"record_id":"bad","timestamp_utc":"now","tool":"run_cia",'
                '"case_path":"ieee14","status":"approve","summary":"bad","data":[]}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "studies.jsonl:1"):
                store.load_records()

    def test_registry_persists_cia_and_capacity_when_memory_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StudyMemoryStore(tmp)
            registry = memory_registry(store)

            cia = registry.call_tool(
                "run_cia",
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
            capacity = registry.call_tool(
                "find_max_capacity",
                {
                    "case_path": "ieee2",
                    "bus": 1,
                    "connection_type": "load",
                    "max_mw": 10.0,
                },
            )

            self.assertIn("memory_record", cia)
            self.assertIn("memory_record", capacity)
            records = store.load_records()
            self.assertEqual([record.tool for record in records], ["run_cia", "find_max_capacity"])

    def test_registry_skips_memory_for_unsuccessful_cia_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StudyMemoryStore(tmp)
            registry = memory_registry(store, solver_cls=FailingMemorySolver)

            result = registry.call_tool(
                "run_cia",
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

            self.assertFalse(result["ok"])
            self.assertEqual(result["error_type"], "baseline_powerflow_failed")
            self.assertEqual(store.load_records(), [])


if __name__ == "__main__":
    unittest.main()

"""Tool registry for the Mini Grid-Mind reproduction.

The registry is the boundary the future LLM agent will see. It exposes
solver-backed operations through JSON-schema tool definitions and keeps
unimplemented Grid-Mind tools explicit rather than silently faking behavior.
"""

from __future__ import annotations

import contextlib
import math
import sys
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .capacity import CapacitySearchRunner
from .cia import ConnectionRequest, SteadyStateCIARunner
from .contingency import ContingencyRunner
from .dynamic import DynamicSimulationError, TransientStabilityRunner, list_dynamic_cases
from .emt import EMTScreeningError, EMTScreeningRunner
from .integrated_assessment import (
    EMTAssessmentRequest,
    IntegratedAssessmentRunner,
    TransientAssessmentRequest,
)
from .real_case_dossier import (
    inspect_real_case_summary,
    inspect_real_dynamic_channels,
    inspect_real_model_inventory,
    inspect_real_network_neighborhood,
    inspect_real_poc_context,
    inspect_real_static_operating_point,
)
from .memory import StudyMemoryStore
from .real_data import list_real_psse_cases, run_real_psse_assessment
from .real_interconnection import (
    list_real_interconnection_actions,
    run_real_interconnection_assessment,
)
from .remote_psse import (
    list_remote_psse_m1m2_cases,
    run_remote_psse_m1m2,
)
from .solver import GridSolver, PandaPowerSolver, PowerFlowError, SolverDependencyError
from .violations import LimitProfile, ViolationInspector, profile_from_name


class ToolRegistryError(RuntimeError):
    """Raised when a tool call is invalid or cannot be executed."""


@dataclass(frozen=True)
class BackendDefinition:
    """Factory and metadata hooks for one solver backend."""

    name: str
    solver_factory: Callable[[], GridSolver]
    supported_cases: Callable[[], List[str]]
    is_available: Callable[[], bool]
    description: str

    def to_dict(self, active: bool = False) -> Dict[str, Any]:
        return {
            "name": self.name,
            "active": active,
            "available": self.is_available(),
            "supported_cases": self.supported_cases(),
            "description": self.description,
        }


@dataclass(frozen=True)
class ToolDefinition:
    """Agent-facing tool metadata."""

    name: str
    description: str
    parameters: Dict[str, Any]
    group: str
    implemented: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "group": self.group,
            "implemented": self.implemented,
            "parameters": self.parameters,
        }

    def to_openai_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def default_backends() -> Dict[str, BackendDefinition]:
    return {
        "pandapower": BackendDefinition(
            name="pandapower",
            solver_factory=PandaPowerSolver,
            supported_cases=PandaPowerSolver.supported_cases,
            is_available=PandaPowerSolver.is_available,
            description="Steady-state AC power-flow backend used for screening.",
        )
    }


class ToolRegistry:
    """Small OpenAI-style action registry modeled on Grid-Mind's tool layer."""

    def __init__(
        self,
        backends: Optional[Mapping[str, BackendDefinition]] = None,
        active_backend: str = "pandapower",
        memory_store: Optional[StudyMemoryStore] = None,
    ) -> None:
        self._backends = dict(backends or default_backends())
        self._tool_definitions = _build_tool_definitions()
        self._active_backend = self._validate_backend(active_backend)
        self._memory_store = memory_store

    @property
    def active_backend(self) -> str:
        return self._active_backend

    def list_tools(self, include_unimplemented: bool = True) -> Dict[str, Any]:
        tools = [
            definition.to_dict()
            for definition in self._tool_definitions.values()
            if include_unimplemented or definition.implemented
        ]
        return {
            "active_backend": self.active_backend,
            "tool_count": len(tools),
            "tools": tools,
        }

    def openai_tool_specs(self, include_unimplemented: bool = False) -> List[Dict[str, Any]]:
        return [
            definition.to_openai_spec()
            for definition in self._tool_definitions.values()
            if include_unimplemented or definition.implemented
        ]

    def call_tool(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        if name not in self._tool_definitions:
            raise ToolRegistryError(f"Unknown tool '{name}'")
        definition = self._tool_definitions[name]
        args = dict(arguments or {})
        _validate_json_args(definition.parameters, args, path=name)

        if not definition.implemented:
            return {
                "ok": False,
                "error_type": "not_implemented",
                "tool": name,
                "message": "This Grid-Mind tool is declared for the roadmap but not implemented yet.",
            }

        handlers = {
            "list_backends": self._tool_list_backends,
            "list_cases": self._tool_list_cases,
            "set_backend": self._tool_set_backend,
            "run_powerflow": self._tool_run_powerflow,
            "inspect_violations": self._tool_inspect_violations,
            "run_contingency": self._tool_run_contingency,
            "run_emt_screening": self._tool_run_emt_screening,
            "run_cia": self._tool_run_cia,
            "run_integrated_assessment": self._tool_run_integrated_assessment,
            "find_max_capacity": self._tool_find_max_capacity,
            "query_network_data": self._tool_query_network_data,
            "list_dynamic_cases": self._tool_list_dynamic_cases,
            "run_transient_stability": self._tool_run_transient_stability,
            "list_real_psse_cases": self._tool_list_real_psse_cases,
            "run_real_psse_assessment": self._tool_run_real_psse_assessment,
            "list_real_interconnection_actions": self._tool_list_real_interconnection_actions,
            "run_real_interconnection_assessment": self._tool_run_real_interconnection_assessment,
            "inspect_real_case_summary": self._tool_inspect_real_case_summary,
            "inspect_real_poc_context": self._tool_inspect_real_poc_context,
            "inspect_real_network_neighborhood": self._tool_inspect_real_network_neighborhood,
            "inspect_real_model_inventory": self._tool_inspect_real_model_inventory,
            "inspect_real_static_operating_point": self._tool_inspect_real_static_operating_point,
            "inspect_real_dynamic_channels": self._tool_inspect_real_dynamic_channels,
            "list_remote_psse_m1m2_cases": self._tool_list_remote_psse_m1m2_cases,
            "run_remote_psse_m1m2": self._tool_run_remote_psse_m1m2,
        }
        with contextlib.redirect_stdout(sys.stderr):
            return handlers[name](args)

    def _tool_list_backends(self, _args: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "ok": True,
            "active_backend": self.active_backend,
            "backends": [
                backend.to_dict(active=name == self.active_backend)
                for name, backend in self._backends.items()
            ],
        }

    def _tool_list_cases(self, _args: Mapping[str, Any]) -> Dict[str, Any]:
        backend = self._backend()
        return {
            "ok": True,
            "backend": backend.name,
            "cases": backend.supported_cases(),
        }

    def _tool_list_dynamic_cases(self, _args: Mapping[str, Any]) -> Dict[str, Any]:
        return list_dynamic_cases()

    def _tool_list_real_psse_cases(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        return list_real_psse_cases(processed_dir=args.get("processed_dir"))

    def _tool_list_real_interconnection_actions(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        return list_real_interconnection_actions(
            processed_dir=args.get("processed_dir"),
            inventory_dir=args.get("inventory_dir"),
            case_id=str(args.get("case_id") or "pif6_2026_05_17"),
        )

    def _tool_list_remote_psse_m1m2_cases(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        return list_remote_psse_m1m2_cases(
            check_health=bool(args.get("check_health", False)),
        )

    def _tool_inspect_real_case_summary(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        return inspect_real_case_summary(
            case_id=_required_str(args, "case_id"),
            processed_dir=args.get("processed_dir"),
            max_poc_candidates=_optional_int(args, "max_poc_candidates", 8),
        )

    def _tool_inspect_real_poc_context(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        return inspect_real_poc_context(
            case_id=_required_str(args, "case_id"),
            poc_label_or_bus=args.get("poc_label_or_bus"),
            processed_dir=args.get("processed_dir"),
            max_candidates=_optional_int(args, "max_candidates", 12),
            max_branches=_optional_int(args, "max_branches", 12),
        )

    def _tool_inspect_real_network_neighborhood(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        return inspect_real_network_neighborhood(
            case_id=_required_str(args, "case_id"),
            bus=_optional_int(args, "bus", 0),
            depth=_optional_int(args, "depth", 1),
            max_rows=_optional_int(args, "max_rows", 20),
            processed_dir=args.get("processed_dir"),
        )

    def _tool_inspect_real_model_inventory(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        return inspect_real_model_inventory(
            case_id=_required_str(args, "case_id"),
            model_scope=str(args.get("model_scope") or "summary"),
            processed_dir=args.get("processed_dir"),
            max_rows=_optional_int(args, "max_rows", 20),
        )

    def _tool_inspect_real_static_operating_point(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        return inspect_real_static_operating_point(
            case_id=_required_str(args, "case_id"),
            processed_dir=args.get("processed_dir"),
            max_voltage_rows=_optional_int(args, "max_voltage_rows", 6),
            max_overload_rows=_optional_int(args, "max_overload_rows", 6),
        )

    def _tool_inspect_real_dynamic_channels(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        return inspect_real_dynamic_channels(
            case_id=_required_str(args, "case_id"),
            scenario_type=str(args.get("scenario_type") or "no_disturbance_5s"),
            processed_dir=args.get("processed_dir"),
            max_samples=_optional_int(args, "max_samples", 5),
        )

    def _tool_set_backend(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        backend_name = _required_str(args, "backend")
        self._active_backend = self._validate_backend(backend_name)
        return {
            "ok": True,
            "active_backend": self.active_backend,
        }

    def _tool_run_powerflow(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        solver, powerflow_summary = self._load_and_solve(args)
        profile = _limit_profile_from_args(args)
        report = ViolationInspector(profile).inspect_solver(solver)
        max_bus_results = _optional_int(args, "max_bus_results", 20)
        max_branch_results = _optional_int(args, "max_branch_results", 20)
        max_violations = _optional_int(args, "max_violations", 50)

        bus_rows = solver.bus_results()
        branch_rows = solver.branch_results()
        return {
            "ok": True,
            "tool": "run_powerflow",
            "backend": self.active_backend,
            "case_info": solver.case_info().to_dict(),
            "powerflow_summary": powerflow_summary,
            "violation_report": _limit_violation_report(report.to_dict(), max_violations),
            "bus_results": _limit_rows(bus_rows, max_bus_results),
            "branch_results": {
                "lines": _limit_rows(branch_rows.get("lines", []), max_branch_results),
                "transformers": _limit_rows(branch_rows.get("transformers", []), max_branch_results),
            },
        }

    def _tool_inspect_violations(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        solver, _powerflow_summary = self._load_and_solve(args)
        profile = _limit_profile_from_args(args)
        report = ViolationInspector(profile).inspect_solver(solver)
        max_violations = _optional_int(args, "max_violations", 50)
        return {
            "ok": True,
            "tool": "inspect_violations",
            "backend": self.active_backend,
            "case_info": solver.case_info().to_dict(),
            "violation_report": _limit_violation_report(report.to_dict(), max_violations),
        }

    def _tool_query_network_data(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        solver = self._load_case(args)
        max_rows = _optional_int(args, "max_rows", 50)
        return {
            "ok": True,
            "tool": "query_network_data",
            "backend": self.active_backend,
            "network_data": solver.network_data(max_rows=max_rows),
            "solved_operating_point_claimed": False,
        }

    def _tool_run_contingency(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        backend = self._backend()
        if not backend.is_available():
            raise ToolRegistryError(f"Backend '{backend.name}' is not available in this environment")

        case_path = _case_name_from_args(args)
        profile = _limit_profile_from_args(args, default_profile="emergency")
        max_contingencies = _optional_int(args, "max_contingencies", -1)
        max_violations = _optional_int(args, "max_violations", 50)
        max_failed_contingencies = _optional_int(args, "max_failed_contingencies", 50)
        include_contingency_results = bool(args.get("include_contingency_results", False))
        try:
            result = ContingencyRunner(
                solver_factory=backend.solver_factory,
                limit_profile=profile,
            ).run(
                case_path=case_path,
                max_contingencies=max_contingencies,
                max_violations=max_violations,
                max_failed_contingencies=max_failed_contingencies,
            )
        except (PowerFlowError, SolverDependencyError, ValueError) as exc:
            raise ToolRegistryError(str(exc)) from exc
        if not include_contingency_results:
            result.pop("contingency_results", None)
        result["backend"] = self.active_backend
        return result

    def _tool_run_cia(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        backend = self._backend()
        if not backend.is_available():
            raise ToolRegistryError(f"Backend '{backend.name}' is not available in this environment")

        case_path = _case_name_from_args(args)
        try:
            connection = ConnectionRequest.from_mapping(args["connection"])
        except ValueError as exc:
            raise ToolRegistryError(str(exc)) from exc
        profile = _limit_profile_from_args(args)
        worsening_threshold = _optional_float(
            args,
            "material_worsening_threshold_percent",
            2.0,
        )
        max_violations = _optional_int(args, "max_violations", 50)
        max_contingencies = _optional_int(args, "max_contingencies", -1)
        max_failed_contingencies = _optional_int(args, "max_failed_contingencies", 50)
        runner = SteadyStateCIARunner(
            solver_factory=backend.solver_factory,
            limit_profile=profile,
            material_worsening_threshold_percent=worsening_threshold,
        )
        try:
            result = runner.run(
                case_path=case_path,
                connection=connection,
                enable_contingency=bool(args.get("enable_contingency", False)),
                enable_transient=bool(args.get("enable_transient", False)),
                enable_emt=bool(args.get("enable_emt", False)),
                max_violations=max_violations,
                max_contingencies=max_contingencies,
                max_failed_contingencies=max_failed_contingencies,
                fail_on_contingency_material_worsening=bool(
                    args.get("fail_on_contingency_material_worsening", False)
                ),
            )
        except (PowerFlowError, SolverDependencyError, ValueError) as exc:
            raise ToolRegistryError(str(exc)) from exc
        result["backend"] = self.active_backend
        self._persist_study_memory(result)
        return result

    def _tool_run_integrated_assessment(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        backend = self._backend()
        if not backend.is_available():
            raise ToolRegistryError(f"Backend '{backend.name}' is not available in this environment")

        case_path = _case_name_from_args(args)
        try:
            connection = ConnectionRequest.from_mapping(args["connection"])
            transient = TransientAssessmentRequest.from_mapping(args.get("transient"))
            emt = EMTAssessmentRequest.from_mapping(args.get("emt"))
        except ValueError as exc:
            raise ToolRegistryError(str(exc)) from exc
        profile = _limit_profile_from_args(args)
        try:
            result = IntegratedAssessmentRunner(
                solver_factory=backend.solver_factory,
                limit_profile=profile,
                material_worsening_threshold_percent=_optional_float(
                    args,
                    "material_worsening_threshold_percent",
                    2.0,
                ),
            ).run(
                case_path=case_path,
                connection=connection,
                transient=transient,
                emt=emt,
                enable_contingency=bool(args.get("enable_contingency", False)),
                max_violations=_optional_int(args, "max_violations", 50),
                max_contingencies=_optional_int(args, "max_contingencies", -1),
                max_failed_contingencies=_optional_int(args, "max_failed_contingencies", 50),
                fail_on_contingency_material_worsening=bool(
                    args.get("fail_on_contingency_material_worsening", False)
                ),
            )
        except (PowerFlowError, SolverDependencyError, ValueError) as exc:
            raise ToolRegistryError(str(exc)) from exc
        result["backend"]["m1"] = self.active_backend
        return result

    def _tool_find_max_capacity(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        backend = self._backend()
        if not backend.is_available():
            raise ToolRegistryError(f"Backend '{backend.name}' is not available in this environment")

        case_path = _case_name_from_args(args)
        profile = _limit_profile_from_args(args)
        connection_type = _required_str(args, "connection_type").strip().lower()
        is_ibr = args.get("is_ibr")
        if is_ibr is not None and not isinstance(is_ibr, bool):
            raise ToolRegistryError("Tool argument 'is_ibr' must be a boolean")
        try:
            result = CapacitySearchRunner(
                solver_factory=backend.solver_factory,
                limit_profile=profile,
                material_worsening_threshold_percent=_optional_float(
                    args,
                    "material_worsening_threshold_percent",
                    2.0,
                ),
            ).run(
                case_path=case_path,
                bus=_optional_int(args, "bus", 0),
                connection_type=connection_type,
                is_ibr=is_ibr,
                q_mvar=_optional_number(args, "q_mvar", 0.0),
                vm_pu=_optional_number(args, "vm_pu", 1.0),
                min_mw=_optional_float(args, "min_mw", 0.0),
                max_mw=_optional_float(args, "max_mw", 500.0),
                tolerance_mw=_positive_float(args, "tolerance_mw", 1.0),
                max_iterations=_positive_int(args, "max_iterations", 12),
                coarse_scan_points=_positive_int(args, "coarse_scan_points", 11),
                enable_contingency=bool(args.get("enable_contingency", False)),
                max_contingencies=_optional_int(args, "max_contingencies", -1),
                max_failed_contingencies=_optional_int(args, "max_failed_contingencies", 50),
                fail_on_contingency_material_worsening=bool(
                    args.get("fail_on_contingency_material_worsening", False)
                ),
                max_violations=_optional_int(args, "max_violations", 10),
                max_samples=_positive_int(args, "max_samples", 100),
                include_reports=bool(args.get("include_reports", False)),
            )
        except (PowerFlowError, SolverDependencyError, ValueError) as exc:
            raise ToolRegistryError(str(exc)) from exc
        result["backend"] = self.active_backend
        self._persist_study_memory(result)
        return result

    def _tool_run_transient_stability(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            return TransientStabilityRunner().run(
                case_path=_case_name_from_args(args),
                disturbance=args["disturbance"],
                connection=args.get("connection"),
                simulation_time_s=_optional_number(args, "simulation_time_s", 5.0),
                max_angle_spread_rad=_optional_number(
                    args,
                    "max_angle_spread_rad",
                    2.0 * math.pi,
                ),
                max_speed_deviation_pu=_optional_number(
                    args,
                    "max_speed_deviation_pu",
                    0.05,
                ),
                min_voltage_pu=_optional_number(args, "min_voltage_pu", 0.7),
                max_samples=_positive_int(args, "max_samples", 200),
            )
        except DynamicSimulationError as exc:
            raise ToolRegistryError(str(exc)) from exc

    def _tool_run_emt_screening(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        backend = self._backend()
        if not backend.is_available():
            raise ToolRegistryError(f"Backend '{backend.name}' is not available in this environment")
        try:
            connection = ConnectionRequest.from_mapping(args["connection"])
            result = EMTScreeningRunner(backend.solver_factory).run(
                case_path=_case_name_from_args(args),
                connection=connection,
                scr_threshold=_optional_number(args, "scr_threshold", 3.0),
                borderline_margin=_optional_number(args, "borderline_margin", 0.25),
            )
        except (EMTScreeningError, PowerFlowError, SolverDependencyError, ValueError) as exc:
            raise ToolRegistryError(str(exc)) from exc
        result["backend"] = self.active_backend
        return result

    def _tool_run_real_psse_assessment(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        return run_real_psse_assessment(
            case_id=_required_str(args, "case_id"),
            processed_dir=args.get("processed_dir"),
            include_channel_preview=bool(args.get("include_channel_preview", False)),
            max_channel_samples=_optional_int(args, "max_channel_samples", 20),
        )

    def _tool_run_real_interconnection_assessment(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        return run_real_interconnection_assessment(
            case_id=_required_str(args, "case_id"),
            connection=_mapping_arg(args, "connection"),
            disturbance=_mapping_arg(args, "disturbance"),
            processed_dir=args.get("processed_dir"),
        )

    def _tool_run_remote_psse_m1m2(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        return run_remote_psse_m1m2(
            case_id=_required_str(args, "case_id"),
            scenario_type=_required_str(args, "scenario_type"),
            request_id=args.get("request_id"),
            include_artifacts=bool(args.get("include_artifacts", False)),
        )

    def _load_and_solve(self, args: Mapping[str, Any]) -> tuple[GridSolver, Dict[str, Any]]:
        solver = self._load_case(args)
        try:
            powerflow_summary = solver.run_powerflow()
        except PowerFlowError as exc:
            raise ToolRegistryError(str(exc)) from exc
        return solver, powerflow_summary

    def _load_case(self, args: Mapping[str, Any]) -> GridSolver:
        case_name = _case_name_from_args(args)
        backend = self._backend()
        if not backend.is_available():
            raise ToolRegistryError(f"Backend '{backend.name}' is not available in this environment")
        try:
            solver = backend.solver_factory()
            solver.load_case(case_name)
        except (PowerFlowError, SolverDependencyError, ValueError) as exc:
            raise ToolRegistryError(str(exc)) from exc
        return solver

    def _backend(self) -> BackendDefinition:
        return self._backends[self.active_backend]

    def _persist_study_memory(self, result: Dict[str, Any]) -> None:
        if self._memory_store is None:
            return
        if not result.get("ok", False):
            return
        try:
            record = self._memory_store.append_tool_result(result)
        except (OSError, ValueError) as exc:
            raise ToolRegistryError(f"Failed to persist study memory: {exc}") from exc
        result["memory_record"] = record.to_reference()

    def _validate_backend(self, backend_name: str) -> str:
        key = backend_name.strip().lower()
        if key not in self._backends:
            raise ToolRegistryError(
                f"Unknown backend '{backend_name}'. Available backends: {', '.join(self._backends)}"
            )
        return key


def _build_tool_definitions() -> Dict[str, ToolDefinition]:
    definitions = [
        ToolDefinition(
            name="list_backends",
            group="system",
            description="List registered solver backends and availability.",
            parameters=_object_schema({}, required=[]),
        ),
        ToolDefinition(
            name="list_cases",
            group="system",
            description="List built-in test cases supported by the active backend.",
            parameters=_object_schema({}, required=[]),
        ),
        ToolDefinition(
            name="list_dynamic_cases",
            group="system",
            description=(
                "List ANDES dynamic cases available for M2 transient-stability "
                "screening, including dynamic-data availability and disturbance types."
            ),
            parameters=_object_schema({}, required=[]),
        ),
        ToolDefinition(
            name="set_backend",
            group="system",
            description="Set the active solver backend after validating the backend name.",
            parameters=_object_schema(
                {
                    "backend": {
                        "type": "string",
                        "description": "Backend name, e.g. pandapower.",
                    }
                },
                required=["backend"],
            ),
        ),
        ToolDefinition(
            name="run_powerflow",
            group="analysis",
            description=(
                "Load a case, run AC power flow, return convergence summary, bounded bus/branch "
                "results, and a violation report grounded in solver outputs."
            ),
            parameters=_case_profile_schema(
                extra={
                    "max_bus_results": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum bus-result rows to return; -1 returns all.",
                    },
                    "max_branch_results": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum line/transformer rows to return per table; -1 returns all.",
                    },
                    "max_violations": {
                        "type": "integer",
                        "default": 50,
                        "description": "Maximum violation rows to return; -1 returns all.",
                    },
                }
            ),
        ),
        ToolDefinition(
            name="run_opf",
            group="analysis",
            description="Run OPF and inspect post-OPF violations. Roadmap placeholder.",
            parameters=_case_schema(),
            implemented=False,
        ),
        ToolDefinition(
            name="inspect_violations",
            group="analysis",
            description=(
                "Run AC power flow and return a structured voltage, thermal, and optional "
                "angle-difference violation report."
            ),
            parameters=_case_profile_schema(),
        ),
        ToolDefinition(
            name="run_contingency",
            group="analysis",
            description=(
                "Run single line/transformer N-1 contingency screening under emergency limits."
            ),
            parameters=_contingency_schema(),
        ),
        ToolDefinition(
            name="run_transient_stability",
            group="dynamic_analysis",
            description=(
                "Run ANDES time-domain transient-stability simulation for supported "
                "dynamic cases with a bus fault or line-trip disturbance. Optionally "
                "adds a proposed interconnection as a static PQ load/injection before "
                "PFlow and TDS."
            ),
            parameters=_transient_schema(),
        ),
        ToolDefinition(
            name="run_emt_screening",
            group="dynamic_analysis",
            description=(
                "Run an EMT/SCR screening proxy for an inverter-based proposed "
                "connection. This computes short-circuit ratio at the point of "
                "interconnection from the solved pandapower network and applies "
                "the Grid-Mind-style SCR threshold; it is not a waveform EMT simulation."
            ),
            parameters=_emt_schema(),
        ),
        ToolDefinition(
            name="run_cia",
            group="assessment",
            description=(
                "Run baseline-aware CIA with steady-state screening, optional N-1 "
                "contingency screening, and optional F4 EMT/SCR screening. The "
                "classic F3 transient stage remains a placeholder in this tool; "
                "use run_integrated_assessment for executable M2 transient screening."
            ),
            parameters=_cia_schema(),
        ),
        ToolDefinition(
            name="run_integrated_assessment",
            group="assessment",
            description=(
                "Run one integrated assessment: M1 steady-state CIA for the "
                "proposed connection plus optional M2 ANDES transient-stability "
                "screening and optional F4 EMT/SCR screening. M2 uses a static "
                "PQ representation of the same connection when enabled."
            ),
            parameters=_integrated_assessment_schema(),
        ),
        ToolDefinition(
            name="list_real_psse_cases",
            group="real_data",
            description=(
                "List frozen real-data PSS/E cases available from processed PSSE "
                "JSON/CSV outputs. This does not execute PSS/E."
            ),
            parameters=_real_psse_list_schema(),
        ),
        ToolDefinition(
            name="run_real_psse_assessment",
            group="real_data",
            description=(
                "Read a frozen PSS/E result bundle and return a grounded real-data "
                "M1 power-flow plus M2 RMS-dynamic assessment. This consumes "
                "processed JSON/CSV files and does not load SAV/DYR/DLL files."
            ),
            parameters=_real_psse_assessment_schema(),
        ),
        ToolDefinition(
            name="list_real_interconnection_actions",
            group="real_data",
            description=(
                "List the offline frozen-PSS/E interconnection action space built "
                "from real inventory exports, including PIF6 candidate POC buses, "
                "static-PQ project types, disturbances, catalog size, and result "
                "availability. This does not execute PSS/E."
            ),
            parameters=_real_interconnection_list_schema(),
        ),
        ToolDefinition(
            name="run_real_interconnection_assessment",
            group="real_data",
            description=(
                "Run an exact-match lookup against precomputed PSS/E interconnection "
                "study results. Use this for real-data PIF6 questions that add or "
                "connect a solar, wind, BESS, or load project and ask for approval, "
                "RMS dynamic survival, or interconnection impact."
            ),
            parameters=_real_interconnection_assessment_schema(),
        ),
        ToolDefinition(
            name="inspect_real_case_summary",
            group="real_data",
            description=(
                "Inspect a processed real PSS/E case package: inventory counts, "
                "available remote scenarios, POC candidates, static/dynamic snapshot, "
                "artifact availability, and data-quality warnings. Read-only."
            ),
            parameters=_real_case_summary_schema(),
        ),
        ToolDefinition(
            name="inspect_real_poc_context",
            group="real_data",
            description=(
                "Inspect candidate POC buses/branches and nearby controlled machines "
                "for a processed real PSS/E case. Use this before choosing or claiming "
                "the POC bus when labels such as POC2, POC2_0, or DUMMY are ambiguous."
            ),
            parameters=_real_poc_context_schema(),
        ),
        ToolDefinition(
            name="inspect_real_network_neighborhood",
            group="real_data",
            description=(
                "Inspect a bounded topology neighborhood around one real PSS/E bus, "
                "including bus voltages, adjacent branches, transformers, and top loading. "
                "Read-only; it does not run contingency or a new load flow."
            ),
            parameters=_real_network_neighborhood_schema(),
        ),
        ToolDefinition(
            name="inspect_real_model_inventory",
            group="real_data",
            description=(
                "Inspect processed machine and dynamic-model inventory for a real PSS/E case, "
                "including model categories, model names, POC-related machines, and limitations."
            ),
            parameters=_real_model_inventory_schema(),
        ),
        ToolDefinition(
            name="inspect_real_static_operating_point",
            group="real_data",
            description=(
                "Inspect processed static operating-point evidence: convergence, voltage range, "
                "POC P/Q, bus counts, and overload rows. Read-only; no new PSS/E solve."
            ),
            parameters=_real_static_operating_point_schema(),
        ),
        ToolDefinition(
            name="inspect_real_dynamic_channels",
            group="real_data",
            description=(
                "Inspect processed no-disturbance RMS dynamic channel evidence, including "
                "channel names, final POC P/Q/V/frequency values, extrema, and row count."
            ),
            parameters=_real_dynamic_channels_schema(),
        ),
        ToolDefinition(
            name="list_remote_psse_m1m2_cases",
            group="real_data",
            description=(
                "List live remote PSS/E M1+M2 cases and allowlisted scenarios "
                "available through the TCP/IP Windows worker. This does not load "
                "SAV/DYR/DLL files on the GPU side."
            ),
            parameters=_remote_psse_list_schema(),
        ),
        ToolDefinition(
            name="run_remote_psse_m1m2",
            group="real_data",
            description=(
                "Submit one allowlisted live PSS/E job to the remote Windows worker "
                "and return a compact M1 load-flow plus M2 RMS-dynamic assessment. "
                "Use this for the real M1+M2 gym over TCP/IP, not for arbitrary "
                "new interconnection projects, faults, or controller edits."
            ),
            parameters=_remote_psse_m1m2_schema(),
        ),
        ToolDefinition(
            name="run_cia_with_mitigation",
            group="assessment",
            description="Run CIA with traceable mitigation options. Roadmap placeholder.",
            parameters=_cia_schema(include_mitigations=True),
            implemented=False,
        ),
        ToolDefinition(
            name="find_max_capacity",
            group="capacity_search",
            description=(
                "Binary-search the maximum MW at a bus that remains approved by the CIA pipeline."
            ),
            parameters=_capacity_schema(),
        ),
        ToolDefinition(
            name="query_network_data",
            group="topology",
            description="Return read-only topology/equipment data without claiming a solved operating point.",
            parameters=_case_schema(
                extra={
                    "max_rows": {
                        "type": "integer",
                        "default": 50,
                        "description": "Maximum rows per equipment table; -1 returns all.",
                    }
                }
            ),
        ),
    ]
    return {definition.name: definition for definition in definitions}


def _case_schema(extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    properties = {
        "case_path": {
            "type": "string",
            "description": "Case name/path. Built-in aliases include ieee14, ieee30, ieee57, ieee118.",
        }
    }
    properties.update(extra or {})
    return _object_schema(properties, required=["case_path"])


def _case_profile_schema(extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    properties = {
        "case_path": {
            "type": "string",
            "description": "Case name/path. Built-in aliases include ieee14, ieee30, ieee57, ieee118.",
        },
        "profile": {
            "type": "string",
            "enum": ["normal", "emergency"],
            "default": "normal",
            "description": "Screening limit profile.",
        },
        "angle_limit_degree": {
            "type": "number",
            "description": "Optional angle-difference limit. Omit to keep angle screening disabled.",
        },
        "max_violations": {
            "type": "integer",
            "default": 50,
            "description": "Maximum violation rows to return; -1 returns all.",
        },
    }
    properties.update(extra or {})
    return _object_schema(properties, required=["case_path"])


def _contingency_schema() -> Dict[str, Any]:
    return _case_profile_schema(
        extra={
            "profile": {
                "type": "string",
                "enum": ["normal", "emergency"],
                "default": "emergency",
                "description": "Contingency screening limit profile. Defaults to emergency.",
            },
            "max_contingencies": {
                "type": "integer",
                "default": -1,
                "description": "Maximum outage cases to evaluate; -1 evaluates all candidates.",
            },
            "max_failed_contingencies": {
                "type": "integer",
                "default": 50,
                "description": "Maximum failed contingency rows to return; -1 returns all.",
            },
            "include_contingency_results": {
                "type": "boolean",
                "default": False,
                "description": "Whether to include every solved outage result in the tool output.",
            },
        }
    )


def _connection_schema(
    *,
    enum_values: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    values = enum_values or ["load", "solar", "wind", "bess", "hybrid", "synchronous"]
    schema = {
        "type": "object",
        "properties": {
            "bus": {"type": "integer"},
            "p_mw": {"type": "number"},
            "connection_type": {
                "type": "string",
                "enum": values,
            },
            "is_ibr": {"type": "boolean"},
            "q_mvar": {"type": "number"},
            "name": {"type": "string"},
            "vm_pu": {"type": "number"},
        },
        "required": ["bus", "p_mw", "connection_type", "is_ibr"],
        "additionalProperties": False,
    }
    if description:
        schema["description"] = description
    return schema


def _transient_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "case_path": {
                "type": "string",
                "description": "Dynamic case alias. Supported aliases include kundur_full, kundur, ieee14_dynamic, ieee14, ieee118_dynamic, ieee118_public, and ieee118. IEEE 118 uses public benchmark RAW+DYR data resolved from local override paths, bundled GitHub data, or powerfulcases when available.",
            },
            "disturbance": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["bus_fault", "line_trip"],
                    },
                    "bus": {
                        "type": "integer",
                        "description": "Faulted bus for bus_fault disturbances.",
                    },
                    "fault_start_s": {
                        "type": "number",
                        "description": "Absolute fault start time in seconds.",
                    },
                    "clearing_time_s": {
                        "type": "number",
                        "description": "Absolute fault clearing timestamp in seconds. If a fault starts at 1.0 s and clears after 100 ms, use 1.1.",
                    },
                    "fault_r_pu": {
                        "type": "number",
                        "description": "Optional fault resistance in system p.u.",
                    },
                    "fault_x_pu": {
                        "type": "number",
                        "description": "Optional fault reactance in system p.u.",
                    },
                    "model": {
                        "type": "string",
                        "description": "ANDES model name to toggle for line_trip, usually Line.",
                    },
                    "device": {
                        "type": "string",
                        "description": "ANDES device id to toggle for line_trip, e.g. Line_5.",
                    },
                    "trip_time_s": {
                        "type": "number",
                        "description": "Line-trip time in seconds.",
                    },
                    "reclose_time_s": {
                        "type": "number",
                        "description": "Optional second Toggle time for reclosing.",
                    },
                },
                "required": ["type"],
                "additionalProperties": False,
            },
            "connection": _connection_schema(
                enum_values=[
                    "load",
                    "solar",
                    "wind",
                    "bess",
                    "generator",
                    "hybrid",
                    "synchronous",
                ],
                description=(
                    "Optional proposed interconnection to apply inside the ANDES "
                    "dynamic case as a static PQ load/injection before PFlow and TDS."
                ),
            ),
            "simulation_time_s": {"type": "number", "default": 5.0},
            "max_angle_spread_rad": {
                "type": "number",
                "default": 6.283185307179586,
            },
            "max_speed_deviation_pu": {"type": "number", "default": 0.05},
            "min_voltage_pu": {"type": "number", "default": 0.7},
            "max_samples": {"type": "integer", "default": 200},
        },
        required=["case_path", "disturbance"],
    )


def _integrated_assessment_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "case_path": {
                "type": "string",
                "description": "M1 steady-state study case, e.g. ieee14 or ieee118.",
            },
            "connection": {
                **_connection_schema(),
            },
            "transient": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to run the M2 transient stage.",
                    },
                    "required_for_approval": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether incomplete/unavailable M2 should prevent a complete approval.",
                    },
                    "case_path": {
                        "type": "string",
                        "description": "M2 dynamic case alias. Defaults to the M1 case_path when omitted. IEEE 118 integrated transient requests use strict public RAW-derived M1/M2 case alignment when public RAW+DYR data is available.",
                    },
                    "disturbance": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["bus_fault", "line_trip"],
                            },
                            "bus": {"type": "integer"},
                            "fault_start_s": {"type": "number"},
                            "clearing_time_s": {
                                "type": "number",
                                "description": "Absolute fault clearing timestamp in seconds. If a fault starts at 1.0 s and clears after 100 ms, use 1.1.",
                            },
                            "fault_r_pu": {"type": "number"},
                            "fault_x_pu": {"type": "number"},
                            "model": {"type": "string"},
                            "device": {"type": "string"},
                            "trip_time_s": {"type": "number"},
                            "reclose_time_s": {"type": "number"},
                        },
                        "required": ["type"],
                        "additionalProperties": False,
                    },
                    "simulation_time_s": {"type": "number", "default": 5.0},
                    "max_angle_spread_rad": {
                        "type": "number",
                        "default": 6.283185307179586,
                    },
                    "max_speed_deviation_pu": {"type": "number", "default": 0.05},
                    "min_voltage_pu": {"type": "number", "default": 0.7},
                    "max_samples": {"type": "integer", "default": 200},
                },
                "required": [],
                "additionalProperties": False,
            },
            "emt": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to run the F4 EMT/SCR screening proxy.",
                    },
                    "required_for_approval": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether incomplete/unavailable EMT/SCR screening should prevent complete approval.",
                    },
                    "case_path": {
                        "type": "string",
                        "description": "Steady-state case for SCR screening. Defaults to the top-level case_path.",
                    },
                    "scr_threshold": {
                        "type": "number",
                        "default": 3.0,
                        "description": "Minimum SCR threshold. Default 3.0 follows the Grid-Mind screening criterion.",
                    },
                    "borderline_margin": {
                        "type": "number",
                        "default": 0.25,
                        "description": "SCR margin above the threshold treated as borderline instead of clear pass.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            "enable_contingency": {
                "type": "boolean",
                "default": False,
                "description": "Whether M1 should include N-1 contingency screening.",
            },
            "profile": {
                "type": "string",
                "enum": ["normal", "emergency"],
                "default": "normal",
            },
            "angle_limit_degree": {
                "type": "number",
                "description": "Optional M1 angle-difference limit.",
            },
            "material_worsening_threshold_percent": {"type": "number", "default": 2.0},
            "fail_on_contingency_material_worsening": {"type": "boolean", "default": False},
            "max_contingencies": {"type": "integer", "default": -1},
            "max_failed_contingencies": {"type": "integer", "default": 50},
            "max_violations": {"type": "integer", "default": 50},
        },
        required=["case_path", "connection"],
    )


def _real_psse_list_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "processed_dir": {
                "type": "string",
                "description": (
                    "Optional directory containing processed PSSE JSON/CSV outputs. "
                    "Defaults to POWERGYM_REAL_DATA_DIR or real-data-new/processed_file."
                ),
            }
        },
        required=[],
    )


def _real_psse_assessment_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "case_id": {
                "type": "string",
                "enum": [
                    "test_cases_v36_no_disturbance",
                    "test_cases_v36_pq_target",
                    "pif6_baseline",
                ],
                "description": "Frozen real-data PSSE case id to assess.",
            },
            "processed_dir": {
                "type": "string",
                "description": (
                    "Optional directory containing processed PSSE JSON/CSV outputs. "
                    "Defaults to POWERGYM_REAL_DATA_DIR or real-data-new/processed_file."
                ),
            },
            "include_channel_preview": {
                "type": "boolean",
                "default": False,
                "description": "Whether to include a bounded preview of dynamic channel rows.",
            },
            "max_channel_samples": {
                "type": "integer",
                "default": 20,
                "description": "Maximum channel rows to include when include_channel_preview is true.",
            },
        },
        required=["case_id"],
    )


def _real_interconnection_list_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "case_id": {
                "type": "string",
                "default": "pif6_2026_05_17",
                "description": "Inventory case id. v1 default is pif6_2026_05_17.",
            },
            "processed_dir": {
                "type": "string",
                "description": (
                    "Optional frozen interconnection directory containing "
                    "scenario_catalog.jsonl, index.json, and results/*.json."
                ),
            },
            "inventory_dir": {
                "type": "string",
                "description": (
                    "Optional processed PSSE inventory directory containing "
                    "inventories/<case_id>/*.csv. Defaults to POWERGYM_REAL_DATA_DIR "
                    "or real-data-new/processed_file."
                ),
            },
        },
        required=[],
    )


def _real_interconnection_assessment_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "case_id": {
                "type": "string",
                "description": "Frozen real interconnection case id, usually pif6_2026_05_17.",
            },
            "connection": {
                "type": "object",
                "properties": {
                    "bus": {
                        "type": "integer",
                        "description": "Candidate POC/interconnection bus, e.g. 2 for POC2.",
                    },
                    "connection_type": {
                        "type": "string",
                        "enum": [
                            "solar_static_pq",
                            "wind_static_pq",
                            "bess_static_pq",
                            "load_static_pq",
                            "solar",
                            "wind",
                            "bess",
                            "load",
                            "data_center",
                        ],
                    },
                    "p_mw": {
                        "type": "number",
                        "description": "Project MW magnitude from the precomputed action grid.",
                    },
                    "q_mode": {
                        "type": "string",
                        "enum": [
                            "unity_pf",
                            "pf_0_95_lag",
                            "pf_0_95_lead",
                            "unity",
                            "0.95_lag",
                            "0.95_lead",
                        ],
                        "default": "unity_pf",
                    },
                    "q_mvar": {
                        "type": "number",
                        "description": "Optional explicit Q. If omitted, q_mode determines Q.",
                    },
                    "bess_mode": {
                        "type": "string",
                        "enum": ["charge", "discharge"],
                        "description": "Required only when selecting a BESS charge scenario.",
                    },
                },
                "required": ["bus", "connection_type", "p_mw"],
                "additionalProperties": False,
            },
            "disturbance": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["no_disturbance", "no_disturbance_5s", "bus_fault", "line_trip"],
                    },
                    "bus": {
                        "type": "integer",
                        "description": "Faulted bus for bus_fault disturbances.",
                    },
                    "fault_start_s": {
                        "type": "number",
                        "default": 1.0,
                        "description": "Absolute fault start time in seconds.",
                    },
                    "clearing_time_s": {
                        "type": "number",
                        "description": "Absolute fault clearing timestamp, e.g. 1.1 for 100 ms after 1.0 s.",
                    },
                    "from_bus": {
                        "type": "integer",
                        "description": "From bus for line_trip disturbances.",
                    },
                    "to_bus": {
                        "type": "integer",
                        "description": "To bus for line_trip disturbances.",
                    },
                    "circuit_id": {
                        "type": "string",
                        "default": "1",
                    },
                    "trip_time_s": {
                        "type": "number",
                        "default": 1.0,
                    },
                    "simulation_time_s": {
                        "type": "number",
                        "default": 5.0,
                    },
                },
                "required": ["type"],
                "additionalProperties": False,
            },
            "processed_dir": {
                "type": "string",
                "description": (
                    "Optional frozen interconnection directory containing "
                    "scenario_catalog.jsonl, index.json, and results/*.json."
                ),
            },
        },
        required=["case_id", "connection", "disturbance"],
    )


def _real_case_id_property() -> Dict[str, Any]:
    return {
        "type": "string",
        "enum": ["pif6_2026_05_17", "test_cases_v36"],
        "description": "Processed real PSS/E case package id.",
    }


def _processed_dir_property() -> Dict[str, Any]:
    return {
        "type": "string",
        "description": (
            "Optional processed PSSE artifact directory. Defaults to "
            "POWERGYM_REAL_DATA_DIR or real-data-new/processed_file."
        ),
    }


def _real_case_summary_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "case_id": _real_case_id_property(),
            "processed_dir": _processed_dir_property(),
            "max_poc_candidates": {
                "type": "integer",
                "default": 8,
                "description": "Maximum POC candidate rows to return.",
            },
        },
        required=["case_id"],
    )


def _real_poc_context_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "case_id": _real_case_id_property(),
            "poc_label_or_bus": {
                "type": "string",
                "description": "Optional POC label or bus selector, e.g. POC2 or 2.",
            },
            "processed_dir": _processed_dir_property(),
            "max_candidates": {"type": "integer", "default": 12},
            "max_branches": {"type": "integer", "default": 12},
        },
        required=["case_id"],
    )


def _real_network_neighborhood_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "case_id": _real_case_id_property(),
            "bus": {
                "type": "integer",
                "description": "Real PSS/E bus number to inspect.",
            },
            "depth": {
                "type": "integer",
                "default": 1,
                "description": "Graph depth, clamped to 1..3.",
            },
            "max_rows": {"type": "integer", "default": 20},
            "processed_dir": _processed_dir_property(),
        },
        required=["case_id", "bus"],
    )


def _real_model_inventory_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "case_id": _real_case_id_property(),
            "model_scope": {
                "type": "string",
                "enum": ["summary", "poc", "machines", "controllers", "dynamic", "all"],
                "default": "summary",
            },
            "max_rows": {"type": "integer", "default": 20},
            "processed_dir": _processed_dir_property(),
        },
        required=["case_id"],
    )


def _real_static_operating_point_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "case_id": _real_case_id_property(),
            "processed_dir": _processed_dir_property(),
            "max_voltage_rows": {"type": "integer", "default": 6},
            "max_overload_rows": {"type": "integer", "default": 6},
        },
        required=["case_id"],
    )


def _real_dynamic_channels_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "case_id": _real_case_id_property(),
            "scenario_type": {
                "type": "string",
                "enum": ["no_disturbance_5s", "no_disturbance", "baseline"],
                "default": "no_disturbance_5s",
            },
            "processed_dir": _processed_dir_property(),
            "max_samples": {"type": "integer", "default": 5},
        },
        required=["case_id"],
    )


def _remote_psse_list_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "check_health": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Whether to call the configured remote worker health endpoint. "
                    "The worker endpoint itself is read from PSSE_REMOTE_BASE_URL."
                ),
            }
        },
        required=[],
    )


def _remote_psse_m1m2_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "case_id": {
                "type": "string",
                "enum": ["test_cases_v36", "pif6_2026_05_17"],
                "description": "Allowlisted remote PSS/E case id.",
            },
            "scenario_type": {
                "type": "string",
                "enum": [
                    "static",
                    "static_pf",
                    "load_flow",
                    "powerflow",
                    "no_disturbance_5s",
                    "no_disturbance",
                    "baseline",
                    "baseline_5s",
                    "none",
                    "pq_target_step",
                    "pq_target",
                    "p_q_target",
                ],
                "description": (
                    "Remote PSS/E scenario or supported alias. PIF6 currently "
                    "supports static and no_disturbance_5s only; unsupported "
                    "case/scenario pairs fail closed before network submission."
                ),
            },
            "request_id": {
                "type": "string",
                "description": "Optional caller request id for tracing/idempotency in the GPU-side gym.",
            },
            "step_index": {
                "type": "integer",
                "description": "Optional multi-turn gym step index tracked by the GPU-side VERL wrapper.",
            },
            "expected_state_id": {
                "type": "string",
                "description": "Optional stale-state guard checked by the GPU-side VERL wrapper.",
            },
            "include_artifacts": {
                "type": "boolean",
                "default": False,
                "description": "Whether to fetch the remote artifact manifest after the job completes.",
            },
        },
        required=["case_id", "scenario_type"],
    )


def _emt_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "case_path": {
                "type": "string",
                "description": "Steady-state case name/path for the SCR screen, e.g. ieee14.",
            },
            "connection": _connection_schema(
                description=(
                    "Proposed interconnection. EMT/SCR screening is relevant for "
                    "IBR resources such as solar, wind, BESS, and hybrid projects."
                )
            ),
            "scr_threshold": {
                "type": "number",
                "default": 3.0,
                "description": "Minimum short-circuit ratio threshold. Grid-Mind-style default is 3.0.",
            },
            "borderline_margin": {
                "type": "number",
                "default": 0.25,
                "description": "SCR margin above the threshold treated as borderline instead of clear pass.",
            },
        },
        required=["case_path", "connection"],
    )


def _cia_schema(include_mitigations: bool = False) -> Dict[str, Any]:
    properties: Dict[str, Any] = {
        "case_path": {"type": "string"},
        "connection": {
            **_connection_schema(),
        },
        "enable_contingency": {
            "type": "boolean",
            "description": "Whether to enable N-1 screening in the later CIA pipeline.",
        },
        "enable_transient": {
            "type": "boolean",
            "description": "Whether to enable transient stability screening for IBR requests.",
        },
        "enable_emt": {
            "type": "boolean",
            "description": "Whether to enable EMT/SCR screening for IBR requests.",
        },
        "profile": {
            "type": "string",
            "enum": ["normal", "emergency"],
            "default": "normal",
            "description": "Screening limit profile for the steady-state stage.",
        },
        "angle_limit_degree": {
            "type": "number",
            "description": "Optional angle-difference limit. Omit to keep angle screening disabled.",
        },
        "material_worsening_threshold_percent": {
            "type": "number",
            "default": 2.0,
            "description": "Margin-percent delta that flags material worsening of pre-existing violations.",
        },
        "fail_on_contingency_material_worsening": {
            "type": "boolean",
            "default": False,
            "description": "Whether f2 should fail on material worsening of pre-existing N-1 failures.",
        },
        "max_contingencies": {
            "type": "integer",
            "default": -1,
            "description": "Maximum N-1 outage cases to evaluate when contingency is enabled.",
        },
        "max_failed_contingencies": {
            "type": "integer",
            "default": 50,
            "description": "Maximum failed N-1 contingency rows to return.",
        },
        "max_violations": {
            "type": "integer",
            "default": 50,
            "description": "Maximum violation rows to return per report; -1 returns all.",
        },
    }
    required = ["case_path", "connection"]
    if include_mitigations:
        properties["mitigations"] = {"type": "array", "items": {"type": "object"}}
        required.append("mitigations")
    return _object_schema(properties, required=required)


def _capacity_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "case_path": {"type": "string"},
            "bus": {"type": "integer"},
            "connection_type": {
                "type": "string",
                "enum": ["load", "solar", "wind", "bess", "hybrid", "synchronous"],
            },
            "is_ibr": {
                "type": "boolean",
                "description": "Override automatic IBR classification from connection_type.",
            },
            "q_mvar": {"type": "number", "default": 0.0},
            "vm_pu": {"type": "number", "default": 1.0},
            "min_mw": {"type": "number", "default": 0.0},
            "max_mw": {"type": "number", "default": 500.0},
            "tolerance_mw": {"type": "number", "default": 1.0},
            "max_iterations": {"type": "integer", "default": 12},
            "coarse_scan_points": {"type": "integer", "default": 11},
            "enable_contingency": {
                "type": "boolean",
                "default": False,
                "description": "Whether each sampled CIA should include f2 N-1 screening.",
            },
            "max_contingencies": {
                "type": "integer",
                "default": -1,
                "description": "Maximum N-1 outage cases per sampled CIA when enabled.",
            },
            "max_failed_contingencies": {"type": "integer", "default": 50},
            "material_worsening_threshold_percent": {"type": "number", "default": 2.0},
            "fail_on_contingency_material_worsening": {"type": "boolean", "default": False},
            "profile": {
                "type": "string",
                "enum": ["normal", "emergency"],
                "default": "normal",
            },
            "angle_limit_degree": {
                "type": "number",
                "description": "Optional f1 angle-difference limit for each sampled CIA.",
            },
            "max_violations": {"type": "integer", "default": 10},
            "max_samples": {"type": "integer", "default": 100},
            "include_reports": {
                "type": "boolean",
                "default": False,
                "description": "Whether each sample should include its full CIA report.",
            },
        },
        required=["case_path", "bus", "connection_type"],
    )


def _object_schema(properties: Mapping[str, Any], required: Iterable[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _case_name_from_args(args: Mapping[str, Any]) -> str:
    case_name = args.get("case_path", args.get("case_name"))
    if not isinstance(case_name, str) or not case_name.strip():
        raise ToolRegistryError("Tool argument 'case_path' is required")
    return case_name


def _limit_profile_from_args(
    args: Mapping[str, Any],
    default_profile: str = "normal",
) -> LimitProfile:
    try:
        profile = profile_from_name(str(args.get("profile", default_profile)))
    except ValueError as exc:
        raise ToolRegistryError(str(exc)) from exc
    angle_limit = args.get("angle_limit_degree")
    if angle_limit is not None:
        angle_limit_degree = _numeric_argument(angle_limit, "angle_limit_degree")
        if angle_limit_degree < 0:
            raise ToolRegistryError("Tool argument 'angle_limit_degree' must be non-negative")
        profile = replace(profile, angle_diff_limit_degree=angle_limit_degree)
    return profile


def _required_str(args: Mapping[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolRegistryError(f"Tool argument '{key}' is required")
    return value


def _mapping_arg(args: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = args.get(key)
    if not isinstance(value, Mapping):
        raise ToolRegistryError(f"Tool argument '{key}' must be an object")
    return value


def _optional_int(args: Mapping[str, Any], key: str, default: int) -> int:
    value = args.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolRegistryError(f"Tool argument '{key}' must be an integer")
    return value


def _optional_float(args: Mapping[str, Any], key: str, default: float) -> float:
    value = args.get(key, default)
    result = _numeric_argument(value, key)
    if result < 0:
        raise ToolRegistryError(f"Tool argument '{key}' must be non-negative")
    return result


def _positive_float(args: Mapping[str, Any], key: str, default: float) -> float:
    result = _numeric_argument(args.get(key, default), key)
    if result <= 0:
        raise ToolRegistryError(f"Tool argument '{key}' must be positive")
    return result


def _positive_int(args: Mapping[str, Any], key: str, default: int) -> int:
    value = _optional_int(args, key, default)
    if value <= 0:
        raise ToolRegistryError(f"Tool argument '{key}' must be positive")
    return value


def _optional_number(args: Mapping[str, Any], key: str, default: float) -> float:
    return _numeric_argument(args.get(key, default), key)


def _numeric_argument(value: Any, key: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ToolRegistryError(f"Tool argument '{key}' must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ToolRegistryError(f"Tool argument '{key}' must be finite")
    return result


def _limit_rows(rows: List[Dict[str, Any]], max_rows: int) -> Dict[str, Any]:
    limited = rows if max_rows < 0 else rows[:max_rows]
    return {
        "rows": limited,
        "total_rows": len(rows),
        "truncated_rows": max(0, len(rows) - len(limited)),
    }


def _limit_violation_report(report: Dict[str, Any], max_violations: int) -> Dict[str, Any]:
    violations = list(report.get("violations", []))
    limited = violations if max_violations < 0 else violations[:max_violations]
    report["violations"] = limited
    report["truncated_violations"] = max(0, len(violations) - len(limited))
    return report


def _validate_json_args(schema: Mapping[str, Any], args: Any, path: str) -> None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(args, Mapping):
            raise ToolRegistryError(f"{path} arguments must be a JSON object")

        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in args:
                raise ToolRegistryError(f"Missing required argument '{path}.{key}'")

        if schema.get("additionalProperties") is False:
            extras = sorted(set(args) - set(properties))
            if extras:
                raise ToolRegistryError(
                    f"Unexpected argument(s) for {path}: {', '.join(extras)}"
                )

        for key, value in args.items():
            if key in properties:
                _validate_json_args(properties[key], value, f"{path}.{key}")
        return

    if expected_type == "array":
        if not isinstance(args, list):
            raise ToolRegistryError(f"Argument '{path}' must be an array")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(args):
                _validate_json_args(item_schema, item, f"{path}[{index}]")
        return

    if "enum" in schema and args not in schema["enum"]:
        allowed = ", ".join(str(item) for item in schema["enum"])
        raise ToolRegistryError(f"Argument '{path}' must be one of: {allowed}")

    if expected_type == "string" and not isinstance(args, str):
        raise ToolRegistryError(f"Argument '{path}' must be a string")
    if expected_type == "integer" and (not isinstance(args, int) or isinstance(args, bool)):
        raise ToolRegistryError(f"Argument '{path}' must be an integer")
    if expected_type == "number" and (
        not isinstance(args, (int, float)) or isinstance(args, bool)
    ):
        raise ToolRegistryError(f"Argument '{path}' must be numeric")
    if expected_type == "boolean" and not isinstance(args, bool):
        raise ToolRegistryError(f"Argument '{path}' must be a boolean")

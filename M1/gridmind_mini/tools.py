"""Tool registry for the Mini Grid-Mind reproduction.

The registry is the boundary the future LLM agent will see. It exposes
solver-backed operations through JSON-schema tool definitions and keeps
unimplemented Grid-Mind tools explicit rather than silently faking behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .capacity import CapacitySearchRunner
from .cia import ConnectionRequest, SteadyStateCIARunner
from .contingency import ContingencyRunner
from .memory import StudyMemoryStore
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
            "run_cia": self._tool_run_cia,
            "find_max_capacity": self._tool_find_max_capacity,
            "query_network_data": self._tool_query_network_data,
        }
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
            name="run_cia",
            group="assessment",
            description=(
                "Run baseline-aware CIA with steady-state screening and optional N-1 "
                "contingency screening. Dynamic stages remain explicit placeholders."
            ),
            parameters=_cia_schema(),
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


def _cia_schema(include_mitigations: bool = False) -> Dict[str, Any]:
    properties: Dict[str, Any] = {
        "case_path": {"type": "string"},
        "connection": {
            "type": "object",
            "properties": {
                "bus": {"type": "integer"},
                "p_mw": {"type": "number"},
                "connection_type": {
                    "type": "string",
                    "enum": ["load", "solar", "wind", "bess", "hybrid", "synchronous"],
                },
                "is_ibr": {"type": "boolean"},
                "q_mvar": {"type": "number"},
                "name": {"type": "string"},
                "vm_pu": {"type": "number"},
            },
            "required": ["bus", "p_mw", "connection_type", "is_ibr"],
            "additionalProperties": False,
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

"""ANDES-backed transient-stability tools for the M2 layer.

This module is intentionally separate from the pandapower ``GridSolver``
interface. M1 remains a steady-state screening stack, while M2 uses ANDES for
time-domain simulation on cases that include dynamic data.
"""

from __future__ import annotations

import importlib
import contextlib
import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .public_cases import (
    IEEE118_PUBLIC_LIMITATION,
    IEEE118_PUBLIC_SOURCE,
    PublicCaseDataError,
    PublicDynamicCaseFiles,
    resolve_ieee118_public_dynamic_files,
)


DEFAULT_MAX_ANGLE_SPREAD_RAD = 2.0 * math.pi
DEFAULT_MAX_SPEED_DEVIATION_PU = 0.05
DEFAULT_MIN_VOLTAGE_PU = 0.7
DEFAULT_SYSTEM_BASE_MVA = 100.0
M2_LOAD_CONNECTION_TYPES = {"load"}
M2_GENERATION_CONNECTION_TYPES = {
    "solar",
    "wind",
    "bess",
    "generator",
    "synchronous",
    "hybrid",
}
M2_SUPPORTED_CONNECTION_TYPES = (
    M2_LOAD_CONNECTION_TYPES | M2_GENERATION_CONNECTION_TYPES
)


class DynamicSimulationError(RuntimeError):
    """Raised for invalid M2 transient-stability inputs."""


@dataclass(frozen=True)
class DynamicCaseDefinition:
    """Metadata for one ANDES dynamic case."""

    case_id: str
    display_name: str
    source_type: str
    case_locator: str
    addfile_locator: Optional[str]
    aliases: Tuple[str, ...]
    supported_disturbance_types: Tuple[str, ...]
    dynamic_data_available: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "display_name": self.display_name,
            "source_type": self.source_type,
            "case_locator": self.case_locator,
            "addfile_locator": self.addfile_locator,
            "aliases": list(self.aliases),
            "dynamic_data_available": self.dynamic_data_available,
            "supported_disturbance_types": list(self.supported_disturbance_types),
            "description": self.description,
        }


@dataclass(frozen=True)
class TimeSeries:
    """Small dependency-free representation of a sampled time series."""

    time_s: Tuple[float, ...]
    columns: Tuple[str, ...]
    values: Tuple[Tuple[float, ...], ...]

    @property
    def row_count(self) -> int:
        return len(self.values)

    @property
    def column_count(self) -> int:
        return len(self.columns)


DYNAMIC_CASES: Tuple[DynamicCaseDefinition, ...] = (
    DynamicCaseDefinition(
        case_id="kundur_full",
        display_name="Kundur two-area system",
        source_type="andes_builtin_xlsx",
        case_locator="kundur/kundur_full.xlsx",
        addfile_locator=None,
        aliases=("kundur_full", "kundur", "kundur/kundur_full.xlsx"),
        supported_disturbance_types=("bus_fault", "line_trip"),
        description="ANDES built-in Kundur dynamic case with synchronous-generator models.",
    ),
    DynamicCaseDefinition(
        case_id="ieee14_dynamic",
        display_name="IEEE 14-bus dynamic case",
        source_type="andes_builtin_raw_dyr",
        case_locator="ieee14/ieee14.raw",
        addfile_locator="ieee14/ieee14.dyr",
        aliases=("ieee14_dynamic", "ieee14", "ieee-14", "case14", "case-14"),
        supported_disturbance_types=("bus_fault", "line_trip"),
        description="ANDES built-in IEEE 14 RAW+DYR case.",
    ),
    DynamicCaseDefinition(
        case_id="ieee118_public_dynamic",
        display_name="IEEE 118-bus public dynamic case",
        source_type="public_raw_dyr",
        case_locator="local override, bundled public data, or powerfulcases:ieee118.raw",
        addfile_locator="local override, bundled public data, or powerfulcases:ieee118.dyr:genrou",
        aliases=(
            "ieee118_public_dynamic",
            "ieee118_dynamic",
            "ieee118_public",
            "ieee118",
            "ieee-118",
            "case118",
            "case-118",
            "118",
        ),
        supported_disturbance_types=("bus_fault", "line_trip"),
        description=(
            "Public IEEE 118 RAW+DYR dynamic benchmark resolved at runtime "
            "from local override paths first, then bundled public data, then "
            "powerfulcases; not customer-validated data."
        ),
    ),
)

UNSUPPORTED_DYNAMIC_CASES = {
}


def list_dynamic_cases() -> Dict[str, Any]:
    """Return M2 dynamic-case metadata without requiring ANDES to be installed."""

    return {
        "ok": True,
        "tool": "list_dynamic_cases",
        "backend": "andes",
        "andes_available": is_andes_available(),
        "cases": [case.to_dict() for case in DYNAMIC_CASES],
        "unsupported_cases": [
            {
                "case_path": case_path,
                "dynamic_data_available": False,
                "reason": reason,
            }
            for case_path, reason in sorted(UNSUPPORTED_DYNAMIC_CASES.items())
        ],
    }


def is_andes_available() -> bool:
    """Return whether the ANDES package can be imported."""

    try:
        importlib.import_module("andes")
    except ImportError:
        return False
    return True


def resolve_dynamic_case(case_path: str) -> DynamicCaseDefinition:
    """Resolve a public case alias to a supported dynamic case."""

    key = _normalize_case_alias(case_path)
    for case in DYNAMIC_CASES:
        if key == _normalize_case_alias(case.case_id):
            return case
        if key in {_normalize_case_alias(alias) for alias in case.aliases}:
            return case
    raise DynamicSimulationError(_unsupported_case_message(case_path))


def is_unsupported_dynamic_case(case_path: str) -> bool:
    return _normalize_case_alias(case_path) in {
        _normalize_case_alias(alias) for alias in UNSUPPORTED_DYNAMIC_CASES
    }


def unsupported_dynamic_case_result(case_path: str) -> Dict[str, Any]:
    """Return the structured result for a known case with missing dynamics."""

    return {
        "ok": False,
        "tool": "run_transient_stability",
        "backend": "andes",
        "case_path": case_path,
        "case_info": {
            "case_id": _normalize_case_alias(case_path),
            "dynamic_data_available": False,
        },
        "connection_model": None,
        "connection_application": _connection_not_requested(),
        "dynamic_interconnection_modeling": False,
        "limitations": ["dynamic_data_unavailable"],
        "stability": {"status": "error"},
        "metrics": {
            "tds_converged": False,
            "max_angle_spread_rad": None,
            "final_angle_spread_rad": None,
            "max_speed_deviation_pu": None,
            "min_voltage_pu": None,
            "reason_codes": ["dynamic_data_unavailable"],
        },
        "error_type": "dynamic_data_unavailable",
        "message": _unsupported_case_message(case_path),
        "recommended_action": "Provide a validated RAW+DYR dynamic model, or use one of list_dynamic_cases().",
        "reason_codes": ["dynamic_data_unavailable"],
    }


class TransientStabilityRunner:
    """Run ANDES time-domain simulation and extract compact stability metrics."""

    def __init__(self, andes_module: Optional[Any] = None) -> None:
        self._andes_module = andes_module

    def run(
        self,
        *,
        case_path: str,
        disturbance: Mapping[str, Any],
        connection: Optional[Any] = None,
        simulation_time_s: float = 5.0,
        max_angle_spread_rad: float = DEFAULT_MAX_ANGLE_SPREAD_RAD,
        max_speed_deviation_pu: float = DEFAULT_MAX_SPEED_DEVIATION_PU,
        min_voltage_pu: float = DEFAULT_MIN_VOLTAGE_PU,
        max_samples: int = 200,
        public_case_files: Optional[PublicDynamicCaseFiles] = None,
    ) -> Dict[str, Any]:
        """Run transient stability and return a bounded, agent-facing result."""

        if not isinstance(case_path, str) or not case_path.strip():
            raise DynamicSimulationError("case_path must be a non-empty string")
        if is_unsupported_dynamic_case(case_path):
            return unsupported_dynamic_case_result(case_path)

        try:
            case = resolve_dynamic_case(case_path)
        except DynamicSimulationError:
            return {
                "ok": False,
                "tool": "run_transient_stability",
                "backend": "andes",
                "case_path": case_path,
                "stability": {"status": "error"},
                "connection_model": None,
                "connection_application": _connection_not_requested(),
                "dynamic_interconnection_modeling": False,
                "limitations": ["unsupported_dynamic_case"],
                "metrics": _empty_metrics(["unsupported_dynamic_case"]),
                "error_type": "unsupported_dynamic_case",
                "message": _unsupported_case_message(case_path),
                "reason_codes": ["unsupported_dynamic_case"],
            }

        try:
            simulation_time_s = _positive_float(
                simulation_time_s,
                "simulation_time_s",
            )
            thresholds = {
                "max_angle_spread_rad": _positive_float(
                    max_angle_spread_rad,
                    "max_angle_spread_rad",
                ),
                "max_speed_deviation_pu": _positive_float(
                    max_speed_deviation_pu,
                    "max_speed_deviation_pu",
                ),
                "min_voltage_pu": _non_negative_float(min_voltage_pu, "min_voltage_pu"),
            }
            max_samples = _positive_int(max_samples, "max_samples")
            disturbance_spec = _validate_disturbance(
                disturbance,
                simulation_time_s=simulation_time_s,
            )
        except DynamicSimulationError as exc:
            raise exc

        try:
            connection_spec = _validate_m2_connection(connection)
        except DynamicSimulationError as exc:
            return _connection_error_result(
                case=case,
                case_path=case_path,
                disturbance=disturbance_spec,
                simulation_time_s=simulation_time_s,
                thresholds=thresholds,
                load_metadata={},
                connection_model=None,
                connection_application={
                    "requested": True,
                    "applied": False,
                    "error_type": "connection_input_error",
                    "message": str(exc),
                },
                message=str(exc),
                reason_codes=["connection_input_error"],
                error_type="connection_input_error",
            )
        connection_model = _connection_model_metadata(connection_spec)
        connection_application = _connection_not_requested()
        if connection_spec is not None:
            connection_application = _connection_pending()

        andes = self._import_andes()
        if andes is None:
            return {
                "ok": False,
                "tool": "run_transient_stability",
                "backend": "andes",
                "case_path": case_path,
                "case_info": case.to_dict(),
                "disturbance": disturbance_spec,
                "simulation": {
                    "simulation_time_s": simulation_time_s,
                    "thresholds": thresholds,
                },
                "connection_model": connection_model,
                "connection_application": connection_application,
                "dynamic_interconnection_modeling": False,
                "limitations": _dynamic_limitations(connection_model, case=case),
                "stability": {"status": "error"},
                "metrics": _empty_metrics(["andes_unavailable"]),
                "error_type": "dependency_unavailable",
                "message": "ANDES is not installed in this environment. Install andes>=1.10.1,<2.0 on the runtime node.",
                "reason_codes": ["andes_unavailable"],
            }

        try:
            with contextlib.redirect_stdout(sys.stderr):
                system, load_metadata = self._load_system(
                    andes,
                    case,
                    public_case_files=public_case_files,
                )
                if connection_spec is not None:
                    connection_application = _apply_static_connection(system, connection_spec)
                _disable_existing_toggles(system)
                _add_disturbance(system, disturbance_spec)
                _setup_system(system)
                pflow_converged, pflow_diagnostics = _run_pflow_with_retries(system, case)
                if not pflow_converged:
                    return _routine_error_result(
                        case=case,
                        case_path=case_path,
                        disturbance=disturbance_spec,
                        simulation_time_s=simulation_time_s,
                        thresholds=thresholds,
                        load_metadata=load_metadata,
                        connection_model=connection_model,
                        connection_application=connection_application,
                        routine="PFlow",
                        message="ANDES power flow did not converge before TDS.",
                        reason_codes=["pflow_not_converged"],
                        error_type=_runtime_error_type(case, "simulation_error"),
                        routine_diagnostics=pflow_diagnostics,
                    )
                system.TDS.config.tf = simulation_time_s
                tds_converged = _run_routine(system.TDS, "TDS")
        except PublicCaseDataError as exc:
            return _public_case_data_error_result(
                case=case,
                case_path=case_path,
                disturbance=disturbance_spec,
                simulation_time_s=simulation_time_s,
                thresholds=thresholds,
                connection_model=connection_model,
                connection_application=connection_application,
                error_type=exc.error_type,
                message=str(exc),
            )
        except DynamicSimulationError as exc:
            return _connection_error_result(
                case=case,
                case_path=case_path,
                disturbance=disturbance_spec,
                simulation_time_s=simulation_time_s,
                thresholds=thresholds,
                load_metadata=locals().get("load_metadata", {}),
                connection_model=connection_model,
                connection_application={
                    **connection_application,
                    "requested": connection_spec is not None,
                    "applied": False,
                    "error_type": "connection_application_error",
                    "message": str(exc),
                },
                message=str(exc),
                reason_codes=["connection_application_error"],
                error_type="connection_application_error",
            )
        except Exception as exc:
            return _routine_error_result(
                case=case,
                case_path=case_path,
                disturbance=disturbance_spec,
                simulation_time_s=simulation_time_s,
                thresholds=thresholds,
                load_metadata=locals().get("load_metadata", {}),
                connection_model=connection_model,
                connection_application=connection_application,
                routine="ANDES",
                message=str(exc),
                reason_codes=["andes_runtime_error"],
                error_type=_runtime_error_type(case, type(exc).__name__),
            )

        series_bundle = _extract_timeseries_bundle(system)
        trajectories = _compact_trajectories(series_bundle, max_samples=max_samples)
        metrics = _stability_metrics(
            series_bundle,
            tds_converged=tds_converged,
            thresholds=thresholds,
        )
        status = _classify_stability(metrics)
        return {
            "ok": True,
            "tool": "run_transient_stability",
            "backend": "andes",
            "case_path": case_path,
            "case_info": {
                **case.to_dict(),
                **load_metadata,
            },
            "disturbance": disturbance_spec,
            "simulation": {
                "simulation_time_s": simulation_time_s,
                "thresholds": thresholds,
                "pflow_converged": True,
                "pflow_attempt": pflow_diagnostics.get("successful_attempt"),
                "pflow_attempts": pflow_diagnostics.get("attempts", []),
                "tds_converged": tds_converged,
                "max_samples": max_samples,
                "sample_count": len(trajectories.get("time_s", [])),
            },
            "connection_model": connection_model,
            "connection_application": connection_application,
            "dynamic_interconnection_modeling": bool(connection_application.get("applied")),
            "limitations": _dynamic_limitations(connection_model, case=case),
            "stability": {"status": status},
            "metrics": metrics,
            "trajectories": trajectories,
        }

    def _import_andes(self) -> Optional[Any]:
        if self._andes_module is not None:
            return self._andes_module
        try:
            return importlib.import_module("andes")
        except ImportError:
            return None

    def _load_system(
        self,
        andes: Any,
        case: DynamicCaseDefinition,
        *,
        public_case_files: Optional[PublicDynamicCaseFiles] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        if case.source_type in {"public_raw_dyr", "powerfulcases_raw_dyr"}:
            files = public_case_files or resolve_ieee118_public_dynamic_files()
            system = andes.load(files.raw_path, addfile=files.dyr_path, setup=False)
            return system, {
                "resolved_case_file": files.raw_path,
                "resolved_addfile": files.dyr_path,
                "case_source": files.source,
                "dyr_variant": files.dyr_variant,
                "raw_sha256": files.raw_sha256,
                "dyr_sha256": files.dyr_sha256,
                "data_label": "public_benchmark_data_not_customer_validated",
            }
        case_file = _resolve_andes_case_file(andes, case.case_locator)
        addfile = (
            _resolve_andes_case_file(andes, case.addfile_locator)
            if case.addfile_locator
            else None
        )
        if addfile is None:
            system = andes.load(case_file, setup=False)
        else:
            system = andes.load(case_file, addfile=addfile, setup=False)
        return system, {
            "resolved_case_file": str(case_file),
            "resolved_addfile": None if addfile is None else str(addfile),
        }


def _validate_disturbance(
    disturbance: Mapping[str, Any],
    *,
    simulation_time_s: float,
) -> Dict[str, Any]:
    if not isinstance(disturbance, Mapping):
        raise DynamicSimulationError("disturbance must be an object")
    dtype = disturbance.get("type")
    if not isinstance(dtype, str):
        raise DynamicSimulationError("disturbance.type is required")
    dtype = dtype.strip().lower()
    if dtype == "bus_fault":
        bus = _required_int(disturbance, "bus")
        fault_start = _non_negative_float(
            _required_number(disturbance, "fault_start_s"),
            "fault_start_s",
        )
        clearing = _non_negative_float(
            _required_number(disturbance, "clearing_time_s"),
            "clearing_time_s",
        )
        if clearing <= fault_start:
            raise DynamicSimulationError(
                "disturbance.clearing_time_s must be greater than fault_start_s"
            )
        if clearing > simulation_time_s:
            raise DynamicSimulationError(
                "disturbance.clearing_time_s must not exceed simulation_time_s"
            )
        result: Dict[str, Any] = {
            "type": "bus_fault",
            "bus": bus,
            "fault_start_s": fault_start,
            "clearing_time_s": clearing,
            "clearing_duration_s": clearing - fault_start,
        }
        if "fault_r_pu" in disturbance and disturbance.get("fault_r_pu") is not None:
            result["fault_r_pu"] = _non_negative_float(
                _required_number(disturbance, "fault_r_pu"),
                "fault_r_pu",
            )
        if "fault_x_pu" in disturbance and disturbance.get("fault_x_pu") is not None:
            result["fault_x_pu"] = _non_negative_float(
                _required_number(disturbance, "fault_x_pu"),
                "fault_x_pu",
            )
        return result

    if dtype == "line_trip":
        model = _required_string(disturbance, "model")
        device = _required_string(disturbance, "device")
        trip_time = _non_negative_float(
            _required_number(disturbance, "trip_time_s"),
            "trip_time_s",
        )
        if trip_time > simulation_time_s:
            raise DynamicSimulationError(
                "disturbance.trip_time_s must not exceed simulation_time_s"
            )
        result = {
            "type": "line_trip",
            "model": model,
            "device": device,
            "trip_time_s": trip_time,
        }
        if "reclose_time_s" in disturbance and disturbance.get("reclose_time_s") is not None:
            reclose = _non_negative_float(
                _required_number(disturbance, "reclose_time_s"),
                "reclose_time_s",
            )
            if reclose <= trip_time:
                raise DynamicSimulationError(
                    "disturbance.reclose_time_s must be greater than trip_time_s"
                )
            if reclose > simulation_time_s:
                raise DynamicSimulationError(
                    "disturbance.reclose_time_s must not exceed simulation_time_s"
                )
            result["reclose_time_s"] = reclose
        return result

    raise DynamicSimulationError(
        "disturbance.type must be one of: bus_fault, line_trip"
    )


def _add_disturbance(system: Any, disturbance: Mapping[str, Any]) -> None:
    dtype = disturbance["type"]
    if dtype == "bus_fault":
        kwargs = {
            "bus": disturbance["bus"],
            "tf": disturbance["fault_start_s"],
            "tc": disturbance["clearing_time_s"],
        }
        if "fault_r_pu" in disturbance:
            kwargs["rf"] = disturbance["fault_r_pu"]
        if "fault_x_pu" in disturbance:
            kwargs["xf"] = disturbance["fault_x_pu"]
        _andes_add(system, "Fault", kwargs)
        return
    if dtype == "line_trip":
        _andes_add(
            system,
            "Toggle",
            {
                "model": disturbance["model"],
                "dev": disturbance["device"],
                "t": disturbance["trip_time_s"],
            },
        )
        if "reclose_time_s" in disturbance:
            _andes_add(
                system,
                "Toggle",
                {
                    "model": disturbance["model"],
                    "dev": disturbance["device"],
                    "t": disturbance["reclose_time_s"],
                },
            )
        return
    raise DynamicSimulationError("unsupported disturbance type")


def _validate_m2_connection(connection: Optional[Any]) -> Optional[Dict[str, Any]]:
    if connection is None:
        return None
    if isinstance(connection, Mapping):
        payload = dict(connection)
    else:
        to_dict = getattr(connection, "to_dict", None)
        if not callable(to_dict):
            raise DynamicSimulationError("connection must be an object")
        payload = dict(to_dict())

    missing = [
        key
        for key in ("bus", "p_mw", "connection_type", "is_ibr")
        if key not in payload
    ]
    if missing:
        raise DynamicSimulationError(
            "Missing required connection field(s): " + ", ".join(missing)
        )
    bus = payload.get("bus")
    if not isinstance(bus, int) or isinstance(bus, bool):
        raise DynamicSimulationError("connection.bus must be an integer")
    p_mw = _non_negative_float(payload.get("p_mw"), "connection.p_mw")
    q_mvar = _finite_float(payload.get("q_mvar", 0.0), "connection.q_mvar")
    vm_pu = _positive_float(payload.get("vm_pu", 1.0), "connection.vm_pu")
    connection_type = payload.get("connection_type")
    if not isinstance(connection_type, str) or not connection_type.strip():
        raise DynamicSimulationError("connection.connection_type must be a non-empty string")
    connection_type = connection_type.strip().lower()
    if connection_type not in M2_SUPPORTED_CONNECTION_TYPES:
        raise DynamicSimulationError(
            "connection.connection_type must be one of: "
            + ", ".join(sorted(M2_SUPPORTED_CONNECTION_TYPES))
        )
    is_ibr = payload.get("is_ibr")
    if not isinstance(is_ibr, bool):
        raise DynamicSimulationError("connection.is_ibr must be a boolean")
    name = payload.get("name")
    if name is not None and not isinstance(name, str):
        raise DynamicSimulationError("connection.name must be a string")

    return {
        "bus": bus,
        "p_mw": p_mw,
        "q_mvar": q_mvar,
        "connection_type": connection_type,
        "is_ibr": is_ibr,
        "name": name,
        "vm_pu": vm_pu,
    }


def _connection_model_metadata(
    connection: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    if connection is None:
        return None
    connection_type = str(connection["connection_type"])
    direction = "load" if connection_type in M2_LOAD_CONNECTION_TYPES else "generation"
    return {
        "bus": connection["bus"],
        "p_mw": connection["p_mw"],
        "q_mvar": connection["q_mvar"],
        "connection_type": connection_type,
        "is_ibr": connection["is_ibr"],
        "name": connection.get("name"),
        "vm_pu": connection["vm_pu"],
        "andes_model": "PQ",
        "resource_direction": direction,
        "modeling_assumption": (
            "constant_power_load"
            if direction == "load"
            else "constant_power_generation_as_negative_pq_load"
        ),
        "dynamic_detail": "static_network_injection_no_machine_or_ibr_controls",
    }


def _apply_static_connection(
    system: Any,
    connection: Mapping[str, Any],
) -> Dict[str, Any]:
    bus_status = _validate_connection_bus(system, connection["bus"])
    if bus_status is False:
        raise DynamicSimulationError(
            f"connection.bus {connection['bus']} does not exist in the ANDES dynamic case"
        )

    base_mva = _system_base_mva(system)
    connection_type = str(connection["connection_type"])
    sign = 1.0 if connection_type in M2_LOAD_CONNECTION_TYPES else -1.0
    p0 = sign * float(connection["p_mw"]) / base_mva
    q0 = sign * float(connection["q_mvar"]) / base_mva
    params: Dict[str, Any] = {
        "idx": _connection_idx(connection),
        "bus": connection["bus"],
        "p0": p0,
        "q0": q0,
        "u": 1,
    }
    name = connection.get("name")
    if isinstance(name, str) and name.strip():
        params["name"] = name.strip()
    vn = _bus_vn(system, connection["bus"])
    if vn is not None:
        params["Vn"] = vn

    try:
        _andes_add(system, "PQ", params)
    except Exception as exc:
        raise DynamicSimulationError(
            f"Failed to add static PQ interconnection model to ANDES: {exc}"
        ) from exc

    return {
        "requested": True,
        "applied": True,
        "model": "PQ",
        "mode": (
            "static_load"
            if connection_type in M2_LOAD_CONNECTION_TYPES
            else "static_generation_as_negative_pq_load"
        ),
        "system_base_mva": base_mva,
        "p0_pu": _round_float(p0),
        "q0_pu": _round_float(q0),
        "params": {
            key: _round_float(value) if isinstance(value, float) else value
            for key, value in params.items()
        },
        "bus_validation": "validated" if bus_status is True else "not_available",
        "power_sign_convention": (
            "In ANDES PQ, positive p0/q0 consumes load; generation-like "
            "connections are represented as negative PQ demand in M2 v1."
        ),
    }


def _connection_not_requested() -> Dict[str, Any]:
    return {
        "requested": False,
        "applied": False,
        "mode": "standalone_dynamic_case",
    }


def _connection_pending() -> Dict[str, Any]:
    return {
        "requested": True,
        "applied": False,
        "mode": "pending_static_pq_application",
    }


def _validate_connection_bus(system: Any, bus: int) -> Optional[bool]:
    bus_model = getattr(system, "Bus", None)
    if bus_model is None:
        return None
    ids = _values_from_param(getattr(bus_model, "idx", None))
    if not ids:
        return None
    return str(bus) in {str(item) for item in ids}


def _bus_vn(system: Any, bus: int) -> Optional[float]:
    bus_model = getattr(system, "Bus", None)
    if bus_model is None:
        return None
    ids = _values_from_param(getattr(bus_model, "idx", None))
    vn_values = _values_from_param(getattr(bus_model, "Vn", None))
    if not ids or not vn_values:
        return None
    for index, bus_id in enumerate(ids):
        if str(bus_id) != str(bus):
            continue
        if index >= len(vn_values):
            return None
        try:
            value = float(vn_values[index])
        except (TypeError, ValueError):
            return None
        if math.isfinite(value) and value > 0:
            return value
    return None


def _connection_idx(connection: Mapping[str, Any]) -> str:
    raw_name = connection.get("name")
    if isinstance(raw_name, str) and raw_name.strip():
        base = raw_name.strip()
    else:
        base = f"gridmind_{connection['connection_type']}_{connection['bus']}"
    return (
        base.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _system_base_mva(system: Any) -> float:
    for value in (
        _nested_attr(system, ("config", "mva")),
        _nested_attr(system, ("config", "MVA")),
        _nested_attr(system, ("config", "base_mva")),
        _nested_attr(system, ("config", "baseMVA")),
        getattr(system, "mva", None),
        getattr(system, "base_mva", None),
        getattr(system, "baseMVA", None),
    ):
        number = _optional_positive_float(value)
        if number is not None:
            return number
    return DEFAULT_SYSTEM_BASE_MVA


def _nested_attr(root: Any, path: Sequence[str]) -> Any:
    current = root
    for part in path:
        if current is None:
            return None
        current = getattr(current, part, None)
    return current


def _optional_positive_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "v"):
        value = value.v
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number) and number > 0:
        return number
    return None


def _dynamic_limitations(
    connection_model: Optional[Mapping[str, Any]],
    *,
    case: Optional[DynamicCaseDefinition] = None,
) -> List[str]:
    if connection_model is None:
        limitations = ["standalone_transient_screen_no_interconnection_model"]
    else:
        limitations = [
            "m2_connection_model_is_static_pq",
            "m2_connection_does_not_add_machine_or_inverter_dynamics",
        ]
        if connection_model.get("is_ibr"):
            limitations.append("m2_v1_does_not_model_detailed_ibr_controls")
    if case is not None and case.case_id == "ieee118_public_dynamic":
        limitations.append(IEEE118_PUBLIC_LIMITATION)
    return limitations


def _andes_add(system: Any, model_name: str, params: Mapping[str, Any]) -> None:
    add = getattr(system, "add", None)
    if not callable(add):
        raise DynamicSimulationError("ANDES system does not expose add()")
    payload = dict(params)
    try:
        add(model_name, **payload)
    except TypeError as exc:
        try:
            add(model_name, payload)
        except TypeError:
            raise exc


def _setup_system(system: Any) -> None:
    setup = getattr(system, "setup", None)
    if callable(setup):
        setup()


def _run_routine(routine: Any, name: str) -> bool:
    run = getattr(routine, "run", None)
    if not callable(run):
        raise DynamicSimulationError(f"ANDES {name} routine does not expose run()")
    result = run()
    return False if result is False else True


def _run_pflow_with_retries(
    system: Any,
    case: DynamicCaseDefinition,
) -> Tuple[bool, Dict[str, Any]]:
    routine = getattr(system, "PFlow", None)
    attempts = []
    for label, config in _andes_pflow_attempts(case):
        applied = _apply_routine_config(routine, config)
        try:
            converged = _run_routine(routine, "PFlow")
        except Exception as exc:
            attempts.append(
                {
                    "label": label,
                    "config": dict(config),
                    "applied_config": applied,
                    "converged": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        attempts.append(
            {
                "label": label,
                "config": dict(config),
                "applied_config": applied,
                "converged": bool(converged),
            }
        )
        if converged:
            return True, {
                "successful_attempt": label,
                "attempts": attempts,
            }
    return False, {
        "successful_attempt": None,
        "attempts": attempts,
    }


def _andes_pflow_attempts(case: DynamicCaseDefinition) -> List[Tuple[str, Dict[str, Any]]]:
    base_attempts: List[Tuple[str, Dict[str, Any]]] = [
        (
            "default",
            {
                "report": 0,
            },
        ),
    ]
    if case.case_id != "ieee118_public_dynamic":
        return base_attempts
    return base_attempts + [
        (
            "nr_linesearch_100",
            {
                "method": "NR",
                "linesearch": 1,
                "max_iter": 100,
                "tol": 1e-6,
                "report": 0,
            },
        ),
        (
            "nr_no_linesearch_100",
            {
                "method": "NR",
                "linesearch": 0,
                "max_iter": 100,
                "tol": 1e-6,
                "report": 0,
            },
        ),
        (
            "dishonest_100",
            {
                "method": "dishonest",
                "max_iter": 100,
                "tol": 1e-6,
                "report": 0,
            },
        ),
        (
            "nk_100",
            {
                "method": "NK",
                "max_iter": 100,
                "tol": 1e-6,
                "report": 0,
            },
        ),
    ]


def _apply_routine_config(routine: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
    target = getattr(routine, "config", None)
    if target is None:
        return {}
    applied: Dict[str, Any] = {}
    for key, value in config.items():
        try:
            setattr(target, key, value)
        except Exception:
            continue
        applied[key] = value
    return applied


def _disable_existing_toggles(system: Any) -> None:
    toggle = getattr(system, "Toggle", None)
    if toggle is None:
        return
    idx_values = _values_from_param(getattr(toggle, "idx", None))
    set_method = getattr(toggle, "set", None)
    if callable(set_method) and idx_values:
        for idx in idx_values:
            try:
                set_method("u", idx, 0)
            except Exception:
                continue
        return
    u = getattr(toggle, "u", None)
    values = getattr(u, "v", None)
    try:
        for index in range(len(values)):
            values[index] = 0
    except Exception:
        return


def _extract_timeseries_bundle(system: Any) -> Dict[str, Optional[TimeSeries]]:
    angle = _merged_timeseries(
        [
            _timeseries_for(system, "GENROU", "delta"),
            _timeseries_for(system, "GENCLS", "delta"),
        ]
    )
    speed = _merged_timeseries(
        [
            _timeseries_for(system, "GENROU", "omega"),
            _timeseries_for(system, "GENCLS", "omega"),
        ]
    )
    voltage = _timeseries_for(system, "Bus", "v")
    return {
        "generator_angle_rad": angle,
        "generator_speed_pu": speed,
        "bus_voltage_pu": voltage,
    }


def _compact_trajectories(
    series_bundle: Mapping[str, Optional[TimeSeries]],
    *,
    max_samples: int,
) -> Dict[str, Any]:
    angle = series_bundle.get("generator_angle_rad")
    speed = series_bundle.get("generator_speed_pu")
    voltage = series_bundle.get("bus_voltage_pu")
    base_time = _first_time_vector(angle, speed, voltage)
    sampled_indices = _sample_indices(len(base_time), max_samples)
    trajectories: Dict[str, Any] = {
        "time_s": [_round_float(base_time[index]) for index in sampled_indices],
        "raw_sample_count": len(base_time),
        "truncated_samples": max(0, len(base_time) - len(sampled_indices)),
    }
    if angle is not None:
        trajectories["generator_angle_rad"] = _series_payload(angle, sampled_indices)
    if speed is not None:
        trajectories["generator_speed_pu"] = _series_payload(speed, sampled_indices)
    if voltage is not None:
        trajectories["bus_voltage_pu"] = _series_payload(voltage, sampled_indices)
    return trajectories


def _stability_metrics(
    series_bundle: Mapping[str, Optional[TimeSeries]],
    *,
    tds_converged: bool,
    thresholds: Mapping[str, float],
) -> Dict[str, Any]:
    reason_codes: List[str] = []
    angle_values = _series_values(series_bundle.get("generator_angle_rad"))
    speed_values = _series_values(series_bundle.get("generator_speed_pu"))
    voltage_values = _series_values(series_bundle.get("bus_voltage_pu"))

    max_angle = _max_row_spread(angle_values)
    final_angle = _row_spread(angle_values[-1]) if angle_values else None
    max_speed = _max_abs_deviation(speed_values, reference=1.0)
    min_voltage = _min_value(voltage_values)

    if not tds_converged:
        reason_codes.append("tds_not_converged")
    if max_angle is None:
        reason_codes.append("angle_timeseries_unavailable")
    elif max_angle > thresholds["max_angle_spread_rad"]:
        reason_codes.append("angle_spread_exceeds_threshold")
    if max_speed is None:
        reason_codes.append("speed_timeseries_unavailable")
    elif max_speed > thresholds["max_speed_deviation_pu"]:
        reason_codes.append("speed_deviation_exceeds_threshold")
    if min_voltage is None:
        reason_codes.append("voltage_timeseries_unavailable")
    elif min_voltage < thresholds["min_voltage_pu"]:
        reason_codes.append("voltage_below_threshold")
    if not reason_codes:
        reason_codes.append("transient_metrics_within_thresholds")

    return {
        "tds_converged": bool(tds_converged),
        "max_angle_spread_rad": _optional_round(max_angle),
        "final_angle_spread_rad": _optional_round(final_angle),
        "max_speed_deviation_pu": _optional_round(max_speed),
        "min_voltage_pu": _optional_round(min_voltage),
        "reason_codes": reason_codes,
    }


def _classify_stability(metrics: Mapping[str, Any]) -> str:
    codes = set(str(code) for code in metrics.get("reason_codes", []))
    failure_codes = {
        "tds_not_converged",
        "angle_spread_exceeds_threshold",
        "speed_deviation_exceeds_threshold",
        "voltage_below_threshold",
    }
    if codes.intersection(failure_codes):
        return "fail"
    missing_codes = {
        "angle_timeseries_unavailable",
        "speed_timeseries_unavailable",
        "voltage_timeseries_unavailable",
    }
    if codes.intersection(missing_codes):
        return "borderline"
    return "pass"


def _routine_error_result(
    *,
    case: DynamicCaseDefinition,
    case_path: str,
    disturbance: Mapping[str, Any],
    simulation_time_s: float,
    thresholds: Mapping[str, Any],
    load_metadata: Mapping[str, Any],
    routine: str,
    message: str,
    reason_codes: Sequence[str],
    connection_model: Optional[Mapping[str, Any]] = None,
    connection_application: Optional[Mapping[str, Any]] = None,
    error_type: str = "simulation_error",
    routine_diagnostics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    simulation = {
        "simulation_time_s": simulation_time_s,
        "thresholds": dict(thresholds),
    }
    if routine_diagnostics is not None:
        simulation["routine_diagnostics"] = dict(routine_diagnostics)
    return {
        "ok": False,
        "tool": "run_transient_stability",
        "backend": "andes",
        "case_path": case_path,
        "case_info": {**case.to_dict(), **dict(load_metadata)},
        "disturbance": dict(disturbance),
        "simulation": simulation,
        "connection_model": None if connection_model is None else dict(connection_model),
        "connection_application": (
            _connection_not_requested()
            if connection_application is None
            else dict(connection_application)
        ),
        "dynamic_interconnection_modeling": bool(
            connection_application and connection_application.get("applied")
        ),
        "limitations": _dynamic_limitations(connection_model, case=case),
        "stability": {"status": "error"},
        "metrics": _empty_metrics(reason_codes),
        "error_type": error_type,
        "routine": routine,
        "message": message,
        "reason_codes": list(reason_codes),
    }


def _connection_error_result(
    *,
    case: DynamicCaseDefinition,
    case_path: str,
    disturbance: Mapping[str, Any],
    simulation_time_s: float,
    thresholds: Mapping[str, Any],
    load_metadata: Mapping[str, Any],
    connection_model: Optional[Mapping[str, Any]],
    connection_application: Mapping[str, Any],
    message: str,
    reason_codes: Sequence[str],
    error_type: str,
) -> Dict[str, Any]:
    return _routine_error_result(
        case=case,
        case_path=case_path,
        disturbance=disturbance,
        simulation_time_s=simulation_time_s,
        thresholds=thresholds,
        load_metadata=load_metadata,
        connection_model=connection_model,
        connection_application=connection_application,
        routine="connection_modeling",
        message=message,
        reason_codes=reason_codes,
        error_type=error_type,
    )


def _public_case_data_error_result(
    *,
    case: DynamicCaseDefinition,
    case_path: str,
    disturbance: Mapping[str, Any],
    simulation_time_s: float,
    thresholds: Mapping[str, Any],
    connection_model: Optional[Mapping[str, Any]],
    connection_application: Mapping[str, Any],
    error_type: str,
    message: str,
) -> Dict[str, Any]:
    reason = (
        "dynamic_data_unavailable"
        if error_type == "dynamic_data_unavailable"
        else "case_data_unavailable"
    )
    return _routine_error_result(
        case=case,
        case_path=case_path,
        disturbance=disturbance,
        simulation_time_s=simulation_time_s,
        thresholds=thresholds,
        load_metadata={
            "case_source": IEEE118_PUBLIC_SOURCE,
            "data_label": "public_benchmark_data_not_customer_validated",
        },
        connection_model=connection_model,
        connection_application={
            **dict(connection_application),
            "requested": bool(connection_application.get("requested")),
            "applied": False,
            "error_type": error_type,
            "message": message,
        },
        routine="case_data_resolution",
        message=message,
        reason_codes=[reason],
        error_type=error_type,
    )


def _runtime_error_type(case: DynamicCaseDefinition, default: str) -> str:
    if case.case_id == "ieee118_public_dynamic":
        return "ieee118_dynamic_validation_failed"
    return default


def _empty_metrics(reason_codes: Sequence[str]) -> Dict[str, Any]:
    return {
        "tds_converged": False,
        "max_angle_spread_rad": None,
        "final_angle_spread_rad": None,
        "max_speed_deviation_pu": None,
        "min_voltage_pu": None,
        "reason_codes": list(reason_codes),
    }


def _timeseries_for(system: Any, model_name: str, var_name: str) -> Optional[TimeSeries]:
    model = getattr(system, model_name, None)
    var = getattr(model, var_name, None) if model is not None else None
    if var is None:
        return None
    tds = getattr(system, "TDS", None)
    getter = getattr(tds, "get_timeseries", None)
    if not callable(getter):
        return _direct_timeseries(system, var, var_name=var_name)
    try:
        return _coerce_timeseries(getter(var))
    except Exception:
        return _direct_timeseries(system, var, var_name=var_name)


def _direct_timeseries(system: Any, var: Any, *, var_name: str) -> Optional[TimeSeries]:
    dae = getattr(system, "dae", None)
    ts = getattr(dae, "ts", None)
    if ts is None:
        return None
    addresses = _values_from_param(getattr(var, "a", None))
    if not addresses:
        return None
    source_name = "x" if var_name in {"delta", "omega"} else "y"
    matrix = getattr(ts, source_name, None)
    time_vector = _float_list(getattr(ts, "t", []))
    rows = _matrix_subset(matrix, addresses)
    if not rows:
        return None
    return TimeSeries(
        time_s=tuple(time_vector[: len(rows)]),
        columns=tuple(str(addr) for addr in addresses),
        values=tuple(tuple(row) for row in rows),
    )


def _coerce_timeseries(value: Any) -> Optional[TimeSeries]:
    if value is None:
        return None
    columns = _string_list(getattr(value, "columns", []))
    index = _float_list(getattr(value, "index", []))
    matrix_obj = None
    to_numpy = getattr(value, "to_numpy", None)
    if callable(to_numpy):
        matrix_obj = to_numpy()
    elif hasattr(value, "values"):
        matrix_obj = value.values
    else:
        matrix_obj = value
    rows = _matrix_rows(matrix_obj)
    if not rows:
        return None
    if not index:
        index = [float(i) for i in range(len(rows))]
    if not columns:
        columns = [str(i + 1) for i in range(len(rows[0]))]
    return TimeSeries(
        time_s=tuple(index[: len(rows)]),
        columns=tuple(columns[: len(rows[0])]),
        values=tuple(tuple(row) for row in rows),
    )


def _merged_timeseries(series: Sequence[Optional[TimeSeries]]) -> Optional[TimeSeries]:
    selected = [item for item in series if item is not None and item.row_count > 0]
    if not selected:
        return None
    base = selected[0]
    row_count = min(item.row_count for item in selected)
    columns: List[str] = []
    rows: List[List[float]] = [[] for _ in range(row_count)]
    for item in selected:
        columns.extend(item.columns)
        for idx in range(row_count):
            rows[idx].extend(item.values[idx])
    return TimeSeries(
        time_s=base.time_s[:row_count],
        columns=tuple(columns),
        values=tuple(tuple(row) for row in rows),
    )


def _first_time_vector(*series: Optional[TimeSeries]) -> Tuple[float, ...]:
    for item in series:
        if item is not None and item.time_s:
            return item.time_s
    return tuple()


def _series_payload(series: TimeSeries, indices: Sequence[int]) -> Dict[str, Any]:
    return {
        "columns": list(series.columns),
        "values": [
            [_round_float(value) for value in series.values[index]]
            for index in indices
            if 0 <= index < series.row_count
        ],
    }


def _series_values(series: Optional[TimeSeries]) -> List[List[float]]:
    if series is None:
        return []
    return [list(row) for row in series.values]


def _sample_indices(length: int, max_samples: int) -> List[int]:
    if length <= 0:
        return []
    if length <= max_samples:
        return list(range(length))
    if max_samples == 1:
        return [0]
    step = (length - 1) / float(max_samples - 1)
    indices = [int(round(i * step)) for i in range(max_samples)]
    deduped: List[int] = []
    for index in indices:
        index = min(length - 1, max(0, index))
        if not deduped or index != deduped[-1]:
            deduped.append(index)
    return deduped


def _matrix_rows(value: Any) -> List[List[float]]:
    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return []
    rows: List[List[float]] = []
    if value and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        value = [[item] for item in value]
    for row in value:
        if isinstance(row, tuple):
            row = list(row)
        if not isinstance(row, list):
            continue
        converted = []
        for item in row:
            if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)):
                converted.append(float(item))
        if converted:
            rows.append(converted)
    return rows


def _matrix_subset(matrix: Any, addresses: Sequence[Any]) -> List[List[float]]:
    rows = _matrix_rows(matrix)
    indices = []
    for address in addresses:
        if isinstance(address, int) and not isinstance(address, bool):
            indices.append(address)
    if not rows or not indices:
        return []
    selected = []
    for row in rows:
        selected.append([row[index] for index in indices if 0 <= index < len(row)])
    return [row for row in selected if row]


def _max_row_spread(rows: Sequence[Sequence[float]]) -> Optional[float]:
    spreads = [_row_spread(row) for row in rows if row]
    return max(spreads) if spreads else None


def _row_spread(row: Sequence[float]) -> float:
    if len(row) <= 1:
        return 0.0
    return max(row) - min(row)


def _max_abs_deviation(
    rows: Sequence[Sequence[float]],
    *,
    reference: float,
) -> Optional[float]:
    values = [
        abs(value - reference)
        for row in rows
        for value in row
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return max(values) if values else None


def _min_value(rows: Sequence[Sequence[float]]) -> Optional[float]:
    values = [
        value
        for row in rows
        for value in row
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return min(values) if values else None


def _resolve_andes_case_file(andes: Any, locator: Optional[str]) -> Optional[str]:
    if locator is None:
        return None
    get_case = getattr(andes, "get_case", None)
    if callable(get_case):
        return str(get_case(locator))
    return str(locator)


def _unsupported_case_message(case_path: str) -> str:
    key = _normalize_case_alias(case_path)
    for alias, reason in UNSUPPORTED_DYNAMIC_CASES.items():
        if _normalize_case_alias(alias) == key:
            return reason
    supported = ", ".join(case.case_id for case in DYNAMIC_CASES)
    return (
        f"Dynamic case '{case_path}' is not supported in M2 v1. "
        f"Supported dynamic cases are: {supported}."
    )


def _normalize_case_alias(value: Any) -> str:
    if not isinstance(value, str):
        return str(value)
    return (
        value.strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def _required_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DynamicSimulationError(f"disturbance.{key} must be an integer")
    return value


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DynamicSimulationError(f"disturbance.{key} must be a non-empty string")
    return value.strip()


def _required_number(mapping: Mapping[str, Any], key: str) -> float:
    if key not in mapping:
        raise DynamicSimulationError(f"disturbance.{key} is required")
    return _finite_float(mapping[key], f"disturbance.{key}")


def _positive_float(value: Any, key: str) -> float:
    number = _finite_float(value, key)
    if number <= 0:
        raise DynamicSimulationError(f"{key} must be positive")
    return number


def _non_negative_float(value: Any, key: str) -> float:
    number = _finite_float(value, key)
    if number < 0:
        raise DynamicSimulationError(f"{key} must be non-negative")
    return number


def _positive_int(value: Any, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DynamicSimulationError(f"{key} must be an integer")
    if value <= 0:
        raise DynamicSimulationError(f"{key} must be positive")
    return value


def _finite_float(value: Any, key: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DynamicSimulationError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise DynamicSimulationError(f"{key} must be finite")
    return number


def _values_from_param(value: Any) -> List[Any]:
    if value is None:
        return []
    if hasattr(value, "v"):
        value = value.v
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return list(value)
    return [value]


def _string_list(value: Any) -> List[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _float_list(value: Any) -> List[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            result.append(float(item))
    return result


def _optional_round(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return _round_float(value)


def _round_float(value: float) -> float:
    return round(float(value), 9)

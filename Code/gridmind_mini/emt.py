"""EMT/SCR screening tools for the Mini Grid-Mind F4 layer.

The full Grid-Mind paper describes EMT screening as the highest-fidelity stage.
This module implements the first practical IEEE14 version: a solver-grounded
short-circuit-ratio screen at the proposed point of interconnection. It is not a
time-domain EMT waveform simulation and does not model inverter controls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional

from .solver import GridSolver, PowerFlowError


DEFAULT_SCR_THRESHOLD = 3.0
DEFAULT_BORDERLINE_MARGIN = 0.25
IBR_TYPES = {"solar", "wind", "bess", "hybrid"}

if TYPE_CHECKING:
    from .cia import ConnectionRequest


class EMTScreeningError(RuntimeError):
    """Raised when EMT/SCR screening cannot be executed."""


@dataclass(frozen=True)
class EMTScreeningMetrics:
    """Compact SCR-screen metrics."""

    scr: Optional[float]
    short_circuit_mva: Optional[float]
    project_mva: float
    threshold: float
    borderline_margin: float
    reason_codes: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scr": self.scr,
            "short_circuit_mva": self.short_circuit_mva,
            "project_mva": self.project_mva,
            "threshold": self.threshold,
            "borderline_margin": self.borderline_margin,
            "reason_codes": list(self.reason_codes),
        }


class EMTScreeningRunner:
    """Run an SCR-style EMT screening proxy on a steady-state network."""

    def __init__(self, solver_factory: Callable[[], GridSolver]) -> None:
        self.solver_factory = solver_factory

    def run(
        self,
        *,
        case_path: str,
        connection: "ConnectionRequest | Mapping[str, Any]",
        scr_threshold: float = DEFAULT_SCR_THRESHOLD,
        borderline_margin: float = DEFAULT_BORDERLINE_MARGIN,
    ) -> Dict[str, Any]:
        if not isinstance(case_path, str) or not case_path.strip():
            raise ValueError("case_path must be a non-empty string")
        if not _finite_positive(scr_threshold):
            raise ValueError("scr_threshold must be a positive finite number")
        if not _finite_nonnegative(borderline_margin):
            raise ValueError("borderline_margin must be a non-negative finite number")

        if isinstance(connection, Mapping):
            from .cia import ConnectionRequest

            request = ConnectionRequest.from_mapping(connection)
        else:
            request = connection
        effective_is_ibr = request.is_ibr or request.connection_type in IBR_TYPES
        if not effective_is_ibr:
            metrics = EMTScreeningMetrics(
                scr=None,
                short_circuit_mva=None,
                project_mva=_project_mva(request),
                threshold=float(scr_threshold),
                borderline_margin=float(borderline_margin),
                reason_codes=["emt_screen_not_required_for_non_ibr_connection"],
            )
            return _result(
                ok=True,
                case_path=case_path,
                connection=request,
                status="skipped",
                passed=True,
                metrics=metrics,
                case_info=None,
                powerflow_summary=None,
                connection_application={
                    "requested": True,
                    "applied": False,
                    "reason": "non_ibr_connection",
                },
            )

        project_mva = _project_mva(request)
        if project_mva <= 0.0:
            metrics = EMTScreeningMetrics(
                scr=None,
                short_circuit_mva=None,
                project_mva=project_mva,
                threshold=float(scr_threshold),
                borderline_margin=float(borderline_margin),
                reason_codes=["emt_project_mva_must_be_positive"],
            )
            return _result(
                ok=True,
                case_path=case_path,
                connection=request,
                status="fail",
                passed=False,
                metrics=metrics,
                case_info=None,
                powerflow_summary=None,
                connection_application={"requested": True, "applied": False},
            )

        solver = self.solver_factory()
        case_info = solver.load_case(case_path)
        connection_application = solver.add_connection(**request.to_dict())
        try:
            powerflow_summary = solver.run_powerflow()
        except PowerFlowError as exc:
            return {
                "ok": False,
                "tool": "run_emt_screening",
                "backend": "pandapower",
                "screening_type": "scr_proxy",
                "case_path": case_path,
                "connection": request.to_dict(),
                "connection_application": {
                    "requested": True,
                    "applied": True,
                    "error_type": "post_connection_powerflow_failed",
                    "message": str(exc),
                    **dict(connection_application),
                },
                "emt": {"status": "error", "passed": False},
                "metrics": {
                    "scr": None,
                    "short_circuit_mva": None,
                    "project_mva": project_mva,
                    "threshold": float(scr_threshold),
                    "borderline_margin": float(borderline_margin),
                    "reason_codes": ["emt_post_connection_powerflow_failed"],
                },
                "limitations": _limitations(),
                "error_type": "post_connection_powerflow_failed",
                "message": str(exc),
            }

        bus_index = _connection_bus_index(request, connection_application)
        strength = _short_circuit_strength_mva(solver, bus_index)
        scr = strength / project_mva
        status, passed, reason_codes = _scr_status(
            scr=scr,
            threshold=float(scr_threshold),
            borderline_margin=float(borderline_margin),
        )
        metrics = EMTScreeningMetrics(
            scr=scr,
            short_circuit_mva=strength,
            project_mva=project_mva,
            threshold=float(scr_threshold),
            borderline_margin=float(borderline_margin),
            reason_codes=reason_codes,
        )
        return _result(
            ok=True,
            case_path=case_path,
            connection=request,
            status=status,
            passed=passed,
            metrics=metrics,
            case_info=case_info.to_dict(),
            powerflow_summary=powerflow_summary,
            connection_application={
                "requested": True,
                "applied": True,
                **dict(connection_application),
            },
            poi_bus={
                "requested_bus": request.bus,
                "bus_index": bus_index,
                "resolved_bus": connection_application.get("resolved_bus"),
            },
        )


def _result(
    *,
    ok: bool,
    case_path: str,
    connection: "ConnectionRequest",
    status: str,
    passed: Optional[bool],
    metrics: EMTScreeningMetrics,
    case_info: Optional[Mapping[str, Any]],
    powerflow_summary: Optional[Mapping[str, Any]],
    connection_application: Mapping[str, Any],
    poi_bus: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "ok": ok,
        "tool": "run_emt_screening",
        "backend": "pandapower",
        "screening_type": "scr_proxy",
        "case_path": case_path,
        "case_info": dict(case_info or {}),
        "connection": connection.to_dict(),
        "poi_bus": dict(poi_bus or {}),
        "connection_application": dict(connection_application),
        "emt": {
            "status": status,
            "passed": passed,
            "criterion": "scr_greater_equal_threshold",
        },
        "metrics": metrics.to_dict(),
        "powerflow_summary": dict(powerflow_summary or {}),
        "limitations": _limitations(),
    }


def _short_circuit_strength_mva(solver: GridSolver, bus_index: int) -> float:
    custom = getattr(solver, "emt_screening_data", None)
    if callable(custom):
        data = custom(bus_index)
        if isinstance(data, Mapping) and data.get("short_circuit_mva") is not None:
            return _positive_float(data["short_circuit_mva"], "short_circuit_mva")

    by_bus = getattr(solver, "short_circuit_mva_by_bus", None)
    if isinstance(by_bus, Mapping) and bus_index in by_bus:
        return _positive_float(by_bus[bus_index], "short_circuit_mva_by_bus")

    net = getattr(solver, "net", None)
    if net is None:
        raise EMTScreeningError(
            "SCR screening requires a pandapower-style solver with solved Ybus data."
        )
    return _pandapower_short_circuit_strength_mva(net, bus_index)


def _pandapower_short_circuit_strength_mva(net: Any, bus_index: int) -> float:
    ppc = getattr(net, "_ppc", None)
    if not isinstance(ppc, Mapping):
        try:
            ppc = net["_ppc"]
        except Exception as exc:
            raise EMTScreeningError("pandapower internal _ppc data is unavailable") from exc
    internal = ppc.get("internal") if isinstance(ppc, Mapping) else None
    if not isinstance(internal, Mapping) or internal.get("Ybus") is None:
        raise EMTScreeningError("pandapower internal Ybus data is unavailable")

    ybus = internal["Ybus"]
    try:
        matrix = ybus.toarray()
    except AttributeError:
        matrix = ybus

    try:
        import numpy as np
    except Exception as exc:
        raise EMTScreeningError("numpy is required for SCR screening") from exc

    lookup_index = _ppc_bus_index(net, bus_index, len(matrix))
    zbus = np.linalg.pinv(matrix)
    zth = complex(zbus[lookup_index, lookup_index])
    zth_abs = abs(zth)
    if not math.isfinite(zth_abs) or zth_abs <= 0.0:
        raise EMTScreeningError("invalid Thevenin impedance from Ybus")

    base_mva = _ppc_base_mva(ppc, net)
    vm_pu = _bus_voltage_pu(net, bus_index)
    strength = (vm_pu * vm_pu) * base_mva / zth_abs
    if not math.isfinite(strength) or strength <= 0.0:
        raise EMTScreeningError("invalid short-circuit strength estimate")
    return float(strength)


def _ppc_bus_index(net: Any, bus_index: int, matrix_size: int) -> int:
    lookups = getattr(net, "_pd2ppc_lookups", None)
    lookup = None
    if isinstance(lookups, Mapping):
        lookup = lookups.get("bus")
    if lookup is not None:
        try:
            mapped = int(lookup[bus_index])
            if 0 <= mapped < matrix_size:
                return mapped
        except Exception:
            pass
    if 0 <= int(bus_index) < matrix_size:
        return int(bus_index)
    raise EMTScreeningError(f"bus index {bus_index} is outside the solved Ybus")


def _ppc_base_mva(ppc: Mapping[str, Any], net: Any) -> float:
    base = ppc.get("baseMVA")
    if base is None:
        base = getattr(net, "sn_mva", 100.0)
    return _positive_float(base, "baseMVA")


def _bus_voltage_pu(net: Any, bus_index: int) -> float:
    try:
        value = net.res_bus.loc[bus_index, "vm_pu"]
        result = float(value)
    except Exception:
        result = 1.0
    if not math.isfinite(result) or result <= 0.0:
        return 1.0
    return result


def _connection_bus_index(
    connection: "ConnectionRequest",
    connection_application: Mapping[str, Any],
) -> int:
    resolved = connection_application.get("resolved_bus")
    if isinstance(resolved, Mapping) and resolved.get("bus_index") is not None:
        return int(resolved["bus_index"])
    return int(connection.bus)


def _project_mva(connection: "ConnectionRequest") -> float:
    return float(math.hypot(connection.p_mw, connection.q_mvar))


def _scr_status(
    *,
    scr: float,
    threshold: float,
    borderline_margin: float,
) -> tuple[str, bool, list[str]]:
    if scr >= threshold + borderline_margin:
        return "pass", True, ["emt_scr_above_threshold"]
    if scr >= threshold:
        return "borderline", False, ["emt_scr_near_threshold"]
    return "fail", False, ["emt_scr_below_threshold"]


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise EMTScreeningError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise EMTScreeningError(f"{name} must be positive and finite")
    return result


def _finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0.0


def _finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0.0


def _limitations() -> list[str]:
    return [
        "emt_v1_scr_screen_only_no_time_domain_waveform",
        "emt_v1_does_not_model_inverter_controls_harmonics_or_protection",
        "scr_estimate_uses_steady_state_network_thevenin_equivalent",
    ]

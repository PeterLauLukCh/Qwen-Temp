"""Integrated M1+M2 assessment orchestration.

This is the project-facing assessment module. It runs M1 steady-state CIA and,
when requested, M2 transient stability through ANDES, then returns one combined
result with explicit stage linkage and limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from .cia import ConnectionRequest, SteadyStateCIARunner
from .dynamic import DynamicSimulationError, TransientStabilityRunner
from .emt import DEFAULT_BORDERLINE_MARGIN, DEFAULT_SCR_THRESHOLD, EMTScreeningRunner
from .public_cases import (
    IEEE118_BUNDLED_SOURCE,
    IEEE118_LOCAL_OVERRIDE_SOURCE,
    IEEE118_PUBLIC_LIMITATION,
    IEEE118_PUBLIC_SOURCE,
    PublicCaseDataError,
    PublicDynamicCaseFiles,
    ieee118_bundled_data_available,
    ieee118_local_override_requested,
    is_ieee118_public_dynamic_alias,
    resolve_ieee118_public_dynamic_files,
)
from .solver import AndesPandaPowerSolver, GridSolver
from .violations import LimitProfile


@dataclass(frozen=True)
class TransientAssessmentRequest:
    """Optional M2 block inside an integrated assessment."""

    enabled: bool = False
    required_for_approval: bool = False
    case_path: Optional[str] = None
    disturbance: Optional[Mapping[str, Any]] = None
    simulation_time_s: float = 5.0
    max_angle_spread_rad: float = 6.283185307179586
    max_speed_deviation_pu: float = 0.05
    min_voltage_pu: float = 0.7
    max_samples: int = 200

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "TransientAssessmentRequest":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("transient must be an object")
        enabled = _optional_bool(payload, "enabled", default=True)
        return cls(
            enabled=enabled,
            required_for_approval=_optional_bool(
                payload,
                "required_for_approval",
                default=enabled,
            ),
            case_path=_optional_str(payload, "case_path"),
            disturbance=_optional_mapping(payload, "disturbance"),
            simulation_time_s=_optional_number(payload, "simulation_time_s", 5.0),
            max_angle_spread_rad=_optional_number(
                payload,
                "max_angle_spread_rad",
                6.283185307179586,
            ),
            max_speed_deviation_pu=_optional_number(
                payload,
                "max_speed_deviation_pu",
                0.05,
            ),
            min_voltage_pu=_optional_number(payload, "min_voltage_pu", 0.7),
            max_samples=_optional_int(payload, "max_samples", 200),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "required_for_approval": self.required_for_approval,
            "case_path": self.case_path,
            "disturbance": None if self.disturbance is None else dict(self.disturbance),
            "simulation_time_s": self.simulation_time_s,
            "max_angle_spread_rad": self.max_angle_spread_rad,
            "max_speed_deviation_pu": self.max_speed_deviation_pu,
            "min_voltage_pu": self.min_voltage_pu,
            "max_samples": self.max_samples,
        }


@dataclass(frozen=True)
class EMTAssessmentRequest:
    """Optional F4 EMT/SCR screening block inside an integrated assessment."""

    enabled: bool = False
    required_for_approval: bool = False
    case_path: Optional[str] = None
    scr_threshold: float = DEFAULT_SCR_THRESHOLD
    borderline_margin: float = DEFAULT_BORDERLINE_MARGIN

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "EMTAssessmentRequest":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("emt must be an object")
        enabled = _optional_bool(payload, "enabled", default=True)
        return cls(
            enabled=enabled,
            required_for_approval=_optional_bool(
                payload,
                "required_for_approval",
                default=enabled,
            ),
            case_path=_optional_str(payload, "case_path"),
            scr_threshold=_optional_number(payload, "scr_threshold", DEFAULT_SCR_THRESHOLD),
            borderline_margin=_optional_number(
                payload,
                "borderline_margin",
                DEFAULT_BORDERLINE_MARGIN,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "required_for_approval": self.required_for_approval,
            "case_path": self.case_path,
            "scr_threshold": self.scr_threshold,
            "borderline_margin": self.borderline_margin,
        }


class IntegratedAssessmentRunner:
    """Run M1 and optional M2 stages in one assessment result."""

    def __init__(
        self,
        *,
        solver_factory: Callable[[], GridSolver],
        limit_profile: Optional[LimitProfile] = None,
        material_worsening_threshold_percent: float = 2.0,
        dynamic_runner: Optional[Any] = None,
        strict_match_solver_factory: Optional[Callable[[], GridSolver]] = None,
    ) -> None:
        self.solver_factory = solver_factory
        self.limit_profile = limit_profile or LimitProfile.normal()
        self.material_worsening_threshold_percent = material_worsening_threshold_percent
        self.dynamic_runner = dynamic_runner or TransientStabilityRunner()
        self.strict_match_solver_factory = strict_match_solver_factory

    def run(
        self,
        *,
        case_path: str,
        connection: ConnectionRequest,
        transient: Optional[TransientAssessmentRequest] = None,
        emt: Optional[EMTAssessmentRequest] = None,
        enable_contingency: bool = False,
        max_violations: int = 50,
        max_contingencies: int = -1,
        max_failed_contingencies: int = 50,
        fail_on_contingency_material_worsening: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(case_path, str) or not case_path.strip():
            raise ValueError("case_path must be a non-empty string")
        transient_request = transient or TransientAssessmentRequest()
        emt_request = emt or EMTAssessmentRequest()
        strict_alignment = _requires_ieee118_strict_alignment(case_path, transient_request)
        alignment_metadata: Optional[Dict[str, Any]] = None
        public_case_files: Optional[PublicDynamicCaseFiles] = None
        solver_factory = self.solver_factory
        if strict_alignment:
            try:
                public_case_files = resolve_ieee118_public_dynamic_files()
                alignment_metadata = public_case_files.to_metadata()
            except PublicCaseDataError as exc:
                return _integrated_case_data_unavailable_result(
                    case_path=case_path,
                    connection=connection,
                    transient_request=transient_request,
                    error_type=exc.error_type,
                    message=str(exc),
                )
            solver_factory = (
                self.strict_match_solver_factory
                or (lambda: AndesPandaPowerSolver(case_files=public_case_files))
            )

        try:
            m1_result = SteadyStateCIARunner(
                solver_factory=solver_factory,
                limit_profile=self.limit_profile,
                material_worsening_threshold_percent=self.material_worsening_threshold_percent,
            ).run(
                case_path=case_path,
                connection=connection,
                enable_contingency=enable_contingency,
                enable_transient=False,
                enable_emt=False,
                max_violations=max_violations,
                max_contingencies=max_contingencies,
                max_failed_contingencies=max_failed_contingencies,
                fail_on_contingency_material_worsening=fail_on_contingency_material_worsening,
            )
        except Exception as exc:
            if not strict_alignment:
                raise
            return _integrated_case_data_unavailable_result(
                case_path=case_path,
                connection=connection,
                transient_request=transient_request,
                error_type="ieee118_strict_m1_conversion_failed",
                message=str(exc),
                alignment_metadata=alignment_metadata,
            )

        m1_stage = _m1_stage_report(m1_result)
        m2_stage = self._m2_stage_report(
            case_path,
            connection,
            transient_request,
            public_case_files=public_case_files,
        )
        emt_stage = self._emt_stage_report(
            case_path=case_path,
            connection=connection,
            emt=emt_request,
            solver_factory=solver_factory,
        )
        recommendation, complete, reason_codes = _integrated_recommendation(
            m1_stage,
            m2_stage,
            emt_stage,
            transient_request=transient_request,
            emt_request=emt_request,
        )
        m2_models_connection = bool(m2_stage.get("dynamic_interconnection_modeling"))
        emt_models_connection = bool(emt_stage.get("emt_interconnection_modeling"))

        return {
            "ok": True,
            "tool": "run_integrated_assessment",
            "backend": {
                "m1": "pandapower",
                "m2": "andes",
                "emt": "pandapower",
            },
            "case_path": case_path,
            "connection": connection.to_dict(),
            "recommendation": recommendation,
            "complete": complete,
            "reason_codes": reason_codes,
            "stage_reports": [m1_stage, m2_stage, emt_stage],
            "m1_result": m1_result,
            "m2_result": m2_stage.get("result"),
            "emt_result": emt_stage.get("result"),
            "case_alignment": _case_alignment_report(
                strict_alignment=strict_alignment,
                alignment_metadata=alignment_metadata,
                m2_stage=m2_stage,
            ),
            "m1_case_source": _m1_case_source(strict_alignment, alignment_metadata),
            "m2_case_source": _m2_case_source(m2_stage),
            "limitations": _integrated_limitations(m2_stage, transient_request, emt_stage, emt_request),
            "linkage": {
                "m1_models_new_connection": True,
                "m2_models_new_connection": m2_models_connection,
                "emt_models_new_connection": emt_models_connection,
                "m2_linkage": _m2_linkage(m2_stage, transient_request),
                "emt_linkage": _emt_linkage(emt_stage, emt_request),
                "note": _m2_linkage_note(m2_models_connection, transient_request),
            },
            "summary": {
                "m1_recommendation": m1_result.get("recommendation"),
                "m2_status": m2_stage.get("status"),
                "m2_stability_status": _nested_get(m2_stage.get("result"), "stability.status"),
                "transient_required_for_approval": transient_request.required_for_approval,
                "emt_status": emt_stage.get("status"),
                "emt_scr": _nested_get(emt_stage.get("result"), "metrics.scr"),
                "emt_required_for_approval": emt_request.required_for_approval,
            },
        }

    def _m2_stage_report(
        self,
        case_path: str,
        connection: ConnectionRequest,
        transient: TransientAssessmentRequest,
        public_case_files: Optional[PublicDynamicCaseFiles] = None,
    ) -> Dict[str, Any]:
        if not transient.enabled:
            return {
                "stage": "m2_transient_stability",
                "status": "skipped",
                "passed": None,
                "required_for_approval": False,
                "reason_codes": ["transient_not_requested"],
                "result": None,
            }
        if transient.disturbance is None:
            return {
                "stage": "m2_transient_stability",
                "status": "not_ready",
                "passed": False if transient.required_for_approval else None,
                "required_for_approval": transient.required_for_approval,
                "reason_codes": ["transient_disturbance_missing"],
                "result": None,
            }

        dynamic_case_path = transient.case_path or case_path
        try:
            run_kwargs = {
                "case_path": dynamic_case_path,
                "disturbance": transient.disturbance,
                "connection": connection,
                "simulation_time_s": transient.simulation_time_s,
                "max_angle_spread_rad": transient.max_angle_spread_rad,
                "max_speed_deviation_pu": transient.max_speed_deviation_pu,
                "min_voltage_pu": transient.min_voltage_pu,
                "max_samples": transient.max_samples,
            }
            if public_case_files is not None:
                run_kwargs["public_case_files"] = public_case_files
            result = self.dynamic_runner.run(**run_kwargs)
        except DynamicSimulationError as exc:
            result = {
                "ok": False,
                "tool": "run_transient_stability",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "stability": {"status": "error"},
                "metrics": {"reason_codes": ["transient_input_error"]},
            }

        stability_status = _nested_get(result, "stability.status")
        passed = bool(result.get("ok", False) and stability_status == "pass")
        if passed:
            status = "pass"
        elif stability_status == "borderline":
            status = "borderline"
        elif result.get("ok", False):
            status = "fail"
        else:
            status = "unavailable"
        reason_codes = _list_of_strings(_nested_get(result, "metrics.reason_codes"))
        if not reason_codes:
            reason_codes = _list_of_strings(result.get("reason_codes"))
        if not reason_codes:
            reason_codes = ["transient_result_without_reason_codes"]
        dynamic_interconnection_modeling = bool(
            result.get("dynamic_interconnection_modeling", False)
        )

        return {
            "stage": "m2_transient_stability",
            "status": status,
            "passed": passed,
            "required_for_approval": transient.required_for_approval,
            "case_path": dynamic_case_path,
            "reason_codes": reason_codes,
            "result": result,
            "dynamic_interconnection_modeling": dynamic_interconnection_modeling,
            "connection_model": result.get("connection_model"),
            "connection_application": result.get("connection_application"),
        }

    def _emt_stage_report(
        self,
        *,
        case_path: str,
        connection: ConnectionRequest,
        emt: EMTAssessmentRequest,
        solver_factory: Callable[[], GridSolver],
    ) -> Dict[str, Any]:
        if not emt.enabled:
            return {
                "stage": "f4_emt_screening",
                "status": "skipped",
                "passed": None,
                "required_for_approval": False,
                "reason_codes": ["emt_not_requested"],
                "result": None,
                "emt_interconnection_modeling": False,
            }

        emt_case_path = emt.case_path or case_path
        try:
            result = EMTScreeningRunner(solver_factory).run(
                case_path=emt_case_path,
                connection=connection,
                scr_threshold=emt.scr_threshold,
                borderline_margin=emt.borderline_margin,
            )
        except Exception as exc:
            result = {
                "ok": False,
                "tool": "run_emt_screening",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "emt": {"status": "error", "passed": False},
                "metrics": {"reason_codes": ["emt_screening_unavailable"]},
            }

        emt_status = _nested_get(result, "emt.status")
        if result.get("ok", False) and emt_status in {"pass", "skipped"}:
            status = str(emt_status)
            passed = True
        elif result.get("ok", False) and emt_status == "borderline":
            status = "borderline"
            passed = False
        elif result.get("ok", False):
            status = "fail"
            passed = False
        else:
            status = "unavailable"
            passed = False if emt.required_for_approval else None
        reason_codes = _list_of_strings(_nested_get(result, "metrics.reason_codes"))
        if not reason_codes:
            reason_codes = ["emt_result_without_reason_codes"]

        connection_application = result.get("connection_application")
        if not isinstance(connection_application, Mapping):
            connection_application = {}
        return {
            "stage": "f4_emt_screening",
            "status": status,
            "passed": passed,
            "required_for_approval": emt.required_for_approval,
            "case_path": emt_case_path,
            "reason_codes": reason_codes,
            "result": result,
            "emt_interconnection_modeling": bool(connection_application.get("applied")),
            "connection_application": dict(connection_application),
        }


def _requires_ieee118_strict_alignment(
    case_path: str,
    transient: TransientAssessmentRequest,
) -> bool:
    if not transient.enabled:
        return False
    return is_ieee118_public_dynamic_alias(case_path) or (
        transient.case_path is not None
        and is_ieee118_public_dynamic_alias(transient.case_path)
    )


def _integrated_case_data_unavailable_result(
    *,
    case_path: str,
    connection: ConnectionRequest,
    transient_request: TransientAssessmentRequest,
    error_type: str,
    message: str,
    alignment_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    reason = (
        "dynamic_data_unavailable"
        if error_type == "dynamic_data_unavailable"
        else "case_data_unavailable"
    )
    metadata = dict(alignment_metadata or {})
    source = str(
        metadata.get(
            "source",
            IEEE118_LOCAL_OVERRIDE_SOURCE
            if ieee118_local_override_requested()
            else (
                IEEE118_BUNDLED_SOURCE
                if ieee118_bundled_data_available()
                else IEEE118_PUBLIC_SOURCE
            ),
        )
    )
    m1_stage = {
        "stage": "m1_steady_state_cia",
        "status": "unavailable",
        "passed": False,
        "recommendation": None,
        "required_for_approval": True,
        "reason_codes": [reason],
    }
    m2_result = {
        "ok": False,
        "tool": "run_transient_stability",
        "backend": "andes",
        "case_path": transient_request.case_path or case_path,
        "case_info": {
            "case_id": "ieee118_public_dynamic",
            "case_source": source,
            "dynamic_data_available": error_type != "dynamic_data_unavailable",
        },
        "connection_model": None,
        "connection_application": {
            "requested": True,
            "applied": False,
            "error_type": error_type,
            "message": message,
        },
        "dynamic_interconnection_modeling": False,
        "limitations": [IEEE118_PUBLIC_LIMITATION],
        "stability": {"status": "error"},
        "metrics": {
            "tds_converged": False,
            "max_angle_spread_rad": None,
            "final_angle_spread_rad": None,
            "max_speed_deviation_pu": None,
            "min_voltage_pu": None,
            "reason_codes": [reason],
        },
        "error_type": error_type,
        "message": message,
        "reason_codes": [reason],
    }
    m2_stage = {
        "stage": "m2_transient_stability",
        "status": "unavailable",
        "passed": False if transient_request.required_for_approval else None,
        "required_for_approval": transient_request.required_for_approval,
        "case_path": transient_request.case_path or case_path,
        "reason_codes": [reason],
        "result": m2_result,
        "dynamic_interconnection_modeling": False,
        "connection_model": None,
        "connection_application": m2_result["connection_application"],
    }
    return {
        "ok": True,
        "tool": "run_integrated_assessment",
        "backend": {"m1": "pandapower", "m2": "andes"},
        "case_path": case_path,
        "connection": connection.to_dict(),
        "recommendation": "borderline",
        "complete": False,
        "reason_codes": [
            "integrated_assessment",
            reason,
            "required_m2_unavailable"
            if transient_request.required_for_approval
            else "m2_unavailable_not_required",
        ],
        "stage_reports": [m1_stage, m2_stage],
        "m1_result": None,
        "m2_result": m2_result,
        "case_alignment": {
            "strict_match": True,
            "source": source,
            "available": False,
            "error_type": error_type,
            "message": message,
            "data_label": "public_benchmark_data_not_customer_validated",
            **metadata,
        },
        "m1_case_source": {
            "source": source,
            "available": False,
            "error_type": error_type,
        },
        "m2_case_source": {
            "source": source,
            "available": False,
            "error_type": error_type,
        },
        "limitations": [
            "m2_v1_does_not_model_new_connection_in_dynamic_case",
            IEEE118_PUBLIC_LIMITATION,
            "unavailable",
        ],
        "linkage": {
            "m1_models_new_connection": False,
            "m2_models_new_connection": False,
            "m2_linkage": "dynamic_interconnection_model_not_applied",
            "note": "Strict IEEE118 M1+M2 alignment could not run because public case data was unavailable or failed validation.",
        },
        "summary": {
            "m1_recommendation": None,
            "m2_status": "unavailable",
            "m2_stability_status": "error",
            "transient_required_for_approval": transient_request.required_for_approval,
        },
    }


def _case_alignment_report(
    *,
    strict_alignment: bool,
    alignment_metadata: Optional[Mapping[str, Any]],
    m2_stage: Mapping[str, Any],
) -> Dict[str, Any]:
    if not strict_alignment:
        return {
            "strict_match": False,
            "source": None,
        }
    m2_source = _m2_case_source(m2_stage)
    return {
        "strict_match": True,
        "source": IEEE118_PUBLIC_SOURCE,
        "available": True,
        "data_label": "public_benchmark_data_not_customer_validated",
        "m1_case_source": _m1_case_source(True, alignment_metadata),
        "m2_case_source": m2_source,
        **dict(alignment_metadata or {}),
    }


def _m1_case_source(
    strict_alignment: bool,
    alignment_metadata: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not strict_alignment:
        return None
    return {
        **dict(alignment_metadata or {}),
        "m1_static_conversion": "andes_to_pandapower",
    }


def _m2_case_source(m2_stage: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    result = m2_stage.get("result")
    if not isinstance(result, Mapping):
        return None
    case_info = result.get("case_info")
    if not isinstance(case_info, Mapping):
        return None
    keys = {
        "case_id",
        "case_source",
        "source",
        "resolved_case_file",
        "resolved_addfile",
        "raw_sha256",
        "dyr_sha256",
        "dyr_variant",
        "data_label",
    }
    return {key: case_info.get(key) for key in keys if key in case_info}


def _m1_stage_report(result: Mapping[str, Any]) -> Dict[str, Any]:
    ok = bool(result.get("ok", False))
    recommendation = result.get("recommendation")
    passed = ok and recommendation == "approve"
    if not ok:
        status = "error"
    elif recommendation == "approve":
        status = "pass"
    elif recommendation == "borderline":
        status = "borderline"
    else:
        status = "fail"
    return {
        "stage": "m1_steady_state_cia",
        "status": status,
        "passed": passed,
        "recommendation": recommendation,
        "required_for_approval": True,
        "reason_codes": _list_of_strings(result.get("reason_codes")),
    }


def _integrated_recommendation(
    m1_stage: Mapping[str, Any],
    m2_stage: Mapping[str, Any],
    emt_stage: Mapping[str, Any],
    *,
    transient_request: TransientAssessmentRequest,
    emt_request: EMTAssessmentRequest,
) -> tuple[str, bool, list[str]]:
    reason_codes = ["integrated_assessment"]
    m1_status = m1_stage.get("status")
    m2_status = m2_stage.get("status")
    emt_status = emt_stage.get("status")
    reason_codes.extend(f"m1_{code}" for code in _list_of_strings(m1_stage.get("reason_codes")))
    reason_codes.extend(f"m2_{code}" for code in _list_of_strings(m2_stage.get("reason_codes")))
    reason_codes.extend(f"emt_{code}" for code in _list_of_strings(emt_stage.get("reason_codes")))

    if m1_status in {"error", "fail"}:
        reason_codes.append("m1_not_approved")
        return "reject", bool(m1_status != "error"), reason_codes
    if m1_status == "borderline":
        reason_codes.append("m1_borderline")
        return "borderline", True, reason_codes

    if transient_request.enabled:
        if m2_status == "borderline":
            reason_codes.append("m2_borderline")
            return "borderline", True, reason_codes
        if m2_status == "fail":
            reason_codes.append("m2_failed")
            return "reject", True, reason_codes
        if m2_status != "pass":
            if transient_request.required_for_approval:
                reason_codes.append("required_m2_unavailable")
                return "borderline", False, reason_codes
            reason_codes.append("m2_unavailable_not_required")

    if emt_request.enabled:
        if emt_status == "pass" or emt_status == "skipped":
            reason_codes.append("emt_screen_passed")
        elif emt_status == "borderline":
            reason_codes.append("emt_borderline")
            return "borderline", True, reason_codes
        elif emt_status == "fail":
            reason_codes.append("emt_failed")
            return "reject", True, reason_codes
        elif emt_request.required_for_approval:
            reason_codes.append("required_emt_unavailable")
            return "borderline", False, reason_codes
        else:
            reason_codes.append("emt_unavailable_not_required")

    if transient_request.enabled and m2_status == "pass":
        reason_codes.append("m1_approved_m2_passed")
    elif not transient_request.enabled and not emt_request.enabled:
        reason_codes.append("m1_only_assessment")
    return "approve", True, reason_codes


def _integrated_limitations(
    m2_stage: Mapping[str, Any],
    transient_request: TransientAssessmentRequest,
    emt_stage: Mapping[str, Any],
    emt_request: EMTAssessmentRequest,
) -> list[str]:
    limitations = []
    if not transient_request.enabled:
        limitations.append("transient_stability_not_requested")
    elif m2_stage.get("dynamic_interconnection_modeling"):
        limitations.append("m2_connection_model_is_static_pq_not_full_dynamic_resource")
        result = m2_stage.get("result")
        if isinstance(result, Mapping):
            limitations.extend(_list_of_strings(result.get("limitations")))
    else:
        limitations.append(
            "m2_static_pq_connection_model_not_applied_to_dynamic_case"
        )
    if m2_stage.get("status") in {"not_ready", "unavailable", "skipped"}:
        limitations.append(f"m2_{m2_stage.get('status')}")
    if not emt_request.enabled:
        limitations.append("emt_screening_not_requested")
    else:
        result = emt_stage.get("result")
        if isinstance(result, Mapping):
            limitations.extend(_list_of_strings(result.get("limitations")))
        if emt_stage.get("status") in {"unavailable", "skipped"}:
            limitations.append(f"emt_{emt_stage.get('status')}")
    return list(dict.fromkeys(limitations))


def _emt_linkage(
    emt_stage: Mapping[str, Any],
    emt_request: EMTAssessmentRequest,
) -> str:
    if not emt_request.enabled:
        return "not_requested"
    if emt_stage.get("emt_interconnection_modeling"):
        return "post_connection_scr_screen"
    if emt_stage.get("status") in {"unavailable", "skipped"}:
        return "emt_screen_not_applied"
    return "standalone_emt_scr_screen"


def _m2_linkage(
    m2_stage: Mapping[str, Any],
    transient_request: TransientAssessmentRequest,
) -> str:
    if not transient_request.enabled:
        return "not_requested"
    if m2_stage.get("dynamic_interconnection_modeling"):
        return "dynamic_case_with_static_interconnection_model"
    if m2_stage.get("status") in {"not_ready", "unavailable", "skipped"}:
        return "dynamic_interconnection_model_not_applied"
    return "standalone_dynamic_case_screen"


def _m2_linkage_note(
    m2_models_connection: bool,
    transient_request: TransientAssessmentRequest,
) -> str:
    if not transient_request.enabled:
        return "M2 transient stability was not requested."
    if m2_models_connection:
        return (
            "M2 v1 adds the proposed connection to the selected ANDES dynamic case "
            "as a static PQ load/injection before PFlow and TDS. It does not add "
            "detailed machine or inverter controller dynamics for the new resource."
        )
    return (
        "M2 v1 did not apply a dynamic-case interconnection model for this run; "
        "check the M2 stage reason codes and limitations."
    )


def _nested_get(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _optional_bool(payload: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"transient.{key} must be a boolean")
    return value


def _optional_int(payload: Mapping[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"transient.{key} must be an integer")
    return value


def _optional_mapping(payload: Mapping[str, Any], key: str) -> Optional[Mapping[str, Any]]:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"transient.{key} must be an object")
    return dict(value)


def _optional_number(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"transient.{key} must be numeric")
    return float(value)


def _optional_str(payload: Mapping[str, Any], key: str) -> Optional[str]:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"transient.{key} must be a non-empty string")
    return value.strip()


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]

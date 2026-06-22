"""Connection impact assessment pipeline.

Step 4 implements Grid-Mind's first fidelity stage: steady-state AC power
flow before and after a proposed connection, followed by baseline-aware
violation comparison and a structured recommendation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .contingency import (
    ContingencyRunner,
    compare_contingency_studies,
    contingency_stage_report,
)
from .emt import EMTScreeningError, EMTScreeningRunner
from .solver import GridSolver, PowerFlowError, SolverDependencyError
from .violations import InspectionReport, LimitProfile, ViolationInspector


CONNECTION_TYPES = {"load", "solar", "wind", "bess", "hybrid", "synchronous"}
IBR_TYPES = {"solar", "wind", "bess", "hybrid"}
DEFAULT_WORSENING_THRESHOLD_PERCENT = 2.0


@dataclass(frozen=True)
class ConnectionRequest:
    """Structured interconnection request extracted from user/tool input."""

    bus: int
    p_mw: float
    connection_type: str
    is_ibr: bool
    q_mvar: float = 0.0
    name: Optional[str] = None
    vm_pu: float = 1.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ConnectionRequest":
        missing = [key for key in ("bus", "p_mw", "connection_type", "is_ibr") if key not in payload]
        if missing:
            raise ValueError(f"Missing required connection field(s): {', '.join(missing)}")

        if not isinstance(payload["bus"], int) or isinstance(payload["bus"], bool):
            raise ValueError("connection.bus must be an integer")

        if not isinstance(payload["connection_type"], str):
            raise ValueError("connection.connection_type must be a string")
        connection_type = payload["connection_type"].strip().lower()
        if connection_type not in CONNECTION_TYPES:
            raise ValueError(
                "connection_type must be one of: "
                + ", ".join(sorted(CONNECTION_TYPES))
            )

        if not isinstance(payload["is_ibr"], bool):
            raise ValueError("connection.is_ibr must be a boolean")

        p_mw = _required_numeric(payload, "p_mw")
        if p_mw < 0:
            raise ValueError("connection.p_mw must be non-negative")

        q_mvar = _optional_numeric(payload, "q_mvar", 0.0)
        vm_pu = _optional_numeric(payload, "vm_pu", 1.0)
        name = payload.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError("connection.name must be a string")

        return cls(
            bus=payload["bus"],
            p_mw=p_mw,
            connection_type=connection_type,
            is_ibr=bool(payload["is_ibr"]),
            q_mvar=q_mvar,
            name=name,
            vm_pu=vm_pu,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bus": self.bus,
            "p_mw": self.p_mw,
            "connection_type": self.connection_type,
            "is_ibr": self.is_ibr,
            "q_mvar": self.q_mvar,
            "name": self.name,
            "vm_pu": self.vm_pu,
        }


class SteadyStateCIARunner:
    """Run the baseline-aware steady-state slice of Grid-Mind CIA."""

    def __init__(
        self,
        solver_factory: Callable[[], GridSolver],
        limit_profile: Optional[LimitProfile] = None,
        material_worsening_threshold_percent: float = DEFAULT_WORSENING_THRESHOLD_PERCENT,
    ) -> None:
        self.solver_factory = solver_factory
        self.limit_profile = limit_profile or LimitProfile.normal()
        if material_worsening_threshold_percent < 0:
            raise ValueError("material_worsening_threshold_percent must be non-negative")
        if not math.isfinite(material_worsening_threshold_percent):
            raise ValueError("material_worsening_threshold_percent must be finite")
        self.material_worsening_threshold_percent = material_worsening_threshold_percent

    def run(
        self,
        case_path: str,
        connection: ConnectionRequest,
        enable_contingency: bool = False,
        enable_transient: bool = False,
        enable_emt: bool = False,
        max_violations: int = 50,
        max_contingencies: int = -1,
        max_failed_contingencies: int = 50,
        fail_on_contingency_material_worsening: bool = False,
    ) -> Dict[str, Any]:
        baseline_solver = self.solver_factory()
        baseline_solver.load_case(case_path)
        try:
            baseline_powerflow = baseline_solver.run_powerflow()
        except PowerFlowError as exc:
            return {
                "ok": False,
                "error_type": "baseline_powerflow_failed",
                "error": str(exc),
                "case_path": case_path,
                "connection": connection.to_dict(),
            }

        inspector = ViolationInspector(self.limit_profile)
        baseline_report = inspector.inspect_solver(baseline_solver)
        connection_warnings = _connection_warnings(connection)
        effective_is_ibr = _effective_is_ibr(connection)

        study_solver = self.solver_factory()
        study_solver.load_case(case_path)
        try:
            added_connection = study_solver.add_connection(**connection.to_dict())
            post_powerflow = study_solver.run_powerflow()
            post_report = inspector.inspect_solver(study_solver)
        except PowerFlowError as exc:
            return self._post_nonconvergence_report(
                case_path=case_path,
                connection=connection,
                baseline_powerflow=baseline_powerflow,
                baseline_report=baseline_report,
                error=str(exc),
                connection_warnings=connection_warnings,
                requested_stages=_requested_stages(enable_contingency, enable_transient, enable_emt),
                max_violations=max_violations,
            )

        comparison = compare_violation_reports(
            baseline_report,
            post_report,
            material_worsening_threshold_percent=self.material_worsening_threshold_percent,
        )
        stage_report = _steady_state_stage_report(
            baseline_powerflow=baseline_powerflow,
            post_powerflow=post_powerflow,
            baseline_report=baseline_report,
            post_report=post_report,
            comparison=comparison,
            max_violations=max_violations,
        )
        downstream_stages = self._downstream_stage_reports(
            case_path=case_path,
            connection=connection,
            steady_state_passed=stage_report["passed"],
            enable_contingency=enable_contingency,
            enable_transient=enable_transient,
            enable_emt=enable_emt,
            is_ibr=effective_is_ibr,
            max_violations=max_violations,
            max_contingencies=max_contingencies,
            max_failed_contingencies=max_failed_contingencies,
            fail_on_contingency_material_worsening=fail_on_contingency_material_worsening,
        )
        recommendation, reason_codes, complete = _recommend(stage_report, downstream_stages)

        return {
            "ok": True,
            "tool": "run_cia",
            "case_path": case_path,
            "connection": connection.to_dict(),
            "connection_classification": {"effective_is_ibr": effective_is_ibr},
            "connection_warnings": connection_warnings,
            "added_connection": added_connection,
            "recommendation": recommendation,
            "complete": complete,
            "reason_codes": reason_codes,
            "stage_reports": [stage_report] + downstream_stages,
            "summary": {
                "baseline_status": baseline_report.status,
                "post_status": post_report.status,
                "project_hard_violations": comparison["project_hard_count"],
                "project_borderline_violations": comparison["project_borderline_count"],
                "pre_existing_hard_violations": baseline_report.hard_count,
                "pre_existing_borderline_violations": baseline_report.borderline_count,
            },
        }

    def _downstream_stage_reports(
        self,
        *,
        case_path: str,
        connection: ConnectionRequest,
        steady_state_passed: bool,
        enable_contingency: bool,
        enable_transient: bool,
        enable_emt: bool,
        is_ibr: bool,
        max_violations: int,
        max_contingencies: int,
        max_failed_contingencies: int,
        fail_on_contingency_material_worsening: bool,
    ) -> List[Dict[str, Any]]:
        if not steady_state_passed:
            return [
                _not_run_stage("f2_n1_contingency", "upstream_steady_state_failed")
                if enable_contingency
                else _skipped_stage("f2_n1_contingency", "contingency_disabled"),
            ] + _dynamic_stage_reports(
                case_path=case_path,
                connection=connection,
                solver_factory=self.solver_factory,
                enable_transient=enable_transient,
                enable_emt=enable_emt,
                is_ibr=is_ibr,
                blocked_reason="upstream_steady_state_failed",
            )

        if enable_contingency:
            contingency_runner = ContingencyRunner(
                solver_factory=self.solver_factory,
                limit_profile=LimitProfile.emergency(),
            )
            baseline_study = contingency_runner.run(
                case_path,
                max_contingencies=max_contingencies,
                max_violations=max_violations,
                max_failed_contingencies=max_failed_contingencies,
            )
            post_study = contingency_runner.run(
                case_path,
                connection=connection.to_dict(),
                max_contingencies=max_contingencies,
                max_violations=max_violations,
                max_failed_contingencies=max_failed_contingencies,
            )
            comparison = compare_contingency_studies(
                baseline_study,
                post_study,
                material_worsening_threshold_percent=self.material_worsening_threshold_percent,
                fail_on_material_worsening=fail_on_contingency_material_worsening,
            )
            n1_stage = contingency_stage_report(
                baseline_study,
                post_study,
                comparison,
                max_failed_contingencies=max_failed_contingencies,
            )
        else:
            n1_stage = _skipped_stage("f2_n1_contingency", "contingency_disabled")

        blocked_reason = "upstream_contingency_failed" if n1_stage["status"] == "fail" else None
        return [n1_stage] + _dynamic_stage_reports(
            case_path=case_path,
            connection=connection,
            solver_factory=self.solver_factory,
            enable_transient=enable_transient,
            enable_emt=enable_emt,
            is_ibr=is_ibr,
            blocked_reason=blocked_reason,
        )

    def _post_nonconvergence_report(
        self,
        *,
        case_path: str,
        connection: ConnectionRequest,
        baseline_powerflow: Dict[str, Any],
        baseline_report: InspectionReport,
        error: str,
        connection_warnings: List[str],
        requested_stages: List[str],
        max_violations: int,
    ) -> Dict[str, Any]:
        stage_report = {
            "stage": "f1_steady_state",
            "status": "fail",
            "passed": False,
            "reason_codes": ["post_powerflow_non_convergence"],
            "baseline_powerflow": baseline_powerflow,
            "baseline_violation_report": _limit_violation_report(
                baseline_report.to_dict(), max_violations
            ),
            "post_powerflow": {
                "converged": False,
                "error": error,
            },
            "project_violation_comparison": {
                "material_worsening_threshold_percent": self.material_worsening_threshold_percent,
                "baseline_total_violations": len(baseline_report.violations),
                "post_total_violations": None,
                "project_caused_violations": _limit_items([], max_violations),
                "project_hard_count": 0,
                "project_borderline_count": 0,
            },
        }
        downstream_stages = [
            _not_run_stage(stage, "upstream_steady_state_failed") for stage in requested_stages
        ]
        return {
            "ok": True,
            "tool": "run_cia",
            "case_path": case_path,
            "connection": connection.to_dict(),
            "connection_classification": {
                "effective_is_ibr": _effective_is_ibr(connection),
            },
            "connection_warnings": connection_warnings,
            "recommendation": "reject",
            "complete": len(requested_stages) == 0,
            "reason_codes": ["post_powerflow_non_convergence"],
            "stage_reports": [stage_report] + downstream_stages,
            "summary": {
                "baseline_status": baseline_report.status,
                "post_status": "non_converged",
                "project_hard_violations": 0,
                "project_borderline_violations": 0,
                "pre_existing_hard_violations": baseline_report.hard_count,
                "pre_existing_borderline_violations": baseline_report.borderline_count,
            },
        }


def compare_violation_reports(
    baseline_report: InspectionReport,
    post_report: InspectionReport,
    material_worsening_threshold_percent: float = DEFAULT_WORSENING_THRESHOLD_PERCENT,
) -> Dict[str, Any]:
    """Compare post-connection violations against baseline violations."""

    baseline_by_key = {
        _violation_key(violation.to_dict()): violation.to_dict()
        for violation in baseline_report.violations
    }
    project_caused: List[Dict[str, Any]] = []

    for violation in post_report.violations:
        post = violation.to_dict()
        key = _violation_key(post)
        baseline = baseline_by_key.get(key)
        change_type = _project_change_type(
            baseline,
            post,
            material_worsening_threshold_percent,
        )
        if change_type is None:
            continue

        delta = None
        if baseline is not None:
            delta = post["margin_percent"] - baseline["margin_percent"]
        project_caused.append(
            {
                "change_type": change_type,
                "severity": post["severity"],
                "post_violation": post,
                "baseline_violation": baseline,
                "margin_percent_delta": delta,
            }
        )

    project_caused.sort(key=_project_caused_sort_key, reverse=True)
    hard_count = sum(1 for item in project_caused if item["severity"] == "hard")
    borderline_count = sum(1 for item in project_caused if item["severity"] == "borderline")
    return {
        "material_worsening_threshold_percent": material_worsening_threshold_percent,
        "baseline_total_violations": len(baseline_report.violations),
        "post_total_violations": len(post_report.violations),
        "project_caused_violations": project_caused,
        "project_hard_count": hard_count,
        "project_borderline_count": borderline_count,
    }


def _project_change_type(
    baseline: Optional[Mapping[str, Any]],
    post: Mapping[str, Any],
    material_worsening_threshold_percent: float,
) -> Optional[str]:
    if baseline is None:
        return "new_violation"

    if _severity_rank(post["severity"]) > _severity_rank(str(baseline["severity"])):
        return "severity_worsened"

    if float(baseline["margin_percent"]) <= 0 < float(post["margin_percent"]):
        return "limit_crossed_within_borderline"

    margin_delta = float(post["margin_percent"]) - float(baseline["margin_percent"])
    if margin_delta > material_worsening_threshold_percent:
        return "material_worsening"

    return None


def _required_numeric(payload: Mapping[str, Any], key: str) -> float:
    return _numeric_value(payload[key], key)


def _optional_numeric(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    return _numeric_value(value, key)


def _numeric_value(value: Any, key: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"connection.{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"connection.{key} must be finite")
    return result


def _effective_is_ibr(connection: ConnectionRequest) -> bool:
    return connection.is_ibr or connection.connection_type in IBR_TYPES


def _connection_warnings(connection: ConnectionRequest) -> List[str]:
    if connection.connection_type in IBR_TYPES and not connection.is_ibr:
        return ["ibr_flag_false_for_inverter_based_connection_type"]
    if connection.connection_type == "synchronous" and connection.is_ibr:
        return ["ibr_flag_true_for_synchronous_connection_type"]
    return []


def _steady_state_stage_report(
    *,
    baseline_powerflow: Dict[str, Any],
    post_powerflow: Dict[str, Any],
    baseline_report: InspectionReport,
    post_report: InspectionReport,
    comparison: Dict[str, Any],
    max_violations: int,
) -> Dict[str, Any]:
    hard_count = comparison["project_hard_count"]
    borderline_count = comparison["project_borderline_count"]
    if hard_count:
        status = "fail"
    elif borderline_count:
        status = "borderline"
    else:
        status = "pass"

    reason_codes = []
    if hard_count:
        reason_codes.append("steady_state_project_hard_violations")
    if borderline_count:
        reason_codes.append("steady_state_project_borderline_violations")
    if not reason_codes:
        reason_codes.append("steady_state_no_project_caused_violations")
    if baseline_report.hard_count:
        reason_codes.append("baseline_pre_existing_hard_violations")
    elif baseline_report.borderline_count:
        reason_codes.append("baseline_pre_existing_borderline_violations")

    limited_comparison = dict(comparison)
    limited_comparison["project_caused_violations"] = _limit_items(
        comparison["project_caused_violations"], max_violations
    )

    return {
        "stage": "f1_steady_state",
        "status": status,
        "passed": hard_count == 0,
        "reason_codes": reason_codes,
        "baseline_powerflow": baseline_powerflow,
        "post_powerflow": post_powerflow,
        "baseline_violation_report": _limit_violation_report(
            baseline_report.to_dict(), max_violations
        ),
        "post_violation_report": _limit_violation_report(post_report.to_dict(), max_violations),
        "project_violation_comparison": limited_comparison,
    }


def _recommend(
    steady_state_stage: Mapping[str, Any],
    downstream_stages: Iterable[Mapping[str, Any]],
) -> Tuple[str, List[str], bool]:
    reason_codes = list(steady_state_stage["reason_codes"])
    if steady_state_stage["status"] == "fail":
        return "reject", reason_codes, _downstream_complete(downstream_stages)

    failed_stages = [stage for stage in downstream_stages if stage["status"] == "fail"]
    for stage in failed_stages:
        reason_codes.extend(stage["reason_codes"])
    if failed_stages:
        return "reject", reason_codes, _downstream_complete(downstream_stages)

    incomplete_stages = [stage for stage in downstream_stages if stage["status"] == "not_implemented"]
    for stage in incomplete_stages:
        reason_codes.extend(stage["reason_codes"])
    if incomplete_stages:
        return "borderline", reason_codes, False

    if steady_state_stage["status"] == "borderline":
        return "borderline", reason_codes, True
    return "approve", reason_codes, True


def _dynamic_stage_reports(
    *,
    case_path: str,
    connection: ConnectionRequest,
    solver_factory: Callable[[], GridSolver],
    enable_transient: bool,
    enable_emt: bool,
    is_ibr: bool,
    blocked_reason: Optional[str] = None,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []

    if blocked_reason is not None and enable_transient and is_ibr:
        reports.append(_not_run_stage("f3_transient_stability", blocked_reason))
    elif enable_transient and is_ibr:
        reports.append(_not_implemented_stage("f3_transient_stability"))
    else:
        reason = "non_ibr_request" if enable_transient and not is_ibr else "transient_disabled"
        reports.append(_skipped_stage("f3_transient_stability", reason))

    if blocked_reason is not None and enable_emt and is_ibr:
        reports.append(_not_run_stage("f4_emt_screening", blocked_reason))
    elif enable_emt and is_ibr:
        reports.append(_emt_stage_report(case_path, connection, solver_factory))
    else:
        reason = "non_ibr_request" if enable_emt and not is_ibr else "emt_disabled"
        reports.append(_skipped_stage("f4_emt_screening", reason))

    return reports


def _emt_stage_report(
    case_path: str,
    connection: ConnectionRequest,
    solver_factory: Callable[[], GridSolver],
) -> Dict[str, Any]:
    try:
        result = EMTScreeningRunner(solver_factory=solver_factory).run(
            case_path=case_path,
            connection=connection,
        )
    except (EMTScreeningError, PowerFlowError, SolverDependencyError, ValueError) as exc:
        return {
            "stage": "f4_emt_screening",
            "status": "unavailable",
            "passed": False,
            "reason_codes": ["emt_screening_unavailable"],
            "result": {
                "ok": False,
                "tool": "run_emt_screening",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        }

    status = str(result.get("emt", {}).get("status", "unavailable"))
    reason_codes = result.get("metrics", {}).get("reason_codes")
    if not isinstance(reason_codes, list):
        reason_codes = ["emt_result_without_reason_codes"]
    return {
        "stage": "f4_emt_screening",
        "status": status,
        "passed": bool(result.get("emt", {}).get("passed", False)),
        "reason_codes": [str(code) for code in reason_codes],
        "result": result,
    }


def _requested_stages(
    enable_contingency: bool,
    enable_transient: bool,
    enable_emt: bool,
) -> List[str]:
    stages = []
    if enable_contingency:
        stages.append("f2_n1_contingency")
    if enable_transient:
        stages.append("f3_transient_stability")
    if enable_emt:
        stages.append("f4_emt_screening")
    return stages


def _not_implemented_stage(stage: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "status": "not_implemented",
        "passed": False,
        "reason_codes": [f"{stage}_not_implemented"],
    }


def _not_run_stage(stage: str, reason: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "status": "not_run",
        "passed": False,
        "reason_codes": [reason],
    }


def _skipped_stage(stage: str, reason: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "status": "skipped",
        "passed": True,
        "reason_codes": [reason],
    }


def _downstream_complete(stages: Iterable[Mapping[str, Any]]) -> bool:
    return not any(stage["status"] == "not_implemented" for stage in stages)


def _violation_key(violation: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        violation.get("element_type"),
        violation.get("element_index"),
        violation.get("violation_type"),
        violation.get("limit_relation"),
        round(float(violation.get("limit_value", 0.0)), 9),
    )


def _severity_rank(severity: str) -> int:
    return {"borderline": 1, "hard": 2}.get(severity, 0)


def _project_caused_sort_key(item: Mapping[str, Any]) -> Tuple[int, float]:
    post = item.get("post_violation", {})
    margin_percent = post.get("margin_percent", 0.0)
    try:
        margin = float(margin_percent)
    except Exception:
        margin = 0.0
    return (_severity_rank(str(item.get("severity"))), margin)


def _limit_items(items: List[Dict[str, Any]], max_items: int) -> Dict[str, Any]:
    limited = items if max_items < 0 else items[:max_items]
    return {
        "items": limited,
        "total_items": len(items),
        "truncated_items": max(0, len(items) - len(limited)),
    }


def _limit_violation_report(report: Dict[str, Any], max_violations: int) -> Dict[str, Any]:
    violations = list(report.get("violations", []))
    limited = violations if max_violations < 0 else violations[:max_violations]
    report["violations"] = limited
    report["truncated_violations"] = max(0, len(violations) - len(limited))
    return report

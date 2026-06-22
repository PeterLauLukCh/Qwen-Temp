"""Capacity search for the Mini Grid-Mind reproduction.

Step 6 implements Grid-Mind's binary-search capacity tool. Every sampled MW
value is evaluated through the CIA pipeline so returned limits stay grounded in
deterministic solver and violation-inspector outputs.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional

from .cia import CONNECTION_TYPES, IBR_TYPES, ConnectionRequest, SteadyStateCIARunner
from .solver import GridSolver
from .violations import LimitProfile


class CapacitySearchRunner:
    """Find the largest MW value accepted by the CIA pipeline."""

    def __init__(
        self,
        solver_factory: Callable[[], GridSolver],
        limit_profile: Optional[LimitProfile] = None,
        material_worsening_threshold_percent: float = 2.0,
    ) -> None:
        threshold = _finite_number(
            "material_worsening_threshold_percent",
            material_worsening_threshold_percent,
        )
        if threshold < 0:
            raise ValueError("material_worsening_threshold_percent must be non-negative")
        self.solver_factory = solver_factory
        self.limit_profile = limit_profile or LimitProfile.normal()
        self.material_worsening_threshold_percent = threshold

    def run(
        self,
        *,
        case_path: str,
        bus: int,
        connection_type: str,
        is_ibr: Optional[bool] = None,
        q_mvar: float = 0.0,
        vm_pu: float = 1.0,
        min_mw: float = 0.0,
        max_mw: float = 500.0,
        tolerance_mw: float = 1.0,
        max_iterations: int = 12,
        coarse_scan_points: int = 11,
        enable_contingency: bool = False,
        max_contingencies: int = -1,
        max_failed_contingencies: int = 50,
        fail_on_contingency_material_worsening: bool = False,
        max_violations: int = 10,
        max_samples: int = 100,
        include_reports: bool = False,
    ) -> Dict[str, Any]:
        case_path = _case_path_argument(case_path)
        bus = _integer_argument("bus", bus)
        resource_type = _connection_type_argument(connection_type)
        q_mvar = _finite_number("q_mvar", q_mvar)
        vm_pu = _finite_number("vm_pu", vm_pu)
        min_mw = _finite_number("min_mw", min_mw)
        max_mw = _finite_number("max_mw", max_mw)
        tolerance_mw = _finite_number("tolerance_mw", tolerance_mw)
        max_iterations = _integer_argument("max_iterations", max_iterations)
        coarse_scan_points = _integer_argument("coarse_scan_points", coarse_scan_points)
        max_contingencies = _integer_argument("max_contingencies", max_contingencies)
        max_failed_contingencies = _integer_argument(
            "max_failed_contingencies", max_failed_contingencies
        )
        enable_contingency = _boolean_argument("enable_contingency", enable_contingency)
        fail_on_contingency_material_worsening = _boolean_argument(
            "fail_on_contingency_material_worsening",
            fail_on_contingency_material_worsening,
        )
        max_violations = _integer_argument("max_violations", max_violations)
        max_samples = _integer_argument("max_samples", max_samples)
        include_reports = _boolean_argument("include_reports", include_reports)
        if is_ibr is not None and not isinstance(is_ibr, bool):
            raise ValueError("is_ibr must be a boolean when provided")

        _validate_search_bounds(min_mw, max_mw, tolerance_mw, max_iterations, coarse_scan_points)
        effective_is_ibr = is_ibr if is_ibr is not None else resource_type in IBR_TYPES

        cia_runner = SteadyStateCIARunner(
            solver_factory=self.solver_factory,
            limit_profile=self.limit_profile,
            material_worsening_threshold_percent=self.material_worsening_threshold_percent,
        )
        samples: List[Dict[str, Any]] = []
        samples_by_key: Dict[float, Dict[str, Any]] = {}

        def evaluate(mw: float) -> Dict[str, Any]:
            key = round(float(mw), 9)
            if key in samples_by_key:
                return samples_by_key[key]
            connection = ConnectionRequest(
                bus=bus,
                p_mw=float(mw),
                connection_type=resource_type,
                is_ibr=effective_is_ibr,
                q_mvar=q_mvar,
                vm_pu=vm_pu,
            )
            report = cia_runner.run(
                case_path=case_path,
                connection=connection,
                enable_contingency=enable_contingency,
                max_violations=max_violations,
                max_contingencies=max_contingencies,
                max_failed_contingencies=max_failed_contingencies,
                fail_on_contingency_material_worsening=fail_on_contingency_material_worsening,
            )
            sample = _sample_from_report(float(mw), report, include_reports=include_reports)
            samples.append(sample)
            samples_by_key[key] = sample
            return sample

        lower_sample = evaluate(min_mw)
        upper_sample = evaluate(max_mw)
        contradictions = _monotonicity_contradictions(samples)
        if contradictions:
            return self._fallback_scan_result(
                case_path=case_path,
                bus=bus,
                connection_type=resource_type,
                is_ibr=effective_is_ibr,
                min_mw=min_mw,
                max_mw=max_mw,
                tolerance_mw=tolerance_mw,
                coarse_scan_points=coarse_scan_points,
                enable_contingency=enable_contingency,
                evaluate=evaluate,
                samples=samples,
                initial_contradictions=contradictions,
                bisection_iterations_before_fallback=0,
                max_samples=max_samples,
            )

        if not lower_sample["accepted"]:
            return _result(
                case_path=case_path,
                bus=bus,
                connection_type=resource_type,
                is_ibr=effective_is_ibr,
                status="min_bound_rejected",
                max_approved_mw=None,
                lower_bound_mw=None,
                upper_bound_mw=float(min_mw),
                tolerance_mw=tolerance_mw,
                iterations=0,
                samples=samples,
                max_samples=max_samples,
                boundary_samples={"best_approved": None, "first_rejected": lower_sample},
                diagnostics={
                    "fallback_used": False,
                    "monotonicity_contradictions": [],
                    "message": "Minimum MW bound is not approved by CIA.",
                },
            )

        if upper_sample["accepted"]:
            return _result(
                case_path=case_path,
                bus=bus,
                connection_type=resource_type,
                is_ibr=effective_is_ibr,
                status="max_bound_approved",
                max_approved_mw=float(max_mw),
                lower_bound_mw=float(max_mw),
                upper_bound_mw=None,
                tolerance_mw=tolerance_mw,
                iterations=0,
                samples=samples,
                max_samples=max_samples,
                boundary_samples={"best_approved": upper_sample, "first_rejected": None},
                diagnostics={
                    "fallback_used": False,
                    "monotonicity_contradictions": [],
                    "message": "Maximum MW bound is still approved by CIA.",
                },
            )

        lower_mw = float(min_mw)
        upper_mw = float(max_mw)
        iterations = 0
        first_rejected = upper_sample
        best_approved = lower_sample
        while (upper_mw - lower_mw) > tolerance_mw and iterations < max_iterations:
            midpoint = (lower_mw + upper_mw) / 2.0
            sample = evaluate(midpoint)
            contradictions = _monotonicity_contradictions(samples)
            if contradictions:
                return self._fallback_scan_result(
                    case_path=case_path,
                    bus=bus,
                    connection_type=resource_type,
                    is_ibr=effective_is_ibr,
                    min_mw=min_mw,
                    max_mw=max_mw,
                    tolerance_mw=tolerance_mw,
                    coarse_scan_points=coarse_scan_points,
                    enable_contingency=enable_contingency,
                    evaluate=evaluate,
                    samples=samples,
                    initial_contradictions=contradictions,
                    bisection_iterations_before_fallback=iterations + 1,
                    max_samples=max_samples,
                )
            if sample["accepted"]:
                lower_mw = float(sample["mw"])
                best_approved = sample
            else:
                upper_mw = float(sample["mw"])
                first_rejected = sample
            iterations += 1

        status = "bounded" if (upper_mw - lower_mw) <= tolerance_mw else "iteration_limit"
        return _result(
            case_path=case_path,
            bus=bus,
            connection_type=resource_type,
            is_ibr=effective_is_ibr,
            status=status,
            max_approved_mw=float(best_approved["mw"]),
            lower_bound_mw=lower_mw,
            upper_bound_mw=upper_mw,
            tolerance_mw=tolerance_mw,
            iterations=iterations,
            samples=samples,
            max_samples=max_samples,
            boundary_samples={"best_approved": best_approved, "first_rejected": first_rejected},
            diagnostics={
                "fallback_used": False,
                "monotonicity_contradictions": [],
                "message": "Bisection completed within tolerance."
                if status == "bounded"
                else "Bisection stopped at the iteration limit.",
            },
        )

    def _fallback_scan_result(
        self,
        *,
        case_path: str,
        bus: int,
        connection_type: str,
        is_ibr: bool,
        min_mw: float,
        max_mw: float,
        tolerance_mw: float,
        coarse_scan_points: int,
        enable_contingency: bool,
        evaluate: Any,
        samples: List[Dict[str, Any]],
        initial_contradictions: List[Dict[str, Any]],
        bisection_iterations_before_fallback: int,
        max_samples: int,
    ) -> Dict[str, Any]:
        denominator = max(1, coarse_scan_points - 1)
        samples_before_scan = len(samples)
        for index in range(coarse_scan_points):
            mw = min_mw + (max_mw - min_mw) * index / denominator
            evaluate(mw)
        coarse_scan_evaluations = len(samples) - samples_before_scan

        accepted_samples = [sample for sample in samples if sample["accepted"]]
        rejected_samples = [sample for sample in samples if not sample["accepted"]]
        best_approved = max(accepted_samples, key=lambda sample: sample["mw"]) if accepted_samples else None
        first_rejected = None
        if best_approved is not None:
            rejected_above = [
                sample for sample in rejected_samples if sample["mw"] > best_approved["mw"]
            ]
            if rejected_above:
                first_rejected = min(rejected_above, key=lambda sample: sample["mw"])
        elif rejected_samples:
            first_rejected = min(rejected_samples, key=lambda sample: sample["mw"])

        return _result(
            case_path=case_path,
            bus=bus,
            connection_type=connection_type,
            is_ibr=is_ibr,
            status="monotonicity_fallback",
            max_approved_mw=float(best_approved["mw"]) if best_approved is not None else None,
            lower_bound_mw=float(best_approved["mw"]) if best_approved is not None else None,
            upper_bound_mw=float(first_rejected["mw"]) if first_rejected is not None else None,
            tolerance_mw=tolerance_mw,
            iterations=bisection_iterations_before_fallback,
            samples=samples,
            max_samples=max_samples,
            boundary_samples={"best_approved": best_approved, "first_rejected": first_rejected},
            diagnostics={
                "fallback_used": True,
                "bisection_iterations_before_fallback": bisection_iterations_before_fallback,
                "coarse_scan_points": coarse_scan_points,
                "coarse_scan_evaluations": coarse_scan_evaluations,
                "enable_contingency": enable_contingency,
                "monotonicity_contradictions": initial_contradictions,
                "message": (
                    "Sampled CIA outcomes violated monotone bisection assumptions; "
                    "reported capacity comes from coarse scan."
                ),
            },
        )


def _sample_from_report(
    mw: float,
    report: Mapping[str, Any],
    *,
    include_reports: bool,
) -> Dict[str, Any]:
    recommendation = str(report.get("recommendation", "error"))
    accepted = bool(report.get("ok", False)) and recommendation == "approve"
    sample: Dict[str, Any] = {
        "mw": float(mw),
        "accepted": accepted,
        "recommendation": recommendation,
        "complete": bool(report.get("complete", False)),
        "reason_codes": list(report.get("reason_codes", [])),
        "limiting_summary": None if accepted else _limiting_summary(report),
    }
    if include_reports:
        sample["cia_report"] = dict(report)
    return sample


def _limiting_summary(report: Mapping[str, Any]) -> Dict[str, Any]:
    if not report.get("ok", False):
        return {
            "limiting_stage": "baseline",
            "status": "error",
            "reason_codes": [str(report.get("error_type", "cia_error"))],
            "message": str(report.get("error", "CIA did not complete.")),
        }

    for stage in report.get("stage_reports", []):
        status = stage.get("status")
        if status not in {"fail", "borderline", "not_implemented", "not_run"}:
            continue
        summary = {
            "limiting_stage": stage.get("stage"),
            "status": status,
            "reason_codes": list(stage.get("reason_codes", [])),
        }
        if stage.get("stage") == "f1_steady_state":
            comparison = stage.get("project_violation_comparison", {})
            summary["project_caused_violations"] = comparison.get("project_caused_violations")
            summary["post_powerflow"] = stage.get("post_powerflow")
        elif stage.get("stage") == "f2_n1_contingency":
            comparison = stage.get("project_contingency_comparison", {})
            summary["project_caused_failures"] = comparison.get("project_caused_failures")
        return summary

    return {
        "limiting_stage": None,
        "status": report.get("recommendation"),
        "reason_codes": list(report.get("reason_codes", [])),
    }


def _result(
    *,
    case_path: str,
    bus: int,
    connection_type: str,
    is_ibr: bool,
    status: str,
    max_approved_mw: Optional[float],
    lower_bound_mw: Optional[float],
    upper_bound_mw: Optional[float],
    tolerance_mw: float,
    iterations: int,
    samples: List[Dict[str, Any]],
    max_samples: int,
    boundary_samples: Mapping[str, Optional[Dict[str, Any]]],
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    first_rejected = boundary_samples.get("first_rejected")
    return {
        "ok": True,
        "tool": "find_max_capacity",
        "case_path": case_path,
        "request": {
            "bus": bus,
            "connection_type": connection_type,
            "is_ibr": is_ibr,
        },
        "status": status,
        "max_approved_mw": max_approved_mw,
        "lower_bound_mw": lower_bound_mw,
        "upper_bound_mw": upper_bound_mw,
        "tolerance_mw": tolerance_mw,
        "iterations": iterations,
        "boundary_samples": {
            "best_approved": boundary_samples.get("best_approved"),
            "first_rejected": first_rejected,
        },
        "rejection_explanation": None
        if first_rejected is None
        else first_rejected.get("limiting_summary"),
        "samples": _limit_items(sorted(samples, key=lambda sample: sample["mw"]), max_samples),
        "diagnostics": dict(diagnostics),
    }


def _monotonicity_contradictions(samples: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    contradictions = []
    ordered = sorted(samples, key=lambda sample: sample["mw"])
    for lower_index, lower in enumerate(ordered):
        if lower["accepted"]:
            continue
        for higher in ordered[lower_index + 1 :]:
            if higher["accepted"]:
                contradictions.append(
                    {
                        "lower_rejected_mw": lower["mw"],
                        "higher_approved_mw": higher["mw"],
                        "lower_recommendation": lower["recommendation"],
                        "higher_recommendation": higher["recommendation"],
                    }
                )
                break
    return contradictions


def _validate_search_bounds(
    min_mw: float,
    max_mw: float,
    tolerance_mw: float,
    max_iterations: int,
    coarse_scan_points: int,
) -> None:
    for key, value in {
        "min_mw": min_mw,
        "max_mw": max_mw,
        "tolerance_mw": tolerance_mw,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if min_mw < 0 or max_mw < 0:
        raise ValueError("min_mw and max_mw must be non-negative")
    if max_mw < min_mw:
        raise ValueError("max_mw must be greater than or equal to min_mw")
    if tolerance_mw <= 0:
        raise ValueError("tolerance_mw must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if coarse_scan_points < 2:
        raise ValueError("coarse_scan_points must be at least 2")


def _case_path_argument(case_path: Any) -> str:
    if not isinstance(case_path, str) or not case_path.strip():
        raise ValueError("case_path must be a non-empty string")
    return case_path.strip()


def _connection_type_argument(connection_type: Any) -> str:
    if not isinstance(connection_type, str):
        raise ValueError("connection_type must be a string")
    resource_type = connection_type.strip().lower()
    if resource_type not in CONNECTION_TYPES:
        raise ValueError(
            "connection_type must be one of: " + ", ".join(sorted(CONNECTION_TYPES))
        )
    return resource_type


def _finite_number(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer_argument(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _boolean_argument(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _limit_items(items: List[Dict[str, Any]], max_items: int) -> Dict[str, Any]:
    limited = items if max_items < 0 else items[:max_items]
    return {
        "items": limited,
        "total_items": len(items),
        "truncated_items": max(0, len(items) - len(limited)),
    }

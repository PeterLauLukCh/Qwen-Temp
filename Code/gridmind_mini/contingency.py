"""N-1 contingency screening for the Mini Grid-Mind reproduction.

Step 5 implements Grid-Mind's second CIA fidelity stage: systematic
single-element outages screened with emergency voltage and thermal limits.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .solver import GridSolver, PowerFlowError
from .violations import InspectionReport, LimitProfile, ViolationInspector


DEFAULT_MATERIAL_WORSENING_THRESHOLD_PERCENT = 2.0


class ContingencyRunner:
    """Run single line/transformer outage screening with emergency limits."""

    def __init__(
        self,
        solver_factory: Callable[[], GridSolver],
        limit_profile: Optional[LimitProfile] = None,
    ) -> None:
        self.solver_factory = solver_factory
        self.limit_profile = limit_profile or LimitProfile.emergency()

    def run(
        self,
        case_path: str,
        connection: Optional[Any] = None,
        max_contingencies: int = -1,
        max_violations: int = 50,
        max_failed_contingencies: int = 50,
    ) -> Dict[str, Any]:
        seed_solver = self.solver_factory()
        seed_solver.load_case(case_path)
        connection_kwargs = _connection_kwargs(connection)
        base_added_connection = None
        if connection_kwargs is not None:
            base_added_connection = seed_solver.add_connection(**connection_kwargs)
        base_powerflow = seed_solver.run_powerflow()
        available = seed_solver.available_contingencies()
        selected = available if max_contingencies < 0 else available[:max_contingencies]

        results = [
            self._run_one(case_path, contingency, connection_kwargs, max_violations)
            for contingency in selected
        ]
        failed = [result for result in results if not result["passed"]]
        borderline = [result for result in results if result["status"] == "borderline"]
        hard_violation_count = sum(result["hard_violation_count"] for result in results)
        non_converged_count = sum(1 for result in results if result["failure_mode"] == "non_convergence")

        return {
            "ok": True,
            "tool": "run_contingency",
            "stage": "f2_n1_contingency",
            "case_path": case_path,
            "limit_profile": self.limit_profile.to_dict(),
            "connection_applied": connection_kwargs is not None,
            "base_added_connection": base_added_connection,
            "base_powerflow": base_powerflow,
            "total_available_contingencies": len(available),
            "summary": {
                "evaluated_contingencies": len(results),
                "passed_contingencies": sum(1 for result in results if result["passed"]),
                "failed_contingencies": len(failed),
                "borderline_contingencies": len(borderline),
                "non_converged_contingencies": non_converged_count,
                "hard_violation_count": hard_violation_count,
            },
            "failed_contingencies": _limit_items(failed, max_failed_contingencies),
            "borderline_contingencies": _limit_items(borderline, max_failed_contingencies),
            "contingency_results": results,
        }

    def _run_one(
        self,
        case_path: str,
        contingency: Mapping[str, Any],
        connection_kwargs: Optional[Mapping[str, Any]],
        max_violations: int,
    ) -> Dict[str, Any]:
        solver = self.solver_factory()
        solver.load_case(case_path)
        added_connection = None
        if connection_kwargs is not None:
            added_connection = solver.add_connection(**connection_kwargs)

        applied = solver.apply_contingency(
            str(contingency["element_type"]),
            int(contingency["element_index"]),
        )
        try:
            powerflow = solver.run_powerflow()
        except PowerFlowError as exc:
            failure_item = _non_convergence_failure(applied, str(exc))
            return {
                "contingency": applied,
                "added_connection": added_connection,
                "status": "fail",
                "passed": False,
                "failure_mode": "non_convergence",
                "powerflow": {"converged": False, "error": str(exc)},
                "violation_report": None,
                "hard_violation_count": 0,
                "borderline_violation_count": 0,
                "failure_items": [failure_item],
                "max_failure_margin_percent": None,
            }

        report = ViolationInspector(self.limit_profile).inspect_solver(solver)
        failure_items = _hard_failure_items(applied, report)
        if failure_items:
            status = "fail"
        elif report.borderline_count:
            status = "borderline"
        else:
            status = "pass"

        return {
            "contingency": applied,
            "added_connection": added_connection,
            "status": status,
            "passed": not failure_items,
            "failure_mode": "hard_violations" if failure_items else None,
            "powerflow": powerflow,
            "violation_report": _limit_violation_report(report.to_dict(), max_violations),
            "hard_violation_count": report.hard_count,
            "borderline_violation_count": report.borderline_count,
            "failure_items": failure_items,
            "max_failure_margin_percent": _max_failure_margin_percent(failure_items),
        }


def compare_contingency_studies(
    baseline_study: Mapping[str, Any],
    post_study: Mapping[str, Any],
    material_worsening_threshold_percent: float = DEFAULT_MATERIAL_WORSENING_THRESHOLD_PERCENT,
    fail_on_material_worsening: bool = False,
) -> Dict[str, Any]:
    """Compare post-connection N-1 failures against baseline N-1 failures."""

    baseline_by_signature = {
        item["signature"]: item
        for result in baseline_study.get("contingency_results", [])
        for item in result.get("failure_items", [])
    }
    project_failures: List[Dict[str, Any]] = []
    material_worsenings: List[Dict[str, Any]] = []

    for result in post_study.get("contingency_results", []):
        for post_failure in result.get("failure_items", []):
            baseline_failure = baseline_by_signature.get(post_failure["signature"])
            if baseline_failure is None:
                project_failures.append(
                    _comparison_item("new_contingency_failure", post_failure, None)
                )
                continue

            delta = _margin_delta_percent(baseline_failure, post_failure)
            if delta is not None and delta > material_worsening_threshold_percent:
                item = _comparison_item(
                    "material_worsening",
                    post_failure,
                    baseline_failure,
                    margin_percent_delta=delta,
                )
                material_worsenings.append(item)
                if fail_on_material_worsening:
                    project_failures.append(item)

    project_failures.sort(key=_failure_sort_key, reverse=True)
    material_worsenings.sort(key=_failure_sort_key, reverse=True)
    return {
        "material_worsening_threshold_percent": material_worsening_threshold_percent,
        "fail_on_material_worsening": fail_on_material_worsening,
        "baseline_failed_contingencies": baseline_study.get("summary", {}).get("failed_contingencies", 0),
        "post_failed_contingencies": post_study.get("summary", {}).get("failed_contingencies", 0),
        "new_failure_count": sum(
            1 for item in project_failures if item["change_type"] == "new_contingency_failure"
        ),
        "material_worsening_count": len(material_worsenings),
        "project_failure_count": len(project_failures),
        "project_caused_failures": project_failures,
        "material_worsenings": material_worsenings,
    }


def contingency_stage_report(
    baseline_study: Mapping[str, Any],
    post_study: Mapping[str, Any],
    comparison: Mapping[str, Any],
    max_failed_contingencies: int = 50,
) -> Dict[str, Any]:
    project_failure_count = int(comparison["project_failure_count"])
    status = "fail" if project_failure_count else "pass"
    reason_codes = []
    if comparison["new_failure_count"]:
        reason_codes.append("n1_project_new_failures")
    if comparison["fail_on_material_worsening"] and comparison["material_worsening_count"]:
        reason_codes.append("n1_project_material_worsening")
    elif comparison["material_worsening_count"]:
        reason_codes.append("n1_material_worsening_observed")
    if not reason_codes:
        reason_codes.append("n1_no_project_caused_failures")

    limited_comparison = dict(comparison)
    limited_comparison["project_caused_failures"] = _limit_items(
        list(comparison["project_caused_failures"]),
        max_failed_contingencies,
    )
    limited_comparison["material_worsenings"] = _limit_items(
        list(comparison["material_worsenings"]),
        max_failed_contingencies,
    )

    return {
        "stage": "f2_n1_contingency",
        "status": status,
        "passed": project_failure_count == 0,
        "reason_codes": reason_codes,
        "baseline_contingency_summary": baseline_study["summary"],
        "post_contingency_summary": post_study["summary"],
        "project_contingency_comparison": limited_comparison,
        "baseline_failed_contingencies": baseline_study["failed_contingencies"],
        "post_failed_contingencies": post_study["failed_contingencies"],
    }


def _connection_kwargs(connection: Optional[Any]) -> Optional[Mapping[str, Any]]:
    if connection is None:
        return None
    if hasattr(connection, "to_dict"):
        return connection.to_dict()
    if isinstance(connection, Mapping):
        return dict(connection)
    raise ValueError("connection must be a mapping or expose to_dict()")


def _hard_failure_items(
    contingency: Mapping[str, Any],
    report: InspectionReport,
) -> List[Dict[str, Any]]:
    items = []
    for violation in report.violations:
        if violation.severity != "hard":
            continue
        violation_dict = violation.to_dict()
        items.append(
            {
                "signature": _failure_signature(contingency, violation_dict),
                "contingency_key": _contingency_key(contingency),
                "contingency": dict(contingency),
                "failure_type": "hard_violation",
                "severity": "hard",
                "violation": violation_dict,
                "margin_percent": violation.margin_percent,
            }
        )
    items.sort(key=_failure_sort_key, reverse=True)
    return items


def _non_convergence_failure(contingency: Mapping[str, Any], error: str) -> Dict[str, Any]:
    contingency_key = _contingency_key(contingency)
    return {
        "signature": f"{contingency_key}|non_convergence",
        "contingency_key": contingency_key,
        "contingency": dict(contingency),
        "failure_type": "non_convergence",
        "severity": "non_converged",
        "error": error,
        "margin_percent": None,
    }


def _comparison_item(
    change_type: str,
    post_failure: Mapping[str, Any],
    baseline_failure: Optional[Mapping[str, Any]],
    margin_percent_delta: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "change_type": change_type,
        "signature": post_failure["signature"],
        "contingency_key": post_failure["contingency_key"],
        "failure_type": post_failure["failure_type"],
        "severity": post_failure["severity"],
        "post_failure": dict(post_failure),
        "baseline_failure": dict(baseline_failure) if baseline_failure is not None else None,
        "margin_percent_delta": margin_percent_delta,
    }


def _margin_delta_percent(
    baseline_failure: Mapping[str, Any],
    post_failure: Mapping[str, Any],
) -> Optional[float]:
    baseline_margin = baseline_failure.get("margin_percent")
    post_margin = post_failure.get("margin_percent")
    if baseline_margin is None or post_margin is None:
        return None
    return float(post_margin) - float(baseline_margin)


def _failure_signature(
    contingency: Mapping[str, Any],
    violation: Mapping[str, Any],
) -> str:
    return (
        f"{_contingency_key(contingency)}|"
        f"{violation.get('element_type')}:{violation.get('element_index')}:"
        f"{violation.get('violation_type')}:{violation.get('limit_relation')}:"
        f"{round(float(violation.get('limit_value', 0.0)), 9)}"
    )


def _contingency_key(contingency: Mapping[str, Any]) -> str:
    return f"{contingency.get('element_type')}:{contingency.get('element_index')}"


def _max_failure_margin_percent(failure_items: List[Mapping[str, Any]]) -> Optional[float]:
    margins = [item.get("margin_percent") for item in failure_items if item.get("margin_percent") is not None]
    if not margins:
        return None
    return max(float(margin) for margin in margins)


def _failure_sort_key(item: Mapping[str, Any]) -> Tuple[int, float]:
    failure_type_rank = {"non_convergence": 3, "hard_violation": 2}.get(
        str(item.get("failure_type")),
        0,
    )
    margin = item.get("margin_percent")
    try:
        margin_value = float(margin) if margin is not None else 0.0
    except Exception:
        margin_value = 0.0
    return (failure_type_rank, margin_value)


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

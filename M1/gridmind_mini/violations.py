"""Solver-agnostic voltage and thermal violation inspection.

Step 2 mirrors Grid-Mind's violation-inspector layer: numerical values come
from a solver adapter, and this module deterministically classifies them
against configurable screening limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union


Severity = str
Status = str
FLOAT_EPSILON = 1e-9


@dataclass(frozen=True)
class LimitProfile:
    """Voltage and thermal screening limits for one study stage."""

    name: str = "normal"
    min_voltage_pu: float = 0.95
    max_voltage_pu: float = 1.05
    max_loading_percent: float = 100.0
    voltage_borderline_pu: float = 0.01
    loading_borderline_percent: float = 5.0
    angle_diff_limit_degree: Optional[float] = None

    @classmethod
    def normal(cls) -> "LimitProfile":
        """Normal steady-state screening limits from the Grid-Mind paper."""

        return cls()

    @classmethod
    def emergency(cls) -> "LimitProfile":
        """Emergency N-1 screening limits from the Grid-Mind paper."""

        return cls(
            name="emergency",
            min_voltage_pu=0.90,
            max_voltage_pu=1.10,
            max_loading_percent=110.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "min_voltage_pu": self.min_voltage_pu,
            "max_voltage_pu": self.max_voltage_pu,
            "max_loading_percent": self.max_loading_percent,
            "voltage_borderline_pu": self.voltage_borderline_pu,
            "loading_borderline_percent": self.loading_borderline_percent,
            "angle_diff_limit_degree": self.angle_diff_limit_degree,
        }


@dataclass(frozen=True)
class Violation:
    """One hard or borderline screening finding."""

    element_type: str
    element_index: int
    violation_type: str
    severity: Severity
    observed_value: float
    limit_value: float
    margin: float
    margin_percent: float
    unit: str
    limit_relation: str
    element_name: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_type": self.element_type,
            "element_index": self.element_index,
            "element_name": self.element_name,
            "violation_type": self.violation_type,
            "severity": self.severity,
            "observed_value": self.observed_value,
            "limit_value": self.limit_value,
            "margin": self.margin,
            "margin_percent": self.margin_percent,
            "unit": self.unit,
            "limit_relation": self.limit_relation,
        }


@dataclass(frozen=True)
class InspectionReport:
    """Structured output from the violation inspector."""

    limit_profile: LimitProfile
    violations: List[Violation]
    skipped_measurements: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def hard_count(self) -> int:
        return sum(1 for violation in self.violations if violation.severity == "hard")

    @property
    def borderline_count(self) -> int:
        return sum(1 for violation in self.violations if violation.severity == "borderline")

    @property
    def status(self) -> Status:
        if self.hard_count:
            return "fail"
        if self.borderline_count:
            return "borderline"
        return "pass"

    @property
    def passed(self) -> bool:
        return self.hard_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "limit_profile": self.limit_profile.to_dict(),
            "total_violations": len(self.violations),
            "hard_count": self.hard_count,
            "borderline_count": self.borderline_count,
            "skipped_measurements": self.skipped_measurements,
            "metadata": self.metadata,
            "violations": [violation.to_dict() for violation in self.violations],
        }


class ViolationInspector:
    """Classify solved bus voltages and branch loadings against limits."""

    def __init__(self, limit_profile: Optional[LimitProfile] = None) -> None:
        self.limit_profile = limit_profile or LimitProfile.normal()

    def inspect(
        self,
        bus_results: Iterable[Mapping[str, Any]],
        branch_results: Mapping[str, Iterable[Mapping[str, Any]]],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> InspectionReport:
        violations: List[Violation] = []
        skipped_measurements = 0
        buses = list(bus_results)
        lines = list(branch_results.get("lines", []))
        transformers = list(branch_results.get("transformers", []))

        for row in buses:
            voltage = _as_float(row.get("vm_pu"))
            if voltage is None:
                skipped_measurements += 1
                continue
            violations.extend(self._inspect_voltage(row, voltage))

        angle_by_bus = _bus_angle_lookup(buses)
        for line in lines:
            if _is_out_of_service(line):
                continue
            loading = _as_float(line.get("loading_percent"))
            if loading is None:
                skipped_measurements += 1
            else:
                finding = self._inspect_loading("line", line, loading)
                if finding is not None:
                    violations.append(finding)

            angle_finding = self._inspect_angle_difference(
                element_type="line",
                row=line,
                from_bus_key="from_bus",
                to_bus_key="to_bus",
                angle_by_bus=angle_by_bus,
            )
            if angle_finding == "skipped":
                skipped_measurements += 1
            elif angle_finding is not None:
                violations.append(angle_finding)

        for transformer in transformers:
            if _is_out_of_service(transformer):
                continue
            loading = _as_float(transformer.get("loading_percent"))
            if loading is None:
                skipped_measurements += 1
            else:
                finding = self._inspect_loading("transformer", transformer, loading)
                if finding is not None:
                    violations.append(finding)

            angle_finding = self._inspect_angle_difference(
                element_type="transformer",
                row=transformer,
                from_bus_key="hv_bus",
                to_bus_key="lv_bus",
                angle_by_bus=angle_by_bus,
            )
            if angle_finding == "skipped":
                skipped_measurements += 1
            elif angle_finding is not None:
                violations.append(angle_finding)

        return InspectionReport(
            limit_profile=self.limit_profile,
            violations=violations,
            skipped_measurements=skipped_measurements,
            metadata=dict(metadata or {}),
        )

    def inspect_solver(self, solver: Any) -> InspectionReport:
        """Inspect an already-solved GridSolver-like object."""

        metadata = {}
        if hasattr(solver, "case_info"):
            metadata["case"] = solver.case_info().to_dict()
        return self.inspect(solver.bus_results(), solver.branch_results(), metadata)

    def _inspect_voltage(self, row: Mapping[str, Any], voltage: float) -> List[Violation]:
        limits = self.limit_profile
        findings: List[Violation] = []

        low_margin = limits.min_voltage_pu - voltage
        if low_margin >= -limits.voltage_borderline_pu - FLOAT_EPSILON:
            severity = (
                "hard"
                if low_margin > limits.voltage_borderline_pu + FLOAT_EPSILON
                else "borderline"
            )
            findings.append(
                _make_violation(
                    element_type="bus",
                    element_index=_element_index(row, "bus_index"),
                    element_name=row.get("name"),
                    violation_type="low_voltage",
                    severity=severity,
                    observed_value=voltage,
                    limit_value=limits.min_voltage_pu,
                    margin=low_margin,
                    unit="p.u.",
                    limit_relation=">=",
                )
            )

        high_margin = voltage - limits.max_voltage_pu
        if high_margin >= -limits.voltage_borderline_pu - FLOAT_EPSILON:
            severity = (
                "hard"
                if high_margin > limits.voltage_borderline_pu + FLOAT_EPSILON
                else "borderline"
            )
            findings.append(
                _make_violation(
                    element_type="bus",
                    element_index=_element_index(row, "bus_index"),
                    element_name=row.get("name"),
                    violation_type="high_voltage",
                    severity=severity,
                    observed_value=voltage,
                    limit_value=limits.max_voltage_pu,
                    margin=high_margin,
                    unit="p.u.",
                    limit_relation="<=",
                )
            )

        return findings

    def _inspect_loading(
        self,
        element_type: str,
        row: Mapping[str, Any],
        loading_percent: float,
    ) -> Optional[Violation]:
        limits = self.limit_profile
        margin = loading_percent - limits.max_loading_percent
        if margin < -limits.loading_borderline_percent - FLOAT_EPSILON:
            return None

        severity = (
            "hard"
            if margin > limits.loading_borderline_percent + FLOAT_EPSILON
            else "borderline"
        )
        index_key = "line_index" if element_type == "line" else "trafo_index"
        return _make_violation(
            element_type=element_type,
            element_index=_element_index(row, index_key),
            element_name=row.get("name"),
            violation_type="thermal_loading",
            severity=severity,
            observed_value=loading_percent,
            limit_value=limits.max_loading_percent,
            margin=margin,
            unit="%",
            limit_relation="<=",
        )

    def _inspect_angle_difference(
        self,
        *,
        element_type: str,
        row: Mapping[str, Any],
        from_bus_key: str,
        to_bus_key: str,
        angle_by_bus: Mapping[int, float],
    ) -> Union[Violation, str, None]:
        limit = self.limit_profile.angle_diff_limit_degree
        if limit is None:
            return None

        from_bus = _as_int(row.get(from_bus_key))
        to_bus = _as_int(row.get(to_bus_key))
        if from_bus is None or to_bus is None:
            return "skipped"
        if from_bus not in angle_by_bus or to_bus not in angle_by_bus:
            return "skipped"

        angle_difference = _angle_difference_degree(angle_by_bus[from_bus], angle_by_bus[to_bus])
        margin = angle_difference - limit
        if margin <= FLOAT_EPSILON:
            return None

        index_key = "line_index" if element_type == "line" else "trafo_index"
        return _make_violation(
            element_type=element_type,
            element_index=_element_index(row, index_key),
            element_name=row.get("name"),
            violation_type="angle_difference",
            severity="hard",
            observed_value=angle_difference,
            limit_value=limit,
            margin=margin,
            unit="degree",
            limit_relation="<=",
        )


def profile_from_name(name: str) -> LimitProfile:
    """Build a known limit profile by name."""

    key = name.strip().lower().replace("-", "_")
    if key == "normal":
        return LimitProfile.normal()
    if key == "emergency":
        return LimitProfile.emergency()
    raise ValueError("Unknown limit profile. Expected 'normal' or 'emergency'.")


def _make_violation(
    *,
    element_type: str,
    element_index: int,
    element_name: Optional[Any],
    violation_type: str,
    severity: Severity,
    observed_value: float,
    limit_value: float,
    margin: float,
    unit: str,
    limit_relation: str,
) -> Violation:
    margin_percent = (margin / limit_value * 100.0) if limit_value else 0.0
    return Violation(
        element_type=element_type,
        element_index=element_index,
        element_name=element_name,
        violation_type=violation_type,
        severity=severity,
        observed_value=observed_value,
        limit_value=limit_value,
        margin=margin,
        margin_percent=margin_percent,
        unit=unit,
        limit_relation=limit_relation,
    )


def _element_index(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    try:
        return int(value)
    except Exception:
        return -1


def _as_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if value != value:
            return None
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if value != value:
            return None
        return float(value)
    except Exception:
        return None


def _is_out_of_service(row: Mapping[str, Any]) -> bool:
    in_service = row.get("in_service")
    if in_service is None:
        return False
    if isinstance(in_service, str):
        return in_service.strip().lower() in {"false", "0", "no", "off"}
    return not bool(in_service)


def _bus_angle_lookup(bus_results: Iterable[Mapping[str, Any]]) -> Dict[int, float]:
    angles: Dict[int, float] = {}
    for row in bus_results:
        bus_index = _as_int(row.get("bus_index"))
        angle = _as_float(row.get("va_degree"))
        if bus_index is not None and angle is not None:
            angles[bus_index] = angle
    return angles


def _angle_difference_degree(angle_a: float, angle_b: float) -> float:
    difference = abs(angle_a - angle_b) % 360.0
    if difference > 180.0:
        return 360.0 - difference
    return difference

"""Benchmark runner for generated live remote PSS/E M1+M2 cases.

The cases come from ``real_m1m2_interconnection_generator``. They are
evidence-only: positive labels are allowlisted remote PSS/E baseline jobs, and
negative labels are unsupported-action checks. The runner grades the model's
actual tool calls and final answer; by default it does not run separate oracle
PSS/E jobs, so it does not double the Windows-worker load.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .real_m1m2_interconnection_generator import (
    RealM1M2InterconnectionTestCase,
    real_m1m2_interconnection_testcases_from_payload,
)


@dataclass(frozen=True)
class RealM1M2BenchmarkCheckResult:
    """One benchmark check."""

    name: str
    passed: bool
    actual: Any = None
    expected: Any = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "actual": self.actual,
            "expected": self.expected,
            "message": self.message,
        }


@dataclass(frozen=True)
class RealM1M2BenchmarkResult:
    """Complete benchmark result for one generated testcase."""

    scenario: RealM1M2InterconnectionTestCase
    agent_result: Any
    check_results: Sequence[RealM1M2BenchmarkCheckResult]
    duration_s: float

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.check_results)

    def to_dict(
        self,
        *,
        include_raw_result: bool = True,
        include_messages: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "scenario": self.scenario.to_dict(),
            "passed": self.passed,
            "duration_s": self.duration_s,
            "check_results": [check.to_dict() for check in self.check_results],
        }
        if include_raw_result:
            payload["agent_result"] = self.agent_result.to_dict(
                include_messages=include_messages
            )
        else:
            payload["agent_summary"] = _agent_summary(self.agent_result)
        return payload


@dataclass(frozen=True)
class RealM1M2BenchmarkSuiteResult:
    """Aggregate benchmark result."""

    results: Sequence[RealM1M2BenchmarkResult]
    duration_s: float

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def to_dict(
        self,
        *,
        include_raw_results: bool = True,
        include_messages: bool = False,
    ) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "duration_s": self.duration_s,
            "by_label": _counts_by(self.results, "oracle_label"),
            "by_difficulty": _counts_by(self.results, "difficulty"),
            "results": [
                result.to_dict(
                    include_raw_result=include_raw_results,
                    include_messages=include_messages,
                )
                for result in self.results
            ],
        }


class RealM1M2InterconnectionBenchmarkRunner:
    """Run generated live remote PSS/E M1+M2 benchmark cases."""

    def __init__(self, agent: Any) -> None:
        if not hasattr(agent, "run_turn"):
            raise ValueError("agent must expose run_turn(message, context=...)")
        self.agent = agent

    def run_scenario(
        self,
        scenario: RealM1M2InterconnectionTestCase,
    ) -> RealM1M2BenchmarkResult:
        start = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            agent_result = self.agent.run_turn(
                scenario.user_message,
                context=_agent_visible_context(scenario),
            )
        checks = evaluate_real_m1m2_interconnection_result(
            scenario,
            agent_result=agent_result,
        )
        return RealM1M2BenchmarkResult(
            scenario=scenario,
            agent_result=agent_result,
            check_results=checks,
            duration_s=time.perf_counter() - start,
        )

    def run_suite(
        self,
        scenarios: Sequence[RealM1M2InterconnectionTestCase],
    ) -> RealM1M2BenchmarkSuiteResult:
        start = time.perf_counter()
        results: List[RealM1M2BenchmarkResult] = []
        total = len(scenarios)
        for index, scenario in enumerate(scenarios, start=1):
            case_start = time.perf_counter()
            print(
                _progress_json(
                    event="scenario_start",
                    index=index,
                    total=total,
                    scenario_id=scenario.scenario_id,
                    difficulty=scenario.difficulty,
                    oracle_label=scenario.oracle_label,
                    expected_tool=scenario.expected_tool,
                ),
                file=sys.stderr,
                flush=True,
            )
            result = self.run_scenario(scenario)
            results.append(result)
            print(
                _progress_json(
                    event="scenario_done",
                    index=index,
                    total=total,
                    scenario_id=scenario.scenario_id,
                    passed=result.passed,
                    duration_s=time.perf_counter() - case_start,
                    checks_failed=[
                        check.name for check in result.check_results if not check.passed
                    ],
                ),
                file=sys.stderr,
                flush=True,
            )
        return RealM1M2BenchmarkSuiteResult(
            results=results,
            duration_s=time.perf_counter() - start,
        )


def load_real_m1m2_interconnection_testcases(
    path: str | Path,
) -> List[RealM1M2InterconnectionTestCase]:
    """Load generated real M1+M2 testcases from JSON or JSONL."""

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(str(source))
    if source.suffix.lower() == ".jsonl":
        payload = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
    return real_m1m2_interconnection_testcases_from_payload(payload)


def filter_real_m1m2_interconnection_testcases(
    scenarios: Sequence[RealM1M2InterconnectionTestCase],
    *,
    scenario_ids: Sequence[str] = (),
    tags: Sequence[str] = (),
    difficulties: Sequence[str] = (),
    labels: Sequence[str] = (),
    limit: Optional[int] = None,
) -> List[RealM1M2InterconnectionTestCase]:
    """Filter generated scenarios by id, tag, difficulty, label, and limit."""

    wanted_ids = {item for item in scenario_ids if item}
    wanted_tags = {item for item in tags if item}
    wanted_difficulties = {item.lower() for item in difficulties if item}
    wanted_labels = {item for item in labels if item}
    selected = []
    for scenario in scenarios:
        if wanted_ids and scenario.scenario_id not in wanted_ids:
            continue
        if wanted_tags and not wanted_tags.intersection(set(scenario.tags)):
            continue
        if wanted_difficulties and scenario.difficulty.lower() not in wanted_difficulties:
            continue
        if wanted_labels and scenario.oracle_label not in wanted_labels:
            continue
        selected.append(scenario)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive when provided")
        selected = selected[:limit]
    return selected


def evaluate_real_m1m2_interconnection_result(
    scenario: RealM1M2InterconnectionTestCase,
    *,
    agent_result: Any,
) -> List[RealM1M2BenchmarkCheckResult]:
    """Evaluate one agent result against a generated real M1+M2 testcase."""

    checks: List[RealM1M2BenchmarkCheckResult] = []
    status = getattr(agent_result, "status", None)
    checks.append(
        RealM1M2BenchmarkCheckResult(
            name="agent_status",
            passed=status == "completed",
            actual=status,
            expected="completed",
        )
    )
    records = list(getattr(agent_result, "tool_records", []) or [])
    called_names = [getattr(record, "name", None) for record in records]
    expected_records = [
        record for record in records if getattr(record, "name", None) == scenario.expected_tool
    ]
    checks.append(
        RealM1M2BenchmarkCheckResult(
            name="expected_tool",
            passed=bool(expected_records),
            actual=called_names,
            expected=scenario.expected_tool,
        )
    )
    for forbidden in scenario.forbidden_successful_tools:
        used = [
            record
            for record in records
            if getattr(record, "name", None) == forbidden and getattr(record, "ok", False)
        ]
        checks.append(
            RealM1M2BenchmarkCheckResult(
                name=f"forbidden_tool:{forbidden}",
                passed=not used,
                actual=bool(used),
                expected=False,
            )
        )
    if scenario.oracle_label != "m1_m2_pass":
        used_remote_proxy = [
            record
            for record in records
            if getattr(record, "name", None) == "run_remote_psse_m1m2"
            and getattr(record, "ok", False)
        ]
        checks.append(
            RealM1M2BenchmarkCheckResult(
                name="forbidden_remote_baseline_proxy",
                passed=not used_remote_proxy,
                actual=bool(used_remote_proxy),
                expected=False,
                message=(
                    "Negative remote-gym scenarios must not run an allowlisted "
                    "baseline as a proxy for the unsupported requested action."
                ),
            )
        )

    if expected_records:
        record = expected_records[-1]
        checks.append(
            RealM1M2BenchmarkCheckResult(
                name="expected_tool_ok",
                passed=bool(getattr(record, "ok", False)),
                actual=getattr(record, "ok", None),
                expected=True,
            )
        )
        checks.extend(_tool_path_checks(scenario, getattr(record, "result", {})))
    else:
        checks.extend(
            RealM1M2BenchmarkCheckResult(
                name=f"result:{item.path}",
                passed=False,
                expected=item.expected,
                message="expected tool was not called",
            )
            for item in scenario.expected_paths
        )

    text = str(getattr(agent_result, "output_text", "") or "").lower()
    for expected in scenario.output_contains:
        checks.append(
            RealM1M2BenchmarkCheckResult(
                name=f"output_contains:{expected}",
                passed=str(expected).lower() in text,
                actual=text[:500],
                expected=expected,
            )
        )
    for forbidden in scenario.forbidden_claims:
        claim = str(forbidden).lower()
        checks.append(
            RealM1M2BenchmarkCheckResult(
                name=f"forbidden_claim:{forbidden}",
                passed=claim not in text,
                actual=text[:500] if claim in text else None,
                expected=f"not containing {forbidden!r}",
            )
        )
    return checks


def _tool_path_checks(
    scenario: RealM1M2InterconnectionTestCase,
    result: Mapping[str, Any],
) -> List[RealM1M2BenchmarkCheckResult]:
    checks = []
    for expected in scenario.expected_paths:
        actual = _path_get(result, expected.path)
        if expected.tolerance is not None and _is_number(actual) and _is_number(expected.expected):
            passed = abs(float(actual) - float(expected.expected)) <= expected.tolerance
        else:
            passed = actual == expected.expected
        checks.append(
            RealM1M2BenchmarkCheckResult(
                name=f"result:{expected.path}",
                passed=passed,
                actual=actual,
                expected=expected.expected,
                message=None
                if passed or expected.tolerance is None
                else f"tolerance={expected.tolerance}",
            )
        )
    return checks


def _path_get(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            return None
    return current


def _agent_summary(agent_result: Any) -> Dict[str, Any]:
    records = list(getattr(agent_result, "tool_records", []) or [])
    return {
        "status": getattr(agent_result, "status", None),
        "output_text": getattr(agent_result, "output_text", None),
        "invoked_tools": list(getattr(agent_result, "invoked_tools", []) or []),
        "llm_rounds": getattr(agent_result, "llm_rounds", None),
        "tool_records": [
            {
                "name": getattr(record, "name", None),
                "ok": getattr(record, "ok", None),
                "arguments": getattr(record, "arguments", None),
                "source": getattr(record, "source", None),
                "error": getattr(record, "error", None),
                "result": _compact_tool_result(getattr(record, "result", {})),
            }
            for record in records
        ],
    }


def _compact_tool_result(result: Any) -> Dict[str, Any]:
    if not isinstance(result, Mapping):
        return {}
    return {
        "ok": result.get("ok"),
        "tool": result.get("tool"),
        "backend": result.get("backend"),
        "case_id": result.get("case_id"),
        "scenario_type": result.get("scenario_type"),
        "recommendation": result.get("recommendation"),
        "complete": result.get("complete"),
        "summary": result.get("summary"),
        "case_count": result.get("case_count"),
        "error_type": result.get("error_type"),
        "message": result.get("message"),
    }


def _agent_visible_context(
    scenario: RealM1M2InterconnectionTestCase,
) -> Optional[Dict[str, Any]]:
    """Return testcase context that is safe to expose to the model.

    Generated scenarios carry oracle metadata for grading. Do not leak labels,
    answer policy, or pass/fail flags into the agent prompt.
    """

    context = dict(scenario.context) if scenario.context else {}
    for key in (
        "answer_policy",
        "expected_tool",
        "label_source",
        "oracle_label",
        "trap",
        "validated_remote_job",
    ):
        context.pop(key, None)
    context["remote_psse_m1m2_gym"] = True
    context["remote_psse_m1m2_scope"] = "live_tcp_ip_windows_worker"
    return context or None


def _progress_json(**payload: Any) -> str:
    return json.dumps({"real_m1m2_progress": payload}, sort_keys=True)


def _counts_by(
    results: Sequence[RealM1M2BenchmarkResult],
    attribute: str,
) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for result in results:
        key = str(getattr(result.scenario, attribute))
        bucket = counts.setdefault(key, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if result.passed:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
    return counts


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

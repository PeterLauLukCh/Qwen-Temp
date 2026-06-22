"""Integrated M1+M2 benchmark orchestration.

This module keeps M1 and M2 physically separate while making evaluation a
single command. M1 remains pandapower steady-state/CIA screening; M2 remains
ANDES transient-stability screening.
"""

from __future__ import annotations

import time
from collections.abc import Mapping as ABCMapping
from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from .m1_benchmark import M1BenchmarkRunner, run_m1_oracles
from .m2_benchmark import M2BenchmarkRunner, run_m2_oracles


@dataclass(frozen=True)
class IntegratedSuiteSection:
    """One suite section inside an integrated benchmark run."""

    suite: str
    mode: str
    scenario_ids: Sequence[str]
    result: Any
    oracle_executed: bool = False

    @property
    def enabled(self) -> bool:
        return self.result is not None

    @property
    def total(self) -> int:
        if self.result is None:
            return 0
        if isinstance(self.result, ABCSequence) and not isinstance(self.result, (str, bytes, ABCMapping)):
            return len(self.result)
        value = getattr(self.result, "total", None)
        return int(value) if isinstance(value, int) else 0

    @property
    def passed(self) -> int:
        if self.result is None:
            return 0
        if isinstance(self.result, ABCSequence) and not isinstance(self.result, (str, bytes, ABCMapping)):
            return sum(
                1 for item in self.result if isinstance(item, ABCMapping) and bool(item.get("ok", False))
            )
        value = getattr(self.result, "passed", None)
        return int(value) if isinstance(value, int) else 0

    @property
    def failed(self) -> int:
        return max(0, self.total - self.passed)

    @property
    def ok(self) -> bool:
        if self.result is None:
            return True
        if isinstance(self.result, ABCSequence) and not isinstance(self.result, (str, bytes, ABCMapping)):
            return all(
                isinstance(item, ABCMapping) and bool(item.get("ok", False))
                for item in self.result
            )
        value = getattr(self.result, "ok", None)
        return bool(value)

    def to_dict(
        self,
        *,
        include_raw_results: bool = True,
        include_messages: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "suite": self.suite,
            "mode": self.mode,
            "enabled": self.enabled,
            "ok": self.ok,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "scenario_ids": list(self.scenario_ids),
            "oracle_executed": self.oracle_executed,
        }
        if self.result is None:
            payload["results"] = None
            return payload
        if isinstance(self.result, ABCSequence) and not isinstance(self.result, (str, bytes, ABCMapping)):
            payload["results"] = (
                list(self.result)
                if include_raw_results
                else [_oracle_summary(item) for item in self.result]
            )
            return payload
        to_dict = getattr(self.result, "to_dict", None)
        if callable(to_dict):
            payload["suite_result"] = to_dict(
                include_raw_results=include_raw_results,
                include_messages=include_messages,
            )
            return payload
        payload["result"] = self.result
        return payload


@dataclass(frozen=True)
class IntegratedBenchmarkResult:
    """Aggregate result for a combined M1+M2 benchmark run."""

    mode: str
    m1: IntegratedSuiteSection
    m2: IntegratedSuiteSection
    duration_s: float
    execution_mode: str = "sequential_single_process"

    @property
    def ok(self) -> bool:
        return self.m1.ok and self.m2.ok

    @property
    def total(self) -> int:
        return self.m1.total + self.m2.total

    @property
    def passed(self) -> int:
        return self.m1.passed + self.m2.passed

    @property
    def failed(self) -> int:
        return self.m1.failed + self.m2.failed

    def to_dict(
        self,
        *,
        include_raw_results: bool = True,
        include_messages: bool = False,
    ) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "execution_mode": self.execution_mode,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "duration_s": self.duration_s,
            "m1": self.m1.to_dict(
                include_raw_results=include_raw_results,
                include_messages=include_messages,
            ),
            "m2": self.m2.to_dict(
                include_raw_results=include_raw_results,
                include_messages=include_messages,
            ),
        }


def run_integrated_oracles(
    *,
    m1_scenarios: Sequence[Any] = (),
    m2_scenarios: Sequence[Any] = (),
    oracle_registry: Any,
    execute_m2_oracle: bool = False,
) -> IntegratedBenchmarkResult:
    """Run M1 and M2 oracle checks in one job."""

    if not hasattr(oracle_registry, "call_tool"):
        raise ValueError("oracle_registry must expose call_tool(name, arguments)")
    start = time.perf_counter()
    m1_result = run_m1_oracles(m1_scenarios, oracle_registry) if m1_scenarios else None
    m2_result = (
        run_m2_oracles(
            m2_scenarios,
            oracle_registry,
            execute_tools=execute_m2_oracle,
        )
        if m2_scenarios
        else None
    )
    return IntegratedBenchmarkResult(
        mode="oracle_only",
        m1=IntegratedSuiteSection(
            suite="m1",
            mode="oracle_only",
            scenario_ids=_scenario_ids(m1_scenarios),
            result=m1_result,
            oracle_executed=True if m1_scenarios else False,
        ),
        m2=IntegratedSuiteSection(
            suite="m2",
            mode="oracle_only",
            scenario_ids=_scenario_ids(m2_scenarios),
            result=m2_result,
            oracle_executed=execute_m2_oracle if m2_scenarios else False,
        ),
        duration_s=time.perf_counter() - start,
    )


def run_integrated_live_agent(
    *,
    agent: Any,
    m1_scenarios: Sequence[Any] = (),
    m2_scenarios: Sequence[Any] = (),
    oracle_registry: Any,
    execute_m2_oracle: bool = False,
) -> IntegratedBenchmarkResult:
    """Run M1 and M2 live-agent benchmarks in one job."""

    if not hasattr(agent, "run_turn"):
        raise ValueError("agent must expose run_turn(message, context=...)")
    if not hasattr(oracle_registry, "call_tool"):
        raise ValueError("oracle_registry must expose call_tool(name, arguments)")
    start = time.perf_counter()
    m1_result = (
        M1BenchmarkRunner(agent, oracle_registry).run_suite(m1_scenarios)
        if m1_scenarios
        else None
    )
    m2_result = (
        M2BenchmarkRunner(
            agent,
            oracle_registry,
            execute_oracle=execute_m2_oracle,
        ).run_suite(m2_scenarios)
        if m2_scenarios
        else None
    )
    return IntegratedBenchmarkResult(
        mode="live_agent",
        m1=IntegratedSuiteSection(
            suite="m1",
            mode="live_agent",
            scenario_ids=_scenario_ids(m1_scenarios),
            result=m1_result,
            oracle_executed=True if m1_scenarios else False,
        ),
        m2=IntegratedSuiteSection(
            suite="m2",
            mode="live_agent",
            scenario_ids=_scenario_ids(m2_scenarios),
            result=m2_result,
            oracle_executed=execute_m2_oracle if m2_scenarios else False,
        ),
        duration_s=time.perf_counter() - start,
    )


def _scenario_ids(scenarios: Sequence[Any]) -> Sequence[str]:
    return [str(getattr(scenario, "scenario_id", "")) for scenario in scenarios]


def _oracle_summary(item: Any) -> Dict[str, Any]:
    if not isinstance(item, ABCMapping):
        return {"ok": False, "error_type": "invalid_oracle_item"}
    scenario = item.get("scenario", {})
    result = item.get("oracle_result")
    summary: Dict[str, Any] = {
        "scenario_id": scenario.get("scenario_id") if isinstance(scenario, ABCMapping) else None,
        "ok": bool(item.get("ok", False)),
    }
    if isinstance(result, ABCMapping):
        summary["tool"] = result.get("tool")
        summary["error_type"] = result.get("error_type")
        for key in ("case_path", "recommendation", "status", "max_approved_mw"):
            if key in result:
                summary[key] = result[key]
        stability = result.get("stability")
        if isinstance(stability, ABCMapping):
            summary["stability_status"] = stability.get("status")
        metrics = result.get("metrics")
        if isinstance(metrics, ABCMapping):
            summary["reason_codes"] = metrics.get("reason_codes")
    else:
        summary["tool"] = None
        summary["note"] = item.get("note")
    for source_key, output_key in (
        ("oracle_argument_checks", "oracle_argument_checks_ok"),
        ("oracle_path_checks", "oracle_path_checks_ok"),
    ):
        checks = item.get(source_key, [])
        if isinstance(checks, list):
            summary[output_key] = all(
                bool(check.get("passed", False))
                for check in checks
                if isinstance(check, ABCMapping)
            )
    if "oracle_executed" in item:
        summary["oracle_executed"] = bool(item.get("oracle_executed"))
    return summary

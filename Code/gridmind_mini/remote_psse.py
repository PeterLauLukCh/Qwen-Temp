"""Live remote PSS/E M1+M2 integration.

This module is the GPU-side boundary for the Windows PSS/E worker. It never
loads SAV/DYR/DLL files locally; it sends allowlisted case/scenario identifiers
to the remote queued worker and adapts the returned compact JSON into the
Grid-Mind result shape used by observations, reports, and tools.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REMOTE_PSSE_BASE_URL_ENV = "PSSE_REMOTE_BASE_URL"
REMOTE_PSSE_TOKEN_ENV = "PSSE_REMOTE_TOKEN"
REMOTE_PSSE_TIMEOUT_ENV = "PSSE_REMOTE_TIMEOUT_S"
REMOTE_PSSE_POLL_INTERVAL_ENV = "PSSE_REMOTE_POLL_INTERVAL_S"
REMOTE_PSSE_MAX_WAIT_ENV = "PSSE_REMOTE_MAX_WAIT_S"

DEFAULT_REMOTE_TIMEOUT_S = 10.0
DEFAULT_REMOTE_POLL_INTERVAL_S = 2.0
DEFAULT_REMOTE_MAX_WAIT_S = 900.0
REMOTE_M1M2_SCHEMA_VERSION = "remote_psse_m1m2_v1"
REMOTE_M1M2_TOOL = "run_remote_psse_m1m2"


class RemotePsseError(RuntimeError):
    """Raised for remote-worker configuration or communication failures."""

    def __init__(self, message: str, *, error_type: str = "remote_psse_error") -> None:
        super().__init__(message)
        self.error_type = error_type


@dataclass(frozen=True)
class RemotePsseCaseDefinition:
    """GPU-side mirror of the remote worker's public allowlist."""

    case_id: str
    display_name: str
    allowed_scenarios: Tuple[str, ...]
    description: str
    dynamic_scenarios: Tuple[str, ...] = ("no_disturbance_5s",)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "display_name": self.display_name,
            "allowed_scenarios": list(self.allowed_scenarios),
            "dynamic_scenarios": list(self.dynamic_scenarios),
            "description": self.description,
        }


REMOTE_PSSE_M1M2_CASES: Dict[str, RemotePsseCaseDefinition] = {
    "test_cases_v36": RemotePsseCaseDefinition(
        case_id="test_cases_v36",
        display_name="Small PSS/E PPC test case",
        allowed_scenarios=("static", "no_disturbance_5s", "pq_target_step"),
        description=(
            "Small PPC benchmark. Supports static load flow, 5-second "
            "no-disturbance RMS dynamics, and the provided P/Q target step."
        ),
    ),
    "pif6_2026_05_17": RemotePsseCaseDefinition(
        case_id="pif6_2026_05_17",
        display_name="PIF6 POC2 786-bus case",
        allowed_scenarios=("static", "no_disturbance_5s"),
        description=(
            "Real PIF6 786-bus POC2 case. V1 supports static load flow and "
            "5-second no-disturbance RMS baseline only."
        ),
    ),
}


@dataclass(frozen=True)
class RemotePsseConfig:
    """Connection settings for the remote PSS/E worker."""

    base_url: str
    token: str = ""
    timeout_s: float = DEFAULT_REMOTE_TIMEOUT_S
    poll_interval_s: float = DEFAULT_REMOTE_POLL_INTERVAL_S
    max_wait_s: float = DEFAULT_REMOTE_MAX_WAIT_S

    @classmethod
    def from_env(
        cls,
        *,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout_s: Optional[float] = None,
        poll_interval_s: Optional[float] = None,
        max_wait_s: Optional[float] = None,
    ) -> "RemotePsseConfig":
        resolved_base_url = (base_url or os.environ.get(REMOTE_PSSE_BASE_URL_ENV, "")).strip()
        if not resolved_base_url:
            raise RemotePsseError(
                f"Remote PSS/E base URL is not configured. Set ${REMOTE_PSSE_BASE_URL_ENV}.",
                error_type="remote_psse_not_configured",
            )
        return cls(
            base_url=resolved_base_url.rstrip("/"),
            token=os.environ.get(REMOTE_PSSE_TOKEN_ENV, "") if token is None else str(token),
            timeout_s=_positive_float(
                timeout_s,
                REMOTE_PSSE_TIMEOUT_ENV,
                DEFAULT_REMOTE_TIMEOUT_S,
            ),
            poll_interval_s=_positive_float(
                poll_interval_s,
                REMOTE_PSSE_POLL_INTERVAL_ENV,
                DEFAULT_REMOTE_POLL_INTERVAL_S,
            ),
            max_wait_s=_positive_float(
                max_wait_s,
                REMOTE_PSSE_MAX_WAIT_ENV,
                DEFAULT_REMOTE_MAX_WAIT_S,
            ),
        )


def list_remote_psse_m1m2_cases(
    *,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    timeout_s: Optional[float] = None,
    check_health: bool = False,
) -> Dict[str, Any]:
    """List live remote PSS/E M1+M2 cases known to the GPU-side client."""

    remote_configured = bool((base_url or os.environ.get(REMOTE_PSSE_BASE_URL_ENV, "")).strip())
    health: Optional[Dict[str, Any]] = None
    if check_health and remote_configured:
        try:
            config = RemotePsseConfig.from_env(
                base_url=base_url,
                token=token,
                timeout_s=timeout_s,
            )
            health = request_json("GET", config, "/health")
        except RemotePsseError as exc:
            health = {
                "ok": False,
                "error_type": exc.error_type,
                "message": str(exc),
            }
    return {
        "ok": True,
        "tool": "list_remote_psse_m1m2_cases",
        "backend": "psse_remote_worker",
        "schema_version": REMOTE_M1M2_SCHEMA_VERSION,
        "remote_configured": remote_configured,
        "base_url_env": REMOTE_PSSE_BASE_URL_ENV,
        "token_env": REMOTE_PSSE_TOKEN_ENV,
        "case_count": len(REMOTE_PSSE_M1M2_CASES),
        "cases": [case.to_dict() for case in REMOTE_PSSE_M1M2_CASES.values()],
        "health": health,
        "limitations": [
            "gpu_side_does_not_load_sav_dyr_or_dll",
            "remote_jobs_reload_original_case_each_call",
            "v1_no_persistent_dynamic_episode_state",
            "v1_allowlisted_cases_and_scenarios_only",
            "psse_rms_dynamic_not_emt_waveform",
        ],
    }


def run_remote_psse_m1m2(
    *,
    case_id: str,
    scenario_type: str,
    request_id: Optional[str] = None,
    include_artifacts: bool = False,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    timeout_s: Optional[float] = None,
    poll_interval_s: Optional[float] = None,
    max_wait_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Submit one allowlisted M1/M2 PSS/E job and return compact JSON."""

    try:
        normalized_case, normalized_scenario = validate_remote_m1m2_request(
            case_id=case_id,
            scenario_type=scenario_type,
        )
        config = RemotePsseConfig.from_env(
            base_url=base_url,
            token=token,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            max_wait_s=max_wait_s,
        )
        body: Dict[str, Any] = {
            "case_id": normalized_case,
            "scenario_type": normalized_scenario,
        }
        if request_id:
            body["request_id"] = str(request_id)
        submission = request_json("POST", config, "/jobs", body=body)
        if not submission.get("ok") or not submission.get("job_id"):
            return _remote_error_result(
                case_id=normalized_case,
                scenario_type=normalized_scenario,
                error_type=str(submission.get("error_type") or "remote_submission_failed"),
                message=str(submission.get("message") or "Remote worker did not accept the job."),
                extra={"submission": submission},
            )
        job_id = str(submission["job_id"])
        status_payload = wait_for_remote_job(config, job_id)
        result = request_json("GET", config, f"/jobs/{job_id}/result")
        artifacts = None
        if include_artifacts:
            try:
                artifacts = request_json("GET", config, f"/jobs/{job_id}/artifacts")
            except RemotePsseError as exc:
                artifacts = {
                    "ok": False,
                    "error_type": exc.error_type,
                    "message": str(exc),
                }
        return adapt_remote_worker_result(
            result,
            submission=submission,
            final_status=status_payload,
            artifacts=artifacts,
        )
    except RemotePsseError as exc:
        return _remote_error_result(
            case_id=str(case_id or ""),
            scenario_type=str(scenario_type or ""),
            error_type=exc.error_type,
            message=str(exc),
        )


def validate_remote_m1m2_request(*, case_id: str, scenario_type: str) -> Tuple[str, str]:
    """Validate and normalize the GPU-side public case/scenario allowlist."""

    normalized_case = str(case_id or "").strip()
    if normalized_case not in REMOTE_PSSE_M1M2_CASES:
        allowed = ", ".join(sorted(REMOTE_PSSE_M1M2_CASES))
        raise RemotePsseError(
            f"Unsupported remote PSS/E case_id '{case_id}'. Allowed: {allowed}.",
            error_type="unsupported_remote_case_id",
        )
    normalized_scenario = normalize_remote_scenario_type(scenario_type)
    allowed_scenarios = REMOTE_PSSE_M1M2_CASES[normalized_case].allowed_scenarios
    if normalized_scenario not in allowed_scenarios:
        raise RemotePsseError(
            (
                f"Unsupported remote PSS/E scenario_type '{scenario_type}' for "
                f"{normalized_case}. Allowed: {', '.join(allowed_scenarios)}."
            ),
            error_type="unsupported_remote_scenario_type",
        )
    return normalized_case, normalized_scenario


def normalize_remote_scenario_type(value: Any) -> str:
    """Normalize common scenario aliases without expanding the validated scope."""

    aliases = {
        "none": "no_disturbance_5s",
        "no_disturbance": "no_disturbance_5s",
        "baseline": "no_disturbance_5s",
        "baseline_5s": "no_disturbance_5s",
        "pq_target": "pq_target_step",
        "p_q_target": "pq_target_step",
        "static_pf": "static",
        "load_flow": "static",
        "powerflow": "static",
    }
    normalized = aliases.get(str(value or "").strip().lower(), str(value or "").strip().lower())
    if not normalized:
        raise RemotePsseError("scenario_type is required.", error_type="invalid_remote_job_request")
    if normalized in {"bus_fault", "fault", "line_trip", "qref_step", "pref_step", "vref_step"}:
        raise RemotePsseError(
            f"Scenario '{normalized}' is intentionally blocked until validated for remote PSS/E.",
            error_type="remote_scenario_not_validated",
        )
    return normalized


def request_json(
    method: str,
    config: RemotePsseConfig,
    path: str,
    *,
    body: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Perform one JSON HTTP request against the worker."""

    url = config.base_url.rstrip("/") + path
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(dict(body)).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        payload = _http_error_payload(exc)
        raise RemotePsseError(
            str(payload.get("message") or payload.get("error") or exc.reason),
            error_type=str(payload.get("error_type") or "remote_http_error"),
        ) from exc
    except urllib.error.URLError as exc:
        raise RemotePsseError(
            str(exc.reason),
            error_type="remote_url_error",
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemotePsseError(
            f"Remote worker returned invalid JSON: {exc}",
            error_type="remote_invalid_json",
        ) from exc
    if not isinstance(payload, dict):
        raise RemotePsseError(
            "Remote worker response must be a JSON object.",
            error_type="remote_invalid_json",
        )
    return payload


def wait_for_remote_job(config: RemotePsseConfig, job_id: str) -> Dict[str, Any]:
    """Poll the queued worker until a job reaches completed or error."""

    started = time.monotonic()
    last_status: Dict[str, Any] = {}
    while True:
        status = request_json("GET", config, f"/jobs/{job_id}")
        last_status = status
        job = status.get("job")
        if isinstance(job, Mapping) and job.get("status") in {"completed", "error"}:
            return status
        if time.monotonic() - started > config.max_wait_s:
            raise RemotePsseError(
                f"Timed out waiting for remote PSS/E job {job_id}.",
                error_type="remote_job_wait_timeout",
            )
        time.sleep(max(config.poll_interval_s, 0.1))


def adapt_remote_worker_result(
    worker_result: Mapping[str, Any],
    *,
    submission: Optional[Mapping[str, Any]] = None,
    final_status: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Adapt worker JSON into the Grid-Mind M1+M2 result contract."""

    case_id = _optional_str(worker_result.get("case_id")) or "unknown"
    scenario_type = _optional_str(worker_result.get("scenario_type")) or "unknown"
    m1_result = _mapping_or_empty(worker_result.get("m1_powerflow"))
    m2_result = _mapping_or_empty(worker_result.get("m2_dynamic"))
    m1_stage = _m1_stage(m1_result)
    m2_stage = _m2_stage(
        m2_result,
        dynamic_required=scenario_type == "no_disturbance_5s",
    )
    emt_stage = _emt_stage()
    summary = _summary(worker_result, m1_stage, m2_stage)
    ok = bool(worker_result.get("ok"))
    payload: Dict[str, Any] = {
        "ok": ok,
        "tool": REMOTE_M1M2_TOOL,
        "schema_version": REMOTE_M1M2_SCHEMA_VERSION,
        "backend": "psse_remote_worker",
        "source_worker_tool": _optional_str(worker_result.get("tool")),
        "source_worker_schema_version": _optional_str(worker_result.get("schema_version")),
        "job_id": _optional_str(worker_result.get("job_id")),
        "case_id": case_id,
        "scenario_type": scenario_type,
        "display_name": _optional_str(worker_result.get("display_name")),
        "psse_version": _optional_str(worker_result.get("psse_version")),
        "stage_reports": [m1_stage, m2_stage, emt_stage],
        "m1_result": m1_result,
        "m2_result": m2_result,
        "emt_result": _mapping_or_empty(worker_result.get("emt_result"))
        or {
            "status": "not_run",
            "backend": "not_available_in_remote_v1",
            "reason_codes": ["psse_rms_dynamic_not_emt_waveform"],
        },
        "recommendation": _optional_str(worker_result.get("recommendation"))
        or ("approve" if ok else "reject"),
        "complete": bool(worker_result.get("complete")),
        "reason_codes": _string_list(worker_result.get("reason_codes")),
        "summary": summary,
        "limitations": _string_list(worker_result.get("limitations")),
        "execution": _mapping_or_empty(worker_result.get("execution")),
    }
    if not ok:
        payload["error_type"] = _optional_str(worker_result.get("error_type")) or "remote_psse_job_failed"
        payload["message"] = _optional_str(worker_result.get("message")) or "Remote PSS/E M1+M2 job failed."
    if submission is not None:
        payload["submission"] = dict(submission)
    if final_status is not None:
        payload["final_status"] = dict(final_status)
    if artifacts is not None:
        payload["artifacts"] = dict(artifacts)
    if "psse_rms_dynamic_not_emt_waveform" not in payload["limitations"]:
        payload["limitations"].append("psse_rms_dynamic_not_emt_waveform")
    return payload


def _m1_stage(m1: Mapping[str, Any]) -> Dict[str, Any]:
    status = "pass" if str(m1.get("status")) == "pass" and bool(m1.get("converged")) else "fail"
    overloaded = _optional_int(m1.get("overloaded_branch_count"))
    reason_codes = ["psse_powerflow_converged"] if status == "pass" else ["psse_powerflow_failed"]
    if overloaded is not None and overloaded > 0:
        status = "fail"
        reason_codes.append("psse_overloaded_branches")
    return {
        "stage": "m1_psse_powerflow",
        "status": status,
        "passed": status == "pass",
        "required_for_approval": True,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "metrics": {
            "converged": bool(m1.get("converged")),
            "bus_voltage_min_pu": _optional_number(m1.get("bus_voltage_min_pu")),
            "bus_voltage_max_pu": _optional_number(m1.get("bus_voltage_max_pu")),
            "poc_p_mw": _optional_number(m1.get("poc_p_mw")),
            "poc_q_mvar": _optional_number(m1.get("poc_q_mvar")),
            "bus_count": _optional_int(m1.get("bus_count")),
            "branch_count": _optional_int(m1.get("branch_count")),
            "machine_count": _optional_int(m1.get("machine_count")),
            "load_count": _optional_int(m1.get("load_count")),
            "overloaded_branch_count": overloaded,
        },
        "error_summary": _string_list(m1.get("error_summary")),
    }


def _m2_stage(m2: Mapping[str, Any], *, dynamic_required: bool) -> Dict[str, Any]:
    raw_status = str(m2.get("status", "unknown"))
    initialized = bool(m2.get("initialized"))
    simulation_converged = bool(m2.get("simulation_converged"))
    metrics = _mapping_or_empty(m2.get("metrics"))
    if raw_status == "pass" and initialized and simulation_converged:
        status = "pass"
        passed: Optional[bool] = True
        reason_codes = ["psse_rms_dynamic_converged"]
    elif raw_status in {"not_run", "skipped"} and not dynamic_required:
        status = "skipped"
        passed = None
        reason_codes = ["dynamic_not_run"]
    else:
        status = "fail"
        passed = False
        reason_codes = ["psse_rms_dynamic_failed"]
    return {
        "stage": "m2_psse_rms_dynamic",
        "status": status,
        "passed": passed,
        "required_for_approval": dynamic_required,
        "event": _optional_str(m2.get("event")),
        "reason_codes": reason_codes,
        "metrics": {
            "initialized": initialized,
            "simulation_converged": simulation_converged,
            "min_voltage_pu": _optional_number(metrics.get("min_voltage_pu")),
            "max_voltage_pu": _optional_number(metrics.get("max_voltage_pu")),
            "final_poc_p_mw": _optional_number(metrics.get("final_poc_p_mw")),
            "final_poc_q_mvar": _optional_number(metrics.get("final_poc_q_mvar")),
            "max_oscillation": _optional_number(metrics.get("max_oscillation")),
            "final_time_s": _optional_number(metrics.get("final_time_s")),
            "channel_row_count": _optional_int(metrics.get("channel_row_count")),
            "max_frequency_deviation_pu": _optional_number(metrics.get("max_frequency_deviation_pu")),
            "max_frequency_deviation_hz_at_60hz": _optional_number(
                metrics.get("max_frequency_deviation_hz_at_60hz")
            ),
        },
        "error_summary": _string_list(m2.get("error_summary")),
    }


def _emt_stage() -> Dict[str, Any]:
    return {
        "stage": "f4_emt",
        "status": "not_run",
        "passed": None,
        "required_for_approval": False,
        "reason_codes": ["psse_rms_dynamic_not_emt_waveform"],
    }


def _summary(
    worker_result: Mapping[str, Any],
    m1_stage: Mapping[str, Any],
    m2_stage: Mapping[str, Any],
) -> Dict[str, Any]:
    del worker_result
    m1_metrics = _mapping_or_empty(m1_stage.get("metrics"))
    m2_metrics = _mapping_or_empty(m2_stage.get("metrics"))
    return {
        "m1_status": _optional_str(m1_stage.get("status")),
        "m1_converged": m1_metrics.get("converged"),
        "m1_bus_voltage_min_pu": m1_metrics.get("bus_voltage_min_pu"),
        "m1_bus_voltage_max_pu": m1_metrics.get("bus_voltage_max_pu"),
        "m1_poc_p_mw": m1_metrics.get("poc_p_mw"),
        "m1_poc_q_mvar": m1_metrics.get("poc_q_mvar"),
        "m1_bus_count": m1_metrics.get("bus_count"),
        "m1_branch_count": m1_metrics.get("branch_count"),
        "m1_machine_count": m1_metrics.get("machine_count"),
        "m1_load_count": m1_metrics.get("load_count"),
        "m2_status": _optional_str(m2_stage.get("status")),
        "m2_event": _optional_str(m2_stage.get("event")),
        "m2_initialized": m2_metrics.get("initialized"),
        "m2_simulation_converged": m2_metrics.get("simulation_converged"),
        "m2_min_voltage_pu": m2_metrics.get("min_voltage_pu"),
        "m2_max_voltage_pu": m2_metrics.get("max_voltage_pu"),
        "m2_final_poc_p_mw": m2_metrics.get("final_poc_p_mw"),
        "m2_final_poc_q_mvar": m2_metrics.get("final_poc_q_mvar"),
        "m2_max_oscillation": m2_metrics.get("max_oscillation"),
        "m2_final_time_s": m2_metrics.get("final_time_s"),
        "m2_channel_row_count": m2_metrics.get("channel_row_count"),
        "m2_max_frequency_deviation_pu": m2_metrics.get("max_frequency_deviation_pu"),
        "m2_max_frequency_deviation_hz_at_60hz": m2_metrics.get(
            "max_frequency_deviation_hz_at_60hz"
        ),
        "emt_status": "not_run",
        "backend": "psse_remote_worker",
    }


def _remote_error_result(
    *,
    case_id: str,
    scenario_type: str,
    error_type: str,
    message: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": False,
        "tool": REMOTE_M1M2_TOOL,
        "schema_version": REMOTE_M1M2_SCHEMA_VERSION,
        "backend": "psse_remote_worker",
        "case_id": case_id,
        "scenario_type": scenario_type,
        "error_type": error_type,
        "message": message,
        "recommendation": "reject",
        "complete": False,
        "reason_codes": [error_type],
        "limitations": [
            "remote_psse_result_unavailable",
            "psse_rms_dynamic_not_emt_waveform",
        ],
    }
    payload.update(dict(extra or {}))
    return payload


def _http_error_payload(exc: urllib.error.HTTPError) -> Dict[str, Any]:
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except Exception:
        payload = {
            "ok": False,
            "error_type": "remote_http_error",
            "status": exc.code,
            "message": exc.reason,
        }
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "error_type": "remote_http_error",
            "status": exc.code,
            "message": exc.reason,
        }
    return payload


def _positive_float(value: Optional[float], env_name: str, default: float) -> float:
    raw: Any = value
    if raw is None:
        raw = os.environ.get(env_name)
    if raw is None or raw == "":
        return default
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as exc:
        raise RemotePsseError(
            f"{env_name} must be a positive number.",
            error_type="invalid_remote_psse_config",
        ) from exc
    if parsed <= 0:
        raise RemotePsseError(
            f"{env_name} must be positive.",
            error_type="invalid_remote_psse_config",
        )
    return parsed


def _mapping_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> List[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return []


def _optional_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _optional_int(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _optional_str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value:
        return value
    return None

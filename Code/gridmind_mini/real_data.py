"""Frozen real-data PSS/E result integration.

This module reads compact JSON/CSV outputs produced by an external PSS/E
runner. It never opens SAV/DYR/DLL files and never attempts to execute PSS/E.
The goal is to expose real-data study results through the same tool/reporting
shape used by the rest of Mini Grid-Mind.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


REAL_DATA_DIR_ENV = "POWERGYM_REAL_DATA_DIR"
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "real-data-new" / "processed_file"
LEGACY_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "real-data-new" / "processed_files"
REAL_PSSE_CASE_IDS = (
    "test_cases_v36_no_disturbance",
    "test_cases_v36_pq_target",
    "pif6_baseline",
)
PSSE_RMS_NOT_EMT_LIMITATION = "psse_rms_dynamic_not_emt_waveform"


class RealDataError(RuntimeError):
    """Raised for invalid or unavailable frozen real-data artifacts."""

    def __init__(self, message: str, *, error_type: str = "real_data_error") -> None:
        super().__init__(message)
        self.error_type = error_type


@dataclass(frozen=True)
class RealPsseCaseDefinition:
    """Definition of one frozen PSS/E result case."""

    case_id: str
    display_name: str
    bundle_id: str
    result_file: str
    bus_voltage_csv: Optional[str] = None
    channel_csv: Optional[str] = None
    overload_csv: Optional[str] = None
    poc_candidates_csv: Optional[str] = None
    pq_iterations_csv: Optional[str] = None
    dynamic_required_for_completion: bool = True
    description: str = ""


REAL_PSSE_CASES: Dict[str, RealPsseCaseDefinition] = {
    "test_cases_v36_no_disturbance": RealPsseCaseDefinition(
        case_id="test_cases_v36_no_disturbance",
        display_name="Small PSS/E PPC no-disturbance baseline",
        bundle_id="test_cases_v36",
        result_file="test_cases_v36_no_disturbance_result.json",
        bus_voltage_csv="test_cases_v36_bus_voltages.csv",
        channel_csv="test_cases_v36_no_disturbance_channels.csv",
        overload_csv="test_cases_v36_overloaded_branches.csv",
        description="Small PPC benchmark with load flow and 5 s no-disturbance RMS dynamics.",
    ),
    "test_cases_v36_pq_target": RealPsseCaseDefinition(
        case_id="test_cases_v36_pq_target",
        display_name="Small PSS/E PPC P/Q target reproduction",
        bundle_id="test_cases_v36",
        result_file="test_cases_v36_pq_target_result.json",
        bus_voltage_csv="test_cases_v36_bus_voltages.csv",
        overload_csv="test_cases_v36_overloaded_branches.csv",
        pq_iterations_csv="test_cases_v36_pq_target_iterations.csv",
        dynamic_required_for_completion=False,
        description="Static P/Q target reproduction from the provided step_pmin_qmax script.",
    ),
    "pif6_baseline": RealPsseCaseDefinition(
        case_id="pif6_baseline",
        display_name="PIF6 786-bus PSS/E baseline",
        bundle_id="pif6_2026_05_17",
        result_file="pif6_baseline_result.json",
        bus_voltage_csv="pif6_bus_voltages.csv",
        channel_csv="pif6_no_disturbance_channels.csv",
        overload_csv="pif6_overloaded_branches.csv",
        poc_candidates_csv="pif6_poc_branch_candidates.csv",
        description="786-bus PIF6 renewable-plant case with 5 s no-disturbance RMS dynamics.",
    ),
}


def list_real_psse_cases(processed_dir: Optional[str] = None) -> Dict[str, Any]:
    """List frozen real-data PSS/E cases and artifact availability."""

    root = resolve_processed_dir(processed_dir)
    preflight = _load_preflight_if_available(root)
    cases = []
    for definition in REAL_PSSE_CASES.values():
        artifact_status = _artifact_status(root, definition)
        cases.append(
            {
                "case_id": definition.case_id,
                "display_name": definition.display_name,
                "bundle_id": definition.bundle_id,
                "description": definition.description,
                "available": artifact_status["required_available"],
                "result_file": definition.result_file,
                "artifacts": artifact_status,
                "dynamic_required_for_completion": definition.dynamic_required_for_completion,
            }
        )
    return {
        "ok": True,
        "tool": "list_real_psse_cases",
        "backend": "psse_frozen",
        "processed_dir": str(root),
        "processed_dir_exists": root.exists(),
        "preflight_available": bool(preflight),
        "psse_version": _preflight_psse_version(preflight),
        "case_count": len(cases),
        "cases": cases,
        "limitations": [
            "frozen_psse_outputs_only_no_local_psse_execution",
            "labels_not_yet_expert_validated",
        ],
    }


def run_real_psse_assessment(
    *,
    case_id: str,
    processed_dir: Optional[str] = None,
    include_channel_preview: bool = False,
    max_channel_samples: int = 20,
) -> Dict[str, Any]:
    """Return a Grid-Mind-style assessment from frozen PSS/E artifacts."""

    if case_id not in REAL_PSSE_CASES:
        return _error_result(
            case_id=case_id,
            processed_dir=resolve_processed_dir(processed_dir),
            error_type="unknown_real_psse_case",
            message=f"Unknown real PSS/E case_id '{case_id}'.",
        )
    if not isinstance(include_channel_preview, bool):
        return _error_result(
            case_id=case_id,
            processed_dir=resolve_processed_dir(processed_dir),
            error_type="invalid_argument",
            message="include_channel_preview must be a boolean.",
        )
    if not isinstance(max_channel_samples, int) or isinstance(max_channel_samples, bool):
        return _error_result(
            case_id=case_id,
            processed_dir=resolve_processed_dir(processed_dir),
            error_type="invalid_argument",
            message="max_channel_samples must be an integer.",
        )
    if max_channel_samples < 0:
        return _error_result(
            case_id=case_id,
            processed_dir=resolve_processed_dir(processed_dir),
            error_type="invalid_argument",
            message="max_channel_samples must be non-negative.",
        )

    root = resolve_processed_dir(processed_dir)
    definition = REAL_PSSE_CASES[case_id]
    try:
        result = _load_result(root, definition)
        preflight = _load_preflight(root)
    except RealDataError as exc:
        return _error_result(
            case_id=case_id,
            processed_dir=root,
            error_type=exc.error_type,
            message=str(exc),
        )

    try:
        m1_result = _mapping(result.get("m1_powerflow"), "m1_powerflow")
        m2_result = _mapping(result.get("m2_dynamic"), "m2_dynamic")
        m1_stage = _m1_stage(m1_result)
        m2_stage = _m2_stage(
            m2_result,
            dynamic_required=definition.dynamic_required_for_completion,
        )
        emt_stage = _emt_stage()
        recommendation, complete, reason_codes = _recommendation(
            m1_stage,
            m2_stage,
            dynamic_required=definition.dynamic_required_for_completion,
        )
        artifacts = _artifacts(root, definition)
        limitations = _limitations(result.get("limitations"), definition)
    except RealDataError as exc:
        return _error_result(
            case_id=case_id,
            processed_dir=root,
            error_type=exc.error_type,
            message=str(exc),
        )
    channel_preview = None
    if include_channel_preview and definition.channel_csv:
        channel_preview = _csv_preview(root / definition.channel_csv, max_rows=max_channel_samples)

    payload: Dict[str, Any] = {
        "ok": True,
        "tool": "run_real_psse_assessment",
        "backend": "psse_frozen",
        "case_id": case_id,
        "display_name": definition.display_name,
        "processed_dir": str(root),
        "psse_version": _optional_str(result.get("psse_version")) or _preflight_psse_version(preflight),
        "source_files": _mapping_or_empty(result.get("files")),
        "preflight": _preflight_summary(preflight, definition.bundle_id),
        "stage_reports": [m1_stage, m2_stage, emt_stage],
        "m1_result": dict(m1_result),
        "m2_result": dict(m2_result),
        "emt_result": {
            "status": "not_run",
            "backend": "not_available_in_v0",
            "reason_codes": ["psse_rms_dynamic_not_emt_waveform"],
        },
        "recommendation": recommendation,
        "complete": complete,
        "reason_codes": reason_codes,
        "summary": _summary(result, m1_stage, m2_stage),
        "limitations": limitations,
        "artifacts": artifacts,
    }
    if channel_preview is not None:
        payload["channel_preview"] = channel_preview
    return payload


def resolve_processed_dir(processed_dir: Optional[str] = None) -> Path:
    """Resolve the frozen processed-output directory."""

    candidate = processed_dir or os.environ.get(REAL_DATA_DIR_ENV)
    if candidate:
        return Path(candidate).expanduser()
    if DEFAULT_PROCESSED_DIR.exists() or not LEGACY_PROCESSED_DIR.exists():
        return DEFAULT_PROCESSED_DIR
    return LEGACY_PROCESSED_DIR


def _load_result(root: Path, definition: RealPsseCaseDefinition) -> Mapping[str, Any]:
    path = root / definition.result_file
    return _load_json_mapping(path, error_type="real_data_unavailable")


def _load_preflight(root: Path) -> Mapping[str, Any]:
    return _load_json_mapping(root / "preflight_summary.json", error_type="real_data_unavailable")


def _load_preflight_if_available(root: Path) -> Dict[str, Any]:
    try:
        return dict(_load_preflight(root))
    except RealDataError:
        return {}


def _load_json_mapping(path: Path, *, error_type: str) -> Mapping[str, Any]:
    if not path.exists():
        raise RealDataError(f"Required real-data artifact is missing: {path}", error_type=error_type)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RealDataError(
            f"Invalid JSON in real-data artifact {path}: {exc}",
            error_type="real_data_validation_error",
        ) from exc
    except OSError as exc:
        raise RealDataError(
            f"Cannot read real-data artifact {path}: {exc}",
            error_type="real_data_unavailable",
        ) from exc
    if not isinstance(data, Mapping):
        raise RealDataError(
            f"Real-data artifact {path} must contain a JSON object.",
            error_type="real_data_validation_error",
        )
    return data


def _mapping(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RealDataError(
            f"Real-data result field '{key}' must be an object.",
            error_type="real_data_validation_error",
        )
    return value


def _mapping_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _artifact_status(root: Path, definition: RealPsseCaseDefinition) -> Dict[str, Any]:
    files = {
        "result": definition.result_file,
        "bus_voltages": definition.bus_voltage_csv,
        "channels": definition.channel_csv,
        "overloads": definition.overload_csv,
        "poc_candidates": definition.poc_candidates_csv,
        "pq_iterations": definition.pq_iterations_csv,
    }
    artifacts: Dict[str, Any] = {}
    for name, filename in files.items():
        if filename is None:
            continue
        path = root / filename
        artifacts[name] = {
            "path": str(path),
            "exists": path.exists(),
        }
    artifacts["required_available"] = bool((root / definition.result_file).exists())
    return artifacts


def _artifacts(root: Path, definition: RealPsseCaseDefinition) -> Dict[str, Any]:
    artifacts = _artifact_status(root, definition)
    for name, payload in list(artifacts.items()):
        if name == "required_available" or not isinstance(payload, Mapping):
            continue
        path = Path(str(payload.get("path", "")))
        if path.exists() and path.suffix.lower() == ".csv":
            payload = dict(payload)
            payload.update(_csv_summary(path))
            artifacts[name] = payload
    return artifacts


def _csv_summary(path: Path) -> Dict[str, Any]:
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return {"row_count": 0, "columns": []}
            rows = sum(1 for _ in reader)
        return {"row_count": rows, "columns": list(header)}
    except OSError as exc:
        return {"csv_error": str(exc)}


def _csv_preview(path: Path, *, max_rows: int) -> Dict[str, Any]:
    if not path.exists():
        return {
            "available": False,
            "path": str(path),
            "row_count": 0,
            "rows": [],
            "error_type": "csv_missing",
        }
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
    except OSError as exc:
        return {
            "available": False,
            "path": str(path),
            "row_count": 0,
            "rows": [],
            "error_type": "csv_read_error",
            "message": str(exc),
        }
    limited = rows[:max_rows]
    return {
        "available": True,
        "path": str(path),
        "row_count": len(rows),
        "returned_rows": len(limited),
        "truncated_rows": max(0, len(rows) - len(limited)),
        "columns": fieldnames,
        "rows": limited,
    }


def _m1_stage(m1: Mapping[str, Any]) -> Dict[str, Any]:
    converged = bool(m1.get("converged"))
    status = "pass" if str(m1.get("status")) == "pass" and converged else "fail"
    reason_codes = ["psse_powerflow_converged"] if status == "pass" else ["psse_powerflow_failed"]
    overloaded = _optional_int(m1.get("overloaded_branch_count"))
    if overloaded is not None and overloaded > 0:
        status = "fail"
        reason_codes.append("psse_overloaded_branches")
    return {
        "stage": "m1_psse_powerflow",
        "status": status,
        "passed": status == "pass",
        "required_for_approval": True,
        "reason_codes": reason_codes,
        "metrics": {
            "converged": converged,
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
    elif raw_status == "not_run" and not dynamic_required:
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
        },
        "error_summary": list(m2.get("error_summary", []))
        if isinstance(m2.get("error_summary"), list)
        else [],
    }


def _emt_stage() -> Dict[str, Any]:
    return {
        "stage": "f4_emt",
        "status": "not_run",
        "passed": None,
        "required_for_approval": False,
        "reason_codes": ["psse_rms_dynamic_not_emt_waveform"],
    }


def _recommendation(
    m1_stage: Mapping[str, Any],
    m2_stage: Mapping[str, Any],
    *,
    dynamic_required: bool,
) -> tuple[str, bool, List[str]]:
    if m1_stage.get("passed") is not True:
        return "reject", False, ["m1_psse_powerflow_failed"]
    if dynamic_required and m2_stage.get("passed") is not True:
        return "reject", False, ["m2_psse_rms_dynamic_failed"]
    reason_codes = ["real_psse_frozen_result_pass"]
    if m2_stage.get("status") == "skipped":
        reason_codes.append("dynamic_not_run")
    return "approve", True, reason_codes


def _summary(
    result: Mapping[str, Any],
    m1_stage: Mapping[str, Any],
    m2_stage: Mapping[str, Any],
) -> Dict[str, Any]:
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
        "emt_status": "not_run",
        "backend": _optional_str(result.get("backend")),
    }


def _preflight_summary(preflight: Mapping[str, Any], bundle_id: str) -> Dict[str, Any]:
    bundle = _mapping_or_empty(_mapping_or_empty(preflight.get("bundles")).get(bundle_id))
    dlls = _mapping_or_empty(bundle.get("dll_load_check")).get("dlls", [])
    dll_load_successful = (
        all(bool(item.get("loaded")) for item in dlls)
        if isinstance(dlls, list)
        else None
    )
    return {
        "psse_version": _preflight_psse_version(preflight),
        "python_version": _optional_str(preflight.get("python_version")),
        "bundle_id": bundle_id,
        "dyr_model_counts": _mapping_or_empty(bundle.get("dyr_model_counts")),
        "dll_load_successful": dll_load_successful,
        "dynamic_error_summary": list(bundle.get("dynamic_error_summary", []))
        if isinstance(bundle.get("dynamic_error_summary"), list)
        else [],
    }


def _preflight_psse_version(preflight: Mapping[str, Any]) -> Optional[str]:
    return _optional_str(preflight.get("psse_version"))


def _limitations(value: Any, definition: RealPsseCaseDefinition) -> List[str]:
    limitations = [str(item) for item in value] if isinstance(value, list) else []
    limitations.extend(
        [
            "frozen_psse_outputs_only_no_local_psse_execution",
            "v0_no_new_interconnection_modeling_on_real_case",
            "labels_not_yet_expert_validated",
        ]
    )
    if not definition.dynamic_required_for_completion:
        limitations.append("dynamic_not_run")
    if PSSE_RMS_NOT_EMT_LIMITATION not in limitations:
        limitations.append(PSSE_RMS_NOT_EMT_LIMITATION)
    return list(dict.fromkeys(limitations))


def _error_result(
    *,
    case_id: str,
    processed_dir: Path,
    error_type: str,
    message: str,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "tool": "run_real_psse_assessment",
        "backend": "psse_frozen",
        "case_id": case_id,
        "processed_dir": str(processed_dir),
        "error_type": error_type,
        "message": message,
        "recommendation": "reject",
        "complete": False,
        "reason_codes": [error_type],
    }


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

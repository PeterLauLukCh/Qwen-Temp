"""Frozen real-data PSS/E interconnection replay environment.

This module defines the offline interconnection gym used for real PSS/E data.
It never executes PSS/E. Instead, it reads inventory exports to generate a
scenario catalog, then serves exact-match lookups from precomputed result JSONs
created by an external PSS/E batch runner.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .real_data import REAL_DATA_DIR_ENV, resolve_processed_dir


REAL_INTERCONNECTION_DIR_ENV = "POWERGYM_REAL_INTERCONNECTION_DIR"
DEFAULT_FROZEN_INTERCONNECTION_DIR = (
    Path(__file__).resolve().parents[2] / "real-data-new" / "frozen_interconnection"
)
DEFAULT_CASE_ID = "pif6_2026_05_17"
DEFAULT_SIMULATION_TIME_S = 5.0
DEFAULT_FAULT_START_S = 1.0
DEFAULT_PROJECT_SIZES_MW = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
DEFAULT_CONNECTION_TYPES = (
    "solar_static_pq",
    "wind_static_pq",
    "bess_static_pq",
    "load_static_pq",
)
DEFAULT_Q_MODES = ("unity_pf", "pf_0_95_lag", "pf_0_95_lead")
DEFAULT_CLEARING_TIMES_S = (1.08, 1.10, 1.15)
REAL_INTERCONNECTION_SCHEMA_VERSION = "real_interconnection_frozen_v1"
STATIC_PQ_LIMITATION = "static_pq_interconnection_only_no_full_device_dynamics"
FROZEN_REPLAY_LIMITATION = "frozen_psse_precomputed_results_only_no_live_psse"
EXACT_MATCH_LIMITATION = "exact_match_lookup_only"


class RealInterconnectionError(RuntimeError):
    """Raised for invalid frozen interconnection artifacts."""

    def __init__(self, message: str, *, error_type: str = "real_interconnection_error") -> None:
        super().__init__(message)
        self.error_type = error_type


@dataclass(frozen=True)
class ScenarioCatalogBuildResult:
    """Generated scenario catalog plus metadata."""

    case_id: str
    scenarios: Sequence[Dict[str, Any]]
    criteria: Dict[str, Any]
    action_space: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "case_id": self.case_id,
            "scenario_count": len(self.scenarios),
            "criteria": self.criteria,
            "action_space": self.action_space,
            "scenarios": list(self.scenarios),
        }


def resolve_frozen_interconnection_dir(processed_dir: Optional[str] = None) -> Path:
    """Resolve the frozen interconnection result directory."""

    candidate = processed_dir or os.environ.get(REAL_INTERCONNECTION_DIR_ENV)
    if candidate:
        return Path(candidate).expanduser()
    return DEFAULT_FROZEN_INTERCONNECTION_DIR


def list_real_interconnection_actions(
    *,
    processed_dir: Optional[str] = None,
    inventory_dir: Optional[str] = None,
    case_id: str = DEFAULT_CASE_ID,
) -> Dict[str, Any]:
    """List action-space metadata and frozen-result availability."""

    frozen_root = resolve_frozen_interconnection_dir(processed_dir)
    inventory_root = resolve_processed_dir(inventory_dir)
    try:
        inventory = load_inventory(case_id=case_id, processed_dir=str(inventory_root))
        action_space = build_action_space(inventory)
    except RealInterconnectionError as exc:
        return {
            "ok": False,
            "tool": "list_real_interconnection_actions",
            "backend": "psse_frozen_interconnection",
            "case_id": case_id,
            "processed_dir": str(frozen_root),
            "inventory_dir": str(inventory_root),
            "error_type": exc.error_type,
            "message": str(exc),
            "limitations": _base_limitations(),
        }

    catalog = _load_catalog_if_available(frozen_root)
    index = _load_index_if_available(frozen_root)
    result_count = _result_count(frozen_root, index)
    return {
        "ok": True,
        "tool": "list_real_interconnection_actions",
        "backend": "psse_frozen_interconnection",
        "case_id": case_id,
        "processed_dir": str(frozen_root),
        "inventory_dir": str(inventory_root),
        "catalog_available": bool(catalog),
        "index_available": bool(index),
        "catalog_scenario_count": len(catalog),
        "precomputed_result_count": result_count,
        "action_space": action_space,
        "limitations": _base_limitations(),
    }


def run_real_interconnection_assessment(
    *,
    case_id: str,
    connection: Mapping[str, Any],
    disturbance: Mapping[str, Any],
    processed_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Return one exact-match frozen PSS/E interconnection assessment."""

    frozen_root = resolve_frozen_interconnection_dir(processed_dir)
    try:
        normalized_connection = normalize_connection(connection)
        normalized_disturbance = normalize_disturbance(disturbance)
        scenario_key = build_scenario_key(
            case_id=case_id,
            connection=normalized_connection,
            disturbance=normalized_disturbance,
        )
        catalog_by_key = _catalog_by_key(frozen_root)
        index = _index_by_key(frozen_root, catalog_by_key)
        entry = index.get(scenario_key)
        if entry is None:
            return _missing_precomputed_result(
                case_id=case_id,
                processed_dir=frozen_root,
                scenario_key=scenario_key,
                connection=normalized_connection,
                disturbance=normalized_disturbance,
                message="No exact precomputed PSS/E result exists for this scenario.",
            )
        result_path = frozen_root / str(entry.get("result_file", ""))
        if not result_path.exists():
            return _missing_precomputed_result(
                case_id=case_id,
                processed_dir=frozen_root,
                scenario_key=scenario_key,
                connection=normalized_connection,
                disturbance=normalized_disturbance,
                scenario_id=_optional_str(entry.get("scenario_id")),
                message=f"Precomputed scenario is indexed but result JSON is missing: {result_path}",
            )
        raw_result = _load_json_mapping(
            result_path,
            error_type="real_interconnection_validation_error",
        )
        return _adapt_precomputed_result(
            raw_result,
            case_id=case_id,
            scenario_key=scenario_key,
            processed_dir=frozen_root,
            connection=normalized_connection,
            disturbance=normalized_disturbance,
            result_file=str(result_path),
            index_entry=entry,
        )
    except RealInterconnectionError as exc:
        return {
            "ok": False,
            "tool": "run_real_interconnection_assessment",
            "backend": "psse_frozen_interconnection",
            "case_id": case_id,
            "processed_dir": str(frozen_root),
            "error_type": exc.error_type,
            "message": str(exc),
            "recommendation": "reject",
            "complete": False,
            "reason_codes": [exc.error_type],
            "limitations": _base_limitations(),
        }


def generate_real_interconnection_scenarios(
    *,
    case_id: str = DEFAULT_CASE_ID,
    processed_dir: Optional[str] = None,
    project_sizes_mw: Sequence[float] = DEFAULT_PROJECT_SIZES_MW,
    connection_types: Sequence[str] = DEFAULT_CONNECTION_TYPES,
    q_modes: Sequence[str] = DEFAULT_Q_MODES,
    clearing_times_s: Sequence[float] = DEFAULT_CLEARING_TIMES_S,
    simulation_time_s: float = DEFAULT_SIMULATION_TIME_S,
) -> ScenarioCatalogBuildResult:
    """Generate deterministic scenario catalog rows from inventory exports."""

    inventory = load_inventory(case_id=case_id, processed_dir=processed_dir)
    action_space = build_action_space(
        inventory,
        project_sizes_mw=project_sizes_mw,
        connection_types=connection_types,
        q_modes=q_modes,
        clearing_times_s=clearing_times_s,
        simulation_time_s=simulation_time_s,
    )
    criteria = default_interconnection_criteria()
    scenarios: List[Dict[str, Any]] = []
    for bus in action_space["connection_buses"]:
        for connection_type in action_space["connection_types"]:
            for p_mw in action_space["project_sizes_mw"]:
                bess_modes = ["charge", "discharge"] if connection_type == "bess_static_pq" else [None]
                for bess_mode in bess_modes:
                    for q_mode in action_space["q_modes"]:
                        connection = normalize_connection(
                            {
                                "bus": bus["bus"],
                                "connection_type": connection_type,
                                "p_mw": p_mw,
                                "q_mode": q_mode,
                                **({"bess_mode": bess_mode} if bess_mode else {}),
                            }
                        )
                        for disturbance in action_space["disturbances"]:
                            normalized_disturbance = normalize_disturbance(disturbance)
                            scenarios.append(
                                _scenario_catalog_row(
                                    case_id=case_id,
                                    connection=connection,
                                    disturbance=normalized_disturbance,
                                    criteria=criteria,
                                )
                            )
    return ScenarioCatalogBuildResult(
        case_id=case_id,
        scenarios=scenarios,
        criteria=criteria,
        action_space=action_space,
    )


def write_real_interconnection_catalog(
    *,
    output: str,
    build_result: ScenarioCatalogBuildResult,
) -> Dict[str, Any]:
    """Write scenario_catalog.jsonl, index.json, and criteria.json."""

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for scenario in build_result.scenarios:
            handle.write(json.dumps(scenario, sort_keys=True) + "\n")
    index = {
        "schema_version": REAL_INTERCONNECTION_SCHEMA_VERSION,
        "case_id": build_result.case_id,
        "scenario_count": len(build_result.scenarios),
        "lookup_policy": "exact_match_only",
        "scenarios": {
            str(scenario["scenario_key"]): {
                "scenario_id": scenario["scenario_id"],
                "result_file": scenario["result_file"],
            }
            for scenario in build_result.scenarios
        },
    }
    criteria_path = output_path.parent / "criteria.json"
    index_path = output_path.parent / "index.json"
    criteria_path.write_text(
        json.dumps(build_result.criteria, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_path.parent / "results").mkdir(exist_ok=True)
    return {
        "ok": True,
        "case_id": build_result.case_id,
        "scenario_count": len(build_result.scenarios),
        "catalog_path": str(output_path),
        "index_path": str(index_path),
        "criteria_path": str(criteria_path),
        "results_dir": str(output_path.parent / "results"),
    }


def load_inventory(*, case_id: str, processed_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load CSV inventory tables for one processed PSS/E case."""

    root = resolve_processed_dir(processed_dir)
    case_dir = root / "inventories" / case_id
    if not case_dir.exists():
        raise RealInterconnectionError(
            f"Inventory directory is missing: {case_dir}",
            error_type="inventory_unavailable",
        )
    tables = {}
    for name in (
        "branches",
        "dynamic_models",
        "loads",
        "machines",
        "poc_candidates",
        "shunts",
        "switched_shunts",
        "transformers_2w",
        "transformers_3w",
    ):
        tables[name] = _read_csv_dicts(case_dir / f"{name}.csv")
    return {
        "case_id": case_id,
        "processed_dir": str(root),
        "case_dir": str(case_dir),
        "tables": tables,
        "summary": {name: len(rows) for name, rows in tables.items()},
    }


def build_action_space(
    inventory: Mapping[str, Any],
    *,
    project_sizes_mw: Sequence[float] = DEFAULT_PROJECT_SIZES_MW,
    connection_types: Sequence[str] = DEFAULT_CONNECTION_TYPES,
    q_modes: Sequence[str] = DEFAULT_Q_MODES,
    clearing_times_s: Sequence[float] = DEFAULT_CLEARING_TIMES_S,
    simulation_time_s: float = DEFAULT_SIMULATION_TIME_S,
) -> Dict[str, Any]:
    """Build a conservative v1 action space from inventory tables."""

    tables = _mapping(inventory.get("tables"), "tables")
    poc_rows = _list_of_mappings(tables.get("poc_candidates"), "poc_candidates")
    bus_candidates = []
    seen_buses = set()
    for row in poc_rows:
        if str(row.get("candidate_type", "")).strip().lower() != "bus":
            continue
        bus = _optional_int(row.get("bus"))
        if bus is None or bus in seen_buses:
            continue
        seen_buses.add(bus)
        bus_candidates.append(
            {
                "bus": bus,
                "name": _optional_str(row.get("name")),
                "voltage_pu": _optional_number(row.get("voltage_pu")),
                "reason": _optional_str(row.get("reason")),
            }
        )
    if not bus_candidates:
        raise RealInterconnectionError(
            "No candidate POC buses found in poc_candidates.csv.",
            error_type="inventory_validation_error",
        )

    branch_events = []
    seen_lines = set()
    for row in poc_rows:
        if str(row.get("candidate_type", "")).strip().lower() != "branch":
            continue
        from_bus = _optional_int(row.get("from_bus"))
        to_bus = _optional_int(row.get("to_bus"))
        circuit_id = _optional_str(row.get("circuit_id")) or "1"
        if from_bus is None or to_bus is None:
            continue
        key = (from_bus, to_bus, circuit_id)
        if key in seen_lines:
            continue
        seen_lines.add(key)
        branch_events.append(
            {
                "type": "line_trip",
                "from_bus": from_bus,
                "to_bus": to_bus,
                "circuit_id": circuit_id,
                "trip_time_s": DEFAULT_FAULT_START_S,
                "simulation_time_s": simulation_time_s,
            }
        )

    bus_faults = []
    for bus in bus_candidates:
        for clearing_time_s in clearing_times_s:
            bus_faults.append(
                {
                    "type": "bus_fault",
                    "bus": bus["bus"],
                    "fault_start_s": DEFAULT_FAULT_START_S,
                    "clearing_time_s": _round_float(clearing_time_s),
                    "simulation_time_s": simulation_time_s,
                }
            )

    disturbances = [
        {"type": "no_disturbance", "simulation_time_s": simulation_time_s},
        *bus_faults,
        *branch_events,
    ]
    return {
        "case_id": _optional_str(inventory.get("case_id")),
        "connection_buses": bus_candidates,
        "connection_types": [normalize_connection_type(item) for item in connection_types],
        "project_sizes_mw": [_round_float(item) for item in project_sizes_mw],
        "q_modes": [normalize_q_mode(item) for item in q_modes],
        "bess_modes": ["charge", "discharge"],
        "disturbances": disturbances,
        "disturbance_count": len(disturbances),
        "scenario_count_estimate": (
            len(bus_candidates)
            * sum(2 if normalize_connection_type(item) == "bess_static_pq" else 1 for item in connection_types)
            * len(project_sizes_mw)
            * len(q_modes)
            * len(disturbances)
        ),
        "inventory_summary": dict(inventory.get("summary", {})),
    }


def normalize_connection(connection: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a real interconnection action for exact lookup."""

    if not isinstance(connection, Mapping):
        raise RealInterconnectionError("connection must be an object.", error_type="invalid_argument")
    bus = _required_int(connection, "bus")
    p_mw = _required_number(connection, "p_mw")
    if p_mw <= 0:
        raise RealInterconnectionError("connection.p_mw must be positive.", error_type="invalid_argument")
    connection_type = normalize_connection_type(_required_str(connection, "connection_type"))
    q_mode = normalize_q_mode(_optional_str(connection.get("q_mode")) or "unity_pf")
    bess_mode = _optional_str(connection.get("bess_mode"))
    if connection_type == "bess_static_pq":
        bess_mode = (bess_mode or "discharge").lower()
        if bess_mode not in {"charge", "discharge"}:
            raise RealInterconnectionError(
                "connection.bess_mode must be charge or discharge.",
                error_type="invalid_argument",
            )
    elif bess_mode:
        raise RealInterconnectionError(
            "connection.bess_mode is only valid for bess_static_pq.",
            error_type="invalid_argument",
        )
    q_mvar = _optional_number(connection.get("q_mvar"))
    if q_mvar is None:
        q_mvar = _q_from_mode(p_mw, q_mode)
    p_load_mw = _static_load_equivalent_p(p_mw, connection_type, bess_mode)
    return {
        "bus": bus,
        "connection_type": connection_type,
        "p_mw": _round_float(p_mw),
        "q_mode": q_mode,
        "q_mvar": _round_float(q_mvar),
        "bess_mode": bess_mode,
        "psse_static_model": "load_equivalent",
        "p_load_mw": _round_float(p_load_mw),
        "q_load_mvar": _round_float(-q_mvar if p_load_mw < 0 else q_mvar),
    }


def normalize_disturbance(disturbance: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize disturbance parameters for exact lookup."""

    if not isinstance(disturbance, Mapping):
        raise RealInterconnectionError("disturbance must be an object.", error_type="invalid_argument")
    dtype = _required_str(disturbance, "type").lower()
    simulation_time_s = _optional_number(disturbance.get("simulation_time_s"))
    if simulation_time_s is None:
        simulation_time_s = DEFAULT_SIMULATION_TIME_S
    if dtype in {"no_disturbance", "no_disturbance_5s", "none"}:
        return {
            "type": "no_disturbance",
            "simulation_time_s": _round_float(simulation_time_s),
        }
    if dtype == "bus_fault":
        return {
            "type": "bus_fault",
            "bus": _required_int(disturbance, "bus"),
            "fault_start_s": _round_float(
                _optional_number(disturbance.get("fault_start_s")) or DEFAULT_FAULT_START_S
            ),
            "clearing_time_s": _round_float(_required_number(disturbance, "clearing_time_s")),
            "simulation_time_s": _round_float(simulation_time_s),
        }
    if dtype == "line_trip":
        return {
            "type": "line_trip",
            "from_bus": _required_int(disturbance, "from_bus"),
            "to_bus": _required_int(disturbance, "to_bus"),
            "circuit_id": _optional_str(disturbance.get("circuit_id")) or "1",
            "trip_time_s": _round_float(
                _optional_number(disturbance.get("trip_time_s")) or DEFAULT_FAULT_START_S
            ),
            "simulation_time_s": _round_float(simulation_time_s),
        }
    raise RealInterconnectionError(
        f"Unsupported disturbance type '{dtype}'.",
        error_type="invalid_argument",
    )


def normalize_connection_type(value: Any) -> str:
    text = str(value).strip().lower()
    aliases = {
        "solar": "solar_static_pq",
        "pv": "solar_static_pq",
        "wind": "wind_static_pq",
        "bess": "bess_static_pq",
        "battery": "bess_static_pq",
        "load": "load_static_pq",
        "data_center": "load_static_pq",
        "datacenter": "load_static_pq",
    }
    normalized = aliases.get(text, text)
    if normalized not in DEFAULT_CONNECTION_TYPES:
        raise RealInterconnectionError(
            f"Unsupported connection_type '{value}'.",
            error_type="invalid_argument",
        )
    return normalized


def normalize_q_mode(value: Any) -> str:
    text = str(value).strip().lower()
    aliases = {
        "unity": "unity_pf",
        "unity_power_factor": "unity_pf",
        "0.95_lag": "pf_0_95_lag",
        "0.95_lead": "pf_0_95_lead",
    }
    normalized = aliases.get(text, text)
    if normalized not in DEFAULT_Q_MODES:
        raise RealInterconnectionError(
            f"Unsupported q_mode '{value}'.",
            error_type="invalid_argument",
        )
    return normalized


def build_scenario_key(
    *,
    case_id: str,
    connection: Mapping[str, Any],
    disturbance: Mapping[str, Any],
) -> str:
    """Build the canonical exact-match key for one scenario."""

    payload = {
        "case_id": str(case_id),
        "connection": _canonicalize(connection),
        "disturbance": _canonicalize(disturbance),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def default_interconnection_criteria() -> Dict[str, Any]:
    """Default pseudo-label criteria stored with generated catalogs."""

    return {
        "schema_version": REAL_INTERCONNECTION_SCHEMA_VERSION,
        "label_source": "psse_agent_pseudo_label_not_expert_final_planning_label",
        "m1": {
            "powerflow_converged_required": True,
            "max_overloaded_branch_count": 0,
            "voltage_min_pu": 0.90,
            "voltage_max_pu": 1.10,
        },
        "m2": {
            "rms_initialized_required": True,
            "rms_simulation_converged_required": True,
            "min_voltage_pu": 0.70,
            "max_frequency_deviation_hz": 0.50,
        },
        "labels": ["approve", "reject", "borderline"],
    }


def _scenario_catalog_row(
    *,
    case_id: str,
    connection: Mapping[str, Any],
    disturbance: Mapping[str, Any],
    criteria: Mapping[str, Any],
) -> Dict[str, Any]:
    scenario_key = build_scenario_key(
        case_id=case_id,
        connection=connection,
        disturbance=disturbance,
    )
    scenario_id = _scenario_id(case_id, connection, disturbance, scenario_key)
    return {
        "schema_version": REAL_INTERCONNECTION_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "scenario_key": scenario_key,
        "case_id": case_id,
        "connection": dict(connection),
        "disturbance": dict(disturbance),
        "criteria_version": criteria.get("schema_version"),
        "psse_batch_instructions": {
            "apply_interconnection": "static_pq_load_equivalent",
            "run_powerflow": True,
            "run_rms_dynamic_if_powerflow_passes": True,
            "emit_compact_json": True,
        },
        "result_file": f"results/{scenario_id}.json",
        "limitations": _base_limitations(),
    }


def _scenario_id(
    case_id: str,
    connection: Mapping[str, Any],
    disturbance: Mapping[str, Any],
    scenario_key: str,
) -> str:
    parts = [
        _slug(case_id),
        _slug(str(connection.get("connection_type"))),
        f"bus{connection.get('bus')}",
        f"p{_number_slug(connection.get('p_mw'))}mw",
        _slug(str(connection.get("q_mode"))),
    ]
    if connection.get("bess_mode"):
        parts.append(_slug(str(connection.get("bess_mode"))))
    dtype = str(disturbance.get("type"))
    if dtype == "bus_fault":
        parts.extend(
            [
                "fault",
                f"bus{disturbance.get('bus')}",
                f"clear{_number_slug(disturbance.get('clearing_time_s'))}s",
            ]
        )
    elif dtype == "line_trip":
        parts.extend(
            [
                "trip",
                f"{disturbance.get('from_bus')}_{disturbance.get('to_bus')}_{disturbance.get('circuit_id')}",
            ]
        )
    else:
        parts.append("no_disturbance")
    parts.append(scenario_key[:10])
    return "_".join(parts)


def _adapt_precomputed_result(
    raw_result: Mapping[str, Any],
    *,
    case_id: str,
    scenario_key: str,
    processed_dir: Path,
    connection: Mapping[str, Any],
    disturbance: Mapping[str, Any],
    result_file: str,
    index_entry: Mapping[str, Any],
) -> Dict[str, Any]:
    m1_result = _mapping_or_empty(raw_result.get("m1_result") or raw_result.get("m1_powerflow"))
    m2_result = _mapping_or_empty(raw_result.get("m2_result") or raw_result.get("m2_dynamic"))
    emt_or_scr = _mapping_or_empty(
        raw_result.get("emt_or_scr_result")
        or raw_result.get("scr_result")
        or raw_result.get("scr_screen")
    )
    stage_reports = raw_result.get("stage_reports")
    if not isinstance(stage_reports, list):
        stage_reports = [
            _m1_stage(m1_result),
            _m2_stage(m2_result),
            _scr_stage(emt_or_scr),
        ]
    recommendation = _optional_str(raw_result.get("recommendation") or raw_result.get("label"))
    if recommendation is None:
        recommendation = _recommend_from_stages(stage_reports)
    complete = raw_result.get("complete")
    if not isinstance(complete, bool):
        complete = recommendation in {"approve", "borderline"} and all(
            stage.get("passed") is not False
            for stage in stage_reports
            if isinstance(stage, Mapping)
        )
    summary = _mapping_or_empty(raw_result.get("summary"))
    if not summary:
        summary = _summary(m1_result, m2_result, emt_or_scr, stage_reports)
    limitations = _string_list(raw_result.get("limitations"))
    for item in _base_limitations():
        if item not in limitations:
            limitations.append(item)
    return {
        "ok": True,
        "tool": "run_real_interconnection_assessment",
        "backend": "psse_frozen_interconnection",
        "schema_version": REAL_INTERCONNECTION_SCHEMA_VERSION,
        "case_id": case_id,
        "scenario_id": _optional_str(raw_result.get("scenario_id"))
        or _optional_str(index_entry.get("scenario_id")),
        "scenario_key": scenario_key,
        "processed_dir": str(processed_dir),
        "result_file": result_file,
        "lookup_policy": "exact_match_only",
        "connection": dict(connection),
        "disturbance": dict(disturbance),
        "source_files": _mapping_or_empty(raw_result.get("source_files") or raw_result.get("files")),
        "stage_reports": stage_reports,
        "m1_result": m1_result,
        "m2_result": m2_result,
        "emt_or_scr_result": emt_or_scr,
        "recommendation": recommendation,
        "complete": complete,
        "summary": summary,
        "reason_codes": _string_list(raw_result.get("reason_codes")),
        "limitations": limitations,
        "raw_label_source": _optional_str(raw_result.get("label_source")),
    }


def _missing_precomputed_result(
    *,
    case_id: str,
    processed_dir: Path,
    scenario_key: str,
    connection: Mapping[str, Any],
    disturbance: Mapping[str, Any],
    message: str,
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "tool": "run_real_interconnection_assessment",
        "backend": "psse_frozen_interconnection",
        "case_id": case_id,
        "scenario_id": scenario_id,
        "scenario_key": scenario_key,
        "processed_dir": str(processed_dir),
        "error_type": "precomputed_result_unavailable",
        "message": message,
        "connection": dict(connection),
        "disturbance": dict(disturbance),
        "recommendation": "reject",
        "complete": False,
        "reason_codes": ["precomputed_result_unavailable"],
        "limitations": _base_limitations(),
    }


def _m1_stage(m1: Mapping[str, Any]) -> Dict[str, Any]:
    status = str(m1.get("status", "unknown"))
    converged = bool(m1.get("converged"))
    overloaded = _optional_int(m1.get("overloaded_branch_count"))
    passed = status == "pass" and converged and (overloaded is None or overloaded == 0)
    return {
        "stage": "m1_psse_powerflow",
        "status": "pass" if passed else "fail",
        "passed": passed,
        "required_for_approval": True,
        "metrics": {
            "converged": converged,
            "bus_voltage_min_pu": _optional_number(m1.get("bus_voltage_min_pu")),
            "bus_voltage_max_pu": _optional_number(m1.get("bus_voltage_max_pu")),
            "overloaded_branch_count": overloaded,
            "poc_p_mw": _optional_number(m1.get("poc_p_mw")),
            "poc_q_mvar": _optional_number(m1.get("poc_q_mvar")),
        },
    }


def _m2_stage(m2: Mapping[str, Any]) -> Dict[str, Any]:
    status = str(m2.get("status", "unknown"))
    initialized = bool(m2.get("initialized"))
    simulation_converged = bool(m2.get("simulation_converged"))
    passed = status == "pass" and initialized and simulation_converged
    metrics = _mapping_or_empty(m2.get("metrics"))
    return {
        "stage": "m2_psse_rms_dynamic",
        "status": "pass" if passed else "fail",
        "passed": passed,
        "required_for_approval": True,
        "event": _optional_str(m2.get("event")),
        "metrics": {
            "initialized": initialized,
            "simulation_converged": simulation_converged,
            "min_voltage_pu": _optional_number(metrics.get("min_voltage_pu")),
            "max_voltage_pu": _optional_number(metrics.get("max_voltage_pu")),
            "max_frequency_deviation_hz": _optional_number(
                metrics.get("max_frequency_deviation_hz")
            ),
        },
    }


def _scr_stage(scr: Mapping[str, Any]) -> Dict[str, Any]:
    if not scr:
        return {
            "stage": "f4_scr_or_emt_proxy",
            "status": "not_available",
            "passed": None,
            "required_for_approval": False,
            "reason_codes": ["scr_or_emt_not_exported"],
        }
    status = str(scr.get("status", "unknown"))
    passed = scr.get("passed")
    if not isinstance(passed, bool):
        passed = status == "pass"
    return {
        "stage": "f4_scr_or_emt_proxy",
        "status": status,
        "passed": passed,
        "required_for_approval": bool(scr.get("required_for_approval", False)),
        "metrics": _mapping_or_empty(scr.get("metrics")),
    }


def _summary(
    m1: Mapping[str, Any],
    m2: Mapping[str, Any],
    scr: Mapping[str, Any],
    stage_reports: Sequence[Any],
) -> Dict[str, Any]:
    m2_metrics = _mapping_or_empty(m2.get("metrics"))
    scr_metrics = _mapping_or_empty(scr.get("metrics"))
    return {
        "m1_status": _stage_status(stage_reports, "m1_psse_powerflow"),
        "m2_status": _stage_status(stage_reports, "m2_psse_rms_dynamic"),
        "scr_or_emt_status": _stage_status(stage_reports, "f4_scr_or_emt_proxy"),
        "m1_converged": bool(m1.get("converged")),
        "m1_bus_voltage_min_pu": _optional_number(m1.get("bus_voltage_min_pu")),
        "m1_bus_voltage_max_pu": _optional_number(m1.get("bus_voltage_max_pu")),
        "m1_overloaded_branch_count": _optional_int(m1.get("overloaded_branch_count")),
        "m2_initialized": bool(m2.get("initialized")),
        "m2_simulation_converged": bool(m2.get("simulation_converged")),
        "m2_min_voltage_pu": _optional_number(m2_metrics.get("min_voltage_pu")),
        "scr": _optional_number(scr_metrics.get("scr")),
    }


def _stage_status(stage_reports: Sequence[Any], stage: str) -> Optional[str]:
    for item in stage_reports:
        if isinstance(item, Mapping) and item.get("stage") == stage:
            return _optional_str(item.get("status"))
    return None


def _recommend_from_stages(stage_reports: Sequence[Any]) -> str:
    statuses = [
        item.get("status")
        for item in stage_reports
        if isinstance(item, Mapping) and item.get("required_for_approval") is True
    ]
    if any(status == "fail" for status in statuses):
        return "reject"
    if any(status == "borderline" for status in statuses):
        return "borderline"
    return "approve"


def _load_catalog_if_available(root: Path) -> List[Dict[str, Any]]:
    path = root / "scenario_catalog.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RealInterconnectionError(
                    f"Invalid JSONL at {path}:{line_no}: {exc}",
                    error_type="real_interconnection_validation_error",
                ) from exc
            if not isinstance(row, Mapping):
                raise RealInterconnectionError(
                    f"Catalog row at {path}:{line_no} must be an object.",
                    error_type="real_interconnection_validation_error",
                )
            rows.append(dict(row))
    return rows


def _load_index_if_available(root: Path) -> Dict[str, Any]:
    path = root / "index.json"
    if not path.exists():
        return {}
    return dict(_load_json_mapping(path, error_type="real_interconnection_validation_error"))


def _catalog_by_key(root: Path) -> Dict[str, Dict[str, Any]]:
    catalog = _load_catalog_if_available(root)
    return {
        str(row.get("scenario_key")): row
        for row in catalog
        if row.get("scenario_key")
    }


def _index_by_key(root: Path, catalog_by_key: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index = _load_index_if_available(root)
    scenarios = index.get("scenarios") if isinstance(index, Mapping) else None
    if isinstance(scenarios, Mapping):
        return {
            str(key): dict(value)
            for key, value in scenarios.items()
            if isinstance(value, Mapping)
        }
    return {
        key: {
            "scenario_id": value.get("scenario_id"),
            "result_file": value.get("result_file"),
        }
        for key, value in catalog_by_key.items()
    }


def _result_count(root: Path, index: Mapping[str, Any]) -> int:
    scenarios = index.get("scenarios") if isinstance(index, Mapping) else None
    if not isinstance(scenarios, Mapping):
        return len(list((root / "results").glob("*.json"))) if (root / "results").exists() else 0
    count = 0
    for value in scenarios.values():
        if isinstance(value, Mapping) and (root / str(value.get("result_file", ""))).exists():
            count += 1
    return count


def _load_json_mapping(path: Path, *, error_type: str) -> Mapping[str, Any]:
    if not path.exists():
        raise RealInterconnectionError(f"Required artifact is missing: {path}", error_type=error_type)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RealInterconnectionError(
            f"Invalid JSON in artifact {path}: {exc}",
            error_type=error_type,
        ) from exc
    if not isinstance(data, Mapping):
        raise RealInterconnectionError(
            f"Artifact {path} must contain a JSON object.",
            error_type=error_type,
        )
    return data


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise RealInterconnectionError(
            f"Inventory CSV is missing: {path}",
            error_type="inventory_unavailable",
        )
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise RealInterconnectionError(
            f"Cannot read inventory CSV {path}: {exc}",
            error_type="inventory_unavailable",
        ) from exc


def _q_from_mode(p_mw: float, q_mode: str) -> float:
    if q_mode == "unity_pf":
        return 0.0
    q = abs(p_mw) * math.tan(math.acos(0.95))
    return q if q_mode == "pf_0_95_lag" else -q


def _static_load_equivalent_p(
    p_mw: float,
    connection_type: str,
    bess_mode: Optional[str],
) -> float:
    if connection_type == "load_static_pq":
        return p_mw
    if connection_type == "bess_static_pq" and bess_mode == "charge":
        return p_mw
    return -p_mw


def _base_limitations() -> List[str]:
    return [
        FROZEN_REPLAY_LIMITATION,
        EXACT_MATCH_LIMITATION,
        STATIC_PQ_LIMITATION,
        "labels_not_yet_expert_validated",
        "psse_rms_dynamic_not_emt_waveform_unless_emt_data_exported",
    ]


def _required_str(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RealInterconnectionError(f"{key} must be a non-empty string.", error_type="invalid_argument")
    return value.strip()


def _required_int(mapping: Mapping[str, Any], key: str) -> int:
    value = _optional_int(mapping.get(key))
    if value is None:
        raise RealInterconnectionError(f"{key} must be an integer.", error_type="invalid_argument")
    return value


def _required_number(mapping: Mapping[str, Any], key: str) -> float:
    value = _optional_number(mapping.get(key))
    if value is None:
        raise RealInterconnectionError(f"{key} must be a number.", error_type="invalid_argument")
    return value


def _optional_str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _optional_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            as_float = float(value)
        except ValueError:
            return None
        if as_float.is_integer():
            return int(as_float)
    return None


def _optional_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _round_float(value: Any) -> float:
    number = _optional_number(value)
    if number is None:
        raise RealInterconnectionError(f"Expected number, got {value!r}", error_type="invalid_argument")
    return round(float(number), 8)


def _number_slug(value: Any) -> str:
    number = _optional_number(value)
    if number is None:
        return "unknown"
    return str(_round_float(number)).replace("-", "m").replace(".", "p")


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        return _round_float(value)
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RealInterconnectionError(f"{name} must be an object.", error_type="inventory_validation_error")
    return value


def _mapping_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any, name: str) -> List[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise RealInterconnectionError(f"{name} must be a list.", error_type="inventory_validation_error")
    return [item for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]

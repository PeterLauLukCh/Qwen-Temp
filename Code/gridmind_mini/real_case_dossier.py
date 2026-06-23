"""Read-only engineer dossier tools for processed real PSS/E cases.

These helpers expose compact, bounded views of the processed PSS/E artifacts
already present on the GPU side. They never load SAV/DYR/DLL files and never
execute PSS/E; live execution remains behind the remote worker allowlist.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .real_data import resolve_processed_dir
from .remote_psse import REMOTE_PSSE_M1M2_CASES


REAL_CASE_DOSSIER_SCHEMA_VERSION = "real_case_dossier_v1"
REAL_DOSSIER_TOOL_NAMES = (
    "inspect_real_case_summary",
    "inspect_real_poc_context",
    "inspect_real_network_neighborhood",
    "inspect_real_model_inventory",
    "inspect_real_static_operating_point",
    "inspect_real_dynamic_channels",
)


class RealCaseDossierError(RuntimeError):
    """Raised for missing or invalid processed real-case artifacts."""

    def __init__(self, message: str, *, error_type: str = "real_case_dossier_error") -> None:
        super().__init__(message)
        self.error_type = error_type


@dataclass(frozen=True)
class RealCaseDossierDefinition:
    """Processed-artifact mapping for one engineer-visible case package."""

    case_id: str
    display_name: str
    inventory_case_id: str
    result_file: str
    bus_voltage_csv: str
    channel_csv: Optional[str]
    overload_csv: Optional[str]
    description: str


REAL_CASE_DOSSIERS: Dict[str, RealCaseDossierDefinition] = {
    "pif6_2026_05_17": RealCaseDossierDefinition(
        case_id="pif6_2026_05_17",
        display_name="PIF6 POC2 786-bus case",
        inventory_case_id="pif6_2026_05_17",
        result_file="pif6_baseline_result.json",
        bus_voltage_csv="pif6_bus_voltages.csv",
        channel_csv="pif6_no_disturbance_channels.csv",
        overload_csv="pif6_overloaded_branches.csv",
        description="Real PIF6 renewable-plant POC2 package with processed static and no-disturbance RMS artifacts.",
    ),
    "test_cases_v36": RealCaseDossierDefinition(
        case_id="test_cases_v36",
        display_name="Small PSS/E PPC test case",
        inventory_case_id="test_cases_v36",
        result_file="test_cases_v36_no_disturbance_result.json",
        bus_voltage_csv="test_cases_v36_bus_voltages.csv",
        channel_csv="test_cases_v36_no_disturbance_channels.csv",
        overload_csv="test_cases_v36_overloaded_branches.csv",
        description="Small PPC benchmark package with processed static and no-disturbance RMS artifacts.",
    ),
}

_CASE_ALIASES = {
    "pif6": "pif6_2026_05_17",
    "pif6_baseline": "pif6_2026_05_17",
    "pif6_2026_05_17": "pif6_2026_05_17",
    "test_cases_v36": "test_cases_v36",
    "test_cases_v36_no_disturbance": "test_cases_v36",
    "small": "test_cases_v36",
}


def inspect_real_case_summary(
    *,
    case_id: str,
    processed_dir: Optional[str] = None,
    max_poc_candidates: int = 8,
) -> Dict[str, Any]:
    """Return a compact case package summary and POC candidates."""

    try:
        root, definition = _resolve_case(case_id, processed_dir)
        inventory = _load_inventory(root, definition)
        summary = _inventory_summary(root, definition)
        result = _load_result(root, definition)
        poc_candidates = _poc_candidate_preview(
            inventory["poc_candidates"],
            max_rows=_positive_int(max_poc_candidates, "max_poc_candidates"),
        )
        m1 = _mapping(result.get("m1_powerflow"))
        m2 = _mapping(result.get("m2_dynamic"))
        return {
            "ok": True,
            "tool": "inspect_real_case_summary",
            "schema_version": REAL_CASE_DOSSIER_SCHEMA_VERSION,
            "backend": "processed_psse_case_dossier",
            "case_id": definition.case_id,
            "display_name": definition.display_name,
            "description": definition.description,
            "processed_dir": str(root),
            "inventory_counts": summary,
            "case_size": {
                "bus_count": _optional_int(m1.get("bus_count")),
                "branch_count": _optional_int(m1.get("branch_count")),
                "machine_count": _optional_int(m1.get("machine_count")),
                "load_count": _optional_int(m1.get("load_count")),
            },
            "available_remote_scenarios": list(
                REMOTE_PSSE_M1M2_CASES.get(definition.case_id, _empty_remote()).allowed_scenarios
            ),
            "processed_artifacts": _artifact_status(root, definition),
            "poc_candidates": poc_candidates,
            "static_snapshot": {
                "converged": _optional_bool(m1.get("converged")),
                "bus_voltage_min_pu": _optional_number(m1.get("bus_voltage_min_pu")),
                "bus_voltage_max_pu": _optional_number(m1.get("bus_voltage_max_pu")),
                "poc_p_mw": _optional_number(m1.get("poc_p_mw")),
                "poc_q_mvar": _optional_number(m1.get("poc_q_mvar")),
                "overloaded_branch_count": _optional_int(m1.get("overloaded_branch_count")),
            },
            "dynamic_snapshot": {
                "status": _optional_str(m2.get("status")),
                "event": _optional_str(m2.get("event")),
                "initialized": _optional_bool(m2.get("initialized")),
                "simulation_converged": _optional_bool(m2.get("simulation_converged")),
            },
            "data_quality_warnings": _data_quality_warnings(result, inventory),
            "limitations": _limitations(result),
        }
    except RealCaseDossierError as exc:
        return _error_result("inspect_real_case_summary", case_id, exc)


def inspect_real_poc_context(
    *,
    case_id: str,
    poc_label_or_bus: Optional[Any] = None,
    processed_dir: Optional[str] = None,
    max_candidates: int = 12,
    max_branches: int = 12,
) -> Dict[str, Any]:
    """Return candidate POC buses/branches and nearby controlled machines."""

    try:
        root, definition = _resolve_case(case_id, processed_dir)
        inventory = _load_inventory(root, definition)
        selector = "" if poc_label_or_bus is None else str(poc_label_or_bus).strip()
        bus_candidates = _filter_poc_buses(
            inventory["poc_candidates"],
            selector=selector,
            max_rows=_positive_int(max_candidates, "max_candidates"),
        )
        selected_bus = _selected_bus(bus_candidates)
        branch_candidates = _filter_poc_branches(
            inventory["poc_candidates"],
            selected_bus=selected_bus,
            max_rows=_positive_int(max_branches, "max_branches"),
        )
        machines = _machines_for_poc(
            inventory["machines"],
            selected_bus=selected_bus,
            max_rows=8,
        )
        return {
            "ok": True,
            "tool": "inspect_real_poc_context",
            "schema_version": REAL_CASE_DOSSIER_SCHEMA_VERSION,
            "backend": "processed_psse_case_dossier",
            "case_id": definition.case_id,
            "query": {"poc_label_or_bus": selector or None},
            "selected_poc": bus_candidates[0] if bus_candidates else None,
            "candidate_count": len(bus_candidates),
            "ambiguous": len(bus_candidates) > 1,
            "candidate_buses": bus_candidates,
            "candidate_branches": branch_candidates,
            "controlled_or_nearby_machines": machines,
            "engineering_note": (
                "Treat similarly named buses as candidates until a metering branch or project documentation confirms the POC."
                if len(bus_candidates) > 1
                else "POC candidate selection is based on processed names/known POC labels."
            ),
            "limitations": [
                "processed_inventory_only_no_live_case_mutation",
                "poc_labels_not_yet_expert_validated",
            ],
        }
    except RealCaseDossierError as exc:
        return _error_result("inspect_real_poc_context", case_id, exc)


def inspect_real_network_neighborhood(
    *,
    case_id: str,
    bus: int,
    depth: int = 1,
    max_rows: int = 20,
    processed_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a bounded graph neighborhood around one bus."""

    try:
        root, definition = _resolve_case(case_id, processed_dir)
        inventory = _load_inventory(root, definition)
        bus_id = _required_int(bus, "bus")
        depth_value = min(max(_positive_int(depth, "depth"), 1), 3)
        max_value = _positive_int(max_rows, "max_rows")
        edges = _network_edges(inventory)
        visited = _bfs_buses(edges, start=bus_id, depth=depth_value)
        voltages = _read_csv_dicts(root / definition.bus_voltage_csv)
        bus_rows = _bus_rows(voltages, visited, max_rows=max_value)
        branches = _rows_touching_bus_set(inventory["branches"], visited, max_rows=max_value)
        transformers_2w = _rows_touching_bus_set(inventory["transformers_2w"], visited, max_rows=max_value)
        transformers_3w = _three_winding_touching_bus_set(inventory["transformers_3w"], visited, max_rows=max_value)
        return {
            "ok": True,
            "tool": "inspect_real_network_neighborhood",
            "schema_version": REAL_CASE_DOSSIER_SCHEMA_VERSION,
            "backend": "processed_psse_case_dossier",
            "case_id": definition.case_id,
            "bus": bus_id,
            "depth": depth_value,
            "visited_bus_count": len(visited),
            "buses": bus_rows,
            "branches": branches,
            "transformers_2w": transformers_2w,
            "transformers_3w": transformers_3w,
            "top_loading_percent": _max_loading(branches + transformers_2w),
            "limitations": ["processed_topology_only_no_contingency_recalculation"],
        }
    except RealCaseDossierError as exc:
        return _error_result("inspect_real_network_neighborhood", case_id, exc)


def inspect_real_model_inventory(
    *,
    case_id: str,
    model_scope: str = "summary",
    processed_dir: Optional[str] = None,
    max_rows: int = 20,
) -> Dict[str, Any]:
    """Return compact machine and dynamic-model inventory information."""

    try:
        root, definition = _resolve_case(case_id, processed_dir)
        del root
        inventory = _load_inventory(resolve_processed_dir(processed_dir), definition)
        scope = str(model_scope or "summary").strip().lower()
        max_value = _positive_int(max_rows, "max_rows")
        dynamic_models = inventory["dynamic_models"]
        machines = inventory["machines"]
        categories = Counter(str(row.get("model_category") or "unknown") for row in dynamic_models)
        model_names = Counter(str(row.get("model_name") or "unknown") for row in dynamic_models)
        if scope in {"poc", "poc_models"}:
            poc_buses = {
                _optional_int(row.get("bus"))
                for row in inventory["poc_candidates"]
                if str(row.get("candidate_type", "")).lower() == "bus"
            }
            poc_buses.discard(None)
            selected_models = [
                row for row in dynamic_models if _optional_int(row.get("bus")) in poc_buses
            ]
            selected_machines = [
                row
                for row in machines
                if _optional_int(row.get("bus")) in poc_buses
                or _optional_int(row.get("regulated_bus")) in poc_buses
            ]
        elif scope in {"machines", "machine"}:
            selected_models = []
            selected_machines = machines
        elif scope in {"controllers", "dynamic", "all"}:
            selected_models = dynamic_models
            selected_machines = machines if scope == "all" else []
        else:
            selected_models = dynamic_models[:max_value]
            selected_machines = machines[:max_value]
        return {
            "ok": True,
            "tool": "inspect_real_model_inventory",
            "schema_version": REAL_CASE_DOSSIER_SCHEMA_VERSION,
            "backend": "processed_psse_case_dossier",
            "case_id": definition.case_id,
            "model_scope": scope,
            "counts": {
                "machines": len(machines),
                "dynamic_models": len(dynamic_models),
                "model_categories": dict(sorted(categories.items())),
                "top_model_names": dict(model_names.most_common(12)),
            },
            "machines": [_machine_view(row) for row in selected_machines[:max_value]],
            "dynamic_models": [_dynamic_model_view(row) for row in selected_models[:max_value]],
            "limitations": ["model_inventory_only_no_parameter_validation"],
        }
    except RealCaseDossierError as exc:
        return _error_result("inspect_real_model_inventory", case_id, exc)


def inspect_real_static_operating_point(
    *,
    case_id: str,
    processed_dir: Optional[str] = None,
    max_voltage_rows: int = 6,
    max_overload_rows: int = 6,
) -> Dict[str, Any]:
    """Return compact solved static operating-point evidence."""

    try:
        root, definition = _resolve_case(case_id, processed_dir)
        result = _load_result(root, definition)
        m1 = _mapping(result.get("m1_powerflow"))
        voltages = _read_csv_dicts(root / definition.bus_voltage_csv)
        overloads = (
            _read_csv_dicts(root / definition.overload_csv)
            if definition.overload_csv
            else []
        )
        voltage_rows = [_voltage_view(row) for row in voltages]
        voltage_values = [
            row["voltage_pu"]
            for row in voltage_rows
            if row.get("voltage_pu") is not None
        ]
        low = sorted(voltage_rows, key=lambda row: row.get("voltage_pu") if row.get("voltage_pu") is not None else 999)
        high = sorted(
            voltage_rows,
            key=lambda row: row.get("voltage_pu") if row.get("voltage_pu") is not None else -999,
            reverse=True,
        )
        return {
            "ok": True,
            "tool": "inspect_real_static_operating_point",
            "schema_version": REAL_CASE_DOSSIER_SCHEMA_VERSION,
            "backend": "processed_psse_case_dossier",
            "case_id": definition.case_id,
            "converged": _optional_bool(m1.get("converged")),
            "voltage_range_pu": {
                "min": min(voltage_values) if voltage_values else _optional_number(m1.get("bus_voltage_min_pu")),
                "max": max(voltage_values) if voltage_values else _optional_number(m1.get("bus_voltage_max_pu")),
            },
            "poc": {
                "p_mw": _optional_number(m1.get("poc_p_mw")),
                "q_mvar": _optional_number(m1.get("poc_q_mvar")),
            },
            "counts": {
                "buses": _optional_int(m1.get("bus_count")) or len(voltages),
                "branches": _optional_int(m1.get("branch_count")),
                "machines": _optional_int(m1.get("machine_count")),
                "loads": _optional_int(m1.get("load_count")),
                "overloaded_branches": _optional_int(m1.get("overloaded_branch_count")) or len(overloads),
            },
            "lowest_voltage_buses": low[: _positive_int(max_voltage_rows, "max_voltage_rows")],
            "highest_voltage_buses": high[: _positive_int(max_voltage_rows, "max_voltage_rows")],
            "overloaded_or_monitored_branches": [
                _branch_view(row) for row in overloads[: _positive_int(max_overload_rows, "max_overload_rows")]
            ],
            "limitations": ["processed_static_snapshot_only_no_new_load_flow"],
        }
    except RealCaseDossierError as exc:
        return _error_result("inspect_real_static_operating_point", case_id, exc)


def inspect_real_dynamic_channels(
    *,
    case_id: str,
    scenario_type: str = "no_disturbance_5s",
    processed_dir: Optional[str] = None,
    max_samples: int = 5,
) -> Dict[str, Any]:
    """Return compact dynamic-channel evidence for processed no-disturbance runs."""

    try:
        root, definition = _resolve_case(case_id, processed_dir)
        scenario = str(scenario_type or "no_disturbance_5s").strip().lower()
        if scenario not in {"no_disturbance_5s", "no_disturbance", "baseline"}:
            raise RealCaseDossierError(
                f"Dynamic channel dossier supports only no_disturbance_5s, not {scenario_type!r}.",
                error_type="unsupported_dynamic_channel_scenario",
            )
        if not definition.channel_csv:
            raise RealCaseDossierError(
                f"No processed channel CSV is configured for {definition.case_id}.",
                error_type="dynamic_channels_unavailable",
            )
        rows = _read_csv_dicts(root / definition.channel_csv)
        if not rows:
            raise RealCaseDossierError(
                f"Processed channel CSV is empty for {definition.case_id}.",
                error_type="dynamic_channels_empty",
            )
        columns = list(rows[0])
        final = rows[-1]
        voltage_cols = [col for col in columns if col.lower().endswith("_v") or "voltage" in col.lower()]
        freq_cols = [col for col in columns if "freq" in col.lower()]
        p_cols = [col for col in columns if "poc_p" in col.lower()]
        q_cols = [col for col in columns if "poc_q" in col.lower()]
        sample_count = _positive_int(max_samples, "max_samples")
        return {
            "ok": True,
            "tool": "inspect_real_dynamic_channels",
            "schema_version": REAL_CASE_DOSSIER_SCHEMA_VERSION,
            "backend": "processed_psse_case_dossier",
            "case_id": definition.case_id,
            "scenario_type": "no_disturbance_5s",
            "row_count": len(rows),
            "channel_names": columns,
            "final_time_s": _optional_number(final.get("time") or final.get("Time(s)")),
            "final_values": _selected_final_values(final, p_cols + q_cols + voltage_cols[:3] + freq_cols[:2]),
            "voltage_extrema": {
                col: _column_min_max(rows, col) for col in voltage_cols[:6]
            },
            "frequency_extrema": {
                col: _column_min_max(rows, col) for col in freq_cols[:3]
            },
            "sample_rows": rows[:sample_count],
            "limitations": ["processed_no_disturbance_channels_only_not_fault_or_emt"],
        }
    except RealCaseDossierError as exc:
        return _error_result("inspect_real_dynamic_channels", case_id, exc)


def _empty_remote() -> Any:
    return type("_EmptyRemote", (), {"allowed_scenarios": ()})()


def _resolve_case(
    case_id: str,
    processed_dir: Optional[str],
) -> Tuple[Path, RealCaseDossierDefinition]:
    normalized = _CASE_ALIASES.get(str(case_id or "").strip(), str(case_id or "").strip())
    if normalized not in REAL_CASE_DOSSIERS:
        allowed = ", ".join(sorted(REAL_CASE_DOSSIERS))
        raise RealCaseDossierError(
            f"Unknown real case_id {case_id!r}. Allowed: {allowed}.",
            error_type="unknown_real_case_id",
        )
    return resolve_processed_dir(processed_dir), REAL_CASE_DOSSIERS[normalized]


def _load_inventory(root: Path, definition: RealCaseDossierDefinition) -> Dict[str, List[Dict[str, str]]]:
    case_dir = root / "inventories" / definition.inventory_case_id
    if not case_dir.exists():
        raise RealCaseDossierError(
            f"Inventory directory is missing: {case_dir}",
            error_type="inventory_unavailable",
        )
    return {
        name: _read_csv_dicts(case_dir / f"{name}.csv")
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
        )
    }


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise RealCaseDossierError(
            f"Required processed CSV is missing: {path}",
            error_type="processed_csv_missing",
        )
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise RealCaseDossierError(
            f"Cannot read processed CSV {path}: {exc}",
            error_type="processed_csv_unavailable",
        ) from exc


def _load_result(root: Path, definition: RealCaseDossierDefinition) -> Dict[str, Any]:
    path = root / definition.result_file
    if not path.exists():
        raise RealCaseDossierError(
            f"Required processed result JSON is missing: {path}",
            error_type="processed_result_missing",
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealCaseDossierError(
            f"Cannot read processed result JSON {path}: {exc}",
            error_type="processed_result_invalid",
        ) from exc
    if not isinstance(data, Mapping):
        raise RealCaseDossierError(
            f"Processed result JSON must be an object: {path}",
            error_type="processed_result_invalid",
        )
    return dict(data)


def _inventory_summary(root: Path, definition: RealCaseDossierDefinition) -> Dict[str, Any]:
    summary_path = root / "inventories" / "inventory_summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            for item in payload.get("cases", []):
                if isinstance(item, Mapping) and item.get("case_id") == definition.inventory_case_id:
                    return {
                        key: item.get(key)
                        for key in (
                            "branches",
                            "transformers_2w",
                            "transformers_3w",
                            "machines",
                            "loads",
                            "shunts",
                            "switched_shunts",
                            "dynamic_models",
                            "poc_candidates",
                        )
                    }
        except (OSError, json.JSONDecodeError):
            pass
    inventory = _load_inventory(root, definition)
    return {name: len(rows) for name, rows in inventory.items()}


def _artifact_status(root: Path, definition: RealCaseDossierDefinition) -> Dict[str, Any]:
    files = {
        "result": definition.result_file,
        "bus_voltages": definition.bus_voltage_csv,
        "channels": definition.channel_csv,
        "overloads": definition.overload_csv,
        "inventory": f"inventories/{definition.inventory_case_id}",
    }
    return {
        key: {"path": str(root / value), "exists": (root / value).exists()}
        for key, value in files.items()
        if value
    }


def _poc_candidate_preview(rows: Sequence[Mapping[str, Any]], *, max_rows: int) -> List[Dict[str, Any]]:
    return [_poc_view(row) for row in rows[:max_rows]]


def _filter_poc_buses(
    rows: Sequence[Mapping[str, Any]],
    *,
    selector: str,
    max_rows: int,
) -> List[Dict[str, Any]]:
    selected = []
    selector_lower = selector.lower()
    selector_bus = _optional_int(selector)
    for row in rows:
        if str(row.get("candidate_type", "")).strip().lower() != "bus":
            continue
        bus = _optional_int(row.get("bus"))
        name = str(row.get("name") or "")
        if selector:
            if selector_bus is not None and bus != selector_bus:
                continue
            if selector_bus is None and selector_lower not in name.lower():
                continue
        selected.append(_poc_view(row))
    if not selected and selector:
        for row in rows:
            if str(row.get("candidate_type", "")).strip().lower() == "bus":
                selected.append(_poc_view(row))
    return selected[:max_rows]


def _filter_poc_branches(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_bus: Optional[int],
    max_rows: int,
) -> List[Dict[str, Any]]:
    selected = []
    for row in rows:
        if str(row.get("candidate_type", "")).strip().lower() != "branch":
            continue
        if selected_bus is not None and selected_bus not in {
            _optional_int(row.get("from_bus")),
            _optional_int(row.get("to_bus")),
            _optional_int(row.get("bus")),
        }:
            continue
        selected.append(_poc_branch_view(row))
    return selected[:max_rows]


def _selected_bus(candidates: Sequence[Mapping[str, Any]]) -> Optional[int]:
    if not candidates:
        return None
    return _optional_int(candidates[0].get("bus"))


def _machines_for_poc(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_bus: Optional[int],
    max_rows: int,
) -> List[Dict[str, Any]]:
    if selected_bus is None:
        return []
    selected = [
        row
        for row in rows
        if _optional_int(row.get("bus")) == selected_bus
        or _optional_int(row.get("regulated_bus")) == selected_bus
    ]
    return [_machine_view(row) for row in selected[:max_rows]]


def _network_edges(inventory: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[int, set[int]]:
    edges: Dict[int, set[int]] = {}
    for row in inventory["branches"]:
        _add_edge(edges, _optional_int(row.get("from_bus")), _optional_int(row.get("to_bus")))
    for row in inventory["transformers_2w"]:
        _add_edge(edges, _optional_int(row.get("from_bus")), _optional_int(row.get("to_bus")))
    for row in inventory["transformers_3w"]:
        buses = [_optional_int(row.get("bus1")), _optional_int(row.get("bus2")), _optional_int(row.get("bus3"))]
        buses = [bus for bus in buses if bus is not None]
        for left in buses:
            for right in buses:
                if left != right:
                    _add_edge(edges, left, right)
    return edges


def _add_edge(edges: Dict[int, set[int]], left: Optional[int], right: Optional[int]) -> None:
    if left is None or right is None:
        return
    edges.setdefault(left, set()).add(right)
    edges.setdefault(right, set()).add(left)


def _bfs_buses(edges: Mapping[int, set[int]], *, start: int, depth: int) -> set[int]:
    visited = {start}
    queue: deque[Tuple[int, int]] = deque([(start, 0)])
    while queue:
        bus, distance = queue.popleft()
        if distance >= depth:
            continue
        for next_bus in sorted(edges.get(bus, ())):
            if next_bus in visited:
                continue
            visited.add(next_bus)
            queue.append((next_bus, distance + 1))
    return visited


def _bus_rows(rows: Sequence[Mapping[str, Any]], buses: set[int], *, max_rows: int) -> List[Dict[str, Any]]:
    selected = [
        _voltage_view(row)
        for row in rows
        if _optional_int(row.get("bus")) in buses
    ]
    return selected[:max_rows]


def _rows_touching_bus_set(
    rows: Sequence[Mapping[str, Any]],
    buses: set[int],
    *,
    max_rows: int,
) -> List[Dict[str, Any]]:
    selected = []
    for row in rows:
        if _optional_int(row.get("from_bus")) in buses or _optional_int(row.get("to_bus")) in buses:
            selected.append(_branch_view(row))
    return selected[:max_rows]


def _three_winding_touching_bus_set(
    rows: Sequence[Mapping[str, Any]],
    buses: set[int],
    *,
    max_rows: int,
) -> List[Dict[str, Any]]:
    selected = []
    for row in rows:
        row_buses = {_optional_int(row.get("bus1")), _optional_int(row.get("bus2")), _optional_int(row.get("bus3"))}
        if any(bus in buses for bus in row_buses if bus is not None):
            selected.append(
                {
                    "bus1": _optional_int(row.get("bus1")),
                    "bus2": _optional_int(row.get("bus2")),
                    "bus3": _optional_int(row.get("bus3")),
                    "id": _optional_str(row.get("id")),
                    "status": _optional_int(row.get("status")),
                    "name": _optional_str(row.get("name")),
                    "flow_source": _optional_str(row.get("flow_source")),
                }
            )
    return selected[:max_rows]


def _max_loading(rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    values = [
        _optional_number(row.get("loading_percent"))
        for row in rows
        if _optional_number(row.get("loading_percent")) is not None
    ]
    return max(values) if values else None


def _column_min_max(rows: Sequence[Mapping[str, Any]], column: str) -> Dict[str, Optional[float]]:
    values = [
        _optional_number(row.get(column))
        for row in rows
        if _optional_number(row.get(column)) is not None
    ]
    return {"min": min(values), "max": max(values)} if values else {"min": None, "max": None}


def _selected_final_values(row: Mapping[str, Any], columns: Sequence[str]) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {}
    for column in columns:
        if column in result:
            continue
        result[column] = _optional_number(row.get(column))
    return result


def _poc_view(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_type": _optional_str(row.get("candidate_type")),
        "bus": _optional_int(row.get("bus")),
        "name": _optional_str(row.get("name")),
        "voltage_pu": _optional_number(row.get("voltage_pu")),
        "from_bus": _optional_int(row.get("from_bus")),
        "to_bus": _optional_int(row.get("to_bus")),
        "circuit_id": _optional_str(row.get("circuit_id")),
        "p_mw": _optional_number(row.get("P_MW")),
        "q_mvar": _optional_number(row.get("Q_Mvar")),
        "reason": _optional_str(row.get("reason")),
    }


def _poc_branch_view(row: Mapping[str, Any]) -> Dict[str, Any]:
    view = _poc_view(row)
    return {
        key: view[key]
        for key in ("bus", "name", "voltage_pu", "from_bus", "to_bus", "circuit_id", "p_mw", "q_mvar", "reason")
    }


def _branch_view(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "from_bus": _optional_int(row.get("from_bus")),
        "to_bus": _optional_int(row.get("to_bus")),
        "circuit_id": _optional_str(row.get("circuit_id")),
        "status": _optional_int(row.get("status")),
        "p_mw": _optional_number(row.get("P_MW")),
        "q_mvar": _optional_number(row.get("Q_Mvar")),
        "loading_percent": _optional_number(row.get("loading_percent")),
        "rate_a": _optional_number(row.get("RATEA")),
        "flow_source": _optional_str(row.get("flow_source")),
        "name": _optional_str(row.get("name")),
    }


def _machine_view(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "bus": _optional_int(row.get("bus")),
        "id": _optional_str(row.get("id")),
        "status": _optional_int(row.get("status")),
        "pgen_mw": _optional_number(row.get("Pgen")),
        "qgen_mvar": _optional_number(row.get("Qgen")),
        "pmax_mw": _optional_number(row.get("Pmax")),
        "pmin_mw": _optional_number(row.get("Pmin")),
        "qmax_mvar": _optional_number(row.get("Qmax")),
        "qmin_mvar": _optional_number(row.get("Qmin")),
        "mbase": _optional_number(row.get("MBASE")),
        "voltage_setpoint": _optional_number(row.get("voltage_setpoint")),
        "regulated_bus": _optional_int(row.get("regulated_bus")),
        "wmod": _optional_int(row.get("WMOD")),
        "wpf": _optional_number(row.get("WPF")),
    }


def _dynamic_model_view(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "bus": _optional_int(row.get("bus")),
        "id": _optional_str(row.get("id")),
        "model_name": _optional_str(row.get("model_name")),
        "model_category": _optional_str(row.get("model_category")),
        "controlled_bus_candidates": _optional_str(row.get("connected_controlled_bus_candidates")),
    }


def _voltage_view(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "bus": _optional_int(row.get("bus")),
        "name": _optional_str(row.get("name")),
        "base_kv": _optional_number(row.get("base_kv")),
        "type": _optional_int(row.get("type")),
        "voltage_pu": _optional_number(row.get("voltage_pu")),
        "angle_deg": _optional_number(row.get("angle_deg")),
    }


def _data_quality_warnings(
    result: Mapping[str, Any],
    inventory: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[str]:
    warnings = []
    limitations = _limitations(result)
    if "labels_not_yet_expert_validated" in limitations:
        warnings.append("labels_not_yet_expert_validated")
    transformer_flows = [
        row.get("flow_source")
        for row in list(inventory.get("transformers_2w", ())) + list(inventory.get("transformers_3w", ()))
    ]
    if any(str(item) == "not_available" for item in transformer_flows):
        warnings.append("some_transformer_flows_not_available")
    if not inventory.get("poc_candidates"):
        warnings.append("no_poc_candidates_in_inventory")
    return list(dict.fromkeys(warnings))


def _limitations(result: Mapping[str, Any]) -> List[str]:
    raw = result.get("limitations")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(item) for item in raw]
    return []


def _error_result(tool: str, case_id: str, exc: RealCaseDossierError) -> Dict[str, Any]:
    return {
        "ok": False,
        "tool": tool,
        "schema_version": REAL_CASE_DOSSIER_SCHEMA_VERSION,
        "backend": "processed_psse_case_dossier",
        "case_id": str(case_id or ""),
        "error_type": exc.error_type,
        "message": str(exc),
    }


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "1", "yes"}:
            return True
        if lower in {"false", "0", "no"}:
            return False
    return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _optional_number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _required_int(value: Any, name: str) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise RealCaseDossierError(f"{name} must be an integer.", error_type="invalid_argument")
    return parsed


def _positive_int(value: Any, name: str) -> int:
    parsed = _required_int(value, name)
    if parsed < 1:
        raise RealCaseDossierError(f"{name} must be positive.", error_type="invalid_argument")
    return parsed

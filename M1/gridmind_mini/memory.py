"""Persistent study memory for the Mini Grid-Mind reproduction.

Step 7 implements the paper's append-only memory layer for completed CIA and
capacity-search studies. The store keeps structured JSONL records for machine
retrieval and regenerates a small Markdown ledger for human audit.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


MEMORY_CONTEXT_CAVEAT = (
    "Memory entries below come from earlier simulations recorded by this local "
    "Mini Grid-Mind store. Treat them as supplementary session context, not as "
    "independent historical studies, and prefer fresh simulations for new "
    "quantitative questions."
)


@dataclass(frozen=True)
class StudyMemoryRecord:
    """Compact structured memory entry for one completed study."""

    record_id: str
    timestamp_utc: str
    tool: str
    case_path: str
    bus: Optional[int]
    connection_type: Optional[str]
    mw: Optional[float]
    status: str
    summary: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp_utc": self.timestamp_utc,
            "tool": self.tool,
            "case_path": self.case_path,
            "bus": self.bus,
            "connection_type": self.connection_type,
            "mw": self.mw,
            "status": self.status,
            "summary": self.summary,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StudyMemoryRecord":
        data = payload.get("data", {})
        if not isinstance(data, Mapping):
            raise ValueError("data must be a JSON object")
        return cls(
            record_id=_required_str(payload, "record_id"),
            timestamp_utc=_required_str(payload, "timestamp_utc"),
            tool=_required_str(payload, "tool"),
            case_path=_required_str(payload, "case_path"),
            bus=_optional_int(payload.get("bus")),
            connection_type=_optional_str(payload.get("connection_type")),
            mw=_optional_float(payload.get("mw")),
            status=_required_str(payload, "status"),
            summary=_required_str(payload, "summary"),
            data=dict(data),
        )

    def to_reference(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp_utc": self.timestamp_utc,
            "summary": self.summary,
        }


class StudyMemoryStore:
    """Append-only JSONL study memory with deterministic recall helpers."""

    def __init__(
        self,
        root: str | Path,
        *,
        records_filename: str = "studies.jsonl",
        ledger_filename: str = "ledger.md",
    ) -> None:
        self.root = Path(root)
        self.records_path = self.root / records_filename
        self.ledger_path = self.root / ledger_filename

    def append_tool_result(self, result: Mapping[str, Any]) -> StudyMemoryRecord:
        record = record_from_tool_result(result)
        self.append_record(record)
        return record

    def append_record(self, record: StudyMemoryRecord) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), allow_nan=False, sort_keys=True) + "\n")
        self.regenerate_ledger()

    def load_records(self) -> List[StudyMemoryRecord]:
        if not self.records_path.exists():
            return []

        records: List[StudyMemoryRecord] = []
        with self.records_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(StudyMemoryRecord.from_dict(json.loads(stripped)))
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid memory record at {self.records_path}:{line_number}: {exc}"
                    ) from exc
        return records

    def recent(self, limit: int = 5) -> List[StudyMemoryRecord]:
        limit = _limit_argument(limit)
        return _limit_records(_sort_newest(self.load_records()), limit)

    def recall_bus(
        self,
        *,
        case_path: str,
        bus: int,
        limit: int = 5,
    ) -> List[StudyMemoryRecord]:
        case_path = _case_path_argument(case_path)
        bus = _integer_argument("bus", bus)
        limit = _limit_argument(limit)
        case_key = _case_key(case_path)
        matches = [
            record
            for record in self.load_records()
            if _case_key(record.case_path) == case_key and record.bus == bus
        ]
        return _limit_records(_sort_newest(matches), limit)

    def recall_case(self, *, case_path: str, limit: int = 10) -> List[StudyMemoryRecord]:
        case_path = _case_path_argument(case_path)
        limit = _limit_argument(limit)
        case_key = _case_key(case_path)
        matches = [
            record for record in self.load_records() if _case_key(record.case_path) == case_key
        ]
        return _limit_records(_sort_newest(matches), limit)

    def search(
        self,
        query: str,
        *,
        case_path: Optional[str] = None,
        limit: int = 10,
    ) -> List[StudyMemoryRecord]:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if case_path is not None:
            case_path = _case_path_argument(case_path)
        limit = _limit_argument(limit)
        terms = [term for term in query.lower().split() if term]
        if not terms:
            return []

        case_key = _case_key(case_path) if case_path is not None else None
        matches = []
        for record in self.load_records():
            if case_key is not None and _case_key(record.case_path) != case_key:
                continue
            haystack = _record_haystack(record)
            if all(term in haystack for term in terms):
                matches.append(record)
        return _limit_records(_sort_newest(matches), limit)

    def recall_max_capacity(
        self,
        *,
        case_path: Optional[str] = None,
        bus: Optional[int] = None,
        connection_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[StudyMemoryRecord]:
        if case_path is not None:
            case_path = _case_path_argument(case_path)
        if bus is not None:
            bus = _integer_argument("bus", bus)
        type_key = _optional_filter_str("connection_type", connection_type)
        limit = _limit_argument(limit)
        case_key = _case_key(case_path) if case_path is not None else None
        matches = []
        for record in self.load_records():
            if record.tool != "find_max_capacity":
                continue
            if case_key is not None and _case_key(record.case_path) != case_key:
                continue
            if bus is not None and record.bus != bus:
                continue
            if type_key is not None and record.connection_type != type_key:
                continue
            matches.append(record)
        return _limit_records(_sort_newest(matches), limit)

    def build_prompt_context(self, records: Iterable[StudyMemoryRecord]) -> str:
        selected = list(records)
        if not selected:
            return ""

        lines = [MEMORY_CONTEXT_CAVEAT, "", "Relevant memory entries:"]
        for record in selected:
            lines.append(
                "- "
                + f"[{record.record_id}] {record.timestamp_utc} | "
                + f"{record.tool} | {record.summary}"
            )
        return "\n".join(lines)

    def regenerate_ledger(self) -> None:
        records = _sort_newest(self.load_records())
        self.root.mkdir(parents=True, exist_ok=True)
        generated_at = _now_utc()
        lines = [
            "# Mini Grid-Mind Study Ledger",
            "",
            f"Generated UTC: {generated_at}",
            "",
            MEMORY_CONTEXT_CAVEAT,
            "",
            "| Time UTC | Tool | Case | Bus | Type | MW | Status | Summary |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for record in records:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(record.timestamp_utc),
                        _cell(record.tool),
                        _cell(record.case_path),
                        _cell("" if record.bus is None else str(record.bus)),
                        _cell(record.connection_type or ""),
                        _cell(_format_mw(record.mw)),
                        _cell(record.status),
                        _cell(record.summary),
                    ]
                )
                + " |"
            )
        self.ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def record_from_tool_result(result: Mapping[str, Any]) -> StudyMemoryRecord:
    if not result.get("ok", False):
        raise ValueError("Only successful tool results can be stored in memory")

    tool = str(result.get("tool", ""))
    if tool == "run_cia":
        return _record_from_cia(result)
    if tool == "find_max_capacity":
        return _record_from_capacity(result)
    raise ValueError(f"Tool result '{tool}' is not supported by study memory")


def _record_from_cia(result: Mapping[str, Any]) -> StudyMemoryRecord:
    raw_connection = result.get("connection", {})
    if not isinstance(raw_connection, Mapping):
        raise ValueError("run_cia result connection must be an object")
    connection = dict(raw_connection)
    case_path = _required_str(result, "case_path")
    bus = _optional_int(connection.get("bus"))
    connection_type = _optional_str(connection.get("connection_type"))
    if connection_type is not None:
        connection_type = connection_type.strip().lower()
    mw = _optional_float(connection.get("p_mw"))
    status = str(result.get("recommendation", "unknown"))
    reason_codes = list(result.get("reason_codes", []))
    raw_summary = result.get("summary", {})
    if not isinstance(raw_summary, Mapping):
        raise ValueError("run_cia result summary must be an object")
    summary = dict(raw_summary)
    stage_statuses = _stage_statuses(result.get("stage_reports", []))
    text = (
        f"CIA {status} for {_format_mw(mw)} MW {connection_type or 'connection'} "
        f"at bus {bus if bus is not None else 'unknown'} on {case_path}; "
        f"project hard={summary.get('project_hard_violations', 'unknown')}, "
        f"borderline={summary.get('project_borderline_violations', 'unknown')}; "
        f"reasons={', '.join(str(code) for code in reason_codes) or 'none'}."
    )
    return StudyMemoryRecord(
        record_id=_new_record_id(),
        timestamp_utc=_now_utc(),
        tool="run_cia",
        case_path=case_path,
        bus=bus,
        connection_type=connection_type,
        mw=mw,
        status=status,
        summary=text,
        data={
            "recommendation": status,
            "complete": bool(result.get("complete", False)),
            "reason_codes": reason_codes,
            "summary": summary,
            "stage_statuses": stage_statuses,
            "limiting_issues": _limiting_issues_from_cia(result.get("stage_reports", [])),
        },
    )


def _record_from_capacity(result: Mapping[str, Any]) -> StudyMemoryRecord:
    raw_request = result.get("request", {})
    if not isinstance(raw_request, Mapping):
        raise ValueError("find_max_capacity result request must be an object")
    request = dict(raw_request)
    case_path = _required_str(result, "case_path")
    bus = _optional_int(request.get("bus"))
    connection_type = _optional_str(request.get("connection_type"))
    if connection_type is not None:
        connection_type = connection_type.strip().lower()
    max_approved_mw = _optional_float(result.get("max_approved_mw"))
    status = str(result.get("status", "unknown"))
    rejection = result.get("rejection_explanation")
    limiting_stage = None
    if isinstance(rejection, Mapping):
        limiting_stage = rejection.get("limiting_stage")
    text = (
        f"Capacity search {status} for {connection_type or 'connection'} "
        f"at bus {bus if bus is not None else 'unknown'} on {case_path}; "
        f"max approved={_format_mw(max_approved_mw)} MW"
    )
    if limiting_stage:
        text += f"; limiting stage={limiting_stage}"
    text += "."
    samples = result.get("samples", {})
    total_samples = samples.get("total_items") if isinstance(samples, Mapping) else None
    return StudyMemoryRecord(
        record_id=_new_record_id(),
        timestamp_utc=_now_utc(),
        tool="find_max_capacity",
        case_path=case_path,
        bus=bus,
        connection_type=connection_type,
        mw=max_approved_mw,
        status=status,
        summary=text,
        data={
            "capacity_status": status,
            "max_approved_mw": max_approved_mw,
            "lower_bound_mw": result.get("lower_bound_mw"),
            "upper_bound_mw": result.get("upper_bound_mw"),
            "tolerance_mw": result.get("tolerance_mw"),
            "iterations": result.get("iterations"),
            "sample_count": total_samples,
            "boundary_samples": result.get("boundary_samples"),
            "rejection_explanation": rejection,
            "diagnostics": result.get("diagnostics"),
        },
    )


def _limiting_issues_from_cia(stage_reports: Any) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    if not isinstance(stage_reports, list):
        return issues
    for stage in stage_reports:
        if not isinstance(stage, Mapping):
            continue
        if stage.get("status") not in {"fail", "borderline", "not_run", "not_implemented"}:
            continue
        item: Dict[str, Any] = {
            "stage": stage.get("stage"),
            "status": stage.get("status"),
            "reason_codes": list(stage.get("reason_codes", [])),
        }
        if stage.get("stage") == "f1_steady_state":
            comparison = stage.get("project_violation_comparison", {})
            if isinstance(comparison, Mapping):
                item["project_caused_violations"] = comparison.get("project_caused_violations")
        elif stage.get("stage") == "f2_n1_contingency":
            comparison = stage.get("project_contingency_comparison", {})
            if isinstance(comparison, Mapping):
                item["project_caused_failures"] = comparison.get("project_caused_failures")
        issues.append(item)
    return issues


def _stage_statuses(stage_reports: Any) -> List[Dict[str, Any]]:
    statuses = []
    if not isinstance(stage_reports, list):
        return statuses
    for stage in stage_reports:
        if not isinstance(stage, Mapping):
            continue
        statuses.append(
            {
                "stage": stage.get("stage"),
                "status": stage.get("status"),
                "reason_codes": list(stage.get("reason_codes", [])),
            }
        )
    return statuses


def _sort_newest(records: Iterable[StudyMemoryRecord]) -> List[StudyMemoryRecord]:
    return sorted(records, key=lambda record: record.timestamp_utc, reverse=True)


def _limit_records(records: List[StudyMemoryRecord], limit: int) -> List[StudyMemoryRecord]:
    if limit < 0:
        return records
    return records[:limit]


def _record_haystack(record: StudyMemoryRecord) -> str:
    return " ".join(
        [
            record.tool,
            record.case_path,
            record.connection_type or "",
            record.status,
            record.summary,
            json.dumps(record.data, sort_keys=True),
        ]
    ).lower()


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional string field must be a string")
    return value.strip()


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("optional integer field must be an integer")
    return value


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("optional float field must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("optional float field must be finite")
    return result


def _case_path_argument(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("case_path must be a non-empty string")
    return value.strip()


def _integer_argument(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _limit_argument(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("limit must be an integer")
    return value


def _optional_filter_str(name: str, value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string when provided")
    return value.strip().lower()


def _case_key(case_path: str) -> str:
    return case_path.strip().lower()


def _new_record_id() -> str:
    return "mem_" + uuid.uuid4().hex[:12]


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _format_mw(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return f"{value:g}"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")

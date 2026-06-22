#!/usr/bin/env python3
"""Windows-side queued JSON worker for remote PSS/E jobs.

The worker is intentionally narrow:

* health/echo endpoints remain for network smoke tests;
* every PSS/E job runs in a subprocess, never in an HTTP request thread;
* jobs are serialized through a single worker queue;
* case IDs and scenario types are allowlisted;
* PIF6 always uses pssecmd36.exe with ``-buses 50000`` and a generated IDV file;
* original SAV/DYR/DLL files are read-only inputs.

This script uses only the Python standard library in the HTTP layer.  PSS/E,
psspy, and dyntools are imported only by subprocesses or explicit preflight
actions on the Windows host.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import platform
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse


DEFAULT_PORT = 8765
TOKEN_ENV = "PSSE_REMOTE_TOKEN"
DEFAULT_PSSE_CMD = r"C:\Program Files\PTI\PSSE36\36.2\PSSBIN\pssecmd36.exe"
DEFAULT_PSSE_PYVER = "312"
DEFAULT_JOB_TIMEOUT_S = 900.0
REMOTE_RESULT_SCHEMA_VERSION = "psse_remote_result_v1"


class WorkerError(RuntimeError):
    """Structured worker error."""

    def __init__(self, message: str, *, error_type: str = "worker_error") -> None:
        super().__init__(message)
        self.error_type = error_type


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    display_name: str
    folder_parts: Tuple[str, ...]
    sav: str
    dyr: Optional[str]
    dlls: Tuple[str, ...]
    allowed_scenarios: Tuple[str, ...]
    executor: str
    bus_count: int
    branch_count: Optional[int]
    machine_count: Optional[int]
    load_count: Optional[int]
    poc_bus: Optional[int]
    poc_branch: Optional[Tuple[int, int, str]]
    key_voltage_buses: Tuple[int, ...]
    dynamic_channels: Tuple[Tuple[str, Tuple[Any, ...]], ...]
    limitations: Tuple[str, ...] = ()


CASE_DEFINITIONS: Dict[str, CaseDefinition] = {
    "test_cases_v36": CaseDefinition(
        case_id="test_cases_v36",
        display_name="Small PSS/E PPC test case",
        folder_parts=("real-data-new", "test_cases _v36"),
        sav="psse_ppc_test_bench_PMIN_QMAX.sav",
        dyr="psse_ppc_test_dynamic.dyr",
        dlls=("sippc_r5gz_v36_5_2_6_test.dll",),
        allowed_scenarios=("static", "no_disturbance_5s", "pq_target_step"),
        executor="standalone_psspy_subprocess",
        bus_count=11,
        branch_count=10,
        machine_count=5,
        load_count=0,
        poc_bus=900,
        poc_branch=(800, 900, "1"),
        key_voltage_buses=(900, 800, 700, 40, 41, 200, 201),
        dynamic_channels=(),
        limitations=(
            "small_case_uses_standalone_psspy_50_bus_path",
            "psse_rms_dynamic_not_emt_waveform",
        ),
    ),
    "pif6_2026_05_17": CaseDefinition(
        case_id="pif6_2026_05_17",
        display_name="PIF6 POC2 786-bus case",
        folder_parts=("real-data-new", "2026_05_17"),
        sav="PIF6_POC2_KLS_V9_updt.sav",
        dyr="PIF6_POC2_Disaggregated.dyr",
        dlls=(
            "sippc_pif6_v36_5_2_5_5.dll",
            "SG1100UD_PSSE36_V0131101_260420.dll",
            "NWSTAT01_V2_20250829_V36.dll",
        ),
        allowed_scenarios=("static", "no_disturbance_5s"),
        executor="pssecmd36_50000_bus_idv",
        bus_count=786,
        branch_count=790,
        machine_count=251,
        load_count=2,
        poc_bus=2,
        poc_branch=(2001, 2, "1"),
        key_voltage_buses=(2, 2000, 2001, 9001),
        dynamic_channels=(
            ("voltage", (2, "POC2_V")),
            ("voltage", (2000, "POC2_0_V")),
            ("voltage", (2001, "DUMMY_110_V")),
            ("branch_pq", (2001, 2, "1", "POC_P_2001_2", "POC_Q_2001_2")),
            ("machine_array", (2, 2000, "1", "GEN_P_2000")),
            ("machine_array", (3, 2000, "1", "GEN_Q_2000")),
            ("machine_array", (7, 2000, "1", "GEN_SPEED_2000")),
            ("bus_frequency", (2, "FREQ_2")),
        ),
        limitations=(
            "pif6_uses_pssecmd36_50000_bus_idv_path",
            "no_faults_or_unvalidated_ppc_control_changes_in_v1",
            "psse_rms_dynamic_not_emt_waveform",
        ),
    ),
}


@dataclass(frozen=True)
class WorkerConfig:
    repo_root: Path
    output_root: Path
    psse_cmd: Path
    psse_pyver: str
    job_timeout_s: float
    token: str


@dataclass
class JobRecord:
    job_id: str
    request: Dict[str, Any]
    case_id: str
    scenario_type: str
    status: str = "queued"
    created_utc: str = field(default_factory=utc_now)
    started_utc: Optional[str] = None
    ended_utc: Optional[str] = None
    artifact_dir: Optional[str] = None
    result_path: Optional[str] = None
    artifacts_path: Optional[str] = None
    error_type: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "case_id": self.case_id,
            "scenario_type": self.scenario_type,
            "status": self.status,
            "created_utc": self.created_utc,
            "started_utc": self.started_utc,
            "ended_utc": self.ended_utc,
            "artifact_dir": self.artifact_dir,
            "result_path": self.result_path,
            "artifacts_path": self.artifacts_path,
            "error_type": self.error_type,
            "message": self.message,
        }


class JobManager:
    """Single-worker serialized job queue."""

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread = threading.Thread(target=self._worker_loop, name="psse-job-worker", daemon=True)
        self._thread.start()

    def submit(self, payload: Mapping[str, Any]) -> JobRecord:
        case_id, scenario_type = validate_job_request(payload)
        job_id = uuid.uuid4().hex
        record = JobRecord(
            job_id=job_id,
            request=jsonable(dict(payload)),
            case_id=case_id,
            scenario_type=scenario_type,
        )
        with self._lock:
            self._jobs[job_id] = record
        self._queue.put(job_id)
        return record

    def status(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def result(self, job_id: str) -> Mapping[str, Any]:
        record = self.status(job_id)
        if record is None:
            raise WorkerError(f"Unknown job_id: {job_id}", error_type="job_not_found")
        if record.status not in {"completed", "error"}:
            raise WorkerError(f"Job is not finished: {record.status}", error_type="job_not_finished")
        if record.result_path is None:
            raise WorkerError("Job has no result path.", error_type="job_result_missing")
        return load_json(Path(record.result_path))

    def artifacts(self, job_id: str) -> Mapping[str, Any]:
        record = self.status(job_id)
        if record is None:
            raise WorkerError(f"Unknown job_id: {job_id}", error_type="job_not_found")
        if record.artifacts_path is None:
            raise WorkerError("Job has no artifact manifest yet.", error_type="job_artifacts_missing")
        return load_json(Path(record.artifacts_path))

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        record = self.status(job_id)
        if record is None:
            return
        with self._lock:
            record.status = "running"
            record.started_utc = utc_now()
        try:
            result, artifacts = execute_job(record, self.config)
            artifact_dir = Path(str(artifacts["artifact_dir"]))
            result_path = artifact_dir / "result.json"
            artifacts_path = artifact_dir / "artifacts_manifest.json"
            write_json(result_path, result)
            artifacts = write_final_artifact_manifest(artifact_dir, artifacts_path)
            with self._lock:
                record.status = "completed" if result.get("ok") else "error"
                record.ended_utc = utc_now()
                record.artifact_dir = str(artifacts["artifact_dir"])
                record.result_path = str(result_path)
                record.artifacts_path = str(artifacts_path)
                if not result.get("ok"):
                    record.error_type = str(result.get("error_type") or "psse_job_failed")
                    record.message = str(result.get("message") or "PSS/E job failed.")
        except Exception as exc:
            artifact_dir = create_artifact_dir(self.config, record)
            failure = failure_result(record, artifact_dir, exc)
            result_path = artifact_dir / "result.json"
            artifacts_path = artifact_dir / "artifacts_manifest.json"
            write_json(result_path, failure)
            artifacts = write_final_artifact_manifest(artifact_dir, artifacts_path)
            with self._lock:
                record.status = "error"
                record.ended_utc = utc_now()
                record.artifact_dir = str(artifact_dir)
                record.result_path = str(result_path)
                record.artifacts_path = str(artifacts_path)
                record.error_type = getattr(exc, "error_type", type(exc).__name__)
                record.message = str(exc)


def main() -> int:
    default_repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the queued PSS/E remote worker.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind TCP port.")
    parser.add_argument(
        "--token",
        default=os.environ.get(TOKEN_ENV, ""),
        help=f"Shared token. Defaults to ${TOKEN_ENV}. Empty disables auth.",
    )
    parser.add_argument("--repo-root", default=str(default_repo), help="Repo root containing real-data-new.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Derived-output root. Defaults to <repo>/real-data-new/derived_outputs/remote_psse_jobs.",
    )
    parser.add_argument("--psse-cmd", default=DEFAULT_PSSE_CMD, help="pssecmd36.exe path.")
    parser.add_argument("--psse-pyver", default=DEFAULT_PSSE_PYVER, help="PSS/E Python version code.")
    parser.add_argument("--job-timeout", type=float, default=DEFAULT_JOB_TIMEOUT_S, help="Job timeout in seconds.")
    args = parser.parse_args()

    if args.port <= 0 or args.port > 65535:
        raise SystemExit("--port must be in 1..65535")
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else repo_root / "real-data-new" / "derived_outputs" / "remote_psse_jobs"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    config = WorkerConfig(
        repo_root=repo_root,
        output_root=output_root,
        psse_cmd=Path(args.psse_cmd),
        psse_pyver=str(args.psse_pyver),
        job_timeout_s=float(args.job_timeout),
        token=args.token,
    )
    manager = JobManager(config)
    handler = build_handler(config=config, manager=manager)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"PSS/E remote worker listening on http://{args.host}:{args.port}", flush=True)
    print(f"Repo root: {repo_root}", flush=True)
    print(f"Output root: {output_root}", flush=True)
    print(f"PSS/E command: {config.psse_cmd}", flush=True)
    print("Queue: single serialized worker", flush=True)
    print("Auth: enabled via bearer token" if args.token else "Auth: disabled; use only on a private trusted network", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping worker.", flush=True)
    finally:
        server.server_close()
    return 0


def build_handler(*, config: WorkerConfig, manager: JobManager) -> type[BaseHTTPRequestHandler]:
    class PsseRemoteHandler(BaseHTTPRequestHandler):
        server_version = "PsseRemoteWorker/0.2"

        def do_GET(self) -> None:  # noqa: N802 - stdlib hook
            if not self.authorized():
                self.write_json({"ok": False, "error_type": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
                return
            try:
                self.handle_get()
            except WorkerError as exc:
                self.write_json({"ok": False, "error_type": exc.error_type, "message": str(exc)}, status_for_error(exc))
            except Exception as exc:
                self.write_json(
                    {"ok": False, "error_type": type(exc).__name__, "message": str(exc)},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def do_POST(self) -> None:  # noqa: N802 - stdlib hook
            if not self.authorized():
                self.write_json({"ok": False, "error_type": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
                return
            try:
                self.handle_post()
            except WorkerError as exc:
                self.write_json({"ok": False, "error_type": exc.error_type, "message": str(exc)}, status_for_error(exc))
            except ValueError as exc:
                self.write_json({"ok": False, "error_type": "invalid_json", "message": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self.write_json(
                    {"ok": False, "error_type": type(exc).__name__, "message": str(exc)},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def handle_get(self) -> None:
            parts = self.path_parts
            if parts == ["health"]:
                self.write_json(
                    {
                        "ok": True,
                        "service": "psse_remote_worker",
                        "version": "0.2",
                        "time_utc": utc_now(),
                        "host": socket.gethostname(),
                        "user": getpass.getuser(),
                        "platform": platform.platform(),
                        "python": sys.version,
                        "cwd": os.getcwd(),
                        "repo_root": str(config.repo_root),
                        "output_root": str(config.output_root),
                        "psse_cmd": str(config.psse_cmd),
                        "queue_policy": "single_serial_worker",
                        "allowlisted_cases": {
                            case_id: {
                                "display_name": case.display_name,
                                "allowed_scenarios": list(case.allowed_scenarios),
                                "executor": case.executor,
                            }
                            for case_id, case in CASE_DEFINITIONS.items()
                        },
                    }
                )
                return
            if len(parts) == 2 and parts[0] == "jobs":
                record = manager.status(parts[1])
                if record is None:
                    raise WorkerError(f"Unknown job_id: {parts[1]}", error_type="job_not_found")
                self.write_json({"ok": True, "job": record.to_dict()})
                return
            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "result":
                self.write_json(dict(manager.result(parts[1])))
                return
            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "artifacts":
                self.write_json(dict(manager.artifacts(parts[1])))
                return
            self.write_json({"ok": False, "error_type": "not_found"}, HTTPStatus.NOT_FOUND)

        def handle_post(self) -> None:
            parts = self.path_parts
            if parts == ["echo"]:
                self.write_json({"ok": True, "echo": self.read_json()})
                return
            if parts == ["action"]:
                payload = self.read_json()
                action = str(payload.get("action", "")).strip()
                if action == "echo":
                    self.write_json({"ok": True, "action": "echo", "payload": payload.get("payload")})
                    return
                if action in {"preflight", "preflight_check", "psse_preflight"}:
                    self.write_json(run_preflight(config))
                    return
                if action == "psse_import_check":
                    self.write_json(psse_import_check(config))
                    return
                self.write_json(
                    {
                        "ok": False,
                        "error_type": "unsupported_action",
                        "message": "Supported actions: echo, preflight_check, psse_import_check.",
                    },
                    HTTPStatus.BAD_REQUEST,
                )
                return
            if parts == ["jobs"]:
                record = manager.submit(self.read_json())
                self.write_json({"ok": True, "job_id": record.job_id, "job": record.to_dict()}, HTTPStatus.ACCEPTED)
                return
            self.write_json({"ok": False, "error_type": "not_found"}, HTTPStatus.NOT_FOUND)

        @property
        def path_parts(self) -> List[str]:
            parsed = urlparse(self.path)
            return [part for part in parsed.path.split("/") if part]

        def authorized(self) -> bool:
            if not config.token:
                return True
            header = self.headers.get("Authorization", "")
            if header == f"Bearer {config.token}":
                return True
            return self.headers.get("X-PSSE-Token", "") == config.token

        def read_json(self) -> Dict[str, Any]:
            length_text = self.headers.get("Content-Length", "0")
            try:
                length = int(length_text)
            except ValueError:
                length = 0
            raw = self.rfile.read(max(length, 0))
            if not raw:
                return {}
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON request body: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON request body must be an object.")
            return payload

        def write_json(self, payload: Mapping[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(jsonable(payload), indent=2, sort_keys=True).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write(
                "{0} - - [{1}] {2}\n".format(
                    self.address_string(),
                    self.log_date_time_string(),
                    fmt % args,
                )
            )

    return PsseRemoteHandler


def validate_job_request(payload: Mapping[str, Any]) -> Tuple[str, str]:
    if not isinstance(payload, Mapping):
        raise WorkerError("Job request must be a JSON object.", error_type="invalid_job_request")
    case_id = str(payload.get("case_id", "")).strip()
    if case_id not in CASE_DEFINITIONS:
        raise WorkerError(
            f"Unsupported case_id '{case_id}'. Allowed: {', '.join(sorted(CASE_DEFINITIONS))}",
            error_type="unsupported_case_id",
        )
    scenario = payload.get("scenario")
    scenario_type = str(payload.get("scenario_type") or payload.get("scenario_id") or "").strip()
    if isinstance(scenario, Mapping):
        scenario_type = str(scenario.get("type") or scenario.get("scenario_type") or scenario_type).strip()
    elif isinstance(scenario, str) and scenario.strip():
        scenario_type = scenario.strip()
    scenario_type = normalize_scenario_type(scenario_type)
    allowed = CASE_DEFINITIONS[case_id].allowed_scenarios
    if scenario_type not in allowed:
        raise WorkerError(
            f"Unsupported scenario_type '{scenario_type}' for {case_id}. Allowed: {', '.join(allowed)}",
            error_type="unsupported_scenario_type",
        )
    return case_id, scenario_type


def normalize_scenario_type(value: str) -> str:
    aliases = {
        "none": "no_disturbance_5s",
        "no_disturbance": "no_disturbance_5s",
        "baseline": "no_disturbance_5s",
        "pq_target": "pq_target_step",
        "p_q_target": "pq_target_step",
        "static_pf": "static",
        "load_flow": "static",
        "powerflow": "static",
    }
    normalized = aliases.get(str(value).strip().lower(), str(value).strip().lower())
    if not normalized:
        raise WorkerError("scenario_type is required.", error_type="invalid_job_request")
    if normalized in {"bus_fault", "fault", "line_trip", "qref_step", "pref_step", "vref_step"}:
        raise WorkerError(
            f"Scenario '{normalized}' is intentionally blocked until validated.",
            error_type="scenario_not_validated",
        )
    return normalized


def scenario_requires_dynamic_m2(scenario_type: str) -> bool:
    return scenario_type in {"no_disturbance_5s"}


def execute_job(record: JobRecord, config: WorkerConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    case = CASE_DEFINITIONS[record.case_id]
    artifact_dir = create_artifact_dir(config, record)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "request.json", record.request)
    validate_case_files(config, case)
    if case.executor == "pssecmd36_50000_bus_idv":
        result = execute_pssecmd_case(record, case, config, artifact_dir)
    elif case.executor == "standalone_psspy_subprocess":
        result = execute_small_case_python(record, case, config, artifact_dir)
    else:
        raise WorkerError(f"Unknown executor: {case.executor}", error_type="executor_unavailable")
    artifacts = artifact_manifest(artifact_dir)
    return result, artifacts


def execute_pssecmd_case(
    record: JobRecord,
    case: CaseDefinition,
    config: WorkerConfig,
    artifact_dir: Path,
) -> Dict[str, Any]:
    idv_path = artifact_dir / "run.idv"
    outx_path = artifact_dir / f"{record.case_id}_{record.scenario_type}.outx"
    channel_csv = artifact_dir / "channels.csv"
    idv_text = build_pssecmd_idv(case, config, record.scenario_type, outx_path)
    idv_path.write_text(idv_text, encoding="utf-8")
    command = [
        str(config.psse_cmd),
        "-buses",
        "50000",
        "-pyver",
        config.psse_pyver,
        "-inpdev",
        str(idv_path),
    ]
    proc = run_subprocess(command, cwd=config.repo_root, timeout=config.job_timeout_s)
    write_text(artifact_dir / "stdout.txt", proc["stdout"])
    write_text(artifact_dir / "stderr.txt", proc["stderr"])
    write_json(artifact_dir / "subprocess.json", proc)
    error_lines = extract_error_lines(proc["stdout"], proc["stderr"])
    converged = "Reached tolerance" in proc["stdout"]
    psse_version = parse_psse_banner(proc["stdout"])

    m1 = {
        "status": "pass" if proc["returncode"] == 0 and converged else "fail",
        "converged": bool(converged),
        "bus_count": case.bus_count,
        "branch_count": case.branch_count,
        "machine_count": case.machine_count,
        "load_count": case.load_count,
        "poc_bus": case.poc_bus,
        "poc_branch": branch_dict(case.poc_branch),
        "bus_voltage_min_pu": None,
        "bus_voltage_max_pu": None,
        "poc_p_mw": None,
        "poc_q_mvar": None,
        "source": "pssecmd36_stdout_plus_case_allowlist_metadata",
        "topology_count_source": "allowlist_metadata_validated_by_prior_psse_inventory",
        "static_measurements_available": False,
        "static_measurements_note": (
            "The pssecmd36 IDV static path currently proves load-flow convergence. "
            "POC P/Q and voltage extrema are populated by the no_disturbance_5s channel path."
        ),
        "error_summary": error_lines,
    }
    m2 = {
        "status": "not_run",
        "initialized": False,
        "simulation_converged": False,
        "event": record.scenario_type,
        "metrics": {
            "min_voltage_pu": None,
            "max_voltage_pu": None,
            "final_poc_p_mw": None,
            "final_poc_q_mvar": None,
            "max_frequency_deviation_pu": None,
            "max_frequency_deviation_hz_at_60hz": None,
        },
        "error_summary": [],
    }
    if record.scenario_type == "no_disturbance_5s":
        dynamic_status = psse_dynamic_status_from_stdout(proc["stdout"], expected_end_time_s=5.0)
        channel_export = export_outx_channels(outx_path, channel_csv, config)
        write_json(artifact_dir / "channel_export.json", channel_export)
        if channel_export.get("ok"):
            metrics = metrics_from_channel_csv(channel_csv, case)
            final_time = optional_float(metrics.get("final_time_s"))
            reached_target_time = final_time is not None and final_time >= 4.99
            simulation_ok = (
                proc["returncode"] == 0
                and dynamic_status["initial_conditions_ok"]
                and dynamic_status["bat_run_seen"]
                and dynamic_status["response_file_completed"]
                and not dynamic_status["initialization_failed"]
                and not dynamic_status["run_failed"]
                and reached_target_time
                and not error_lines
            )
            m2.update(
                {
                    "status": "pass" if simulation_ok else "fail",
                    "initialized": bool(dynamic_status["initial_conditions_ok"]),
                    "simulation_converged": bool(simulation_ok),
                    "metrics": metrics,
                    "channel_csv": str(channel_csv),
                    "psse_dynamic_status": dynamic_status,
                    "error_summary": error_lines,
                }
            )
            m1["poc_p_mw"] = metrics.get("initial_poc_p_mw")
            m1["poc_q_mvar"] = metrics.get("initial_poc_q_mvar")
            m1["bus_voltage_min_pu"] = metrics.get("min_voltage_pu")
            m1["bus_voltage_max_pu"] = metrics.get("max_voltage_pu")
            m1["static_measurements_available"] = True
            m1["static_measurements_note"] = "Initial POC P/Q are taken from the exported dynamic channel rows."
        else:
            m2.update(
                {
                    "status": "fail",
                    "initialized": bool(dynamic_status["initial_conditions_ok"]),
                    "simulation_converged": False,
                    "psse_dynamic_status": dynamic_status,
                    "error_summary": [str(channel_export.get("message") or channel_export.get("error_type"))],
                }
            )
    ok = proc["returncode"] == 0 and m1["status"] == "pass" and (record.scenario_type == "static" or m2["status"] == "pass")
    return remote_result(
        record=record,
        case=case,
        ok=ok,
        psse_version=psse_version,
        m1=m1,
        m2=m2,
        artifact_dir=artifact_dir,
        execution={
            "executor": case.executor,
            "command": command,
            "returncode": proc["returncode"],
            "stdout_path": str(artifact_dir / "stdout.txt"),
            "stderr_path": str(artifact_dir / "stderr.txt"),
            "idv_path": str(idv_path),
            "outx_path": str(outx_path) if outx_path.exists() else None,
            "error_lines": error_lines,
        },
    )


def build_pssecmd_idv(case: CaseDefinition, config: WorkerConfig, scenario_type: str, outx_path: Path) -> str:
    case_dir = case_dir_path(config, case)
    sav = case_dir / case.sav
    lines = [
        f"case '{sav}'",
        "BAT_FDNS 0 0 0 1 1 0 99 0",
    ]
    if scenario_type == "no_disturbance_5s":
        if not case.dyr:
            raise WorkerError(f"Case {case.case_id} has no DYR file.", error_type="dyr_unavailable")
        lines.append(f"BAT_DYRE_NEW_2 1 1 1 1 '{case_dir / case.dyr}'")
        for dll in case.dlls:
            lines.append(f"BAT_ADDMODELLIBRARY '{case_dir / dll}'")
        lines.extend(
            [
                "BAT_DYNAMICS_SOLUTION_PARAM_2 50 100000000 100000000 100000000 100000000 100000000 100000000 100000000 0.3 1.000000020041e+20 0.001 1.000000020041e+20 1.000000020041e+20 1.000000020041e+20 1.000000020041e+20 1.000000020041e+20",
                "BAT_CONG 0",
                "BAT_CONL 0 1 1 0 0 100.00 0.00 0.00 100.00",
                "BAT_CONL 0 1 2 0 0 100.00 0.00 0.00 100.00",
                "BAT_CONL 0 1 3 0 0 100.00 0.00 0.00 100.00",
            ]
        )
        for kind, args in case.dynamic_channels:
            if kind == "voltage":
                bus, label = args
                lines.append(f"BAT_VOLTAGE_CHANNEL -1 -1 -1 {bus} '{label}'")
            elif kind == "branch_pq":
                from_bus, to_bus, ckt, p_label, q_label = args
                lines.append(
                    f"BAT_BRANCH_P_AND_Q_CHANNEL -1 -1 -1 {from_bus} {to_bus} '{ckt}' '{p_label}' '{q_label}'"
                )
            elif kind == "machine_array":
                quantity, bus, machine_id, label = args
                lines.append(f"BAT_MACHINE_ARRAY_CHANNEL -1 {quantity} {bus} '{machine_id}' '{label}'")
            elif kind == "bus_frequency":
                bus, label = args
                lines.append(f"BAT_BUS_FREQUENCY_CHANNEL -1 {bus} '{label}'")
        lines.extend(
            [
                f"BAT_STRT_2 1 1 '{outx_path}'",
                "BAT_RUN 0 5.0 1 1 0",
            ]
        )
    return "\n".join(lines) + "\n"


def execute_small_case_python(
    record: JobRecord,
    case: CaseDefinition,
    config: WorkerConfig,
    artifact_dir: Path,
) -> Dict[str, Any]:
    script_path = artifact_dir / "small_case_job.py"
    output_json = artifact_dir / "small_case_output.json"
    channel_csv = artifact_dir / "channels.csv"
    script_path.write_text(
        build_small_case_script(case, config, record.scenario_type, output_json, channel_csv),
        encoding="utf-8",
    )
    proc = run_subprocess([sys.executable, str(script_path)], cwd=config.repo_root, timeout=config.job_timeout_s)
    write_text(artifact_dir / "stdout.txt", proc["stdout"])
    write_text(artifact_dir / "stderr.txt", proc["stderr"])
    write_json(artifact_dir / "subprocess.json", proc)
    error_lines = extract_error_lines(proc["stdout"], proc["stderr"])
    if output_json.exists():
        payload = dict(load_json(output_json))
    else:
        payload = {
            "ok": False,
            "error_type": "small_case_output_missing",
            "message": "Generated small-case script did not write output JSON.",
        }
    m1 = payload.get("m1_powerflow") if isinstance(payload.get("m1_powerflow"), Mapping) else {}
    m2 = payload.get("m2_dynamic") if isinstance(payload.get("m2_dynamic"), Mapping) else {}
    ok = (
        bool(payload.get("ok"))
        and proc["returncode"] == 0
        and dict(m1).get("status") == "pass"
        and (not scenario_requires_dynamic_m2(record.scenario_type) or dict(m2).get("status") == "pass")
    )
    return remote_result(
        record=record,
        case=case,
        ok=ok,
        psse_version=payload.get("psse_version"),
        m1={
            **dict(m1),
            "bus_count": dict(m1).get("bus_count", case.bus_count),
            "branch_count": dict(m1).get("branch_count", case.branch_count),
            "machine_count": dict(m1).get("machine_count", case.machine_count),
            "load_count": dict(m1).get("load_count", case.load_count),
            "error_summary": error_lines,
        },
        m2=dict(m2) if m2 else {
            "status": "not_run",
            "initialized": False,
            "simulation_converged": False,
            "event": record.scenario_type,
            "metrics": {},
            "error_summary": [],
        },
        artifact_dir=artifact_dir,
        execution={
            "executor": case.executor,
            "command": [sys.executable, str(script_path)],
            "returncode": proc["returncode"],
            "stdout_path": str(artifact_dir / "stdout.txt"),
            "stderr_path": str(artifact_dir / "stderr.txt"),
            "script_path": str(script_path),
            "channel_csv": str(channel_csv) if channel_csv.exists() else None,
            "error_lines": error_lines,
        },
    )


def build_small_case_script(
    case: CaseDefinition,
    config: WorkerConfig,
    scenario_type: str,
    output_json: Path,
    channel_csv: Path,
) -> str:
    case_dir = case_dir_path(config, case)
    sav = case_dir / case.sav
    dyr = case_dir / case.dyr if case.dyr else None
    dlls = [case_dir / dll for dll in case.dlls]
    return f'''from __future__ import annotations
import csv
import json
import math
import traceback
from pathlib import Path

try:
    try:
        import psse3602  # type: ignore
    except Exception:
        import psse36  # type: ignore
    import psspy  # type: ignore
except Exception as exc:
    Path(r"{output_json}").write_text(json.dumps({{"ok": False, "error_type": type(exc).__name__, "message": str(exc)}}), encoding="utf-8")
    raise SystemExit(1)

SAV = r"{sav}"
DYR = {repr(str(dyr) if dyr else "")}
DLLS = {[str(path) for path in dlls]!r}
SCENARIO = {scenario_type!r}
OUTPUT_JSON = Path(r"{output_json}")
CHANNEL_CSV = Path(r"{channel_csv}")

def ierr_value(value):
    if isinstance(value, int):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], int):
        return value[0]
    return None

def psse_value(fn, *args):
    try:
        value = fn(*args)
        if isinstance(value, tuple) and len(value) >= 2:
            return None if value[0] != 0 else value[1]
        return value
    except Exception:
        return None

def poc():
    value = psse_value(psspy.brnflo, 800, 900, '1')
    if isinstance(value, complex):
        return value.real, value.imag
    return None, None

def vbus(bus):
    return psse_value(psspy.busdat, bus, 'PU')

def set_wind(bus, pg, q):
    ig = [1,0,0,0,0,1,0]
    rl = [pg, q, q, q, 9999, -9999, 520, 0,1,0,0,1,1,0,0,0,1]
    try:
        return psspy.machine_data_4(bus, '1', ig, rl)
    except Exception:
        return psspy.machine_data_5(bus, '1', ig, rl)

def set_stat(bus, q):
    ig = [1,0,0,0,0,1,0]
    rl = [0.0, q, q, q, 9999, -9999, 1000, 0,1,0,0,1,1,0,0,0,1]
    try:
        return psspy.machine_data_4(bus, '1', ig, rl)
    except Exception:
        return psspy.machine_data_5(bus, '1', ig, rl)

def base_m1():
    p, q = poc()
    voltages = [vbus(b) for b in [900, 800, 700, 40, 41, 200, 201]]
    voltages = [v for v in voltages if isinstance(v, (int, float))]
    converged = psspy.solved() == 0
    return {{
        "status": "pass" if converged else "fail",
        "converged": converged,
        "bus_voltage_min_pu": min(voltages) if voltages else None,
        "bus_voltage_max_pu": max(voltages) if voltages else None,
        "poc_p_mw": p,
        "poc_q_mvar": q,
        "poc_bus": 900,
        "poc_branch": {{"from_bus": 800, "to_bus": 900, "circuit_id": "1"}},
        "bus_count": 11,
        "branch_count": 10,
        "machine_count": 5,
        "load_count": 0,
        "overloaded_branch_count": 0,
    }}

def run_static():
    psspy.psseinit(50)
    psspy.case(SAV)
    psspy.fdns([0,0,0,1,1,0,99,0])
    return {{"ok": True, "m1_powerflow": base_m1(), "m2_dynamic": {{"status": "not_run", "initialized": False, "simulation_converged": False, "event": "static", "metrics": {{}}}}}}

def run_pq_target():
    rows = []
    psspy.psseinit(50)
    psspy.case(SAV)
    psspy.fdns([0,0,0,1,1,0,99,0])
    p, q = poc()
    rows.append({{"stage": "base", "poc_p_mw": p, "poc_q_mvar": q}})
    target_p = 200.0
    target_q = 330.0
    wp = 500.0
    for it in range(15):
        set_wind(40, wp, 0.0)
        set_wind(41, wp, 0.0)
        psspy.fdns([0,0,0,1,1,0,99,0])
        p, q = poc()
        err = p - target_p if p is not None else 0.0
        rows.append({{"stage": "step_a_reduce_p", "iteration": it, "wind_p_mw_each": wp, "wind_q_mvar_each": 0.0, "poc_p_mw": p, "poc_q_mvar": q, "p_error_mw": err}})
        if abs(err) <= 0.5:
            break
        wp -= err / 2.0
    set_wind(40, wp, 0.0)
    set_wind(41, wp, 0.0)
    psspy.fdns([0,0,0,1,1,0,99,0])
    p, q = poc()
    rows.append({{"stage": "step_b_confirm_wind_q0", "wind_p_mw_each": wp, "wind_q_mvar_each": 0.0, "poc_p_mw": p, "poc_q_mvar": q}})
    set_stat(200, 150.0)
    set_stat(201, 150.0)
    psspy.fdns([0,0,0,1,1,0,99,0])
    p, q = poc()
    rows.append({{"stage": "step_c_statcom_qmax", "stat_q_mvar_each": 150.0, "poc_p_mw": p, "poc_q_mvar": q, "q_needed_mvar": None if q is None else target_q - q}})
    if q is not None and q < target_q - 0.1:
        wq = 0.0
        for it in range(15):
            set_wind(40, wp, wq)
            set_wind(41, wp, wq)
            psspy.fdns([0,0,0,1,1,0,99,0])
            p, q = poc()
            err = q - target_q if q is not None else 0.0
            rows.append({{"stage": "step_d_increase_wind_q", "iteration": it, "wind_p_mw_each": wp, "wind_q_mvar_each": wq, "poc_p_mw": p, "poc_q_mvar": q, "q_error_mvar": err}})
            if abs(err) <= 0.1:
                break
            wq -= err / 2.0
    m1 = base_m1()
    with CHANNEL_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "iteration", "wind_p_mw_each", "wind_q_mvar_each", "stat_q_mvar_each", "poc_p_mw", "poc_q_mvar", "p_error_mw", "q_needed_mvar", "q_error_mvar"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return {{"ok": True, "m1_powerflow": m1, "m2_dynamic": {{"status": "not_run", "initialized": False, "simulation_converged": False, "event": "static_pq_target_reproduction", "metrics": {{"final_poc_p_mw": m1.get("poc_p_mw"), "final_poc_q_mvar": m1.get("poc_q_mvar")}}}}}}

def run_no_disturbance():
    psspy.psseinit(50)
    psspy.case(SAV)
    psspy.fdns([0,0,0,1,1,0,99,0])
    psspy.dyre_new_2([1,1,1,1], DYR)
    for dll in DLLS:
        psspy.addmodellibrary(dll)
    _i = psspy.getdefaultint()
    _f = psspy.getdefaultreal()
    psspy.dynamics_solution_param_2([50,_i,_i,_i,_i,_i,_i,_i], [0.3,_f,0.001,_f,_f,_f,_f,_f])
    psspy.cong(0)
    psspy.conl(0,1,1,[0,0],[100.0,0.0,0.0,100.0])
    psspy.conl(0,1,2,[0,0],[100.0,0.0,0.0,100.0])
    psspy.conl(0,1,3,[0,0],[100.0,0.0,0.0,100.0])
    outx = str(CHANNEL_CSV.with_suffix(".outx"))
    strt_ierr = ierr_value(psspy.strt_2([1,1], outx))
    rows = []
    run_ierrs = []
    for i in range(0, 101):
        t = round(i * 0.05, 6)
        run_ierr = ierr_value(psspy.run(0, t, 1, 1, 0))
        run_ierrs.append({{"time": t, "ierr": run_ierr}})
        p, q = poc()
        rows.append({{"time": t, "POC_P_800_900": p, "POC_Q_800_900": q, "V_900": vbus(900), "V_800": vbus(800)}})
    with CHANNEL_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "POC_P_800_900", "POC_Q_800_900", "V_900", "V_800"])
        writer.writeheader()
        writer.writerows(rows)
    m1 = base_m1()
    p_vals = [r["POC_P_800_900"] for r in rows if isinstance(r["POC_P_800_900"], (int, float))]
    q_vals = [r["POC_Q_800_900"] for r in rows if isinstance(r["POC_Q_800_900"], (int, float))]
    v_vals = [r["V_900"] for r in rows if isinstance(r["V_900"], (int, float))] + [r["V_800"] for r in rows if isinstance(r["V_800"], (int, float))]
    failed_run_ierrs = [item for item in run_ierrs if item["ierr"] not in (0, None)]
    simulation_ok = strt_ierr in (0, None) and not failed_run_ierrs
    m2 = {{
        "status": "pass" if simulation_ok else "fail",
        "initialized": strt_ierr in (0, None),
        "simulation_converged": simulation_ok,
        "event": "no_disturbance_5s",
        "metrics": {{
            "min_voltage_pu": min(v_vals) if v_vals else None,
            "max_voltage_pu": max(v_vals) if v_vals else None,
            "final_poc_p_mw": p_vals[-1] if p_vals else None,
            "final_poc_q_mvar": q_vals[-1] if q_vals else None,
            "max_oscillation": max([abs(x - p_vals[0]) for x in p_vals]) if p_vals else None,
            "run_step_count": len(run_ierrs),
            "failed_run_step_count": len(failed_run_ierrs),
        }},
        "strt_ierr": strt_ierr,
        "failed_run_ierrs": failed_run_ierrs[:10],
    }}
    return {{"ok": simulation_ok, "m1_powerflow": m1, "m2_dynamic": m2}}

try:
    if SCENARIO == "static":
        out = run_static()
    elif SCENARIO == "pq_target_step":
        out = run_pq_target()
    elif SCENARIO == "no_disturbance_5s":
        out = run_no_disturbance()
    else:
        out = {{"ok": False, "error_type": "unsupported_scenario", "message": SCENARIO}}
    try:
        out["psse_version"] = psspy.psseversion()
    except Exception:
        out["psse_version"] = None
    OUTPUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
except Exception as exc:
    OUTPUT_JSON.write_text(json.dumps({{"ok": False, "error_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}, indent=2, sort_keys=True), encoding="utf-8")
    raise SystemExit(1)
'''


def remote_result(
    *,
    record: JobRecord,
    case: CaseDefinition,
    ok: bool,
    psse_version: Any,
    m1: Mapping[str, Any],
    m2: Mapping[str, Any],
    artifact_dir: Path,
    execution: Mapping[str, Any],
) -> Dict[str, Any]:
    recommendation = "approve" if ok else "reject"
    reason_codes = ["psse_remote_execution_pass"] if ok else ["psse_remote_execution_failed"]
    if m1.get("status") != "pass":
        reason_codes.append("m1_psse_powerflow_failed")
    if scenario_requires_dynamic_m2(record.scenario_type) and m2.get("status") != "pass":
        reason_codes.append("m2_psse_rms_dynamic_failed")
    return {
        "ok": bool(ok),
        "schema_version": REMOTE_RESULT_SCHEMA_VERSION,
        "tool": "run_remote_psse_job",
        "backend": "psse_remote_worker",
        "job_id": record.job_id,
        "case_id": record.case_id,
        "scenario_type": record.scenario_type,
        "display_name": case.display_name,
        "psse_version": psse_version,
        "artifact_dir": str(artifact_dir),
        "execution": jsonable(execution),
        "m1_powerflow": jsonable(m1),
        "m2_dynamic": jsonable(m2),
        "emt_result": {
            "status": "not_run",
            "backend": "not_available_in_remote_v1",
            "reason_codes": ["psse_rms_dynamic_not_emt_waveform"],
        },
        "recommendation": recommendation,
        "complete": bool(ok),
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "limitations": list(case.limitations),
        "message": None if ok else "PSS/E remote job failed; inspect execution.error_lines and artifacts.",
    }


def run_preflight(config: WorkerConfig) -> Dict[str, Any]:
    psse_import = psse_import_check(config)
    capacity_probe = pssecmd_capacity_probe(config)
    dll_checks = {case_id: dll_load_check(config, case) for case_id, case in CASE_DEFINITIONS.items()}
    return {
        "ok": bool(capacity_probe.get("ok")),
        "action": "preflight_check",
        "time_utc": utc_now(),
        "psspy_import": psse_import,
        "standalone_psspy_capacity": {
            "bus_limit": 50,
            "source": "observed_normal_python_xplore_limit_not_a_license_check",
            "note": "Do not use standalone psspy capacity as the PIF6 execution capability.",
        },
        "pssecmd36_capacity": capacity_probe,
        "dll_load_results": dll_checks,
        "warnings": [
            "Close the PSS/E GUI before running jobs to avoid session/license contention.",
            "PIF6 jobs must use pssecmd36 -buses 50000; standalone psspy is intentionally not used for PIF6.",
        ],
    }


def psse_import_check(config: WorkerConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    ensure_psse_path(config)
    try:
        try:
            import psse3602 as psse_module  # type: ignore
        except Exception:
            import psse36 as psse_module  # type: ignore
        import psspy  # type: ignore
    except Exception as exc:
        return {
            "ok": False,
            "action": "psse_import_check",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "duration_s": round(time.perf_counter() - started, 6),
            "hint": "Run worker with PSS/E-compatible Python or set PATH/sys.path to PSSBIN.",
        }
    try:
        version = psspy.psseversion()
    except Exception:
        version = None
    return {
        "ok": True,
        "action": "psse_import_check",
        "psse_module": repr(psse_module),
        "psse_version": version,
        "duration_s": round(time.perf_counter() - started, 6),
        "not_a_capacity_check": True,
    }


def pssecmd_capacity_probe(config: WorkerConfig) -> Dict[str, Any]:
    probe_dir = config.output_root / "_preflight"
    probe_dir.mkdir(parents=True, exist_ok=True)
    idv = probe_dir / f"capacity_probe_{int(time.time())}.idv"
    idv.write_text("\n", encoding="utf-8")
    command = [
        str(config.psse_cmd),
        "-buses",
        "50000",
        "-pyver",
        config.psse_pyver,
        "-inpdev",
        str(idv),
    ]
    try:
        proc = run_subprocess(command, cwd=config.repo_root, timeout=60.0)
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "command": command,
        }
    banner = parse_psse_banner(proc["stdout"])
    return {
        "ok": proc["returncode"] == 0 and "50000 BUS POWER SYSTEM SIMULATOR" in proc["stdout"],
        "command": command,
        "returncode": proc["returncode"],
        "banner": banner,
        "capacity_bus_limit": 50000 if "50000 BUS POWER SYSTEM SIMULATOR" in proc["stdout"] else None,
        "stdout_tail": tail_lines(proc["stdout"], 40),
        "stderr_tail": tail_lines(proc["stderr"], 40),
    }


def dll_load_check(config: WorkerConfig, case: CaseDefinition) -> Dict[str, Any]:
    if not case.dlls:
        return {"ok": True, "dlls": []}
    ensure_psse_path(config)
    try:
        try:
            import psse3602  # type: ignore  # noqa: F401
        except Exception:
            import psse36  # type: ignore  # noqa: F401
        import psspy  # type: ignore
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "message": str(exc), "dlls": []}
    try:
        psspy.psseinit(50)
    except Exception:
        pass
    dlls = []
    for dll in case.dlls:
        path = case_dir_path(config, case) / dll
        try:
            value = psspy.addmodellibrary(str(path))
            ierr = extract_ierr(value)
            dlls.append({"path": str(path), "loaded": ierr in (0, None), "ierr": ierr, "error": None})
        except Exception as exc:
            dlls.append({"path": str(path), "loaded": False, "ierr": None, "error": f"{type(exc).__name__}: {exc}"})
    return {"ok": all(item["loaded"] for item in dlls), "dlls": dlls}


def export_outx_channels(outx_path: Path, csv_path: Path, config: WorkerConfig) -> Dict[str, Any]:
    if not outx_path.exists():
        return {"ok": False, "error_type": "outx_missing", "message": f"OUTX file not found: {outx_path}"}
    ensure_psse_path(config)
    try:
        import dyntools  # type: ignore
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "message": f"Cannot import dyntools: {exc}"}
    try:
        chnf = dyntools.CHNF(str(outx_path))
        _short_title, chanid, chandata = chnf.get_data()
        time_values = list(chandata.get("time") or chandata.get("Time(s)") or [])
        labels = []
        channel_keys = []
        for key, label in chanid.items():
            if str(key).lower() == "time":
                continue
            channel_keys.append(key)
            labels.append(str(label).strip())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", *labels])
            for index, t in enumerate(time_values):
                row = [t]
                for key in channel_keys:
                    series = chandata.get(key) or []
                    row.append(series[index] if index < len(series) else None)
                writer.writerow(row)
        return {"ok": True, "path": str(csv_path), "row_count": len(time_values), "channels": labels}
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }


def metrics_from_channel_csv(path: Path, case: CaseDefinition) -> Dict[str, Any]:
    rows = read_csv_dicts(path)
    if not rows:
        return {}
    columns = rows[0].keys()
    numeric: Dict[str, List[float]] = {}
    for col in columns:
        values = []
        for row in rows:
            value = optional_float(row.get(col))
            if value is not None:
                values.append(value)
        if values:
            numeric[col] = values
    voltage_cols = [col for col in numeric if col.upper().endswith("_V") or col.startswith("V_")]
    freq_cols = [col for col in numeric if "FREQ" in col.upper()]
    p_col = "POC_P_2001_2" if case.case_id == "pif6_2026_05_17" else "POC_P_800_900"
    q_col = "POC_Q_2001_2" if case.case_id == "pif6_2026_05_17" else "POC_Q_800_900"
    p_values = numeric.get(p_col, [])
    q_values = numeric.get(q_col, [])
    time_values = numeric.get("time", [])
    v_values = [item for col in voltage_cols for item in numeric.get(col, [])]
    f_values = [item for col in freq_cols for item in numeric.get(col, [])]
    max_frequency_deviation_pu = max(abs(value) for value in f_values) if f_values else None
    return {
        "min_voltage_pu": min(v_values) if v_values else None,
        "max_voltage_pu": max(v_values) if v_values else None,
        "initial_poc_p_mw": p_values[0] if p_values else None,
        "initial_poc_q_mvar": q_values[0] if q_values else None,
        "final_poc_p_mw": p_values[-1] if p_values else None,
        "final_poc_q_mvar": q_values[-1] if q_values else None,
        "max_oscillation": max(abs(value - p_values[0]) for value in p_values) if p_values else None,
        "max_frequency_deviation_pu": max_frequency_deviation_pu,
        "max_frequency_deviation_hz_at_60hz": (
            max_frequency_deviation_pu * 60.0 if max_frequency_deviation_pu is not None else None
        ),
        "frequency_channel_unit": "per_unit_frequency_deviation",
        "initial_time_s": time_values[0] if time_values else None,
        "final_time_s": time_values[-1] if time_values else None,
        "channel_row_count": len(rows),
        "channel_columns": list(columns),
    }


def run_subprocess(command: Sequence[str], *, cwd: Path, timeout: float) -> Dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        list(command),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        "command": list(command),
        "returncode": proc.returncode,
        "duration_s": round(time.perf_counter() - started, 6),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def validate_case_files(config: WorkerConfig, case: CaseDefinition) -> None:
    case_dir = case_dir_path(config, case)
    paths = [case_dir / case.sav]
    if case.dyr:
        paths.append(case_dir / case.dyr)
    paths.extend(case_dir / dll for dll in case.dlls)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise WorkerError(f"Missing allowlisted case files: {missing}", error_type="case_files_missing")


def case_dir_path(config: WorkerConfig, case: CaseDefinition) -> Path:
    path = config.repo_root
    for part in case.folder_parts:
        path = path / part
    return path


def create_artifact_dir(config: WorkerConfig, record: JobRecord) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{record.case_id}_{record.scenario_type}")
    return config.output_root / f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{record.job_id[:10]}_{safe}"


def artifact_manifest(artifact_dir: Path) -> Dict[str, Any]:
    files = []
    if artifact_dir.exists():
        for path in sorted(item for item in artifact_dir.rglob("*") if item.is_file()):
            files.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "relative_path": str(path.relative_to(artifact_dir)),
                    "bytes": path.stat().st_size,
                }
            )
    return {
        "ok": True,
        "artifact_dir": str(artifact_dir),
        "file_count": len(files),
        "files": files,
    }


def write_final_artifact_manifest(artifact_dir: Path, artifacts_path: Path) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {}
    previous_text: Optional[str] = None
    for _ in range(4):
        manifest = artifact_manifest(artifact_dir)
        text = json.dumps(jsonable(manifest), indent=2, sort_keys=True)
        if text == previous_text:
            break
        artifacts_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts_path.write_text(text, encoding="utf-8")
        previous_text = text
    return manifest


def failure_result(record: JobRecord, artifact_dir: Path, exc: Exception) -> Dict[str, Any]:
    return {
        "ok": False,
        "schema_version": REMOTE_RESULT_SCHEMA_VERSION,
        "tool": "run_remote_psse_job",
        "backend": "psse_remote_worker",
        "job_id": record.job_id,
        "case_id": record.case_id,
        "scenario_type": record.scenario_type,
        "artifact_dir": str(artifact_dir),
        "error_type": getattr(exc, "error_type", type(exc).__name__),
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "recommendation": "reject",
        "complete": False,
        "reason_codes": ["psse_remote_worker_exception"],
    }


def ensure_psse_path(config: WorkerConfig) -> None:
    psse_bin = config.psse_cmd.parent
    if str(psse_bin) not in sys.path:
        sys.path.insert(0, str(psse_bin))
    path_text = os.environ.get("PATH", "")
    if str(psse_bin) not in path_text:
        os.environ["PATH"] = str(psse_bin) + os.pathsep + path_text
    add_dll_dir = getattr(os, "add_dll_directory", None)
    if add_dll_dir is not None and psse_bin.exists():
        try:
            add_dll_dir(str(psse_bin))
        except OSError:
            pass


def extract_error_lines(stdout: str, stderr: str, limit: int = 80) -> List[str]:
    patterns = re.compile(r"(error|fatal|traceback|not found|input error|failed|allowed)", re.I)
    lines = []
    for stream_name, text in (("stdout", stdout), ("stderr", stderr)):
        for line_no, line in enumerate(text.splitlines(), start=1):
            if patterns.search(line):
                lines.append(f"{stream_name}:{line_no}: {line.strip()}")
    return lines[:limit]


def psse_dynamic_status_from_stdout(stdout: str, *, expected_end_time_s: float) -> Dict[str, Any]:
    lower = stdout.lower()
    initial_conditions_ok = "initial conditions check o.k." in lower
    bat_run_seen = bool(re.search(r"\bBAT_RUN\b", stdout, re.I))
    response_file_completed = "response file operation terminated" in lower or "end-of-file from terminal" in lower
    initialization_failed = bool(
        re.search(r"(initial conditions.*(?:fail|suspect|not ok)|network not in factored form)", stdout, re.I)
    )
    run_failed = bool(re.search(r"(activity run.*(?:fail|error)|simulation.*(?:fail|error))", stdout, re.I))
    return {
        "initial_conditions_ok": initial_conditions_ok,
        "bat_run_seen": bat_run_seen,
        "response_file_completed": response_file_completed,
        "initialization_failed": initialization_failed,
        "run_failed": run_failed,
        "expected_end_time_s": expected_end_time_s,
    }


def parse_psse_banner(stdout: str) -> Optional[str]:
    for line in stdout.splitlines():
        if "BUS POWER SYSTEM SIMULATOR--PSS" in line:
            return line.strip()
    for line in stdout.splitlines():
        if "PSS(R)E Xplore Version" in line:
            return line.strip()
    return None


def tail_lines(text: str, limit: int) -> List[str]:
    return text.splitlines()[-limit:]


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def branch_dict(branch: Optional[Tuple[int, int, str]]) -> Optional[Dict[str, Any]]:
    if branch is None:
        return None
    return {"from_bus": branch[0], "to_bus": branch[1], "circuit_id": branch[2]}


def optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_ierr(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], int):
        return value[0]
    return None


def status_for_error(exc: WorkerError) -> HTTPStatus:
    if exc.error_type in {"job_not_found"}:
        return HTTPStatus.NOT_FOUND
    if exc.error_type in {"job_not_finished"}:
        return HTTPStatus.CONFLICT
    return HTTPStatus.BAD_REQUEST


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise WorkerError(f"JSON artifact must be an object: {path}", error_type="invalid_json_artifact")
    return data


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    return value


if __name__ == "__main__":
    raise SystemExit(main())

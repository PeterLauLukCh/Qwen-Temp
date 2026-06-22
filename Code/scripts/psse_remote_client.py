#!/usr/bin/env python3
"""GPU-side client for the queued PSS/E remote worker."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def main() -> int:
    parser = argparse.ArgumentParser(description="Call a PSS/E remote worker.")
    parser.add_argument("--base-url", required=True, help="Worker base URL, e.g. http://192.168.1.50:8765.")
    parser.add_argument("--token", default="", help="Shared token, if worker auth is enabled.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("health", help="GET /health.")
    echo = sub.add_parser("echo", help="POST /echo.")
    echo.add_argument("--message", default="hello from gpu node")
    sub.add_parser("psse-import-check", help="POST /action action=psse_import_check.")
    sub.add_parser("preflight", help="POST /action action=preflight_check.")

    submit = sub.add_parser("submit-job", help="POST /jobs.")
    submit.add_argument("--case-id", required=True, choices=("test_cases_v36", "pif6_2026_05_17"))
    submit.add_argument(
        "--scenario-type",
        required=True,
        choices=("static", "no_disturbance_5s", "pq_target_step"),
        help="Scenario must also be allowed for the selected case.",
    )
    submit.add_argument(
        "--wait",
        action="store_true",
        help="Poll until the job reaches completed/error, then print the result JSON.",
    )
    submit.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between status polls with --wait.")
    submit.add_argument("--max-wait", type=float, default=900.0, help="Maximum wait seconds with --wait.")

    status = sub.add_parser("job-status", help="GET /jobs/{job_id}.")
    status.add_argument("job_id")

    result = sub.add_parser("job-result", help="GET /jobs/{job_id}/result.")
    result.add_argument("job_id")

    artifacts = sub.add_parser("job-artifacts", help="GET /jobs/{job_id}/artifacts.")
    artifacts.add_argument("job_id")
    args = parser.parse_args()

    try:
        if args.command == "health":
            payload = request_json(
                "GET",
                args.base_url,
                "/health",
                token=args.token,
                timeout=args.timeout,
            )
        elif args.command == "echo":
            payload = request_json(
                "POST",
                args.base_url,
                "/echo",
                token=args.token,
                timeout=args.timeout,
                body={"message": args.message},
            )
        elif args.command == "psse-import-check":
            payload = request_json(
                "POST",
                args.base_url,
                "/action",
                token=args.token,
                timeout=args.timeout,
                body={"action": "psse_import_check"},
            )
        elif args.command == "preflight":
            payload = request_json(
                "POST",
                args.base_url,
                "/action",
                token=args.token,
                timeout=args.timeout,
                body={"action": "preflight_check"},
            )
        elif args.command == "submit-job":
            payload = request_json(
                "POST",
                args.base_url,
                "/jobs",
                token=args.token,
                timeout=args.timeout,
                body={"case_id": args.case_id, "scenario_type": args.scenario_type},
            )
            if args.wait and payload.get("ok") and payload.get("job_id"):
                submission = payload
                job_id = str(payload["job_id"])
                payload = wait_for_result(
                    args.base_url,
                    job_id,
                    token=args.token,
                    timeout=args.timeout,
                    poll_interval=args.poll_interval,
                    max_wait=args.max_wait,
                )
                payload.setdefault("job_id", job_id)
                payload.setdefault("submission", submission)
        elif args.command == "job-status":
            payload = request_json(
                "GET",
                args.base_url,
                "/jobs/{0}".format(args.job_id),
                token=args.token,
                timeout=args.timeout,
            )
        elif args.command == "job-result":
            payload = request_json(
                "GET",
                args.base_url,
                "/jobs/{0}/result".format(args.job_id),
                token=args.token,
                timeout=args.timeout,
            )
        elif args.command == "job-artifacts":
            payload = request_json(
                "GET",
                args.base_url,
                "/jobs/{0}/artifacts".format(args.job_id),
                token=args.token,
                timeout=args.timeout,
            )
        else:  # pragma: no cover - argparse prevents this
            raise ValueError("Unsupported command: {0}".format(args.command))
    except urllib.error.HTTPError as exc:
        print(error_payload(exc), file=sys.stderr)
        return 2
    except urllib.error.URLError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": "url_error",
                    "message": str(exc.reason),
                    "hint": "Check IP, port, VPN/routing, Windows Firewall, and worker process.",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("ok") else 2


def wait_for_result(
    base_url: str,
    job_id: str,
    *,
    token: str,
    timeout: float,
    poll_interval: float,
    max_wait: float,
) -> Dict[str, Any]:
    started = time.monotonic()
    while True:
        status = request_json(
            "GET",
            base_url,
            "/jobs/{0}".format(job_id),
            token=token,
            timeout=timeout,
        )
        job = status.get("job")
        if isinstance(job, dict) and job.get("status") in {"completed", "error"}:
            return request_json(
                "GET",
                base_url,
                "/jobs/{0}/result".format(job_id),
                token=token,
                timeout=timeout,
            )
        if time.monotonic() - started > max_wait:
            return {
                "ok": False,
                "error_type": "job_wait_timeout",
                "message": "Timed out waiting for job {0}.".format(job_id),
                "last_status": status,
            }
        time.sleep(max(poll_interval, 0.1))


def request_json(
    method: str,
    base_url: str,
    path: str,
    *,
    token: str,
    timeout: float,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + path
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer {0}".format(token)
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Worker response must be a JSON object.")
    return payload


def error_payload(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read()
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        payload = {
            "ok": False,
            "error_type": "http_error",
            "status": exc.code,
            "message": exc.reason,
        }
    return json.dumps(payload, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())

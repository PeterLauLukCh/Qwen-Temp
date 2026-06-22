# Remote PSS/E Bridge

Goal: let the GPU node ask the Windows PSS/E host to run narrow,
allowlisted PSS/E studies and return compact M1/M2 JSON.

Ping only proves ICMP routing. The first acceptance layer is still HTTP
`/health` and `/echo`; real PSS/E studies use the queued `/jobs` API.

## Architecture

```text
GPU node / agent
  -> HTTP JSON request
  -> Windows PSS/E worker
  -> single serialized job queue
  -> PSS/E subprocess
  -> unique derived-output directory
  -> compact JSON + artifacts
  -> GPU node / agent
```

The worker does not accept arbitrary shell commands or arbitrary paths. It only
accepts the case IDs and scenarios in the local allowlist.

## Confirmed Windows Host

- IP: `192.168.6.85`
- Python: `3.12.7`
- PSS/E: `Xplore 36.2.0`
- Install: `C:\Program Files\PTI\PSSE36\36.2`
- Large-case command path:
  `C:\Program Files\PTI\PSSE36\36.2\PSSBIN\pssecmd36.exe`

Normal Python `psspy` is limited to 50 buses in this environment. That is fine
for `test_cases_v36`, but not for PIF6. PIF6 must run through:

```bat
"C:\Program Files\PTI\PSSE36\36.2\PSSBIN\pssecmd36.exe" -buses 50000 -pyver 312 -inpdev <generated.idv>
```

Do not treat `psse_import_check` as a capacity or license check.

## 1. Start Worker On Windows

Close the PSS/E GUI first to avoid session/license contention.

From `cmd.exe`:

```bat
cd /d "C:\Users\pchen\Desktop\Qwen-Grid-main"
set PSSE_REMOTE_TOKEN=<set-a-strong-token>
py -3 Code\scripts\psse_remote_worker.py --host 0.0.0.0 --port 8765 --token %PSSE_REMOTE_TOKEN% --psse-cmd "C:\Program Files\PTI\PSSE36\36.2\PSSBIN\pssecmd36.exe"
```

The worker writes derived outputs under:

```text
real-data-new\derived_outputs\remote_psse_jobs\
```

Each job gets its own subdirectory containing `request.json`, `result.json`,
logs, generated IDV/Python files, channel CSVs, OUTX files, and an artifact
manifest when available. The original SAV, DYR, and DLL files are read-only
inputs and should not be modified.

## 2. Open Windows Firewall

Use a private network/VPN only. Do not expose this service directly to the
public internet.

PowerShell as Administrator. Replace `GPU_NODE_IP` with the GPU node address
that will call the worker; do not open this port to all remote addresses,
especially when Windows reports the network profile as `Public`.

```powershell
$GpuNodeIp = "GPU_NODE_IP"
New-NetFirewallRule -DisplayName "PSS/E Remote Worker 8765 from GPU only" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8765 -RemoteAddress $GpuNodeIp -Profile Public,Private
```

If an earlier broad test rule exists, remove it after confirming the restricted
rule works:

```powershell
Get-NetFirewallRule -DisplayName "PSS/E Remote Worker 8765" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
```

## 3. Network Smoke Test From GPU Node

```bash
export PSSE_REMOTE_TOKEN='<set-a-strong-token>'
python3 Code/scripts/psse_remote_client.py \
  --base-url http://192.168.6.85:8765 \
  --token "$PSSE_REMOTE_TOKEN" \
  health
```

```bash
python3 Code/scripts/psse_remote_client.py \
  --base-url http://192.168.6.85:8765 \
  --token "$PSSE_REMOTE_TOKEN" \
  echo --message "hello from gpu"
```

If HTTP fails, test the TCP path:

```bash
nc -vz 192.168.6.85 8765
```

Common outcomes:

- `succeeded`: TCP path is open; debug URL/token/worker logs.
- `timed out`: routing/VPN/firewall issue.
- `connection refused`: host reachable, but the worker is not listening.

## 4. Preflight

Preflight reports four separate facts:

- `psspy_import`: import/version only, not capacity.
- `standalone_psspy_capacity`: expected 50-bus normal Python/Xplore limit.
- `pssecmd36_capacity`: the `50000 BUS POWER SYSTEM SIMULATOR` banner probe.
- `dll_load_results`: DLL load checks for the allowlisted cases.

```bash
python3 Code/scripts/psse_remote_client.py \
  --base-url http://192.168.6.85:8765 \
  --token "$PSSE_REMOTE_TOKEN" \
  --timeout 120 \
  preflight
```

## 5. Job API

Endpoints:

- `POST /jobs` -> returns `job_id`
- `GET /jobs/{job_id}` -> `queued`, `running`, `completed`, or `error`
- `GET /jobs/{job_id}/result` -> compact M1/M2 JSON
- `GET /jobs/{job_id}/artifacts` -> CSV/log/OUTX manifest

Initially allowed:

- `test_cases_v36`
  - `static`
  - `no_disturbance_5s`
  - `pq_target_step`
- `pif6_2026_05_17`
  - `static`
  - `no_disturbance_5s`

No faults, arbitrary control edits, arbitrary files, or arbitrary commands are
accepted in this version.

Submit PIF6 static:

```bash
python3 Code/scripts/psse_remote_client.py \
  --base-url http://192.168.6.85:8765 \
  --token "$PSSE_REMOTE_TOKEN" \
  submit-job --case-id pif6_2026_05_17 --scenario-type static
```

Poll:

```bash
python3 Code/scripts/psse_remote_client.py \
  --base-url http://192.168.6.85:8765 \
  --token "$PSSE_REMOTE_TOKEN" \
  job-status JOB_ID
```

Fetch result:

```bash
python3 Code/scripts/psse_remote_client.py \
  --base-url http://192.168.6.85:8765 \
  --token "$PSSE_REMOTE_TOKEN" \
  job-result JOB_ID
```

Fetch artifact manifest:

```bash
python3 Code/scripts/psse_remote_client.py \
  --base-url http://192.168.6.85:8765 \
  --token "$PSSE_REMOTE_TOKEN" \
  job-artifacts JOB_ID
```

Or submit and wait in one command:

```bash
python3 Code/scripts/psse_remote_client.py \
  --base-url http://192.168.6.85:8765 \
  --token "$PSSE_REMOTE_TOKEN" \
  --timeout 120 \
  submit-job --case-id pif6_2026_05_17 --scenario-type no_disturbance_5s --wait
```

Small case P/Q target reproduction:

```bash
python3 Code/scripts/psse_remote_client.py \
  --base-url http://192.168.6.85:8765 \
  --token "$PSSE_REMOTE_TOKEN" \
  --timeout 120 \
  submit-job --case-id test_cases_v36 --scenario-type pq_target_step --wait
```

## Acceptance Checklist

- GPU reaches `/health`.
- GPU reaches `/echo`.
- Preflight separates `psspy` import from the 50-bus standalone limit and the
  `pssecmd36 -buses 50000` path.
- PIF6 static result reports `bus_count: 786`.
- PIF6 5-second baseline completes and returns POC P/Q/V/frequency channel
  metrics.
- Small case reproduces the existing P/Q target result.
- Concurrent `POST /jobs` requests remain serialized by the worker queue.

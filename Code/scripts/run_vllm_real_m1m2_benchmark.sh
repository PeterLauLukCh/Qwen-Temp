#!/usr/bin/env bash
set -Eeuo pipefail

# Keep user-site packages such as ~/.local/lib/pythonX.Y/site-packages from
# leaking into conda environments. This avoids accidentally importing a vLLM or
# torch wheel built for a newer CUDA runtime than the node driver supports.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

usage() {
  cat <<'EOF'
Launch vLLM, run generated live remote PSS/E M1+M2 testcases, save results,
and stop vLLM.

Run from either the repository root or the Code directory.

Defaults match the current GPU-node setup:
  model: /mindopt/ea120/models/Qwen3.5-27B
  GPUs:  6,7
  cases: real-data-new/generated_real_m1m2_interconnection_cases.json
  count: 100

Required environment:
  PSSE_REMOTE_BASE_URL     e.g. http://71.142.245.200:18765
  PSSE_REMOTE_TOKEN        current Windows-worker bearer token

Common options:
  --model-path PATH
  --cases PATH
  --count N
  --served-model-name NAME
  --benchmark-model NAME
  --gpus LIST
  --tp-size N
  --port PORT
  --output-dir DIR
  --include-raw-results
  --include-messages
  --keep-server
  --vllm-arg ARG                Extra argument forwarded to vLLM. Repeat for flags
                                and values, e.g. --vllm-arg --model-impl --vllm-arg transformers.
  --startup-timeout SECONDS
  --                             Remaining args go to run_real_m1m2_interconnection_benchmark.py

Example:
  bash Code/scripts/run_vllm_real_m1m2_benchmark.sh \
    --model-path /mindopt/ea120/models/Qwen3.5-27B \
    --gpus 6,7 \
    --count 100
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 2
}

count_gpus() {
  local list="$1"
  local count=0
  local item
  IFS=',' read -r -a parts <<<"$list"
  for item in "${parts[@]}"; do
    if [[ -n "${item//[[:space:]]/}" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "$count"
}

safe_name() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9_.-' '_' | sed 's/^_//; s/_$//'
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/mindopt/ea120/models/Qwen3.5-27B}"
CASES_PATH="${CASES_PATH:-${REPO_ROOT}/real-data-new/generated_real_m1m2_interconnection_cases.json}"
COUNT="${COUNT:-100}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
BENCHMARK_MODEL="${BENCHMARK_MODEL:-}"
HOST="${HOST:-0.0.0.0}"
CONNECT_HOST="${CONNECT_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPUS="${CUDA_VISIBLE_DEVICES:-6,7}"
TP_SIZE="${TP_SIZE:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
DTYPE="${DTYPE:-auto}"
REASONING_PARSER="${REASONING_PARSER-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER-qwen3_coder}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/benchmark_results}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
VLLM_BIN="${VLLM_BIN:-vllm}"
KEEP_SERVER=0
INCLUDE_RAW_RESULTS=0
INCLUDE_MESSAGES=0
VLLM_EXTRA_ARGS=()
BENCH_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --model-path)
      MODEL_PATH="${2:?missing value for --model-path}"
      shift 2
      ;;
    --cases)
      CASES_PATH="${2:?missing value for --cases}"
      shift 2
      ;;
    --count|--limit)
      COUNT="${2:?missing value for --count}"
      shift 2
      ;;
    --served-model-name|--model-name)
      SERVED_MODEL_NAME="${2:?missing value for --served-model-name}"
      shift 2
      ;;
    --benchmark-model)
      BENCHMARK_MODEL="${2:?missing value for --benchmark-model}"
      shift 2
      ;;
    --gpus|--cuda-visible-devices)
      GPUS="${2:?missing value for --gpus}"
      shift 2
      ;;
    --tp-size|--tensor-parallel-size)
      TP_SIZE="${2:?missing value for --tp-size}"
      shift 2
      ;;
    --host)
      HOST="${2:?missing value for --host}"
      shift 2
      ;;
    --connect-host)
      CONNECT_HOST="${2:?missing value for --connect-host}"
      shift 2
      ;;
    --port)
      PORT="${2:?missing value for --port}"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="${2:?missing value for --gpu-memory-utilization}"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="${2:?missing value for --max-model-len}"
      shift 2
      ;;
    --max-num-seqs)
      MAX_NUM_SEQS="${2:?missing value for --max-num-seqs}"
      shift 2
      ;;
    --dtype)
      DTYPE="${2:?missing value for --dtype}"
      shift 2
      ;;
    --reasoning-parser)
      REASONING_PARSER="${2-}"
      shift 2
      ;;
    --tool-call-parser)
      TOOL_CALL_PARSER="${2:?missing value for --tool-call-parser}"
      shift 2
      ;;
    --disable-auto-tool-choice)
      ENABLE_AUTO_TOOL_CHOICE=0
      shift
      ;;
    --trust-remote-code)
      TRUST_REMOTE_CODE=1
      shift
      ;;
    --no-trust-remote-code)
      TRUST_REMOTE_CODE=0
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --startup-timeout)
      STARTUP_TIMEOUT="${2:?missing value for --startup-timeout}"
      shift 2
      ;;
    --poll-interval)
      POLL_INTERVAL="${2:?missing value for --poll-interval}"
      shift 2
      ;;
    --include-raw-results)
      INCLUDE_RAW_RESULTS=1
      shift
      ;;
    --include-messages)
      INCLUDE_MESSAGES=1
      shift
      ;;
    --vllm-arg)
      VLLM_EXTRA_ARGS+=("${2:?missing value for --vllm-arg}")
      shift 2
      ;;
    --keep-server)
      KEEP_SERVER=1
      shift
      ;;
    --)
      shift
      BENCH_EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "${PSSE_REMOTE_BASE_URL:-}" ]] || die "Set PSSE_REMOTE_BASE_URL first."
[[ -n "${PSSE_REMOTE_TOKEN:-}" ]] || die "Set PSSE_REMOTE_TOKEN first."
[[ -d "$MODEL_PATH" ]] || die "model path does not exist: $MODEL_PATH"
[[ -f "$CASES_PATH" ]] || die "cases file does not exist: $CASES_PATH"
[[ "$COUNT" =~ ^[0-9]+$ ]] || die "--count must be a positive integer"
[[ "$COUNT" -gt 0 ]] || die "--count must be positive"

if [[ -z "$SERVED_MODEL_NAME" ]]; then
  SERVED_MODEL_NAME="$(basename "$MODEL_PATH")"
fi
if [[ -z "$BENCHMARK_MODEL" ]]; then
  BENCHMARK_MODEL="$SERVED_MODEL_NAME"
fi
if [[ -z "$TP_SIZE" ]]; then
  TP_SIZE="$(count_gpus "$GPUS")"
fi

RUN_NAME="$(safe_name "${SERVED_MODEL_NAME}_real_m1m2_${COUNT}")"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}_${RUN_NAME}_port${PORT}"
VLLM_LOG="${RUN_DIR}/vllm.log"
BENCH_STDERR="${RUN_DIR}/real_m1m2_benchmark.stderr.log"
RESULT_JSON="${RUN_DIR}/real_m1m2_result.json"
META_JSON="${RUN_DIR}/run_metadata.json"
mkdir -p "$RUN_DIR"

VLLM_ARGS=(
  "$VLLM_BIN" serve "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --dtype "$DTYPE"
)
if [[ -n "$REASONING_PARSER" ]]; then
  VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
fi
if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
  VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi
VLLM_ARGS+=("${VLLM_EXTRA_ARGS[@]}")

BENCH_ARGS=(
  python3 "$CODE_DIR/scripts/run_real_m1m2_interconnection_benchmark.py"
  --cases "$CASES_PATH"
  --limit "$COUNT"
  --base-url "http://${CONNECT_HOST}:${PORT}/v1"
  --model "$BENCHMARK_MODEL"
  --output "$RESULT_JSON"
)
if [[ "$INCLUDE_RAW_RESULTS" != "1" ]]; then
  BENCH_ARGS+=(--no-raw-results)
fi
if [[ "$INCLUDE_MESSAGES" == "1" ]]; then
  BENCH_ARGS+=(--include-messages)
fi
BENCH_ARGS+=("${BENCH_EXTRA_ARGS[@]}")

VLLM_PID=""
VLLM_PGID=""

cleanup() {
  local exit_code=$?
  if [[ "$KEEP_SERVER" == "1" ]]; then
    if [[ -n "${VLLM_PID:-}" ]]; then
      log "Keeping vLLM running with PID ${VLLM_PID}"
    fi
    return "$exit_code"
  fi
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    log "Stopping vLLM PID ${VLLM_PID}"
    if [[ -n "${VLLM_PGID:-}" ]]; then
      kill -TERM "-${VLLM_PGID}" 2>/dev/null || true
    else
      kill -TERM "$VLLM_PID" 2>/dev/null || true
    fi
    for _ in $(seq 1 45); do
      if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$VLLM_PID" 2>/dev/null; then
      log "vLLM did not stop after SIGTERM; sending SIGKILL"
      if [[ -n "${VLLM_PGID:-}" ]]; then
        kill -KILL "-${VLLM_PGID}" 2>/dev/null || true
      else
        kill -KILL "$VLLM_PID" 2>/dev/null || true
      fi
    fi
  fi
  return "$exit_code"
}
trap cleanup EXIT

log "Run directory: ${RUN_DIR}"
log "Model path: ${MODEL_PATH}"
log "Served model name: ${SERVED_MODEL_NAME}"
log "Benchmark model: ${BENCHMARK_MODEL}"
log "Cases: ${CASES_PATH}; count: ${COUNT}"
log "GPUs: ${GPUS}; TP size: ${TP_SIZE}; port: ${PORT}"
log "Remote PSS/E endpoint: ${PSSE_REMOTE_BASE_URL}"
log "PYTHONNOUSERSITE: ${PYTHONNOUSERSITE}"

log "Checking Python/vLLM/CUDA environment..."
(
  cd "$REPO_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPUS"
  python3 - <<'PY'
import importlib.metadata
import shutil
import site
import sys

print("python:", sys.executable)
print("user_site_enabled:", site.ENABLE_USER_SITE)
print("vllm_executable:", shutil.which("vllm"))
try:
    import torch
    print("torch:", torch.__version__)
    print("torch_cuda:", torch.version.cuda)
    available = torch.cuda.is_available()
    print("cuda_available:", available)
    if available:
        print("cuda_device_0:", torch.cuda.get_device_name(0))
    else:
        raise SystemExit(
            "CUDA is not available to this Python environment. Check torch CUDA wheel "
            "compatibility with the NVIDIA driver."
        )
except Exception as exc:
    raise SystemExit(f"CUDA/vLLM preflight failed: {type(exc).__name__}: {exc}") from exc
try:
    print("vllm:", importlib.metadata.version("vllm"))
except importlib.metadata.PackageNotFoundError:
    raise SystemExit("CUDA/vLLM preflight failed: vllm is not installed in this environment.")
PY
)

export MODEL_PATH_VALUE="$MODEL_PATH"
export SERVED_MODEL_NAME_VALUE="$SERVED_MODEL_NAME"
export BENCHMARK_MODEL_VALUE="$BENCHMARK_MODEL"
export CASES_PATH_VALUE="$CASES_PATH"
export COUNT_VALUE="$COUNT"
export GPUS_VALUE="$GPUS"
export TP_SIZE_VALUE="$TP_SIZE"
export PORT_VALUE="$PORT"
python3 - "$META_JSON" <<'PY'
import json
import os
import sys

payload = {
    "model_path": os.environ["MODEL_PATH_VALUE"],
    "served_model_name": os.environ["SERVED_MODEL_NAME_VALUE"],
    "benchmark_model": os.environ["BENCHMARK_MODEL_VALUE"],
    "cases_path": os.environ["CASES_PATH_VALUE"],
    "count": int(os.environ["COUNT_VALUE"]),
    "gpus": os.environ["GPUS_VALUE"],
    "tp_size": int(os.environ["TP_SIZE_VALUE"]),
    "port": int(os.environ["PORT_VALUE"]),
    "psse_remote_base_url": os.environ.get("PSSE_REMOTE_BASE_URL", ""),
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

log "Starting vLLM..."
(
  cd "$REPO_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPUS"
  exec "${VLLM_ARGS[@]}"
) >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
VLLM_PGID="$(ps -o pgid= "$VLLM_PID" 2>/dev/null | tr -d ' ' || true)"
log "vLLM PID: ${VLLM_PID}; log: ${VLLM_LOG}"

log "Waiting for vLLM readiness..."
python3 - "$CONNECT_HOST" "$PORT" "$STARTUP_TIMEOUT" "$POLL_INTERVAL" "$VLLM_LOG" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

host, port, timeout_s, poll_s, log_path = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
url = f"http://{host}:{port}/v1/models"
deadline = time.time() + timeout_s
last_error = ""
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("data"):
            print(json.dumps({"ok": True, "url": url, "models": payload.get("data")}, indent=2))
            raise SystemExit(0)
    except Exception as exc:
        last_error = repr(exc)
    time.sleep(poll_s)
print(json.dumps({"ok": False, "url": url, "last_error": last_error, "log_path": log_path}, indent=2))
raise SystemExit(1)
PY

log "Running generated real M1+M2 benchmark..."
set +e
(
  cd "$REPO_ROOT"
  export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
  "${BENCH_ARGS[@]}"
) 2>"$BENCH_STDERR"
BENCH_EXIT=$?
set -e

if [[ "$BENCH_EXIT" -ne 0 ]]; then
  log "Benchmark failed with exit ${BENCH_EXIT}. Stderr tail:"
  tail -120 "$BENCH_STDERR" >&2 || true
  log "vLLM log tail:"
  tail -120 "$VLLM_LOG" >&2 || true
  exit "$BENCH_EXIT"
fi

log "Benchmark complete."
log "Result JSON: ${RESULT_JSON}"
log "Benchmark stderr: ${BENCH_STDERR}"
log "vLLM log: ${VLLM_LOG}"

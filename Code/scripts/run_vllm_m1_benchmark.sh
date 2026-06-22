#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Launch vLLM, run the M1 benchmark, save results, and stop vLLM.

Run from either the repository root or the Code directory.

Required:
  --model-path PATH              Local model directory, e.g. /nas/models/Qwen3.5-27B

Common options:
  --served-model-name NAME       Name exposed by /v1/models. Defaults to basename(model-path).
  --benchmark-model NAME         Model name passed to run_m1_benchmark.py. Defaults to served name.
  --gpus LIST                    CUDA_VISIBLE_DEVICES list, e.g. 0,1,2,3. Default: CUDA_VISIBLE_DEVICES or 0.
  --tp-size N                    Tensor parallel size. Default: number of GPUs in --gpus.
  --port PORT                    vLLM port. Default: 8000.
  --host HOST                    vLLM bind host. Default: 0.0.0.0.
  --connect-host HOST            Host used by the local benchmark client. Default: 127.0.0.1.
  --output-dir DIR               Directory for logs/results. Default: Code/benchmark_results.

vLLM options:
  --cuda-home PATH               Optional CUDA toolkit path. Auto-uses /nas/peter.c/cuda-12.8.1 if present.
  --gpu-memory-utilization X     Default: 0.90.
  --max-model-len N              Default: 16384.
  --max-num-seqs N               Default: 4.
  --dtype DTYPE                  Default: auto.
  --moe-backend BACKEND          Default: triton. Use empty string to omit.
  --reasoning-parser PARSER      Default: qwen3. Use empty string to omit.
  --tool-call-parser PARSER      Default: qwen3_coder.
  --disable-auto-tool-choice     Do not pass --enable-auto-tool-choice.
  --trust-remote-code / --no-trust-remote-code
                                  Default: trust remote code.

Benchmark options:
  --include-raw-results          Keep full agent/oracle JSON instead of compact summaries.
  --include-messages             Include full LLM messages in benchmark output.
  --startup-timeout SECONDS      vLLM readiness timeout. Default: 900.
  --poll-interval SECONDS        Readiness polling interval. Default: 5.
  --keep-server                  Do not stop vLLM after the benchmark.
  --                             Remaining arguments are forwarded to run_m1_benchmark.py.

Examples:
  bash scripts/run_vllm_m1_benchmark.sh \
    --model-path /nas/models/Qwen3.5-27B \
    --served-model-name qwen35-27b \
    --gpus 0,1,2,3 \
    --port 8000

  bash scripts/run_vllm_m1_benchmark.sh \
    --model-path /nas/models/Qwen3.6-35B-A3B \
    --served-model-name qwen36-35b-a3b \
    --gpus 4,5,6,7 \
    --port 8001 \
    -- --tag cia
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

MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
BENCHMARK_MODEL="${BENCHMARK_MODEL:-}"
HOST="${HOST:-0.0.0.0}"
CONNECT_HOST="${CONNECT_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
TP_SIZE="${TP_SIZE:-}"
CUDA_HOME_ARG="${CUDA_HOME:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
DTYPE="${DTYPE:-auto}"
MOE_BACKEND="${MOE_BACKEND:-triton}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/benchmark_results}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
KEEP_SERVER=0
INCLUDE_RAW_RESULTS=0
INCLUDE_MESSAGES=0
VLLM_BIN="${VLLM_BIN:-vllm}"
M1_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="${2:?missing value for --model-path}"
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
    --cuda-home)
      CUDA_HOME_ARG="${2:?missing value for --cuda-home}"
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
    --moe-backend)
      MOE_BACKEND="${2-}"
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
    --keep-server)
      KEEP_SERVER=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      M1_EXTRA_ARGS=("$@")
      break
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  if [[ -t 0 ]]; then
    read -r -p "Model path: " MODEL_PATH
  else
    die "--model-path is required"
  fi
fi
[[ -d "$MODEL_PATH" ]] || die "model path does not exist: $MODEL_PATH"
[[ -n "$GPUS" ]] || die "--gpus must not be empty"

if [[ -z "$SERVED_MODEL_NAME" ]]; then
  SERVED_MODEL_NAME="$(basename "$MODEL_PATH")"
fi
if [[ -z "$BENCHMARK_MODEL" ]]; then
  BENCHMARK_MODEL="$SERVED_MODEL_NAME"
fi
if [[ -z "$TP_SIZE" ]]; then
  TP_SIZE="$(count_gpus "$GPUS")"
fi
[[ "$TP_SIZE" =~ ^[0-9]+$ ]] && [[ "$TP_SIZE" -gt 0 ]] || die "--tp-size must be a positive integer"
[[ "$PORT" =~ ^[0-9]+$ ]] && [[ "$PORT" -ge 1 ]] && [[ "$PORT" -le 65535 ]] || die "--port must be 1-65535"

if [[ -z "$CUDA_HOME_ARG" && -d /nas/peter.c/cuda-12.8.1 ]]; then
  CUDA_HOME_ARG="/nas/peter.c/cuda-12.8.1"
fi
if [[ -n "$CUDA_HOME_ARG" ]]; then
  export CUDA_HOME="$CUDA_HOME_ARG"
  export CUDA_PATH="$CUDA_HOME_ARG"
  export PATH="$CUDA_HOME_ARG/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME_ARG/lib64:${LD_LIBRARY_PATH:-}"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPUS"

TIMESTAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
RUN_NAME="$(safe_name "${SERVED_MODEL_NAME}")"
[[ -n "$RUN_NAME" ]] || RUN_NAME="model"
RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}_${RUN_NAME}_port${PORT}"
VLLM_LOG="${RUN_DIR}/vllm.log"
BENCH_STDERR="${RUN_DIR}/m1_benchmark.stderr.log"
RESULT_JSON="${RUN_DIR}/m1_result.json"
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
if [[ -n "$MOE_BACKEND" ]]; then
  VLLM_ARGS+=(--moe-backend "$MOE_BACKEND")
fi
if [[ -n "$REASONING_PARSER" ]]; then
  VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
fi
if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
  VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

M1_ARGS=(
  python3 scripts/run_m1_benchmark.py
  --base-url "http://${CONNECT_HOST}:${PORT}/v1"
  --model "$BENCHMARK_MODEL"
)
if [[ "$INCLUDE_RAW_RESULTS" != "1" ]]; then
  M1_ARGS+=(--no-raw-results)
fi
if [[ "$INCLUDE_MESSAGES" == "1" ]]; then
  M1_ARGS+=(--include-messages)
fi
M1_ARGS+=("${M1_EXTRA_ARGS[@]}")

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
trap cleanup EXIT INT TERM

wait_for_vllm() {
  local url="http://${CONNECT_HOST}:${PORT}/v1/models"
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  log "Waiting for vLLM readiness at ${url}"
  while (( SECONDS < deadline )); do
    if python3 - "$url" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data")
    if isinstance(data, list):
        raise SystemExit(0)
except Exception:
    pass
raise SystemExit(1)
PY
    then
      log "vLLM is ready"
      return 0
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      log "vLLM exited before readiness. Last log lines:"
      tail -120 "$VLLM_LOG" >&2 || true
      return 1
    fi
    sleep "$POLL_INTERVAL"
  done
  log "Timed out waiting for vLLM. Last log lines:"
  tail -120 "$VLLM_LOG" >&2 || true
  return 1
}

log "Run directory: ${RUN_DIR}"
log "Model path: ${MODEL_PATH}"
log "Served model name: ${SERVED_MODEL_NAME}"
log "Benchmark model: ${BENCHMARK_MODEL}"
log "GPUs: ${CUDA_VISIBLE_DEVICES}; TP size: ${TP_SIZE}; port: ${PORT}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  log "CUDA_HOME: ${CUDA_HOME}"
fi

if command -v setsid >/dev/null 2>&1; then
  setsid "${VLLM_ARGS[@]}" >"$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  VLLM_PGID=$VLLM_PID
else
  "${VLLM_ARGS[@]}" >"$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
fi
log "Started vLLM PID ${VLLM_PID}; log: ${VLLM_LOG}"

wait_for_vllm

log "Running M1 benchmark; result: ${RESULT_JSON}"
set +e
(
  cd "$CODE_DIR"
  "${M1_ARGS[@]}"
) >"$RESULT_JSON" 2>"$BENCH_STDERR"
BENCH_EXIT=$?
set -e

export RUN_META_PATH="$META_JSON"
export RUN_DIR_PATH="$RUN_DIR"
export MODEL_PATH_VALUE="$MODEL_PATH"
export SERVED_MODEL_NAME_VALUE="$SERVED_MODEL_NAME"
export BENCHMARK_MODEL_VALUE="$BENCHMARK_MODEL"
export HOST_VALUE="$HOST"
export CONNECT_HOST_VALUE="$CONNECT_HOST"
export PORT_VALUE="$PORT"
export GPUS_VALUE="$GPUS"
export TP_SIZE_VALUE="$TP_SIZE"
export GPU_MEMORY_UTILIZATION_VALUE="$GPU_MEMORY_UTILIZATION"
export MAX_MODEL_LEN_VALUE="$MAX_MODEL_LEN"
export MAX_NUM_SEQS_VALUE="$MAX_NUM_SEQS"
export DTYPE_VALUE="$DTYPE"
export MOE_BACKEND_VALUE="$MOE_BACKEND"
export RESULT_JSON_PATH="$RESULT_JSON"
export BENCH_STDERR_PATH="$BENCH_STDERR"
export VLLM_LOG_PATH="$VLLM_LOG"
export BENCH_EXIT_VALUE="$BENCH_EXIT"

python3 - <<'PY'
import json
import os
import time

result = None
result_path = os.environ["RESULT_JSON_PATH"]
try:
    with open(result_path, "r", encoding="utf-8") as handle:
        result = json.load(handle)
except Exception as exc:
    result = {"parse_error": str(exc)}

suite = result.get("suite", {}) if isinstance(result, dict) else {}
metadata = {
    "created_unix": int(time.time()),
    "run_dir": os.environ["RUN_DIR_PATH"],
    "model_path": os.environ["MODEL_PATH_VALUE"],
    "served_model_name": os.environ["SERVED_MODEL_NAME_VALUE"],
    "benchmark_model": os.environ["BENCHMARK_MODEL_VALUE"],
    "host": os.environ["HOST_VALUE"],
    "connect_host": os.environ["CONNECT_HOST_VALUE"],
    "port": int(os.environ["PORT_VALUE"]),
    "gpus": os.environ["GPUS_VALUE"],
    "tensor_parallel_size": int(os.environ["TP_SIZE_VALUE"]),
    "gpu_memory_utilization": os.environ["GPU_MEMORY_UTILIZATION_VALUE"],
    "max_model_len": os.environ["MAX_MODEL_LEN_VALUE"],
    "max_num_seqs": os.environ["MAX_NUM_SEQS_VALUE"],
    "dtype": os.environ["DTYPE_VALUE"],
    "moe_backend": os.environ["MOE_BACKEND_VALUE"],
    "benchmark_exit_code": int(os.environ["BENCH_EXIT_VALUE"]),
    "benchmark_ok": bool(result.get("ok")) if isinstance(result, dict) else False,
    "benchmark_total": suite.get("total"),
    "benchmark_passed": suite.get("passed"),
    "benchmark_failed": suite.get("failed"),
    "benchmark_duration_s": suite.get("duration_s"),
    "files": {
        "result_json": result_path,
        "benchmark_stderr": os.environ["BENCH_STDERR_PATH"],
        "vllm_log": os.environ["VLLM_LOG_PATH"],
    },
}
with open(os.environ["RUN_META_PATH"], "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

log "Benchmark exit code: ${BENCH_EXIT}"
log "Saved metadata: ${META_JSON}"
if [[ "$BENCH_EXIT" -ne 0 ]]; then
  log "Benchmark failed or had failed cases. stderr:"
  tail -80 "$BENCH_STDERR" >&2 || true
fi
exit "$BENCH_EXIT"

#!/usr/bin/env bash
# Reproducible bare-metal Ray Serve benchmark for Runpod/A100.
#
# Runs Ray Serve REST and Ray Serve gRPC directly as Python processes:
#   - no Docker Compose
#   - no LitServe baseline
#   - Locust --csv-full-history enabled for confidence-band plots
set -Eeuo pipefail
shopt -s nullglob

export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"

ENV_FILE="${ENV_FILE:-.env.ray-a100-4w.example}"
if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    . "${ENV_FILE}"
    set +a
else
    echo "Missing env file: ${ENV_FILE}" >&2
    exit 1
fi

UV_PYTHON="${UV_PYTHON:-3.13}"
MODEL_ID="${MODEL_ID:-hivetrace/gliner-guard-uniencoder}"
MODEL_SHORT="${MODEL_SHORT:-uni}"
SCHEMA_MODE="${SCHEMA_MODE:-minimal}"
NUM_REPLICAS="${NUM_REPLICAS:-4}"
NUM_GPUS_PER_REPLICA="${NUM_GPUS_PER_REPLICA:-0.25}"
NUM_CPUS_PER_REPLICA="${NUM_CPUS_PER_REPLICA:-3}"
MAX_ONGOING_REQUESTS="${MAX_ONGOING_REQUESTS:-256}"
MAX_CONCURRENT_BATCHES="${MAX_CONCURRENT_BATCHES:-1}"
BATCH_CONFIGS="${BATCH_CONFIGS:-16:0.05,32:0.05,64:0.05}"
REPEATS="${REPEATS:-3}"
DURATION="${DURATION:-${LOCUST_RUN_TIME:-15m}}"
USERS="${USERS:-${LOCUST_USERS:-100}}"
SPAWN_RATE="${SPAWN_RATE:-${LOCUST_SPAWN_RATE:-1}}"
USER_THROUGHPUT="${USER_THROUGHPUT:-${LOCUST_USER_THROUGHPUT:-5}}"
DATASET="${DATASET:-prompts}"
WARMUP_REQS="${WARMUP_REQS:-50}"
PROTOCOLS="${PROTOCOLS:-rest,grpc}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-1800}"
BETWEEN_RUN_SLEEP_SECONDS="${BETWEEN_RUN_SLEEP_SECONDS:-5}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-2000000000}"
RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
RAY_WORK_DIR="${RAY_WORK_DIR:-/tmp/gliner-guard-ray-workdir}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
DEFAULT_RESULT_DIR="results-runpod/a100-baremetal-$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -d /workspace ]]; then
    DEFAULT_RESULT_DIR="/workspace/gliner-guard-results/a100-baremetal-$(date -u +%Y%m%dT%H%M%SZ)"
fi
RESULT_DIR="${RESULT_DIR:-${DEFAULT_RESULT_DIR}}"

if [[ "${RESULT_DIR}" != /* ]]; then
    RESULT_DIR="${REPO_DIR}/${RESULT_DIR}"
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
    BATCH_CONFIGS="${SMOKE_BATCH_CONFIGS:-16:0.05}"
    REPEATS="${SMOKE_REPEATS:-1}"
    DURATION="${SMOKE_DURATION:-30s}"
    USERS="${SMOKE_USERS:-2}"
    SPAWN_RATE="${SMOKE_SPAWN_RATE:-1}"
    USER_THROUGHPUT="${SMOKE_USER_THROUGHPUT:-1}"
    WARMUP_REQS="${SMOKE_WARMUP_REQS:-2}"
fi

mkdir -p "${RESULT_DIR}/logs" "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

SERVER_PID=""
GPU_PID=""

duration_seconds() {
    local value="$1"
    case "${value}" in
        *s) echo "${value%s}" ;;
        *m) echo "$(( ${value%m} * 60 ))" ;;
        *h) echo "$(( ${value%h} * 3600 ))" ;;
        *) echo "${value}" ;;
    esac
}

stop_gpu_metrics() {
    if [[ -n "${GPU_PID:-}" ]] && kill -0 "${GPU_PID}" 2>/dev/null; then
        kill "${GPU_PID}" 2>/dev/null || true
        wait "${GPU_PID}" 2>/dev/null || true
    fi
    GPU_PID=""
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${SERVER_PID}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill -9 "${SERVER_PID}" 2>/dev/null || true
        fi
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""

    (
        cd "${REPO_DIR}/ray-serve"
        UV_PYTHON="${UV_PYTHON}" uv run ray stop --force >/dev/null 2>&1 || true
    )
    if [[ -x "${REPO_DIR}/ray-serve/.venv/bin/ray" ]]; then
        "${REPO_DIR}/ray-serve/.venv/bin/ray" stop --force >/dev/null 2>&1 || true
    fi
    pkill -f "python .*serve_app.py" >/dev/null 2>&1 || true
    pkill -f "python .*serve_app_grpc.py" >/dev/null 2>&1 || true
    sleep "${BETWEEN_RUN_SLEEP_SECONDS}"
}

cleanup() {
    stop_gpu_metrics
    stop_server
}
trap cleanup EXIT INT TERM

record_environment() {
    {
        echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "repo_dir=${REPO_DIR}"
        echo "env_file=${ENV_FILE}"
        echo "uv_python=${UV_PYTHON}"
        echo "model_id=${MODEL_ID}"
        echo "schema_mode=${SCHEMA_MODE}"
        echo "num_replicas=${NUM_REPLICAS}"
        echo "num_gpus_per_replica=${NUM_GPUS_PER_REPLICA}"
        echo "num_cpus_per_replica=${NUM_CPUS_PER_REPLICA}"
        echo "max_ongoing_requests=${MAX_ONGOING_REQUESTS}"
        echo "max_concurrent_batches=${MAX_CONCURRENT_BATCHES}"
        echo "batch_configs=${BATCH_CONFIGS}"
        echo "protocols=${PROTOCOLS}"
        echo "repeats=${REPEATS}"
        echo "duration=${DURATION}"
        echo "users=${USERS}"
        echo "spawn_rate=${SPAWN_RATE}"
        echo "user_throughput=${USER_THROUGHPUT}"
        echo "dataset=${DATASET}"
        echo "warmup_reqs=${WARMUP_REQS}"
        echo "ray_object_store_memory=${RAY_OBJECT_STORE_MEMORY}"
        echo "ray_work_dir=${RAY_WORK_DIR}"
        echo ""
        echo "uv:"
        uv --version || true
        echo ""
        echo "python:"
        UV_PYTHON="${UV_PYTHON}" uv run python --version || true
        echo ""
        echo "git:"
        git rev-parse --short HEAD 2>/dev/null || true
        echo ""
        echo "gpu:"
        nvidia-smi || true
    } > "${RESULT_DIR}/environment.txt"
}

ensure_grpc_stubs() {
    local site_packages

    (
        cd "${REPO_DIR}/ray-serve"
        if [[ ! -f gliner_guard_pb2.py || ! -f gliner_guard_pb2_grpc.py ]]; then
            UV_PYTHON="${UV_PYTHON}" uv run python -m grpc_tools.protoc \
                -I=proto --python_out=. --grpc_python_out=. \
                proto/gliner_guard.proto
        fi

        site_packages="$(UV_PYTHON="${UV_PYTHON}" uv run python -c 'import site; print(site.getsitepackages()[0])')"
        cp -f gliner_guard_pb2.py gliner_guard_pb2_grpc.py "${site_packages}/"
    )
}

wait_ready_rest() {
    local log_file="$1"
    local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
    local attempts=0

    echo "  Waiting for REST server on :8000..."
    while (( SECONDS < deadline )); do
        if curl -sf -o /dev/null http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d '{"text":"healthcheck"}'; then
            echo "  REST server ready."
            return 0
        fi
        attempts=$((attempts + 1))
        if (( attempts % 12 == 0 )); then
            echo "  Still waiting; latest server log:"
            tail -40 "${log_file}" || true
        fi
        sleep 5
    done

    echo "  TIMEOUT waiting for REST server" >&2
    tail -120 "${log_file}" >&2 || true
    return 1
}

wait_ready_grpc() {
    local log_file="$1"
    local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
    local attempts=0

    echo "  Waiting for gRPC server on :9000..."
    while (( SECONDS < deadline )); do
        if (
            cd "${REPO_DIR}/ray-serve"
            PYTHONPATH="${REPO_DIR}/ray-serve:${PYTHONPATH:-}" \
                UV_PYTHON="${UV_PYTHON}" uv run python - <<'PY'
import grpc
import gliner_guard_pb2
import gliner_guard_pb2_grpc

channel = grpc.insecure_channel("localhost:9000")
stub = gliner_guard_pb2_grpc.GLiNERGuardServiceStub(channel)
stub.Predict(gliner_guard_pb2.PredictRequest(text="healthcheck"), timeout=5)
PY
        ) >/dev/null 2>&1; then
            echo "  gRPC server ready."
            return 0
        fi
        attempts=$((attempts + 1))
        if (( attempts % 12 == 0 )); then
            echo "  Still waiting; latest server log:"
            tail -40 "${log_file}" || true
        fi
        sleep 5
    done

    echo "  TIMEOUT waiting for gRPC server" >&2
    tail -120 "${log_file}" >&2 || true
    return 1
}

warmup_rest() {
    local pids=()
    local pid

    echo "  Warmup REST: ${WARMUP_REQS} requests..."
    for i in $(seq 1 "${WARMUP_REQS}"); do
        curl -sf -o /dev/null http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d '{"text":"warmup request '"${i}"'"}' &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
}

warmup_grpc() {
    echo "  Warmup gRPC: ${WARMUP_REQS} requests..."
    (
        cd "${REPO_DIR}/ray-serve"
        PYTHONPATH="${REPO_DIR}/ray-serve:${PYTHONPATH:-}" \
            UV_PYTHON="${UV_PYTHON}" WARMUP_REQS="${WARMUP_REQS}" uv run python - <<'PY'
import os

import grpc
import gliner_guard_pb2
import gliner_guard_pb2_grpc

channel = grpc.insecure_channel("localhost:9000")
stub = gliner_guard_pb2_grpc.GLiNERGuardServiceStub(channel)
for i in range(int(os.environ["WARMUP_REQS"])):
    stub.Predict(gliner_guard_pb2.PredictRequest(text=f"warmup {i}"), timeout=30)
PY
    )
}

start_server() {
    local protocol="$1"
    local batch_size="$2"
    local timeout="$3"
    local prefix="$4"
    local app="serve_app.py"
    local server_log="${RESULT_DIR}/logs/${prefix}-server.log"

    if [[ "${protocol}" == "grpc" ]]; then
        app="serve_app_grpc.py"
    fi

    stop_server

    echo "  Starting Ray Serve ${protocol} server..."
    mkdir -p "${RAY_WORK_DIR}"
    (
        cd "${RAY_WORK_DIR}"
        export PYTHONPATH="${REPO_DIR}/ray-serve:${PYTHONPATH:-}"
        export UV_PYTHON
        export MODEL_ID
        export SCHEMA_MODE
        export NUM_REPLICAS
        export NUM_GPUS_PER_REPLICA
        export NUM_CPUS_PER_REPLICA
        export MAX_ONGOING_REQUESTS
        export MAX_CONCURRENT_BATCHES
        export RAY_OBJECT_STORE_MEMORY
        export RAY_memory_monitor_refresh_ms
        export HF_HOME
        export HF_HUB_CACHE
        export TRANSFORMERS_CACHE
        export MAX_BATCH_SIZE="${batch_size}"
        export BATCH_WAIT_TIMEOUT="${timeout}"
        exec "${REPO_DIR}/ray-serve/.venv/bin/python" "${REPO_DIR}/ray-serve/${app}"
    ) > "${server_log}" 2>&1 &
    SERVER_PID=$!
    echo "${SERVER_PID}" > "${RESULT_DIR}/logs/${prefix}-server.pid"

    if [[ "${protocol}" == "grpc" ]]; then
        wait_ready_grpc "${server_log}"
    else
        wait_ready_rest "${server_log}"
    fi
}

run_locust() {
    local protocol="$1"
    local prefix="$2"
    local locust_file="test-gliner.py"
    local host="http://localhost:8000"
    local duration_secs
    local locust_log="${RESULT_DIR}/logs/${prefix}-locust.log"
    local gpu_log="${RESULT_DIR}/logs/${prefix}-gpu.log"
    local gpu_csv="${RESULT_DIR}/gpu-${prefix}.csv"
    local locust_status=0

    if [[ "${protocol}" == "grpc" ]]; then
        locust_file="test-gliner-grpc.py"
        host="localhost:9000"
    fi

    duration_secs="$(duration_seconds "${DURATION}")"
    bash scripts/collect_gpu_metrics.sh "${gpu_csv}" "${duration_secs}" > "${gpu_log}" 2>&1 &
    GPU_PID=$!

    (
        cd test-script
        DATASET="${DATASET}" \
            GLINER_HOST="${host}" \
            LOCUST_USER_THROUGHPUT="${USER_THROUGHPUT}" \
            UV_PYTHON="${UV_PYTHON}" \
            uv run locust -f "${locust_file}" \
            --headless -u "${USERS}" -r "${SPAWN_RATE}" --run-time "${DURATION}" \
            --csv-full-history \
            --csv="${RESULT_DIR}/${prefix}" \
            --html="${RESULT_DIR}/${prefix}.html"
    ) > "${locust_log}" 2>&1 || locust_status=$?

    stop_gpu_metrics
    tail -30 "${locust_log}" || true
    return "${locust_status}"
}

print_result() {
    local prefix="$1"
    local stats_file="${RESULT_DIR}/${prefix}_stats.csv"
    if [[ -f "${stats_file}" ]]; then
        grep "Aggregated" "${stats_file}" \
            | tail -1 \
            | awk -F',' '{printf "    RPS=%.2f  P50=%sms  P95=%sms  Failures=%s\n", $10, $6, $8, $4}' \
            || true
    else
        echo "    missing stats: ${stats_file}"
    fi
}

append_summary() {
    local prefix="$1"
    local protocol="$2"
    local batch_size="$3"
    local run_num="$4"
    local stats_file="${RESULT_DIR}/${prefix}_stats.csv"
    local summary_file="${RESULT_DIR}/summary.csv"

    if [[ ! -f "${stats_file}" ]]; then
        return 0
    fi

    grep "Aggregated" "${stats_file}" \
        | tail -1 \
        | awk -F',' \
            -v protocol="${protocol}" \
            -v batch_size="${batch_size}" \
            -v run_num="${run_num}" \
            -v prefix="${prefix}" \
            '{printf "%s,%s,%s,%s,%.6f,%s,%s,%s\n", prefix, protocol, batch_size, run_num, $10, $6, $8, $4}' \
        >> "${summary_file}" || true
}

run_one() {
    local protocol="$1"
    local batch_size="$2"
    local timeout="$3"
    local run_num="$4"
    local prefix="ray-${protocol}-B${batch_size}-${MODEL_SHORT}-${DATASET}-run${run_num}"

    echo ""
    echo "=========================================="
    echo "  Ray Serve ${protocol}: ${prefix}"
    echo "  replicas=${NUM_REPLICAS} gpu_per_replica=${NUM_GPUS_PER_REPLICA} batch=${batch_size} timeout=${timeout}"
    echo "=========================================="

    start_server "${protocol}" "${batch_size}" "${timeout}" "${prefix}"
    if [[ "${protocol}" == "grpc" ]]; then
        warmup_grpc
    else
        warmup_rest
    fi
    run_locust "${protocol}" "${prefix}"
    print_result "${prefix}"
    append_summary "${prefix}" "${protocol}" "${batch_size}" "${run_num}"
    stop_server
}

generate_plots() {
    local cfg batch_size protocol pattern output
    IFS=',' read -r -a cfgs <<< "${BATCH_CONFIGS}"
    IFS=',' read -r -a protocols <<< "${PROTOCOLS}"

    echo ""
    echo "Generating history plots..."
    for cfg in "${cfgs[@]}"; do
        IFS=':' read -r batch_size _ <<< "${cfg}"
        for protocol in "${protocols[@]}"; do
            pattern="${RESULT_DIR}/ray-${protocol}-B${batch_size}-${MODEL_SHORT}-${DATASET}-run*_stats_history.csv"
            output="${RESULT_DIR}/ray-${protocol}-B${batch_size}-${MODEL_SHORT}-${DATASET}-history.png"
            if compgen -G "${pattern}" >/dev/null; then
                MPLBACKEND=Agg UV_PYTHON="${UV_PYTHON}" \
                    uv run --with pandas --with matplotlib --with seaborn \
                    python scripts/plot_locust_history.py \
                    --glob "${pattern}" \
                    --output "${output}" \
                    --max-users "${USERS}" \
                    --latency-percentile "95%" \
                    --throughput-scale 1000 \
                    || true
            fi
        done
    done
}

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Run scripts/setup-a100-baremetal.sh first." >&2
    exit 1
fi
if [[ ! -x "${REPO_DIR}/ray-serve/.venv/bin/python" ]]; then
    echo "Ray Serve venv is missing. Run scripts/setup-a100-baremetal.sh first." >&2
    exit 1
fi

ensure_grpc_stubs
record_environment
echo "prefix,protocol,batch_size,run,rps,p50_ms,p95_ms,failures" > "${RESULT_DIR}/summary.csv"

IFS=',' read -r -a configs <<< "${BATCH_CONFIGS}"
IFS=',' read -r -a protocols <<< "${PROTOCOLS}"
duration_secs="$(duration_seconds "${DURATION}")"
total_runs=$(( ${#configs[@]} * REPEATS * ${#protocols[@]} ))

echo "======================================================"
echo "  Ray Serve A100 bare-metal REST/gRPC benchmark"
echo "  Result dir: ${RESULT_DIR}"
echo "  Model: ${MODEL_ID}"
echo "  Dataset: ${DATASET}"
echo "  Batches: ${BATCH_CONFIGS}"
echo "  Protocols: ${PROTOCOLS}"
echo "  Repeats: ${REPEATS}"
echo "  Total runs: ${total_runs}"
echo "  Load: users=${USERS}, spawn_rate=${SPAWN_RATE}, duration=${DURATION} (${duration_secs}s), per_user_throughput=${USER_THROUGHPUT}"
echo "  Ray: replicas=${NUM_REPLICAS}, gpu_per_replica=${NUM_GPUS_PER_REPLICA}, max_ongoing=${MAX_ONGOING_REQUESTS}"
echo "======================================================"

for cfg in "${configs[@]}"; do
    IFS=':' read -r batch_size timeout <<< "${cfg}"
    for run in $(seq 1 "${REPEATS}"); do
        for protocol in "${protocols[@]}"; do
            run_one "${protocol}" "${batch_size}" "${timeout}" "${run}"
        done
    done
done

generate_plots

echo ""
echo "Summary CSV: ${RESULT_DIR}/summary.csv"
if command -v column >/dev/null 2>&1; then
    column -s, -t "${RESULT_DIR}/summary.csv"
else
    cat "${RESULT_DIR}/summary.csv"
fi
echo ""
echo "Done: ${RESULT_DIR}"

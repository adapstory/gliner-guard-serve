#!/usr/bin/env bash
# Run Ray Serve gRPC benchmarks: REST vs gRPC comparison
# Usage: ./scripts/run-grpc-benchmarks.sh
#   or:  REPEATS=3 USERS=100 ./scripts/run-grpc-benchmarks.sh  (cloud VM)
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/.."

ENV_FILE="${ENV_FILE:-.env}"
COMPOSE=(docker compose)
if [[ -f "${ENV_FILE}" ]]; then
    export ENV_FILE
    set -a
    # shellcheck source=/dev/null
    . "${ENV_FILE}"
    set +a
    COMPOSE=(docker compose --env-file "${ENV_FILE}")
fi

DURATION="${DURATION:-15m}"
USERS="${USERS:-20}"
SPAWN_RATE="${SPAWN_RATE:-1}"
USER_THROUGHPUT="${USER_THROUGHPUT:-${LOCUST_USER_THROUGHPUT:-5}}"
WARMUP_REQS="${WARMUP_REQS:-50}"
DATASET="${DATASET:-prompts}"
REPEATS="${REPEATS:-1}"
MODEL_ID="${MODEL_ID:-hivetrace/gliner-guard-uniencoder}"
MODEL_SHORT="${MODEL_SHORT:-uni}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-0}"
BATCH_WAIT_TIMEOUT="${BATCH_WAIT_TIMEOUT:-0.05}"

mkdir -p results

wait_ready_rest() {
    echo "  Waiting for REST on :8000..."
    for i in $(seq 1 120); do
        if curl -sf -o /dev/null http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d '{"text":"healthcheck"}'; then
            echo "  REST server ready!"
            return 0
        fi
        sleep 2
    done
    echo "  TIMEOUT waiting for REST server"
    return 1
}

wait_ready_grpc() {
    echo "  Waiting for gRPC on :9000..."
    # Use python grpc health check
    for i in $(seq 1 120); do
        if python3 -c "
import grpc, sys
sys.path.insert(0, 'ray-serve')
import gliner_guard_pb2, gliner_guard_pb2_grpc
ch = grpc.insecure_channel('localhost:9000')
stub = gliner_guard_pb2_grpc.GLiNERGuardServiceStub(ch)
try:
    stub.Predict(gliner_guard_pb2.PredictRequest(text='ping'), timeout=5)
    print('ok')
except: pass
" 2>/dev/null | grep -q "ok"; then
            echo "  gRPC server ready!"
            return 0
        fi
        sleep 2
    done
    echo "  TIMEOUT waiting for gRPC server"
    return 1
}

warmup_rest() {
    echo "  Warmup REST: ${WARMUP_REQS} requests..."
    for i in $(seq 1 "${WARMUP_REQS}"); do
        curl -sf -o /dev/null http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d '{"text":"warmup request '"$i"'"}' &
    done
    wait
    echo "  Warmup done."
}

warmup_grpc() {
    echo "  Warmup gRPC: ${WARMUP_REQS} requests..."
    python3 -c "
import grpc, sys
sys.path.insert(0, 'ray-serve')
import gliner_guard_pb2, gliner_guard_pb2_grpc
ch = grpc.insecure_channel('localhost:9000')
stub = gliner_guard_pb2_grpc.GLiNERGuardServiceStub(ch)
for i in range(${WARMUP_REQS}):
    stub.Predict(gliner_guard_pb2.PredictRequest(text=f'warmup {i}'))
" 2>/dev/null
    echo "  Warmup done."
}

run_rest_bench() {
    local run_num="$1"
    local batch_tag=""
    [ "${MAX_BATCH_SIZE}" -gt 0 ] && batch_tag="-B${MAX_BATCH_SIZE}"
    local prefix="ray-rest${batch_tag}-${MODEL_SHORT}-${DATASET}-run${run_num}"

    echo ""
    echo "=========================================="
    echo "  REST Benchmark: ${prefix}"
    echo "  Batch: ${MAX_BATCH_SIZE}, Run: ${run_num}/${REPEATS}"
    echo "=========================================="

    MODEL_ID="${MODEL_ID}" MAX_BATCH_SIZE="${MAX_BATCH_SIZE}" \
        BATCH_WAIT_TIMEOUT="${BATCH_WAIT_TIMEOUT}" \
        "${COMPOSE[@]}" --profile ray-serve up -d ray-serve 2>&1 | tail -2
    wait_ready_rest || return 1
    warmup_rest

    local gpu_csv="results/gpu-${prefix}.csv"
    local duration_secs
    duration_secs=$(echo "${DURATION}" | sed 's/m//' | awk '{print $1*60}')
    bash scripts/collect_gpu_metrics.sh "${gpu_csv}" "${duration_secs}" &
    local gpu_pid=$!

    cd test-script
    DATASET="${DATASET}" GLINER_HOST=http://localhost:8000 \
        LOCUST_USER_THROUGHPUT="${USER_THROUGHPUT}" \
        uv run locust -f test-gliner.py \
        --headless -u "${USERS}" -r "${SPAWN_RATE}" --run-time "${DURATION}" \
        --csv-full-history \
        --csv="../results/${prefix}" \
        --html="../results/${prefix}.html" 2>&1 | tail -20
    cd ..

    wait "${gpu_pid}" 2>/dev/null || true
    "${COMPOSE[@]}" --profile ray-serve down 2>&1 | tail -2

    local stats_file="results/${prefix}_stats.csv"
    if [ -f "${stats_file}" ]; then
        echo "  Results:"
        grep "Aggregated" "${stats_file}" | awk -F',' '{printf "    RPS=%.1f  P50=%sms  P95=%sms  Failures=%s\n", $10, $6, $8, $4}' || true
    fi
    echo "  Done: ${prefix}"
    sleep 5
}

run_grpc_bench() {
    local run_num="$1"
    local batch_tag=""
    [ "${MAX_BATCH_SIZE}" -gt 0 ] && batch_tag="-B${MAX_BATCH_SIZE}"
    local prefix="ray-grpc${batch_tag}-${MODEL_SHORT}-${DATASET}-run${run_num}"

    echo ""
    echo "=========================================="
    echo "  gRPC Benchmark: ${prefix}"
    echo "  Batch: ${MAX_BATCH_SIZE}, Run: ${run_num}/${REPEATS}"
    echo "=========================================="

    MODEL_ID="${MODEL_ID}" MAX_BATCH_SIZE="${MAX_BATCH_SIZE}" \
        BATCH_WAIT_TIMEOUT="${BATCH_WAIT_TIMEOUT}" \
        "${COMPOSE[@]}" --profile ray-serve-grpc up -d ray-serve-grpc 2>&1 | tail -2
    wait_ready_rest || return 1  # gRPC app also serves REST on 8000
    sleep 5  # extra time for gRPC proxy to start
    warmup_grpc

    local gpu_csv="results/gpu-${prefix}.csv"
    local duration_secs
    duration_secs=$(echo "${DURATION}" | sed 's/m//' | awk '{print $1*60}')
    bash scripts/collect_gpu_metrics.sh "${gpu_csv}" "${duration_secs}" &
    local gpu_pid=$!

    cd test-script
    DATASET="${DATASET}" GLINER_HOST=localhost:9000 \
        LOCUST_USER_THROUGHPUT="${USER_THROUGHPUT}" \
        uv run locust -f test-gliner-grpc.py \
        --headless -u "${USERS}" -r "${SPAWN_RATE}" --run-time "${DURATION}" \
        --csv-full-history \
        --csv="../results/${prefix}" \
        --html="../results/${prefix}.html" 2>&1 | tail -20
    cd ..

    wait "${gpu_pid}" 2>/dev/null || true
    "${COMPOSE[@]}" --profile ray-serve-grpc down 2>&1 | tail -2

    local stats_file="results/${prefix}_stats.csv"
    if [ -f "${stats_file}" ]; then
        echo "  Results:"
        grep "Aggregated" "${stats_file}" | awk -F',' '{printf "    RPS=%.1f  P50=%sms  P95=%sms  Failures=%s\n", $10, $6, $8, $4}' || true
    fi
    echo "  Done: ${prefix}"
    sleep 5
}

total_runs=$((REPEATS * 2))
echo "======================================================"
echo "  Ray Serve REST vs gRPC Benchmark"
echo "  Model: ${MODEL_SHORT} (${MODEL_ID})"
echo "  Batch: ${MAX_BATCH_SIZE}"
echo "  Repeats: ${REPEATS} per protocol"
echo "  Total runs: ${total_runs} (${REPEATS} REST + ${REPEATS} gRPC)"
echo "  Duration: ${DURATION} per run"
echo "  Users: ${USERS}"
echo "  Per-user throughput target: ${USER_THROUGHPUT}"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

echo ""
echo ">>> Protocol: REST"
for run in $(seq 1 "${REPEATS}"); do
    run_rest_bench "${run}"
done

echo ""
echo ">>> Protocol: gRPC"
for run in $(seq 1 "${REPEATS}"); do
    run_grpc_bench "${run}"
done

echo ""
echo "======================================================"
echo "  REST vs gRPC comparison complete!"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

echo ""
echo "Summary:"
echo "| Benchmark | RPS | P50 (ms) | P95 (ms) | Failures |"
echo "|-----------|----:|--------:|---------:|---------:|"
for f in results/ray-rest-*_stats.csv results/ray-grpc-*_stats.csv; do
    [ -f "$f" ] || continue
    name=$(basename "$f" _stats.csv)
    tail -1 "$f" | awk -F',' -v n="$name" '{printf "| %s | %.1f | %s | %s | %s |\n", n, $10, $6, $8, $4}'
done

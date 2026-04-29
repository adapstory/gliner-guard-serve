#!/usr/bin/env bash
# Reproducible A100 four-worker REST benchmark:
#   LitServe REST vs Ray Serve REST, batch sizes 16/32/64, same Locust profile.
#
# Usage:
#   ./scripts/run-a100-4worker-rest-benchmarks.sh
#   ENV_FILE=.env.ray-a100-4w.example REPEATS=1 DURATION=2m ./scripts/run-a100-4worker-rest-benchmarks.sh
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/.."

ENV_FILE="${ENV_FILE:-.env.ray-a100-4w.example}"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing env file: ${ENV_FILE}" >&2
    exit 1
fi
export ENV_FILE

set -a
# shellcheck source=/dev/null
. "${ENV_FILE}"
set +a

DURATION="${DURATION:-${LOCUST_RUN_TIME:-15m}}"
USERS="${USERS:-${LOCUST_USERS:-100}}"
SPAWN_RATE="${SPAWN_RATE:-${LOCUST_SPAWN_RATE:-1}}"
USER_THROUGHPUT="${USER_THROUGHPUT:-${LOCUST_USER_THROUGHPUT:-5}}"
DATASET="${DATASET:-prompts}"
REPEATS="${REPEATS:-3}"
MODEL_ID="${MODEL_ID:-hivetrace/gliner-guard-uniencoder}"
MODEL_SHORT="${MODEL_SHORT:-uni}"
WARMUP_REQS="${WARMUP_REQS:-50}"
BATCH_CONFIGS="${BATCH_CONFIGS:-16:0.05,32:0.05,64:0.05}"
LITSERVE_WORKERS_PER_DEVICE="${LITSERVE_WORKERS_PER_DEVICE:-4}"

mkdir -p results

COMPOSE=(docker compose --env-file "${ENV_FILE}")

duration_seconds() {
    echo "${DURATION}" | sed 's/m//' | awk '{print $1*60}'
}

wait_ready() {
    echo "  Waiting for REST server on :8000..."
    for _ in $(seq 1 120); do
        if curl -sf -o /dev/null http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d '{"text":"healthcheck"}'; then
            echo "  REST server ready."
            return 0
        fi
        sleep 2
    done
    echo "  TIMEOUT waiting for REST server" >&2
    return 1
}

warmup() {
    echo "  Warmup: ${WARMUP_REQS} requests..."
    for i in $(seq 1 "${WARMUP_REQS}"); do
        curl -sf -o /dev/null http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d '{"text":"warmup request '"${i}"'"}' &
    done
    wait
}

run_locust() {
    local prefix="$1"
    local gpu_csv="results/gpu-${prefix}.csv"
    local duration_secs
    duration_secs="$(duration_seconds)"

    bash scripts/collect_gpu_metrics.sh "${gpu_csv}" "${duration_secs}" &
    local gpu_pid=$!

    cd test-script
    DATASET="${DATASET}" \
        GLINER_HOST=http://localhost:8000 \
        LOCUST_USER_THROUGHPUT="${USER_THROUGHPUT}" \
        uv run locust -f test-gliner.py \
        --headless -u "${USERS}" -r "${SPAWN_RATE}" --run-time "${DURATION}" \
        --csv-full-history \
        --csv="../results/${prefix}" \
        --html="../results/${prefix}.html" 2>&1 | tail -20
    cd ..

    wait "${gpu_pid}" 2>/dev/null || true
}

print_result() {
    local prefix="$1"
    local stats_file="results/${prefix}_stats.csv"
    if [[ -f "${stats_file}" ]]; then
        grep "Aggregated" "${stats_file}" \
            | awk -F',' '{printf "    RPS=%.1f  P50=%sms  P95=%sms  Failures=%s\n", $10, $6, $8, $4}' \
            || true
    else
        echo "    missing stats: ${stats_file}"
    fi
}

run_litserve() {
    local batch_size="$1"
    local timeout="$2"
    local run_num="$3"
    local prefix="litserve-rest-B${batch_size}-${MODEL_SHORT}-${DATASET}-run${run_num}"

    echo ""
    echo "=========================================="
    echo "  LitServe REST: ${prefix}"
    echo "  workers_per_device=${LITSERVE_WORKERS_PER_DEVICE} batch=${batch_size} timeout=${timeout}"
    echo "=========================================="

    MODEL_ID="${MODEL_ID}" \
        LITSERVE_MAX_BATCH_SIZE="${batch_size}" \
        LITSERVE_BATCH_TIMEOUT="${timeout}" \
        LITSERVE_WORKERS_PER_DEVICE="${LITSERVE_WORKERS_PER_DEVICE}" \
        "${COMPOSE[@]}" --profile litserve up -d litserve
    wait_ready
    warmup
    run_locust "${prefix}"
    "${COMPOSE[@]}" --profile litserve down
    print_result "${prefix}"
}

run_ray() {
    local batch_size="$1"
    local timeout="$2"
    local run_num="$3"
    local prefix="ray-rest-B${batch_size}-${MODEL_SHORT}-${DATASET}-run${run_num}"

    echo ""
    echo "=========================================="
    echo "  Ray Serve REST: ${prefix}"
    echo "  replicas=${NUM_REPLICAS:-1} gpu_per_replica=${NUM_GPUS_PER_REPLICA:-1} batch=${batch_size} timeout=${timeout}"
    echo "=========================================="

    MODEL_ID="${MODEL_ID}" \
        MAX_BATCH_SIZE="${batch_size}" \
        BATCH_WAIT_TIMEOUT="${timeout}" \
        "${COMPOSE[@]}" --profile ray-serve up -d ray-serve
    wait_ready
    warmup
    run_locust "${prefix}"
    "${COMPOSE[@]}" --profile ray-serve down
    print_result "${prefix}"
}

echo "======================================================"
echo "  A100 4-worker REST comparison"
echo "  Env file: ${ENV_FILE}"
echo "  Model: ${MODEL_ID}"
echo "  Dataset: ${DATASET}"
echo "  Batches: ${BATCH_CONFIGS}"
echo "  Repeats: ${REPEATS}"
echo "  Load: users=${USERS}, spawn_rate=${SPAWN_RATE}, duration=${DURATION}, per_user_throughput=${USER_THROUGHPUT}"
echo "======================================================"

IFS=',' read -r -a configs <<< "${BATCH_CONFIGS}"
for cfg in "${configs[@]}"; do
    IFS=':' read -r batch_size timeout <<< "${cfg}"
    for run in $(seq 1 "${REPEATS}"); do
        run_litserve "${batch_size}" "${timeout}" "${run}"
        run_ray "${batch_size}" "${timeout}" "${run}"
    done
done

echo ""
echo "Summary:"
echo "| Benchmark | RPS | P50 (ms) | P95 (ms) | Failures |"
echo "|-----------|----:|--------:|---------:|---------:|"
for f in results/litserve-rest-B*_stats.csv results/ray-rest-B*_stats.csv; do
    [[ -f "${f}" ]] || continue
    name="$(basename "${f}" _stats.csv)"
    tail -1 "${f}" | awk -F',' -v n="${name}" '{printf "| %s | %.1f | %s | %s | %s |\n", n, $10, $6, $8, $4}'
done

echo ""
echo "Generating history plots..."
for cfg in "${configs[@]}"; do
    IFS=':' read -r batch_size _ <<< "${cfg}"
    for framework in litserve ray; do
        pattern="results/${framework}-rest-B${batch_size}-${MODEL_SHORT}-${DATASET}-run*_stats_history.csv"
        output="results/${framework}-rest-B${batch_size}-${MODEL_SHORT}-${DATASET}-history.png"
        uv run --with pandas --with matplotlib --with seaborn \
            python scripts/plot_locust_history.py \
            --glob "${pattern}" \
            --output "${output}" \
            --title "${framework} REST B${batch_size} ${MODEL_SHORT} ${DATASET}" \
            --max-users "${USERS}" \
            --latency-percentile "95%" \
            --throughput-scale 1000 \
            || true
    done
done

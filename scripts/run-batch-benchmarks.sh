#!/usr/bin/env bash
# Run Ray Serve Dynamic Batching sweep: B1-B4 configs × 1 repeat (dev GPU quick validation)
# Usage: ./scripts/run-batch-benchmarks.sh
#   or:  REPEATS=3 USERS=100 ./scripts/run-batch-benchmarks.sh   (cloud VM full sweep)
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/.."

DURATION="${DURATION:-15m}"
USERS="${USERS:-20}"
SPAWN_RATE="${SPAWN_RATE:-1}"
WARMUP_REQS="${WARMUP_REQS:-50}"
DATASET="${DATASET:-prompts}"
REPEATS="${REPEATS:-1}"
MODEL_ID="${MODEL_ID:-hivetrace/gliner-guard-uniencoder}"
MODEL_SHORT="${MODEL_SHORT:-uni}"

# Batch configurations: "ID:max_batch_size:batch_wait_timeout"
CONFIGS=(
    "B1:8:0.01"
    "B2:16:0.05"
    "B3:32:0.05"
    "B4:64:0.10"
)

mkdir -p results

wait_ready() {
    echo "  Waiting for server on :8000..."
    for i in $(seq 1 120); do
        if curl -sf -o /dev/null http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d '{"text":"healthcheck"}'; then
            echo "  Server ready!"
            return 0
        fi
        sleep 2
    done
    echo "  TIMEOUT waiting for server"
    return 1
}

warmup() {
    echo "  Warmup: ${WARMUP_REQS} requests..."
    for i in $(seq 1 "${WARMUP_REQS}"); do
        curl -sf -o /dev/null http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d '{"text":"warmup request number '"$i"'"}' &
    done
    wait
    echo "  Warmup done."
}

run_bench() {
    local batch_id="$1"
    local batch_size="$2"
    local batch_timeout="$3"
    local run_num="$4"
    local prefix="ray-rest-${batch_id}-${MODEL_SHORT}-${DATASET}-run${run_num}"

    echo ""
    echo "=========================================="
    echo "  Benchmark: ${prefix}"
    echo "  Model: ${MODEL_ID}"
    echo "  Batch: size=${batch_size}, timeout=${batch_timeout}"
    echo "  Run: ${run_num}/${REPEATS}"
    echo "=========================================="

    # Start server with batch config
    echo "  Starting Ray Serve (batch_size=${batch_size}, timeout=${batch_timeout})..."
    MAX_BATCH_SIZE="${batch_size}" BATCH_WAIT_TIMEOUT="${batch_timeout}" \
        MODEL_ID="${MODEL_ID}" \
        docker compose --profile ray-serve up -d ray-serve 2>&1 | tail -2
    wait_ready || return 1

    # Warmup
    warmup

    # GPU metrics in background
    local gpu_csv="results/gpu-${prefix}.csv"
    local duration_secs
    duration_secs=$(echo "${DURATION}" | sed 's/m//' | awk '{print $1*60}')
    bash scripts/collect_gpu_metrics.sh "${gpu_csv}" "${duration_secs}" &
    local gpu_pid=$!

    # Run Locust
    echo "  Running Locust: ${USERS} users, ${SPAWN_RATE}/s, ${DURATION}..."
    cd test-script
    DATASET="${DATASET}" GLINER_HOST=http://localhost:8000 \
        uv run locust -f test-gliner.py \
        --headless -u "${USERS}" -r "${SPAWN_RATE}" --run-time "${DURATION}" \
        --csv="../results/${prefix}" \
        --html="../results/${prefix}.html" 2>&1 | tail -20
    cd ..

    # Wait for GPU metrics
    wait "${gpu_pid}" 2>/dev/null || true

    # Stop server
    echo "  Stopping server..."
    docker compose --profile ray-serve down 2>&1 | tail -2

    # Extract summary
    local stats_file="results/${prefix}_stats.csv"
    if [ -f "${stats_file}" ]; then
        echo "  Results:"
        grep "Aggregated" "${stats_file}" | awk -F',' '{printf "    RPS=%.1f  P50=%sms  P95=%sms  Failures=%s\n", $10, $6, $8, $4}' || echo "    (could not parse stats)"
    else
        echo "  WARNING: stats file not found: ${stats_file}"
    fi

    echo "  Done: ${prefix}"
    echo ""
    sleep 5
}

total_runs=$(( ${#CONFIGS[@]} * REPEATS ))
echo "======================================================"
echo "  Ray Serve Dynamic Batching Sweep"
echo "  Model: ${MODEL_SHORT} (${MODEL_ID})"
echo "  Configs: ${#CONFIGS[@]} (B1-B4)"
echo "  Repeats: ${REPEATS} per config"
echo "  Total runs: ${total_runs}"
echo "  Duration: ${DURATION} per run"
echo "  Users: ${USERS}"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r batch_id batch_size batch_timeout <<< "${config}"
    echo ""
    echo ">>> Config: ${batch_id} (batch_size=${batch_size}, timeout=${batch_timeout})"
    echo ""
    for run in $(seq 1 "${REPEATS}"); do
        run_bench "${batch_id}" "${batch_size}" "${batch_timeout}" "${run}"
    done
done

echo ""
echo "======================================================"
echo "  All benchmarks complete!"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results in: results/"
echo "======================================================"

# Summary table
echo ""
echo "Summary:"
echo "| Benchmark | RPS | P50 (ms) | P95 (ms) | Failures |"
echo "|-----------|----:|--------:|---------:|---------:|"
for f in results/ray-rest-B*_stats.csv; do
    [ -f "$f" ] || continue
    name=$(basename "$f" _stats.csv)
    tail -1 "$f" | awk -F',' -v n="$name" '{printf "| %s | %.1f | %s | %s | %s |\n", n, $10, $6, $8, $4}'
done

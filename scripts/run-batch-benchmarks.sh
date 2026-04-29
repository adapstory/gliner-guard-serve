#!/usr/bin/env bash
# Run Ray Serve Dynamic Batching sweep: B1-B8 configs × N repeats
# Usage: ./scripts/run-batch-benchmarks.sh                          (dev GPU: 1 repeat, 20 users)
#   or:  REPEATS=3 USERS=100 ./scripts/run-batch-benchmarks.sh     (cloud VM: full sweep)
#   or:  DURATION=2m USERS=10 ./scripts/run-batch-benchmarks.sh    (demo: verify scripts work)
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
# B1-B8: systematic sweep (batch_size × timeout)
# B9-B11: special configs (token-aware, max_ongoing_requests) — separate script
CONFIGS=(
    "B1:8:0.01"
    "B2:8:0.05"
    "B3:16:0.01"
    "B4:16:0.05"
    "B5:32:0.05"
    "B6:32:0.10"
    "B7:64:0.05"
    "B8:64:0.10"
)

FAILED_CONFIGS=()

mkdir -p results

container_alive() {
    # Check if the ray-serve container is still running (not exited/crashed)
    local state
    state=$(docker compose --profile ray-serve ps --format '{{.State}}' ray-serve 2>/dev/null)
    [[ "${state}" == "running" ]]
}

wait_ready() {
    echo "  Waiting for server on :8000..."
    for i in $(seq 1 60); do
        if curl -sf -o /dev/null http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d '{"text":"healthcheck"}'; then
            echo "  Server ready!"
            return 0
        fi
        # Early exit: if container crashed, don't keep waiting
        if ! container_alive; then
            echo "  FATAL: container exited (model load failure?)"
            return 1
        fi
        sleep 2
    done
    echo "  TIMEOUT waiting for server (120s)"
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
    if ! wait_ready; then
        echo "  FATAL: server didn't start. Stopping container..."
        docker compose --profile ray-serve down 2>&1 | tail -2
        sleep 5
        return 1
    fi

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
        LOCUST_USER_THROUGHPUT="${LOCUST_USER_THROUGHPUT:-5}" \
        uv run locust -f test-gliner.py \
        --headless -u "${USERS}" -r "${SPAWN_RATE}" --run-time "${DURATION}" \
        --csv-full-history \
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
echo "  Configs: ${#CONFIGS[@]} (B1-B8)"
echo "  Repeats: ${REPEATS} per config"
echo "  Total runs: ${total_runs}"
echo "  Duration: ${DURATION} per run"
echo "  Users: ${USERS}"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

model_broken=false

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r batch_id batch_size batch_timeout <<< "${config}"

    if $model_broken; then
        echo ""
        echo ">>> SKIP: ${batch_id} — model failed to load, skipping remaining configs"
        FAILED_CONFIGS+=("${batch_id}")
        continue
    fi

    echo ""
    echo ">>> Config: ${batch_id} (batch_size=${batch_size}, timeout=${batch_timeout})"
    echo ""
    for run in $(seq 1 "${REPEATS}"); do
        if ! run_bench "${batch_id}" "${batch_size}" "${batch_timeout}" "${run}"; then
            FAILED_CONFIGS+=("${batch_id}")
            # If first config fails, model itself is likely broken — skip the rest
            if [[ "${#FAILED_CONFIGS[@]}" -ge 2 ]]; then
                echo "  WARNING: 2+ configs failed — model appears broken, skipping remaining"
                model_broken=true
            fi
            break  # skip remaining repeats for this config
        fi
    done
done

echo ""
echo "======================================================"
if [[ ${#FAILED_CONFIGS[@]} -gt 0 ]]; then
    echo "  Sweep finished with failures: ${FAILED_CONFIGS[*]}"
else
    echo "  All benchmarks complete!"
fi
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

#!/usr/bin/env bash
# Run Ray Serve REST No-Batch benchmarks: 3 repeats × 2 models
# Usage: ./scripts/run-nobatch-benchmarks.sh
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/.."

DURATION="${DURATION:-15m}"
USERS="${USERS:-20}"
SPAWN_RATE="${SPAWN_RATE:-1}"
WARMUP_REQS="${WARMUP_REQS:-50}"
DATASET="${DATASET:-prompts}"

MODELS=("hivetrace/gliner-guard-uniencoder" "hivetrace/gliner-guard-biencoder")
MODEL_SHORTS=("uni" "bi")

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
    local model_id="$1"
    local model_short="$2"
    local run_num="$3"
    local prefix="ray-rest-nobatch-${model_short}-${DATASET}-run${run_num}"

    echo ""
    echo "=========================================="
    echo "  Benchmark: ${prefix}"
    echo "  Model: ${model_id}"
    echo "  Run: ${run_num}/3"
    echo "=========================================="

    # Start server
    echo "  Starting Ray Serve..."
    MODEL_ID="${model_id}" docker compose --profile ray-serve up -d ray-serve 2>&1 | tail -2
    wait_ready

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

echo "======================================================"
echo "  Ray Serve REST No-Batch Benchmark Suite"
echo "  Models: uniencoder + biencoder"
echo "  Repeats: 3 per model"
echo "  Duration: ${DURATION} per run"
echo "  Total estimated time: ~90 minutes"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

for model_idx in 0 1; do
    model_id="${MODELS[$model_idx]}"
    model_short="${MODEL_SHORTS[$model_idx]}"

    echo ""
    echo ">>> Model: ${model_short} (${model_id})"
    echo ""

    for run in 1 2 3; do
        run_bench "${model_id}" "${model_short}" "${run}"
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
for f in results/ray-rest-nobatch-*_stats.csv; do
    [ -f "$f" ] || continue
    name=$(basename "$f" _stats.csv)
    tail -1 "$f" | awk -F',' -v n="$name" '{printf "| %s | %.1f | %s | %s | %s |\n", n, $10, $6, $8, $4}'
done

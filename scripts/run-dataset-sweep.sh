#!/usr/bin/env bash
# Run dataset sweep with optimal batch config across all 5 datasets × 2 models (Day 11)
# Usage: BATCH_ID=B4 MAX_BATCH_SIZE=16 BATCH_WAIT_TIMEOUT=0.05 ./scripts/run-dataset-sweep.sh
#   or:  DURATION=2m USERS=10 REPEATS=1 ./scripts/run-dataset-sweep.sh   (demo mode)
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/.."

DURATION="${DURATION:-15m}"
USERS="${USERS:-100}"
SPAWN_RATE="${SPAWN_RATE:-1}"
WARMUP_REQS="${WARMUP_REQS:-50}"
REPEATS="${REPEATS:-3}"

# Optimal batch config (set from Phase 2 results)
BATCH_ID="${BATCH_ID:-B4}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"
BATCH_WAIT_TIMEOUT="${BATCH_WAIT_TIMEOUT:-0.05}"

# Datasets to sweep (synthetic-medium already done in batch sweep)
DATASETS=(
    "prompts-short"
    "prompts-long"
    "xstest"
    "aya-rus"
)

# Models
MODELS=(
    "uni:hivetrace/gliner-guard-uniencoder"
    "bi:hivetrace/gliner-guard-biencoder"
)

FAILED_MODELS=()

mkdir -p results

port_free() {
    ! curl -sf -o /dev/null --connect-timeout 1 http://localhost:8000/ 2>/dev/null
}

container_alive() {
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
    local model_short="$1"
    local model_id="$2"
    local dataset="$3"
    local run_num="$4"
    local prefix="ray-rest-${BATCH_ID}-${model_short}-${dataset}-run${run_num}"

    echo ""
    echo "=========================================="
    echo "  Dataset Sweep: ${prefix}"
    echo "  Model: ${model_id}"
    echo "  Batch: ${BATCH_ID} (size=${MAX_BATCH_SIZE}, timeout=${BATCH_WAIT_TIMEOUT})"
    echo "  Dataset: ${dataset}"
    echo "  Run: ${run_num}/${REPEATS}"
    echo "=========================================="

    # Start server
    echo "  Starting Ray Serve..."
    MAX_BATCH_SIZE="${MAX_BATCH_SIZE}" BATCH_WAIT_TIMEOUT="${BATCH_WAIT_TIMEOUT}" \
        MODEL_ID="${model_id}" \
        docker compose --profile ray-serve up -d ray-serve 2>&1 | tail -2
    if ! wait_ready; then
        echo "  FATAL: server didn't start. Stopping container..."
        docker compose --profile ray-serve down 2>&1 | tail -2
        sleep 5
        return 1
    fi

    warmup

    # GPU metrics
    local gpu_csv="results/gpu-${prefix}.csv"
    local duration_secs
    duration_secs=$(echo "${DURATION}" | sed 's/m//' | awk '{print $1*60}')
    bash scripts/collect_gpu_metrics.sh "${gpu_csv}" "${duration_secs}" &
    local gpu_pid=$!

    # Locust
    echo "  Running Locust: ${USERS} users, ${SPAWN_RATE}/s, ${DURATION}..."
    cd test-script
    DATASET="${dataset}" GLINER_HOST=http://localhost:8000 \
        uv run locust -f test-gliner.py \
        --headless -u "${USERS}" -r "${SPAWN_RATE}" --run-time "${DURATION}" \
        --csv="../results/${prefix}" \
        --html="../results/${prefix}.html" 2>&1 | tail -20
    cd ..

    wait "${gpu_pid}" 2>/dev/null || true

    # Stop server
    echo "  Stopping server..."
    docker compose --profile ray-serve down 2>&1 | tail -2

    # Summary
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

total_runs=$(( ${#DATASETS[@]} * ${#MODELS[@]} * REPEATS ))
echo "======================================================"
echo "  Dataset Sweep (Day 11)"
echo "  Batch config: ${BATCH_ID} (size=${MAX_BATCH_SIZE}, timeout=${BATCH_WAIT_TIMEOUT})"
echo "  Datasets: ${#DATASETS[@]} (${DATASETS[*]})"
echo "  Models: ${#MODELS[@]} (uni + bi)"
echo "  Repeats: ${REPEATS} per combination"
echo "  Total runs: ${total_runs}"
echo "  Duration: ${DURATION} per run"
echo "  Users: ${USERS}"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

# Pre-flight: check port 8000 is free
if ! port_free; then
    echo "ERROR: port 8000 is already in use. Stop other services first:"
    echo "  docker compose --profile ray-serve down"
    echo "  docker compose --profile litserve down"
    exit 1
fi

for model_entry in "${MODELS[@]}"; do
    IFS=':' read -r model_short model_id <<< "${model_entry}"
    echo ""
    echo ">>> Model: ${model_short} (${model_id})"

    model_broken=false

    for dataset in "${DATASETS[@]}"; do
        if $model_broken; then
            echo "  SKIP: ${dataset} — model ${model_short} failed to load"
            continue
        fi

        echo ""
        echo ">>> Dataset: ${dataset}"
        for run in $(seq 1 "${REPEATS}"); do
            if ! run_bench "${model_short}" "${model_id}" "${dataset}" "${run}"; then
                FAILED_MODELS+=("${model_short}")
                echo "  WARNING: ${model_short} failed on ${dataset} — skipping remaining datasets for this model"
                model_broken=true
                break
            fi
        done
    done
done

echo ""
echo "======================================================"
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo "  Dataset sweep finished with failures: ${FAILED_MODELS[*]}"
else
    echo "  Dataset sweep complete!"
fi
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results in: results/"
echo "======================================================"

# Summary table
echo ""
echo "Summary:"
echo "| Benchmark | RPS | P50 (ms) | P95 (ms) | Failures |"
echo "|-----------|----:|--------:|---------:|---------:|"
for f in results/ray-rest-${BATCH_ID}-*_stats.csv; do
    [ -f "$f" ] || continue
    name=$(basename "$f" _stats.csv)
    tail -1 "$f" | awk -F',' -v n="$name" '{printf "| %s | %.1f | %s | %s | %s |\n", n, $10, $6, $8, $4}'
done

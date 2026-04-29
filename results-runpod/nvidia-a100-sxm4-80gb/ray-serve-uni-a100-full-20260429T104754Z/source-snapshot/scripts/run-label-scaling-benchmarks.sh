#!/usr/bin/env bash
# --------------------------------------------------------------------------
# Label Scaling Benchmark: minimal (6 labels) vs full (56 labels)
#
# Runs Locust load tests for both schema modes on the same model,
# measuring throughput and latency impact of expanded taxonomy.
#
# Prerequisites:
#   - Docker images built (make build-ray)
#   - GPU available
#
# Usage:
#   ./scripts/run-label-scaling-benchmarks.sh [USERS] [DURATION] [REPEATS]
#   Default: 20 users, 5m duration, 3 repeats
# --------------------------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")/.."

USERS="${1:-20}"
DURATION="${2:-5m}"
REPEATS="${3:-3}"
DATASET="${DATASET:-prompts}"
MODEL_ID="${MODEL_ID:-hivetrace/gliner-guard-uniencoder}"
RESULTS_DIR="results/label-scaling"
COMPOSE="docker compose"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  Label Scaling Benchmark"
echo "  Users=$USERS  Duration=$DURATION  Repeats=$REPEATS"
echo "  Dataset=$DATASET  Model=$MODEL_ID"
echo "============================================================"

run_locust() {
    local schema_mode=$1
    local run_num=$2
    local csv_prefix="$RESULTS_DIR/ray-rest-${schema_mode}-${DATASET}-run${run_num}"

    echo ""
    echo "--- Run $run_num: SCHEMA_MODE=$schema_mode ---"

    # Start Ray Serve with the specified schema mode
    SCHEMA_MODE="$schema_mode" MODEL_ID="$MODEL_ID" \
        $COMPOSE --profile ray-serve up -d --wait

    # Wait for model to load
    echo "Waiting for model readiness..."
    for i in $(seq 1 60); do
        if curl -sf http://localhost:8000/-/healthz > /dev/null 2>&1; then
            echo "  Ready after ${i}s"
            break
        fi
        sleep 1
    done

    # Run Locust headless
    DATASET="$DATASET" $COMPOSE run --rm \
        -e LOCUST_USERS="$USERS" \
        -e LOCUST_SPAWN_RATE="$USERS" \
        -e LOCUST_RUN_TIME="$DURATION" \
        locust \
        --headless \
        --csv="/results/$(basename "$csv_prefix")" \
        2>&1 | tail -5

    # Copy results from container volume
    echo "  Results → $csv_prefix"

    # Stop Ray Serve
    $COMPOSE --profile ray-serve down
    sleep 2
}

for run in $(seq 1 "$REPEATS"); do
    run_locust "minimal" "$run"
    run_locust "full" "$run"
done

echo ""
echo "============================================================"
echo "  Done. Results in $RESULTS_DIR/"
echo "============================================================"
echo ""
echo "Compare with:"
echo "  python scripts/gen-benchmark-table.py $RESULTS_DIR/"

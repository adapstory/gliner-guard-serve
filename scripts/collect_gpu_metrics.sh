#!/bin/bash
# Collect GPU metrics during benchmark runs.
# Usage: ./collect_gpu_metrics.sh <output_file> <duration_seconds>
#
# Example: ./collect_gpu_metrics.sh results/gpu-litserve-run1.csv 900

set -euo pipefail

OUTPUT="${1:?Usage: $0 <output_file> <duration_seconds>}"
DURATION="${2:?Usage: $0 <output_file> <duration_seconds>}"

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv,nounits -l 1 > "$OUTPUT" &
PID=$!

echo "GPU metrics → $OUTPUT (PID=$PID, duration=${DURATION}s)"
sleep "$DURATION"
kill "$PID" 2>/dev/null || true
echo "Done. $(wc -l < "$OUTPUT") samples collected."

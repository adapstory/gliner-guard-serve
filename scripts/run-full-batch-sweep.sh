#!/usr/bin/env bash
# Full batch sweep: B1-B8 × 2 models × N repeats (Days 8-10)
# Usage: REPEATS=3 USERS=100 DURATION=15m ./scripts/run-full-batch-sweep.sh   (cloud VM)
#   or:  DURATION=2m USERS=10 REPEATS=1 ./scripts/run-full-batch-sweep.sh     (demo)
#
# Each model sweep runs independently — if biencoder fails, uniencoder results are kept.
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/.."

REPEATS="${REPEATS:-3}"
DURATION="${DURATION:-15m}"
USERS="${USERS:-100}"

uni_rc=0
bi_rc=0

echo "======================================================"
echo "  Full Batch Sweep (Days 8-10)"
echo "  Duration: ${DURATION}, Users: ${USERS}, Repeats: ${REPEATS}"
echo "  Total: 8 configs × 2 models × ${REPEATS} repeats = $(( 8 * 2 * REPEATS )) runs"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

# Day 8-9: Uniencoder B1-B8
echo ""
echo "========== UNIENCODER (Day 8-9) =========="
DURATION="${DURATION}" USERS="${USERS}" REPEATS="${REPEATS}" \
    MODEL_ID="hivetrace/gliner-guard-uniencoder" MODEL_SHORT="uni" \
    bash scripts/run-batch-benchmarks.sh || uni_rc=$?

if [[ $uni_rc -ne 0 ]]; then
    echo "  WARNING: Uniencoder sweep exited with code ${uni_rc}"
fi

# Day 10: Biencoder B1-B8
echo ""
echo "========== BIENCODER (Day 10) =========="
DURATION="${DURATION}" USERS="${USERS}" REPEATS="${REPEATS}" \
    MODEL_ID="hivetrace/gliner-guard-biencoder" MODEL_SHORT="bi" \
    bash scripts/run-batch-benchmarks.sh || bi_rc=$?

if [[ $bi_rc -ne 0 ]]; then
    echo "  WARNING: Biencoder sweep exited with code ${bi_rc}"
fi

echo ""
echo "======================================================"
if [[ $uni_rc -eq 0 && $bi_rc -eq 0 ]]; then
    echo "  Full sweep complete!"
else
    echo "  Sweep finished with issues (uni=${uni_rc}, bi=${bi_rc})"
fi
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

# Combined summary
echo ""
echo "Combined Summary:"
echo "| Benchmark | RPS | P50 (ms) | P95 (ms) | Failures |"
echo "|-----------|----:|--------:|---------:|---------:|"
for f in results/ray-rest-B*_stats.csv; do
    [ -f "$f" ] || continue
    name=$(basename "$f" _stats.csv)
    tail -1 "$f" | awk -F',' -v n="$name" '{printf "| %s | %.1f | %s | %s | %s |\n", n, $10, $6, $8, $4}'
done

#!/usr/bin/env bash
# Persistently watch a bare-metal A100 benchmark and write status files.
set -Eeuo pipefail

cd "$(dirname "$0")/.."

RESULT_DIR="${RESULT_DIR:?RESULT_DIR is required}"
PID_ARG=()
if [[ -n "${EXPERIMENT_PID:-}" ]]; then
    PID_ARG=(--pid "${EXPERIMENT_PID}")
fi

INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
EXPECTED_RUNS="${EXPECTED_RUNS:-18}"
STALE_MINUTES="${STALE_MINUTES:-25}"
MONITOR_DIR="${RESULT_DIR}/monitor"
mkdir -p "${MONITOR_DIR}"

echo "$$" > "${MONITOR_DIR}/watchdog.pid"
echo "watchdog_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${MONITOR_DIR}/watchdog.log"

while true; do
    status_json="$(
        python3 scripts/monitor_a100_experiment.py \
            --result-dir "${RESULT_DIR}" \
            "${PID_ARG[@]}" \
            --expected-runs "${EXPECTED_RUNS}" \
            --stale-minutes "${STALE_MINUTES}" \
            --write-status \
            --json
    )"
    state="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])' <<< "${status_json}")"
    completed="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["completed_runs"])' <<< "${status_json}")"
    active="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("active_prefix") or "")' <<< "${status_json}")"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) state=${state} completed=${completed}/${EXPECTED_RUNS} active=${active}" >> "${MONITOR_DIR}/watchdog.log"

    case "${state}" in
        completed|failed|stalled)
            exit 0
            ;;
    esac

    sleep "${INTERVAL_SECONDS}"
done

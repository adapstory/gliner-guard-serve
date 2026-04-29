# Ray Serve A100 4-Worker Reproducible Config

> Purpose: make the Ray Serve REST experiment directly comparable with the
> LitServe baseline on A100 and keep the PR reproducible.

## Source Baseline

LitServe baseline:

- `workers_per_device=4`
- dynamic batching enabled
- `max_batch_size=64`
- `batch_timeout=0.05s`
- Locust: `100` users, `1` user/s spawn rate, `15m` run

The matching Ray Serve shape is not a single replica. For one A100 shared by
four workers, use four Ray Serve replicas with fractional GPU scheduling:

- `NUM_REPLICAS=4`
- `NUM_GPUS_PER_REPLICA=0.25`
- `GPU_COUNT=1`
- `MAX_ONGOING_REQUESTS=256`
- `MAX_CONCURRENT_BATCHES=1`

For four physical A100 GPUs, keep the same replica count but use:

- `GPU_COUNT=4`
- `NUM_GPUS_PER_REPLICA=1`

Ray supports fractional GPU resources for actors/replicas, but GPU memory
is still the application's responsibility.

## Config Files

- `.env.ray-a100-4w.example` is the reproducible A100 profile.
- `scripts/run-a100-4worker-rest-benchmarks.sh` runs LitServe REST and Ray
  Serve REST with the same batch sizes and Locust profile.
- `scripts/setup-a100-baremetal.sh` prepares a Runpod/A100 host without Docker.
- `scripts/run-a100-4worker-ray-baremetal-benchmarks.sh` runs only Ray Serve
  REST and Ray Serve gRPC without Docker Compose and without LitServe.

The sweep intentionally focuses on the PR-critical configs:

| Config | Batch size | Timeout |
|---|---:|---:|
| B16 | 16 | 0.05s |
| B32 | 32 | 0.05s |
| B64 | 64 | 0.05s |

This avoids mixing the main head-to-head comparison with exploratory timeout
variants.

## Run

```bash
cd gliner-guard-serve
cp .env.ray-a100-4w.example .env.ray-a100-4w
# Fill HF_TOKEN if the model download requires it.

ENV_FILE=.env.ray-a100-4w \
  ./scripts/run-a100-4worker-rest-benchmarks.sh
```

Equivalent Make target:

```bash
ENV_FILE=.env.ray-a100-4w make bench-a100-4worker-rest
```

## Bare-Metal Runpod Run

Use this path when Docker Compose is unavailable or intentionally excluded:

```bash
cd /workspace/gliner-guard-serve
bash scripts/setup-a100-baremetal.sh

ENV_FILE=.env.ray-a100-4w.example \
  bash scripts/run-a100-4worker-ray-baremetal-benchmarks.sh
```

Quick smoke before the full run:

```bash
SMOKE=1 \
ENV_FILE=.env.ray-a100-4w.example \
  bash scripts/run-a100-4worker-ray-baremetal-benchmarks.sh
```

The bare-metal runner starts `ray-serve/serve_app.py` and
`ray-serve/serve_app_grpc.py` directly with uv, tears Ray down between runs, and
writes each benchmark into an isolated timestamped directory. On Runpod, the
default result path is `/workspace/gliner-guard-results/a100-baremetal-*`; on
local machines it is `results-runpod/a100-baremetal-*`.

If the Runpod `/workspace` volume returns `Stale file handle` while installing
Python wheels, keep the checked-out code and `.venv` on the root disk, for
example `/root/gliner-guard-serve`, while leaving `HF_HOME` and `RESULT_DIR`
under `/workspace`.

Expected Ray-only full run count:

- `3` batch sizes
- `2` protocols: Ray Serve REST and Ray Serve gRPC
- `3` repeats
- total: `18` benchmark runs

For a quick smoke run before the full A100 benchmark:

```bash
ENV_FILE=.env.ray-a100-4w.example \
REPEATS=1 LOCUST_RUN_TIME=2m LOCUST_USERS=10 \
  ./scripts/run-a100-4worker-rest-benchmarks.sh
```

Expected full run count:

- `3` batch sizes
- `2` frameworks: LitServe REST and Ray Serve REST
- `3` repeats
- total: `18` benchmark runs

## Result Files

Each run writes:

- `results/litserve-rest-B{16,32,64}-uni-prompts-runN_stats.csv`
- `results/litserve-rest-B{16,32,64}-uni-prompts-runN_stats_history.csv`
- `results/ray-rest-B{16,32,64}-uni-prompts-runN_stats.csv`
- `results/ray-rest-B{16,32,64}-uni-prompts-runN_stats_history.csv`
- matching `.html` Locust reports
- matching `results/gpu-*.csv` `nvidia-smi` samples
- generated history plots:
  `results/{litserve,ray}-rest-B{16,32,64}-uni-prompts-history.png`

## Monitoring

For a long Runpod run, keep a status snapshot next to the result artifacts:

```bash
RESULT_DIR=/workspace/gliner-guard-results/a100-full-YYYYMMDDTHHMMSSZ \
EXPERIMENT_PID=<runner-pid> \
INTERVAL_SECONDS=300 \
  nohup bash scripts/watch-a100-experiment.sh \
  > "$RESULT_DIR/monitor/watchdog.out" 2>&1 &
```

The watchdog writes:

- `$RESULT_DIR/monitor/status.json`
- `$RESULT_DIR/monitor/status.md`
- `$RESULT_DIR/monitor/status-history.jsonl`
- `$RESULT_DIR/monitor/watchdog.log`

One-shot status check:

```bash
python3 scripts/monitor_a100_experiment.py \
  --result-dir /workspace/gliner-guard-results/a100-full-YYYYMMDDTHHMMSSZ \
  --pid <runner-pid> \
  --write-status
```

The PR should include the generated result CSV/HTML only when they are intended
as evidence for the benchmark result. Otherwise, keep this as a runnable
benchmark config PR.

## Current Best Starting Point

Run the full A100 PR sweep in this order:

1. LitServe B64, B32, B16 baseline with `workers_per_device=4`.
2. Ray Serve REST B64, B32, B16 with `NUM_REPLICAS=4` and
   `NUM_GPUS_PER_REPLICA=0.25`.
3. Pick the winner by mean RPS first, then P95 latency, with zero failures.

My prior expectation for A100 is:

- B64 may win raw throughput if GPU memory and padding overhead are fine.
- B32 is the safest likely default for mixed-length prompts.
- B16 is the latency-protecting fallback if B64/B32 have bad P95.

Do not select a final "best" config from the time-sliced dev GPU results. Those
results are useful only for functional validation.

## Full-History Plot

The benchmark runner passes Locust `--csv-full-history`, which produces
`*_stats_history.csv` files. The plot generator is:

```bash
uv run --with pandas --with matplotlib --with seaborn \
  python scripts/plot_locust_history.py \
  --glob 'results/ray-rest-B64-uni-prompts-run*_stats_history.csv' \
  --output results/ray-rest-B64-uni-prompts-history.png \
  --title 'Ray REST B64 uni prompts' \
  --max-users 100 \
  --latency-percentile '95%' \
  --throughput-scale 1000
```

The generated graph follows the example notebook:

- x-axis: `User Count`
- throughput: `Requests/s`
- latency: `95%`
- confidence band: mean ± std across repeats
- optimal marker: max `mean_throughput / mean_latency`

## gRPC Low-RPS Investigation

Ray Serve's public `gRPCOptions` currently exposes port, servicer functions, and
request timeout. It does not expose a general "increase server worker threads"
knob for the external gRPC proxy. In the Ray source used here, the proxy starts
`grpc.aio.server` with `maximum_concurrent_rpcs=None`, so the first suspect is
not a hard gRPC server concurrency cap.

The observed pattern, low latency around `100-200ms` but unchanged RPS, is most
likely a load-generator artifact until proven otherwise:

- Locust uses gevent greenlets.
- `grpcio` uses C-core I/O that is not automatically gevent-cooperative.
- Without `grpc.experimental.gevent.init_gevent()`, one blocking gRPC call can
  effectively block the Locust worker process, capping RPS while per-call
  latency still looks low.

This PR patches the gRPC Locust file to initialize gRPC gevent compatibility
before creating channels. Re-run gRPC after this patch before tuning Ray Serve.

If RPS is still low after the client patch:

1. Increase client-side load generator capacity or run distributed Locust.
2. Compare with a small `grpc.aio` benchmark outside Locust.
3. Check Ray Serve metrics:
   `serve_num_ongoing_grpc_requests`, `serve_deployment_processing_latency_ms`,
   `serve_batch_wait_time_ms`.
4. Check GPU utilization and batch fill ratio.

If GPU utilization is already saturated, gRPC can reduce transport overhead and
latency but should not be expected to increase model-bound throughput.

## References

- Ray Serve dynamic request batching:
  https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html
- Ray Serve gRPC guide:
  https://docs.ray.io/en/latest/serve/advanced-guides/grpc-guide.html
- Ray Serve resource allocation:
  https://docs.ray.io/en/latest/serve/resource-allocation.html
- Ray Serve performance tuning:
  https://docs.ray.io/en/latest/serve/advanced-guides/performance.html
- Locust gRPC guidance:
  https://docs.locust.io/en/stable/testing-other-systems.html#grpc

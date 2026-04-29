# A100 Ray Serve UNI Benchmark Results, 2026-04-29

This document preserves the PR evidence for the Ray Serve A100 run.

## Scope

- Hardware: Runpod `NVIDIA A100-SXM4-80GB`
- Model family: GLiNER guard uniencoder only (`uni`)
- Frameworks: Ray Serve REST and Ray Serve gRPC
- Workers: `4` Ray Serve replicas on one A100
- GPU placement: `NUM_REPLICAS=4`, `NUM_GPUS_PER_REPLICA=0.25`
- Batch sizes: `16`, `32`, `64`
- Repeats: `3` per protocol/batch-size pair
- Total benchmark runs: `18`
- Load profile: Locust `100` users, `1` user/s spawn rate, `15m` run
- Excluded from this run: LitServe baseline, biencoder (`bi`), Docker Compose

The run was executed bare-metal on Runpod because Docker Compose was not
available. The runner used the Ray-only path documented in
`docs/ray-serve-a100-4worker-config.md`.

## Artifact Location

Full copied result directory:

```text
results-runpod/nvidia-a100-sxm4-80gb/ray-serve-uni-a100-full-20260429T104754Z
```

Original VM result directory:

```text
/workspace/gliner-guard-results/a100-full-20260429T104754Z
```

Sibling VM files also copied into the result directory:

```text
/workspace/gliner-guard-results/a100-full-20260429T104754Z.driver.log
/workspace/gliner-guard-results/a100-full-20260429T104754Z.pid
```

## Artifact Inventory

| Artifact type | Count |
|---|---:|
| Total files committed in this result directory | 226 |
| `summary.csv` | 1 |
| `MANIFEST.sha256` checksum manifest | 1 |
| `FILE_INVENTORY.txt` byte-size inventory | 1 |
| Driver log copied from VM sibling path | 1 |
| Runner PID copied from VM sibling path | 1 |
| VM source snapshot files | 28 |
| Locust `*_stats.csv` | 18 |
| Locust `*_stats_history.csv` | 18 |
| Locust `*_failures.csv` | 18 |
| Locust `*_exceptions.csv` | 18 |
| Locust HTML reports | 18 |
| GPU CSV samples | 18 |
| History plots | 6 |
| Benchmark run logs | 54 |
| Benchmark server PID files | 18 |
| Monitor status/history files | 6 |

`source-snapshot/` preserves the compact VM source state for the runner, Ray
Serve apps, gRPC proto/stub files, and benchmark scripts. This includes the
generated `ray-serve/gliner_guard_pb2*.py` files that were present on the VM.

`MANIFEST.sha256` contains checksums for the 224 copied run/source evidence
files, excluding only `MANIFEST.sha256` and `FILE_INVENTORY.txt` themselves.

History plots:

- `ray-rest-B16-uni-prompts-history.png`
- `ray-grpc-B16-uni-prompts-history.png`
- `ray-rest-B32-uni-prompts-history.png`
- `ray-grpc-B32-uni-prompts-history.png`
- `ray-rest-B64-uni-prompts-history.png`
- `ray-grpc-B64-uni-prompts-history.png`

## Aggregate Results

All runs completed with `0` Locust failures.

| Protocol | Batch | Runs | Mean RPS | RPS std | Mean P50, ms | Mean P95, ms | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| gRPC | 16 | 3 | 156.47 | 1.27 | 600.2 | 2005.7 | 0 |
| REST | 16 | 3 | 147.22 | 0.40 | 636.3 | 1850.8 | 0 |
| gRPC | 32 | 3 | 139.03 | 1.91 | 672.4 | 3163.8 | 0 |
| REST | 32 | 3 | 137.42 | 1.32 | 680.4 | 3360.9 | 0 |
| gRPC | 64 | 3 | 138.48 | 5.43 | 678.5 | 4040.7 | 0 |
| REST | 64 | 3 | 141.43 | 1.10 | 662.6 | 3464.5 | 0 |

## Per-Run Results

| Run | RPS | P50, ms | P95, ms | Failures |
|---|---:|---:|---:|---:|
| ray-rest-B16-run1 | 147.03 | 638.52 | 1356.65 | 0 |
| ray-grpc-B16-run1 | 155.45 | 605.15 | 2804.18 | 0 |
| ray-rest-B16-run2 | 147.68 | 634.28 | 2089.43 | 0 |
| ray-grpc-B16-run2 | 157.90 | 594.14 | 1265.38 | 0 |
| ray-rest-B16-run3 | 146.97 | 636.10 | 2106.42 | 0 |
| ray-grpc-B16-run3 | 156.08 | 601.29 | 1947.61 | 0 |
| ray-rest-B32-run1 | 138.46 | 675.33 | 1802.05 | 0 |
| ray-grpc-B32-run1 | 140.05 | 668.52 | 2692.82 | 0 |
| ray-rest-B32-run2 | 137.85 | 678.32 | 2550.71 | 0 |
| ray-grpc-B32-run2 | 140.20 | 666.23 | 1268.12 | 0 |
| ray-rest-B32-run3 | 135.93 | 687.63 | 5729.99 | 0 |
| ray-grpc-B32-run3 | 136.82 | 682.40 | 5530.34 | 0 |
| ray-rest-B64-run1 | 140.73 | 663.54 | 5965.39 | 0 |
| ray-grpc-B64-run1 | 140.79 | 666.76 | 5004.82 | 0 |
| ray-rest-B64-run2 | 140.86 | 665.16 | 2838.53 | 0 |
| ray-grpc-B64-run2 | 142.38 | 660.12 | 1426.28 | 0 |
| ray-rest-B64-run3 | 142.70 | 659.11 | 1589.69 | 0 |
| ray-grpc-B64-run3 | 132.28 | 708.52 | 5690.91 | 0 |

## Readout

- The best mean throughput in this run is Ray Serve gRPC B16:
  `156.47` RPS, mean P50 `600.2 ms`, mean P95 `2005.7 ms`.
- REST B16 is the closest REST point:
  `147.22` RPS, mean P50 `636.3 ms`, mean P95 `1850.8 ms`.
- B64 does not win on this A100 run. REST B64 slightly beats gRPC B64 on mean
  RPS, while both show noisy high-tail latency in individual repeats.
- Because this run is `uni` only, do not use it as evidence for `bi`.
- Because LitServe was intentionally not run here, this PR evidence is for the
  Ray Serve REST vs gRPC A100 sweep only.

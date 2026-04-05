# Ray Serve Dynamic Batching — Dev GPU Results

> **Phase:** 2 (Dynamic Batching Sweep)
> **Date:** 2026-04-05
> **GPU:** NVIDIA RTX 5070 Ti 16GB, 1/8 time-sliced via K3s GPU Operator (~2GB effective VRAM)
> **Model:** `hivetrace/gliner-guard-uniencoder` (147M params, DeBERTa-v2 backbone)
> **Load:** 20 Locust users, 15 min per config, `prompts.csv` (500 rows, ~2500 chars avg)
> **Framework:** Ray Serve 2.46.0, PyTorch bf16, `@serve.batch` decorator

---

## Implementation

Added `@serve.batch` support to `ray-serve/serve_app.py` via env var toggle:

- `MAX_BATCH_SIZE=0` → no batching (uses `model.extract()`, backward compatible with Phase 1)
- `MAX_BATCH_SIZE>0` → batching enabled (uses `model.batch_extract()`, `@serve.batch` decorator)
- `BATCH_WAIT_TIMEOUT` → controls how long Ray waits to collect a full batch (seconds)

The deployment class is selected at module load time via `_build_deployment()` factory.

---

## Results: B1–B4 Sweep (dev GPU, uniencoder)

| Config | max_batch_size | batch_wait_timeout_s | RPS | P50 (ms) | P95 (ms) | Errors |
|--------|:-------------:|:-------------------:|----:|--------:|---------:|-------:|
| **no-batch** | — | — | **4.8** | **4,124** | **5,780** | 0 |
| B1 | 8 | 0.01 | 3.2 | 6,093 | 8,359 | 0 |
| B2 | 16 | 0.05 | 2.7 | 7,373 | 14,645 | 0 |
| B3 | 32 | 0.05 | 2.6 | 7,739 | 14,298 | 0 |
| B4 | 64 | 0.10 | 2.6 | 7,609 | 13,994 | 0 |

### Observations

1. **Batching is slower than no-batch on this GPU.** All B1–B4 configs produce lower RPS (2.6–3.2) vs no-batch (4.8). This is expected — see analysis below.

2. **Smaller batch + tighter timeout = less degradation.** B1 (batch=8, timeout=10ms) is closest to no-batch performance. Larger batches (B3, B4) plateau at ~2.6 RPS.

3. **P95 latency spikes with larger batches.** B2 P95 (14.6s) is 2.5× the no-batch P95 (5.8s). B3 and B4 are similar (~14s), suggesting the bottleneck is GPU processing time, not batch collection.

4. **Zero errors across all configs.** No OOMs, no timeouts, no HTTP errors. The `RAY_memory_monitor_refresh_ms=0` setting prevents Ray's OOM killer from interfering.

### Why Batching Hurts on Dev GPU

The dev GPU is **1/8 time-sliced** via NVIDIA MPS/time-slicing through K3s GPU Operator:

- **~2GB effective VRAM** — batch processing needs to fit all texts' tensors simultaneously. Larger batches cause more memory pressure and potential GPU memory swapping.
- **~12.5% compute** — the GPU's CUDA cores are shared 8 ways. Batch parallelism can't help when compute is the bottleneck (not scheduling overhead).
- **Only 20 concurrent users** — with low concurrency, batches rarely fill up. A `max_batch_size=64` with 20 users means batches are small anyway, but the timeout wait adds pure latency.
- **`batch_extract()` overhead** — even if batch size is small, `batch_extract()` has collation/padding overhead that `extract()` (single text) avoids.

### Expected on Full GPU

On a dedicated A100/H100 with full VRAM and compute:
- Larger batches should show **throughput improvement** as GPU can parallelize across batch dimension
- The crossover point (batch > no-batch) likely requires >50 concurrent users to keep batches full
- Optimal config is expected to be B2 or B3 range (16–32 batch size)

---

## Files

| File | Description |
|------|-------------|
| `results/ray-rest-B{1-4}-uni-prompts-run1_stats.csv` | Locust stats per config |
| `results/ray-rest-B{1-4}-uni-prompts-run1.html` | Locust HTML reports |
| `results/gpu-ray-rest-B{1-4}-uni-prompts-run1.csv` | GPU metrics (nvidia-smi, 1s interval) |
| `ray-serve/serve_app.py` | Updated with `@serve.batch` support |
| `scripts/run-batch-benchmarks.sh` | Automated sweep script |

---

## Next Steps

- [ ] Run same sweep on cloud VM with full GPU (A100/H100)
- [ ] Test with 100 users (expect batching to outperform no-batch at higher concurrency)
- [ ] Expand to biencoder model
- [ ] Add token-aware batching (B9 config from experiment plan)

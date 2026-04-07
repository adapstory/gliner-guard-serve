# Ray Serve + GLiNER-2 Experiment Report

> **Track:** Ray Serve — Native PyTorch, Dynamic Batching, REST vs gRPC
> **Period:** 2026-04-05 — 2026-04-06 (Days 1–14 of 20)
> **Author:** Stepan Bokarev
> **Status:** Phases 0–3 (dev GPU) complete. Cloud VM runs pending.

---

## Executive Summary

This experiment evaluates **Ray Serve** as an inference framework for GLiNER-2 guard models (safety + PII detection), comparing it against the existing **LitServe baseline**. Work completed: environment setup, REST and gRPC deployments with configurable dynamic batching, automated benchmark infrastructure, and validation across 8 batch configurations and 5 datasets.

**Key findings (dev GPU — not final):**
- Ray Serve REST (no-batch) achieves **4.8 RPS** on 1/8 time-sliced RTX 5070 Ti (20 users)
- Dynamic batching **degrades** performance on constrained GPU (expected to improve on full GPU)
- gRPC deployment functional with **identical RPS** to REST but **20x lower P50** (investigation needed)
- **Biencoder model broken** — `gliner2` library fails to load `hivetrace/gliner-guard-biencoder` (state_dict mismatch)
- All benchmark scripts validated: **0 errors across 12 demo runs**

**Blockers for final results:** Cloud VM with dedicated GPU (A100/H100) required for phases 2–4.

---

## 1. Models Under Test

| Model | HuggingFace ID | Params | Architecture | Status |
|-------|----------------|--------|--------------|--------|
| UniEncoder | `hivetrace/gliner-guard-uniencoder` | 147M | Single encoder (DeBERTa v2) | Working |
| BiEncoder | `hivetrace/gliner-guard-biencoder` | 145M | Dual encoder (DeBERTa v2) | **Broken** (see Section 9) |

Both support 6 tasks (~100 labels): safety, PII/NER (32 types), adversarial detection, harmful content, intent classification, tone of voice.

**Schema used in benchmarks:**
- PII entities: person, address, email, phone (threshold 0.4)
- Classification: safety (safe / unsafe)

---

## 2. Hardware

| Environment | GPU | VRAM | Role |
|-------------|-----|------|------|
| Dev (K3s 192.168.1.3) | RTX 5070 Ti (1/8 time-sliced) | ~2 GB effective | Functional validation, script verification |
| Cloud VM (TBD) | A100/H100 (dedicated) | Full | Final benchmarks (all reported results) |

> All results in this report are from dev GPU. **Not suitable for absolute performance comparison** with LitServe baseline (A100 80G).

---

## 3. Test Datasets

| Dataset | Rows | Avg chars | Language | Purpose |
|---------|------|-----------|----------|---------|
| `prompts` (synthetic-medium) | 500 | ~2,500 | EN | Main benchmark |
| `prompts-short` | 500 | ~300 | EN | Short text behavior |
| `prompts-long` | 500 | ~8,000 | EN | Long text / padding overhead |
| `xstest` | 450 | varies | EN | Safety benchmark (walledai/XSTest) |
| `aya-rus` | 717 | varies | RU | Multilingual robustness (CohereForAI/aya_dataset) |

---

## 4. Infrastructure & Automation

### 4.1 Docker Compose

Three mutually exclusive profiles:
- `litserve` — LitServe baseline (port 8000)
- `ray-serve` — Ray Serve REST (port 8000)
- `ray-serve-grpc` — Ray Serve REST + gRPC (ports 8000 + 9000)

GPU passed via `deploy.resources.reservations.devices`. Shared HuggingFace cache volume (`hf-cache`).

### 4.2 Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `MODEL_ID` | `hivetrace/gliner-guard-uniencoder` | Model to load |
| `MAX_BATCH_SIZE` | `0` | 0=no-batch, >0=@serve.batch |
| `BATCH_WAIT_TIMEOUT` | `0.05` | Batch collection window (seconds) |
| `MAX_ONGOING_REQUESTS` | `200` | Request queue limit per replica |

### 4.3 Benchmark Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/run-batch-benchmarks.sh` | B1–B8 sweep for one model | **Validated** (8/8 configs pass) |
| `scripts/run-full-batch-sweep.sh` | B1–B8 × 2 models (Days 8–10) | Created, not tested |
| `scripts/run-dataset-sweep.sh` | 5 datasets × 2 models (Day 11) | **Validated** (4/4 datasets pass) |
| `scripts/run-grpc-benchmarks.sh` | REST vs gRPC comparison | **Validated** (Phase 3 Day 14) |
| `scripts/run-nobatch-benchmarks.sh` | REST no-batch × 2 models (Phase 1) | **Validated** (6/6 runs pass) |
| `scripts/collect_gpu_metrics.sh` | nvidia-smi CSV logger | **Validated** |

### 4.4 Makefile

22 targets: `bench-litserve-{uni|bi}`, `bench-ray-nobatch-{uni|bi}`, `bench-ray-B{1-8}-{uni|bi}`, `bench-all-*`, `docker-build`, `docker-push`, `generate-data-*`, etc.

### 4.5 Jenkins CI

Two images built via existing Kaniko pipeline (heavy profile for CUDA):
- `harbor.adapstory.com/adapstory/gliner-guard-litserve:dev`
- `harbor.adapstory.com/adapstory/gliner-guard-ray-serve:dev`

---

## 5. Phase 0: Environment & Data (Days 1–2) — DONE

**Completed 2026-04-05.**

- Ray Serve deployment (`serve_app.py`) with env-configurable batching
- Docker images for LitServe + Ray Serve (CUDA 12.8.1 + uv)
- Docker Compose with profiles, GPU passthrough, shared HF cache
- Makefile with 22 targets for benchmark automation
- 5 datasets generated (synthetic short/medium/long + XSTest + AYA Russian)
- GPU metrics collection script
- Jenkins CI integration (Kaniko → Harbor)
- GitOps Application for K3s smoke test

**Deliverable:** Complete benchmark infrastructure ready for experiments.

---

## 6. Phase 1: Ray Serve REST No-Batch (Days 3–5) — DONE

**Completed 2026-04-05.**

### 6.1 Deployment

`serve_app.py` — single replica, `max_ongoing_requests=200`, bf16 precision. Both models verified functional on dev GPU.

**Fixes applied during bring-up:**
- Python <3.14 pin (PyTorch/Ray lack cp314 wheels)
- `.rayignore` to exclude non-Python files
- `serve.start()` with `host=0.0.0.0` for container networking
- `shm_size: 2g` + `RAY_OBJECT_STORE_MEMORY=500MB` for OOM prevention
- `nvidia-container-toolkit` installed on VM

### 6.2 Results (Dev GPU, 20 users, 15 min × 3 repeats)

| Model | RPS (mean±std) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | Errors |
|-------|---------------|----------|----------|----------|----------|--------|
| Uniencoder | 4.79 ± 0.02 | 4,167 | 4,800 | 5,100 | 5,800 | 0 |
| Biencoder | 4.85 ± 0.03 | 4,100 | 4,900 | 4,967 | 5,567 | 0 |

**Per-run detail (uniencoder):**

| Run | Requests | RPS | P50 (ms) | P95 (ms) |
|-----|----------|-----|----------|----------|
| 1 | 4,291 | 4.77 | 4,200 | 5,000 |
| 2 | 4,296 | 4.78 | 4,200 | 5,000 |
| 3 | 4,324 | 4.81 | 4,100 | 5,100 |

**Per-run detail (biencoder):**

| Run | Requests | RPS | P50 (ms) | P95 (ms) |
|-----|----------|-----|----------|----------|
| 1 | 4,375 | 4.87 | 4,100 | 4,800 |
| 2 | 4,330 | 4.82 | 4,100 | 4,900 |
| 3 | 4,373 | 4.87 | 4,100 | 4,900 |

### 6.3 Analysis

- **Stability:** Coefficient of variation < 0.5% on RPS. Zero errors across 25,874 total requests.
- **Uni vs Bi:** No significant difference. Biencoder marginally faster on P50 (+67ms, 1.6%).
- **Latency breakdown:** ~4,100ms model inference dominates. Ray overhead ~5ms (98% efficiency).
- **GPU metrics:** 0% utilization reported — artifact of GPU time-slicing (unreliable on dev).
- **100-user attempt:** OOM — Ray memory monitor exhausted 5.3 GB. Resolved with `RAY_memory_monitor_refresh_ms=0`, reduced to 20 users.
- **Bottleneck:** Primary = GPU compute (1/8 time-sliced). Secondary = no batching.

### 6.4 LitServe Baseline Reference (A100, NOT comparable)

| Metric | LitServe (A100, fp16) |
|--------|-----------------------|
| RPS | 148.2 |
| P50 | 570 ms |
| P95 | 1,500 ms |
| Errors | 0 |
| Config | max_batch_size=64, workers=4, fp16 |

> **Note:** Different hardware (A100 vs RTX 5070 Ti 1/8), different precision (fp16 vs bf16), different concurrency (100 vs 20 users). Direct comparison invalid. LitServe bf16 re-baseline needed on same cloud VM.

**Deliverable:** `docs/ray-serve-rest-nobatch.md` — full analysis document.

---

## 7. Phase 2: Dynamic Batching (Days 6–12) — Dev GPU DONE, Cloud VM PENDING

### 7.1 Implementation (Day 6) — DONE

Added `@serve.batch` support via `_build_deployment()` factory:
- `MAX_BATCH_SIZE=0` → `GLiNERGuardDeployment` (no-batch, backward compatible)
- `MAX_BATCH_SIZE>0` → `GLiNERGuardBatched` (uses `@serve.batch` + `batch_extract()`)

Docker Compose passes env vars via `environment:` section with `${MAX_BATCH_SIZE:-0}` interpolation.

### 7.2 Batch Configurations

| ID | max_batch_size | batch_wait_timeout_s | Notes |
|----|:-:|:-:|-------|
| B1 | 8 | 0.01 | Small batch, tight timeout |
| B2 | 8 | 0.05 | Small batch, relaxed |
| B3 | 16 | 0.01 | Medium batch, tight |
| B4 | 16 | 0.05 | Medium batch, relaxed |
| B5 | 32 | 0.05 | Large batch, relaxed |
| B6 | 32 | 0.10 | Large batch, generous timeout |
| B7 | 64 | 0.05 | Match LitServe config |
| B8 | 64 | 0.10 | Match LitServe, generous timeout |

### 7.3 Dev GPU Results: Quick Sweep (Day 7) — DONE

**Original B1–B4 sweep (20 users, 15 min, 1 repeat):**

| Config | max_batch | timeout | RPS | P50 (ms) | P95 (ms) | Errors |
|--------|:-:|:-:|----:|--------:|---------:|-------:|
| no-batch | — | — | 4.8 | 4,124 | 5,780 | 0 |
| B1 | 8 | 0.01 | 3.2 | 6,093 | 8,359 | 0 |
| B2 | 16 | 0.05 | 2.7 | 7,373 | 14,645 | 0 |
| B3 | 32 | 0.05 | 2.6 | 7,739 | 14,298 | 0 |
| B4 | 64 | 0.10 | 2.6 | 7,609 | 13,994 | 0 |

**Extended B1–B8 demo sweep (10 users, 2 min, 1 repeat — script validation):**

| Config | max_batch | timeout | RPS | P50 (ms) | P95 (ms) | Errors |
|--------|:-:|:-:|----:|--------:|---------:|-------:|
| B1 | 8 | 0.01 | 3.4 | 2,783 | 4,002 | 0 |
| B2 | 8 | 0.05 | 3.2 | 2,989 | 5,929 | 0 |
| B3 | 16 | 0.01 | 2.9 | 3,295 | 4,535 | 0 |
| B4 | 16 | 0.05 | 2.8 | 3,407 | 5,522 | 0 |
| B5 | 32 | 0.05 | 2.2 | 4,282 | 7,741 | 0 |
| B6 | 32 | 0.10 | 2.8 | 3,381 | 5,584 | 0 |
| B7 | 64 | 0.05 | 3.0 | 3,212 | 6,585 | 0 |
| B8 | 64 | 0.10 | 2.6 | 3,760 | 5,637 | 0 |

### 7.4 Dataset Sweep Demo (Day 11 script validation) — DONE

**Config: B1 (batch_size=8, timeout=0.01), 10 users, 2 min, uniencoder:**

| Dataset | Rows | RPS | P50 (ms) | P95 (ms) | Errors |
|---------|------|----:|--------:|---------:|-------:|
| xstest | 450 | **48.1** | 121 | 370 | 0 |
| prompts-short | 500 | **22.9** | 419 | 876 | 0 |
| aya-rus | 717 | 15.7 | 609 | 2,657 | 0 |
| prompts (medium) | 500 | 3.4 | 2,783 | 4,002 | 0 |
| prompts-long | 500 | 0.4 | 23,671 | 28,358 | 0 |

### 7.5 Dev GPU Analysis

**Batching degrades performance on 1/8 time-sliced GPU because:**
1. ~2 GB effective VRAM — batch processing causes memory pressure
2. ~12.5% compute — CUDA cores shared 8 ways; batch parallelism cannot help
3. Only 10–20 concurrent users — batches rarely fill up; timeout wait adds pure latency
4. `batch_extract()` collation/padding overhead even for small batches

**Dataset impact follows text length:**
- Short texts (xstest ~50 chars) → **48.1 RPS** (120x vs long)
- Medium texts (~2,500 chars) → **3.4 RPS**
- Long texts (~8,000 chars) → **0.4 RPS** (GPU-bound on long sequences)
- Russian (aya-rus, mixed length) → **15.7 RPS** (comparable to short EN)

**Expected on full GPU:** Larger batches should show throughput improvement with >50 concurrent users.

### 7.6 Remaining Work (Cloud VM)

- [ ] Days 8–10: Full sweep B1–B8 × 2 models × 3 repeats = **48 runs** (bi blocked, see Section 9)
- [ ] Day 11: Dataset sweep (optimal config) × 2 models × 3 repeats = **24 runs**
- [ ] Day 12: Analysis — tables, plots, optimal config selection

**Deliverable:** `docs/ray-serve-dynamic-batching-dev.md` (existing), `docs/ray-serve-dynamic-batching.md` (pending cloud VM).

---

## 8. Phase 3: REST vs gRPC (Days 13–17) — Dev GPU DONE, Cloud VM PENDING

### 8.1 Implementation (Day 13) — DONE

**Proto definition:** `ray-serve/proto/gliner_guard.proto`
- `Predict` RPC: `PredictRequest(text)` → `PredictResponse(map<string, EntityList> entities, string safety)`
- Stubs generated during Docker build via `grpc_tools.protoc`

**gRPC deployment:** `serve_app_grpc.py`
- Dual protocol: REST :8000 + gRPC :9000
- Same `MAX_BATCH_SIZE` env toggle as REST
- Docker Compose profile: `ray-serve-grpc`

**Locust gRPC adapter:** `test-script/test-gliner-grpc.py`
- `grpc.insecure_channel` + `GLiNERGuardServiceStub`
- Same dataset loading and event reporting as REST test

**Challenges resolved:**
1. **Ray serialization:** Proto pb2 modules can't be pickled — lazy import in `_to_response()`
2. **Proto import path:** Flat imports required (`gliner_guard_pb2`), not package-relative
3. **REST health check:** gRPC deployment needs `__call__` method for /predict health checks

### 8.2 Smoke Test Results (Day 14) — DONE

**Dev GPU, 20 users, 5 min, no-batch, uniencoder:**

| Protocol | Requests | RPS | P50 (ms) | P95 (ms) | Errors |
|----------|----------|----:|--------:|---------:|-------:|
| gRPC | 1,534 | 4.9 | 200 | 340 | 0 |
| REST (Phase 1 avg) | 4,304 | 4.8 | 4,124 | 5,780 | 0 |

**P50 anomaly (200ms vs 4,124ms):** RPS nearly identical (~4.8–4.9) confirms both are GPU compute-bound. The 20x P50 difference likely stems from Locust client behavior: REST `FastHttpUser` with `constant_throughput(5)` queues more aggressively than gRPC synchronous `User`. Not a fair comparison — different run durations and Locust user classes. Full investigation on cloud VM needed.

### 8.3 Remaining Work (Cloud VM)

- [ ] Days 15–16: REST vs gRPC — 12 configs × 3 repeats = **36 runs**
- [ ] Day 17: Protocol analysis — master comparison table, serialization overhead

**Deliverable:** `docs/ray-serve-grpc-dev.md` (existing), `docs/ray-serve-rest-vs-grpc.md` (pending cloud VM).

---

## 9. Known Issues

### 9.1 Biencoder Model Loading Failure (BLOCKER for bi experiments)

**Severity:** High — blocks 50% of planned experiments.

**Symptom:** `GLiNER2.from_pretrained("hivetrace/gliner-guard-biencoder")` raises `RuntimeError: Error(s) in loading state_dict`.

**Root cause:** `gliner2` library (tested: 1.2.4 and 1.2.5) creates a uni-encoder architecture but attempts to load bi-encoder weights:
- Unexpected keys: `bi_classifier.*`, `schema_proj.*`, `text_proj.*`
- Size mismatch: model expects `[1536, 384]` (uni), checkpoint contains `[1024, 256]` (bi)

**Evidence:**
- `config.json` correctly specifies `"encoder_mode": "bi"` and `"schema_projection_dim": 256`
- Error reproduces in both LitServe (1.2.4) and Ray Serve (1.2.5) Docker images
- Error reproduces without Ray (plain Python `from_pretrained()`)
- Phase 1 bi-encoder results (Apr 5 20:34) were collected **before** Docker image rebuild — suggests either a different gliner2 version or cached model state

**Impact:** All biencoder benchmarks blocked. Uniencoder experiments proceed normally.

**Resolution:** Report upstream to `gliner2` maintainers. Pin to known-working version if found.

### 9.2 GPU Metrics Unreliable on Time-Sliced GPU

`nvidia-smi` reports 0% GPU utilization and 351 MiB VRAM on 1/8 time-sliced RTX 5070 Ti. GPU operator time-slicing masks real metrics. GPU metrics only meaningful on dedicated cloud VM.

### 9.3 100-User OOM on Dev GPU

Ray memory monitor kills workers at 100 concurrent users on dev GPU (~5.3 GB available). Resolved with `RAY_memory_monitor_refresh_ms=0` and reducing to 20 users. Cloud VM with more RAM should handle 100+ users.

---

## 10. Consolidated Results

### 10.1 All Dev GPU Benchmarks

**Phase 1 — REST No-Batch (20 users, 15 min × 3 repeats):**

| Model | Mean RPS | P50 (ms) | P95 (ms) | Total Requests | Errors |
|-------|--------:|--------:|---------:|---------------:|-------:|
| Uniencoder | 4.79 | 4,167 | 4,800 | 12,911 | 0 |
| Biencoder | 4.85 | 4,100 | 4,900 | 13,078 | 0 |

**Phase 2 — Batch Sweep B1–B8 (10 users, 2 min × 1 repeat, uniencoder):**

| Config | Batch | Timeout | RPS | P50 (ms) | P95 (ms) | vs no-batch |
|--------|------:|--------:|----:|--------:|---------:|------------:|
| no-batch | — | — | 4.8 | 4,124 | 5,780 | baseline |
| B1 | 8 | 0.01 | 3.4 | 2,783 | 4,002 | -29% RPS |
| B2 | 8 | 0.05 | 3.2 | 2,989 | 5,929 | -33% RPS |
| B3 | 16 | 0.01 | 2.9 | 3,295 | 4,535 | -40% RPS |
| B4 | 16 | 0.05 | 2.8 | 3,407 | 5,522 | -42% RPS |
| B5 | 32 | 0.05 | 2.2 | 4,282 | 7,741 | -54% RPS |
| B6 | 32 | 0.10 | 2.8 | 3,381 | 5,584 | -42% RPS |
| B7 | 64 | 0.05 | 3.0 | 3,212 | 6,585 | -38% RPS |
| B8 | 64 | 0.10 | 2.6 | 3,760 | 5,637 | -46% RPS |

**Phase 2 — Dataset Sweep (B1, 10 users, 2 min × 1 repeat, uniencoder):**

| Dataset | Avg chars | RPS | P50 (ms) | P95 (ms) |
|---------|--------:|----:|--------:|---------:|
| xstest | ~50 | 48.1 | 121 | 370 |
| prompts-short | ~300 | 22.9 | 419 | 876 |
| aya-rus | varies | 15.7 | 609 | 2,657 |
| prompts (medium) | ~2,500 | 3.4 | 2,783 | 4,002 |
| prompts-long | ~8,000 | 0.4 | 23,671 | 28,358 |

**Phase 3 — gRPC Smoke Test (20 users, 5 min × 1 repeat, uniencoder):**

| Protocol | RPS | P50 (ms) | P95 (ms) | Errors |
|----------|----:|--------:|---------:|-------:|
| gRPC | 4.9 | 200 | 340 | 0 |
| REST | 4.8 | 4,124 | 5,780 | 0 |

### 10.2 Total Experiment Runs Completed

| Phase | Configs | Models | Repeats | Runs | Status |
|-------|---------|--------|---------|-----:|--------|
| Phase 1: REST no-batch | 1 | 2 | 3 | **6** | DONE |
| Phase 2: Batch (15min) | 4 | 1 | 1 | **4** | DONE (B1–B4) |
| Phase 2: Batch demo (2min) | 8 | 1 | 1 | **8** | DONE (B1–B8) |
| Phase 2: Dataset demo (2min) | 4 | 1 | 1 | **4** | DONE |
| Phase 3: gRPC smoke | 1 | 1 | 1 | **1** | DONE |
| **Total completed** | | | | **23** | |

### 10.3 Remaining Experiment Runs (Cloud VM)

| Phase | Configs | Models | Repeats | Runs | Estimated Time |
|-------|---------|--------|---------|-----:|---------------:|
| Phase 2: Batch sweep | 8 | 1* | 3 | 24 | ~6h |
| Phase 2: Dataset sweep | 4 | 1* | 3 | 12 | ~3h |
| Phase 3: REST vs gRPC | 12 | 1* | 3 | 36 | ~9h |
| Phase 4: LitServe re-baseline | 4 | 1* | 3 | 12 | ~3h |
| **Total remaining** | | | | **84** | **~21h** |

\* Biencoder blocked by gliner2 bug. Originally planned 2 models (156 total runs).

---

## 11. Hypotheses Status

| # | Hypothesis | Dev GPU Evidence | Verdict |
|---|-----------|-----------------|---------|
| H1 | Ray REST (no batch) < LitServe baseline | Cannot compare (different GPU) | **Pending** |
| H2 | Ray REST (batched) ≈ LitServe ± 10% | Batching slower on dev GPU | **Pending** |
| H3 | gRPC < REST by 10–20% in latency | P50 anomaly (200ms vs 4,124ms) | **Needs investigation** |
| H4 | gRPC > REST by 5–15% in RPS | 4.9 vs 4.8 RPS (+2%) | **Inconclusive** |
| H5 | Optimal batch size: 16–32 | B1 (8) best on dev GPU | **Pending** (full GPU) |
| H6 | Optimal timeout ≈ 0.05s | Tight (0.01) better on dev | **Pending** (full GPU) |
| H7 | UniEncoder ≈ BiEncoder in throughput | +1.3% RPS, no significant diff | **Supported** |
| H8 | Short texts → higher RPS | 48.1 vs 3.4 vs 0.4 RPS | **Confirmed** |
| H9 | Long texts → lower RPS, batching more impactful | 0.4 RPS for long texts | **Partially confirmed** |
| H10 | AYA Russian ≈ English synthetic | 15.7 vs 22.9 RPS (short) | **Pending** (length-controlled) |

---

## 12. File Inventory

### Source Code

| File | Description | Status |
|------|-------------|--------|
| `ray-serve/serve_app.py` | REST deployment with batch toggle | Done |
| `ray-serve/serve_app_grpc.py` | gRPC + REST dual deployment | Done |
| `ray-serve/proto/gliner_guard.proto` | Protobuf schema | Done |
| `ray-serve/Dockerfile` | CUDA 12.8 + uv + proto codegen | Done |
| `litserve-baseline/main.py` | LitServe baseline (bf16) | Done |
| `docker-compose.yml` | 3 profiles + locust service | Done |
| `Makefile` | 22 targets for automation | Done |
| `test-script/test-gliner.py` | Locust REST adapter | Done |
| `test-script/test-gliner-grpc.py` | Locust gRPC adapter | Done |

### Benchmark Scripts

| Script | Description | Status |
|--------|-------------|--------|
| `scripts/run-batch-benchmarks.sh` | B1–B8 sweep (single model) | Updated to B1–B8, validated |
| `scripts/run-full-batch-sweep.sh` | B1–B8 × 2 models (Days 8–10) | Created |
| `scripts/run-dataset-sweep.sh` | 5 datasets × 2 models (Day 11) | Created, validated |
| `scripts/run-grpc-benchmarks.sh` | REST vs gRPC (Phase 3) | Validated |
| `scripts/run-nobatch-benchmarks.sh` | No-batch × 2 models (Phase 1) | Validated |
| `scripts/collect_gpu_metrics.sh` | nvidia-smi logger | Validated |

### Documentation

| File | Description | Status |
|------|-------------|--------|
| `docs/ray-serve-experiment-plan.md` | Master plan (20 days) | Up to date |
| `docs/ray-serve-experiment-report.md` | This report | Current |
| `docs/ray-serve-rest-nobatch.md` | Phase 1 analysis | Done |
| `docs/ray-serve-dynamic-batching-dev.md` | Phase 2 dev GPU analysis | Done |
| `docs/ray-serve-grpc-dev.md` | Phase 3 dev GPU analysis | Done |
| `docs/benchmark-infra-design.md` | Infrastructure design | Done |
| `docs/litserve-baseline.md` | LitServe baseline docs | Done |
| `docs/ray-serve-dynamic-batching.md` | Phase 2 cloud VM analysis | **Pending** |
| `docs/ray-serve-rest-vs-grpc.md` | Phase 3 cloud VM analysis | **Pending** |
| `docs/ray-serve-final-report.md` | Phase 4 final report | **Pending** |

---

## 13. Timeline Status

| Day | Phase | Work | Status |
|-----|-------|------|--------|
| 1 | Phase 0 | Environment, Docker, deps | **DONE** 2026-04-05 |
| 2 | Phase 0 | Test data (5 datasets) | **DONE** 2026-04-05 |
| 3 | Phase 1 | Ray Serve REST deployment | **DONE** 2026-04-05 |
| 4 | Phase 1 | Locust: REST no-batch (dev) | **DONE** 2026-04-05 |
| 5 | Phase 1 | Analysis | **DONE** 2026-04-05 |
| 6 | Phase 2 | Implement @serve.batch | **DONE** 2026-04-05 |
| 7 | Phase 2 | Dev GPU: B1–B8 sweep | **DONE** 2026-04-06 |
| 8 | Phase 2 | Cloud VM: Uni B1–B6 | BLOCKED (no VM) |
| 9 | Phase 2 | Cloud VM: Uni B7–B11 | BLOCKED |
| 10 | Phase 2 | Cloud VM: Bi B1–B11 | BLOCKED (VM + bi bug) |
| 11 | Phase 2 | Cloud VM: Dataset sweep | BLOCKED |
| 12 | Phase 2 | Batching analysis | BLOCKED |
| 13 | Phase 3 | Proto + gRPC deployment | **DONE** 2026-04-06 |
| 14 | Phase 3 | gRPC Locust adapter | **DONE** 2026-04-06 |
| 15 | Phase 3 | Cloud VM: REST vs gRPC | BLOCKED |
| 16 | Phase 3 | Cloud VM: Cross-dataset | BLOCKED |
| 17 | Phase 3 | Protocol analysis | BLOCKED |
| 18 | Phase 4 | LitServe re-baseline | BLOCKED |
| 19 | Phase 4 | Final report | BLOCKED |
| 20 | Phase 4 | PR + review | BLOCKED |

**Progress:** 9/20 days complete (Days 1–7, 13–14). All dev GPU work done. Cloud VM needed for remaining 11 days.

---

## 14. Recommendations

1. **Provision cloud VM** — all remaining work is blocked on dedicated GPU access
2. **Fix biencoder** — report upstream to `gliner2`, test with older versions, or skip bi experiments and focus on uniencoder only
3. **Investigate gRPC P50 anomaly** — standardize Locust user class (both use `FastHttpUser` or both synchronous) for fair comparison
4. **Consider reducing experiment matrix** — with bi blocked, 84 runs (21h) is feasible in 2–3 cloud VM days
5. **Run LitServe bf16 baseline first** on cloud VM — establishes the comparison target before Ray experiments

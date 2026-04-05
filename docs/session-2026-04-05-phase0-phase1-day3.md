# Session Report: Phase 0 Completion + Phase 1 Day 3

**Date:** 2026-04-05
**Duration:** ~2 hours
**Scope:** GLiNER Guard Serve — Ray Serve experiment infrastructure bootstrap and smoke test
**VM:** 192.168.1.3 (K3s single-node, RTX 5070 Ti 16GB, 39GB RAM)

---

## Objectives

1. Close remaining Phase 0 items (datasets, Docker images)
2. Deploy Ray Serve on dev GPU and verify both models
3. Establish docker compose workflow on the VM

---

## Results

### Phase 0 Closure

| Item | Status | Notes |
|------|--------|-------|
| `xstest.csv` (450 rows) | Already generated | Was marked as pending in plan, but file existed |
| `aya-rus.csv` (717 rows) | Already generated | 717 rows (not 500 — full Russian subset sample) |
| Docker image: litserve | Built on VM | `docker compose build litserve` — 17.6GB |
| Docker image: ray-serve | Built on VM | `docker compose build ray-serve` — 18.1GB |
| Jenkins CI builds #55, #56 | FAILURE | Triggered but failed on old code (pre-Python fix). Non-blocking — images built locally |

### Phase 1 Day 3

| Check | Result |
|-------|--------|
| Ray Serve starts on dev GPU | Yes (RTX 5070 Ti via docker compose) |
| Uniencoder (`hivetrace/gliner-guard-uniencoder`) | Working — correct PII + safety output |
| Biencoder (`hivetrace/gliner-guard-biencoder`) | Working — correct PII + safety output |
| Functional correctness vs LitServe | Matched. Ray Serve additionally detected `address: 123 Main St` |

**Smoke test input:**
```
My name is John Smith and I live at 123 Main St. Call me at 555-123-4567
```

**Ray Serve output (both models):**
```json
{
  "entities": {
    "person": ["John Smith"],
    "address": ["123 Main St"],
    "email": [],
    "phone": ["555-123-4567"]
  },
  "safety": "unsafe"
}
```

**LitServe output (for comparison):**
```json
{
  "entities": {
    "person": ["John Smith"],
    "address": [],
    "email": [],
    "phone": ["555-123-4567"]
  },
  "safety": "unsafe"
}
```

---

## Issues Encountered and Fixes

### 1. Python 3.14 incompatibility

**Problem:** `uv` downloads the latest Python (3.14.3) inside the container. PyTorch 2.8.0 only ships `cp313` wheels, Ray 2.54.1 ships `cp312`/`cp313`.

**Error:**
```
error: Distribution `torch==2.8.0` can't be installed because it doesn't have a
wheel for the current platform
hint: You're using CPython 3.14 (`cp314`), but `torch` (v2.8.0) only has wheels
with the following Python ABI tags: `cp313`, `cp313t`
```

**Fix:**
- `pyproject.toml`: `requires-python = ">=3.12, <3.14"` (both litserve + ray-serve)
- `Dockerfile`: `ENV UV_PYTHON=3.13`
- Regenerated `uv.lock` files

**Commit:** `18a724c`

### 2. Locust Dockerfile build context mismatch

**Problem:** `Dockerfile.locust` references `COPY test-script/pyproject.toml` (relative to project root), but `docker-compose.yml` set `context: test-script`.

**Fix:** Changed compose to `context: .` and `dockerfile: Dockerfile.locust`.

**Commit:** `a9ca008`

### 3. Ray Serve CLI `--host`/`--port` not supported

**Problem:** `serve run serve_app:app --host 0.0.0.0 --port 8000` fails with `No such option: --host`.

**Error:**
```
Usage: serve run [OPTIONS] CONFIG_OR_IMPORT_PATH [ARGUMENTS]...
Error: No such option: --host
```

**Fix:** Switched from CLI to Python-based startup:
```python
if __name__ == "__main__":
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    serve.run(app, route_prefix="/")
    signal.pause()
```
Dockerfile CMD changed to `["uv", "run", "python", "serve_app.py"]`.

**Note:** `RAY_SERVE_HTTP_HOST` env var was also tried but is not honored by Ray.

**Commit:** `1b933ce`

### 4. Ray working_dir upload exceeds 512MB limit

**Problem:** Ray Serve uploads the entire `/app` directory as `working_dir`, which includes `.venv` (1.2GB), exceeding the 512MB default limit.

**Error:**
```
RuntimeEnvSetupError: Package size (1187.18MiB) exceeds the maximum size of
512.00MiB. To exclude large files, add them to '.rayignore'.
```

**Fix:** Created `ray-serve/.rayignore`:
```
.venv/
__pycache__/
*.pyc
.cache/
.ruff_cache/
uv.lock
```

**Commit:** `5cfad9d`

### 5. Ray Serve OOM (40+ workers killed)

**Problem:** VM has 39GB RAM but K3s uses 87% (29GB). Ray spawns head node + GCS + dashboard + proxy + controller + replica — each loading the model (~560MB). Workers killed by Ray's memory monitor in a crash loop.

**Error:**
```
40 Workers (tasks / actors) killed due to memory pressure (OOM)
```

**Fix (multi-pronged):**
1. `docker-compose.yml`: `shm_size: "2g"` + `RAY_OBJECT_STORE_MEMORY=500000000`
2. Removed `ray_actor_options={"num_gpus": 1}` from deployment (GPU passed via Docker runtime)
3. Temporarily scaled down K3s workloads to free ~3GB:
   - OpenSearch (1GB), Kafka Connect (1GB), Dify API+Worker (1.1GB), Open-WebUI (608MB)
   - All restored after smoke test

**Commit:** `3f31e10`

### 6. nvidia-container-toolkit not installed

**Problem:** Docker on VM (192.168.1.3) had no NVIDIA container runtime. GPU passthrough via `deploy.resources.reservations.devices` requires it.

**Fix:**
```bash
# Add NVIDIA repo
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Install
apt-get install -y nvidia-container-toolkit

# Configure Docker
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

**Verification:** `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi` returned `NVIDIA GeForce RTX 5070 Ti, 16303 MiB`.

---

## Infrastructure Changes

| Component | Before | After |
|-----------|--------|-------|
| nvidia-container-toolkit on VM | Not installed | Installed, Docker runtime configured |
| `gliner-guard-serve` repo on VM | Not cloned | Cloned at `~/gliner-guard-serve` |
| Docker images on VM | None | litserve:dev (17.6GB), ray-serve:dev (18.1GB) |
| HuggingFace model cache | Empty | `gliner-guard-uniencoder` + `biencoder` cached in Docker volume `hf-cache` |

---

## Commits (chronological)

| SHA | Message |
|-----|---------|
| `d28b4e9` | docs: mark all 5/5 datasets as generated in experiment plan |
| `18a724c` | fix: pin Python <3.14 — PyTorch/Ray lack cp314 wheels |
| `a9ca008` | fix: correct locust build context in docker-compose.yml |
| `2c8ccee` | fix: use RAY_SERVE_HTTP env vars instead of --host/--port CLI flags |
| `5cfad9d` | fix: add .rayignore to exclude .venv from Ray working_dir upload |
| `3f31e10` | fix: resolve Ray Serve OOM — limit object store, remove num_gpus |
| `1b933ce` | fix: use serve.start() with explicit http_options host 0.0.0.0 |
| `816032d` | docs: mark Phase 1 Day 3 complete — both models verified on dev GPU |

---

## Next Steps (Day 4)

1. **Quick bench** (`bench.py`) — sanity check RPS on dev GPU
2. **Locust benchmarks:** Ray REST no-batch, 3 repeats x 2 models
   - `make bench-ray-nobatch-uni RUN=1` (requires K3s workloads scaled down for RAM)
   - Save results to `results/ray-rest-nobatch-{uni,bi}-run{1,2,3}.csv`
3. **GPU metrics:** `scripts/collect_gpu_metrics.sh` in parallel with Locust

### Known Constraints for Dev GPU Benchmarks

- **RAM pressure:** Must scale down ~4GB of K3s workloads before running Ray Serve
- **GPU:** RTX 5070 Ti 16GB, 1/8 time-sliced — numbers NOT for final comparison (final = cloud VM with dedicated GPU)
- **Locust:** Runs on same VM (not separate machine) — slight measurement noise
- **Jenkins:** Needs retriggering with latest code for Harbor push (builds #55/#56 failed on old code)

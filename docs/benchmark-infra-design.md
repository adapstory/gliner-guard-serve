# GLiNER Guard Benchmark Infrastructure — Design Spec

**Date:** 2026-04-05
**Status:** Approved by PO
**Scope:** Temporary research experiment. Infrastructure will be torn down after results collected. Full experiment runs on cloud VM with dedicated GPU.

## PO Decisions

| # | Question | Decision |
|---|----------|----------|
| 1 | Scope | Temporary experiment, then tear down + cloud |
| 2 | Automation | Makefile targets (granular, per-config) |
| 3 | CI | Reuse Jenkins Kaniko pipeline (heavy profile for CUDA) |
| 4 | GPU | 1/8 time-sliced GPU for smoke test; full GPU on cloud VM |
| 5 | Cloud VM | docker compose (profiles for litserve/ray-serve + locust) |

## Architecture

### Docker Compose (cloud VM execution)

```
docker-compose.yml
  litserve (profile: litserve)  ─── GPU, port 8000
  ray-serve (profile: ray-serve) ── GPU, port 8000
  locust (always)               ─── CPU, port 8089, connects to server:8000
  shared: hf-cache volume, .env config
```

- Profiles: `--profile litserve` or `--profile ray-serve` (mutually exclusive)
- Both servers aliased as `server` so Locust config is unchanged
- Model cache: named volume `hf-cache` at `/app/.cache/huggingface`
- Config via `.env`: MODEL_ID, MAX_BATCH_SIZE, BATCH_WAIT_TIMEOUT, MAX_ONGOING_REQUESTS

### Makefile Targets

```
Server management:  up-litserve, up-ray-nobatch, up-ray-B4, down
Benchmarks:         bench-litserve-uni, bench-ray-nobatch-uni, bench-ray-B4-uni, ...
Utilities:          bench-readme, docker-build, docker-push, warmup, generate-data-*
```

Each `bench-*` target: up server → wait ready → warmup → GPU metrics bg → locust 15m → down → save results.
Run number via `RUN=N` variable.
Result naming: `results/{framework}-{protocol}-{config}-{model}-run{N}.csv`

### Jenkins CI

- Reuse existing Kaniko pipeline (heavy profile for CUDA base image)
- Two images: `gliner-guard-litserve`, `gliner-guard-ray-serve`
- Registry: Harbor (harbor.adapstory.com/adapstory/)
- Tag: CalVer YY.MM.DD-{sha9}
- Stages: checkout → kaniko build → trivy scan → harbor push → update GitOps values
- Manual trigger only

### Files to Create/Modify

**New files:**
- `docker-compose.yml` — server profiles + locust
- `.dockerignore` — shared for both images
- Expanded `Makefile` — benchmark targets + docker management
- `Adapstory-GitOps/infra/ci/jenkins/pipelines/gliner-guard-serve-build.jenkinsfile`

**Modified files:**
- `Adapstory-GitOps/infra/ci/jenkins/job-dsl/jobs.groovy` — register new build job

## Out of Scope

- Production monitoring (OTEL, structured logging, Prometheus)
- Unit/integration tests for model code
- Multi-stage Docker builds
- Pre-commit hooks
- gRPC Locust adapter (Phase 3 of experiment, not infra)

# Ray Serve gRPC — Dev GPU Implementation & Smoke Test

> **Phase:** 3 (REST vs gRPC) — Day 13-14 implementation + dev GPU validation
> **Date:** 2026-04-06
> **GPU:** NVIDIA RTX 5070 Ti 16GB, 1/8 time-sliced
> **Model:** `hivetrace/gliner-guard-uniencoder` (147M params)
> **Framework:** Ray Serve 2.46.0, gRPC via `gRPCOptions` + protobuf

---

## Implementation

### Proto definition
- `ray-serve/proto/gliner_guard.proto`: `Predict` RPC, `PredictRequest` (text), `PredictResponse` (map entities, string safety)
- Stubs generated during Docker build via `grpc_tools.protoc`
- Flat imports (`gliner_guard_pb2`) required for Ray actor serialization — package-based imports cause pickle failures

### gRPC deployment (`serve_app_grpc.py`)
- Dual protocol: REST on port 8000 (`__call__`) + gRPC on port 9000 (`Predict`)
- Same `MAX_BATCH_SIZE` env toggle as REST (0=no-batch, >0=batch)
- Docker Compose profile: `ray-serve-grpc`
- `gRPCOptions(port=9000, grpc_servicer_functions=[...])` for gRPC proxy

### Locust adapter (`test-gliner-grpc.py`)
- `grpc.insecure_channel` + `GLiNERGuardServiceStub`
- Same dataset loading as REST test
- `events.request.fire()` for Locust metric tracking

### Key challenges resolved
1. **Ray serialization**: Proto pb2 module can't be pickled. Solution: lazy import inside `_to_response()`, untyped signatures
2. **Proto import path**: `grpc_tools.protoc` generates flat imports (`import gliner_guard_pb2`), not package-relative. Solution: generate at root level, copy to site-packages
3. **REST health check**: gRPC deployment needs `__call__` method for REST /predict health checks, even when primary protocol is gRPC

---

## Smoke Test Results

### gRPC quick benchmark (5 min, 20 users, no-batch)

| Metric | gRPC | REST (Phase 1 avg) |
|--------|-----:|-------------------:|
| RPS | 4.9 | 4.8 |
| P50 (ms) | 200 | 4,124 |
| P95 (ms) | 340 | 5,780 |
| Errors | 0 | 0 |

### Note on latency difference

The gRPC P50 (200ms) vs REST P50 (4124ms) is a **20x difference** that requires investigation:
- RPS is identical (~4.8-4.9) — both are GPU compute-bound
- The REST P50 of 4.1s with 20 users suggests queueing: 20 users / 4.8 RPS = ~4.2s queue wait per request
- The gRPC P50 of 200ms suggests **much less queueing** — possibly because `constant_throughput(5)` in Locust throttles differently for gRPC (synchronous blocking calls) vs REST (async HTTP)
- **This is not a fair comparison** — the REST baseline ran 15 min with 3 repeats, gRPC ran 5 min with 1 run
- Full comparison on cloud VM with identical load profiles needed

---

## Files

| File | Description |
|------|-------------|
| `ray-serve/proto/gliner_guard.proto` | Proto3 service definition |
| `ray-serve/serve_app_grpc.py` | gRPC+REST dual deployment |
| `test-script/test-gliner-grpc.py` | Locust gRPC adapter |
| `scripts/run-grpc-benchmarks.sh` | Automated REST vs gRPC sweep |
| `docker-compose.yml` | Added `ray-serve-grpc` profile |

---

## Next Steps (Cloud VM)

- [ ] Run `scripts/run-grpc-benchmarks.sh` with `REPEATS=3 USERS=100 DURATION=15m`
- [ ] Compare REST vs gRPC with identical Locust config (same `wait_time`, same users)
- [ ] Test gRPC + batching (MAX_BATCH_SIZE=16)
- [ ] Investigate latency difference root cause

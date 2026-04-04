# Experiment Plan: Ray Serve + GLiNER-2

> **Track:** Ray Serve — Native PyTorch, Dynamic Batching, REST vs gRPC
> **Duration:** 20 days
> **Baseline:** LitServe + GLiNER-2 · PyTorch bf16 · 148.2 RPS, P50 570ms, P95 1500ms (A100 80G)
> **Models:** `hivetrace/gliner-guard-uniencoder` (147M) + `hivetrace/gliner-guard-biencoder` (145M)

---

## 1. Goals

| # | Goal | Success Metric |
|---|------|---------------|
| G1 | Deploy GLiNER-2 (both encoders) through Ray Serve with native PyTorch bf16 | Server starts, handles 100 concurrent users, 0 errors |
| G2 | Find optimal dynamic batching config for each encoder | RPS >= baseline (148 RPS) or latency P50 < 570ms |
| G3 | Measure REST vs gRPC delta | Quantified difference in RPS and latency (P50, P95) |
| G4 | Validate on diverse datasets (synthetic, safety, multilingual) | No accuracy regression across datasets |

---

## 2. Models Under Test

| Model | ID | Params | Architecture | Backbone |
|-------|----|--------|-------------|----------|
| UniEncoder | `hivetrace/gliner-guard-uniencoder` | 147M | Single encoder for all inputs | mmBERT-small (DebertaV2) |
| BiEncoder | `hivetrace/gliner-guard-biencoder` | 145M | Separate encoders for query/candidate | mmBERT-small (DebertaV2) |

Both support 6 tasks, ~100 labels:
- Safety (safe/unsafe)
- PII/NER (32 entity types)
- Adversarial detection (15 labels)
- Harmful content (30 labels)
- Intent classification (13 labels)
- Tone of voice (10 labels)

---

## 3. Test Datasets

| Dataset | Source | Rows | Avg chars | Language | Purpose |
|---------|--------|------|-----------|----------|---------|
| `synthetic-medium` | `generate_data.py` (default) | 500 | ~2500 | EN | Main benchmark (existing baseline) |
| `synthetic-short` | `generate_data.py` (min=20, max=80 words) | 500 | ~300 | EN | Short text behavior |
| `synthetic-long` | `generate_data.py` (min=1000, max=2000 words) | 500 | ~8000 | EN | Long text / padding overhead |
| `xstest` | `walledai/XSTest` | 450 | varies | EN | Safety benchmark (safe + unsafe prompts) |
| `aya-rus` | `CohereForAI/aya_dataset` (filter: `language_code=rus`) | 500* | varies | RU | Multilingual robustness |

\* Random sample of 500 from Russian subset.

### Data Preparation Script

Extend `scripts/generate_data.py` or create `scripts/prepare_datasets.py`:

```python
from datasets import load_dataset
import csv, random

# --- Synthetic short/long ---
# Reuse generate_data.py with configurable min_words/max_words

# --- XSTest ---
ds = load_dataset("walledai/XSTest")
with open("test-script/xstest.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["user_msg"])
    for row in ds["test"]:
        writer.writerow([row["prompt"]])

# --- AYA Russian ---
ds = load_dataset("CohereForAI/aya_dataset")
rus = ds["train"].filter(lambda x: x["language_code"] == "rus")
sample = random.sample(range(len(rus)), min(500, len(rus)))
with open("test-script/aya-rus.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["user_msg"])
    for i in sample:
        writer.writerow([rus[i]["inputs"]])
```

---

## 4. Hardware Progression

Experiments run in two phases on different hardware to validate portability.

| Stage | GPU | VRAM | CPU | Role |
|-------|-----|------|-----|------|
| **Dev** (Days 1–7) | Available GPU (T4 / RTX 3090 / etc.) | varies | varies | Functional validation, initial batching sweep |
| **Prod** (Days 8–18) | A100 80G PCIe | 80 GB | 14 vCPU | Final benchmarks (all results reported from this) |
| **Locust client** | — | — | 16 vCPU (separate VM) | Load generator |

---

## 5. Experiment Structure

```
ray-serve/
├── serve_app.py                    # Ray Serve deployment (REST)
├── serve_app_grpc.py               # Ray Serve deployment (gRPC)
├── proto/
│   ├── gliner_guard.proto          # Protobuf service definition
│   ├── gliner_guard_pb2.py         # Generated stubs
│   └── gliner_guard_pb2_grpc.py    # Generated gRPC stubs
├── config/
│   ├── serve_config_rest.yaml      # Ray Serve config (REST)
│   └── serve_config_grpc.yaml      # Ray Serve config (gRPC)
├── Dockerfile                      # Reproducible environment
├── docker-compose.yml              # Server + monitoring
├── pyproject.toml                  # Dependencies
└── README.md                       # Setup & results

test-script/
├── test-gliner.py                  # Locust REST (existing)
├── test-gliner-grpc.py             # Locust gRPC (new)
├── prompts.csv                     # synthetic-medium (existing)
├── responses.csv                   # synthetic-medium (existing)
├── prompts-short.csv               # synthetic-short (new)
├── prompts-long.csv                # synthetic-long (new)
├── xstest.csv                      # XSTest safety (new)
└── aya-rus.csv                     # AYA Russian (new)

scripts/
├── generate_data.py                # Existing (parameterize min/max words)
├── prepare_datasets.py             # HuggingFace dataset download + convert
├── collect_gpu_metrics.sh          # nvidia-smi logging during benchmarks
└── gen-benchmark-table.py          # Existing
```

---

## 6. Phases & Timeline (20 Days)

### Phase 0: Environment & Data (Days 1–2)

#### Day 1 — Environment Setup

- [ ] Create `ray-serve/` directory, `pyproject.toml`
- [ ] Dependencies:

```toml
[project]
name = "gliner-guard-ray-serve"
requires-python = ">=3.12"
dependencies = [
    "ray[serve]>=2.46",
    "gliner2>=1.2.4",
    "torch==2.8.0",
    "transformers>=5.0.0",
    "grpcio>=1.71",
    "grpcio-tools>=1.71",
    "protobuf>=5.29",
    "pydantic>=2.0.0",
]
```

- [ ] Create `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y python3.12 python3-pip curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .
EXPOSE 8000 9000
CMD ["uv", "run", "serve", "run", "serve_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] Verify Ray Serve starts on dev GPU: `serve run serve_app:app`
- [ ] Smoke test: single `/predict` returns correct result for both models

#### Day 2 — Test Data Preparation

- [ ] Parameterize `generate_data.py` for `--min-words` / `--max-words`
- [ ] Generate `prompts-short.csv` (20–80 words), `prompts-long.csv` (1000–2000 words)
- [ ] Create `scripts/prepare_datasets.py` — download XSTest + AYA Russian
- [ ] Generate `xstest.csv` (450 rows), `aya-rus.csv` (500 rows)
- [ ] Update `test-gliner.py` to accept `--dataset` flag (prompts / prompts-short / prompts-long / xstest / aya-rus)
- [ ] Create `scripts/collect_gpu_metrics.sh`:

```bash
#!/bin/bash
# Usage: ./collect_gpu_metrics.sh <output_file> <duration_seconds>
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv,nounits -l 1 > "$1" &
PID=$!
sleep "$2"
kill $PID
```

**Deliverable:** All test data ready, Docker builds, dev GPU validated.

---

### Phase 1: Ray Serve REST Baseline — Both Encoders (Days 3–5)

#### Day 3 — Implement Ray Serve Deployment

```python
import torch
from ray import serve

from gliner2 import GLiNER2

MODEL_ID = os.environ.get("MODEL_ID", "hivetrace/gliner-guard-uniencoder")

@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=200,
    ray_actor_options={"num_gpus": 1},
)
class GLiNERGuardDeployment:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = GLiNER2.from_pretrained(MODEL_ID)
        self.model.to(self.device).to(torch.bfloat16).eval()
        self.schema = (
            self.model.create_schema()
            .entities(
                entity_types=["person", "address", "email", "phone"],
                threshold=0.4,
            )
            .classification(task="safety", labels=["safe", "unsafe"])
        )

    async def __call__(self, request):
        body = await request.json()
        text = body["text"]
        result = self.model.extract(text, self.schema)
        return result

app = GLiNERGuardDeployment.bind()
```

- [ ] Deploy on dev GPU, verify both models load (swap `MODEL_ID`)
- [ ] Verify functional correctness: same output as LitServe for 5 test cases
- [ ] Quick bench (`bench.py`) — sanity check on dev GPU

#### Day 4 — Locust: Ray REST No-Batch (dev GPU, synthetic-medium)

For each model (uniencoder + biencoder):
- [ ] Run Locust × 3 repeats: `locust -f test-gliner.py -u 100 -r 1 --run-time 15m`
- [ ] Collect GPU metrics in parallel: `./collect_gpu_metrics.sh gpu_metrics.csv 900`
- [ ] Save: `results/ray-rest-nobatch-uni-{run1,run2,run3}.csv`
- [ ] Save: `results/ray-rest-nobatch-bi-{run1,run2,run3}.csv`

#### Day 5 — Analysis

- [ ] Calculate mean ± std for RPS, P50, P95 across 3 runs per model
- [ ] Compare uni vs bi on same hardware
- [ ] Compare vs LitServe baseline (noting different GPU if dev != A100)
- [ ] Identify bottlenecks: GPU util%, VRAM, serialization
- [ ] Document: `docs/ray-serve-rest-nobatch.md`

**Deliverable:** Working deployment for both models + dev GPU benchmarks.

---

### Phase 2: Dynamic Batching Sweep (Days 6–12)

**Objective:** Full parameter sweep for both encoders. On dev GPU first (Days 6–7), then final runs on A100 (Days 8–11).

#### Day 6 — Implement `@serve.batch`

```python
@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=200,
    ray_actor_options={"num_gpus": 1},
)
class GLiNERGuardBatched:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = GLiNER2.from_pretrained(MODEL_ID)
        self.model.to(self.device).to(torch.bfloat16).eval()
        self.schema = (
            self.model.create_schema()
            .entities(entity_types=["person", "address", "email", "phone"], threshold=0.4)
            .classification(task="safety", labels=["safe", "unsafe"])
        )

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.05)
    async def handle_batch(self, texts: list[str]) -> list[dict]:
        results = self.model.batch_extract(
            texts=texts,
            schemas=self.schema,
            batch_size=len(texts),
        )
        return results

    async def __call__(self, request):
        body = await request.json()
        return await self.handle_batch(body["text"])
```

- [ ] Verify batching: send 10 concurrent requests, confirm `batch_extract` called with batch
- [ ] Add env-based config: `MAX_BATCH_SIZE`, `BATCH_WAIT_TIMEOUT`, `MAX_ONGOING_REQUESTS`
- [ ] Smoke test on dev GPU

#### Day 7 — Dev GPU: Quick Sweep (validate matrix works)

Run 1 repeat per config (not 3) — just to validate setup works and catch OOMs:

| ID | `max_batch_size` | `batch_wait_timeout_s` |
|----|-----------------|----------------------|
| B1 | 8 | 0.01 |
| B2 | 16 | 0.05 |
| B3 | 32 | 0.05 |
| B4 | 64 | 0.10 |

- [ ] Run B1–B4 for uniencoder on dev GPU (4 × 15 min = 1h)
- [ ] Verify no OOMs, results are parseable
- [ ] Adjust batch sizes if dev GPU has less VRAM

#### Days 8–10 — A100: Full Batch Sweep (both encoders)

**Experiment matrix** — each run on A100, 3 repeats, synthetic-medium dataset:

| ID | `max_batch_size` | `batch_wait_timeout_s` | `max_ongoing_requests` | Notes |
|----|-----------------|----------------------|----------------------|-------|
| B1 | 8 | 0.01 | 200 | Small batch, tight timeout |
| B2 | 8 | 0.05 | 200 | Small batch, relaxed |
| B3 | 16 | 0.01 | 200 | Medium batch, tight |
| B4 | 16 | 0.05 | 200 | Medium batch, relaxed |
| B5 | 32 | 0.05 | 200 | Large batch, relaxed |
| B6 | 32 | 0.10 | 200 | Large batch, generous timeout |
| B7 | 64 | 0.05 | 200 | Match LitServe config |
| B8 | 64 | 0.10 | 200 | Match LitServe, generous timeout |
| B9 | 64 | 0.05 | token-aware | `batch_size_fn` by total tokens |
| B10 | best | best | 100 | Test `max_ongoing_requests` impact |
| B11 | best | best | 300 | Test `max_ongoing_requests` impact |

**Per experiment protocol:**
```
1. Set env vars: MODEL_ID, MAX_BATCH_SIZE, BATCH_WAIT_TIMEOUT
2. Restart Ray Serve (clean state)
3. Warm-up: 50 requests via bench.py
4. Start GPU metrics: ./collect_gpu_metrics.sh results/gpu-{ID}.csv 900
5. Run Locust: -u 100 -r 1 --run-time 15m --csv results/ray-batch-{model}-{ID}-run{N}
6. Repeat steps 1-5 for run 2 and run 3
```

**Total runs:** 11 configs × 2 models × 3 repeats = **66 Locust runs** (66 × 15 min = ~16.5h)

**Schedule:**
- Day 8: Uniencoder B1–B6 (18 runs ≈ 5h)
- Day 9: Uniencoder B7–B11 (15 runs ≈ 4h)
- Day 10: Biencoder B1–B11 (33 runs ≈ 8.5h, can automate overnight)

#### Day 11 — Dataset Sweep (optimal config)

Use **best config from B1–B11** for each model. Run on all 5 datasets:

| Dataset | Uni (3 runs) | Bi (3 runs) |
|---------|-------------|-------------|
| synthetic-medium | done (Phase 2) | done (Phase 2) |
| synthetic-short | run | run |
| synthetic-long | run | run |
| xstest | run | run |
| aya-rus | run | run |

**Total:** 4 datasets × 2 models × 3 repeats = **24 runs** (~6h)

#### Day 12 — Batching Analysis

- [ ] Compile all results: mean ± std for RPS, P50, P95, P99
- [ ] Build tables:
  - Batch config comparison (uni)
  - Batch config comparison (bi)
  - Uni vs Bi on optimal config
  - Dataset impact on throughput
- [ ] Plots (matplotlib/seaborn):
  - RPS vs `max_batch_size` (grouped by timeout)
  - P50/P95 vs `max_batch_size`
  - GPU utilization vs batch_size
  - RPS by dataset (grouped by model)
- [ ] Select **optimal config** per model (best RPS where P95 < 2000ms)
- [ ] Document: `docs/ray-serve-dynamic-batching.md`

**Deliverable:** Optimal batching config per model + full comparison matrix + dataset analysis.

---

### Phase 3: REST vs gRPC (Days 13–17)

#### Day 13 — Proto + gRPC Deployment

```protobuf
// proto/gliner_guard.proto
syntax = "proto3";
package gliner_guard;

service GLiNERGuardService {
    rpc Predict (PredictRequest) returns (PredictResponse);
}

message PredictRequest {
    string text = 1;
}

message Entity {
    string type = 1;
    string text = 2;
    float confidence = 3;
    int32 start = 4;
    int32 end = 5;
}

message Classification {
    string task = 1;
    string label = 2;
    float confidence = 3;
}

message PredictResponse {
    repeated Entity entities = 1;
    repeated Classification classifications = 2;
}
```

- [ ] Generate stubs: `python -m grpc_tools.protoc -I=proto --python_out=proto --grpc_python_out=proto proto/gliner_guard.proto`
- [ ] Implement `serve_app_grpc.py` with `gRPCOptions(port=9000)`
- [ ] Smoke test: gRPC client → both models → correct response
- [ ] Verify REST still works alongside gRPC (dual port: 8000 + 9000)

#### Day 14 — gRPC Locust Adapter

```python
# test-script/test-gliner-grpc.py
import time, grpc, random, csv
from locust import User, task, events
from proto import gliner_guard_pb2, gliner_guard_pb2_grpc

class GrpcUser(User):
    abstract = False

    def on_start(self):
        host = self.environment.host or "localhost:9000"
        self.channel = grpc.insecure_channel(host, options=[
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 30000),
        ])
        self.stub = gliner_guard_pb2_grpc.GLiNERGuardServiceStub(self.channel)
        # Load same dataset as REST tests
        self.prompts = []
        with open("test-script/prompts.csv") as f:
            reader = csv.DictReader(f)
            self.prompts = [row["user_msg"] for row in reader]

    @task
    def predict_prompt(self):
        text = random.choice(self.prompts)
        request = gliner_guard_pb2.PredictRequest(text=text)
        start = time.perf_counter()
        try:
            response = self.stub.Predict(request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.environment.events.request.fire(
                request_type="gRPC",
                name="predict_prompt",
                response_time=elapsed_ms,
                response_length=response.ByteSize(),
                exception=None,
            )
        except grpc.RpcError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.environment.events.request.fire(
                request_type="gRPC",
                name="predict_prompt",
                response_time=elapsed_ms,
                response_length=0,
                exception=e,
            )

    def on_stop(self):
        self.channel.close()
```

- [ ] Implement + test gRPC Locust user
- [ ] Validate: 10 users × 1 min, metrics appear in Locust stats
- [ ] Verify same data source as REST tests (prompts.csv)
- [ ] Cross-check: gRPC response == REST response for same input

#### Days 15–16 — REST vs gRPC Benchmark (A100)

Use **optimal batching config** from Phase 2. Both models. 3 repeats each.

| ID | Model | Protocol | Batching | Dataset |
|----|-------|----------|----------|---------|
| R1 | uni | REST | optimal | synthetic-medium |
| R2 | uni | gRPC | optimal | synthetic-medium |
| R3 | uni | REST | no batch | synthetic-medium |
| R4 | uni | gRPC | no batch | synthetic-medium |
| R5 | bi | REST | optimal | synthetic-medium |
| R6 | bi | gRPC | optimal | synthetic-medium |
| R7 | bi | REST | no batch | synthetic-medium |
| R8 | bi | gRPC | no batch | synthetic-medium |

**Cross-dataset validation** (optimal batching, optimal protocol):
| R9 | uni | best | optimal | synthetic-short |
| R10 | uni | best | optimal | synthetic-long |
| R11 | uni | best | optimal | xstest |
| R12 | uni | best | optimal | aya-rus |

**Total:** 12 configs × 3 repeats = **36 runs** (~9h)

- Day 15: R1–R8 (24 runs ≈ 6h)
- Day 16: R9–R12 (12 runs ≈ 3h) + troubleshooting buffer

#### Day 17 — Protocol Analysis

- [ ] Build master comparison table:

| Metric | LitServe baseline | Ray REST no-batch | Ray REST batched | Ray gRPC no-batch | Ray gRPC batched |
|--------|------------------|-------------------|------------------|-------------------|------------------|
| | | uni / bi | uni / bi | uni / bi | uni / bi |
| RPS (mean±std) | 148.2 | ? | ? | ? | ? |
| P50 ms | 570 | ? | ? | ? | ? |
| P95 ms | 1500 | ? | ? | ? | ? |
| P99 ms | 1700 | ? | ? | ? | ? |
| GPU util % | ? | ? | ? | ? | ? |
| VRAM peak GB | ? | ? | ? | ? | ? |
| Errors | 0 | ? | ? | ? | ? |

- [ ] Calculate deltas: `(ray - baseline) / baseline × 100%`
- [ ] Analyze:
  - JSON vs protobuf serialization overhead
  - HTTP/1.1 vs HTTP/2 connection overhead
  - Batching efficiency difference across protocols
  - Impact of text length on protocol overhead
- [ ] Document: `docs/ray-serve-rest-vs-grpc.md`

---

### Phase 4: Final Report & PR (Days 18–20)

#### Day 18 — Consolidation

- [ ] Re-run LitServe baseline on A100 (3 repeats) for fair comparison — both models
  - LitServe no-batch (uni + bi)
  - LitServe batched (uni + bi)
- [ ] Ensure all comparisons are A100-to-A100

#### Day 19 — Final Report

- [ ] Master results document: `docs/ray-serve-final-report.md`
  - Executive summary (1 page)
  - Methodology
  - Results tables + plots
  - Statistical significance (mean ± std, mark if overlapping CIs)
  - Recommendations
  - Appendix: raw data links
- [ ] Update README benchmark table: `make bench-readme`
- [ ] Verify Docker reproducibility: fresh `docker build` + benchmark matches

#### Day 20 — PR & Review

- [ ] Create PR with all code + docs + results
- [ ] Peer review checklist:
  - [ ] All experiments reproducible (`docker build` + `uv run`)
  - [ ] Locust CSVs + HTMLs in `results/`
  - [ ] GPU metrics CSVs in `results/`
  - [ ] Each experiment has a doc in `docs/`
  - [ ] README benchmark table updated
  - [ ] Code quality: type hints, docstrings, clean imports
  - [ ] Statistical: 3 repeats, mean ± std reported
  - [ ] Both models tested

---

## 7. Experiment Protocol (every single run)

```
1. Clean state      → restart Ray Serve, torch.cuda.empty_cache()
2. Verify config    → echo $MODEL_ID $MAX_BATCH_SIZE $BATCH_WAIT_TIMEOUT $MAX_ONGOING_REQUESTS
3. Warm-up          → 50 requests via bench.py (discard results)
4. Start GPU logger → ./collect_gpu_metrics.sh results/gpu-{EXP_ID}-run{N}.csv 900
5. Run Locust       → -u 100 -r 1 --run-time 15m --csv results/{EXP_ID}-run{N}
6. Stop GPU logger  → (auto-stops after 900s)
7. Save artifacts   → CSV + HTML + gpu_metrics → results/
8. Log environment  → GPU model, VRAM total, driver version, Ray version, Python version,
                       torch version, gliner2 version → results/{EXP_ID}-env.txt
9. Sanity check     → grep "Aggregated" in CSV, verify 0 failures
```

**Naming convention:** `results/{framework}-{protocol}-{batch_config}-{model}-{dataset}-run{N}.csv`
Example: `results/ray-rest-B4-uni-synthetic-medium-run2.csv`

---

## 8. Variables & Controls

### Independent Variables

| Variable | Values |
|----------|--------|
| Serving framework | LitServe, Ray Serve |
| Model | uniencoder (147M), biencoder (145M) |
| Protocol | REST (JSON), gRPC (protobuf) |
| `max_batch_size` | 0 (no batch), 8, 16, 32, 64 |
| `batch_wait_timeout_s` | 0.01, 0.05, 0.10 |
| `max_ongoing_requests` | 100, 200, 300 |
| `batch_size_fn` | None, token-aware |
| Dataset | synthetic-medium, synthetic-short, synthetic-long, xstest, aya-rus |

### Dependent Variables (measured)

| Metric | Source |
|--------|--------|
| RPS | Locust CSV (Aggregated row) |
| P50, P95, P99 latency (ms) | Locust CSV |
| Error count / rate | Locust CSV |
| GPU utilization % | nvidia-smi log |
| VRAM used (GB) | nvidia-smi log |
| GPU power draw (W) | nvidia-smi log |

### Controls (fixed across all A100 experiments)

| Control | Value |
|---------|-------|
| Precision | bf16 (bfloat16) |
| Schema | 4 PII entity types (threshold=0.4) + safety classification |
| Hardware | A100 80G PCIe, 14 vCPU (server) |
| Locust client | Separate VM, 16 vCPU |
| Load pattern | 100 users, spawn rate 1/s |
| Test duration | 15 min per run |
| Repeats | 3 per configuration |
| `num_replicas` | 1 (single GPU) |

---

## 9. Statistical Approach

- **3 repeats** per experiment configuration
- Report **mean ± standard deviation** for all metrics
- If std > 10% of mean → flag as unstable, investigate
- **No cherry-picking**: report all 3 runs, not "best of 3"
- Comparison significance: if confidence intervals (mean ± 1 std) overlap → "no significant difference"
- GPU metrics: report mean utilization over steady-state period (exclude first 2 min ramp-up)

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Ray Serve + GLiNER2 incompatibility | Blocks Phase 1 | Day 3: smoke test on dev GPU. Fallback: wrap model in plain class |
| Dev GPU OOM on large batches | Limits dev validation | Day 7: start with small batches. Skip B7–B9 on dev, validate on A100 only |
| A100 not available on schedule | Delays Phase 2–3 | Days 1–7 use dev GPU. A100 work can shift to Days 10–20 |
| gRPC Locust adapter unreliable | Blocks Phase 3 | Fallback: use `ghz` (dedicated gRPC bench tool) + convert results to CSV |
| BiEncoder loads differently than UniEncoder | Extra debugging | Day 3: test both models on day 1. Same code, different `MODEL_ID` |
| `batch_size_fn` not in current Ray version | Limits B9 | Check Ray changelog. Fallback: skip B9, document as "not available" |
| 66 + 36 + 24 = 126 Locust runs too slow | Timeline slip | Automate with shell script. Overnight runs. Days 18–20 are buffer |
| Locust client bottleneck at 100 users | Misleading RPS cap | Monitor Locust CPU. If >80%, increase client VM vCPU or reduce users |

---

## 11. Expected Outcomes

| Hypothesis | Basis |
|-----------|-------|
| Ray REST (no batch) < LitServe baseline | Ray actor/routing overhead vs LitServe's direct FastAPI |
| Ray REST (batched, optimal) ≈ LitServe baseline ± 10% | Dynamic batching compensates for Ray overhead |
| gRPC < REST by 10–20% in latency | Protobuf serialization faster than JSON; HTTP/2 multiplexing |
| gRPC > REST by 5–15% in RPS | Lower per-request overhead |
| Optimal batch size: 16–32 | ~150M params; sweet spot between GPU utilization and queue wait |
| `batch_wait_timeout_s` ≈ 0.05s optimal | Too low → small batches; too high → latency penalty |
| UniEncoder ≈ BiEncoder in throughput | Similar param count (147M vs 145M) |
| Short texts → higher RPS | Less compute per request, padding savings |
| Long texts → lower RPS, batching more impactful | GPU-bound; batching amortizes fixed overhead better |
| AYA Russian ≈ English synthetic | Same tokenizer, similar token distribution |

---

## 12. Calendar View

| Day | Phase | Work | Key Output |
|-----|-------|------|-----------|
| 1 | Phase 0 | Environment, Docker, deps | Working container on dev GPU |
| 2 | Phase 0 | Test data preparation (5 datasets) | All CSVs ready |
| 3 | Phase 1 | Ray Serve REST deployment (both models) | `serve_app.py` working |
| 4 | Phase 1 | Locust: REST no-batch (dev GPU, 3×2 runs) | Dev benchmarks |
| 5 | Phase 1 | Analysis, troubleshooting | `docs/ray-serve-rest-nobatch.md` |
| 6 | Phase 2 | Implement `@serve.batch` + env config | Batched deployment working |
| 7 | Phase 2 | Dev GPU: quick sweep B1–B4 (validate) | No OOMs, setup confirmed |
| 8 | Phase 2 | **A100**: Uniencoder B1–B6 (18 runs) | Raw results |
| 9 | Phase 2 | **A100**: Uniencoder B7–B11 (15 runs) | Raw results |
| 10 | Phase 2 | **A100**: Biencoder B1–B11 (33 runs) | Raw results |
| 11 | Phase 2 | **A100**: Dataset sweep (24 runs) | Raw results |
| 12 | Phase 2 | Batching analysis + plots | `docs/ray-serve-dynamic-batching.md`, **checkpoint** |
| 13 | Phase 3 | Proto + gRPC deployment | `serve_app_grpc.py` working |
| 14 | Phase 3 | gRPC Locust adapter | `test-gliner-grpc.py` validated |
| 15 | Phase 3 | **A100**: REST vs gRPC R1–R8 (24 runs) | Raw results |
| 16 | Phase 3 | **A100**: Cross-dataset R9–R12 (12 runs) | Raw results |
| 17 | Phase 3 | Protocol analysis | `docs/ray-serve-rest-vs-grpc.md`, **checkpoint** |
| 18 | Phase 4 | Re-run LitServe baselines on A100 (fair comp) | Apples-to-apples numbers |
| 19 | Phase 4 | Final report | `docs/ray-serve-final-report.md` |
| 20 | Phase 4 | PR + peer review | PR ready |

---

## 13. Total Experiment Count

| Phase | Configs | Models | Repeats | Runs |
|-------|---------|--------|---------|------|
| Phase 1: REST no-batch | 1 | 2 | 3 | 6 |
| Phase 2: Batch sweep | 11 | 2 | 3 | 66 |
| Phase 2: Dataset sweep | 4 | 2 | 3 | 24 |
| Phase 3: REST vs gRPC | 8 | — | 3 | 24 |
| Phase 3: Cross-dataset | 4 | — | 3 | 12 |
| Phase 4: LitServe re-baseline | 4 | 2 | 3 | 24 |
| **Total** | | | | **156 runs** |

**Total Locust time:** 156 × 15 min = **39 hours** (spread across 8 benchmark days)

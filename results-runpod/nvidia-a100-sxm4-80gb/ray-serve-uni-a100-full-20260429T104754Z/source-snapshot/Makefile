# GLiNER Guard Benchmark — Makefile
# Usage: make bench-litserve-uni RUN=1
#        make bench-ray-B4-uni RUN=1
#        make bench-all-litserve

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ── Config ──────────────────────────────────────────────────────────────────
REGISTRY    ?= harbor.adapstory.com/adapstory
TAG         ?= dev
RUN         ?= 1
MODEL       ?= uni
DURATION    ?= 15m
USERS       ?= 100
SPAWN_RATE  ?= 1
DATASET     ?= prompts
WARMUP_REQS ?= 50

# Model IDs
MODEL_UNI := hivetrace/gliner-guard-uniencoder
MODEL_BI  := hivetrace/gliner-guard-biencoder
MODEL_ID   = $(if $(filter bi,$(MODEL)),$(MODEL_BI),$(MODEL_UNI))

# ── Docker ──────────────────────────────────────────────────────────────────
.PHONY: docker-build docker-push

docker-build: ## Build both images (--platform linux/amd64)
	docker compose build litserve
	docker compose build ray-serve

docker-push: ## Push images to Harbor
	docker compose push litserve
	docker compose push ray-serve

# ── Server Management ───────────────────────────────────────────────────────
.PHONY: up-litserve up-ray-nobatch up-ray-batch down wait-ready warmup

up-litserve: ## Start LitServe baseline
	MODEL_ID=$(MODEL_ID) docker compose --profile litserve up -d
	@$(MAKE) wait-ready

up-ray-nobatch: ## Start Ray Serve without batching
	MODEL_ID=$(MODEL_ID) MAX_BATCH_SIZE=0 docker compose --profile ray-serve up -d
	@$(MAKE) wait-ready

up-ray-batch: ## Start Ray Serve with batching (use MAX_BATCH_SIZE, BATCH_WAIT_TIMEOUT)
	MODEL_ID=$(MODEL_ID) docker compose --profile ray-serve up -d
	@$(MAKE) wait-ready

down: ## Stop all services
	docker compose --profile litserve --profile ray-serve down

wait-ready: ## Wait for server to be ready
	@echo "Waiting for server on :8000..."
	@for i in $$(seq 1 120); do \
		curl -sf -o /dev/null http://localhost:8000/predict \
			-H "Content-Type: application/json" \
			-d '{"text":"healthcheck"}' && { echo "Ready!"; exit 0; } || sleep 2; \
	done; echo "TIMEOUT"; exit 1

warmup: ## Send warmup requests
	@echo "Warmup: $(WARMUP_REQS) requests..."
	@cd litserve-baseline && uv run python bench.py 2>/dev/null || true

# ── Benchmark Primitives ────────────────────────────────────────────────────
# Internal: run a single Locust benchmark
# Args: FRAMEWORK, CONFIG, PROTOCOL (set by caller)
RESULT_PREFIX = $(FRAMEWORK)-$(PROTOCOL)-$(CONFIG)-$(MODEL)-$(DATASET)-run$(RUN)

define run-bench
	@echo "=== Benchmark: $(RESULT_PREFIX) ==="
	@mkdir -p results
	@$(MAKE) warmup
	@bash scripts/collect_gpu_metrics.sh results/gpu-$(RESULT_PREFIX).csv $$(echo $(DURATION) | sed 's/m//' | awk '{print $$1*60}') &
	cd test-script && DATASET=$(DATASET) GLINER_HOST=http://localhost:8000 \
		LOCUST_USER_THROUGHPUT=$${LOCUST_USER_THROUGHPUT:-5} \
		uv run locust -f test-gliner.py \
		--headless -u $(USERS) -r $(SPAWN_RATE) --run-time $(DURATION) \
		--csv-full-history \
		--csv=../results/$(RESULT_PREFIX) \
		--html=../results/$(RESULT_PREFIX).html
	@echo "=== Done: results/$(RESULT_PREFIX) ==="
endef

# ── LitServe Benchmarks ────────────────────────────────────────────────────
.PHONY: bench-litserve-uni bench-litserve-bi

bench-litserve-uni: FRAMEWORK=litserve
bench-litserve-uni: PROTOCOL=rest
bench-litserve-uni: CONFIG=baseline
bench-litserve-uni: MODEL=uni
bench-litserve-uni: ## Benchmark LitServe uniencoder
	@$(MAKE) up-litserve MODEL=uni
	$(run-bench)
	@$(MAKE) down

bench-litserve-bi: FRAMEWORK=litserve
bench-litserve-bi: PROTOCOL=rest
bench-litserve-bi: CONFIG=baseline
bench-litserve-bi: MODEL=bi
bench-litserve-bi: ## Benchmark LitServe biencoder
	@$(MAKE) up-litserve MODEL=bi
	$(run-bench)
	@$(MAKE) down

# ── Ray Serve: No Batch ────────────────────────────────────────────────────
.PHONY: bench-ray-nobatch-uni bench-ray-nobatch-bi

bench-ray-nobatch-uni: FRAMEWORK=ray
bench-ray-nobatch-uni: PROTOCOL=rest
bench-ray-nobatch-uni: CONFIG=nobatch
bench-ray-nobatch-uni: MODEL=uni
bench-ray-nobatch-uni: ## Ray Serve no-batch uniencoder
	@$(MAKE) up-ray-nobatch MODEL=uni
	$(run-bench)
	@$(MAKE) down

bench-ray-nobatch-bi: FRAMEWORK=ray
bench-ray-nobatch-bi: PROTOCOL=rest
bench-ray-nobatch-bi: CONFIG=nobatch
bench-ray-nobatch-bi: MODEL=bi
bench-ray-nobatch-bi: ## Ray Serve no-batch biencoder
	@$(MAKE) up-ray-nobatch MODEL=bi
	$(run-bench)
	@$(MAKE) down

# ── Ray Serve: Batch Configs ───────────────────────────────────────────────
# Usage: make bench-ray-B4-uni RUN=1
#        make bench-ray-B7-bi RUN=2 DATASET=prompts-short
.PHONY: bench-ray-B1-uni bench-ray-B2-uni bench-ray-B3-uni bench-ray-B4-uni \
        bench-ray-B5-uni bench-ray-B6-uni bench-ray-B7-uni bench-ray-B8-uni

define ray-batch-target
bench-ray-$(1)-uni: FRAMEWORK=ray
bench-ray-$(1)-uni: PROTOCOL=rest
bench-ray-$(1)-uni: CONFIG=$(1)
bench-ray-$(1)-uni: MODEL=uni
bench-ray-$(1)-uni:
	MAX_BATCH_SIZE=$(2) BATCH_WAIT_TIMEOUT=$(3) $$(MAKE) up-ray-batch MODEL=uni
	$$(run-bench)
	@$$(MAKE) down

bench-ray-$(1)-bi: FRAMEWORK=ray
bench-ray-$(1)-bi: PROTOCOL=rest
bench-ray-$(1)-bi: CONFIG=$(1)
bench-ray-$(1)-bi: MODEL=bi
bench-ray-$(1)-bi:
	MAX_BATCH_SIZE=$(2) BATCH_WAIT_TIMEOUT=$(3) $$(MAKE) up-ray-batch MODEL=bi
	$$(run-bench)
	@$$(MAKE) down
endef

#                    ID   batch  timeout
$(eval $(call ray-batch-target,B1,8,0.01))
$(eval $(call ray-batch-target,B2,8,0.05))
$(eval $(call ray-batch-target,B3,16,0.01))
$(eval $(call ray-batch-target,B4,16,0.05))
$(eval $(call ray-batch-target,B5,32,0.05))
$(eval $(call ray-batch-target,B6,32,0.10))
$(eval $(call ray-batch-target,B7,64,0.05))
$(eval $(call ray-batch-target,B8,64,0.10))

# ── Batch Runners ───────────────────────────────────────────────────────────
.PHONY: bench-all-litserve bench-all-ray-nobatch bench-all-ray-batch bench-a100-4worker-rest setup-a100-baremetal bench-a100-4worker-ray-baremetal

bench-all-litserve: ## Run all LitServe benchmarks (3 repeats × 2 models)
	@for run in 1 2 3; do \
		$(MAKE) bench-litserve-uni RUN=$$run; \
		$(MAKE) bench-litserve-bi RUN=$$run; \
	done

bench-all-ray-nobatch: ## Run all Ray no-batch benchmarks (3 repeats × 2 models)
	@for run in 1 2 3; do \
		$(MAKE) bench-ray-nobatch-uni RUN=$$run; \
		$(MAKE) bench-ray-nobatch-bi RUN=$$run; \
	done

bench-all-ray-batch-uni: ## Run all batch configs for uniencoder (3 repeats × 8 configs)
	@for run in 1 2 3; do \
		for cfg in B1 B2 B3 B4 B5 B6 B7 B8; do \
			$(MAKE) bench-ray-$$cfg-uni RUN=$$run; \
		done; \
	done

bench-a100-4worker-rest: ## Reproducible LitServe vs Ray Serve REST A100 4-worker sweep
	ENV_FILE=$${ENV_FILE:-.env.ray-a100-4w.example} \
		bash scripts/run-a100-4worker-rest-benchmarks.sh

setup-a100-baremetal: ## Prepare uv/Python/stubs on a bare-metal A100 host
	bash scripts/setup-a100-baremetal.sh

bench-a100-4worker-ray-baremetal: ## Ray Serve REST/gRPC A100 4-worker sweep without Docker/LitServe
	ENV_FILE=$${ENV_FILE:-.env.ray-a100-4w.example} \
		bash scripts/run-a100-4worker-ray-baremetal-benchmarks.sh

# ── Data Generation ─────────────────────────────────────────────────────────
.PHONY: generate-data generate-data-short generate-data-long generate-data-external

generate-data: ## Generate default synthetic data (128-512 words)
	cd scripts && python generate_data.py

generate-data-short: ## Generate short text data (20-80 words)
	cd scripts && python generate_data.py --min-words 20 --max-words 80 --suffix short

generate-data-long: ## Generate long text data (1000-2000 words)
	cd scripts && python generate_data.py --min-words 1000 --max-words 2000 --suffix long

generate-data-external: ## Download XSTest + AYA Russian from HuggingFace
	cd scripts && uv run python prepare_datasets.py

# ── Utilities ───────────────────────────────────────────────────────────────
.PHONY: bench-readme gpu-metrics help

bench-readme: ## Update README.md with benchmark results from results/*.csv
	@TABLE=$$(uv run python3 scripts/gen-benchmark-table.py) && \
	uv run python3 -c "\
import sys, re; \
readme = open('README.md').read(); \
table = sys.argv[1]; \
updated = re.sub( \
    r'(<!-- BENCH:START -->).*?(<!-- BENCH:END -->)', \
    r'\1\n' + table + r'\n\2', \
    readme, flags=re.DOTALL); \
open('README.md', 'w').write(updated)" "$$TABLE"
	@echo "README.md updated with benchmark table"

gpu-metrics: ## Start GPU metrics collection (DURATION=15m)
	bash scripts/collect_gpu_metrics.sh results/gpu-manual-$$(date +%s).csv $$(echo $(DURATION) | sed 's/m//' | awk '{print $$1*60}')

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-28s\033[0m %s\n", $$1, $$2}'

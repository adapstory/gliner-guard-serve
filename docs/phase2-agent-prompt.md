# Prompt: Phase 2 — Ray Serve Dynamic Batching

## Задача

Ты продолжаешь эксперимент по бенчмаркингу GLiNER Guard через Ray Serve. Phase 0 и Phase 1 завершены. Тебе нужно выполнить **Phase 2: Dynamic Batching** — реализовать `@serve.batch`, прогнать sweep по конфигурациям на dev GPU и задокументировать результаты.

## Что уже сделано (НЕ ТРОГАЙ, просто знай)

1. **Репозиторий:** `https://github.com/adapstory/gliner-guard-serve.git`  
   На VM: `~/gliner-guard-serve`
2. **Docker images:** собраны локально на VM через `docker compose build`, работают
3. **LitServe baseline:** 148.2 RPS, P50=570ms на A100 (fp16, batched) — файл `results/litserve-baseline.csv`
4. **Ray Serve no-batch (Phase 1):** uni=4.8 RPS, bi=4.9 RPS на dev GPU (RTX 5070 Ti 1/8 time-sliced, 20 users). 0 errors. Результаты в `results/ray-rest-nobatch-*`
5. **Текущий `serve_app.py`** — работает БЕЗ батчинга, route `/predict`, `serve.start()` с `0.0.0.0:8000`
6. **Документация Phase 1:** `docs/ray-serve-rest-nobatch.md`, `docs/session-2026-04-05-phase0-phase1.md`
7. **Experiment plan:** `docs/ray-serve-experiment-plan.md` — полное описание всех фаз

## Инфраструктура

### VM (dev GPU)
- **SSH:** `ssh stepan@192.168.1.3` (пароль: `stepan`)
- **GPU:** NVIDIA RTX 5070 Ti 16GB, 1/8 time-sliced через K3s GPU Operator
- **RAM:** 39GB total, ~5GB available (после scale down K3s workloads)
- **Docker:** 28.2.2, nvidia-container-toolkit установлен
- **uv:** 0.11.3 (path: `$HOME/.local/bin/uv`)
- **Repo:** `~/gliner-guard-serve` (git pull перед работой!)

### КРИТИЧНО — RAM на VM ограничена!
Перед запуском Ray Serve ОБЯЗАТЕЛЬНО освободи ~4GB RAM:
```bash
kubectl scale statefulset opensearch-dev -n env-dev --replicas=0
kubectl scale deployment dev-kafka-kafka-connect0 -n env-dev --replicas=0
kubectl scale deployment dify-api dify-worker -n dify --replicas=0
kubectl scale statefulset open-webui -n ollama --replicas=0
```
**После работы ОБЯЗАТЕЛЬНО верни обратно** (replicas=1 для всех).

### КРИТИЧНО — 100 Locust users = OOM
На dev GPU ТОЛЬКО 20 users. 100 users вызвали 83% failure rate из-за OOM. Это ограничение dev стенда, не Ray Serve.

## Файлы, которые тебе нужно изменить

### 1. `ray-serve/serve_app.py` — добавить batching

Текущий код обрабатывает по одному запросу. Нужно добавить `@serve.batch` декоратор. **Важно:**
- `MAX_BATCH_SIZE` и `BATCH_WAIT_TIMEOUT` берутся из env vars (уже определены в `.env.example`)
- Если `MAX_BATCH_SIZE=0` → работать без батчинга (обратная совместимость)
- Используй `self.model.batch_extract()` для батчевого инференса
- Route остаётся `/predict`
- `serve.start(http_options={"host": "0.0.0.0", "port": 8000})` — НЕ МЕНЯТЬ

Пример из плана (адаптируй):
```python
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "0"))
BATCH_WAIT_TIMEOUT = float(os.environ.get("BATCH_WAIT_TIMEOUT", "0.05"))

# Если MAX_BATCH_SIZE > 0 — создай класс с @serve.batch
# Если MAX_BATCH_SIZE = 0 — оставь текущий класс без батчинга
```

### 2. `scripts/run-batch-benchmarks.sh` — скрипт для sweep

По аналогии с `scripts/run-nobatch-benchmarks.sh`, но:
- Принимает конфигурацию (batch_size, timeout) как параметры
- Прогоняет 1 repeat (не 3) на dev GPU для quick validation (Day 7)
- Naming convention: `ray-rest-B{N}-{model}-{dataset}-run{N}`
- Передаёт `MAX_BATCH_SIZE` и `BATCH_WAIT_TIMEOUT` через env vars при `docker compose up`

### 3. `ray-serve/Dockerfile` — пересобрать образ после изменения serve_app.py

Только `COPY . .` layer изменится. Rebuild быстрый (~5 сек).

## Что КОНКРЕТНО нужно сделать

### Day 6 — Реализация `@serve.batch`

1. Измени `ray-serve/serve_app.py`:
   - Читай `MAX_BATCH_SIZE` и `BATCH_WAIT_TIMEOUT` из env
   - Если batch > 0: используй `@serve.batch(max_batch_size=..., batch_wait_timeout_s=...)`
   - Метод `handle_batch(self, texts: list[str]) -> list[dict]` вызывает `self.model.batch_extract()`
   - `__call__` передаёт `body["text"]` в `handle_batch`
2. Закоммить, запуш, pull на VM, rebuild образ
3. Smoke test с `MAX_BATCH_SIZE=16`:
   ```bash
   MAX_BATCH_SIZE=16 BATCH_WAIT_TIMEOUT=0.05 MODEL_ID=hivetrace/gliner-guard-uniencoder \
     docker compose --profile ray-serve up -d ray-serve
   # Отправь 10 параллельных запросов, убедись что batch_extract вызывается с batch
   ```
4. Smoke test с `MAX_BATCH_SIZE=0` (без батчинга) — регрессии нет

### Day 7 — Quick Sweep на dev GPU

Прогони 4 конфигурации для uniencoder, 1 repeat по 15 мин:

| ID | max_batch_size | batch_wait_timeout_s |
|----|---------------|---------------------|
| B1 | 8 | 0.01 |
| B2 | 16 | 0.05 |
| B3 | 32 | 0.05 |
| B4 | 64 | 0.10 |

Для каждого:
```bash
MAX_BATCH_SIZE={size} BATCH_WAIT_TIMEOUT={timeout} MODEL_ID=hivetrace/gliner-guard-uniencoder \
  docker compose --profile ray-serve up -d ray-serve
# wait ready, warmup, locust 20 users 15min, stop
```

Результаты → `results/ray-rest-B{N}-uni-prompts-run1_stats.csv`

### Day 7 — Анализ + документ

Создай `docs/ray-serve-dynamic-batching-dev.md`:
- Таблица B1-B4: RPS, P50, P95, errors
- Сравнение с no-batch baseline (4.8 RPS)
- Какой batch_size лучше на dev GPU
- Были ли OOM или ошибки
- Рекомендации для cloud VM sweep

### Обновление плана

В `docs/ray-serve-experiment-plan.md`:
- Отметь Day 6 и Day 7 как DONE
- Впиши dev GPU результаты в таблицу

## Как запускать на VM (пошагово)

```bash
# 1. SSH на VM
ssh stepan@192.168.1.3

# 2. Освободить RAM
kubectl scale statefulset opensearch-dev -n env-dev --replicas=0
kubectl scale deployment dev-kafka-kafka-connect0 -n env-dev --replicas=0
kubectl scale deployment dify-api dify-worker -n dify --replicas=0
kubectl scale statefulset open-webui -n ollama --replicas=0

# 3. Pull + rebuild
cd ~/gliner-guard-serve
git pull
docker compose build ray-serve

# 4. Запуск с батчингом
MAX_BATCH_SIZE=16 BATCH_WAIT_TIMEOUT=0.05 MODEL_ID=hivetrace/gliner-guard-uniencoder \
  docker compose --profile ray-serve up -d ray-serve

# 5. Ждать ready (~90 сек, первый раз дольше — скачивание модели)
for i in $(seq 1 120); do
  curl -sf -o /dev/null http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text":"healthcheck"}' && echo "Ready!" && break || sleep 2
done

# 6. Warmup
for i in $(seq 1 50); do
  curl -sf -o /dev/null http://localhost:8000/predict \
    -H "Content-Type: application/json" -d '{"text":"warmup"}' &
done; wait

# 7. Locust
cd test-script
export PATH=$HOME/.local/bin:$PATH
DATASET=prompts GLINER_HOST=http://localhost:8000 \
  uv run locust -f test-gliner.py --headless -u 20 -r 1 --run-time 15m \
  --csv=../results/ray-rest-B2-uni-prompts-run1 \
  --html=../results/ray-rest-B2-uni-prompts-run1.html
cd ..

# 8. Stop
docker compose --profile ray-serve down

# 9. Repeat for B1, B3, B4...

# 10. ОБЯЗАТЕЛЬНО восстановить workloads
kubectl scale statefulset opensearch-dev -n env-dev --replicas=1
kubectl scale deployment dev-kafka-kafka-connect0 -n env-dev --replicas=1
kubectl scale deployment dify-api dify-worker -n dify --replicas=1
kubectl scale statefulset open-webui -n ollama --replicas=1
```

## Грабли, о которых знай

1. **Python 3.14** — `pyproject.toml` уже пинит `<3.14`. Если добавляешь новые зависимости — не ломай это.
2. **`.rayignore`** — уже есть, исключает `.venv/`. Если Ray ругается на working_dir > 512MB — добавь туда.
3. **`serve.start()` обязателен** — `serve run` CLI НЕ поддерживает `--host`. Только Python script.
4. **`RAY_memory_monitor_refresh_ms=0`** — уже в compose, отключает OOM killer. Не убирай.
5. **GPU metrics (nvidia-smi)** — показывают 0% на time-sliced GPU. Не пугайся, это артефакт.
6. **Модель ~560MB** — первый запуск скачивает с HuggingFace. Потом кэшируется в Docker volume `hf-cache`.
7. **`num_gpus=1` УБРАН** из deployment — GPU пробрасывается через Docker runtime, Ray его не scheduling'ит.

## Git workflow

- Всё на ветке `main`
- Коммить после каждого логического шага (реализация batching, benchmark script, результаты, документ)
- `git push` обязателен — работа НЕ закончена пока не запушено
- Convention: `feat:`, `fix:`, `docs:` prefix

## Критерии успеха

- [ ] `serve_app.py` поддерживает `MAX_BATCH_SIZE` env var (0 = no batch, >0 = batch)
- [ ] Smoke test: batching работает (10 concurrent → batch_extract вызван с batch)
- [ ] Smoke test: `MAX_BATCH_SIZE=0` работает (обратная совместимость с Phase 1)
- [ ] 4 конфигурации (B1-B4) прогнаны на dev GPU, 0 errors
- [ ] Результаты в `results/ray-rest-B{N}-*_stats.csv`
- [ ] Документ `docs/ray-serve-dynamic-batching-dev.md`
- [ ] `docs/ray-serve-experiment-plan.md` обновлён (Day 6-7 DONE)
- [ ] Всё закоммичено и запушено
- [ ] K3s workloads восстановлены после работы

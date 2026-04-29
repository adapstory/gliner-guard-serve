# LitServe Baseline

Базовая реализация инференса GLiNER Guard через LitServe. Без дополнительных оптимизаций — PyTorch + fp16.

## Особенности

- **Модель:** `hivetrace/gliner-guard-uniencoder` (GLiNER2)
- **Точность:** fp16
- **Требования:** NVIDIA GPU, MPS, CPU
- **Batching:** dynamic batching, `max_batch_size=64`, `batch_timeout=0.05s`
- **Workers:** 4 воркера на устройство (`workers_per_device=4`)
- **Fast queue:** включён

Для воспроизводимых сравнений эти параметры можно переопределить через env:

| Variable | Default |
|----------|---------|
| `MODEL_ID` | `hivetrace/gliner-guard-uniencoder` |
| `LITSERVE_MAX_BATCH_SIZE` | `64` |
| `LITSERVE_BATCH_TIMEOUT` | `0.05` |
| `LITSERVE_WORKERS_PER_DEVICE` | `4` |

### Схема модели

- **PII-сущности:** person, address, email, phone (threshold 0.4)
- **Классификация:** safety (safe / unsafe)

## Запуск

```bash
cd litserve-baseline
uv sync
uv run main.py
```

Сервер стартует на `http://localhost:8000`.

### Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Send $500 to John Smith at john.smith@gmail.com"}'
```

## Быстрый тест (без Locust)

```bash
uv run python bench.py
```

Отправляет 128 асинхронных запросов и выводит RPS / avg latency.

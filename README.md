# Gliner Guard Serve

## Тестовые данные
Cгенерированы скриптом `scripts/generate_data.py`

Создаёт `prompts.csv` (`user_msg`) и `responses.csv` (`assistant_msg`) — по 500 строк синтетического текста с шумом. Длина 128–512 слов, среднее ~320.

### Статистика по словам

| файл | rows | min words | max words | avg words |
|------|------|-----------|-----------|-----------|
| prompts.csv | 500 | 128 | 512 | ~320 |
| responses.csv | 500 | 128 | 512 | ~320 |

### Статистика по символам

| файл | rows | min chars | max chars | avg chars | median | stdev |
|------|------|-----------|-----------|-----------|--------|-------|
| prompts.csv | 500 | 914 | 4 152 | 2 521 | 2 472 | 702 |
| responses.csv | 500 | 949 | 4 139 | 2 534 | 2 507 | 737 |

## Benchmarks

Пайплайн бенчмаркинга, настройка стенда и инструкция по тестированию — [docs/instruction.md](docs/instruction.md).

Таблица обновляется командой: `make bench-readme`
считывая `.csv` файлы из директории `results`

<!-- BENCH:START -->
| benchmark | RPS | P50 (ms) | P95 (ms) |
|-----------|----:|--------:|---------:|
| litserve-baseline (A100 fp16) | 148.2 | 570 | 1500 |
| ray-nobatch-uni-run1 (RTX 5070 Ti) | 4.8 | 4139 | 6014 |
| ray-nobatch-uni-run2 (RTX 5070 Ti) | 4.8 | 4130 | 5868 |
| ray-nobatch-uni-run3 (RTX 5070 Ti) | 4.8 | 4102 | 5457 |
| ray-nobatch-bi-run1 (RTX 5070 Ti) | 4.9 | 4055 | 5413 |
| ray-nobatch-bi-run2 (RTX 5070 Ti) | 4.8 | 4099 | 5834 |
| ray-nobatch-bi-run3 (RTX 5070 Ti) | 4.9 | 4055 | 5542 |
<!-- BENCH:END -->

# Submission System: Приём работ, анализаторы и анализ сходства

Система приёма, валидации и анализа лабораторных работ для онлайн-школы. Реализована в рамках Лабораторных работ №1–3 по траектории **AI и MLOps**. Проект демонстрирует архитектуру, готовую к интеграции ML-моделей и LLM-анализаторов, воспроизводимое сравнение методов поиска сходства кода и ограниченный исполнитель заданий с тайм-аутами и отменой.

## Функциональность

### Этап А: Приём и валидация
-   **Парсинг JSONL:** Потоковая обработка файлов с поддержкой неограниченного объёма данных.
-   **Валидация схемы:** Проверка обязательных полей (`student_id`, `submission_id`, `timestamp`), формата ISO 8601 и структуры JSON.
-   **Обработка конфликтов:** Различение дубликатов (совпадение ID + хеша) и конфликтов (совпадение ID + разное содержимое). Исходные данные не перезаписываются.
-   **Манифестирование:** Автоматическое формирование манифеста с SHA-256 хешами и версией обработки для каждой принятой сдачи.
-   **Отказоустойчивость:** Повреждённые строки логируются как ошибки с указанием номера строки, но не прерывают импорт.

### Этап Б: Контракты и анализаторы
-   **Расширяемая архитектура:** Использование `Protocol` для контракта `Analyzer`. Центральный сервис не содержит ветвления по конкретным классам.
-   **Анализаторы:**
    -   `LengthAnalyzer`: Подсчёт непустых строк кода и текста.
    -   `TemplateAnalyzer`: Поиск точных совпадений с разрешённым шаблоном с расчётом similarity.
    -   `AIDeclarationChecker`: Проверка соответствия AI-декларации объёму работы (для защиты).
-   **Источники данных:** Единый контракт `SubmissionSource` для In-Memory и JSONL-источников. Ресурс файла управляется через контекстный менеджер.
-   **AI/MLOps инфраструктура:**
    -   Интерфейсы `InputPreparer` и `FeedbackFormatter`.
    -   Stub-модель с детерминированным результатом и фиксацией версии схемы признаков.
    -   Конфигурация режимов `ml`/`llm` с расчётом `config_hash`.

## ЛР2: Анализ сходства и оценка качества

Добавлен отдельный слой `submission_system.similarity`, который не меняет
приём и валидацию ЛР1:

-   **Учебный корпус** `data/corpus/`: 40 пар, 60 сдач, 20 семейств вариантов,
    10 пар с разрешённым шаблоном. У каждой пары есть `assignment_id`,
    `family_id`, `derivation_relation` и `template_only`.
-   **Потоковое чтение JSONL** через генераторы: `iter_jsonl`,
    `iter_corpus_pairs`, `iter_candidate_pairs_streaming`.
-   **Token baseline**: `tokenize` отбрасывает комментарии и служебные токены,
    строки шаблона выносятся отдельно; сходство — Jaccard по 3-граммам.
-   **Второй метод**: TF-IDF по токенам кода и cosine. Векторизатор обучается
    только на tuning-сдачах, evaluation только `transform` (без повторного fit).
-   **Признаки пары** для ML: Jaccard, cosine, длины, доля шаблона, расхождение
    методов и др. (`artifacts/lab2/pair_features.jsonl`).
-   **Retrieval для LLM**: сначала TF-IDF находит кандидатов, затем
    детерминированная LLM-заглушка формирует feedback. Один вопрос «плагиат?»
    к LLM не используется.
-   **Воспроизводимый эксперимент**: порог выбирается на tuning, метрики
    считаются на evaluation; семьи вариантов не пересекают границу split.
-   **`review_required`**: решение по порогу, матрица ошибок, precision, recall,
    F1 и доля неопределённых результатов.
-   **Разбор ошибок**: FP/FN с причинами; обязательный случай общего шаблона.
-   **Профилирование**: время и пиковая память на 100/1000/5000 записей,
    3 повтора, median/mean.
-   **ML против порога**: логистическая регрессия на признаках пары,
    bootstrap и grouped-CV для оценки нестабильности.
-   **MLflow**: параметры, метрики и артефакты разбиения пишутся в локальный
    tracking backend (`mlflow.db` + `mlruns/`); при отсутствии пакета
    предусмотрен JSON-fallback.
-   **Версия отчёта**: `report_version`, `config_hash`, `template_fingerprint`,
    `corpus_fingerprint` в `summary.json` и MLflow.
-   **Демонстрация нового шаблона**: `template-demo` показывает изменение
    конфигурации, score и `review_required` без правки корпуса.

### Запуск ЛР2

```bash
# Собрать корпус из детерминированного генератора
python scripts/build_corpus.py

# Сравнить baseline и TF-IDF, сохранить отчёт
python -m submission_system.similarity.cli evaluate --corpus data/corpus

# Разбор ошибок, включая общий шаблон
python -m submission_system.similarity.cli errors --corpus data/corpus

# Демонстрация добавления нового разрешённого шаблона
python -m submission_system.similarity.cli template-demo --corpus data/corpus

# ML-классификатор против threshold baseline + MLflow-run
python -m submission_system.similarity.cli ml --corpus data/corpus --seeds 20

# Посмотреть MLflow-run
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Retrieval кандидатов для выбранной сдачи
python -m submission_system.similarity.cli retrieve \
    --corpus data/corpus --submission sub-F18-a --top-k 5

# Профилирование 100/1000/5000 записей
python scripts/profile_pipeline.py --sizes 100,1000,5000 --repeats 3 --seed 0

# Замер памяти: потоковое чтение vs полный индекс
python scripts/benchmark_memory.py --corpus data/corpus
```

Подробности, результаты и ограничения — в `docs/LAB2.md`.

## ЛР3, этап А: ограниченный исполнитель

Добавлен пакет `submission_system.jobs`, который превращает анализаторы ЛР1–2
в управляемые задачи, не меняя существующий код приёма и сравнения:

-   **`AnalysisJob`** — `job_id`, `submission_id`, `submission_hash`,
    `analyzer_id`, `config_hash`, `status`, `attempt_no`, тайм-аут и
    timestamps.
-   **Контракт статусов:** `queued → running → succeeded | failed |
    timed_out | cancelled`, плюс `rejected` для явного отказа в постановке.
-   **Постановка отделена от выполнения:** `submit` сохраняет сдачу и решает,
    принять ли задачу в очередь; worker-циклы выполняют её независимо.
-   **Ограничения ресурсов:** одновременно работает не больше
    `concurrency_limit` задач, число ожидающих ограничено `queue_limit`.
    Переполнение возвращает `queue_full`, исходная сдача остаётся в
    репозитории.
-   **Тайм-аут и отмена:** тайм-аут переводит задачу в `timed_out`, отмена
    поддерживается до старта и во время выполнения; `CancelledError` не
    превращается в обычную ошибку анализатора.
-   **Остановка без «сирот»:** `shutdown` дожидается уже запущенных вызовов
    пула, поэтому после остановки не остаётся задач, которыми исполнитель
    больше не управляет.
-   **Политика ошибок — collect-errors:** ошибка или тайм-аут одной задачи не
    отменяют остальные; результаты успешных задач сохраняются.
-   **Неблокирующий event loop:** синхронные анализаторы ЛР1 выполняются в
    отдельном пуле потоков или процессов через `SyncAnalyzerAdapter`.
-   **Модельный адаптер:** `GuardedModelAdapter` ограничивает конкурентные
    обращения к модели, фиксирует `batch_size` и различает коды `timeout`,
    `model_unavailable`, `invalid_model_response`.
-   **Управляемые заглушки:** `SlowAdapter` / `FailingAdapter` /
    `HangingAdapter` позволяют воспроизводимо тестировать задержку, ошибку и
    зависание.

### Запуск ЛР3-A

```bash
# Демонстрация очереди, тайм-аута, отмены и кодов ошибок
python scripts/lab3_demo.py

# Сравнение последовательного запуска, потоков и процессов
python scripts/benchmark_executor.py --jobs 8 --workers 4 --iterations 400000 --repeats 3

# Только тесты этапа А
pytest tests/ -q -k "jobs"
```

Результаты: `artifacts/lab3/stage_a_demo.json` и
`artifacts/lab3/executor_benchmark.json`. Подробности — в `docs/LAB3.md`.

## Установка и запуск

Требуется Python >= 3.12.

1.  Создайте и активируйте виртуальное окружение:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # Windows
    source .venv/bin/activate  # Linux/macOS
    ```

2.  Установите пакет в режиме разработки:
    ```bash
    pip install -e .
    ```

3.  Запустите импорт через CLI:
    ```bash
    python -m submission_system.cli data/test_submissions.jsonl
    ```

4.  Запуск тестов:
    ```bash
    pytest tests/ -v
    ```

## Структура проекта

-   `src/submission_system/models.py` — Доменные модели и отчёты.
-   `src/submission_system/contracts.py` — Протоколы и контекст анализа.
-   `src/submission_system/validator.py` — Логика валидации и хеширования.
-   `src/submission_system/repository.py` — In-memory репозиторий.
-   `src/submission_system/importer.py` — Сервис импорта и генерации манифеста.
-   `src/submission_system/sources.py` — Источники данных (JSONL, In-Memory).
-   `src/submission_system/analyzers.py` — Реализации анализаторов.
-   `src/submission_system/analysis_service.py` — Центральный сервис анализа.
-   `src/submission_system/ml_stub.py` — Stub-модель и интерфейсы подготовки входа.
-   `src/submission_system/config.py` — Конфигурация режимов обработки.
-   `src/submission_system/tokenizer.py` - Токенизация кода для анализа сходства.
-   `src/submission_system/similarity/` - Корпус, baseline, TF-IDF, пары, признаки, retrieval, оценка, разбор ошибок и ML.
-   `src/submission_system/jobs/` - Модели задач, ограниченный исполнитель, адаптеры, модельный адаптер и статусы ЛР3-A.
-   `scripts/build_corpus.py` - Детерминированная сборка учебного корпуса.
-   `scripts/benchmark_memory.py` - Замер памяти для потокового чтения и индекса.
-   `scripts/profile_pipeline.py` - Профилирование времени и памяти на 100/1000/5000 записей.
-   `scripts/lab3_demo.py` - Демонстрация очереди, тайм-аута, отмены и кодов ошибок.
-   `scripts/benchmark_executor.py` - Сравнение последовательного запуска, потоков и процессов.
-   `tests/` - Комплексные тесты приёмки, контрактные тесты, тесты ЛР2 и ЛР3-A.
-   `data/` - Тестовые наборы данных и корпус ЛР2.
-   `docs/LAB2.md` - технический отчёт ЛР2; `ОТЧЁТ_ЛР2.md` - итоговый отчёт.
-   `docs/LAB3.md` - технический отчёт ЛР3-A; `ОТЧЁТ_ЛР3.md` - итоговый отчёт.

## Архитектурные решения

-   **Разделение слоёв:** Доменный слой независим от ввода-вывода, HTTP и stdin.
-   **Неизменяемость:** Входные данные не модифицируются; все преобразования создают новые объекты.
-   **Безопасность ресурсов:** Файловые дескрипторы освобождаются гарантированно через `with open(...)`.
-   **Подготовка к ML:** Сырые данные отделены от метаданных; `ai_declaration` не используется как обучающая метка на этапе валидации.
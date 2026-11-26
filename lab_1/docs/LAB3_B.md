# ЛР3, этап Б: постоянное хранение и FastAPI-интерфейс

Версия отчёта: **3.1.0-stage-b**. Этап Б добавляет SQLite-хранилище,
идемпотентность запуска анализа и HTTP-интерфейс FastAPI. Он использует
исполнитель и адаптеры этапа А без изменения их контракта.

## 1. Что добавлено

| Компонент | Файл | Назначение |
|---|---|---|
| SQLite-хранилище | `src/submission_system/persistence/sqlite_store.py` | Сдачи, задачи, отчёты, idempotency keys |
| API-схемы | `src/submission_system/api/schemas.py` | Pydantic-модели запросов/ответов |
| Прикладной слой | `src/submission_system/api/service.py` | Бизнес-логика, идемпотентность, ошибки |
| FastAPI | `src/submission_system/api/app.py` | Маршруты, API-key auth, lifespan |
| Настройки | `src/submission_system/api/settings.py` | DB, ключи, модель, batch, MLflow |
| Тесты | `tests/test_persistence.py`, `tests/test_api.py` | 21 интеграционный тест этапа Б |

## 2. Схема хранения

SQLite-база по умолчанию: `artifacts/lab3/analysis.db`; PostgreSQL не требуется.
Таблицы:

* `submissions` — неизменяемая исходная сдача (`submission_id` — PK,
  `code`, `text`, `raw_content_hash`);
* `jobs` — `AnalysisJob`: статус, `attempt_no`, timestamps, `error_code`;
* `analysis_reports` — `AnalysisReport`: score, evidence JSON, версия модели,
  версия индекса, версия промпта, `config_hash`, `mlflow_run_id`;
* `idempotency_keys` — уникальный ключ запуска, ссылка на job и payload-поля
  `submission_hash`, `analyzer_id`, `config_hash`.

Индексы: по сдаче, статусу, попыткам и idempotency-ключу.

### 2.1. Неизменяемость сдачи

`SqliteSubmissionRepository.save` вставляет строку только один раз. Повтор с
тем же `submission_id` и другим содержимым не перезаписывает оригинал:
сервис возвращает `409 submission_conflict`. Оригинальный код и хеш остаются
прежними.

### 2.2. Атомарность `succeeded` и отчёта

`SqliteJobStore.save` в одной транзакции:

1. обновляет строку `jobs`;
2. если статус `succeeded`, вставляет/обновляет строку
   `analysis_reports`.

Если `status == succeeded`, а `result is None`, сохранение отклоняется
`PersistenceError`; такой статус не попадает в БД. Дополнительно чтение
`get` не выдаёт `succeeded` без отчёта как успех, а при старте
`recover_unfinished` переводит такие строки в `failed` с кодом
`inconsistent_state`.

### 2.3. Перезапуск и незавершённые попытки

При создании приложения вызывается `SqliteStore.initialize()`, который
выполняет `recover_unfinished()`:

* `queued` и `running` → `failed`, `error_code=interrupted`,
  `error_message` о перезапуске;
* `succeeded` без отчёта → `failed`, `error_code=inconsistent_state`;
* `attempt_no`, timestamps и история задач сохраняются.

Поэтому после падения процесса система видит историю и однозначно обозначает
незавершённые попытки, а не выдаёт их за готовый результат.

## 3. Идемпотентность запуска

`POST /analysis-jobs` требует `idempotency_key` в теле запроса или заголовке
`Idempotency-Key`.

Правила:

* тот же ключ + тот же `submission_hash`, `analyzer_id`, `config_hash` →
  возвращается тот же `job_id` (HTTP 200, `X-Idempotent-Replay: true`);
* тот же ключ + другой payload → `409 idempotency_conflict`;
* повтор после `failed` с тем же ключом → тот же failed job (идемпотентный
  повтор), новый запуск оформляется новым ключом;
* новый ключ после failed/любого завершённого запуска → новая задача с
  `attempt_no = max(previous) + 1`; история сохраняется;
* ключ уникален в БД (`PRIMARY KEY`); job и ключ создаются одной транзакцией
  `add_with_idempotency`, поэтому падение между ними не оставляет задачу без
  ключа;
* проверка и постановка защищены `asyncio.Lock` в одном процессе.

Пример:

```bash
curl -X POST http://127.0.0.1:8000/analysis-jobs \
  -H "X-API-Key: dev-api-key" \
  -H "Content-Type: application/json" \
  -d '{"submission_id":"sub-1","analyzer_id":"baseline","idempotency_key":"demo-key-1"}'
```

Повтор с тем же телом вернёт тот же job, а смена `analyzer_id` при том же
ключе даст `409`.

## 4. API

Запуск:

```bash
uvicorn submission_system.api.app:create_app --factory --reload
```

OpenAPI: `http://127.0.0.1:8000/docs`. API-маршруты (кроме `/health`)
доступны также с префиксом `/api/v1` (например, `/api/v1/submissions`).

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | Проверка сервиса и счётчики recovery; без API-ключа |
| POST | `/submissions` | Создать/получить неизменяемую сдачу |
| GET | `/submissions` | Список сдач |
| GET | `/submissions/{submission_id}` | Оригинальная сдача |
| POST | `/analysis-jobs` | Запуск анализа по `idempotency_key` |
| GET | `/analysis-jobs` | История задач, фильтры `status`, `submission_id` |
| GET | `/analysis-jobs/{job_id}` | Статус и результат |
| GET | `/analysis-jobs/{job_id}/report` | Отчёт (`409`, если не готов) |
| POST | `/analysis-jobs/{job_id}/cancel` | Отмена до старта/во время выполнения |
| GET | `/analyzers` | Доступные адаптеры и версии |

### 4.1. Контроль доступа

Все маршруты, кроме `/health`, требуют ключ:

* `X-API-Key: <ключ>`, или
* `Authorization: Bearer <ключ>`.

Роли:

* `ANALYZER_API_KEY` — admin, полный доступ;
* `ANALYZER_STUDENT_KEYS` — JSON `{"student-key":"student-id"}`;
* `ANALYZER_TEACHER_KEYS` — JSON `{"teacher-key":["hw_1","hw_2"]}`.

Студент может читать и запускать только свои работы и не может создать сдачу
за другого. Преподаватель видит отчёты только по своим `assignment_id`;
запуск и отмена ему запрещены. Admin видит всё.

Ключ по умолчанию для локального запуска — `dev-api-key`. Сравнение
выполняется через `secrets.compare_digest`. Пустая конфигурация всех ключей
отключает проверку только для локальных тестов.

### 4.2. Секреты

`provider_api_key` читается из `MODEL_PROVIDER_API_KEY`, но не входит ни в одну
Pydantic-схему ответа. Тест `test_api_does_not_expose_provider_or_access_secrets`
проверяет, что ни ключ доступа, ни ключ провайдера не появляются в ответах
`/health`, `/analysis-jobs`, отчёте и `/analyzers`.

## 5. Baseline и model-adapter

Оба адаптера реализуют один `AnalysisAdapter`:

* `baseline` — `SyncAnalyzerAdapter(LengthAnalyzer())`, синхронный анализатор
  ЛР1 в пуле;
* `model` — `GuardedModelAdapter(StubModelAdapter())` с ограничением
  конкурентности, batch-конфигурацией и тайм-аутом.

Если внешняя модель не настроена, `ANALYZER_MODEL_MODE=unavailable` включает
`UnavailableAnalysisAdapter`: baseline продолжает работать, а задачи `model`
завершаются `failed` / `model_unavailable`. API и управление задачами при этом
не меняются.

В отчёте сохраняются:

* `model_version`;
* `index_version`;
* `prompt_version`;
* `config_hash`;
* `mlflow_run_id` (из `ANALYZER_MLFLOW_RUN_ID`).

## 6. Настройки окружения

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `ANALYZER_DB_PATH` | `artifacts/lab3/analysis.db` | SQLite-файл |
| `ANALYZER_API_KEY` | `dev-api-key` | Admin-ключ |
| `ANALYZER_STUDENT_KEYS` | `{}` | JSON-карта student-ключей |
| `ANALYZER_TEACHER_KEYS` | `{}` | JSON-карта teacher-ключей и заданий |
| `ANALYZER_EXECUTOR_CONCURRENCY` | `2` | Worker'ы исполнителя |
| `ANALYZER_QUEUE_LIMIT` | `16` | Лимит очереди |
| `ANALYZER_DEFAULT_TIMEOUT` | `30` | Тайм-аут задачи, s |
| `ANALYZER_EXECUTION_MODEL` | `thread` | `thread`/`process` |
| `ANALYZER_BATCH_SIZE` | `4` | Batch модели |
| `ANALYZER_MODEL_CONCURRENCY` | `2` | Конкурентные вызовы модели |
| `ANALYZER_MODEL_TIMEOUT` | `10` | Тайм-аут модели, s |
| `ANALYZER_MODEL_DELAY` | `0` | Управляемая задержка stub |
| `ANALYZER_MODEL_MODE` | `stub` | `stub`/`unavailable` |
| `ANALYZER_MODEL_VERSION` | `stub-model-1.0.0` | Версия модели |
| `ANALYZER_INDEX_VERSION` | `tfidf-lr2` | Версия индекса |
| `ANALYZER_PROMPT_VERSION` | `retrieval-feedback-1.0.0` | Версия промпта |
| `ANALYZER_MLFLOW_RUN_ID` | пусто | Ссылка на MLflow run |
| `MODEL_PROVIDER_API_KEY` | пусто | Секрет провайдера, не отдаётся API |

## 7. Демонстрация

```bash
python scripts/lab3b_demo.py
```

Скрипт создаёт временную SQLite-БД, проходит полный цикл API (создание сдачи,
baseline, model, повтор идемпотентного ключа), затем имитирует падение с
`running`-задачей и показывает восстановление после пересоздания приложения.
Результат: `artifacts/lab3b/api_demo.json`.

## 8. Проверки

```bash
pytest tests/ -q
```

Этап Б покрыт тестами:

* `tests/test_persistence.py` — неизменяемость сдачи, roundtrip задачи и
  отчёта, атомарность `succeeded`+report, атомарная запись job+idempotency key,
  восстановление interrupted/inconsistent, номера попыток;
* `tests/test_api.py` — auth, роли student/teacher/admin, чужой отчёт,
  teacher другого задания, доступ к своей работе, маршруты, идемпотентность
  (включая рестарт), конфликт payload, новая попытка после failed, queue full,
  unavailable model, отмена, восстановление после рестарта, отсутствие
  секретов в ответах.

## 9. Ограничения

1. SQLite подходит для одного сервисного процесса; `recover_unfinished`
   предполагает, что другие процессы не выполняют те же jobs.
2. Idempotency-проверка и постановка защищены `asyncio.Lock` в одном процессе;
   для нескольких реплик нужен внешний уникальный индекс/брокер.
3. Внешний модельный API не вызывается; используется stub или режим
   `unavailable`.
4. Роли student/teacher/admin реализованы на API-ключах; отдельного
   управления пользователями и смены пароля нет.
5. История задач и отчётов хранится в SQLite и растёт; архив/ротация не
   реализованы.
6. MLflow-ссылка хранится как идентификатор run; сам MLflow-сервер не
   поднимается.
7. Пользовательский интерфейс ограничен OpenAPI/Swagger и JSON API.

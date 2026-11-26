# ЛР3, этап А: ограниченный исполнитель анализа

Версия отчёта: **3.0.0-stage-a**. Документ описывает только этап А: очередь,
конкурентность, тайм-аут, отмена, коды ошибок и модельный адаптер. Постоянное
хранение и пользовательский интерфейс относятся к следующему этапу и здесь не
заявляются как готовые.

## 1. Что добавлено к ЛР1–2

| Компонент | Файл | Назначение |
|---|---|---|
| Модель задачи | `src/submission_system/jobs/models.py` | `AnalysisJob`, `JobStatus`, `JobErrorCode`, `SubmitResult`, `JobStore` |
| Исполнитель | `src/submission_system/jobs/executor.py` | `AnalysisExecutor`, `ExecutorConfig`, очередь, worker'ы, тайм-аут и отмена |
| Адаптеры | `src/submission_system/jobs/adapters.py` | `SyncAnalyzerAdapter`, `FunctionAdapter`, `SlowAdapter`, `FailingAdapter`, `HangingAdapter`, `CpuBoundAnalyzer` |
| Модельный адаптер | `src/submission_system/jobs/model_adapter.py` | `GuardedModelAdapter`, `BatchConfig`, `StubModelAdapter`, коды ошибок модели |
| Демонстрация | `scripts/lab3_demo.py` | Очередь, тайм-аут, отмена, коды ошибок, batch |
| Бенчмарк | `scripts/benchmark_executor.py` | Последовательный запуск, потоки и процессы |
| Тесты | `tests/test_jobs_*.py` | 47 тестов этапа А |

Исходные анализаторы и similarity-слой ЛР1–2 не изменены: исполнитель работает
с ними через адаптер и общий контракт `AnalysisReport`.

## 2. Контракт задачи и статусов

`AnalysisJob` содержит обязательные поля:

* `job_id` — уникальный идентификатор задачи;
* `submission_id` — ссылка на сдачу в репозитории;
* `submission_hash` — SHA-256 от `code + text`, фиксирует вход;
* `analyzer_id` — выбранный адаптер;
* `config_hash` — отпечаток конфигурации адаптера;
* `status` — статус из общего контракта;
* `attempt_no` — номер попытки (повторы появятся в ЛР4);
* `timeout_seconds` — тайм-аут конкретной задачи;
* `created_at`, `queued_at`, `started_at`, `finished_at`, `updated_at`.

Дополнительно сохраняются `error_code`, `error_message` и `result`
(`AnalysisReport`).

Общий контракт статусов:

| Статус | Терминальный | Смысл |
|---|---:|---|
| `queued` | нет | Принята и ждёт свободного worker'а |
| `running` | нет | Выполняется |
| `succeeded` | да | Результат сохранён |
| `failed` | да | Ошибка адаптера/анализатора |
| `timed_out` | да | Истёк тайм-аут |
| `cancelled` | да | Отменена до старта или во время выполнения |
| `rejected` | да | Явный отказ в постановке (например, очередь полна) |

`JobStatus` наследуется от `str`, поэтому `job.status == "queued"` работает и
для проверок, и при сериализации через `.value`.

Переходы выполняются explicit-методами `mark_running`, `mark_succeeded`,
`mark_failed`, `mark_timed_out`, `mark_cancelled`, `mark_rejected`. Повторный
старт терминальной задачи поднимает `JobStateError`.

## 3. Постановка и выполнение

`build_analysis_job` создаёт описание задачи, но не ставит её в очередь.
`AnalysisExecutor.submit`:

1. проверяет адаптер и закрытие исполнителя;
2. находит или принимает `Submission`;
3. сохраняет сдачу в `InMemoryRepository` **до** проверки очереди;
4. проверяет конфликт ID/содержимого;
5. создаёт задачу;
6. принимает её в очередь или возвращает `SubmitResult(accepted=False)`.

`submit` не ждёт анализа: он сразу возвращает решение о постановке. Worker'ы
забирают задачи из `deque` под `asyncio.Condition`. Поэтому постановка и
выполнение разделены, а не смешаны в одном вызове.

При переполнении очередь возвращается `JobErrorCode.QUEUE_FULL`, статус задачи
— `rejected`. Исходная сдача при этом уже сохранена и не удаляется. Это
проверяется в `test_queue_limit_rejects_explicitly_and_keeps_submission`.

## 4. Ограничения конкурентности и очереди

`ExecutorConfig`:

* `concurrency_limit` — число worker'ов и одновременно выполняющихся задач;
* `queue_limit` — число задач, ожидающих сверх немедленной мощности;
* `default_timeout` — тайм-аут по умолчанию;
* `executor_kind` — `thread` или `process`;
* `auto_start` — запускать worker'ы при первом `submit` или требовать
  явного `start()`.

Семантика очереди: ближайшие `concurrency_limit` задач лежат в отдельной
очереди `_starting` и резервируют немедленный запуск. Собственно очередь
`_pending` содержит только ожидающие задачи и не может быть длиннее
`queue_limit`. Поэтому при `concurrency_limit=1, queue_limit=1` принимаются
две задачи, а третья получает `queue_full`; при `queue_limit=0` задача
принимается только при свободной мощности. Всплеск `submit` не раздувает
`_pending`: лишние запросы сразу получают `rejected`.

Число одновременно работающих анализаторов не превышает
`concurrency_limit`: это обеспечивается числом worker'ов и размером пула
`ThreadPoolExecutor`/`ProcessPoolExecutor`. Даже если async-задача отменена по
тайм-ауту, CPU-функция в пуле продолжает занимать поток, поэтому новый запуск
не превышает лимит фактических CPU-исполнителей.

## 5. Тайм-аут, отмена и `CancelledError`

Каждая задача получает `timeout_seconds`. Worker ждёт дочернюю задачу через
`asyncio.wait(..., timeout=...)`:

* задача завершилась — сохраняется `result` и статус `succeeded`;
* тайм-аут — дочерняя задача отменяется, статус `timed_out`, код `timeout`;
* задача отменена пользователем — статус `cancelled`, код `cancelled`;
* исключение адаптера — `failed` с кодом из исключения.

`cancel(job_id)`:

* для `queued` удаляет задачу из очереди и сразу ставит `cancelled`;
* для `running` отменяет `asyncio.Task`, а worker фиксирует терминальный
  статус;
* для терминальной задачи возвращает `False`.

`CancelledError` не превращается в `failed`:
* адаптер, который сам поднимает `CancelledError`, даёт `cancelled`;
* отмена worker'а при shutdown перехватывается только для очистки дочерней
  задачи и затем пробрасывается дальше;
* `suppress(CancelledError)` используется только после осознанной отмены по
  тайм-ауту/shutdown, чтобы дождаться завершения дочерней задачи.

Ограничение: для синхронной функции в потоке отмена не убивает поток
принудительно. Задача помечается `cancelled`, но CPU-функция может доработать в
фоне; поток остаётся занятым и продолжает учитываться пулом. Поэтому
`shutdown` всегда ждёт уже запущенные вызовы пула (даже при `wait=False`),
чтобы после остановки не осталось задач, которыми исполнитель уже не
управляет. Для жёсткой отмены предусмотрен `executor_kind="process"`, но по
умолчанию выбран поток — см. раздел 6.

## 6. Модель исполнения и измерения

Синхронные анализаторы ЛР1 запускаются через `SyncAnalyzerAdapter` в отдельном
пуле, а не в event loop. Поэтому event loop продолжает обрабатывать другие
задачи, пока CPU-анализатор работает в потоке/процессе.

Бенчмарк `scripts/benchmark_executor.py`:

```bash
python scripts/benchmark_executor.py --jobs 8 --workers 4 --iterations 400000 --repeats 3
```

Параметры: 8 задач, 4 worker'а, 400 000 итераций CPU-нагрузки, 3 повтора.
Результат `artifacts/lab3/executor_benchmark.json`:

| Режим | Медиана времени | Медиана throughput |
|---|---:|---:|
| последовательный | 0.366 s | 21.84 jobs/s |
| потоки | 0.341 s | 23.48 jobs/s |
| процессы | 0.524 s | 15.27 jobs/s |

Выбран пул потоков: на учебном объёме он не требует сериализации объектов
между процессами и не платит за spawn на Windows. Пул процессов оставлен
опцией для будущих тяжёлых CPU-адаптеров. Управление задачей, тайм-аут и
отмена от выбора пула не зависят.

Отдельно тест `test_required_cpu_analysis_does_not_block_event_loop` считает
тики event loop во время CPU-анализа и требует, чтобы они происходили.

## 7. Модельный адаптер AI/MLOps

`GuardedModelAdapter` оборачивает любой `ModelAdapter`:

* `asyncio.Semaphore(concurrency_limit)` ограничивает конкурентные обращения к
  модели;
* `BatchConfig.batch_size` фиксирует размер пакета; `score_batch` режет вход
  на пакеты этого размера;
* `call_timeout` ограничивает один вызов модели;
* `StubModelAdapter` и настоящий адаптер реализуют один протокол, поэтому
  замена не меняет управление задачей.

Коды ошибок:

| Ситуация | Исключение | `JobErrorCode` | Статус задачи |
|---|---|---|---|
| Тайм-аут модели | `ModelTimeoutError` | `timeout` | `timed_out` |
| Модель недоступна | `ModelUnavailableError` | `model_unavailable` | `failed` |
| NaN/∞/score вне `[0,1]` | `InvalidModelResponseError` | `invalid_model_response` | `failed` |
| Прочая ошибка адаптера | `AdapterError` | `analyzer_error` | `failed` |
| Отмена | `CancelledError` | `cancelled` | `cancelled` |

Невалидный ответ не подменяется нулём: `validate_analysis_report` и
`_validate_score` поднимают исключение, а исполнитель сохраняет отдельный код.
Повторы временных ошибок намеренно не реализованы — они запланированы на ЛР4.

### 7.1. Политика ошибок: collect-errors

Выбрана политика **collect-errors**, а не fail-fast:

* каждая задача изолирована; `failed`, `timed_out` или `cancelled` одной
  задачи не останавливает worker и не отменяет остальные;
* результаты уже успешных задач остаются в `JobStore` и доступны через
  `get_result` / `results`;
* исполнитель продолжает принимать новые задачи после ошибки;
* итоговое состояние видно в `stats()["by_status"]` и в статусе каждой задачи.

Это проверяется тестами `test_one_failure_does_not_lose_other_results` и
`test_timeout_does_not_lose_other_results`. Fail-fast не используется: для
пакетной проверки работ потеря остальных результатов из-за одной ошибки
недопустима.

## 8. Демонстрация

`python scripts/lab3_demo.py` сохраняет
`artifacts/lab3/stage_a_demo.json`. В прогоне видно:

* `concurrency_limit=1, queue_limit=1`: две задачи приняты, третья получила
  `queue_full`, при этом `demo-sub-2` осталась в репозитории;
* задача с тайм-аутом 0.05 s получила `timed_out` / `timeout`;
* отмена до старта и во время выполнения дала `cancelled`;
* недоступность, невалидный ответ и отмена имеют разные коды;
* batch-адаптер при `batch_size=2` наблюдал пакеты `[2, 2, 1]`, максимум
  одновременных вызовов — 2.

## 9. Проверки

Новые тесты:

* `tests/test_jobs_models.py` — контракт задачи, статусы, переходы, store;
* `tests/test_jobs_executor.py` — конкурентность, очередь, тайм-аут, отмена,
  коды, сохранение результата, event loop, конфликт сдачи;
* `tests/test_jobs_model_adapter.py` — batch, лимит модели, коды ошибок,
  замена stub, валидация конфигурации;
* `tests/test_jobs_required_checks.py` — обязательные проверки этапа А.

Полный прогон:

```text
124 passed, 1 warning
```

Из них 47 тестов этапа А; предупреждение относится к SQLAlchemy внутри MLflow и
к ЛР3 не относится.

### 9.1. Приёмка этапа А

| Пункт плана | Проверка | Результат |
|---|---|---|
| Тестовый адаптер подтверждает соблюдение лимита | `test_concurrency_limit_is_respected`, `test_required_queue_and_concurrency_limits`; `SlowAdapter.max_active <= concurrency_limit` | PASS |
| Тайм-аут переводит job в правильный статус | `test_timeout_has_its_own_terminal_status`, `test_required_timeout_is_not_failure` | PASS |
| Отмена переводит job в правильный статус | `test_cancel_before_start`, `test_cancel_during_execution`, `test_required_cancel_before_and_during_execution` | PASS |
| Очередь переполняется предсказуемо и не растёт без ограничения | `test_queue_limit_rejects_explicitly_and_keeps_submission`, `test_queue_limit_zero_allows_only_immediate_capacity`, `test_queue_stays_bounded_under_pressure`; `pending_count <= queue_limit` | PASS |
| После остановки нет задач, которыми исполнитель не управляет | `test_shutdown_waits_for_running_sync_work_and_leaves_no_tasks`, `test_shutdown_without_wait_cancels_pending_and_running`; `shutdown` ждёт пул | PASS |
| Один провал не теряет остальные результаты | `test_one_failure_does_not_lose_other_results`, `test_timeout_does_not_lose_other_results` | PASS |
| Политика fail-fast либо collect-errors документирована | раздел 7.1, `docs/LAB3.md`; выбрана collect-errors | PASS |

## 10. Ограничения этапа А

1. Результаты и задачи хранятся в памяти; после перезапуска процесса они
   теряются, а история задач растёт линейно. Очередь при этом ограничена
   `queue_limit`; постоянное хранение — этап Б.
2. Пользовательский интерфейс не реализован: есть CLI-демонстрация и JSON-
   артефакты.
3. Отмена синхронной функции в потоке кооперативная: поток не убивается
   принудительно. Для жёсткой отмены нужен процессный пул и сериализуемые
   адаптеры.
4. Нет автоматических повторов: `attempt_no` зарезервирован под ЛР4.
5. Нет распределённой очереди и внешнего брокера; исполнитель однопроцессный.
6. Модельный адаптер использует детерминированную заглушку; реальный внешний
   API не вызывается.
7. `ProcessPoolExecutor` на Windows требует запуска через `if __name__ ==
   "__main__"`; это учтено в бенчмарке, но не является режимом по умолчанию.
8. Один экземпляр `AnalysisExecutor` привязан к одному event loop; повторное
   использование после `asyncio.run` не поддерживается.

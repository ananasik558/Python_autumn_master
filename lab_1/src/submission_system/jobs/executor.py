"""Ограниченный исполнитель анализа ЛР3, этап А.

Исполнитель разделяет две операции:

* ``submit`` — принимает сдачу, сохраняет её и ставит задачу в ограниченную
  очередь;
* worker-цикл — забирает задачу из очереди и выполняет адаптер.

Инварианты:

* одновременно выполняющихся задач не больше ``concurrency_limit``;
* число ожидающих задач ограничено ``queue_limit``; при переполнении
  возвращается явный отказ ``queue_full``, а сдача остаётся в репозитории;
* каждая задача имеет тайм-аут;
* отмена работает и до старта, и во время выполнения;
* результат успешной задачи сохраняется в памяти до этапа Б.

Синхронные анализаторы ЛР1 выполняются в отдельном пуле потоков или процессов,
поэтому CPU-работа не блокирует event loop.
"""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Optional

from ..analysis_service import validate_analysis_report
from ..models import AnalysisReport, Submission
from ..repository import InMemoryRepository
from .adapters import AdapterError, AnalysisAdapter
from .models import (
    AnalysisJob,
    JobErrorCode,
    JobStateError,
    JobStatus,
    JobStore,
    SubmitResult,
    build_analysis_job,
    utc_now,
)


ExecutorKind = Literal["thread", "process"]


@dataclass(frozen=True)
class ExecutorConfig:
    """Параметры ограниченного исполнителя."""

    concurrency_limit: int = 2
    queue_limit: int = 16
    default_timeout: float = 30.0
    executor_kind: ExecutorKind = "thread"
    auto_start: bool = True

    def __post_init__(self) -> None:
        if self.concurrency_limit < 1:
            raise ValueError("concurrency_limit должен быть >= 1")
        if self.queue_limit < 0:
            raise ValueError("queue_limit не может быть отрицательным")
        if self.default_timeout < 0:
            raise ValueError("default_timeout не может быть отрицательным")
        if self.executor_kind not in {"thread", "process"}:
            raise ValueError("executor_kind должен быть 'thread' или 'process'")

    @property
    def execution_model(self) -> ExecutorKind:
        return self.executor_kind


class AnalysisExecutor:
    """Асинхронный исполнитель задач анализа с ограничением ресурсов."""

    def __init__(
        self,
        adapters: Mapping[str, AnalysisAdapter] | Iterable[AnalysisAdapter],
        config: ExecutorConfig | None = None,
        *,
        repository: Any | None = None,
        job_store: Any | None = None,
        pool: Executor | None = None,
    ) -> None:
        self.config = config or ExecutorConfig()
        if isinstance(adapters, Mapping):
            self._adapters = dict(adapters)
        else:
            self._adapters = {adapter.analyzer_id: adapter for adapter in adapters}
        if not self._adapters:
            raise ValueError("Нужен хотя бы один адаптер")

        self.repository = (
            repository if repository is not None else InMemoryRepository()
        )
        self.jobs = job_store if job_store is not None else JobStore()
        # Задачи, для которых уже зарезервировано место под немедленный старт.
        self._starting: deque[str] = deque()
        # Собственно очередь ожидания; её размер ограничен queue_limit.
        self._pending: deque[str] = deque()
        self._active: set[str] = set()
        self._tasks: dict[str, asyncio.Task[AnalysisReport]] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._condition = asyncio.Condition()
        self._workers: list[asyncio.Task[Any]] = []
        self._started = False
        self._closed = False

        if pool is not None:
            self._pool = pool
            self._owns_pool = False
        else:
            self._pool = self._create_pool()
            self._owns_pool = True

    @property
    def concurrency_limit(self) -> int:
        return self.config.concurrency_limit

    @property
    def queue_limit(self) -> int:
        return self.config.queue_limit

    @property
    def pending_count(self) -> int:
        """Размер очереди ожидания (не включает зарезервированные старты)."""

        return len(self._pending)

    @property
    def starting_count(self) -> int:
        return len(self._starting)

    @property
    def running_count(self) -> int:
        return len(self._active)

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------
    def _create_pool(self) -> Executor:
        if self.config.executor_kind == "process":
            return ProcessPoolExecutor(max_workers=self.config.concurrency_limit)
        return ThreadPoolExecutor(
            max_workers=self.config.concurrency_limit,
            thread_name_prefix="analysis-job",
        )

    def start(self) -> None:
        """Запускает worker-циклы. Требует активного event loop."""

        if self._closed:
            raise RuntimeError("Исполнитель уже остановлен")
        if self._started:
            return
        self._started = True
        for index in range(self.config.concurrency_limit):
            self._workers.append(
                asyncio.create_task(
                    self._worker_loop(index),
                    name=f"analysis-worker-{index}",
                )
            )

    async def __aenter__(self) -> "AnalysisExecutor":
        self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.shutdown()

    async def shutdown(
        self,
        *,
        wait: bool = True,
        cancel_pending: bool = False,
    ) -> None:
        """Останавливает исполнитель.

        ``wait=True`` дообрабатывает уже принятую очередь. ``wait=False``
        отменяет ожидающие и выполняющиеся задачи. В обоих режимах shutdown
        дожидается уже запущенных вызовов пула: Python-поток нельзя безопасно
        убить, поэтому после возврата из shutdown исполнитель не оставляет
        задач, которыми больше не управляет.
        """

        if self._closed:
            return
        self._closed = True

        if (cancel_pending or not wait or not self._started) and self._pending:
            # Если worker'ы не будут ждать завершения или вообще не запускались,
            # ожидающие задачи нельзя оставлять в подвешенном виде.
            async with self._condition:
                for queue in (self._starting, self._pending):
                    while queue:
                        job_id = queue.popleft()
                        job = self.jobs.get(job_id)
                        if job is not None and job.status is JobStatus.QUEUED:
                            job.mark_cancelled(
                                "Исполнитель остановлен до старта задачи"
                            )
                            self._persist_job(job)
                            self._signal_done(job_id)
                self._condition.notify_all()
        else:
            async with self._condition:
                self._condition.notify_all()

        if wait:
            if self._workers:
                await asyncio.gather(*self._workers, return_exceptions=True)
        else:
            for worker in self._workers:
                worker.cancel()
            if self._workers:
                await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()
        if self._owns_pool:
            # cancel_futures=True не даёт выполнить отложенные CPU-вызовы
            # отменённых/просроченных задач после остановки. wait=True
            # гарантирует, что уже запущенный поток/процесс завершён.
            self._pool.shutdown(  # type: ignore[attr-defined]
                wait=True,
                cancel_futures=True,
            )
        self._active.clear()
        self._starting.clear()
        self._pending.clear()
        self._tasks.clear()

    async def aclose(self) -> None:
        await self.shutdown()

    # ------------------------------------------------------------------
    # Постановка задачи
    # ------------------------------------------------------------------
    async def submit(
        self,
        submission: Submission | str,
        analyzer_id: str,
        *,
        timeout: float | None = None,
        job_id: str | None = None,
        config_hash: str | None = None,
        attempt_no: int = 1,
        submission_hash: str | None = None,
    ) -> SubmitResult:
        """Ставит задачу в очередь и сразу возвращает решение о постановке.

        Метод не ждёт выполнения. Если очередь заполнена, возвращается
        ``SubmitResult(accepted=False, error_code=queue_full)``. Исходная
        сдача сохраняется до проверки очереди, поэтому отказ не теряет данные.
        """

        if self._closed:
            raise RuntimeError("Нельзя ставить задачи после shutdown")
        if self.config.auto_start:
            self.start()
        elif not self._started:
            raise RuntimeError(
                "auto_start=False: вызовите executor.start() перед постановкой задач"
            )

        if analyzer_id not in self._adapters:
            raise KeyError(f"Неизвестный анализатор: {analyzer_id}")

        submission_obj = self._resolve_submission(submission)
        adapter = self._adapters[analyzer_id]

        # Сохраняем вход до проверки очереди: отказ не должен терять сдачу.
        existing = self.repository.get(submission_obj.submission_id)
        submission_digest = _submission_hash(submission_obj)
        existing_digest = (
            _submission_hash(existing) if existing is not None else None
        )
        if existing is not None and existing_digest != submission_digest:
            job = build_analysis_job(
                submission_obj,
                analyzer_id,
                config_hash=config_hash or adapter.config_hash,
                timeout_seconds=(
                    self.config.default_timeout if timeout is None else timeout
                ),
                attempt_no=attempt_no,
                job_id=job_id,
                submission_hash=submission_digest,
            )
            job.mark_rejected(
                JobErrorCode.SUBMISSION_CONFLICT,
                "ID сдачи уже занят другой версией содержимого",
            )
            self.jobs.add(job)
            self._events[job.job_id] = _make_event()
            self._events[job.job_id].set()
            return SubmitResult(
                accepted=False,
                job=job,
                error_code=JobErrorCode.SUBMISSION_CONFLICT,
                message=job.error_message or "",
            )

        if existing is None:
            saved = Submission(
                student_id=submission_obj.student_id,
                submission_id=submission_obj.submission_id,
                timestamp=submission_obj.timestamp,
                code=submission_obj.code,
                text=submission_obj.text,
                ai_declaration=submission_obj.ai_declaration,
                assignment_id=submission_obj.assignment_id,
                raw_content_hash=submission_digest,
            )
            self.repository.save(saved)

        job = build_analysis_job(
            submission_obj,
            analyzer_id,
            config_hash=config_hash or adapter.config_hash,
            timeout_seconds=(
                self.config.default_timeout if timeout is None else timeout
            ),
            attempt_no=attempt_no,
            job_id=job_id,
            submission_hash=submission_digest,
        )

        async with self._condition:
            # Часть мест резервируется под немедленный старт: это не очередь,
            # поэтому queue_limit применяется только к _pending. Так размер
            # очереди остаётся предсказуемым и не растёт при всплеске submit.
            has_immediate_slot = (
                len(self._active) + len(self._starting)
                < self.config.concurrency_limit
            )
            if not has_immediate_slot and self._queue_is_full():
                job.mark_rejected(
                    JobErrorCode.QUEUE_FULL,
                    (
                        f"Очередь заполнена: concurrency_limit="
                        f"{self.config.concurrency_limit}, "
                        f"queue_limit={self.config.queue_limit}"
                    ),
                )
                self.jobs.add(job)
                self._events[job.job_id] = _make_event()
                self._events[job.job_id].set()
                return SubmitResult(
                    accepted=False,
                    job=job,
                    error_code=JobErrorCode.QUEUE_FULL,
                    message=job.error_message or "",
                )

            self.jobs.add(job)
            self._events[job.job_id] = _make_event()
            target = self._starting if has_immediate_slot else self._pending
            target.append(job.job_id)
            self._condition.notify_all()
            return SubmitResult(accepted=True, job=job)

    async def enqueue_prebuilt(self, job: AnalysisJob) -> SubmitResult:
        """Ставит в очередь уже сохранённую задачу.

        Нужно сервису этапа Б: job и idempotency_key записываются в SQLite
        одной транзакцией, после чего задача передаётся исполнителю. Повторная
        постановка того же job_id не создаёт вторую запись.
        """

        if self._closed:
            raise RuntimeError("Нельзя ставить задачи после shutdown")
        if self.config.auto_start:
            self.start()
        elif not self._started:
            raise RuntimeError(
                "auto_start=False: вызовите executor.start() перед постановкой задач"
            )
        if job.analyzer_id not in self._adapters:
            raise KeyError(f"Неизвестный анализатор: {job.analyzer_id}")
        if self.repository.get(job.submission_id) is None:
            raise KeyError(f"Сдача не найдена: {job.submission_id}")
        if job.status is not JobStatus.QUEUED:
            raise JobStateError(
                f"enqueue_prebuilt ожидает queued, получен {job.status.value}"
            )
        if job.job_id not in self.jobs:
            self.jobs.add(job)

        async with self._condition:
            if job.job_id in self._starting or job.job_id in self._pending:
                return SubmitResult(accepted=True, job=job)
            if job.job_id in self._active:
                return SubmitResult(accepted=True, job=self.jobs.require(job.job_id))
            has_immediate_slot = (
                len(self._active) + len(self._starting)
                < self.config.concurrency_limit
            )
            if not has_immediate_slot and self._queue_is_full():
                job.mark_rejected(
                    JobErrorCode.QUEUE_FULL,
                    (
                        f"Очередь заполнена: concurrency_limit="
                        f"{self.config.concurrency_limit}, "
                        f"queue_limit={self.config.queue_limit}"
                    ),
                )
                self.jobs.save(job)
                self._events[job.job_id] = _make_event()
                self._events[job.job_id].set()
                return SubmitResult(
                    accepted=False,
                    job=job,
                    error_code=JobErrorCode.QUEUE_FULL,
                    message=job.error_message or "",
                )

            self._events[job.job_id] = _make_event()
            target = self._starting if has_immediate_slot else self._pending
            target.append(job.job_id)
            self._condition.notify_all()
            return SubmitResult(accepted=True, job=job)

    def _resolve_submission(self, submission: Submission | str) -> Submission:
        if isinstance(submission, Submission):
            return submission
        if isinstance(submission, str):
            existing = self.repository.get(submission)
            if existing is None:
                raise KeyError(f"Сдача не найдена: {submission}")
            return existing
        raise TypeError("submission должен быть Submission или submission_id")

    def _queue_is_full(self) -> bool:
        """Проверяет лимит именно очереди ожидания.

        Задачи, зарезервированные под немедленный старт, лежат в отдельной
        очереди ``_starting`` и не занимают ``queue_limit``. Поэтому
        ``len(_pending)`` никогда не превышает ``queue_limit``.
        """

        return len(self._pending) >= self.config.queue_limit

    # ------------------------------------------------------------------
    # Отмена и ожидание
    # ------------------------------------------------------------------
    async def cancel(self, job_id: str) -> bool:
        """Отменяет задачу до старта или во время выполнения.

        Возвращает ``True``, если задача была активной и отмена запрошена.
        Для уже терминальной задачи возвращается ``False``.
        """

        job = self.jobs.get(job_id)
        if job is None or job.is_terminal:
            return False

        async with self._condition:
            if job.status is JobStatus.QUEUED:
                for queue in (self._starting, self._pending):
                    if job_id in queue:
                        queue.remove(job_id)
                        break
                job.mark_cancelled("Задача отменена до старта")
                self._persist_job(job)
                self._signal_done(job_id)
                self._condition.notify_all()
                return True

            if job.status is JobStatus.RUNNING:
                task = self._tasks.get(job_id)
                if task is not None and not task.done():
                    task.cancel()
                    return True
        return False

    async def wait(
        self,
        job_id: str,
        timeout: float | None = None,
    ) -> AnalysisJob:
        """Ждёт терминального статуса задачи и возвращает её."""

        job = self.jobs.require(job_id)
        event = self._events.setdefault(job_id, _make_event())
        if job.is_terminal:
            return job
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return job

    async def wait_all(
        self,
        job_ids: Iterable[str],
        timeout: float | None = None,
    ) -> list[AnalysisJob]:
        return await asyncio.gather(
            *(self.wait(job_id, timeout=timeout) for job_id in job_ids)
        )

    def get_job(self, job_id: str) -> Optional[AnalysisJob]:
        return self.jobs.get(job_id)

    def get_result(self, job_id: str) -> Optional[AnalysisReport]:
        job = self.jobs.get(job_id)
        return job.result if job is not None else None

    @property
    def results(self) -> dict[str, AnalysisReport]:
        return {
            job.job_id: job.result
            for job in self.jobs.values()
            if job.result is not None
        }

    @property
    def adapters(self) -> dict[str, AnalysisAdapter]:
        return dict(self._adapters)

    def get_adapter(self, analyzer_id: str) -> AnalysisAdapter:
        return self._adapters[analyzer_id]

    def _persist_job(self, job: AnalysisJob) -> None:
        """Сохраняет снимок задачи, если хранилище это поддерживает."""

        save = getattr(self.jobs, "save", None)
        if save is not None:
            save(job)

    def list_jobs(self, status: JobStatus | None = None) -> list[AnalysisJob]:
        return self.jobs.list(status)

    def stats(self) -> dict[str, Any]:
        counts = {status.value: 0 for status in JobStatus}
        for job in self.jobs.values():
            counts[job.status.value] += 1
        return {
            "concurrency_limit": self.config.concurrency_limit,
            "queue_limit": self.config.queue_limit,
            "executor_kind": self.config.executor_kind,
            "queued": len(self._pending),
            "starting": len(self._starting),
            "running": len(self._active),
            "total": len(self.jobs),
            "by_status": counts,
        }

    # Совместимые имена для отчётов/тестов.
    enqueue = submit

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    async def _worker_loop(self, worker_index: int) -> None:
        del worker_index
        while True:
            async with self._condition:
                while not self._starting and not self._pending and not self._closed:
                    await self._condition.wait()
                if self._closed and not self._starting and not self._pending:
                    return
                if self._starting:
                    job_id = self._starting.popleft()
                else:
                    job_id = self._pending.popleft()
                job = self.jobs.get(job_id)
                if job is None or job.status is not JobStatus.QUEUED:
                    continue
                try:
                    job.mark_running()
                except JobStateError:
                    continue
                self._active.add(job_id)
                task = asyncio.create_task(
                    self._execute_job(job),
                    name=f"analysis-{job.job_id}",
                )
                self._tasks[job_id] = task
                self._condition.notify_all()

            self._persist_job(job)
            await self._await_job(job, task)
            # Освободилось место: будим остальных worker'ов, если в очереди
            # уже лежат задачи.
            async with self._condition:
                self._condition.notify_all()

    async def _execute_job(self, job: AnalysisJob) -> AnalysisReport:
        submission = self.repository.get(job.submission_id)
        if submission is None:
            raise AdapterError(
                f"Сдача {job.submission_id} не найдена",
                code=JobErrorCode.SUBMISSION_NOT_FOUND,
            )
        adapter = self._adapters.get(job.analyzer_id)
        if adapter is None:
            raise AdapterError(
                f"Анализатор {job.analyzer_id} не найден",
                code=JobErrorCode.ANALYZER_NOT_FOUND,
            )
        report = await adapter.analyze(submission, job, executor=self._pool)
        validate_analysis_report(report)
        return report

    async def _await_job(
        self,
        job: AnalysisJob,
        task: asyncio.Task[AnalysisReport],
    ) -> None:
        """Ждёт задачу, применяя тайм-аут и обрабатывая отмену."""

        try:
            done, pending = await asyncio.wait(
                {task},
                timeout=job.timeout_seconds,
            )
        except asyncio.CancelledError:
            # Останавливают самого worker'а: отменяем дочернюю задачу, но
            # CancelledError не проглатываем.
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            if job.status is JobStatus.RUNNING:
                job.mark_cancelled("Worker остановлен во время выполнения")
            self._finish_job(job)
            raise

        if pending:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            if job.status is JobStatus.RUNNING:
                job.mark_timed_out(
                    f"Превышен тайм-аут {job.timeout_seconds}s"
                )
            self._finish_job(job)
            return

        if task.cancelled():
            if job.status is JobStatus.RUNNING:
                job.mark_cancelled("Задача отменена во время выполнения")
        else:
            exception = task.exception()
            if exception is None:
                report = task.result()
                if job.status is JobStatus.RUNNING:
                    job.mark_succeeded(report)
            else:
                self._apply_exception(job, exception)
        self._finish_job(job)

    def _apply_exception(self, job: AnalysisJob, exception: BaseException) -> None:
        if job.status is not JobStatus.RUNNING:
            return
        if isinstance(exception, AdapterError):
            if exception.code is JobErrorCode.TIMEOUT:
                job.mark_timed_out(str(exception))
            else:
                job.mark_failed(exception.code, str(exception))
            return
        if isinstance(exception, asyncio.TimeoutError):
            job.mark_timed_out(str(exception) or "Тайм-аут адаптера")
            return
        if isinstance(exception, asyncio.CancelledError):
            job.mark_cancelled("Задача отменена")
            return
        job.mark_failed(
            JobErrorCode.ANALYZER_ERROR,
            f"{type(exception).__name__}: {exception}",
        )

    def _finish_job(self, job: AnalysisJob) -> None:
        self._active.discard(job.job_id)
        self._tasks.pop(job.job_id, None)
        self._persist_job(job)
        self._signal_done(job.job_id)
        # notify_all нельзя вызывать без захвата Condition. В этом месте мы
        # уже вне критической секции; для ожидающих stats/wait событие уже
        # выставлено, а worker сам проверит очередь на следующей итерации.
        if job.updated_at is None:
            job.updated_at = utc_now()

    def _signal_done(self, job_id: str) -> None:
        event = self._events.get(job_id)
        if event is not None:
            event.set()


def _make_event() -> asyncio.Event:
    return asyncio.Event()


def _submission_hash(submission: Submission) -> str:
    import hashlib

    content = (submission.code or "") + (submission.text or "")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

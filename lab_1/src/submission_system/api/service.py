"""Прикладной слой FastAPI-сервиса ЛР3, этап Б.

Сервис связывает SQLite-хранилище, ограниченный исполнитель и API-схемы.
Job и idempotency_key создаются одной транзакцией; дополнительно проверка и
постановка защищены asyncio-замком внутри одного процесса.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from ..jobs import (
    AnalysisExecutor,
    AnalysisJob,
    JobErrorCode,
    JobStatus,
    build_analysis_job,
)
from ..models import AnalysisReport, Submission
from ..persistence import (
    DuplicateIdempotencyKey,
    SqliteStore,
    submission_digest,
)
from ..validator import validate_submission
from .auth import Actor
from .schemas import (
    AnalysisStartRequest,
    AnalyzerInfo,
    SubmissionCreate,
)
from .settings import Settings


class ServiceError(Exception):
    """Ошибка прикладного слоя с HTTP-кодом и машинным кодом."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        job: Optional[AnalysisJob] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.job = job


class AnalysisApiService:
    def __init__(
        self,
        store: SqliteStore,
        executor: AnalysisExecutor,
        settings: Settings,
    ) -> None:
        self.store = store
        self.executor = executor
        self.settings = settings
        self._idempotency_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Сдачи
    # ------------------------------------------------------------------
    def create_submission(
        self,
        request: SubmissionCreate,
        actor: Optional[Actor] = None,
    ) -> tuple[Submission, bool]:
        if actor is not None and actor.is_teacher:
            raise ServiceError(
                403,
                "forbidden",
                "Преподаватель не создаёт студенческие сдачи",
            )
        if (
            actor is not None
            and actor.is_student
            and actor.student_id != request.student_id
        ):
            raise ServiceError(
                403,
                "forbidden",
                "Студент может создать сдачу только для себя",
            )
        payload = request.model_dump()
        submission, error = validate_submission(payload, 1)
        if error is not None or submission is None:
            raise ServiceError(
                422,
                error.code if error is not None else "invalid_submission",
                error.message if error is not None else "Некорректная сдача",
            )

        digest = submission.raw_content_hash or submission_digest(submission)
        existing = self.store.submissions.get(submission.submission_id)
        if existing is not None:
            if submission_digest(existing) != digest:
                raise ServiceError(
                    409,
                    "submission_conflict",
                    "Оригинальная версия сдачи неизменяема: ID уже занят другим содержимым",
                )
            return existing, False

        if not self.store.submissions.save(submission):
            # Гонка: другой запрос успел сохранить эту же сдачу.
            existing = self.store.submissions.get(submission.submission_id)
            if existing is not None and submission_digest(existing) == digest:
                return existing, False
            raise ServiceError(
                409,
                "submission_conflict",
                "Не удалось сохранить сдачу: ID уже занят",
            )
        return submission, True

    def get_submission(
        self,
        submission_id: str,
        actor: Optional[Actor] = None,
    ) -> Submission:
        submission = self.store.submissions.get(submission_id)
        if submission is None:
            raise ServiceError(404, "submission_not_found", "Сдача не найдена")
        self._authorize_read(actor, submission)
        return submission

    def list_submissions(
        self,
        *,
        actor: Optional[Actor] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Submission]:
        submissions = self.store.submissions.list(limit=1_000_000)
        if actor is not None and not actor.is_admin:
            submissions = [
                item
                for item in submissions
                if actor.can_read_submission(item)
            ]
        return submissions[offset:offset + limit]

    def _authorize_read(
        self,
        actor: Optional[Actor],
        submission: Submission,
    ) -> None:
        if actor is None or actor.is_admin:
            return
        if actor.can_read_submission(submission):
            return
        raise ServiceError(
            403,
            "forbidden",
            "Нет доступа к этой работе",
        )

    def _authorize_start(
        self,
        actor: Optional[Actor],
        submission: Submission,
    ) -> None:
        if actor is None or actor.is_admin:
            return
        if actor.can_start_submission(submission):
            return
        raise ServiceError(
            403,
            "forbidden",
            "Запускать анализ может только владелец работы",
        )

    # ------------------------------------------------------------------
    # Запуск анализа и идемпотентность
    # ------------------------------------------------------------------
    async def start_analysis(
        self,
        request: AnalysisStartRequest,
        actor: Optional[Actor] = None,
    ) -> tuple[AnalysisJob, bool]:
        if not request.idempotency_key:
            raise ServiceError(
                422,
                "idempotency_key_required",
                "Нужен idempotency_key в теле или заголовке Idempotency-Key",
            )
        adapter = self.executor.adapters.get(request.analyzer_id)
        if adapter is None:
            raise ServiceError(
                404,
                "analyzer_not_found",
                f"Неизвестный анализатор: {request.analyzer_id}",
            )
        submission = self.get_submission(request.submission_id, actor)
        self._authorize_start(actor, submission)
        digest = submission.raw_content_hash or submission_digest(submission)
        if request.submission_hash and request.submission_hash != digest:
            raise ServiceError(
                409,
                "payload_conflict",
                "submission_hash не совпадает с сохранённой сдачей",
            )
        config_hash = request.config_hash or getattr(adapter, "config_hash", "")

        async with self._idempotency_lock:
            existing_key = self.store.jobs.get_idempotency(request.idempotency_key)
            if existing_key is not None:
                if not existing_key.matches(
                    submission_hash=digest,
                    analyzer_id=request.analyzer_id,
                    config_hash=config_hash,
                ):
                    raise ServiceError(
                        409,
                        "idempotency_conflict",
                        "idempotency_key уже использован с другим payload",
                    )
                try:
                    existing_job = self.store.jobs.require(existing_key.job_id)
                except KeyError as exc:
                    raise ServiceError(
                        500,
                        "idempotency_job_missing",
                        "Ключ идемпотентности ссылается на отсутствующую задачу",
                    ) from exc
                return existing_job, False

            attempt_no = self.store.jobs.next_attempt_no(
                digest,
                request.analyzer_id,
                config_hash,
            )
            timeout_seconds = (
                self.settings.default_timeout
                if request.timeout_seconds is None
                else request.timeout_seconds
            )
            job = build_analysis_job(
                submission,
                request.analyzer_id,
                config_hash=config_hash,
                timeout_seconds=timeout_seconds,
                attempt_no=attempt_no,
            )

            # job и idempotency_key создаются одной транзакцией: при падении
            # между ними не появляется задача без ключа или ключ без задачи.
            try:
                self.store.jobs.add_with_idempotency(
                    job,
                    key=request.idempotency_key,
                    submission_hash=digest,
                    analyzer_id=request.analyzer_id,
                    config_hash=config_hash,
                )
            except DuplicateIdempotencyKey:
                existing_key = self.store.jobs.get_idempotency(
                    request.idempotency_key
                )
                if existing_key is None:
                    raise
                if not existing_key.matches(
                    submission_hash=digest,
                    analyzer_id=request.analyzer_id,
                    config_hash=config_hash,
                ):
                    raise ServiceError(
                        409,
                        "idempotency_conflict",
                        "idempotency_key уже использован с другим payload",
                    )
                return self.store.jobs.require(existing_key.job_id), False

            try:
                result = await self.executor.enqueue_prebuilt(job)
            except Exception as exc:
                # Не оставляем job в queued, если постановка не удалась.
                if job.status is JobStatus.QUEUED:
                    job.mark_failed(
                        JobErrorCode.ANALYZER_ERROR,
                        f"Не удалось поставить задачу: {type(exc).__name__}: {exc}",
                    )
                    self.store.jobs.save(job)
                raise
            if not result.accepted:
                if result.error_code is JobErrorCode.QUEUE_FULL:
                    raise ServiceError(
                        429,
                        "queue_full",
                        result.message or "Очередь заполнена",
                        job=result.job,
                    )
                raise ServiceError(
                    409,
                    result.error_code.value if result.error_code else "rejected",
                    result.message or "Задача отклонена",
                    job=result.job,
                )
            return result.job, True

    # ------------------------------------------------------------------
    # Задачи и отчёты
    # ------------------------------------------------------------------
    def get_job(
        self,
        job_id: str,
        actor: Optional[Actor] = None,
    ) -> AnalysisJob:
        job = self.store.jobs.get(job_id)
        if job is None:
            raise ServiceError(404, "job_not_found", "Задача не найдена")
        submission = self.store.submissions.get(job.submission_id)
        if submission is None:
            raise ServiceError(404, "submission_not_found", "Сдача не найдена")
        self._authorize_read(actor, submission)
        return job

    def list_jobs(
        self,
        *,
        actor: Optional[Actor] = None,
        status: JobStatus | None = None,
        submission_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AnalysisJob]:
        jobs = self.store.jobs.list(
            status=status,
            submission_id=submission_id,
            limit=1_000_000,
        )
        if actor is not None and not actor.is_admin:
            readable: list[AnalysisJob] = []
            for job in jobs:
                submission = self.store.submissions.get(job.submission_id)
                if submission is not None and actor.can_read_submission(submission):
                    readable.append(job)
            jobs = readable
        return jobs[offset:offset + limit]

    def get_report(
        self,
        job_id: str,
        actor: Optional[Actor] = None,
    ) -> AnalysisReport:
        job = self.get_job(job_id, actor)
        if job.status is not JobStatus.SUCCEEDED or job.result is None:
            raise ServiceError(
                409,
                "report_not_ready",
                f"Отчёт недоступен: статус задачи {job.status.value}",
                job=job,
            )
        return job.result

    async def cancel(
        self,
        job_id: str,
        actor: Optional[Actor] = None,
    ) -> AnalysisJob:
        job = self.get_job(job_id, actor)
        submission = self.get_submission(job.submission_id, actor)
        self._authorize_start(actor, submission)
        if job.is_terminal:
            raise ServiceError(
                409,
                "job_already_finished",
                f"Задача уже завершена: {job.status.value}",
                job=job,
            )
        cancelled = await self.executor.cancel(job_id)
        if not cancelled:
            raise ServiceError(
                409,
                "cancel_rejected",
                "Исполнитель не смог отменить задачу",
                job=self.get_job(job_id, actor),
            )
        return self.get_job(job_id, actor)

    def analyzers(self) -> list[AnalyzerInfo]:
        result: list[AnalyzerInfo] = []
        for analyzer_id, adapter in self.executor.adapters.items():
            result.append(
                AnalyzerInfo(
                    analyzer_id=analyzer_id,
                    analyzer_version=getattr(adapter, "analyzer_version", ""),
                    config_hash=getattr(adapter, "config_hash", ""),
                    model_version=getattr(adapter, "model_version", None),
                    index_version=getattr(adapter, "index_version", None),
                    prompt_version=getattr(adapter, "prompt_version", None),
                )
            )
        return result

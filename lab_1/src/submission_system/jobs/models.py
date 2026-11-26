"""Модели управляемого исполнителя заданий ЛР3, этап А.

Здесь намеренно нет хранения в файлах и сетевого интерфейса: этап А отвечает
только за постановку задачи, ограничение ресурсов, тайм-аут, отмену и
сохранение результата в памяти. Постоянное хранение и пользовательский
интерфейс подключаются на следующем этапе.

Контракт статусов общий для всех адаптеров:

``queued -> running -> succeeded | failed | timed_out | cancelled``

``rejected`` — терминальный статус для задачи, которую не приняли в очередь.
Отказ в постановке не меняет и не удаляет исходную сдачу.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Optional

from ..models import AnalysisReport, Submission


def utc_now() -> str:
    """Текущее время в ISO 8601 с явной временной зоной UTC."""

    return datetime.now(timezone.utc).isoformat()


class JobStateError(ValueError):
    """Недопустимый переход статуса задачи."""


class JobErrorCode(str, Enum):
    """Машинно-читаемые причины отказа/ошибки задачи."""

    QUEUE_FULL = "queue_full"
    SUBMISSION_NOT_FOUND = "submission_not_found"
    SUBMISSION_CONFLICT = "submission_conflict"
    ANALYZER_NOT_FOUND = "analyzer_not_found"
    ANALYZER_ERROR = "analyzer_error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    MODEL_UNAVAILABLE = "model_unavailable"
    INVALID_MODEL_RESPONSE = "invalid_model_response"
    CONFIG_MISMATCH = "config_mismatch"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    INTERRUPTED = "interrupted"
    INCONSISTENT_STATE = "inconsistent_state"

    def __str__(self) -> str:
        return self.value


class JobStatus(str, Enum):
    """Общий контракт статусов задачи.

    Наследование от ``str`` позволяет сравнивать статус со строкой
    (``job.status == "queued"``) и без дополнительных преобразований писать
    его в JSON через ``.value``.
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

    def __str__(self) -> str:
        return self.value

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_STATUSES

    @property
    def is_active(self) -> bool:
        return self in {JobStatus.QUEUED, JobStatus.RUNNING}


TERMINAL_STATUSES = frozenset(
    {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.TIMED_OUT,
        JobStatus.CANCELLED,
        JobStatus.REJECTED,
    }
)


@dataclass
class AnalysisJob:
    """Одна задача анализа.

    Поля ``submission_hash`` и ``config_hash`` фиксируют вход и конфигурацию,
    чтобы результат можно было сопоставить с конкретной версией данных и
    анализатора. ``attempt_no`` уже есть в модели, но повторные попытки
    появятся в ЛР4; на этапе А исполнитель не делает автоматических retry.
    """

    job_id: str
    submission_id: str
    submission_hash: str
    analyzer_id: str
    config_hash: str = ""
    status: JobStatus = JobStatus.QUEUED
    attempt_no: int = 1
    timeout_seconds: float = 30.0
    created_at: str = field(default_factory=utc_now)
    queued_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    updated_at: Optional[str] = None
    error_code: Optional[JobErrorCode] = None
    error_message: Optional[str] = None
    result: Optional[AnalysisReport] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.attempt_no < 1:
            raise ValueError("attempt_no должен быть >= 1")
        if self.timeout_seconds < 0:
            raise ValueError("timeout_seconds не может быть отрицательным")
        if not isinstance(self.status, JobStatus):
            self.status = JobStatus(str(self.status))
        if self.error_code is not None and not isinstance(self.error_code, JobErrorCode):
            self.error_code = JobErrorCode(str(self.error_code))
        if self.queued_at is None:
            self.queued_at = self.created_at
        if self.updated_at is None:
            self.updated_at = self.created_at

    # ------------------------------------------------------------------
    # Переходы статусов. Явные методы не дают случайно перевести задачу
    # из терминального состояния в рабочее.
    # ------------------------------------------------------------------
    def mark_running(self) -> None:
        if self.status is not JobStatus.QUEUED:
            raise JobStateError(
                f"Задача {self.job_id} не может стартовать из статуса {self.status.value}"
            )
        self.status = JobStatus.RUNNING
        self.started_at = utc_now()
        self.updated_at = self.started_at

    def mark_succeeded(self, report: AnalysisReport) -> None:
        if self.status is not JobStatus.RUNNING:
            raise JobStateError(
                f"Успех допустим только из running, сейчас {self.status.value}"
            )
        self.status = JobStatus.SUCCEEDED
        self.result = report
        self.finished_at = utc_now()
        self.updated_at = self.finished_at

    def mark_failed(
        self,
        error_code: JobErrorCode,
        message: str,
    ) -> None:
        if self.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            raise JobStateError(
                f"Ошибка допустима только для активной задачи, сейчас {self.status.value}"
            )
        self.status = JobStatus.FAILED
        self.error_code = error_code
        self.error_message = message
        self.finished_at = utc_now()
        self.updated_at = self.finished_at

    def mark_timed_out(self, message: str) -> None:
        if self.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            raise JobStateError(
                f"Тайм-аут допустим только для активной задачи, сейчас {self.status.value}"
            )
        self.status = JobStatus.TIMED_OUT
        self.error_code = JobErrorCode.TIMEOUT
        self.error_message = message
        self.finished_at = utc_now()
        self.updated_at = self.finished_at

    def mark_cancelled(self, message: str = "Задача отменена") -> None:
        if self.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            raise JobStateError(
                f"Отмена допустима только для активной задачи, сейчас {self.status.value}"
            )
        self.status = JobStatus.CANCELLED
        self.error_code = JobErrorCode.CANCELLED
        self.error_message = message
        self.finished_at = utc_now()
        self.updated_at = self.finished_at

    def mark_rejected(self, error_code: JobErrorCode, message: str) -> None:
        if self.status is not JobStatus.QUEUED:
            raise JobStateError(
                f"Отказ допустим только для новой задачи, сейчас {self.status.value}"
            )
        self.status = JobStatus.REJECTED
        self.error_code = error_code
        self.error_message = message
        self.finished_at = utc_now()
        self.updated_at = self.finished_at

    # ------------------------------------------------------------------
    # Производные значения для отчёта и тестов.
    # ------------------------------------------------------------------
    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def duration_seconds(self) -> Optional[float]:
        """Длительность выполнения; ``None``, если задача ещё не стартовала."""

        if not self.started_at or not self.finished_at:
            return None
        return _seconds_between(self.started_at, self.finished_at)

    @property
    def queue_duration_seconds(self) -> Optional[float]:
        """Время от постановки до старта (или до текущего момента)."""

        if not self.queued_at:
            return None
        end = self.started_at or self.finished_at
        if not end:
            return None
        return _seconds_between(self.queued_at, end)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["error_code"] = (
            self.error_code.value if self.error_code is not None else None
        )
        payload["result"] = (
            _report_to_dict(self.result) if self.result is not None else None
        )
        return payload


@dataclass(frozen=True)
class SubmitResult:
    """Результат постановки задачи.

    ``accepted == False`` означает явный отказ: задача не попала в очередь и
    не будет выполнена. При этом исходная сдача уже сохранена в репозитории.
    """

    accepted: bool
    job: AnalysisJob
    error_code: Optional[JobErrorCode] = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "job": self.job.to_dict(),
            "error_code": self.error_code.value if self.error_code else None,
            "message": self.message,
        }


# Совместимое короткое имя: в отчётах и тестах удобно писать JobSubmitResult.
JobSubmitResult = SubmitResult


class JobStore:
    """In-memory реестр задач; на этапе Б заменяется постоянным хранилищем."""

    def __init__(self) -> None:
        self._jobs: dict[str, AnalysisJob] = {}

    def add(self, job: AnalysisJob) -> None:
        if job.job_id in self._jobs:
            raise ValueError(f"Задача с job_id={job.job_id} уже существует")
        self._jobs[job.job_id] = job

    def save(self, job: AnalysisJob) -> None:
        """Вставляет или обновляет задачу; используется исполнителем."""

        self._jobs[job.job_id] = job

    def get(self, job_id: str) -> Optional[AnalysisJob]:
        return self._jobs.get(job_id)

    def require(self, job_id: str) -> AnalysisJob:
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"Задача не найдена: {job_id}")
        return job

    def list(self, status: Optional[JobStatus] = None) -> list[AnalysisJob]:
        jobs = list(self._jobs.values())
        if status is not None:
            jobs = [job for job in jobs if job.status is status]
        return sorted(jobs, key=lambda item: item.created_at)

    def values(self) -> Iterable[AnalysisJob]:
        return self._jobs.values()

    def __len__(self) -> int:
        return len(self._jobs)

    def __contains__(self, job_id: object) -> bool:
        return job_id in self._jobs


def build_analysis_job(
    submission: Submission,
    analyzer_id: str,
    *,
    config_hash: str = "",
    timeout_seconds: float = 30.0,
    attempt_no: int = 1,
    job_id: Optional[str] = None,
    submission_hash: Optional[str] = None,
) -> AnalysisJob:
    """Создаёт описание задачи, но не ставит её в очередь.

    Это отделяет постановку от выполнения: ``build_analysis_job`` можно
    использовать для сохранения намерения, а ``AnalysisExecutor.submit`` —
    для принятия решения о постановке.
    """

    content = (submission.code or "") + (submission.text or "")
    digest = submission_hash or hashlib.sha256(content.encode("utf-8")).hexdigest()
    return AnalysisJob(
        job_id=job_id or f"job-{uuid.uuid4().hex}",
        submission_id=submission.submission_id,
        submission_hash=digest,
        analyzer_id=analyzer_id,
        config_hash=config_hash,
        timeout_seconds=timeout_seconds,
        attempt_no=attempt_no,
    )


def _seconds_between(start: str, end: str) -> Optional[float]:
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except (TypeError, ValueError):
        return None
    return (end_dt - start_dt).total_seconds()


def _report_to_dict(report: AnalysisReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["evidence"] = [dict(item) for item in payload.get("evidence", [])]
    return payload

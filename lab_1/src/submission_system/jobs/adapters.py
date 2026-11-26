"""Адаптеры анализаторов для исполнителя ЛР3.

Адаптер — это тонкая граница между управлением задачей и конкретной моделью.
Исполнителю не важно, синхронный это анализатор из ЛР1, управляемая заглушка
или настоящий модельный адаптер: он видит только ``analyze`` и коды ошибок.

Синхронный анализатор ЛР1 запускается через переданный ``executor``
(``ThreadPoolExecutor`` или ``ProcessPoolExecutor``), поэтому CPU-работа не
блокирует event loop сервисной траектории.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol, runtime_checkable

from ..contracts import AnalysisContext
from ..models import AnalysisReport, Assignment, Submission
from .models import AnalysisJob, JobErrorCode


class AdapterError(Exception):
    """Ошибка адаптера с машинно-читаемым кодом."""

    code: JobErrorCode = JobErrorCode.ANALYZER_ERROR

    def __init__(self, message: str, *, code: Optional[JobErrorCode] = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class AdapterUnavailable(AdapterError):
    """Модель/внешний сервис недоступен; это не тайм-аут и не плохой ответ."""

    code = JobErrorCode.MODEL_UNAVAILABLE


class InvalidAdapterResponse(AdapterError):
    """Адаптер вернул ответ, который нельзя использовать."""

    code = JobErrorCode.INVALID_MODEL_RESPONSE


class AdapterTimeout(AdapterError):
    """Адаптер сам сообщил об истечении времени."""

    code = JobErrorCode.TIMEOUT


@runtime_checkable
class AnalysisAdapter(Protocol):
    """Контракт адаптера для исполнителя."""

    analyzer_id: str
    config_hash: str

    async def analyze(
        self,
        submission: Submission,
        job: AnalysisJob,
        *,
        executor: Any = None,
    ) -> AnalysisReport: ...


@dataclass
class SyncAnalyzerAdapter:
    """Адаптирует синхронный ``Analyzer`` из ЛР1 к асинхронному исполнителю."""

    analyzer: Any
    assignment: Assignment
    config_hash: str = ""
    analyzer_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.analyzer_id = str(self.analyzer.analyzer_id)
        if not self.config_hash:
            payload = (
                f"{self.analyzer.analyzer_id}:{self.analyzer.analyzer_version}:"
                f"{self.assignment.assignment_id}:{self.assignment.allowed_template}"
            )
            self.config_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _analyze_sync(
        self,
        submission: Submission,
        context: AnalysisContext,
    ) -> AnalysisReport:
        return self.analyzer.analyze(submission, context)

    async def analyze(
        self,
        submission: Submission,
        job: AnalysisJob,
        *,
        executor: Any = None,
    ) -> AnalysisReport:
        context = AnalysisContext(
            assignment=self.assignment,
            config_hash=job.config_hash or self.config_hash,
        )
        loop = asyncio.get_running_loop()
        if executor is not None:
            return await loop.run_in_executor(
                executor,
                self._analyze_sync,
                submission,
                context,
            )
        # Fallback для автономного использования адаптера в тестах.
        return await asyncio.to_thread(self._analyze_sync, submission, context)


@dataclass
class FunctionAdapter:
    """Адаптер вокруг обычной или async-функции; удобен для тестов."""

    analyzer_id: str
    func: Callable[[Submission, AnalysisJob], AnalysisReport | Awaitable[AnalysisReport]]
    config_hash: str = ""

    async def analyze(
        self,
        submission: Submission,
        job: AnalysisJob,
        *,
        executor: Any = None,
    ) -> AnalysisReport:
        result = self.func(submission, job)
        if inspect.isawaitable(result):
            return await result
        return result


@dataclass
class SlowAdapter:
    """Управляемая заглушка для тестов тайм-аута, ошибок и отмены.

    ``mode``:
    * ``success`` — вернуть детерминированный отчёт;
    * ``error`` — поднять ``AdapterError`` (код ``analyzer_error``);
    * ``unavailable`` — поднять ``AdapterUnavailable``;
    * ``invalid`` — поднять ``InvalidAdapterResponse``;
    * ``invalid_report`` — вернуть score вне ``[0, 1]`` для проверки валидации;
    * ``hang`` — ждать бесконечно (до отмены/тайм-аута);
    * ``cancelled`` — поднять ``CancelledError`` без перехвата.
    """

    delay: float = 0.0
    mode: str = "success"
    analyzer_id: str = "slow_adapter"
    config_hash: str = "slow-adapter"
    score: float = 0.5
    message: str = "управляемая ошибка адаптера"

    active: int = field(default=0, init=False)
    max_active: int = field(default=0, init=False)
    calls: int = field(default=0, init=False)

    async def analyze(
        self,
        submission: Submission,
        job: AnalysisJob,
        *,
        executor: Any = None,
    ) -> AnalysisReport:
        self.active += 1
        self.calls += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.mode == "hang":
                await asyncio.Event().wait()
            if self.delay > 0:
                await asyncio.sleep(self.delay)
            if self.mode == "error":
                raise AdapterError(self.message)
            if self.mode == "unavailable":
                raise AdapterUnavailable(self.message)
            if self.mode == "invalid":
                raise InvalidAdapterResponse(self.message)
            if self.mode == "cancelled":
                raise asyncio.CancelledError("адаптер отменён")
            if self.mode == "invalid_report":
                return AnalysisReport(
                    analyzer_id=self.analyzer_id,
                    analyzer_version="1.0.0",
                    score=1.5,
                    rubric_feedback="невалидный score для проверки",
                )
            return AnalysisReport(
                analyzer_id=self.analyzer_id,
                analyzer_version="1.0.0",
                score=self.score,
                rubric_feedback=f"Задача {job.job_id} обработана заглушкой",
            )
        finally:
            self.active -= 1


class FailingAdapter(SlowAdapter):
    """Короткий alias для адаптера, который всегда возвращает ошибку."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "error")
        super().__init__(**kwargs)


class HangingAdapter(SlowAdapter):
    """Короткий alias для адаптера, который зависает до отмены/тайм-аута."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "hang")
        super().__init__(**kwargs)


@dataclass
class CpuBoundAnalyzer:
    """Детерминированный CPU-анализатор для проверки и бенчмарка.

    Он не претендует на содержательный анализ: важен только предсказуемый
    объём CPU-работы, чтобы измерить, блокирует ли синхронный вызов event loop.
    """

    iterations: int = 500_000
    analyzer_id: str = "cpu_bound_analyzer"
    analyzer_version: str = "1.0.0"

    def analyze(self, submission: Submission, context: AnalysisContext) -> AnalysisReport:
        total = 0
        for index in range(self.iterations):
            total = (total + index * index) % 1_000_003
        return AnalysisReport(
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            score=0.0,
            rubric_feedback=f"CPU-проверка завершена, контроль={total}",
        )

"""Ограниченный модельный адаптер для варианта AI и MLOps.

Здесь зафиксированы три вещи, которые не должны зависеть от конкретной модели:

* конкурентные обращения к модели ограничены семафором;
* размер пакета для batch-обработки — часть конфигурации, а не случайное
  значение на каждом вызове;
* тайм-аут, недоступность и невалидный ответ имеют разные коды.

Замена ``StubModelAdapter`` на настоящий адаптер не меняет исполнитель:
исполнитель работает с ``AnalysisAdapter``, а не с конкретным классом модели.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    Protocol,
    Sequence,
    runtime_checkable,
)

from ..models import AnalysisReport, Submission
from .adapters import AdapterError, AdapterTimeout, AdapterUnavailable, InvalidAdapterResponse
from .models import AnalysisJob, JobErrorCode


@runtime_checkable
class ModelAdapter(Protocol):
    """Контракт модели: один score или пакет scores в диапазоне [0, 1]."""

    model_version: str

    async def score(self, submission: Submission) -> float: ...

    async def score_batch(self, submissions: Sequence[Submission]) -> Sequence[float]: ...


class ModelTimeoutError(AdapterTimeout):
    """Модель не ответила за отведённое время."""


class ModelUnavailableError(AdapterUnavailable):
    """Модель/сервис недоступен."""


class InvalidModelResponseError(InvalidAdapterResponse):
    """Модель вернула NaN, бесконечность, не число или score вне [0, 1]."""


@dataclass(frozen=True)
class BatchConfig:
    """Конфигурация пакетной обработки и ограничения конкурентности."""

    batch_size: int = 8
    concurrency_limit: int = 2
    call_timeout: float = 30.0

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size должен быть >= 1")
        if self.concurrency_limit < 1:
            raise ValueError("concurrency_limit должен быть >= 1")
        if self.call_timeout < 0:
            raise ValueError("call_timeout не может быть отрицательным")


class StubModelAdapter:
    """Детерминированная модель для демонстрации и тестов.

    ``mode``: ``ok``, ``unavailable``, ``invalid``, ``hang``, ``error``.
    """

    model_version = "stub-model-1.0.0"

    def __init__(
        self,
        *,
        delay: float = 0.0,
        mode: str = "ok",
        message: str = "управляемая ошибка модели",
    ) -> None:
        self.delay = delay
        self.mode = mode
        self.message = message
        self.active_calls = 0
        self.max_active_calls = 0
        self.batch_sizes: list[int] = []

    async def _simulate(self) -> None:
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.mode == "hang":
                await asyncio.Event().wait()
            if self.delay > 0:
                await asyncio.sleep(self.delay)
            if self.mode == "unavailable":
                raise ModelUnavailableError(self.message)
            if self.mode == "error":
                raise AdapterError(self.message)
        finally:
            self.active_calls -= 1

    async def score(self, submission: Submission) -> float:
        await self._simulate()
        if self.mode == "invalid":
            return float("nan")
        content = (submission.code or "") + (submission.text or "")
        return min(len(content) / 100.0, 1.0)

    async def score_batch(self, submissions: Sequence[Submission]) -> Sequence[float]:
        self.batch_sizes.append(len(submissions))
        await self._simulate()
        if self.mode == "invalid":
            return [float("nan")] * len(submissions)
        return [
            min(len((item.code or "") + (item.text or "")) / 100.0, 1.0)
            for item in submissions
        ]


@dataclass
class GuardedModelAdapter:
    """Обёртка над моделью с ограничением конкурентности и batch-конфигурацией."""

    adapter: ModelAdapter
    config: BatchConfig = field(default_factory=BatchConfig)
    analyzer_id: str = "guarded_model_adapter"
    analyzer_version: str = "1.0.0"
    index_version: str = ""
    prompt_version: str = ""
    mlflow_run_id: str = ""
    reported_model_version: str = ""

    active_calls: int = field(default=0, init=False)
    max_active_calls: int = field(default=0, init=False)
    config_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.config.concurrency_limit)
        payload = (
            f"{self.analyzer_id}:{self.model_version}:"
            f"batch={self.config.batch_size}:"
            f"concurrency={self.config.concurrency_limit}:"
            f"timeout={self.config.call_timeout}"
        )
        self.config_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def batch_size(self) -> int:
        return self.config.batch_size

    @property
    def model_version(self) -> str:
        return self.reported_model_version or str(
            getattr(self.adapter, "model_version", "unknown")
        )

    async def _guarded_call(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        async with self._semaphore:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            try:
                try:
                    return await asyncio.wait_for(
                        operation(),
                        timeout=self.config.call_timeout,
                    )
                except asyncio.TimeoutError as exc:
                    raise ModelTimeoutError(
                        f"Модель не ответила за {self.config.call_timeout}s"
                    ) from exc
                except asyncio.CancelledError:
                    # Отмена не является ошибкой модели и не должна превращаться
                    # в failed. Пробрасываем дальше без изменений.
                    raise
                except (ModelTimeoutError, ModelUnavailableError, InvalidModelResponseError):
                    raise
                except AdapterError:
                    raise
                except Exception as exc:
                    raise ModelUnavailableError(
                        f"Модель недоступна: {type(exc).__name__}: {exc}"
                    ) from exc
            finally:
                self.active_calls -= 1

    async def score(self, submission: Submission) -> float:
        raw = await self._guarded_call(lambda: self.adapter.score(submission))
        return _validate_score(raw, index=0)

    async def score_batch(self, submissions: Sequence[Submission]) -> list[float]:
        """Обрабатывает вход пакетами фиксированного размера."""

        items = list(submissions)
        chunks = [
            items[start:start + self.config.batch_size]
            for start in range(0, len(items), self.config.batch_size)
        ]

        async def run(chunk: Sequence[Submission]) -> Sequence[float]:
            return await self._guarded_call(lambda: self.adapter.score_batch(chunk))

        raw_chunks = await asyncio.gather(
            *(run(chunk) for chunk in chunks),
            return_exceptions=True,
        )
        for item in raw_chunks:
            if isinstance(item, BaseException):
                raise item
        scores: list[float] = []
        for chunk_index, raw_chunk in enumerate(raw_chunks):
            if not isinstance(raw_chunk, Sequence) or isinstance(raw_chunk, (str, bytes)):
                raise InvalidModelResponseError(
                    f"Пакет #{chunk_index} вернул не последовательность scores"
                )
            if len(raw_chunk) != len(chunks[chunk_index]):
                raise InvalidModelResponseError(
                    f"Пакет #{chunk_index}: ожидалось {len(chunks[chunk_index])} "
                    f"score, получено {len(raw_chunk)}"
                )
            scores.extend(
                _validate_score(value, index=chunk_index * self.config.batch_size + offset)
                for offset, value in enumerate(raw_chunk)
            )
        return scores

    async def analyze(
        self,
        submission: Submission,
        job: AnalysisJob,
        *,
        executor: Any = None,
    ) -> AnalysisReport:
        score = await self.score(submission)
        return AnalysisReport(
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            score=score,
            rubric_feedback=(
                f"Модель {getattr(self.adapter, 'model_version', 'unknown')} "
                f"вернула score={score:.3f}"
            ),
            model_version=self.model_version,
            index_version=self.index_version or None,
            prompt_version=self.prompt_version or None,
            mlflow_run_id=self.mlflow_run_id or None,
            config_hash=self.config_hash,
        )


def _validate_score(value: Any, *, index: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidModelResponseError(
            f"score #{index} не является числом: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise InvalidModelResponseError(f"score #{index} не конечен: {number!r}")
    if not 0.0 <= number <= 1.0:
        raise InvalidModelResponseError(
            f"score #{index} вне диапазона [0, 1]: {number!r}"
        )
    return number


def iter_batches(
    submissions: Iterable[Submission],
    batch_size: int,
) -> Iterable[list[Submission]]:
    """Делит вход на пакеты фиксированного размера (удобно для отчёта)."""

    if batch_size < 1:
        raise ValueError("batch_size должен быть >= 1")
    batch: list[Submission] = []
    for submission in submissions:
        batch.append(submission)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def model_error_code(exc: BaseException) -> JobErrorCode:
    """Возвращает код ошибки модели для отчёта/исполнителя."""

    if isinstance(exc, AdapterError):
        return exc.code
    if isinstance(exc, asyncio.TimeoutError):
        return JobErrorCode.TIMEOUT
    return JobErrorCode.ANALYZER_ERROR

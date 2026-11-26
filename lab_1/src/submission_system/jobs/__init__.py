"""Управляемый исполнитель заданий ЛР3, этап А.

Публичный API:

``AnalysisJob``, ``JobStatus``, ``JobErrorCode`` — контракт задачи;
``AnalysisExecutor``, ``ExecutorConfig`` — ограниченный исполнитель;
``SyncAnalyzerAdapter`` — мост к синхронным анализаторам ЛР1;
``SlowAdapter`` / ``FailingAdapter`` / ``HangingAdapter`` — управляемые
заглушки для тайм-аутов, ошибок и отмены;
``GuardedModelAdapter``, ``BatchConfig``, ``StubModelAdapter`` — модельный
адаптер с ограничением конкурентности и batch-конфигурацией.
"""

from .adapters import (
    AdapterError,
    AdapterTimeout,
    AdapterUnavailable,
    AnalysisAdapter,
    CpuBoundAnalyzer,
    FailingAdapter,
    FunctionAdapter,
    HangingAdapter,
    InvalidAdapterResponse,
    SlowAdapter,
    SyncAnalyzerAdapter,
    UnavailableAnalysisAdapter,
)
from .executor import AnalysisExecutor, ExecutorConfig
from .model_adapter import (
    BatchConfig,
    GuardedModelAdapter,
    InvalidModelResponseError,
    ModelAdapter,
    ModelTimeoutError,
    ModelUnavailableError,
    StubModelAdapter,
    iter_batches,
    model_error_code,
)
from .models import (
    AnalysisJob,
    JobErrorCode,
    JobStateError,
    JobStatus,
    JobStore,
    JobSubmitResult,
    SubmitResult,
    build_analysis_job,
    utc_now,
)

__all__ = [
    "AdapterError",
    "AdapterTimeout",
    "AdapterUnavailable",
    "AnalysisAdapter",
    "AnalysisExecutor",
    "AnalysisJob",
    "BatchConfig",
    "CpuBoundAnalyzer",
    "ExecutorConfig",
    "FailingAdapter",
    "FunctionAdapter",
    "GuardedModelAdapter",
    "HangingAdapter",
    "InvalidAdapterResponse",
    "InvalidModelResponseError",
    "JobErrorCode",
    "JobStateError",
    "JobStatus",
    "JobStore",
    "JobSubmitResult",
    "ModelAdapter",
    "ModelTimeoutError",
    "ModelUnavailableError",
    "SlowAdapter",
    "StubModelAdapter",
    "SubmitResult",
    "SyncAnalyzerAdapter",
    "UnavailableAnalysisAdapter",
    "build_analysis_job",
    "iter_batches",
    "model_error_code",
    "utc_now",
]

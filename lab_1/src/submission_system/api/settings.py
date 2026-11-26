"""Настройки FastAPI-сервиса ЛР3, этап Б."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class Settings:
    """Конфигурация сервиса.

    ``api_key`` — единственный секрет доступа к API. ``provider_api_key``
    читается из окружения только для будущего внешнего адаптера и никогда не
    попадает в ответы API.
    """

    db_path: str = "artifacts/lab3/analysis.db"
    api_key: str = "dev-api-key"
    student_keys: Mapping[str, str] = field(default_factory=dict)
    teacher_keys: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    executor_concurrency: int = 2
    queue_limit: int = 16
    default_timeout: float = 30.0
    execution_model: str = "thread"
    batch_size: int = 4
    model_concurrency: int = 2
    model_call_timeout: float = 10.0
    model_delay: float = 0.0
    model_mode: str = "stub"  # stub | unavailable
    model_version: str = "stub-model-1.0.0"
    index_version: str = "tfidf-lr2"
    prompt_version: str = "retrieval-feedback-1.0.0"
    mlflow_run_id: str = ""
    provider_api_key: str = ""

    def __post_init__(self) -> None:
        if self.executor_concurrency < 1:
            raise ValueError("executor_concurrency должен быть >= 1")
        if self.queue_limit < 0:
            raise ValueError("queue_limit не может быть отрицательным")
        if self.batch_size < 1:
            raise ValueError("batch_size должен быть >= 1")
        if self.model_concurrency < 1:
            raise ValueError("model_concurrency должен быть >= 1")
        if self.model_delay < 0:
            raise ValueError("model_delay не может быть отрицательным")
        if self.execution_model not in {"thread", "process"}:
            raise ValueError("execution_model должен быть 'thread' или 'process'")
        if self.model_mode not in {"stub", "unavailable"}:
            raise ValueError("model_mode должен быть 'stub' или 'unavailable'")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=os.getenv("ANALYZER_DB_PATH", "artifacts/lab3/analysis.db"),
            api_key=os.getenv("ANALYZER_API_KEY", "dev-api-key"),
            student_keys=_json_mapping(os.getenv("ANALYZER_STUDENT_KEYS", "{}")),
            teacher_keys=_json_teacher_mapping(
                os.getenv("ANALYZER_TEACHER_KEYS", "{}")
            ),
            executor_concurrency=int(os.getenv("ANALYZER_EXECUTOR_CONCURRENCY", "2")),
            queue_limit=int(os.getenv("ANALYZER_QUEUE_LIMIT", "16")),
            default_timeout=float(os.getenv("ANALYZER_DEFAULT_TIMEOUT", "30")),
            execution_model=os.getenv("ANALYZER_EXECUTION_MODEL", "thread"),
            batch_size=int(os.getenv("ANALYZER_BATCH_SIZE", "4")),
            model_concurrency=int(os.getenv("ANALYZER_MODEL_CONCURRENCY", "2")),
            model_call_timeout=float(os.getenv("ANALYZER_MODEL_TIMEOUT", "10")),
            model_delay=float(os.getenv("ANALYZER_MODEL_DELAY", "0")),
            model_mode=os.getenv("ANALYZER_MODEL_MODE", "stub"),
            model_version=os.getenv("ANALYZER_MODEL_VERSION", "stub-model-1.0.0"),
            index_version=os.getenv("ANALYZER_INDEX_VERSION", "tfidf-lr2"),
            prompt_version=os.getenv(
                "ANALYZER_PROMPT_VERSION", "retrieval-feedback-1.0.0"
            ),
            mlflow_run_id=os.getenv("ANALYZER_MLFLOW_RUN_ID", ""),
            provider_api_key=os.getenv("MODEL_PROVIDER_API_KEY", ""),
        )


def _json_mapping(raw: str) -> dict[str, str]:
    value = json.loads(raw or "{}")
    if not isinstance(value, dict):
        raise ValueError("Ожидался JSON-объект ключей")
    return {str(key): str(item) for key, item in value.items()}


def _json_teacher_mapping(raw: str) -> dict[str, tuple[str, ...]]:
    value = json.loads(raw or "{}")
    if not isinstance(value, dict):
        raise ValueError("Ожидался JSON-объект teacher-ключей")
    result: dict[str, tuple[str, ...]] = {}
    for key, assignments in value.items():
        if isinstance(assignments, str):
            assignments = [assignments]
        result[str(key)] = tuple(str(item) for item in assignments)
    return result

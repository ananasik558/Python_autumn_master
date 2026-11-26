"""Тесты модельного адаптера ЛР3, этап А."""

from __future__ import annotations

import asyncio

import pytest

from submission_system.jobs import (
    AnalysisExecutor,
    BatchConfig,
    ExecutorConfig,
    GuardedModelAdapter,
    InvalidModelResponseError,
    JobErrorCode,
    JobStatus,
    ModelTimeoutError,
    ModelUnavailableError,
    StubModelAdapter,
    iter_batches,
)
from submission_system.models import Submission


def _submission(index: int, length: int = 150) -> Submission:
    return Submission(
        student_id=f"student-{index}",
        submission_id=f"sub-{index}",
        timestamp="2026-10-20T12:00:00+03:00",
        code="x" * length,
        text="",
    )


class FakeRealModelAdapter:
    """Имитация настоящего адаптера: другой класс, тот же контракт."""

    model_version = "real-model-0.9.0"

    async def score(self, submission: Submission) -> float:
        return 0.25

    async def score_batch(self, submissions):
        return [0.25 for _ in submissions]


def test_batch_size_is_taken_from_config():
    async def scenario():
        stub = StubModelAdapter()
        adapter = GuardedModelAdapter(
            stub,
            BatchConfig(batch_size=3, concurrency_limit=2, call_timeout=1),
        )
        scores = await adapter.score_batch([_submission(index) for index in range(7)])

        assert len(scores) == 7
        assert sorted(stub.batch_sizes) == [1, 3, 3]
        assert all(0.0 <= score <= 1.0 for score in scores)

    asyncio.run(scenario())


def test_model_concurrency_limit_is_respected():
    async def scenario():
        stub = StubModelAdapter(delay=0.05)
        adapter = GuardedModelAdapter(
            stub,
            BatchConfig(batch_size=2, concurrency_limit=2, call_timeout=2),
        )
        await asyncio.gather(
            *(adapter.score(_submission(index)) for index in range(6))
        )

        assert adapter.max_active_calls <= 2
        assert stub.max_active_calls <= 2

    asyncio.run(scenario())


def test_model_timeout_has_distinct_code():
    async def scenario():
        adapter = GuardedModelAdapter(
            StubModelAdapter(delay=1.0),
            BatchConfig(concurrency_limit=1, call_timeout=0.05),
        )
        with pytest.raises(ModelTimeoutError) as error:
            await adapter.score(_submission(1))
        assert error.value.code is JobErrorCode.TIMEOUT

    asyncio.run(scenario())


def test_unavailable_model_has_distinct_code():
    async def scenario():
        adapter = GuardedModelAdapter(
            StubModelAdapter(mode="unavailable"),
            BatchConfig(concurrency_limit=1, call_timeout=1),
        )
        with pytest.raises(ModelUnavailableError) as error:
            await adapter.score(_submission(1))
        assert error.value.code is JobErrorCode.MODEL_UNAVAILABLE

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["invalid"])
def test_invalid_model_response_has_distinct_code(mode):
    async def scenario():
        adapter = GuardedModelAdapter(
            StubModelAdapter(mode=mode),
            BatchConfig(concurrency_limit=1, call_timeout=1),
        )
        with pytest.raises(InvalidModelResponseError) as error:
            await adapter.score(_submission(1))
        assert error.value.code is JobErrorCode.INVALID_MODEL_RESPONSE

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mode", "expected_status", "expected_code"),
    [
        ("unavailable", JobStatus.FAILED, JobErrorCode.MODEL_UNAVAILABLE),
        ("invalid", JobStatus.FAILED, JobErrorCode.INVALID_MODEL_RESPONSE),
        ("error", JobStatus.FAILED, JobErrorCode.ANALYZER_ERROR),
    ],
)
def test_executor_maps_model_errors_to_job_status(mode, expected_status, expected_code):
    async def scenario():
        model = GuardedModelAdapter(
            StubModelAdapter(mode=mode),
            BatchConfig(concurrency_limit=1, call_timeout=1),
        )
        executor = AnalysisExecutor(
            {"model": model},
            ExecutorConfig(concurrency_limit=1, queue_limit=1),
        )
        result = await executor.submit(_submission(1), "model")
        job = await executor.wait(result.job.job_id, timeout=2)

        assert job.status is expected_status
        assert job.error_code is expected_code
        await executor.shutdown()

    asyncio.run(scenario())


def test_model_timeout_maps_to_timed_out_job():
    async def scenario():
        model = GuardedModelAdapter(
            StubModelAdapter(delay=1.0),
            BatchConfig(concurrency_limit=1, call_timeout=0.05),
        )
        executor = AnalysisExecutor(
            {"model": model},
            ExecutorConfig(concurrency_limit=1, queue_limit=1),
        )
        result = await executor.submit(_submission(1), "model")
        job = await executor.wait(result.job.job_id, timeout=2)

        assert job.status is JobStatus.TIMED_OUT
        assert job.error_code is JobErrorCode.TIMEOUT
        await executor.shutdown()

    asyncio.run(scenario())


def test_stub_can_be_replaced_without_changing_job_management():
    async def scenario():
        config = ExecutorConfig(concurrency_limit=1, queue_limit=1)
        stub_executor = AnalysisExecutor(
            {"model": GuardedModelAdapter(StubModelAdapter())},
            config,
        )
        real_executor = AnalysisExecutor(
            {"model": GuardedModelAdapter(FakeRealModelAdapter())},
            config,
        )
        stub_result = await stub_executor.submit(_submission(1), "model")
        real_result = await real_executor.submit(_submission(2), "model")
        stub_job = await stub_executor.wait(stub_result.job.job_id, timeout=2)
        real_job = await real_executor.wait(real_result.job.job_id, timeout=2)

        assert stub_job.status is JobStatus.SUCCEEDED
        assert real_job.status is JobStatus.SUCCEEDED
        assert stub_job.result is not None
        assert real_job.result is not None
        assert stub_job.result.score != real_job.result.score
        await stub_executor.shutdown()
        await real_executor.shutdown()

    asyncio.run(scenario())


def test_batch_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        BatchConfig(batch_size=0)
    with pytest.raises(ValueError):
        BatchConfig(concurrency_limit=0)
    with pytest.raises(ValueError):
        BatchConfig(call_timeout=-1)


def test_iter_batches_is_fixed_size():
    submissions = [_submission(index) for index in range(5)]
    batches = list(iter_batches(submissions, batch_size=2))
    assert [len(batch) for batch in batches] == [2, 2, 1]

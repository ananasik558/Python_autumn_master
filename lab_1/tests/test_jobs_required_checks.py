"""Обязательные проверки этапа А ЛР3.

Каждый тест соответствует пункту задания: модель задачи, ограничения
очереди и конкурентности, отказ без потери сдачи, тайм-аут, отмена,
неблокирующий event loop, разные коды ошибок и замена модели.
"""

from __future__ import annotations

import asyncio
import time

from submission_system.jobs import (
    AnalysisExecutor,
    BatchConfig,
    CpuBoundAnalyzer,
    ExecutorConfig,
    GuardedModelAdapter,
    JobErrorCode,
    JobStatus,
    SlowAdapter,
    StubModelAdapter,
    SyncAnalyzerAdapter,
    build_analysis_job,
)
from submission_system.models import Assignment, Submission


def _submission(index: int) -> Submission:
    return Submission(
        student_id=f"required-{index}",
        submission_id=f"required-sub-{index}",
        timestamp="2026-10-20T12:00:00+03:00",
        code="print(1)\n",
        text="",
    )


async def _wait_running(executor: AnalysisExecutor, job_id: str) -> None:
    deadline = time.monotonic() + 2
    while executor.get_job(job_id).status is not JobStatus.RUNNING:
        if time.monotonic() > deadline:
            raise AssertionError("Задача не стартовала")
        await asyncio.sleep(0.005)


def test_required_job_model_has_contract_fields():
    job = build_analysis_job(
        _submission(0),
        "length_analyzer",
        config_hash="cfg",
        timeout_seconds=3,
    )
    assert job.job_id
    assert job.submission_hash
    assert job.status is JobStatus.QUEUED
    assert job.attempt_no == 1
    assert job.created_at and job.queued_at
    assert job.timeout_seconds == 3


def test_required_queue_and_concurrency_limits():
    async def scenario():
        adapter = SlowAdapter(delay=0.05)
        executor = AnalysisExecutor(
            {"slow": adapter},
            ExecutorConfig(concurrency_limit=2, queue_limit=3),
        )
        results = [
            await executor.submit(_submission(index), "slow")
            for index in range(5)
        ]
        assert all(result.accepted for result in results)
        jobs = await executor.wait_all(
            [result.job.job_id for result in results],
            timeout=5,
        )
        assert all(job.status is JobStatus.SUCCEEDED for job in jobs)
        assert adapter.max_active <= 2
        await executor.shutdown()

    asyncio.run(scenario())


def test_required_queue_overflow_is_explicit_and_keeps_submission():
    async def scenario():
        executor = AnalysisExecutor(
            {"slow": SlowAdapter(delay=0.2)},
            ExecutorConfig(concurrency_limit=1, queue_limit=1),
        )
        first = await executor.submit(_submission(1), "slow")
        second = await executor.submit(_submission(2), "slow")
        third = await executor.submit(_submission(3), "slow")

        assert first.accepted and second.accepted
        assert not third.accepted
        assert third.error_code is JobErrorCode.QUEUE_FULL
        assert third.job.status is JobStatus.REJECTED
        assert executor.repository.get("required-sub-3") is not None
        await executor.wait_all(
            [first.job.job_id, second.job.job_id],
            timeout=5,
        )
        await executor.shutdown()

    asyncio.run(scenario())


def test_required_timeout_is_not_failure():
    async def scenario():
        executor = AnalysisExecutor(
            {"slow": SlowAdapter(delay=5)},
            ExecutorConfig(concurrency_limit=1, queue_limit=1),
        )
        result = await executor.submit(_submission(1), "slow", timeout=0.05)
        job = await executor.wait(result.job.job_id, timeout=2)
        assert job.status is JobStatus.TIMED_OUT
        assert job.error_code is JobErrorCode.TIMEOUT
        await executor.shutdown()

    asyncio.run(scenario())


def test_required_cancel_before_and_during_execution():
    async def scenario():
        executor = AnalysisExecutor(
            {"slow": SlowAdapter(delay=5)},
            ExecutorConfig(concurrency_limit=1, queue_limit=2),
        )
        running = await executor.submit(_submission(1), "slow")
        queued = await executor.submit(_submission(2), "slow")

        assert await executor.cancel(queued.job.job_id)
        await _wait_running(executor, running.job.job_id)
        assert await executor.cancel(running.job.job_id)

        assert (await executor.wait(queued.job.job_id, 2)).status is JobStatus.CANCELLED
        assert (await executor.wait(running.job.job_id, 2)).status is JobStatus.CANCELLED
        await executor.shutdown()

    asyncio.run(scenario())


def test_required_cpu_analysis_does_not_block_event_loop():
    async def scenario():
        executor = AnalysisExecutor(
            {
                "cpu": SyncAnalyzerAdapter(
                    CpuBoundAnalyzer(iterations=2_500_000),
                    Assignment(assignment_id="hw"),
                )
            },
            ExecutorConfig(concurrency_limit=1, queue_limit=1),
        )
        result = await executor.submit(_submission(1), "cpu")
        ticks = 0
        while not executor.get_job(result.job.job_id).is_terminal:
            await asyncio.sleep(0.005)
            ticks += 1
        assert (await executor.wait(result.job.job_id, 2)).status is JobStatus.SUCCEEDED
        assert ticks >= 2
        await executor.shutdown()

    asyncio.run(scenario())


def test_required_error_codes_are_distinct():
    async def scenario():
        executor = AnalysisExecutor(
            {
                "unavailable": SlowAdapter(mode="unavailable"),
                "invalid": SlowAdapter(mode="invalid"),
                "cancelled": SlowAdapter(mode="cancelled"),
            },
            ExecutorConfig(concurrency_limit=3, queue_limit=3),
        )
        expected = {
            "unavailable": (JobStatus.FAILED, JobErrorCode.MODEL_UNAVAILABLE),
            "invalid": (JobStatus.FAILED, JobErrorCode.INVALID_MODEL_RESPONSE),
            "cancelled": (JobStatus.CANCELLED, JobErrorCode.CANCELLED),
        }
        for index, analyzer_id in enumerate(expected):
            result = await executor.submit(_submission(index), analyzer_id)
            job = await executor.wait(result.job.job_id, timeout=2)
            status, code = expected[analyzer_id]
            assert job.status is status
            assert job.error_code is code
        await executor.shutdown()

    asyncio.run(scenario())


def test_required_model_batch_and_concurrency_configuration():
    async def scenario():
        stub = StubModelAdapter(delay=0.02)
        model = GuardedModelAdapter(
            stub,
            BatchConfig(batch_size=2, concurrency_limit=2, call_timeout=2),
        )
        scores = await model.score_batch([_submission(index) for index in range(5)])
        assert len(scores) == 5
        assert sorted(stub.batch_sizes) == [1, 2, 2]
        assert model.max_active_calls <= 2
        assert stub.max_active_calls <= 2

    asyncio.run(scenario())


def test_required_stub_replacement_keeps_executor_contract():
    class RealModel:
        model_version = "real-1"

        async def score(self, submission):
            return 0.4

        async def score_batch(self, submissions):
            return [0.4 for _ in submissions]

    async def run_one(model) -> JobStatus:
        executor = AnalysisExecutor(
            {"model": GuardedModelAdapter(model)},
            ExecutorConfig(concurrency_limit=1, queue_limit=1),
        )
        result = await executor.submit(_submission(1), "model")
        job = await executor.wait(result.job.job_id, timeout=2)
        await executor.shutdown()
        return job.status

    assert asyncio.run(run_one(StubModelAdapter())) is JobStatus.SUCCEEDED
    assert asyncio.run(run_one(RealModel())) is JobStatus.SUCCEEDED

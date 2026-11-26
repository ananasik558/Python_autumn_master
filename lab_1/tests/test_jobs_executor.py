"""Интеграционные тесты ограниченного исполнителя ЛР3, этап А."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from submission_system.jobs import (
    AnalysisExecutor,
    CpuBoundAnalyzer,
    ExecutorConfig,
    JobErrorCode,
    JobStatus,
    SlowAdapter,
    SyncAnalyzerAdapter,
)
from submission_system.models import AnalysisReport, Assignment, Submission


def _submission(index: int, *, code: str = "print(1)\n") -> Submission:
    return Submission(
        student_id=f"student-{index}",
        submission_id=f"sub-{index}",
        timestamp="2026-10-20T12:00:00+03:00",
        code=code,
        text="",
    )


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Условие не выполнилось за отведённое время")
        await asyncio.sleep(0.005)


class SlowSyncAnalyzer:
    """Синхронный анализатор, который проверяет, что shutdown ждёт поток."""

    analyzer_id = "slow_sync_analyzer"
    analyzer_version = "1.0.0"

    def __init__(self, delay: float = 0.2) -> None:
        self.delay = delay
        self.started = threading.Event()
        self.finished = threading.Event()

    def analyze(self, submission, context) -> AnalysisReport:
        self.started.set()
        time.sleep(self.delay)
        self.finished.set()
        return AnalysisReport(
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            score=0.5,
        )


def test_concurrency_limit_is_respected():
    async def scenario():
        adapter = SlowAdapter(delay=0.05)
        executor = AnalysisExecutor(
            {"slow": adapter},
            ExecutorConfig(concurrency_limit=2, queue_limit=5),
        )
        results = [
            await executor.submit(_submission(index), "slow")
            for index in range(6)
        ]
        assert all(result.accepted for result in results)
        jobs = await executor.wait_all(
            [result.job.job_id for result in results],
            timeout=5,
        )
        assert all(job.status is JobStatus.SUCCEEDED for job in jobs)
        assert adapter.max_active <= 2
        assert executor.stats()["running"] == 0
        await executor.shutdown()

    asyncio.run(scenario())


def test_queue_limit_rejects_explicitly_and_keeps_submission():
    async def scenario():
        adapter = SlowAdapter(delay=0.2)
        executor = AnalysisExecutor(
            {"slow": adapter},
            ExecutorConfig(concurrency_limit=1, queue_limit=1),
        )
        first = await executor.submit(_submission(1), "slow")
        second = await executor.submit(_submission(2), "slow")
        third = await executor.submit(_submission(3), "slow")

        assert first.accepted is True
        assert second.accepted is True
        assert third.accepted is False
        assert third.error_code is JobErrorCode.QUEUE_FULL
        assert third.job.status is JobStatus.REJECTED
        assert executor.repository.get("sub-3") is not None
        assert executor.repository.get("sub-3").code == "print(1)\n"

        await executor.wait_all(
            [first.job.job_id, second.job.job_id],
            timeout=5,
        )
        await executor.shutdown()

    asyncio.run(scenario())


def test_queue_limit_zero_allows_only_immediate_capacity():
    async def scenario():
        executor = AnalysisExecutor(
            {"slow": SlowAdapter(delay=0.05)},
            ExecutorConfig(concurrency_limit=2, queue_limit=0),
        )
        first = await executor.submit(_submission(1), "slow")
        second = await executor.submit(_submission(2), "slow")
        third = await executor.submit(_submission(3), "slow")

        assert first.accepted is True
        assert second.accepted is True
        assert third.accepted is False
        assert third.error_code is JobErrorCode.QUEUE_FULL
        await executor.wait_all(
            [first.job.job_id, second.job.job_id],
            timeout=5,
        )
        await executor.shutdown()

    asyncio.run(scenario())


def test_cancel_before_start():
    async def scenario():
        adapter = SlowAdapter(delay=0.15)
        executor = AnalysisExecutor(
            {"slow": adapter},
            ExecutorConfig(concurrency_limit=1, queue_limit=2),
        )
        running = await executor.submit(_submission(1), "slow")
        queued = await executor.submit(_submission(2), "slow")

        assert await executor.cancel(queued.job.job_id) is True
        assert queued.job.status is JobStatus.CANCELLED
        assert queued.job.error_code is JobErrorCode.CANCELLED

        running_job = await executor.wait(running.job.job_id, timeout=5)
        cancelled_job = await executor.wait(queued.job.job_id, timeout=1)
        assert running_job.status is JobStatus.SUCCEEDED
        assert cancelled_job.status is JobStatus.CANCELLED
        await executor.shutdown()

    asyncio.run(scenario())


def test_cancel_during_execution():
    async def scenario():
        adapter = SlowAdapter(delay=5.0)
        executor = AnalysisExecutor(
            {"slow": adapter},
            ExecutorConfig(concurrency_limit=1, queue_limit=1),
        )
        result = await executor.submit(_submission(1), "slow", timeout=5.0)
        await _wait_until(
            lambda: executor.get_job(result.job.job_id).status is JobStatus.RUNNING
        )

        assert await executor.cancel(result.job.job_id) is True
        job = await executor.wait(result.job.job_id, timeout=2)
        assert job.status is JobStatus.CANCELLED
        assert job.error_code is JobErrorCode.CANCELLED
        assert job.result is None
        await executor.shutdown()

    asyncio.run(scenario())


def test_timeout_has_its_own_terminal_status():
    async def scenario():
        adapter = SlowAdapter(delay=5.0)
        executor = AnalysisExecutor(
            {"slow": adapter},
            ExecutorConfig(concurrency_limit=1, queue_limit=1),
        )
        result = await executor.submit(_submission(1), "slow", timeout=0.05)
        job = await executor.wait(result.job.job_id, timeout=2)

        assert job.status is JobStatus.TIMED_OUT
        assert job.error_code is JobErrorCode.TIMEOUT
        assert job.result is None
        await executor.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mode", "expected_status", "expected_code"),
    [
        ("error", JobStatus.FAILED, JobErrorCode.ANALYZER_ERROR),
        ("unavailable", JobStatus.FAILED, JobErrorCode.MODEL_UNAVAILABLE),
        ("invalid", JobStatus.FAILED, JobErrorCode.INVALID_MODEL_RESPONSE),
        ("invalid_report", JobStatus.FAILED, JobErrorCode.ANALYZER_ERROR),
        ("cancelled", JobStatus.CANCELLED, JobErrorCode.CANCELLED),
    ],
)
def test_adapter_errors_are_distinct(mode, expected_status, expected_code):
    async def scenario():
        executor = AnalysisExecutor(
            {"adapter": SlowAdapter(mode=mode)},
            ExecutorConfig(concurrency_limit=1, queue_limit=1),
        )
        result = await executor.submit(_submission(1), "adapter", timeout=1)
        job = await executor.wait(result.job.job_id, timeout=2)

        assert job.status is expected_status
        assert job.error_code is expected_code
        assert job.result is None
        await executor.shutdown()

    asyncio.run(scenario())


def test_successful_result_is_saved_and_available():
    async def scenario():
        executor = AnalysisExecutor(
            {"slow": SlowAdapter(score=0.75)},
            ExecutorConfig(concurrency_limit=1, queue_limit=1),
        )
        result = await executor.submit(_submission(1), "slow")
        job = await executor.wait(result.job.job_id, timeout=2)

        assert job.status is JobStatus.SUCCEEDED
        assert executor.get_result(job.job_id) is not None
        assert executor.get_result(job.job_id).score == 0.75
        assert job.to_dict()["result"]["score"] == 0.75
        await executor.shutdown()

    asyncio.run(scenario())


def test_sync_cpu_analysis_does_not_block_event_loop():
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
        job = await executor.wait(result.job.job_id, timeout=2)

        assert job.status is JobStatus.SUCCEEDED
        assert ticks >= 2, "event loop не получал управления во время CPU-анализа"
        await executor.shutdown()

    asyncio.run(scenario())


def test_submission_conflict_does_not_overwrite_original():
    async def scenario():
        executor = AnalysisExecutor(
            {"slow": SlowAdapter()},
            ExecutorConfig(concurrency_limit=1, queue_limit=2),
        )
        first = await executor.submit(_submission(1, code="first"), "slow")
        await executor.wait(first.job.job_id, timeout=2)

        conflict = await executor.submit(_submission(1, code="second"), "slow")
        assert conflict.accepted is False
        assert conflict.error_code is JobErrorCode.SUBMISSION_CONFLICT
        assert conflict.job.status is JobStatus.REJECTED
        assert executor.repository.get("sub-1").code == "first"
        await executor.shutdown()

    asyncio.run(scenario())


def test_shutdown_without_wait_cancels_pending_and_running():
    async def scenario():
        executor = AnalysisExecutor(
            {"slow": SlowAdapter(delay=5.0)},
            ExecutorConfig(concurrency_limit=1, queue_limit=2),
        )
        running = await executor.submit(_submission(1), "slow")
        queued = await executor.submit(_submission(2), "slow")
        await _wait_until(
            lambda: executor.get_job(running.job.job_id).status is JobStatus.RUNNING
        )

        await executor.shutdown(wait=False)
        running_job = await executor.wait(running.job.job_id, timeout=1)
        queued_job = await executor.wait(queued.job.job_id, timeout=1)
        assert running_job.status is JobStatus.CANCELLED
        assert queued_job.status is JobStatus.CANCELLED

    asyncio.run(scenario())


def test_shutdown_waits_for_running_sync_work_and_leaves_no_tasks():
    async def scenario():
        analyzer = SlowSyncAnalyzer(delay=0.2)
        executor = AnalysisExecutor(
            {"slow-sync": SyncAnalyzerAdapter(analyzer, Assignment("hw"))},
            ExecutorConfig(concurrency_limit=1, queue_limit=2),
        )
        result = await executor.submit(_submission(1), "slow-sync", timeout=5)
        await _wait_until(analyzer.started.is_set)

        await executor.shutdown(wait=False)

        assert analyzer.finished.is_set(), "shutdown вернулся до конца CPU-вызова"
        assert executor.running_count == 0
        assert executor.pending_count == 0
        assert executor.get_job(result.job.job_id).status is JobStatus.CANCELLED

    asyncio.run(scenario())


def test_one_failure_does_not_lose_other_results():
    async def scenario():
        executor = AnalysisExecutor(
            {
                "ok": SlowAdapter(score=0.9),
                "broken": SlowAdapter(mode="error"),
            },
            ExecutorConfig(concurrency_limit=2, queue_limit=2),
        )
        first = await executor.submit(_submission(1), "ok")
        broken = await executor.submit(_submission(2), "broken")
        second = await executor.submit(_submission(3), "ok")

        jobs = await executor.wait_all(
            [first.job.job_id, broken.job.job_id, second.job.job_id],
            timeout=3,
        )
        assert jobs[0].status is JobStatus.SUCCEEDED
        assert jobs[0].result.score == 0.9
        assert jobs[1].status is JobStatus.FAILED
        assert jobs[1].error_code is JobErrorCode.ANALYZER_ERROR
        assert jobs[2].status is JobStatus.SUCCEEDED
        assert jobs[2].result.score == 0.9
        assert len(executor.results) == 2
        await executor.shutdown()

    asyncio.run(scenario())


def test_timeout_does_not_lose_other_results():
    async def scenario():
        executor = AnalysisExecutor(
            {
                "ok": SlowAdapter(score=0.8),
                "slow": SlowAdapter(delay=5.0),
            },
            ExecutorConfig(concurrency_limit=2, queue_limit=2),
        )
        slow = await executor.submit(_submission(1), "slow", timeout=0.05)
        ok = await executor.submit(_submission(2), "ok")

        slow_job, ok_job = await executor.wait_all(
            [slow.job.job_id, ok.job.job_id],
            timeout=3,
        )
        assert slow_job.status is JobStatus.TIMED_OUT
        assert slow_job.result is None
        assert ok_job.status is JobStatus.SUCCEEDED
        assert ok_job.result.score == 0.8
        assert len(executor.results) == 1
        await executor.shutdown()

    asyncio.run(scenario())


def test_queue_stays_bounded_under_pressure():
    async def scenario():
        executor = AnalysisExecutor(
            {"slow": SlowAdapter(delay=0.5)},
            ExecutorConfig(concurrency_limit=2, queue_limit=2),
        )
        results = [
            await executor.submit(_submission(index), "slow")
            for index in range(20)
        ]
        accepted = [result for result in results if result.accepted]
        rejected = [result for result in results if not result.accepted]

        assert len(accepted) == 4  # concurrency_limit + queue_limit
        assert len(rejected) == 16
        assert executor.pending_count <= 2
        assert executor.starting_count <= 2
        assert executor.running_count <= 2
        assert all(
            result.error_code is JobErrorCode.QUEUE_FULL for result in rejected
        )
        await executor.wait_all(
            [result.job.job_id for result in accepted],
            timeout=5,
        )
        await executor.shutdown()

    asyncio.run(scenario())


def test_start_can_be_separate_from_submit_and_shutdown_is_safe():
    async def scenario():
        executor = AnalysisExecutor(
            {"slow": SlowAdapter()},
            ExecutorConfig(concurrency_limit=1, queue_limit=1, auto_start=False),
        )
        with pytest.raises(RuntimeError):
            await executor.submit(_submission(1), "slow")

        executor.start()
        result = await executor.submit(_submission(1), "slow")
        assert result.accepted is True
        job = await executor.wait(result.job.job_id, timeout=2)
        assert job.status is JobStatus.SUCCEEDED
        await executor.shutdown()
        await executor.shutdown()  # повторный shutdown безопасен

        with pytest.raises(RuntimeError):
            await executor.submit(_submission(2), "slow")

    asyncio.run(scenario())

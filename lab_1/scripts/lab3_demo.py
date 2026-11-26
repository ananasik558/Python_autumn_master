"""Демонстрация этапа А ЛР3: очередь, тайм-аут, отмена и коды ошибок.

Скрипт не является пользовательским интерфейсом: это воспроизводимый прогон
исполнителя, результат которого сохраняется в ``artifacts/lab3``.

Запуск:
    python scripts/lab3_demo.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from submission_system.jobs import (
    AnalysisExecutor,
    BatchConfig,
    ExecutorConfig,
    GuardedModelAdapter,
    JobStatus,
    SlowAdapter,
    StubModelAdapter,
)
from submission_system.models import Submission

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "lab3" / "stage_a_demo.json"


def _submission(index: int) -> Submission:
    return Submission(
        student_id=f"demo-student-{index}",
        submission_id=f"demo-sub-{index}",
        timestamp="2026-10-20T12:00:00+03:00",
        code="print('demo')\n",
        text="",
    )


def _job_snapshot(job) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "submission_id": job.submission_id,
        "status": job.status.value,
        "error_code": job.error_code.value if job.error_code else None,
        "attempt_no": job.attempt_no,
        "timeout_seconds": job.timeout_seconds,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "score": job.result.score if job.result else None,
    }


async def _queue_scenario() -> dict[str, Any]:
    executor = AnalysisExecutor(
        {"slow": SlowAdapter(delay=0.15)},
        ExecutorConfig(concurrency_limit=1, queue_limit=1),
    )
    results = [await executor.submit(_submission(index), "slow") for index in range(3)]
    pending_after_submit = executor.pending_count
    starting_after_submit = executor.starting_count
    await executor.wait_all(
        [result.job.job_id for result in results if result.accepted],
        timeout=5,
    )
    payload = {
        "concurrency_limit": 1,
        "queue_limit": 1,
        "pending_after_submit": pending_after_submit,
        "starting_after_submit": starting_after_submit,
        "submissions": [
            {
                "accepted": result.accepted,
                "error_code": result.error_code.value if result.error_code else None,
                "status": result.job.status.value,
            }
            for result in results
        ],
        "third_submission_saved": executor.repository.get("demo-sub-2") is not None,
    }
    await executor.shutdown()
    return payload


async def _timeout_scenario() -> dict[str, Any]:
    executor = AnalysisExecutor(
        {"slow": SlowAdapter(delay=5.0)},
        ExecutorConfig(concurrency_limit=1, queue_limit=1),
    )
    result = await executor.submit(_submission(10), "slow", timeout=0.05)
    job = await executor.wait(result.job.job_id, timeout=2)
    payload = _job_snapshot(job)
    await executor.shutdown()
    return payload


async def _cancel_scenario() -> dict[str, Any]:
    executor = AnalysisExecutor(
        {"slow": SlowAdapter(delay=5.0)},
        ExecutorConfig(concurrency_limit=1, queue_limit=2),
    )
    running = await executor.submit(_submission(20), "slow")
    queued = await executor.submit(_submission(21), "slow")
    queued_cancelled = await executor.cancel(queued.job.job_id)
    while executor.get_job(running.job.job_id).status is JobStatus.QUEUED:
        await asyncio.sleep(0.005)
    running_cancelled = await executor.cancel(running.job.job_id)
    running_job = await executor.wait(running.job.job_id, timeout=2)
    queued_job = await executor.wait(queued.job.job_id, timeout=2)
    payload = {
        "queued_cancel_accepted": queued_cancelled,
        "running_cancel_accepted": running_cancelled,
        "queued_job": _job_snapshot(queued_job),
        "running_job": _job_snapshot(running_job),
    }
    await executor.shutdown()
    return payload


async def _error_scenario() -> dict[str, Any]:
    executor = AnalysisExecutor(
        {
            "unavailable": SlowAdapter(mode="unavailable"),
            "invalid": SlowAdapter(mode="invalid"),
            "cancelled": SlowAdapter(mode="cancelled"),
        },
        ExecutorConfig(concurrency_limit=3, queue_limit=3),
    )
    jobs = {}
    for index, analyzer_id in enumerate(("unavailable", "invalid", "cancelled"), 30):
        result = await executor.submit(_submission(index), analyzer_id)
        jobs[analyzer_id] = _job_snapshot(
            await executor.wait(result.job.job_id, timeout=2)
        )
    await executor.shutdown()
    return jobs


async def _model_scenario() -> dict[str, Any]:
    stub = StubModelAdapter(delay=0.02)
    model = GuardedModelAdapter(
        stub,
        BatchConfig(batch_size=2, concurrency_limit=2, call_timeout=2),
    )
    scores = await model.score_batch([_submission(index) for index in range(50, 55)])
    payload = {
        "batch_size": model.batch_size,
        "concurrency_limit": model.config.concurrency_limit,
        "scores": scores,
        "observed_batch_sizes": stub.batch_sizes,
        "max_active_model_calls": model.max_active_calls,
    }
    return payload


async def main_async() -> dict[str, Any]:
    return {
        "report_version": "3.0.0-stage-a",
        "status_contract": [
            "queued",
            "running",
            "succeeded",
            "failed",
            "timed_out",
            "cancelled",
            "rejected",
        ],
        "queue": await _queue_scenario(),
        "timeout": await _timeout_scenario(),
        "cancellation": await _cancel_scenario(),
        "error_codes": await _error_scenario(),
        "model_adapter": await _model_scenario(),
    }


def main() -> int:
    payload = asyncio.run(main_async())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

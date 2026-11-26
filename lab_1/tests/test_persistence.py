"""Тесты постоянного SQLite-хранилища ЛР3, этап Б."""

from __future__ import annotations

from pathlib import Path

import pytest

from submission_system.jobs import AnalysisJob, JobErrorCode, JobStatus
from submission_system.models import AnalysisReport, Evidence, Submission
from submission_system.persistence import (
    DuplicateIdempotencyKey,
    PersistenceError,
    SqliteStore,
    submission_digest,
)


def _submission(submission_id: str = "sub-1", code: str = "print(1)\n") -> Submission:
    return Submission(
        student_id="student-1",
        submission_id=submission_id,
        timestamp="2026-10-20T12:00:00+03:00",
        code=code,
        text="",
    )


def _job(job_id: str = "job-1", submission: Submission | None = None) -> AnalysisJob:
    submission = submission or _submission()
    return AnalysisJob(
        job_id=job_id,
        submission_id=submission.submission_id,
        submission_hash=submission_digest(submission),
        analyzer_id="baseline",
        config_hash="cfg-1",
        timeout_seconds=5.0,
    )


def test_submission_is_immutable_after_restart(tmp_path: Path):
    db_path = tmp_path / "analysis.db"
    store = SqliteStore(db_path)
    store.initialize()
    original = _submission(code="original\n")
    assert store.submissions.save(original) is True

    changed = _submission(code="changed\n")
    assert store.submissions.save(changed) is False
    store.close()

    reopened = SqliteStore(db_path)
    reopened.initialize()
    stored = reopened.submissions.get("sub-1")
    assert stored is not None
    assert stored.code == "original\n"
    assert stored.raw_content_hash == submission_digest(original)
    reopened.close()


def test_job_and_report_roundtrip_after_restart(tmp_path: Path):
    db_path = tmp_path / "analysis.db"
    store = SqliteStore(db_path)
    store.initialize()
    submission = _submission()
    store.submissions.save(submission)
    job = _job(submission=submission)
    store.jobs.add(job)
    job.mark_running()
    store.jobs.save(job)
    job.mark_succeeded(
        AnalysisReport(
            analyzer_id="baseline",
            analyzer_version="1.0.0",
            score=0.25,
            rubric_feedback="ok",
            evidence=[Evidence(line_number=1, text="print")],
            model_version="model-1",
            index_version="index-1",
            prompt_version="prompt-1",
            mlflow_run_id="run-1",
            config_hash="cfg-1",
        )
    )
    store.jobs.save(job)
    store.close()

    reopened = SqliteStore(db_path)
    reopened.initialize()
    loaded = reopened.jobs.require("job-1")
    assert loaded.status is JobStatus.SUCCEEDED
    assert loaded.result is not None
    assert loaded.result.score == 0.25
    assert loaded.result.evidence[0].text == "print"
    assert loaded.result.model_version == "model-1"
    assert loaded.result.index_version == "index-1"
    assert loaded.result.prompt_version == "prompt-1"
    assert loaded.result.mlflow_run_id == "run-1"
    assert loaded.result.config_hash == "cfg-1"
    reopened.close()


def test_succeeded_without_report_is_rejected(tmp_path: Path):
    store = SqliteStore(tmp_path / "analysis.db")
    store.initialize()
    submission = _submission()
    store.submissions.save(submission)
    job = _job(submission=submission)
    job.status = JobStatus.SUCCEEDED
    job.result = None

    with pytest.raises(PersistenceError):
        store.jobs.add(job)
    assert store.jobs.get("job-1") is None
    store.close()


def test_failed_ready_write_rolls_back_to_previous_state(tmp_path: Path):
    store = SqliteStore(tmp_path / "analysis.db")
    store.initialize()
    submission = _submission()
    store.submissions.save(submission)
    job = _job(submission=submission)
    store.jobs.add(job)
    job.mark_running()
    store.jobs.save(job)

    # Имитируем попытку записать succeeded без отчёта.
    job.status = JobStatus.SUCCEEDED
    job.result = None
    with pytest.raises(PersistenceError):
        store.jobs.save(job)

    reloaded = store.jobs.require("job-1")
    assert reloaded.status is JobStatus.RUNNING
    assert reloaded.result is None
    store.close()


def test_corrupted_succeeded_job_is_marked_inconsistent(tmp_path: Path):
    store = SqliteStore(tmp_path / "analysis.db")
    store.initialize()
    submission = _submission()
    store.submissions.save(submission)
    job = _job(submission=submission)
    store.jobs.add(job)

    # Имитируем повреждение БД: статус succeeded без строки отчёта.
    with store.connect() as connection:
        with connection:
            connection.execute(
                "UPDATE jobs SET status = ? WHERE job_id = ?",
                (JobStatus.SUCCEEDED.value, "job-1"),
            )

    recovered = store.recover_unfinished()
    loaded = store.jobs.require("job-1")
    assert recovered["inconsistent"] == 1
    assert loaded.status is JobStatus.FAILED
    assert loaded.error_code is JobErrorCode.INCONSISTENT_STATE
    store.close()


def test_interrupted_jobs_are_marked_after_restart(tmp_path: Path):
    db_path = tmp_path / "analysis.db"
    store = SqliteStore(db_path)
    store.initialize()
    submission = _submission()
    store.submissions.save(submission)
    queued = _job("job-queued", submission=submission)
    running = _job("job-running", submission=submission)
    store.jobs.add(queued)
    store.jobs.add(running)
    running.mark_running()
    store.jobs.save(running)
    store.close()

    reopened = SqliteStore(db_path)
    recovered = reopened.initialize()
    loaded_queued = reopened.jobs.require("job-queued")
    loaded_running = reopened.jobs.require("job-running")

    assert recovered["interrupted"] == 2
    for job in (loaded_queued, loaded_running):
        assert job.status is JobStatus.FAILED
        assert job.error_code is JobErrorCode.INTERRUPTED
        assert job.attempt_no == 1
        assert job.finished_at is not None
    assert len(reopened.jobs) == 2
    reopened.close()


def test_job_and_idempotency_key_are_written_atomically(tmp_path: Path):
    store = SqliteStore(tmp_path / "analysis.db")
    store.initialize()
    submission = _submission()
    store.submissions.save(submission)
    digest = submission_digest(submission)
    job = _job("job-1", submission=submission)

    store.jobs.add_with_idempotency(
        job,
        key="key-1",
        submission_hash=digest,
        analyzer_id="baseline",
        config_hash="cfg-1",
    )
    assert store.jobs.get("job-1") is not None
    assert store.jobs.get_idempotency("key-1").job_id == "job-1"

    duplicate = _job("job-2", submission=submission)
    with pytest.raises(DuplicateIdempotencyKey):
        store.jobs.add_with_idempotency(
            duplicate,
            key="key-1",
            submission_hash=digest,
            analyzer_id="baseline",
            config_hash="cfg-1",
        )
    # Транзакция откатилась: «осиротевшей» задачи job-2 нет.
    assert store.jobs.get("job-2") is None
    store.close()


def test_idempotency_record_and_attempt_history(tmp_path: Path):
    store = SqliteStore(tmp_path / "analysis.db")
    store.initialize()
    submission = _submission()
    store.submissions.save(submission)
    digest = submission_digest(submission)

    first = _job("job-1", submission=submission)
    store.jobs.add(first)
    first.mark_running()
    first.mark_failed(JobErrorCode.ANALYZER_ERROR, "boom")
    store.jobs.save(first)

    record = store.jobs.save_idempotency(
        key="key-1",
        job_id="job-1",
        submission_hash=digest,
        analyzer_id="baseline",
        config_hash="cfg-1",
    )
    assert record.job_id == "job-1"
    assert store.jobs.get_idempotency("key-1").matches(
        submission_hash=digest,
        analyzer_id="baseline",
        config_hash="cfg-1",
    )
    assert store.jobs.next_attempt_no(digest, "baseline", "cfg-1") == 2

    second = _job("job-2", submission=submission)
    second.attempt_no = 2
    store.jobs.add(second)
    assert store.jobs.next_attempt_no(digest, "baseline", "cfg-1") == 3

    with pytest.raises(DuplicateIdempotencyKey):
        store.jobs.save_idempotency(
            key="key-1",
            job_id="job-2",
            submission_hash=digest,
            analyzer_id="baseline",
            config_hash="cfg-1",
        )
    store.close()

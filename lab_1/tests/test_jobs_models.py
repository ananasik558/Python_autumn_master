"""Тесты моделей задач ЛР3, этап А."""

from __future__ import annotations

import pytest

from submission_system.jobs import (
    AnalysisJob,
    JobErrorCode,
    JobStateError,
    JobStatus,
    JobStore,
    build_analysis_job,
)
from submission_system.models import AnalysisReport, Submission


def _submission(submission_id: str = "sub-1") -> Submission:
    return Submission(
        student_id="s1",
        submission_id=submission_id,
        timestamp="2026-10-20T12:00:00+03:00",
        code="print(1)\n",
        text="",
    )


def test_status_contract_marks_terminal_states():
    terminal = {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.TIMED_OUT,
        JobStatus.CANCELLED,
        JobStatus.REJECTED,
    }
    assert all(status.is_terminal for status in terminal)
    assert JobStatus.QUEUED.is_active
    assert JobStatus.RUNNING.is_active
    assert JobStatus.QUEUED == "queued"


def test_analysis_job_contains_required_fields_and_timestamps():
    job = build_analysis_job(
        _submission(),
        "length_analyzer",
        config_hash="cfg-1",
        timeout_seconds=2.5,
    )

    assert job.job_id.startswith("job-")
    assert job.submission_id == "sub-1"
    assert job.submission_hash
    assert job.analyzer_id == "length_analyzer"
    assert job.config_hash == "cfg-1"
    assert job.status is JobStatus.QUEUED
    assert job.attempt_no == 1
    assert job.timeout_seconds == 2.5
    assert job.created_at == job.queued_at
    assert job.started_at is None
    assert job.finished_at is None
    assert job.is_terminal is False


def test_job_transitions_and_report_are_recorded():
    job = build_analysis_job(_submission(), "a")
    report = AnalysisReport(analyzer_id="a", analyzer_version="1", score=0.5)

    job.mark_running()
    assert job.status is JobStatus.RUNNING
    assert job.started_at is not None

    job.mark_succeeded(report)
    assert job.status is JobStatus.SUCCEEDED
    assert job.result is report
    assert job.finished_at is not None
    assert job.duration_seconds is not None
    assert job.duration_seconds >= 0.0
    payload = job.to_dict()
    assert payload["status"] == "succeeded"
    assert payload["result"]["score"] == 0.5
    assert payload["error_code"] is None


def test_terminal_job_cannot_be_restarted():
    job = build_analysis_job(_submission(), "a")
    job.mark_running()
    job.mark_timed_out("too slow")

    with pytest.raises(JobStateError):
        job.mark_running()
    with pytest.raises(JobStateError):
        job.mark_succeeded(AnalysisReport(analyzer_id="a", analyzer_version="1"))


def test_rejected_job_has_error_code_and_finish_time():
    job = build_analysis_job(_submission(), "a")
    job.mark_rejected(JobErrorCode.QUEUE_FULL, "queue is full")

    assert job.status is JobStatus.REJECTED
    assert job.error_code is JobErrorCode.QUEUE_FULL
    assert job.finished_at is not None
    assert job.to_dict()["error_code"] == "queue_full"


def test_job_store_lists_jobs_by_status():
    store = JobStore()
    first = build_analysis_job(_submission("a"), "x", job_id="job-a")
    second = build_analysis_job(_submission("b"), "x", job_id="job-b")
    store.add(first)
    store.add(second)
    first.mark_rejected(JobErrorCode.QUEUE_FULL, "full")

    assert store.require("job-a") is first
    assert [job.job_id for job in store.list(JobStatus.REJECTED)] == ["job-a"]
    assert len(store) == 2
    assert "job-b" in store
    with pytest.raises(ValueError):
        store.add(first)

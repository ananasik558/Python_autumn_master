"""Интеграционные тесты FastAPI-сервиса ЛР3, этап Б."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from submission_system.api import Settings, create_app
from submission_system.jobs import AnalysisJob, JobErrorCode, JobStatus
from submission_system.models import Submission
from submission_system.persistence import SqliteStore, submission_digest

HEADERS = {"X-API-Key": "test-key"}


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "db_path": str(tmp_path / "api.db"),
        "api_key": "test-key",
        "executor_concurrency": 2,
        "queue_limit": 4,
        "model_mode": "stub",
        "model_delay": 0.0,
        "mlflow_run_id": "mlflow-run-123",
    }
    values.update(overrides)
    return Settings(**values)


def _submission_payload(
    submission_id: str = "sub-1",
    *,
    code: str = "print(1)\n",
    student_id: str = "student-1",
    assignment_id: str = "hw_1",
) -> dict:
    return {
        "student_id": student_id,
        "submission_id": submission_id,
        "assignment_id": assignment_id,
        "timestamp": "2026-10-20T12:00:00+03:00",
        "code": code,
        "text": "",
    }


def _create_submission(
    client: TestClient,
    *,
    headers: dict | None = None,
    **kwargs,
) -> dict:
    response = client.post(
        "/submissions",
        json=_submission_payload(**kwargs),
        headers=headers or HEADERS,
    )
    assert response.status_code in {200, 201}, response.text
    return response.json()


def _start(
    client: TestClient,
    *,
    submission_id: str = "sub-1",
    analyzer_id: str = "baseline",
    key: str = "key-1",
    **extra,
) -> dict:
    body = {
        "submission_id": submission_id,
        "analyzer_id": analyzer_id,
        "idempotency_key": key,
        **extra,
    }
    return client.post("/analysis-jobs", json=body, headers=HEADERS)


def _wait_job(
    client: TestClient,
    job_id: str,
    *,
    timeout: float = 3.0,
    headers: dict | None = None,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(
            f"/analysis-jobs/{job_id}",
            headers=headers or HEADERS,
        )
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in {
            "succeeded",
            "failed",
            "timed_out",
            "cancelled",
            "rejected",
        }:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Задача {job_id} не завершилась за {timeout}s")


def _wait_status(
    client: TestClient,
    job_id: str,
    expected: str,
    *,
    timeout: float = 3.0,
    headers: dict | None = None,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(
            f"/analysis-jobs/{job_id}",
            headers=headers or HEADERS,
        )
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] == expected:
            return job
        time.sleep(0.005)
    raise AssertionError(f"Задача {job_id} не достигла статуса {expected}")


def test_health_is_open_and_other_routes_require_api_key(tmp_path: Path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/submissions").status_code == 401
        assert (
            client.get(
                "/submissions",
                headers={"X-API-Key": "wrong"},
            ).status_code
            == 401
        )
        assert client.get("/submissions", headers=HEADERS).status_code == 200
        assert (
            client.get(
                "/submissions",
                headers={"Authorization": "Bearer test-key"},
            ).status_code
            == 200
        )
        assert client.get("/api/v1/submissions", headers=HEADERS).status_code == 200


def test_submission_is_immutable(tmp_path: Path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        created = _create_submission(client)
        assert created["code"] == "print(1)\n"

        conflict = client.post(
            "/submissions",
            json=_submission_payload(code="other code\n"),
            headers=HEADERS,
        )
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["code"] == "submission_conflict"

        stored = client.get("/submissions/sub-1", headers=HEADERS)
        assert stored.status_code == 200
        assert stored.json()["code"] == "print(1)\n"


def test_baseline_job_and_report_contain_metadata(tmp_path: Path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        _create_submission(client)
        response = _start(client)
        assert response.status_code == 202, response.text
        job = response.json()
        assert job["status"] == "queued"

        final = _wait_job(client, job["job_id"])
        assert final["status"] == "succeeded"
        assert final["has_result"] is True

        report = client.get(
            f"/analysis-jobs/{job['job_id']}/report",
            headers=HEADERS,
        )
        assert report.status_code == 200
        payload = report.json()
        assert payload["analyzer_id"] == "baseline"
        assert payload["model_version"] == "length-analyzer-1.0.0"
        assert payload["index_version"] is None
        assert payload["mlflow_run_id"] == "mlflow-run-123"
        assert payload["config_hash"] == job["config_hash"]

        model_start = _start(client, analyzer_id="model", key="model-key")
        model_job = _wait_job(client, model_start.json()["job_id"])
        assert model_job["status"] == "succeeded"
        model_report = client.get(
            f"/analysis-jobs/{model_job['job_id']}/report",
            headers=HEADERS,
        ).json()
        assert model_report["model_version"] == "stub-model-1.0.0"
        assert model_report["index_version"] == "tfidf-lr2"
        assert model_report["prompt_version"] == "retrieval-feedback-1.0.0"
        assert model_report["mlflow_run_id"] == "mlflow-run-123"


def test_idempotent_replay_returns_same_job(tmp_path: Path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        _create_submission(client)
        first = _start(client, key="same-key")
        second = _start(client, key="same-key")

        assert first.status_code == 202
        assert second.status_code == 200
        assert second.headers["x-idempotent-replay"] == "true"
        assert first.json()["job_id"] == second.json()["job_id"]

        jobs = client.get("/analysis-jobs", headers=HEADERS).json()
        assert len(jobs) == 1

        header_only = client.post(
            "/analysis-jobs",
            json={
                "submission_id": "sub-1",
                "analyzer_id": "baseline",
            },
            headers={**HEADERS, "Idempotency-Key": "header-key"},
        )
        assert header_only.status_code == 202


def test_idempotency_key_with_different_payload_conflicts(tmp_path: Path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        _create_submission(client)
        first = _start(client, key="key-1")
        assert first.status_code == 202

        different_analyzer = _start(
            client,
            analyzer_id="model",
            key="key-1",
        )
        assert different_analyzer.status_code == 409
        assert different_analyzer.json()["detail"]["code"] == "idempotency_conflict"

        different_config = _start(client, key="key-1", config_hash="other")
        assert different_config.status_code == 409


def test_new_key_after_failure_creates_new_attempt_and_keeps_history(tmp_path: Path):
    app = create_app(_settings(tmp_path, model_mode="unavailable"))
    with TestClient(app) as client:
        _create_submission(client)
        first = _start(client, analyzer_id="model", key="key-1")
        first_job = _wait_job(client, first.json()["job_id"])
        assert first_job["status"] == "failed"
        assert first_job["error_code"] == "model_unavailable"
        assert first_job["attempt_no"] == 1

        replay = _start(client, analyzer_id="model", key="key-1")
        assert replay.status_code == 200
        assert replay.json()["job_id"] == first.json()["job_id"]
        assert replay.json()["status"] == "failed"

        second = _start(client, analyzer_id="model", key="key-2")
        assert second.status_code == 202
        second_job = _wait_job(client, second.json()["job_id"])
        assert second_job["job_id"] != first_job["job_id"]
        assert second_job["attempt_no"] == 2
        assert second_job["status"] == "failed"

        history = client.get(
            "/analysis-jobs",
            params={"submission_id": "sub-1"},
            headers=HEADERS,
        ).json()
        assert len(history) == 2
        assert {item["attempt_no"] for item in history} == {1, 2}


def test_queue_full_returns_429_and_submission_survives(tmp_path: Path):
    app = create_app(
        _settings(
            tmp_path,
            executor_concurrency=1,
            queue_limit=0,
            model_delay=0.5,
        )
    )
    with TestClient(app) as client:
        _create_submission(client)
        first = _start(client, analyzer_id="model", key="key-1")
        assert first.status_code == 202

        second = _start(client, analyzer_id="model", key="key-2")
        assert second.status_code == 429
        assert second.json()["detail"]["code"] == "queue_full"
        assert second.json()["detail"]["job"]["status"] == "rejected"

        assert client.get("/submissions/sub-1", headers=HEADERS).status_code == 200


def test_model_unavailable_does_not_break_baseline(tmp_path: Path):
    app = create_app(_settings(tmp_path, model_mode="unavailable"))
    with TestClient(app) as client:
        _create_submission(client)
        model = _start(client, analyzer_id="model", key="model-key")
        model_job = _wait_job(client, model.json()["job_id"])
        assert model_job["status"] == "failed"
        assert model_job["error_code"] == "model_unavailable"

        baseline = _start(client, analyzer_id="baseline", key="baseline-key")
        baseline_job = _wait_job(client, baseline.json()["job_id"])
        assert baseline_job["status"] == "succeeded"
        assert baseline_job["has_result"] is True


def test_cancel_running_job(tmp_path: Path):
    app = create_app(_settings(tmp_path, model_delay=1.0))
    with TestClient(app) as client:
        _create_submission(client)
        started = _start(client, analyzer_id="model", key="cancel-key")
        job_id = started.json()["job_id"]
        _wait_status(client, job_id, "running")

        cancelled = client.post(
            f"/analysis-jobs/{job_id}/cancel",
            headers=HEADERS,
        )
        assert cancelled.status_code == 200
        final = _wait_job(client, job_id)
        assert final["status"] == "cancelled"
        assert final["has_result"] is False


def test_http_access_controls_for_students_and_teachers(tmp_path: Path):
    app = create_app(
        _settings(
            tmp_path,
            student_keys={
                "student-a-key": "student-a",
                "student-b-key": "student-b",
            },
            teacher_keys={
                "teacher-hw1-key": ("hw_1",),
                "teacher-hw2-key": ("hw_2",),
            },
        )
    )
    student_a = {"X-API-Key": "student-a-key"}
    student_b = {"X-API-Key": "student-b-key"}
    teacher_hw1 = {"X-API-Key": "teacher-hw1-key"}
    teacher_hw2 = {"X-API-Key": "teacher-hw2-key"}

    with TestClient(app) as client:
        _create_submission(
            client,
            headers=student_a,
            submission_id="sub-a",
            student_id="student-a",
            assignment_id="hw_1",
        )
        started = client.post(
            "/analysis-jobs",
            json={
                "submission_id": "sub-a",
                "analyzer_id": "baseline",
                "idempotency_key": "student-a-key-1",
            },
            headers=student_a,
        )
        assert started.status_code == 202, started.text
        job_id = started.json()["job_id"]
        assert _wait_job(client, job_id, headers=student_a)["status"] == "succeeded"

        # Владелец видит свою работу и отчёт.
        assert client.get("/submissions/sub-a", headers=student_a).status_code == 200
        assert client.get(
            f"/analysis-jobs/{job_id}/report",
            headers=student_a,
        ).status_code == 200

        # Другой студент не видит чужой отчёт и чужую задачу.
        assert client.get(
            f"/analysis-jobs/{job_id}/report",
            headers=student_b,
        ).status_code == 403
        assert client.get(
            f"/analysis-jobs/{job_id}",
            headers=student_b,
        ).status_code == 403
        assert client.get("/submissions", headers=student_b).json() == []
        assert client.get("/analysis-jobs", headers=student_b).json() == []

        # Преподаватель другого задания не видит отчёт.
        assert client.get(
            f"/analysis-jobs/{job_id}/report",
            headers=teacher_hw2,
        ).status_code == 403
        assert client.get("/submissions", headers=teacher_hw2).json() == []
        # Преподаватель своего задания видит отчёт.
        assert client.get(
            f"/analysis-jobs/{job_id}/report",
            headers=teacher_hw1,
        ).status_code == 200
        assert len(client.get("/submissions", headers=teacher_hw1).json()) == 1

        # Студент не создаёт сдачу за другого и не запускает чужой анализ.
        forbidden_create = client.post(
            "/submissions",
            json=_submission_payload(
                submission_id="sub-b",
                student_id="student-b",
                assignment_id="hw_1",
            ),
            headers=student_a,
        )
        assert forbidden_create.status_code == 403
        forbidden_start = client.post(
            "/analysis-jobs",
            json={
                "submission_id": "sub-a",
                "analyzer_id": "baseline",
                "idempotency_key": "student-b-key-1",
            },
            headers=student_b,
        )
        assert forbidden_start.status_code == 403


def test_idempotency_survives_restart(tmp_path: Path):
    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        _create_submission(client)
        first = _start(client, key="persisted-key")
        job = _wait_job(client, first.json()["job_id"])
        assert job["status"] == "succeeded"
        job_id = job["job_id"]

    restarted_app = create_app(settings)
    with TestClient(restarted_app) as client:
        replay = _start(client, key="persisted-key")
        assert replay.status_code == 200
        assert replay.json()["job_id"] == job_id
        assert replay.json()["status"] == "succeeded"


def test_restart_recovers_unfinished_job_and_preserves_history(tmp_path: Path):
    db_path = tmp_path / "restart.db"
    store = SqliteStore(db_path)
    store.initialize()
    submission = Submission(
        student_id="student-1",
        submission_id="sub-1",
        timestamp="2026-10-20T12:00:00+03:00",
        code="print(1)\n",
    )
    store.submissions.save(submission)
    job = AnalysisJob(
        job_id="job-interrupted",
        submission_id="sub-1",
        submission_hash=submission_digest(submission),
        analyzer_id="model",
        config_hash="cfg",
        attempt_no=3,
    )
    store.jobs.add(job)
    job.mark_running()
    store.jobs.save(job)
    store.close()

    app = create_app(_settings(tmp_path, db_path=str(db_path)))
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["recovered"]["interrupted"] == 1

        restored = client.get(
            "/analysis-jobs/job-interrupted",
            headers=HEADERS,
        ).json()
        assert restored["status"] == "failed"
        assert restored["error_code"] == "interrupted"
        assert restored["attempt_no"] == 3

        history = client.get("/analysis-jobs", headers=HEADERS).json()
        assert len(history) == 1


def test_api_does_not_expose_provider_or_access_secrets(tmp_path: Path):
    app = create_app(
        _settings(
            tmp_path,
            api_key="access-secret",
            provider_api_key="provider-secret",
        )
    )
    headers = {"X-API-Key": "access-secret"}
    with TestClient(app) as client:
        client.post(
            "/submissions",
            json=_submission_payload(),
            headers=headers,
        )
        start = client.post(
            "/analysis-jobs",
            json={
                "submission_id": "sub-1",
                "analyzer_id": "model",
                "idempotency_key": "secret-key",
            },
            headers=headers,
        )
        job = _wait_job(client, start.json()["job_id"], headers=headers)
        responses = [
            client.get("/health").text,
            start.text,
            client.get(f"/analysis-jobs/{job['job_id']}", headers=headers).text,
            client.get(
                f"/analysis-jobs/{job['job_id']}/report",
                headers=headers,
            ).text,
            client.get("/analyzers", headers=headers).text,
        ]
        for text in responses:
            assert "provider-secret" not in text
            assert "access-secret" not in text

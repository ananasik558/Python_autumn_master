"""Демонстрация этапа Б: SQLite, идемпотентность и FastAPI.

Запуск:
    python scripts/lab3b_demo.py
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from submission_system.api import Settings, create_app
from submission_system.jobs import AnalysisJob
from submission_system.models import Submission
from submission_system.persistence import SqliteStore, submission_digest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "lab3b" / "api_demo.json"
HEADERS = {"X-API-Key": "demo-key"}


def _wait(client: TestClient, job_id: str, timeout: float = 3.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/analysis-jobs/{job_id}", headers=HEADERS)
        response.raise_for_status()
        job = response.json()
        if job["status"] in {"succeeded", "failed", "timed_out", "cancelled"}:
            return job
        time.sleep(0.01)
    raise TimeoutError(job_id)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="lab3b-demo-") as temp_dir:
        db_path = Path(temp_dir) / "analysis.db"
        settings = Settings(
            db_path=str(db_path),
            api_key="demo-key",
            model_delay=0.02,
            mlflow_run_id="demo-mlflow-run",
        )
        app = create_app(settings)
        with TestClient(app) as client:
            health = client.get("/health").json()
            submission_payload = {
                "student_id": "student-demo",
                "submission_id": "sub-demo",
                "timestamp": "2026-10-20T12:00:00+03:00",
                "code": "print('demo')\n",
                "text": "",
            }
            submission = client.post(
                "/submissions",
                json=submission_payload,
                headers=HEADERS,
            ).json()
            baseline_start = client.post(
                "/analysis-jobs",
                json={
                    "submission_id": "sub-demo",
                    "analyzer_id": "baseline",
                    "idempotency_key": "demo-baseline-key",
                },
                headers=HEADERS,
            ).json()
            baseline_job = _wait(client, baseline_start["job_id"])
            baseline_report = client.get(
                f"/analysis-jobs/{baseline_start['job_id']}/report",
                headers=HEADERS,
            ).json()

            model_start = client.post(
                "/analysis-jobs",
                json={
                    "submission_id": "sub-demo",
                    "analyzer_id": "model",
                    "idempotency_key": "demo-model-key",
                },
                headers=HEADERS,
            ).json()
            model_job = _wait(client, model_start["job_id"])
            replay = client.post(
                "/analysis-jobs",
                json={
                    "submission_id": "sub-demo",
                    "analyzer_id": "baseline",
                    "idempotency_key": "demo-baseline-key",
                },
                headers=HEADERS,
            )
            jobs = client.get("/analysis-jobs", headers=HEADERS).json()
            analyzers = client.get("/analyzers", headers=HEADERS).json()

        # Имитируем падение: незавершённая попытка осталась в running.
        store = SqliteStore(db_path)
        store.initialize()
        persisted = store.submissions.get("sub-demo")
        assert persisted is not None
        interrupted = AnalysisJob(
            job_id="job-interrupted-demo",
            submission_id="sub-demo",
            submission_hash=submission_digest(persisted),
            analyzer_id="model",
            config_hash="demo-config",
            attempt_no=4,
        )
        store.jobs.add(interrupted)
        interrupted.mark_running()
        store.jobs.save(interrupted)
        store.close()

        restarted_app = create_app(settings)
        with TestClient(restarted_app) as client:
            restart_health = client.get("/health").json()
            interrupted_job = client.get(
                "/analysis-jobs/job-interrupted-demo",
                headers=HEADERS,
            ).json()

        payload = {
            "report_version": "3.1.0-stage-b",
            "health": health,
            "submission": submission,
            "baseline_job": baseline_job,
            "baseline_report": baseline_report,
            "model_job": model_job,
            "idempotent_replay": {
                "status_code": replay.status_code,
                "replay_header": replay.headers.get("x-idempotent-replay"),
                "same_job": replay.json()["job_id"] == baseline_start["job_id"],
            },
            "jobs_count": len(jobs),
            "analyzers": analyzers,
            "restart_health": restart_health,
            "interrupted_job": interrupted_job,
            "caveat": (
                "Демонстрация использует временную БД; внешняя модель не вызывается."
            ),
        }

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

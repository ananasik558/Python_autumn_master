"""SQLite-хранилище этапа Б ЛР3.

Хранятся три сущности: исходная сдача, задача анализа и отчёт. Сдача
неизменяема: повтор с тем же ``submission_id`` и другим содержимым не
перезаписывает оригинал. Отчёт и статус ``succeeded`` пишутся в одной
транзакции, поэтому готовый статус без отчёта не появляется.

При старте сервис вызывает :meth:`SqliteStore.recover_unfinished`: задачи,
оставшиеся в ``queued``/``running`` после падения процесса, переводятся в
``failed`` с кодом ``interrupted``. История и ``attempt_no`` сохраняются.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from ..jobs.models import AnalysisJob, JobErrorCode, JobStatus, utc_now
from ..models import AnalysisReport, Evidence, Submission


class PersistenceError(RuntimeError):
    """Нарушение целостности хранилища."""


class DuplicateIdempotencyKey(PersistenceError):
    """Тот же idempotency_key уже использован."""


@dataclass(frozen=True)
class IdempotencyRecord:
    idempotency_key: str
    job_id: str
    submission_hash: str
    analyzer_id: str
    config_hash: str
    created_at: str

    def matches(
        self,
        *,
        submission_hash: str,
        analyzer_id: str,
        config_hash: str,
    ) -> bool:
        return (
            self.submission_hash == submission_hash
            and self.analyzer_id == analyzer_id
            and self.config_hash == config_hash
        )


SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    submission_id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    code TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    ai_declaration INTEGER,
    assignment_id TEXT NOT NULL DEFAULT '',
    raw_content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    submission_id TEXT NOT NULL,
    submission_hash TEXT NOT NULL,
    analyzer_id TEXT NOT NULL,
    config_hash TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    attempt_no INTEGER NOT NULL,
    timeout_seconds REAL NOT NULL,
    created_at TEXT NOT NULL,
    queued_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT,
    error_code TEXT,
    error_message TEXT,
    FOREIGN KEY (submission_id) REFERENCES submissions(submission_id)
);

CREATE TABLE IF NOT EXISTS analysis_reports (
    job_id TEXT PRIMARY KEY,
    analyzer_id TEXT NOT NULL,
    analyzer_version TEXT NOT NULL,
    score REAL,
    similarity REAL,
    rubric_feedback TEXT NOT NULL DEFAULT '',
    evidence_json TEXT NOT NULL DEFAULT '[]',
    model_version TEXT,
    index_version TEXT,
    prompt_version TEXT,
    mlflow_run_id TEXT,
    config_hash TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    idempotency_key TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    submission_hash TEXT NOT NULL,
    analyzer_id TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_submission ON jobs(submission_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_attempt
    ON jobs(submission_hash, analyzer_id, config_hash, attempt_no);
CREATE INDEX IF NOT EXISTS idx_idempotency_job ON idempotency_keys(job_id);
"""


class SqliteStore:
    """Файловое SQLite-хранилище с репозиториями сдач и задач."""

    def __init__(self, path: str | Path = "data/analysis.db") -> None:
        self.path = str(path)
        self._memory_connection: Optional[sqlite3.Connection] = None
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.submissions = SqliteSubmissionRepository(self)
        self.jobs = SqliteJobStore(self)
        self._initialized = False

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self.path == ":memory:":
            if self._memory_connection is None:
                self._memory_connection = sqlite3.connect(
                    ":memory:",
                    check_same_thread=False,
                )
                self._memory_connection.row_factory = sqlite3.Row
            yield self._memory_connection
            return

        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self, *, recover: bool = True) -> dict[str, int]:
        with self.connect() as connection:
            with connection:
                connection.executescript(SCHEMA)
                columns = {
                    row["name"]
                    for row in connection.execute(
                        "PRAGMA table_info(submissions)"
                    ).fetchall()
                }
                if "assignment_id" not in columns:
                    connection.execute(
                        "ALTER TABLE submissions "
                        "ADD COLUMN assignment_id TEXT NOT NULL DEFAULT ''"
                    )
        self._initialized = True
        return self.recover_unfinished() if recover else {}

    def close(self) -> None:
        if self._memory_connection is not None:
            self._memory_connection.close()
            self._memory_connection = None

    def recover_unfinished(self) -> dict[str, int]:
        """Однозначно помечает прерванные и несогласованные попытки."""

        now = utc_now()
        with self.connect() as connection:
            with connection:
                interrupted = connection.execute(
                    """
                    UPDATE jobs
                       SET status = ?,
                           error_code = ?,
                           error_message = ?,
                           finished_at = ?,
                           updated_at = ?
                     WHERE status IN (?, ?)
                    """,
                    (
                        JobStatus.FAILED.value,
                        JobErrorCode.INTERRUPTED.value,
                        "Процесс был перезапущен до завершения задачи",
                        now,
                        now,
                        JobStatus.QUEUED.value,
                        JobStatus.RUNNING.value,
                    ),
                ).rowcount
                inconsistent = connection.execute(
                    """
                    UPDATE jobs
                       SET status = ?,
                           error_code = ?,
                           error_message = ?,
                           finished_at = ?,
                           updated_at = ?
                     WHERE status = ?
                       AND NOT EXISTS (
                           SELECT 1 FROM analysis_reports AS r
                            WHERE r.job_id = jobs.job_id
                       )
                    """,
                    (
                        JobStatus.FAILED.value,
                        JobErrorCode.INCONSISTENT_STATE.value,
                        "Статус succeeded без сохранённого отчёта",
                        now,
                        now,
                        JobStatus.SUCCEEDED.value,
                    ),
                ).rowcount
        return {
            "interrupted": interrupted,
            "inconsistent": inconsistent,
        }


class SqliteSubmissionRepository:
    """Неизменяемое хранилище исходных сдач."""

    def __init__(self, store: SqliteStore) -> None:
        self.store = store

    def get(self, submission_id: str) -> Optional[Submission]:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT * FROM submissions WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
        return _submission_from_row(row) if row is not None else None

    def save(self, submission: Submission) -> bool:
        digest = submission.raw_content_hash or submission_digest(submission)
        with self.store.connect() as connection:
            try:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO submissions (
                            submission_id, student_id, timestamp, code, text,
                            ai_declaration, assignment_id, raw_content_hash,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            submission.submission_id,
                            submission.student_id,
                            submission.timestamp,
                            submission.code or "",
                            submission.text or "",
                            None
                            if submission.ai_declaration is None
                            else int(submission.ai_declaration),
                            submission.assignment_id or "",
                            digest,
                            utc_now(),
                        ),
                    )
                return True
            except sqlite3.IntegrityError:
                return False

    def list(self, *, limit: int = 100, offset: int = 0) -> list[Submission]:
        with self.store.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM submissions
                 ORDER BY created_at ASC, submission_id ASC
                 LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [_submission_from_row(row) for row in rows]

    def count(self) -> int:
        with self.store.connect() as connection:
            return int(
                connection.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
            )


class SqliteJobStore:
    """Задачи и отчёты; succeeded и report сохраняются атомарно."""

    def __init__(self, store: SqliteStore) -> None:
        self.store = store

    # ------------------------------------------------------------------
    # JobStore-совместимый интерфейс
    # ------------------------------------------------------------------
    def add(self, job: AnalysisJob) -> None:
        if job.status is JobStatus.SUCCEEDED and job.result is None:
            raise PersistenceError("Нельзя сохранить succeeded без отчёта")
        if job.result is not None and job.status is not JobStatus.SUCCEEDED:
            raise PersistenceError("Отчёт допустим только для статуса succeeded")
        with self.store.connect() as connection:
            with connection:
                exists = connection.execute(
                    "SELECT 1 FROM jobs WHERE job_id = ?",
                    (job.job_id,),
                ).fetchone()
                if exists is not None:
                    raise ValueError(f"Задача с job_id={job.job_id} уже существует")
                self._write_job(connection, job)
                if job.result is not None:
                    self._write_report(connection, job)

    def add_with_idempotency(
        self,
        job: AnalysisJob,
        *,
        key: str,
        submission_hash: str,
        analyzer_id: str,
        config_hash: str,
    ) -> IdempotencyRecord:
        """Атомарно создаёт job и idempotency-ключ.

        Если ключ уже занят, транзакция откатывается, и в БД не появляется
        «осиротевшая» задача без ключа.
        """

        if job.status is not JobStatus.QUEUED:
            raise PersistenceError("Новая задача должна быть в статусе queued")
        record = IdempotencyRecord(
            idempotency_key=key,
            job_id=job.job_id,
            submission_hash=submission_hash,
            analyzer_id=analyzer_id,
            config_hash=config_hash,
            created_at=utc_now(),
        )
        with self.store.connect() as connection:
            try:
                with connection:
                    exists = connection.execute(
                        "SELECT 1 FROM jobs WHERE job_id = ?",
                        (job.job_id,),
                    ).fetchone()
                    if exists is not None:
                        raise ValueError(
                            f"Задача с job_id={job.job_id} уже существует"
                        )
                    self._write_job(connection, job)
                    connection.execute(
                        """
                        INSERT INTO idempotency_keys (
                            idempotency_key, job_id, submission_hash,
                            analyzer_id, config_hash, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.idempotency_key,
                            record.job_id,
                            record.submission_hash,
                            record.analyzer_id,
                            record.config_hash,
                            record.created_at,
                        ),
                    )
                return record
            except sqlite3.IntegrityError as exc:
                raise DuplicateIdempotencyKey(key) from exc

    def save(self, job: AnalysisJob) -> None:
        if job.status is JobStatus.SUCCEEDED and job.result is None:
            raise PersistenceError("Нельзя сохранить succeeded без отчёта")
        if job.result is not None and job.status is not JobStatus.SUCCEEDED:
            raise PersistenceError("Отчёт допустим только для статуса succeeded")
        with self.store.connect() as connection:
            with connection:
                self._write_job(connection, job)
                if job.result is not None:
                    self._write_report(connection, job)

    def get(self, job_id: str) -> Optional[AnalysisJob]:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            report_row = connection.execute(
                "SELECT * FROM analysis_reports WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        job = _job_from_row(row)
        job.result = _report_from_row(report_row) if report_row is not None else None
        if job.status is JobStatus.SUCCEEDED and job.result is None:
            # Защита чтения: готовый статус без отчёта не выдаётся как успех.
            job.status = JobStatus.FAILED
            job.error_code = JobErrorCode.INCONSISTENT_STATE
            job.error_message = "Статус succeeded без сохранённого отчёта"
            job.finished_at = job.finished_at or utc_now()
        return job

    def require(self, job_id: str) -> AnalysisJob:
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"Задача не найдена: {job_id}")
        return job

    def list(
        self,
        status: JobStatus | None = None,
        *,
        submission_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AnalysisJob]:
        query = "SELECT * FROM jobs"
        conditions: list[str] = []
        params: list[Any] = []
        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)
        if submission_id is not None:
            conditions.append("submission_id = ?")
            params.append(submission_id)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at ASC, job_id ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self.store.connect() as connection:
            rows = connection.execute(query, params).fetchall()
            job_ids = [row["job_id"] for row in rows]
            reports = self._reports_for(connection, job_ids)
        jobs: list[AnalysisJob] = []
        for row in rows:
            job = _job_from_row(row)
            job.result = reports.get(job.job_id)
            if job.status is JobStatus.SUCCEEDED and job.result is None:
                job.status = JobStatus.FAILED
                job.error_code = JobErrorCode.INCONSISTENT_STATE
                job.error_message = "Статус succeeded без сохранённого отчёта"
                job.finished_at = job.finished_at or utc_now()
            jobs.append(job)
        return jobs

    def values(self) -> list[AnalysisJob]:
        return self.list(limit=1_000_000)

    def __len__(self) -> int:
        with self.store.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])

    def __contains__(self, job_id: object) -> bool:
        if not isinstance(job_id, str):
            return False
        with self.store.connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                is not None
            )

    # ------------------------------------------------------------------
    # Идемпотентность и попытки
    # ------------------------------------------------------------------
    def next_attempt_no(
        self,
        submission_hash: str,
        analyzer_id: str,
        config_hash: str,
    ) -> int:
        with self.store.connect() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(attempt_no), 0) + 1
                  FROM jobs
                 WHERE submission_hash = ?
                   AND analyzer_id = ?
                   AND config_hash = ?
                """,
                (submission_hash, analyzer_id, config_hash),
            ).fetchone()
        return int(row[0])

    def get_idempotency(self, key: str) -> Optional[IdempotencyRecord]:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT * FROM idempotency_keys WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return IdempotencyRecord(
            idempotency_key=row["idempotency_key"],
            job_id=row["job_id"],
            submission_hash=row["submission_hash"],
            analyzer_id=row["analyzer_id"],
            config_hash=row["config_hash"],
            created_at=row["created_at"],
        )

    def save_idempotency(
        self,
        *,
        key: str,
        job_id: str,
        submission_hash: str,
        analyzer_id: str,
        config_hash: str,
    ) -> IdempotencyRecord:
        record = IdempotencyRecord(
            idempotency_key=key,
            job_id=job_id,
            submission_hash=submission_hash,
            analyzer_id=analyzer_id,
            config_hash=config_hash,
            created_at=utc_now(),
        )
        with self.store.connect() as connection:
            try:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO idempotency_keys (
                            idempotency_key, job_id, submission_hash,
                            analyzer_id, config_hash, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.idempotency_key,
                            record.job_id,
                            record.submission_hash,
                            record.analyzer_id,
                            record.config_hash,
                            record.created_at,
                        ),
                    )
                return record
            except sqlite3.IntegrityError as exc:
                raise DuplicateIdempotencyKey(key) from exc

    # ------------------------------------------------------------------
    # Внутренняя сериализация
    # ------------------------------------------------------------------
    def _write_job(self, connection: sqlite3.Connection, job: AnalysisJob) -> None:
        connection.execute(
            """
            INSERT INTO jobs (
                job_id, submission_id, submission_hash, analyzer_id,
                config_hash, status, attempt_no, timeout_seconds,
                created_at, queued_at, started_at, finished_at, updated_at,
                error_code, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                submission_id = excluded.submission_id,
                submission_hash = excluded.submission_hash,
                analyzer_id = excluded.analyzer_id,
                config_hash = excluded.config_hash,
                status = excluded.status,
                attempt_no = excluded.attempt_no,
                timeout_seconds = excluded.timeout_seconds,
                queued_at = excluded.queued_at,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                updated_at = excluded.updated_at,
                error_code = excluded.error_code,
                error_message = excluded.error_message
            """,
            (
                job.job_id,
                job.submission_id,
                job.submission_hash,
                job.analyzer_id,
                job.config_hash,
                job.status.value,
                job.attempt_no,
                job.timeout_seconds,
                job.created_at,
                job.queued_at,
                job.started_at,
                job.finished_at,
                job.updated_at,
                job.error_code.value if job.error_code is not None else None,
                job.error_message,
            ),
        )

    def _write_report(
        self,
        connection: sqlite3.Connection,
        job: AnalysisJob,
    ) -> None:
        report = job.result
        if report is None:
            raise PersistenceError("Отчёт отсутствует")
        evidence = json.dumps(
            [asdict(item) for item in report.evidence],
            ensure_ascii=False,
        )
        connection.execute(
            """
            INSERT INTO analysis_reports (
                job_id, analyzer_id, analyzer_version, score, similarity,
                rubric_feedback, evidence_json, model_version, index_version,
                prompt_version, mlflow_run_id, config_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                analyzer_id = excluded.analyzer_id,
                analyzer_version = excluded.analyzer_version,
                score = excluded.score,
                similarity = excluded.similarity,
                rubric_feedback = excluded.rubric_feedback,
                evidence_json = excluded.evidence_json,
                model_version = excluded.model_version,
                index_version = excluded.index_version,
                prompt_version = excluded.prompt_version,
                mlflow_run_id = excluded.mlflow_run_id,
                config_hash = excluded.config_hash
            """,
            (
                job.job_id,
                report.analyzer_id,
                report.analyzer_version,
                report.score,
                report.similarity,
                report.rubric_feedback,
                evidence,
                report.model_version,
                report.index_version,
                report.prompt_version,
                report.mlflow_run_id,
                report.config_hash or job.config_hash,
                utc_now(),
            ),
        )

    def _reports_for(
        self,
        connection: sqlite3.Connection,
        job_ids: list[str],
    ) -> dict[str, AnalysisReport]:
        if not job_ids:
            return {}
        placeholders = ",".join("?" for _ in job_ids)
        rows = connection.execute(
            f"SELECT * FROM analysis_reports WHERE job_id IN ({placeholders})",
            job_ids,
        ).fetchall()
        return {row["job_id"]: _report_from_row(row) for row in rows}


# Совместимые имена для внешнего кода/тестов.
SQLiteStore = SqliteStore
SQLiteJobStore = SqliteJobStore


def submission_digest(submission: Submission) -> str:
    content = (submission.code or "") + (submission.text or "")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _submission_from_row(row: sqlite3.Row) -> Submission:
    declaration = row["ai_declaration"]
    return Submission(
        student_id=row["student_id"],
        submission_id=row["submission_id"],
        timestamp=row["timestamp"],
        code=row["code"],
        text=row["text"],
        ai_declaration=None if declaration is None else bool(declaration),
        assignment_id=row["assignment_id"] if "assignment_id" in row.keys() else "",
        raw_content_hash=row["raw_content_hash"],
    )


def _job_from_row(row: sqlite3.Row) -> AnalysisJob:
    error_code = row["error_code"]
    return AnalysisJob(
        job_id=row["job_id"],
        submission_id=row["submission_id"],
        submission_hash=row["submission_hash"],
        analyzer_id=row["analyzer_id"],
        config_hash=row["config_hash"],
        status=JobStatus(row["status"]),
        attempt_no=row["attempt_no"],
        timeout_seconds=row["timeout_seconds"],
        created_at=row["created_at"],
        queued_at=row["queued_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        updated_at=row["updated_at"],
        error_code=JobErrorCode(error_code) if error_code else None,
        error_message=row["error_message"],
    )


def _report_from_row(row: sqlite3.Row) -> AnalysisReport:
    evidence_payload = json.loads(row["evidence_json"] or "[]")
    return AnalysisReport(
        analyzer_id=row["analyzer_id"],
        analyzer_version=row["analyzer_version"],
        score=row["score"],
        similarity=row["similarity"],
        rubric_feedback=row["rubric_feedback"] or "",
        evidence=[Evidence(**item) for item in evidence_payload],
        model_version=row["model_version"],
        index_version=row["index_version"],
        prompt_version=row["prompt_version"],
        mlflow_run_id=row["mlflow_run_id"],
        config_hash=row["config_hash"],
    )

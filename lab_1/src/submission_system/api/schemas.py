"""Pydantic-схемы FastAPI-сервиса ЛР3.

Схемы ответа содержат только доменные поля: никаких API-ключей, токенов
провайдера или путей к секретам.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..jobs.models import AnalysisJob
from ..models import AnalysisReport, Evidence, Submission


class SubmissionCreate(BaseModel):
    student_id: str = Field(min_length=1)
    submission_id: str = Field(min_length=1)
    assignment_id: str = ""
    timestamp: str = Field(min_length=1)
    code: str = ""
    text: str = ""
    ai_declaration: Optional[bool] = None


class SubmissionResponse(BaseModel):
    student_id: str
    submission_id: str
    assignment_id: str
    timestamp: str
    code: str
    text: str
    ai_declaration: Optional[bool] = None
    raw_content_hash: str

    @classmethod
    def from_submission(cls, submission: Submission) -> "SubmissionResponse":
        return cls(
            student_id=submission.student_id,
            submission_id=submission.submission_id,
            assignment_id=submission.assignment_id,
            timestamp=submission.timestamp,
            code=submission.code or "",
            text=submission.text or "",
            ai_declaration=submission.ai_declaration,
            raw_content_hash=submission.raw_content_hash or "",
        )


class AnalysisStartRequest(BaseModel):
    submission_id: str = Field(min_length=1)
    analyzer_id: str = Field(min_length=1)
    idempotency_key: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    config_hash: Optional[str] = None
    submission_hash: Optional[str] = None
    timeout_seconds: Optional[float] = Field(default=None, ge=0)


class EvidenceResponse(BaseModel):
    line_number: int
    text: str
    type: str = "match"

    @classmethod
    def from_evidence(cls, evidence: Evidence) -> "EvidenceResponse":
        return cls(
            line_number=evidence.line_number,
            text=evidence.text,
            type=evidence.type,
        )


class ReportResponse(BaseModel):
    analyzer_id: str
    analyzer_version: str
    score: Optional[float] = None
    similarity: Optional[float] = None
    rubric_feedback: str = ""
    evidence: list[EvidenceResponse] = Field(default_factory=list)
    model_version: Optional[str] = None
    index_version: Optional[str] = None
    prompt_version: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    config_hash: Optional[str] = None

    @classmethod
    def from_report(cls, report: AnalysisReport) -> "ReportResponse":
        return cls(
            analyzer_id=report.analyzer_id,
            analyzer_version=report.analyzer_version,
            score=report.score,
            similarity=report.similarity,
            rubric_feedback=report.rubric_feedback,
            evidence=[
                EvidenceResponse.from_evidence(item) for item in report.evidence
            ],
            model_version=report.model_version,
            index_version=report.index_version,
            prompt_version=report.prompt_version,
            mlflow_run_id=report.mlflow_run_id,
            config_hash=report.config_hash,
        )


class JobResponse(BaseModel):
    job_id: str
    submission_id: str
    submission_hash: str
    analyzer_id: str
    config_hash: str
    status: str
    attempt_no: int
    timeout_seconds: float
    created_at: str
    queued_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    updated_at: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    has_result: bool = False
    result: Optional[ReportResponse] = None

    @classmethod
    def from_job(cls, job: AnalysisJob) -> "JobResponse":
        return cls(
            job_id=job.job_id,
            submission_id=job.submission_id,
            submission_hash=job.submission_hash,
            analyzer_id=job.analyzer_id,
            config_hash=job.config_hash,
            status=job.status.value,
            attempt_no=job.attempt_no,
            timeout_seconds=job.timeout_seconds,
            created_at=job.created_at,
            queued_at=job.queued_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            updated_at=job.updated_at,
            error_code=job.error_code.value if job.error_code else None,
            error_message=job.error_message,
            has_result=job.result is not None,
            result=ReportResponse.from_report(job.result) if job.result else None,
        )


class CancelResponse(BaseModel):
    job_id: str
    cancelled: bool
    status: str


class AnalyzerInfo(BaseModel):
    analyzer_id: str
    analyzer_version: str
    config_hash: str
    model_version: Optional[str] = None
    index_version: Optional[str] = None
    prompt_version: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    database: str
    recovered: dict[str, int] = Field(default_factory=dict)

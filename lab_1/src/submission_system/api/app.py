"""FastAPI-СЃРµСЂРІРёСЃ Р°РЅР°Р»РёР·Р°С‚РѕСЂР° Р›Р 3, СЌС‚Р°Рї Р‘.

Р—Р°РїСѓСЃРє:
    uvicorn submission_system.api.app:create_app --factory --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Response

from ..jobs import (
    AnalysisExecutor,
    BatchConfig,
    ExecutorConfig,
    GuardedModelAdapter,
    JobStatus,
    StubModelAdapter,
    SyncAnalyzerAdapter,
    UnavailableAnalysisAdapter,
)
from ..models import Assignment
from ..analyzers import LengthAnalyzer
from ..persistence import SqliteStore
from .auth import Actor, resolve_actor
from .schemas import (
    AnalysisStartRequest,
    AnalyzerInfo,
    CancelResponse,
    HealthResponse,
    JobResponse,
    ReportResponse,
    SubmissionCreate,
    SubmissionResponse,
)
from .service import AnalysisApiService, ServiceError
from .settings import Settings


def build_adapters(
    settings: Settings,
) -> dict:
    """Baseline Рё model-adapter Р·Р° РѕРґРЅРёРј РєРѕРЅС‚СЂР°РєС‚РѕРј AnalysisAdapter."""

    baseline = SyncAnalyzerAdapter(
        LengthAnalyzer(),
        Assignment(assignment_id="stage-b-baseline"),
        model_version="length-analyzer-1.0.0",
        mlflow_run_id=settings.mlflow_run_id,
        adapter_id="baseline",
    )
    if settings.model_mode == "unavailable":
        model = UnavailableAnalysisAdapter(
            analyzer_id="model",
            config_hash="model-unavailable",
            message="Р’РЅРµС€РЅСЏСЏ РјРѕРґРµР»СЊ РЅРµ РЅР°СЃС‚СЂРѕРµРЅР°; РґРѕСЃС‚СѓРїРµРЅ С‚РѕР»СЊРєРѕ baseline",
        )
    else:
        model = GuardedModelAdapter(
            StubModelAdapter(delay=settings.model_delay),
            BatchConfig(
                batch_size=settings.batch_size,
                concurrency_limit=settings.model_concurrency,
                call_timeout=settings.model_call_timeout,
            ),
            analyzer_id="model",
            index_version=settings.index_version,
            prompt_version=settings.prompt_version,
            mlflow_run_id=settings.mlflow_run_id,
            reported_model_version=settings.model_version,
        )
    return {"baseline": baseline, "model": model}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = SqliteStore(settings.db_path)
    recovered = store.initialize()

    adapters = build_adapters(settings)
    executor = AnalysisExecutor(
        adapters,
        ExecutorConfig(
            concurrency_limit=settings.executor_concurrency,
            queue_limit=settings.queue_limit,
            default_timeout=settings.default_timeout,
            executor_kind=settings.execution_model,  # type: ignore[arg-type]
        ),
        repository=store.submissions,
        job_store=store.jobs,
    )
    service = AnalysisApiService(store, executor, settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        executor.start()
        try:
            yield
        finally:
            await executor.shutdown(wait=True)
            store.close()

    app = FastAPI(
        title="Submission analysis service",
        version="3.1.0-stage-b",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.executor = executor
    app.state.service = service
    app.state.recovered = recovered

    def require_actor(
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> Actor:
        # Р›РѕРєР°Р»СЊРЅС‹Р№ СЂРµР¶РёРј Р±РµР· РєР»СЋС‡РµР№: РїРѕР»РЅС‹Р№ РґРѕСЃС‚СѓРї С‚РѕР»СЊРєРѕ РґР»СЏ СЂР°Р·СЂР°Р±РѕС‚РєРё.
        if not settings.api_key and not settings.student_keys and not settings.teacher_keys:
            return Actor(actor_id="anonymous", role="admin")
        provided = x_api_key
        if not provided and authorization:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() == "bearer":
                provided = token.strip()
        actor = resolve_actor(settings, provided)
        if actor is None:
            raise HTTPException(
                status_code=401,
                detail={"code": "unauthorized", "message": "РўСЂРµР±СѓРµС‚СЃСЏ API-РєР»СЋС‡"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return actor

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            database=settings.db_path,
            recovered=recovered,
        )

    router = APIRouter()

    def raise_service_error(error: ServiceError) -> NoReturn:
        detail: dict = {"code": error.code, "message": error.message}
        if error.job is not None:
            detail["job"] = JobResponse.from_job(error.job).model_dump()
        raise HTTPException(status_code=error.status_code, detail=detail)

    @router.post(
        "/submissions",
        response_model=SubmissionResponse,
        status_code=201,
    )
    def create_submission(
        payload: SubmissionCreate,
        response: Response,
        actor: Actor = Depends(require_actor),
    ) -> SubmissionResponse:
        try:
            submission, created = service.create_submission(payload, actor)
        except ServiceError as error:
            raise_service_error(error)
        response.status_code = 201 if created else 200
        return SubmissionResponse.from_submission(submission)

    @router.get("/submissions", response_model=list[SubmissionResponse])
    def list_submissions(
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        actor: Actor = Depends(require_actor),
    ) -> list[SubmissionResponse]:
        return [
            SubmissionResponse.from_submission(item)
            for item in service.list_submissions(
                actor=actor,
                limit=limit,
                offset=offset,
            )
        ]

    @router.get("/submissions/{submission_id}", response_model=SubmissionResponse)
    def get_submission(
        submission_id: str,
        actor: Actor = Depends(require_actor),
    ) -> SubmissionResponse:
        try:
            submission = service.get_submission(submission_id, actor)
        except ServiceError as error:
            raise_service_error(error)
        return SubmissionResponse.from_submission(submission)

    @router.post("/analysis-jobs", response_model=JobResponse, status_code=202)
    async def start_analysis(
        payload: AnalysisStartRequest,
        response: Response,
        idempotency_header: Annotated[
            str | None,
            Header(alias="Idempotency-Key"),
        ] = None,
        actor: Actor = Depends(require_actor),
    ) -> JobResponse:
        if not payload.idempotency_key and idempotency_header:
            payload = payload.model_copy(
                update={"idempotency_key": idempotency_header}
            )
        try:
            job, created = await service.start_analysis(payload, actor)
        except ServiceError as error:
            raise_service_error(error)
        response.status_code = 202 if created else 200
        response.headers["X-Idempotent-Replay"] = "false" if created else "true"
        return JobResponse.from_job(job)

    @router.get("/analysis-jobs", response_model=list[JobResponse])
    def list_jobs(
        status: str | None = None,
        submission_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        actor: Actor = Depends(require_actor),
    ) -> list[JobResponse]:
        parsed_status = None
        if status is not None:
            try:
                parsed_status = JobStatus(status)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "invalid_status", "message": status},
                ) from exc
        jobs = service.list_jobs(
            actor=actor,
            status=parsed_status,
            submission_id=submission_id,
            limit=limit,
            offset=offset,
        )
        return [JobResponse.from_job(job) for job in jobs]

    @router.get("/analysis-jobs/{job_id}", response_model=JobResponse)
    def get_job(
        job_id: str,
        actor: Actor = Depends(require_actor),
    ) -> JobResponse:
        try:
            job = service.get_job(job_id, actor)
        except ServiceError as error:
            raise_service_error(error)
        return JobResponse.from_job(job)

    @router.get("/analysis-jobs/{job_id}/report", response_model=ReportResponse)
    def get_report(
        job_id: str,
        actor: Actor = Depends(require_actor),
    ) -> ReportResponse:
        try:
            report = service.get_report(job_id, actor)
        except ServiceError as error:
            raise_service_error(error)
        return ReportResponse.from_report(report)

    @router.post(
        "/analysis-jobs/{job_id}/cancel",
        response_model=CancelResponse,
    )
    async def cancel_job(
        job_id: str,
        actor: Actor = Depends(require_actor),
    ) -> CancelResponse:
        try:
            job = await service.cancel(job_id, actor)
        except ServiceError as error:
            raise_service_error(error)
        return CancelResponse(
            job_id=job.job_id,
            cancelled=job.status is JobStatus.CANCELLED,
            status=job.status.value,
        )

    @router.get("/analyzers", response_model=list[AnalyzerInfo])
    def list_analyzers(
        actor: Actor = Depends(require_actor),
    ) -> list[AnalyzerInfo]:
        del actor
        return service.analyzers()

    app.include_router(router)
    # РљРѕСЂРѕС‚РєРёРµ РїСѓС‚Рё СѓРґРѕР±РЅС‹ РґР»СЏ Р»РѕРєР°Р»СЊРЅРѕРіРѕ Р·Р°РїСѓСЃРєР°, РІРµСЂСЃРёСЏ вЂ” РґР»СЏ РєР»РёРµРЅС‚РѕРІ.
    app.include_router(router, prefix="/api/v1")
    return app


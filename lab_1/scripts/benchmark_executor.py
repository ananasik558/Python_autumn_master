"""Бенчмарк модели исполнения исполнителя ЛР3.

Сравнивает последовательный запуск, пул потоков и пул процессов на одной и той
же CPU-работе. Результаты нужны для обоснования выбора: по умолчанию
исполнитель использует потоки, потому что на учебных объёмах накладные расходы
процессов выше, а управление задачами и тайм-ауты остаются одинаковыми.

Запуск:
    python scripts/benchmark_executor.py --jobs 8 --workers 4 --iterations 500000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from submission_system.jobs import (
    AnalysisExecutor,
    CpuBoundAnalyzer,
    ExecutorConfig,
    SyncAnalyzerAdapter,
)
from submission_system.models import Assignment, Submission

ROOT = Path(__file__).resolve().parents[1]


def _submission(index: int) -> Submission:
    return Submission(
        student_id=f"bench-{index}",
        submission_id=f"bench-sub-{index}",
        timestamp="2026-10-20T12:00:00+03:00",
        code="print('bench')\n",
        text="",
    )


async def _run_pool(kind: str, jobs: int, workers: int, iterations: int) -> dict[str, Any]:
    adapter = SyncAnalyzerAdapter(
        CpuBoundAnalyzer(iterations=iterations),
        Assignment(assignment_id="bench"),
    )
    executor = AnalysisExecutor(
        {"cpu": adapter},
        ExecutorConfig(
            concurrency_limit=workers,
            queue_limit=jobs,
            executor_kind=kind,  # type: ignore[arg-type]
        ),
    )
    started = time.perf_counter()
    results = [await executor.submit(_submission(index), "cpu") for index in range(jobs)]
    completed = await executor.wait_all(
        [result.job.job_id for result in results],
        timeout=120,
    )
    elapsed = time.perf_counter() - started
    stats = executor.stats()
    await executor.shutdown()
    if any(job.status.value != "succeeded" for job in completed):
        raise RuntimeError(
            "Не все задачи завершились успешно: "
            + ", ".join(f"{job.job_id}:{job.status.value}" for job in completed)
        )
    return {
        "executor_kind": kind,
        "jobs": jobs,
        "workers": workers,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "jobs_per_second": jobs / elapsed if elapsed else None,
        "stats": stats,
    }


def _run_sequential(jobs: int, iterations: int) -> dict[str, Any]:
    analyzer = CpuBoundAnalyzer(iterations=iterations)
    assignment = Assignment(assignment_id="bench")
    from submission_system.contracts import AnalysisContext

    context = AnalysisContext(assignment=assignment)
    started = time.perf_counter()
    for index in range(jobs):
        analyzer.analyze(_submission(index), context)
    elapsed = time.perf_counter() - started
    return {
        "executor_kind": "sequential",
        "jobs": jobs,
        "workers": 1,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "jobs_per_second": jobs / elapsed if elapsed else None,
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    cases: dict[str, list[dict[str, Any]]] = {
        "sequential": [],
        "thread": [],
        "process": [],
    }
    for _ in range(args.repeats):
        cases["sequential"].append(_run_sequential(args.jobs, args.iterations))
        for kind in ("thread", "process"):
            cases[kind].append(
                await _run_pool(kind, args.jobs, args.workers, args.iterations)
            )

    aggregated = {}
    for name, runs in cases.items():
        aggregated[name] = {
            "elapsed_seconds": _summary([run["elapsed_seconds"] for run in runs]),
            "jobs_per_second": _summary(
                [
                    run["jobs_per_second"]
                    for run in runs
                    if run["jobs_per_second"] is not None
                ]
            ),
            "runs": runs,
        }

    thread_median = aggregated["thread"]["elapsed_seconds"]["median"]
    process_median = aggregated["process"]["elapsed_seconds"]["median"]
    decision = (
        "Выбран пул потоков: на учебном объёме он не требует сериализации "
        "объектов между процессами и запускается без spawn-накладных расходов. "
        "Пул процессов оставлен как опция для будущих тяжёлых CPU-адаптеров; "
        "тайм-аут, отмена и ограничение конкурентности от модели исполнения не зависят."
    )
    return {
        "parameters": {
            "jobs": args.jobs,
            "workers": args.workers,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "corpus": "synthetic CPU load only; quality is not measured",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os_cpu_count(),
        },
        "results": aggregated,
        "comparison": {
            "thread_vs_process_ratio": (
                thread_median / process_median if process_median else None
            ),
            "decision": decision,
        },
    }


def os_cpu_count() -> int | None:
    import os

    return os.cpu_count()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=400_000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", default="artifacts/lab3/executor_benchmark.json")
    args = parser.parse_args()

    payload = asyncio.run(main_async(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("executor benchmark:")
    for name, result in payload["results"].items():
        median = result["elapsed_seconds"]["median"]
        throughput = result["jobs_per_second"]["median"]
        print(f"  {name:<12} {median:8.3f}s  {throughput:8.2f} jobs/s")
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

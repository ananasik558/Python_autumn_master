"""Профилирование потоковой обработки на синтетических данных.

Скрипт отвечает на вопрос «как растут время и пиковая память при увеличении
числа записей», а не «насколько хорошо методы различают плагиат». Синтетические
записи нужны только для нагрузки: метрики качества на них не считаются.

Запуск:
    python scripts/profile_pipeline.py --sizes 100,1000,5000 --repeats 3
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

from submission_system.similarity.corpus import iter_jsonl
from submission_system.similarity.tfidf import TfidfVectorizer, cosine_similarity
from submission_system.tokenizer import ngram_values, tokenize_code

ROOT = Path(__file__).resolve().parents[1]


def synthetic_code(index: int, seed: int = 0) -> str:
    """Детерминированный валидный Python-код с уникальными именами.

    ``seed`` сдвигает выбор шаблона кода, поэтому влияет на содержимое
    синтетических записей, но не на их количество. Это позволяет явно
    зафиксировать seed в параметрах замера.
    """

    variant = (index + seed) % 4
    if variant == 0:
        return (
            f"def sum_even_{index}(values):\n"
            f"    total_{index} = 0\n"
            f"    for value in values:\n"
            f"        if value % 2 == 0:\n"
            f"            total_{index} += value\n"
            f"    return total_{index}\n"
        )
    if variant == 1:
        return (
            "import re\n\n"
            f"def normalize_words_{index}(text):\n"
            f"    pattern_{index} = re.compile(r\"[a-zа-яё0-9]+\")\n"
            f"    return pattern_{index}.findall(text.lower())\n"
        )
    if variant == 2:
        return (
            f"def flatten_{index}(items):\n"
            f"    result_{index} = []\n"
            f"    for item in items:\n"
            f"        if isinstance(item, list):\n"
            f"            result_{index}.extend(flatten_{index}(item))\n"
            f"        else:\n"
            f"            result_{index}.append(item)\n"
            f"    return result_{index}\n"
        )
    return (
        f"LIMIT_{index} = {index % 7 + 1}\n\n"
        f"def count_long_{index}(words):\n"
        f"    total_{index} = 0\n"
        f"    for word in words:\n"
        f"        if len(word) > LIMIT_{index}:\n"
        f"            total_{index} += 1\n"
        f"    return total_{index}\n"
    )


def write_synthetic(path: Path, records: int, seed: int = 0) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(records):
            record = {
                "student_id": f"u{index:06d}",
                "submission_id": f"s{index:06d}",
                "assignment_id": f"hw_{index % 4}",
                "code": synthetic_code(index, seed),
                "text": "",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path.stat().st_size


def prepare_vectorizer(
    path: Path,
    max_features: int,
    max_df: float,
    sample_size: int = 20,
    seed: int = 0,
) -> tuple[TfidfVectorizer, dict[str, float], str]:
    """Обучает словарь на фиксированной выборке до начала замера."""

    documents: list[str] = []
    for index, record in enumerate(iter_jsonl(path)):
        if index >= sample_size:
            break
        documents.append(str(record.get("code") or ""))
    vectorizer = TfidfVectorizer(max_features=max_features, max_df=max_df)
    vectorizer.fit(documents)
    reference_code = synthetic_code(0, seed)
    reference_vector = vectorizer.transform(reference_code)
    return vectorizer, reference_vector, reference_code


def process_file(
    path: Path,
    vectorizer: TfidfVectorizer,
    reference_vector: dict[str, float],
    ngram_size: int,
) -> dict[str, Any]:
    """Потоковая обработка: чтение, токенизация, n-граммы, cosine."""

    records = 0
    token_total = 0
    ngram_total = 0
    scored = 0
    score_sum = 0.0

    for record in iter_jsonl(path):
        code = str(record.get("code") or "")
        features = tokenize_code(code)
        token_total += len(features.tokens)
        ngram_total += len(ngram_values(features.tokens, ngram_size))
        vector = vectorizer.transform(code)
        score = cosine_similarity(vector, reference_vector)
        if score is not None:
            scored += 1
            score_sum += score
        records += 1

    return {
        "records": records,
        "token_total": token_total,
        "ngram_total": ngram_total,
        "scored_records": scored,
        "mean_reference_cosine": (score_sum / scored) if scored else None,
    }


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "runs": values,
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def measure(
    operation: Callable[[], dict[str, Any]],
    repeats: int,
    *,
    memory: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timed: list[float] = []
    peaks: list[float] = []
    last_result: dict[str, Any] = {}
    for _ in range(repeats):
        gc.collect()
        if memory:
            tracemalloc.start()
        started = time.perf_counter()
        last_result = operation()
        elapsed = time.perf_counter() - started
        timed.append(elapsed)
        if memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks.append(peak / 1024.0)
    return _stats(timed), _stats(peaks) if memory else {}, last_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="100,1000,5000")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="artifacts/lab2/profiling.json")
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--max-df", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    if args.repeats < 3:
        raise SystemExit("По заданию нужно не менее 3 повторов каждого замера")
    if any(size <= 0 for size in sizes):
        raise SystemExit("Размеры должны быть положительными")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    measurements: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="similarity-profile-") as temp_dir:
        temp_path = Path(temp_dir) / "synthetic.jsonl"
        for size in sizes:
            file_bytes = write_synthetic(temp_path, size, args.seed)
            vectorizer, reference_vector, reference_code = prepare_vectorizer(
                temp_path,
                max_features=args.max_features,
                max_df=args.max_df,
                seed=args.seed,
            )

            def operation() -> dict[str, Any]:
                return process_file(
                    temp_path,
                    vectorizer,
                    reference_vector,
                    ngram_size=args.ngram_size,
                )

            wall, _, result = measure(operation, args.repeats, memory=False)
            _, peak, _ = measure(operation, args.repeats, memory=True)
            throughput = (
                size / wall["median"] if wall["median"] > 0 else None
            )
            measurements.append(
                {
                    "records": size,
                    "file_bytes": file_bytes,
                    "wall_seconds": wall,
                    "peak_kib": peak,
                    "throughput_records_per_second_median": throughput,
                    "result": result,
                }
            )
            print(
                f"records={size:>6}  "
                f"wall median={wall['median']:.3f}s  "
                f"peak median={peak['median']:.1f} KiB  "
                f"throughput={throughput:.0f} rec/s"
            )

    payload = {
        "parameters": {
            "sizes": sizes,
            "repeats": args.repeats,
            "ngram_size": args.ngram_size,
            "max_features": args.max_features,
            "max_df": args.max_df,
            "seed": args.seed,
            "vectorizer_fit_documents": 20,
            "synthetic_only": True,
            "purpose": "load profiling, not quality evaluation",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "measurements": measurements,
        "notes": [
            "Wall time и пиковая память измеряются отдельными сериями: "
            "tracemalloc замедляет выполнение.",
            "Пайплайн потоковый: записи не накапливаются в списке, поэтому "
            "память должна расти слабо.",
            "Синтетические данные не используются для precision/recall/F1.",
            "seed фиксирует выбор шаблона синтетического кода; размеры и "
            "повторы задаются параметрами.",
        ],
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

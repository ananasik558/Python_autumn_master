"""Замер памяти: потоковое чтение против полного индекса.

Запуск:
    python scripts/benchmark_memory.py --corpus data/corpus --output artifacts/lab2/memory.json

Скрипт показывает, что генератор сам по себе не гарантирует константную
память: если следующим шагом строится полный индекс или пары удерживают всё
задание, пик памяти растёт линейно.
"""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
import tracemalloc
from pathlib import Path
from typing import Any, Callable

from submission_system.similarity.corpus import iter_jsonl
from submission_system.similarity.pairs import iter_candidate_pairs_streaming


def _peak_kib(operation: Callable[[], Any]) -> tuple[int, float]:
    gc.collect()
    tracemalloc.start()
    operation()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, current / 1024.0


def _write_repeated(source: Path, destination: Path, copies: int) -> int:
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            for copy_index in range(copies):
                item = dict(record)
                item["submission_id"] = f"{record['submission_id']}-copy{copy_index}"
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
    return count


def benchmark_file(path: Path) -> dict[str, Any]:
    def stream() -> int:
        return sum(1 for _ in iter_jsonl(path))

    def full_load() -> int:
        return len(list(iter_jsonl(path)))

    def index() -> int:
        return len({record["submission_id"]: record for record in iter_jsonl(path)})

    def streaming_pairs() -> int:
        return sum(1 for _ in iter_candidate_pairs_streaming(path))

    records = stream()
    stream_peak = _peak_kib(stream)[0]
    full_peak, _ = _peak_kib(full_load)
    index_peak, _ = _peak_kib(index)
    if records <= 600:
        pairs_peak = _peak_kib(streaming_pairs)[0]
        pairs_kib: float | None = round(pairs_peak / 1024.0, 2)
    else:
        pairs_kib = None
    return {
        "records": records,
        "stream_peak_kib": round(stream_peak / 1024.0, 2),
        "full_load_peak_kib": round(full_peak / 1024.0, 2),
        "full_index_peak_kib": round(index_peak / 1024.0, 2),
        "streaming_pairs_peak_kib": pairs_kib,
        "stream_peak_bytes_per_record": round(stream_peak / max(records, 1), 2),
        "full_index_bytes_per_record": round(index_peak / max(records, 1), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--output", default="artifacts/lab2/memory.json")
    parser.add_argument("--copies", default="1,5,10")
    args = parser.parse_args()

    source = Path(args.corpus) / "submissions.jsonl"
    copies = [int(value) for value in args.copies.split(",") if value.strip()]
    results: dict[str, Any] = {"source": str(source), "measurements": []}

    with tempfile.TemporaryDirectory(prefix="similarity-memory-") as temp_dir:
        temp_path = Path(temp_dir) / "submissions.jsonl"
        for count in copies:
            records = _write_repeated(source, temp_path, count)
            measurement = benchmark_file(temp_path)
            measurement["requested_copies"] = count
            measurement["written_records"] = records
            results["measurements"].append(measurement)
            print(
                f"records={measurement['records']:>6}  "
                f"stream={measurement['stream_peak_kib']:>8.1f} KiB  "
                f"full={measurement['full_load_peak_kib']:>8.1f} KiB  "
                f"index={measurement['full_index_peak_kib']:>8.1f} KiB  "
                f"pairs={measurement['streaming_pairs_peak_kib']:>8.1f} KiB"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Порождение пар для сравнения.

Правила:
* сравниваются только работы одного задания;
* пара одного и того же студента (self-match) не создаётся;
* повторные версии одного студента остаются отдельными документами и
  сравниваются с работами других студентов, но не друг с другом.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .corpus import CorpusSubmission, iter_corpus_submissions


@dataclass
class PairGenerationStats:
    """Диагностика, которую полезно сохранять рядом с результатами."""

    submissions_seen: int = 0
    assignments_seen: int = 0
    cross_student_pairs: int = 0
    same_student_skipped: int = 0
    repeated_student_submissions: int = 0
    students_with_repeats: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "submissions_seen": self.submissions_seen,
            "assignments_seen": self.assignments_seen,
            "cross_student_pairs": self.cross_student_pairs,
            "same_student_skipped": self.same_student_skipped,
            "repeated_student_submissions": self.repeated_student_submissions,
            "students_with_repeats": self.students_with_repeats,
        }


def _pairs_from_group(
    group: list[CorpusSubmission],
    stats: PairGenerationStats,
) -> Iterator[tuple[CorpusSubmission, CorpusSubmission]]:
    counts = Counter(item.student_id for item in group)
    repeats = sum(count - 1 for count in counts.values() if count > 1)
    stats.repeated_student_submissions += repeats
    stats.students_with_repeats += sum(1 for count in counts.values() if count > 1)

    for left_index, left in enumerate(group):
        for right in group[left_index + 1:]:
            if left.student_id == right.student_id:
                stats.same_student_skipped += 1
                continue
            stats.cross_student_pairs += 1
            yield left, right


def iter_candidate_pairs(
    submissions: Iterable[CorpusSubmission],
    stats: PairGenerationStats | None = None,
) -> Iterator[tuple[CorpusSubmission, CorpusSubmission]]:
    """Пары из произвольного iterable; группировка по заданию в памяти."""

    stats = stats or PairGenerationStats()
    grouped: dict[str, list[CorpusSubmission]] = defaultdict(list)
    for submission in submissions:
        grouped[submission.assignment_id].append(submission)
        stats.submissions_seen += 1

    stats.assignments_seen = len(grouped)
    for assignment_id in sorted(grouped):
        yield from _pairs_from_group(grouped[assignment_id], stats)


def iter_candidate_pairs_streaming(
    path: str | Path,
    stats: PairGenerationStats | None = None,
) -> Iterator[tuple[CorpusSubmission, CorpusSubmission]]:
    """Потоковые пары при условии, что JSONL отсортирован по assignment_id.

    В памяти удерживается только текущее задание. Это не делает полный индекс
    константным по памяти: если все сдачи лежат в одном задании, окно всё
    равно вырастет до размера этого задания.
    """

    stats = stats or PairGenerationStats()
    submissions = iter_corpus_submissions(path)
    for assignment_id, group_iter in itertools.groupby(
        submissions,
        key=lambda item: item.assignment_id,
    ):
        group = list(group_iter)
        stats.assignments_seen += 1
        stats.submissions_seen += len(group)
        yield from _pairs_from_group(group, stats)


def pair_key(left: CorpusSubmission, right: CorpusSubmission) -> tuple[str, str]:
    """Детерминированный ключ пары независимо от порядка сторон."""

    ids = sorted((left.submission_id, right.submission_id))
    return ids[0], ids[1]

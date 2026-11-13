"""Учебный корпус ЛР2: модели, потоковое чтение и запись JSONL.

Корпус специально размечен происхождением (``derivation_relation``), а не
вердиктом «плагиат/не плагиат». Это позволяет оценивать методы сходства, не
подменяя решение о нарушении числом косинуса.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ..models import Assignment, Submission


DERIVATION_RELATIONS = (
    "exact_copy",
    "formatting",
    "renaming",
    "template_only",
    "independent",
)


class CorpusFormatError(ValueError):
    """Ошибка формата строки корпуса с указанием номера строки."""


@dataclass(frozen=True)
class CorpusSubmission:
    """Одна сдача в учебном корпусе.

    ``family_id`` связывает варианты одной задачи. ``split`` задаётся на
    уровне family, чтобы производные работы не попали одновременно в tuning
    и evaluation.
    """

    student_id: str
    submission_id: str
    assignment_id: str
    code: str = ""
    text: str = ""
    timestamp: str = "2026-01-01T00:00:00+00:00"
    family_id: str = ""
    split: str = "tuning"
    variant_role: str = "base"

    def to_submission(self) -> Submission:
        return Submission(
            student_id=self.student_id,
            submission_id=self.submission_id,
            timestamp=self.timestamp,
            code=self.code,
            text=self.text,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "submission_id": self.submission_id,
            "assignment_id": self.assignment_id,
            "code": self.code,
            "text": self.text,
            "timestamp": self.timestamp,
            "family_id": self.family_id,
            "split": self.split,
            "variant_role": self.variant_role,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorpusSubmission":
        required = ("student_id", "submission_id", "assignment_id")
        missing = [name for name in required if not data.get(name)]
        if missing:
            raise CorpusFormatError(
                f"В сдаче отсутствуют обязательные поля: {', '.join(missing)}"
            )
        return cls(
            student_id=str(data["student_id"]),
            submission_id=str(data["submission_id"]),
            assignment_id=str(data["assignment_id"]),
            code=str(data.get("code") or ""),
            text=str(data.get("text") or ""),
            timestamp=str(data.get("timestamp") or "2026-01-01T00:00:00+00:00"),
            family_id=str(data.get("family_id") or ""),
            split=str(data.get("split") or "tuning"),
            variant_role=str(data.get("variant_role") or "base"),
        )


@dataclass(frozen=True)
class CorpusPair:
    """Размеченная пара сдач с происхождением варианта."""

    pair_id: str
    assignment_id: str
    family_id: str
    split: str
    derivation_relation: str
    template_only: bool
    left: CorpusSubmission
    right: CorpusSubmission
    notes: str = ""

    def __post_init__(self) -> None:
        if self.derivation_relation not in DERIVATION_RELATIONS:
            raise CorpusFormatError(
                f"Неизвестное происхождение пары: {self.derivation_relation}"
            )
        if self.left.student_id == self.right.student_id:
            raise CorpusFormatError(
                f"Пара {self.pair_id} содержит self-match одного студента"
            )
        if self.left.assignment_id != self.assignment_id:
            raise CorpusFormatError(f"Левая сдача пары {self.pair_id} из другого задания")
        if self.right.assignment_id != self.assignment_id:
            raise CorpusFormatError(f"Правая сдача пары {self.pair_id} из другого задания")

    @property
    def is_derived_reference(self) -> bool:
        """Справочная метка для эксперимента, а не вердикт о нарушении."""

        return self.derivation_relation != "independent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "assignment_id": self.assignment_id,
            "family_id": self.family_id,
            "split": self.split,
            "derivation_relation": self.derivation_relation,
            "template_only": self.template_only,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorpusPair":
        for required in ("pair_id", "assignment_id", "family_id", "split", "left", "right"):
            if not data.get(required):
                raise CorpusFormatError(f"В паре отсутствует обязательное поле: {required}")
        return cls(
            pair_id=str(data["pair_id"]),
            assignment_id=str(data["assignment_id"]),
            family_id=str(data["family_id"]),
            split=str(data["split"]),
            derivation_relation=str(data.get("derivation_relation") or "independent"),
            template_only=bool(data.get("template_only")),
            left=CorpusSubmission.from_dict(data["left"]),
            right=CorpusSubmission.from_dict(data["right"]),
            notes=str(data.get("notes") or ""),
        )


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Лениво читает JSONL: в памяти находится одна строка, а не весь файл."""

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusFormatError(
                    f"{path}:{line_number}: повреждённый JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise CorpusFormatError(
                    f"{path}:{line_number}: ожидался JSON-объект"
                )
            yield value


def iter_corpus_pairs(path: str | Path) -> Iterator[CorpusPair]:
    for record in iter_jsonl(path):
        yield CorpusPair.from_dict(record)


def iter_corpus_submissions(path: str | Path) -> Iterator[CorpusSubmission]:
    for record in iter_jsonl(path):
        yield CorpusSubmission.from_dict(record)


def load_assignments(path: str | Path) -> dict[str, Assignment]:
    """Загружает небольшой справочник заданий (его допустимо держать в памяти)."""

    assignments: dict[str, Assignment] = {}
    for record in iter_jsonl(path):
        assignment_id = record.get("assignment_id")
        if not assignment_id:
            raise CorpusFormatError("В задании отсутствует assignment_id")
        assignments[str(assignment_id)] = Assignment(
            assignment_id=str(assignment_id),
            allowed_template=str(record.get("allowed_template") or ""),
            requirements=dict(record.get("requirements") or {}),
        )
    return assignments


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False))
            handle.write("\n")
            count += 1
    return count


@dataclass(frozen=True)
class CorpusSummary:
    pairs: int
    submissions: int
    template_pairs: int
    families: int
    tuning_pairs: int
    evaluation_pairs: int
    relations: Mapping[str, int] = field(default_factory=dict)


def summarize_corpus(pairs_path: str | Path, submissions_path: str | Path) -> CorpusSummary:
    relations: dict[str, int] = {}
    families: set[str] = set()
    pair_count = 0
    template_pairs = 0
    tuning_pairs = 0
    evaluation_pairs = 0
    for pair in iter_corpus_pairs(pairs_path):
        pair_count += 1
        families.add(pair.family_id)
        relations[pair.derivation_relation] = relations.get(pair.derivation_relation, 0) + 1
        template_pairs += int(pair.template_only)
        tuning_pairs += int(pair.split == "tuning")
        evaluation_pairs += int(pair.split == "evaluation")

    submission_count = sum(1 for _ in iter_corpus_submissions(submissions_path))
    return CorpusSummary(
        pairs=pair_count,
        submissions=submission_count,
        template_pairs=template_pairs,
        families=len(families),
        tuning_pairs=tuning_pairs,
        evaluation_pairs=evaluation_pairs,
        relations=relations,
    )

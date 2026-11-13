"""Разбор ошибок методов сходства.

Модуль не подбирает порог и ничего не меняет в модели: он берёт уже полученные
на tuning пороги и показывает, какие пары были отправлены на проверку
ошибочно или пропущены. Отдельно выделяются пары с разрешённым шаблоном,
потому что общий шаблон — типичный источник ложных обвинений.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .evaluation import ScoredPair


@dataclass(frozen=True)
class ErrorCase:
    """Одна ошибочная классификация с человекочитаемым объяснением."""

    method_id: str
    pair_id: str
    split: str
    derivation_relation: str
    error_type: str
    template_only: bool
    score: Optional[float]
    threshold: float
    expected_label: int
    predicted_label: int
    cause: str
    template_note: Optional[str] = None
    evidence_preview: tuple[str, ...] = ()
    source_lines: Optional[dict[str, tuple[int, ...]]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "pair_id": self.pair_id,
            "split": self.split,
            "derivation_relation": self.derivation_relation,
            "error_type": self.error_type,
            "template_only": self.template_only,
            "score": self.score,
            "threshold": self.threshold,
            "expected": "derived" if self.expected_label else "independent",
            "predicted": "review_required" if self.predicted_label else "not_required",
            "cause": self.cause,
            "template_note": self.template_note,
            "evidence_preview": list(self.evidence_preview),
            "source_lines": (
                {key: list(value) for key, value in self.source_lines.items()}
                if self.source_lines
                else None
            ),
        }


def _cause_for(
    record: ScoredPair,
    error_type: str,
) -> str:
    if record.template_only and error_type == "false_negative":
        return (
            "Строки разрешённого шаблона исключены из признаков, а собственной "
            "логики в коротком решении оказалось мало. Метод пропустил пару, "
            "хотя работы используют одну и ту же заготовку."
        )
    if record.template_only and error_type == "false_positive":
        return (
            "Пара использует разрешённый шаблон; строки шаблона исключены, но "
            "оставшаяся лексика задачи всё равно повысила score. Это повод "
            "показать шаблон отдельно, а не обвинять студента."
        )
    if record.derivation_relation == "renaming":
        return (
            "Переименование идентификаторов разрушает совпадение токенов: "
            "логика та же, но n-граммы и словарь TF-IDF меняются."
        )
    if record.derivation_relation == "independent":
        return (
            "Независимые решения одной задачи используют общую лексику, "
            "имена стандартных функций и похожую структуру, поэтому score "
            "поднимается выше порога."
        )
    if record.derivation_relation in {"exact_copy", "formatting"}:
        return (
            "Сходство ниже порога при почти одинаковом коде; вероятно, "
            "порог выбран на другом наборе и не переносится на эту пару."
        )
    return "Ошибка классификации на выбранном пороге."


def _evidence_from_comparison(
    comparison: Mapping[str, Any],
    method_id: str,
) -> tuple[tuple[str, ...], Optional[dict[str, tuple[int, ...]]]]:
    method_payload = (comparison.get("methods") or {}).get(method_id) or {}
    evidence = method_payload.get("evidence") or []
    preview = tuple(str(item.get("preview", "")) for item in evidence[:3] if item)
    lines: Optional[dict[str, tuple[int, ...]]] = None
    if evidence:
        first = evidence[0]
        lines = {
            "left": tuple(first.get("left_lines") or ()),
            "right": tuple(first.get("right_lines") or ()),
        }
    return preview, lines


def collect_errors(
    method_evaluations: Iterable[Any],
    tuning_records: Mapping[str, Iterable[ScoredPair]],
    evaluation_records: Mapping[str, Iterable[ScoredPair]],
    comparisons: Iterable[Mapping[str, Any]] = (),
) -> list[ErrorCase]:
    """Собирает FP/FN для каждого метода на tuning и evaluation."""

    comparison_by_pair = {item["pair_id"]: item for item in comparisons}
    threshold_by_method = {
        item.method_id: item.threshold for item in method_evaluations
    }
    errors: list[ErrorCase] = []

    for method_id, threshold in threshold_by_method.items():
        for split, records in (
            ("tuning", tuning_records.get(method_id, ())),
            ("evaluation", evaluation_records.get(method_id, ())),
        ):
            for record in records:
                if record.score is None:
                    # Неопределённый результат — отдельная категория, не FP/FN.
                    predicted = 0
                else:
                    predicted = 1 if record.score >= threshold else 0
                if predicted == record.label:
                    continue

                error_type = "false_positive" if predicted else "false_negative"
                comparison = comparison_by_pair.get(record.pair_id, {})
                preview, lines = _evidence_from_comparison(comparison, method_id)
                template_note = None
                method_payload = (comparison.get("methods") or {}).get(method_id) or {}
                template_note = method_payload.get("template_note")

                errors.append(
                    ErrorCase(
                        method_id=method_id,
                        pair_id=record.pair_id,
                        split=split,
                        derivation_relation=record.derivation_relation,
                        error_type=error_type,
                        template_only=record.template_only,
                        score=record.score,
                        threshold=threshold,
                        expected_label=record.label,
                        predicted_label=predicted,
                        cause=_cause_for(record, error_type),
                        template_note=template_note,
                        evidence_preview=preview,
                        source_lines=lines,
                    )
                )
    return errors


def _priority(case: ErrorCase) -> tuple[int, int, int, float, str, str]:
    """Сначала шаблонные ошибки, затем evaluation, FN и уверенные ошибки."""

    template_priority = 0 if case.template_only else 1
    split_priority = 0 if case.split == "evaluation" else 1
    type_priority = 0 if case.error_type == "false_negative" else 1
    # Для FP/FN берём сначала самые далёкие от порога: они показательнее
    # пограничных случаев.
    score = case.score if case.score is not None else 0.0
    confidence = -score
    return (
        template_priority,
        type_priority,
        split_priority,
        confidence,
        case.pair_id,
        case.method_id,
    )


def summarize_errors(
    errors: Iterable[ErrorCase],
    max_cases: int = 6,
) -> dict[str, Any]:
    """Формирует отчёт: все ошибки, агрегаты и репрезентативные случаи."""

    error_list = list(errors)
    by_type: dict[str, int] = {}
    by_method: dict[str, int] = {}
    by_relation: dict[str, int] = {}
    for case in error_list:
        by_type[case.error_type] = by_type.get(case.error_type, 0) + 1
        by_method[case.method_id] = by_method.get(case.method_id, 0) + 1
        by_relation[case.derivation_relation] = (
            by_relation.get(case.derivation_relation, 0) + 1
        )

    # Одна и та же пара может ошибочно классифицироваться двумя методами;
    # для репрезентативных случаев оставляем по одному объяснению на пару.
    # Дополнительно гарантируем шаблонный случай и хотя бы по одному примеру
    # от каждого метода.
    ordered = sorted(error_list, key=_priority)
    chosen: list[ErrorCase] = []
    seen_pairs: set[str] = set()

    def try_add(case: ErrorCase) -> bool:
        if len(chosen) >= max_cases:
            return False
        if case.pair_id in seen_pairs:
            return False
        chosen.append(case)
        seen_pairs.add(case.pair_id)
        return True

    for case in ordered:
        if case.template_only:
            try_add(case)
            break
    for method_id in sorted(by_method):
        for case in ordered:
            if case.method_id == method_id and case.pair_id not in seen_pairs:
                try_add(case)
                break
    for case in ordered:
        if len(chosen) >= max_cases:
            break
        try_add(case)

    template_errors = [case for case in error_list if case.template_only]
    return {
        "total_errors": len(error_list),
        "by_type": dict(sorted(by_type.items())),
        "by_method": dict(sorted(by_method.items())),
        "by_relation": dict(sorted(by_relation.items())),
        "template_error_count": len(template_errors),
        "template_case_required": True,
        "template_case_included": any(case.template_only for case in chosen),
        "cases": [case.to_dict() for case in chosen],
        "all_errors": [case.to_dict() for case in sorted(error_list, key=_priority)],
        "caveat": (
            "Разбор ошибок не меняет порог и не является вердиктом о нарушении. "
            "Ошибки на tuning показаны для диагностики; итоговые метрики считаются "
            "на evaluation."
        ),
    }


def analyze_errors(
    method_evaluations: Iterable[Any],
    tuning_records: Mapping[str, Iterable[ScoredPair]],
    evaluation_records: Mapping[str, Iterable[ScoredPair]],
    comparisons: Iterable[Mapping[str, Any]] = (),
    max_cases: int = 6,
) -> dict[str, Any]:
    errors = collect_errors(
        method_evaluations,
        tuning_records,
        evaluation_records,
        comparisons,
    )
    return summarize_errors(errors, max_cases=max_cases)


def save_error_analysis(
    analysis: Mapping[str, Any],
    path: str | Path,
) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


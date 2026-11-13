"""Экспериментальная оценка методов сходства.

Метрики считаются относительно происхождения пары (derived/independent).
Порог выбирается только на tuning-парах и затем применяется к evaluation-
парам без переобучения. Это важная часть воспроизводимого сравнения.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence


@dataclass(frozen=True)
class ScoredPair:
    pair_id: str
    score: Optional[float]
    label: int
    split: str
    derivation_relation: str
    template_only: bool = False

    @property
    def label_name(self) -> str:
        return "derived" if self.label else "independent"


@dataclass(frozen=True)
class ClassificationMetrics:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float

    @property
    def support(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def review_rate(self) -> float:
        """Доля работ, отправленных на ручной разбор (predicted positive)."""

        return (self.tp + self.fp) / self.support if self.support else 0.0

    @property
    def confusion_matrix(self) -> dict[str, int]:
        """Матрица ошибок в явном виде.

        Строки — справочная метка, столбцы — предсказание.
        """

        return {
            "true_positive": self.tp,
            "false_positive": self.fp,
            "true_negative": self.tn,
            "false_negative": self.fn,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "confusion_matrix": self.confusion_matrix,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "support": self.support,
            "review_rate": self.review_rate,
        }


@dataclass(frozen=True)
class MethodEvaluation:
    method_id: str
    threshold: float
    tuning: ClassificationMetrics
    evaluation: ClassificationMetrics
    tuning_auc: Optional[float]
    evaluation_auc: Optional[float]
    skipped_tuning: int = 0
    skipped_evaluation: int = 0

    @property
    def tuning_total(self) -> int:
        return self.tuning.support + self.skipped_tuning

    @property
    def evaluation_total(self) -> int:
        return self.evaluation.support + self.skipped_evaluation

    @property
    def tuning_undefined_rate(self) -> float:
        return (
            self.skipped_tuning / self.tuning_total if self.tuning_total else 0.0
        )

    @property
    def evaluation_undefined_rate(self) -> float:
        return (
            self.skipped_evaluation / self.evaluation_total
            if self.evaluation_total
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "threshold_selected_on_tuning": self.threshold,
            "tuning": self.tuning.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "tuning_auc": self.tuning_auc,
            "evaluation_auc": self.evaluation_auc,
            "skipped_tuning": self.skipped_tuning,
            "skipped_evaluation": self.skipped_evaluation,
            "tuning_total": self.tuning_total,
            "evaluation_total": self.evaluation_total,
            "tuning_undefined_rate": self.tuning_undefined_rate,
            "evaluation_undefined_rate": self.evaluation_undefined_rate,
        }


def review_required(score: Optional[float], threshold: float) -> Optional[bool]:
    """Решение «отправить на ручной разбор» по score и выбранному порогу.

    ``None`` означает, что score неопределён (например, в работе нет значимых
    токенов). Это не «нет сходства», поэтому решение не подменяется на ``False``.
    """

    if score is None:
        return None
    return score >= threshold


def relation_to_label(relation: str) -> int:
    """Справочная метка для эксперимента, а не вердикт о нарушении."""

    return 0 if relation == "independent" else 1


def classification_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> ClassificationMetrics:
    if len(scores) != len(labels):
        raise ValueError("Число score и labels должно совпадать")

    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        predicted = 1 if score >= threshold else 0
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and label:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    return ClassificationMetrics(
        threshold=threshold,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
    )


def _candidate_thresholds(scores: Sequence[float]) -> list[float]:
    """Пороги между соседними score.

    Вариант «все positive» (threshold=0) исключён: он часто выигрывает по F1
    на маленьком tuning-наборе, но не является рабочим правилом отбора.
    """

    unique = sorted(set(scores))
    if not unique:
        return [0.0]
    if len(unique) == 1:
        return [unique[0]]
    return [(left + right) / 2.0 for left, right in zip(unique, unique[1:])]


def select_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
) -> float:
    """Выбирает порог по F1 только на переданном наборе.

    При равенстве F1 предпочитается более высокая precision, затем более
    низкий порог (он даёт больше кандидатов на ручную проверку).
    """

    if not scores:
        return 0.5

    best: tuple[float, float, float] | None = None
    best_threshold = 0.5
    for threshold in _candidate_thresholds(scores):
        metrics = classification_metrics(scores, labels, threshold)
        key = (metrics.f1, metrics.precision, -threshold)
        if best is None or key > best:
            best = key
            best_threshold = threshold
    return best_threshold


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """ROC-AUC по рангам, с корректной обработкой одинаковых score."""

    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and scores[ordered[end]] == scores[ordered[index]]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position]] = average_rank
        index = end

    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, labels) if label
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _valid_scores(records: Iterable[ScoredPair]) -> tuple[list[float], list[int], int]:
    scores: list[float] = []
    labels: list[int] = []
    skipped = 0
    for record in records:
        if record.score is None:
            skipped += 1
            continue
        scores.append(record.score)
        labels.append(record.label)
    return scores, labels, skipped


def evaluate_method(
    method_id: str,
    tuning_records: Iterable[ScoredPair],
    evaluation_records: Iterable[ScoredPair],
) -> MethodEvaluation:
    tuning_scores, tuning_labels, skipped_tuning = _valid_scores(tuning_records)
    evaluation_scores, evaluation_labels, skipped_evaluation = _valid_scores(
        evaluation_records
    )

    threshold = select_threshold(tuning_scores, tuning_labels)
    tuning_metrics = classification_metrics(
        tuning_scores, tuning_labels, threshold
    )
    evaluation_metrics = classification_metrics(
        evaluation_scores, evaluation_labels, threshold
    )

    return MethodEvaluation(
        method_id=method_id,
        threshold=threshold,
        tuning=tuning_metrics,
        evaluation=evaluation_metrics,
        tuning_auc=roc_auc(tuning_scores, tuning_labels),
        evaluation_auc=roc_auc(evaluation_scores, evaluation_labels),
        skipped_tuning=skipped_tuning,
        skipped_evaluation=skipped_evaluation,
    )


def mean_score_by_relation(
    records: Iterable[ScoredPair],
) -> dict[str, dict[str, float | int]]:
    """Средний score и число пар по происхождению (для диагностики)."""

    grouped: dict[str, list[float]] = {}
    for record in records:
        if record.score is None:
            continue
        grouped.setdefault(record.derivation_relation, []).append(record.score)
    return {
        relation: {
            "mean_score": sum(values) / len(values),
            "count": len(values),
        }
        for relation, values in sorted(grouped.items())
    }

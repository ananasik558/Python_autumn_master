"""Простой классификатор ``review_required`` на признаках пары.

Реализация на стандартной библиотеке: логистическая регрессия обучается
полным батчем на стандартизованных признаках. Это не production-модель, а
честный эксперимент на маленьком учебном наборе: 18 tuning-пар и 22
evaluation-пары. Поэтому отдельно измеряется разброс метрик при bootstrap и
grouped-CV по семействам, а выводы формулируются как предварительные.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .corpus import write_jsonl
from .evaluation import (
    MethodEvaluation,
    ScoredPair,
    classification_metrics,
    evaluate_method,
    review_required,
    roc_auc,
    select_threshold,
)

FEATURE_NAMES: tuple[str, ...] = (
    "token_jaccard",
    "tfidf_cosine",
    "method_disagreement",
    "ngram_overlap_fraction_of_min",
    "token_count_ratio",
    "char_count_ratio",
    "template_matched_lines",
    "missing_score_count",
)

METHOD_ID = "logistic_regression_pair_features"


def _raw_features(row: Mapping[str, Any]) -> dict[str, Optional[float]]:
    features = row.get("features") or {}
    jaccard = features.get("token_jaccard")
    cosine = features.get("tfidf_cosine")
    missing = int(jaccard is None) + int(cosine is None)
    return {
        "token_jaccard": jaccard,
        "tfidf_cosine": cosine,
        "method_disagreement": features.get("method_disagreement"),
        "ngram_overlap_fraction_of_min": features.get(
            "ngram_overlap_fraction_of_min"
        ),
        "token_count_ratio": features.get("token_count_ratio"),
        "char_count_ratio": features.get("char_count_ratio"),
        "template_matched_lines": features.get("template_matched_lines"),
        "missing_score_count": float(missing),
    }


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def fit_imputer(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Медианы признаков считаются только на обучающем наборе."""

    collected: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}
    for row in rows:
        raw = _raw_features(row)
        for name in FEATURE_NAMES:
            value = raw.get(name)
            if value is not None:
                collected[name].append(float(value))
    return {name: _median(values) for name, values in collected.items()}


def build_matrix(
    rows: Iterable[Mapping[str, Any]],
    medians: Mapping[str, float],
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        raw = _raw_features(row)
        vector = [
            float(raw[name]) if raw.get(name) is not None else float(medians[name])
            for name in FEATURE_NAMES
        ]
        matrix.append(vector)
    return matrix


def labels_of(rows: Iterable[Mapping[str, Any]]) -> list[int]:
    return [1 if row.get("reference_label") == "derived" else 0 for row in rows]


@dataclass
class StandardScaler:
    means: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)

    def fit(self, matrix: Sequence[Sequence[float]]) -> "StandardScaler":
        if not matrix:
            self.means = []
            self.scales = []
            return self
        width = len(matrix[0])
        self.means = [
            sum(row[index] for row in matrix) / len(matrix)
            for index in range(width)
        ]
        self.scales = []
        for index in range(width):
            variance = (
                sum((row[index] - self.means[index]) ** 2 for row in matrix)
                / len(matrix)
            )
            scale = math.sqrt(variance)
            self.scales.append(scale if scale > 1e-12 else 1.0)
        return self

    def transform(self, matrix: Sequence[Sequence[float]]) -> list[list[float]]:
        return [
            [
                (value - self.means[index]) / self.scales[index]
                for index, value in enumerate(row)
            ]
            for row in matrix
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            name: {"mean": self.means[index], "scale": self.scales[index]}
            for index, name in enumerate(FEATURE_NAMES)
        }


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


@dataclass
class LogisticRegression:
    learning_rate: float = 0.2
    epochs: int = 600
    l2: float = 0.05
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0

    def fit(
        self,
        matrix: Sequence[Sequence[float]],
        labels: Sequence[int],
    ) -> "LogisticRegression":
        if not matrix:
            self.weights = []
            self.bias = 0.0
            return self
        width = len(matrix[0])
        weights = [0.0] * width
        bias = 0.0
        count = float(len(matrix))

        for _ in range(self.epochs):
            gradient = [0.0] * width
            gradient_bias = 0.0
            for row, label in zip(matrix, labels):
                score = bias + sum(
                    weight * value for weight, value in zip(weights, row)
                )
                error = _sigmoid(score) - label
                for index, value in enumerate(row):
                    gradient[index] += error * value
                gradient_bias += error
            for index in range(width):
                weights[index] -= self.learning_rate * (
                    gradient[index] / count + self.l2 * weights[index]
                )
            bias -= self.learning_rate * gradient_bias / count

        self.weights = weights
        self.bias = bias
        return self

    def predict_proba(self, matrix: Sequence[Sequence[float]]) -> list[float]:
        return [
            _sigmoid(self.bias + sum(
                weight * value for weight, value in zip(self.weights, row)
            ))
            for row in matrix
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
            "bias": self.bias,
            "weights": {
                name: self.weights[index]
                for index, name in enumerate(FEATURE_NAMES)
                if index < len(self.weights)
            },
        }


@dataclass
class ModelBundle:
    imputer: dict[str, float]
    scaler: StandardScaler
    model: LogisticRegression

    def predict(self, rows: Iterable[Mapping[str, Any]]) -> list[float]:
        row_list = list(rows)
        matrix = build_matrix(row_list, self.imputer)
        return validate_model_scores(self.model.predict_proba(self.scaler.transform(matrix)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "imputer_medians": dict(self.imputer),
            "scaler": self.scaler.to_dict(),
            "logistic_regression": self.model.to_dict(),
        }


def train_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    learning_rate: float = 0.2,
    epochs: int = 600,
    l2: float = 0.05,
) -> ModelBundle:
    rows = list(rows)
    imputer = fit_imputer(rows)
    matrix = build_matrix(rows, imputer)
    scaler = StandardScaler().fit(matrix)
    standardized = scaler.transform(matrix)
    labels = labels_of(rows)
    model = LogisticRegression(
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    ).fit(standardized, labels)
    return ModelBundle(imputer=imputer, scaler=scaler, model=model)


def validate_model_scores(scores: Sequence[float]) -> list[float]:
    """Проверяет, что модель вернула конечные вероятности в диапазоне [0, 1].

    Невалидный результат (NaN, бесконечность, значение вне диапазона или не
    число) — это ошибка модели, а не «нулевое сходство». Поэтому функция
    выбрасывает исключение, а не подменяет значение. Вызывающий код
    (`run_ml_experiment`, bootstrap, CV) обязан передать результат проверки
    дальше, чтобы испорченный score не попал в метрики и артефакты.
    """

    validated: list[float] = []
    for index, value in enumerate(scores):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"model score #{index} is not numeric: {value!r}"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(
                f"model score #{index} is not finite: {number!r}"
            )
        if not 0.0 <= number <= 1.0:
            raise ValueError(
                f"model score #{index} is outside [0, 1]: {number!r}"
            )
        validated.append(number)
    return validated


def _records_from_rows(
    rows: Iterable[Mapping[str, Any]],
    scores: Sequence[float],
) -> list[ScoredPair]:
    records: list[ScoredPair] = []
    scores = validate_model_scores(scores)
    for row, score in zip(rows, scores):
        records.append(
            ScoredPair(
                pair_id=str(row["pair_id"]),
                score=score,
                label=1 if row.get("reference_label") == "derived" else 0,
                split=str(row.get("split", "")),
                derivation_relation=str(row.get("derivation_relation", "")),
                template_only=bool(row.get("template_only")),
            )
        )
    return records


def _metric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def bootstrap_instability(
    tuning_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    seeds: int = 20,
    learning_rate: float = 0.2,
    epochs: int = 600,
    l2: float = 0.05,
) -> dict[str, Any]:
    """Оценивает разброс evaluation-метрик при bootstrap tuning-набора."""

    tuning_rows = list(tuning_rows)
    evaluation_rows = list(evaluation_rows)
    evaluation_labels = labels_of(evaluation_rows)
    runs: list[dict[str, Any]] = []

    for seed in range(seeds):
        rng = random.Random(seed)
        sample = [
            tuning_rows[rng.randrange(len(tuning_rows))]
            for _ in range(len(tuning_rows))
        ]
        if len(set(labels_of(sample))) < 2:
            continue
        bundle = train_model(
            sample,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
        )
        tuning_scores = validate_model_scores(bundle.predict(sample))
        threshold = select_threshold(tuning_scores, labels_of(sample))
        evaluation_scores = validate_model_scores(bundle.predict(evaluation_rows))
        metrics = classification_metrics(
            evaluation_scores,
            evaluation_labels,
            threshold,
        )
        runs.append(
            {
                "seed": seed,
                "threshold": threshold,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "review_rate": metrics.review_rate,
            }
        )

    return {
        "method": "bootstrap tuning set with replacement",
        "seeds_requested": seeds,
        "runs_completed": len(runs),
        "precision": _metric_summary([run["precision"] for run in runs]),
        "recall": _metric_summary([run["recall"] for run in runs]),
        "f1": _metric_summary([run["f1"] for run in runs]),
        "review_rate": _metric_summary([run["review_rate"] for run in runs]),
        "runs": runs,
    }


def grouped_cv_instability(
    tuning_rows: Sequence[Mapping[str, Any]],
    *,
    learning_rate: float = 0.2,
    epochs: int = 600,
    l2: float = 0.05,
) -> dict[str, Any]:
    """Leave-one-family-out CV на tuning: проверка переноса между семьями."""

    tuning_rows = list(tuning_rows)
    families = sorted({str(row.get("family_id", "")) for row in tuning_rows})
    folds: list[dict[str, Any]] = []

    for family in families:
        train_rows = [
            row for row in tuning_rows if str(row.get("family_id", "")) != family
        ]
        test_rows = [
            row for row in tuning_rows if str(row.get("family_id", "")) == family
        ]
        if not test_rows or len(set(labels_of(train_rows))) < 2:
            continue
        bundle = train_model(
            train_rows,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
        )
        train_scores = validate_model_scores(bundle.predict(train_rows))
        threshold = select_threshold(train_scores, labels_of(train_rows))
        test_scores = validate_model_scores(bundle.predict(test_rows))
        metrics = classification_metrics(test_scores, labels_of(test_rows), threshold)
        folds.append(
            {
                "family_id": family,
                "support": metrics.support,
                "threshold": threshold,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
            }
        )

    return {
        "method": "leave-one-family-out on tuning",
        "folds_completed": len(folds),
        "precision": _metric_summary([fold["precision"] for fold in folds]),
        "recall": _metric_summary([fold["recall"] for fold in folds]),
        "f1": _metric_summary([fold["f1"] for fold in folds]),
        "folds": folds,
    }


def _comparison_row(
    evaluation: MethodEvaluation,
) -> dict[str, Any]:
    return {
        "method_id": evaluation.method_id,
        "threshold_selected_on_tuning": evaluation.threshold,
        "evaluation_precision": evaluation.evaluation.precision,
        "evaluation_recall": evaluation.evaluation.recall,
        "evaluation_f1": evaluation.evaluation.f1,
        "evaluation_auc": evaluation.evaluation_auc,
        "review_rate": evaluation.evaluation.review_rate,
        "evaluation_undefined_rate": evaluation.evaluation_undefined_rate,
        "confusion_matrix": evaluation.evaluation.confusion_matrix,
    }


def run_ml_experiment(
    feature_rows: Sequence[Mapping[str, Any]],
    threshold_evaluations: Sequence[MethodEvaluation] = (),
    *,
    seeds: int = 20,
    learning_rate: float = 0.2,
    epochs: int = 600,
    l2: float = 0.05,
) -> dict[str, Any]:
    """Обучает классификатор на tuning и сравнивает с threshold- baseline."""

    tuning_rows = [row for row in feature_rows if row.get("split") == "tuning"]
    evaluation_rows = [
        row for row in feature_rows if row.get("split") == "evaluation"
    ]
    if not tuning_rows or not evaluation_rows:
        raise ValueError("Нужны и tuning-, и evaluation-признаки")

    bundle = train_model(
        tuning_rows,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    tuning_scores = validate_model_scores(bundle.predict(tuning_rows))
    evaluation_scores = validate_model_scores(bundle.predict(evaluation_rows))

    tuning_records = _records_from_rows(tuning_rows, tuning_scores)
    evaluation_records = _records_from_rows(evaluation_rows, evaluation_scores)
    evaluation = evaluate_method(
        METHOD_ID,
        tuning_records,
        evaluation_records,
    )

    predictions: list[dict[str, Any]] = []
    for split, rows, scores in (
        ("tuning", tuning_rows, tuning_scores),
        ("evaluation", evaluation_rows, evaluation_scores),
    ):
        for row, score in zip(rows, scores):
            predictions.append(
                {
                    "method_id": METHOD_ID,
                    "pair_id": row["pair_id"],
                    "family_id": row.get("family_id", ""),
                    "split": split,
                    "derivation_relation": row.get("derivation_relation", ""),
                    "template_only": bool(row.get("template_only")),
                    "score": score,
                    "threshold": evaluation.threshold,
                    "label": 1 if row.get("reference_label") == "derived" else 0,
                    "review_required": review_required(score, evaluation.threshold),
                }
            )

    bootstrap = bootstrap_instability(
        tuning_rows,
        evaluation_rows,
        seeds=seeds,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )
    grouped_cv = grouped_cv_instability(
        tuning_rows,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
    )

    comparison = [_comparison_row(item) for item in threshold_evaluations]
    comparison.append(_comparison_row(evaluation))

    families_tuning = sorted(
        {str(row.get("family_id", "")) for row in tuning_rows}
    )
    families_evaluation = sorted(
        {str(row.get("family_id", "")) for row in evaluation_rows}
    )
    split_artifact = {
        "features": list(FEATURE_NAMES),
        "tuning": {
            "pairs": [row["pair_id"] for row in tuning_rows],
            "families": families_tuning,
            "positives": sum(labels_of(tuning_rows)),
            "negatives": len(tuning_rows) - sum(labels_of(tuning_rows)),
        },
        "evaluation": {
            "pairs": [row["pair_id"] for row in evaluation_rows],
            "families": families_evaluation,
            "positives": sum(labels_of(evaluation_rows)),
            "negatives": len(evaluation_rows) - sum(labels_of(evaluation_rows)),
        },
        "split_unit": "family_id",
        "families_disjoint": set(families_tuning).isdisjoint(families_evaluation),
    }

    results = {
        "method_id": METHOD_ID,
        "features": list(FEATURE_NAMES),
        "dataset": {
            "tuning_pairs": len(tuning_rows),
            "evaluation_pairs": len(evaluation_rows),
            "tuning_families": len(families_tuning),
            "evaluation_families": len(families_evaluation),
        },
        "hyperparameters": {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "l2": l2,
            "threshold_selection": "F1 on tuning only",
            "bootstrap_seeds": seeds,
        },
        "model": bundle.to_dict(),
        "threshold_selected_on_tuning": evaluation.threshold,
        "tuning": evaluation.tuning.to_dict(),
        "evaluation": evaluation.evaluation.to_dict(),
        "evaluation_auc": evaluation.evaluation_auc,
        "comparison": comparison,
        "instability": {
            "bootstrap": bootstrap,
            "grouped_cv_by_family": grouped_cv,
        },
        "split": split_artifact,
        "predictions": predictions,
        "caveat": (
            "Набор мал (18 tuning / 22 evaluation пар). Метрики и разброс "
            "показывают поведение эксперимента, а не готовность к реальным "
            "решениям. Порог и гиперпараметры выбирались только на tuning."
        ),
    }
    return results


def save_ml_results(
    results: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Сохраняет модель, предсказания и описание split."""

    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = dict(results)
    predictions = payload.pop("predictions", [])
    split = payload.pop("split", {})

    results_path = output_dir / "ml_results.json"
    predictions_path = output_dir / "ml_predictions.jsonl"
    split_path = output_dir / "ml_split.json"

    results_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(predictions_path, predictions)
    split_path.write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "results": results_path,
        "predictions": predictions_path,
        "split": split_path,
    }


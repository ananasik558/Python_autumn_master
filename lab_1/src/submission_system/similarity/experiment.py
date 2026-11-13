"""Воспроизводимый эксперимент ЛР2.

Порядок строго фиксирован:
1. читаем корпус потоково;
2. обучаем TF-IDF только на tuning-сдачах;
3. сравниваем пары baseline и TF-IDF (для evaluation — без повторного fit);
4. выбираем порог каждого метода на tuning;
5. считаем метрики на evaluation и сохраняем отчёт.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..models import Assignment
from .baseline import TokenNGramBaseline
from .corpus import (
    CorpusSubmission,
    iter_corpus_pairs,
    iter_corpus_submissions,
    load_assignments,
    summarize_corpus,
    write_jsonl,
)
from .evaluation import (
    MethodEvaluation,
    ScoredPair,
    evaluate_method,
    mean_score_by_relation,
    relation_to_label,
    review_required,
)
from .features import build_pair_features
from .tfidf import TfidfSimilarityMethod, TfidfVectorizer


REPORT_VERSION = "2.2.0"


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _digest_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def build_report_metadata(
    corpus_dir: Path,
    assignments: Mapping[str, Assignment],
    *,
    ngram_size: int,
    max_features: int,
    max_df: float,
) -> dict[str, Any]:
    """Версия отчёта и отпечатки конфигурации/корпуса/шаблонов.

    ``config_hash`` меняется при изменении параметров или разрешённых шаблонов.
    ``template_fingerprint`` позволяет показать на защите, что добавление
    шаблона — это изменение конфигурации, а не «переобучение на результате».
    """

    template_payload = "\n".join(
        f"{assignment_id}\0{assignment.allowed_template}"
        for assignment_id, assignment in sorted(assignments.items())
    )
    config_payload = json.dumps(
        {
            "ngram_size": ngram_size,
            "max_features": max_features,
            "max_df": max_df,
            "fit_split": "tuning",
            "templates": template_payload,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    corpus_paths = [
        corpus_dir / "assignments.jsonl",
        corpus_dir / "submissions.jsonl",
        corpus_dir / "pairs.jsonl",
    ]
    return {
        "report_version": REPORT_VERSION,
        "config_hash": _digest_text(config_payload),
        "corpus_fingerprint": _digest_files(corpus_paths),
        "template_fingerprint": _digest_text(template_payload),
        "parameters": {
            "ngram_size": ngram_size,
            "max_features": max_features,
            "max_df": max_df,
            "fit_split": "tuning",
        },
    }



@dataclass
class ExperimentResult:
    summary: dict[str, Any]
    method_evaluations: list[MethodEvaluation]
    comparisons: list[dict[str, Any]]
    feature_rows: list[dict[str, Any]]
    tuning_records: dict[str, list[ScoredPair]] = field(default_factory=dict)
    evaluation_records: dict[str, list[ScoredPair]] = field(default_factory=dict)
    threshold_predictions: list[dict[str, Any]] = field(default_factory=list)

    def selected_method(self) -> MethodEvaluation:
        """Лучший метод по F1 на tuning; evaluation используется только для отчёта."""

        return max(
            self.method_evaluations,
            key=lambda item: (item.tuning.f1, item.tuning.precision, item.method_id),
        )

    def to_summary_dict(self) -> dict[str, Any]:
        selected = self.selected_method()
        return {
            **self.summary,
            "methods": [item.to_dict() for item in self.method_evaluations],
            "selected_by_tuning": selected.method_id,
            "selected_evaluation": selected.evaluation.to_dict(),
            "caveat": (
                "derivation_relation — происхождение примера, а не вердикт о плагиате. "
                "Корпус учебный; метрики не являются оценкой точности на реальных студентах."
            ),
        }


def _index_submissions(path: Path) -> dict[str, CorpusSubmission]:
    """Полный индекс нужен для доступа к паре по submission_id.

    Это тот самый случай, когда потоковое чтение не делает память константной:
    индекс хранит все сдачи. Для больших корпусов нужен внешний key-value store.
    """

    return {item.submission_id: item for item in iter_corpus_submissions(path)}


def run_experiment(
    corpus_dir: str | Path,
    ngram_size: int = 3,
    max_features: int = 5000,
    max_df: float = 0.9,
    *,
    assignments_override: Mapping[str, Assignment] | None = None,
) -> ExperimentResult:
    corpus_dir = Path(corpus_dir)
    assignments = (
        dict(assignments_override)
        if assignments_override is not None
        else load_assignments(corpus_dir / "assignments.jsonl")
    )
    submissions = _index_submissions(corpus_dir / "submissions.jsonl")
    corpus_summary = summarize_corpus(
        corpus_dir / "pairs.jsonl", corpus_dir / "submissions.jsonl"
    )

    tuning_submissions = [item for item in submissions.values() if item.split == "tuning"]
    evaluation_submissions = [
        item for item in submissions.values() if item.split == "evaluation"
    ]
    if not tuning_submissions:
        raise ValueError("В корпусе нет tuning-сдач: TF-IDF не на чем обучать")

    vectorizer = TfidfVectorizer(max_features=max_features, max_df=max_df)
    tuning_documents = [
        (item.code, assignments[item.assignment_id].allowed_template)
        for item in tuning_submissions
    ]
    vectorizer.fit(tuning_documents)
    tfidf_method = TfidfSimilarityMethod(vectorizer)
    tfidf_method.fitted_on = tuple(item.submission_id for item in tuning_submissions)
    baseline_method = TokenNGramBaseline(n=ngram_size)

    comparisons: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    tuning_records: dict[str, list[ScoredPair]] = {
        baseline_method.method_id: [],
        tfidf_method.method_id: [],
    }
    evaluation_records: dict[str, list[ScoredPair]] = {
        baseline_method.method_id: [],
        tfidf_method.method_id: [],
    }

    for pair in iter_corpus_pairs(corpus_dir / "pairs.jsonl"):
        assignment = assignments[pair.assignment_id]
        left = submissions[pair.left.submission_id]
        right = submissions[pair.right.submission_id]
        baseline_result = baseline_method.compare(left, right, assignment.allowed_template)
        tfidf_result = tfidf_method.compare(left, right, assignment.allowed_template)

        records = {
            baseline_method.method_id: baseline_result,
            tfidf_method.method_id: tfidf_result,
        }
        target = tuning_records if pair.split == "tuning" else evaluation_records
        for method_id, result in records.items():
            target[method_id].append(
                ScoredPair(
                    pair_id=pair.pair_id,
                    score=result.score,
                    label=relation_to_label(pair.derivation_relation),
                    split=pair.split,
                    derivation_relation=pair.derivation_relation,
                    template_only=pair.template_only,
                )
            )

        feature_rows.append(
            build_pair_features(pair, baseline_result, tfidf_result, assignments)
        )
        comparisons.append(
            {
                "pair_id": pair.pair_id,
                "assignment_id": pair.assignment_id,
                "family_id": pair.family_id,
                "split": pair.split,
                "derivation_relation": pair.derivation_relation,
                "template_only": pair.template_only,
                "methods": {
                    baseline_method.method_id: baseline_result.to_dict(),
                    tfidf_method.method_id: tfidf_result.to_dict(),
                },
            }
        )

    evaluations = [
        evaluate_method(
            method_id,
            tuning_records[method_id],
            evaluation_records[method_id],
        )
        for method_id in (baseline_method.method_id, tfidf_method.method_id)
    ]
    threshold_by_method = {item.method_id: item.threshold for item in evaluations}

    threshold_predictions: list[dict[str, Any]] = []
    for method_id in (baseline_method.method_id, tfidf_method.method_id):
        for split, records in (
            ("tuning", tuning_records[method_id]),
            ("evaluation", evaluation_records[method_id]),
        ):
            threshold = threshold_by_method[method_id]
            for record in records:
                threshold_predictions.append(
                    {
                        "method_id": method_id,
                        "pair_id": record.pair_id,
                        "split": split,
                        "derivation_relation": record.derivation_relation,
                        "template_only": record.template_only,
                        "score": record.score,
                        "threshold": threshold,
                        "label": record.label,
                        "review_required": review_required(record.score, threshold),
                    }
                )

    summary = {
        "report": build_report_metadata(
            corpus_dir,
            assignments,
            ngram_size=ngram_size,
            max_features=max_features,
            max_df=max_df,
        ),
        "corpus": {
            "pairs": corpus_summary.pairs,
            "submissions": corpus_summary.submissions,
            "template_pairs": corpus_summary.template_pairs,
            "families": corpus_summary.families,
            "tuning_pairs": corpus_summary.tuning_pairs,
            "evaluation_pairs": corpus_summary.evaluation_pairs,
            "relations": dict(corpus_summary.relations),
        },
        "tfidf": {
            "fit_split": "tuning",
            "fit_documents": vectorizer.document_count,
            "vocabulary_size": len(vectorizer.vocabulary),
            "evaluation_submissions_transformed_without_fit": len(evaluation_submissions),
            "max_features": max_features,
            "max_df": max_df,
        },
        "baseline": {"ngram_size": ngram_size},
        "score_by_relation": {
            method_id: {
                "tuning": mean_score_by_relation(tuning_records[method_id]),
                "evaluation": mean_score_by_relation(evaluation_records[method_id]),
            }
            for method_id in (baseline_method.method_id, tfidf_method.method_id)
        },
    }

    return ExperimentResult(
        summary=summary,
        method_evaluations=evaluations,
        comparisons=comparisons,
        feature_rows=feature_rows,
        tuning_records=tuning_records,
        evaluation_records=evaluation_records,
        threshold_predictions=threshold_predictions,
    )


def save_experiment(result: ExperimentResult, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    comparisons_path = output_dir / "pair_comparisons.jsonl"
    features_path = output_dir / "pair_features.jsonl"
    predictions_path = output_dir / "threshold_predictions.jsonl"

    import json

    summary_path.write_text(
        json.dumps(result.to_summary_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(comparisons_path, result.comparisons)
    write_jsonl(features_path, result.feature_rows)
    write_jsonl(predictions_path, result.threshold_predictions)
    return {
        "summary": summary_path,
        "comparisons": comparisons_path,
        "features": features_path,
        "predictions": predictions_path,
    }

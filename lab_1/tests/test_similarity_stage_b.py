from pathlib import Path

import pytest

from submission_system.similarity.error_analysis import analyze_errors
from submission_system.similarity.evaluation import (
    ScoredPair,
    evaluate_method,
    review_required,
)
from submission_system.similarity.experiment import run_experiment
from submission_system.similarity.ml import (
    METHOD_ID,
    build_matrix,
    fit_imputer,
    run_ml_experiment,
)

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


def _record(pair_id: str, score, label: int, relation: str, split: str) -> ScoredPair:
    return ScoredPair(
        pair_id=pair_id,
        score=score,
        label=label,
        split=split,
        derivation_relation=relation,
        template_only=relation == "template_only",
    )


def test_review_required_keeps_undefined_as_none():
    assert review_required(0.5, 0.5) is True
    assert review_required(0.49, 0.5) is False
    assert review_required(None, 0.5) is None


def test_method_evaluation_exports_confusion_matrix_and_undefined_share():
    tuning = [
        _record("p1", 0.9, 1, "exact_copy", "tuning"),
        _record("p2", 0.1, 0, "independent", "tuning"),
        _record("p3", None, 0, "independent", "tuning"),
    ]
    evaluation = [
        _record("p4", 0.8, 1, "formatting", "evaluation"),
        _record("p5", 0.7, 0, "independent", "evaluation"),
        _record("p6", None, 1, "template_only", "evaluation"),
    ]
    result = evaluate_method("m", tuning, evaluation)
    payload = result.to_dict()

    assert payload["evaluation"]["confusion_matrix"] == {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 0,
        "false_negative": 0,
    }
    assert payload["evaluation"]["review_rate"] == pytest.approx(1.0)
    assert payload["skipped_evaluation"] == 1
    assert payload["evaluation_total"] == 3
    assert payload["evaluation_undefined_rate"] == pytest.approx(1 / 3)
    assert payload["tuning_undefined_rate"] == pytest.approx(1 / 3)


def test_error_analysis_has_at_least_three_cases_and_template_case():
    result = run_experiment(CORPUS)
    analysis = analyze_errors(
        result.method_evaluations,
        result.tuning_records,
        result.evaluation_records,
        result.comparisons,
        max_cases=5,
    )

    assert analysis["total_errors"] >= 3
    assert len(analysis["cases"]) >= 3
    assert analysis["template_error_count"] >= 1
    assert analysis["template_case_included"] is True
    template_cases = [case for case in analysis["cases"] if case["template_only"]]
    assert template_cases
    assert template_cases[0]["template_note"]
    assert template_cases[0]["error_type"] in {"false_negative", "false_positive"}
    for case in analysis["cases"]:
        assert case["cause"]
        assert case["pair_id"]


def test_ml_experiment_is_comparable_with_threshold_baseline():
    result = run_experiment(CORPUS)
    ml = run_ml_experiment(
        result.feature_rows,
        result.method_evaluations,
        seeds=5,
        epochs=300,
    )

    assert ml["method_id"] == METHOD_ID
    assert ml["dataset"]["tuning_pairs"] == 18
    assert ml["dataset"]["evaluation_pairs"] == 22
    assert "template_matched_lines" in ml["features"]
    assert "template_only" not in ml["features"]
    assert "template_shared" not in ml["features"]
    assert ml["threshold_selected_on_tuning"] >= 0.0
    assert 0.0 <= ml["evaluation"]["precision"] <= 1.0
    assert 0.0 <= ml["evaluation"]["recall"] <= 1.0
    assert 0.0 <= ml["evaluation"]["f1"] <= 1.0
    assert ml["evaluation"]["support"] == 22
    assert ml["split"]["families_disjoint"] is True
    assert {row["method_id"] for row in ml["comparison"]} >= {
        METHOD_ID,
        "token_3gram_jaccard",
    }
    assert ml["instability"]["bootstrap"]["runs_completed"] == 5
    assert ml["instability"]["grouped_cv_by_family"]["folds_completed"] >= 3
    assert all(0.0 <= row["score"] <= 1.0 for row in ml["predictions"])
    assert all(row["review_required"] in {True, False} for row in ml["predictions"])


def test_feature_imputer_uses_training_values_only():
    rows = [
        {
            "pair_id": "a",
            "reference_label": "derived",
            "features": {"token_jaccard": 1.0, "tfidf_cosine": None},
        }
    ]
    medians = fit_imputer(rows)
    matrix = build_matrix(rows, medians)
    assert len(matrix[0]) > 0
    # None заменяется медианой, а не попадает в матрицу как NaN/None.
    assert all(isinstance(value, float) for value in matrix[0])

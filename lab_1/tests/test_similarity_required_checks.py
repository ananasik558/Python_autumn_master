"""Обязательные проверки ЛР2 из чек-листа.

Каждый тест соответствует пункту списка: одинаковый код, переименованные
переменные, независимые решения, разрешённый шаблон, пустой ввод, SyntaxError,
self-match, повторная версия студента, отсутствие утечки семейств между
tuning/evaluation, невалидный результат модели и воспроизводимость
сохранённого эксперимента.
"""

from __future__ import annotations

import ast
import statistics
from pathlib import Path

import pytest

from submission_system.analysis_service import (
    AnalysisService,
    validate_analysis_report,
)
from submission_system.models import AnalysisReport, Assignment, Submission
from submission_system.similarity.baseline import TokenNGramBaseline
from submission_system.similarity.corpus import (
    CorpusSubmission,
    iter_corpus_pairs,
)
from submission_system.similarity.evaluation import review_required
from submission_system.similarity.experiment import (
    REPORT_VERSION,
    run_experiment,
    save_experiment,
)
from submission_system.similarity.ml import (
    run_ml_experiment,
    save_ml_results,
    validate_model_scores,
)
from submission_system.similarity.mlflow_logging import (
    build_metrics,
    build_params,
    build_tags,
)
from submission_system.similarity.template_demo import run_template_change_demo
from submission_system.similarity.pairs import (
    PairGenerationStats,
    iter_candidate_pairs,
)
from submission_system.similarity.retrieval import TfidfIndex
from submission_system.similarity.tfidf import TfidfSimilarityMethod, TfidfVectorizer
from submission_system.tokenizer import tokenize_code
from submission_system.sources import InMemorySource

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"
METHODS = ("token_3gram_jaccard", "tfidf_cosine")


@pytest.fixture(scope="module")
def experiment_result():
    return run_experiment(CORPUS)


@pytest.fixture(scope="module")
def ml_result(experiment_result):
    return run_ml_experiment(
        experiment_result.feature_rows,
        experiment_result.method_evaluations,
        seeds=5,
        epochs=300,
    )


def _thresholds(result) -> dict[str, float]:
    return {item.method_id: item.threshold for item in result.method_evaluations}


def _by_pair(result) -> dict[str, dict]:
    return {item["pair_id"]: item for item in result.comparisons}


def test_required_same_code_gets_maximum_score(experiment_result):
    exact = [
        item
        for item in experiment_result.comparisons
        if item["derivation_relation"] == "exact_copy"
    ]
    assert exact
    thresholds = _thresholds(experiment_result)

    for item in exact:
        for method_id in METHODS:
            score = item["methods"][method_id]["score"]
            assert score == pytest.approx(1.0)
            assert review_required(score, thresholds[method_id]) is True


def test_required_renamed_variables_are_detected_but_not_exact(experiment_result):
    renamed = [
        item
        for item in experiment_result.comparisons
        if item["derivation_relation"] == "renaming"
    ]
    assert len(renamed) >= 1
    thresholds = _thresholds(experiment_result)

    for item in renamed:
        for method_id in METHODS:
            score = item["methods"][method_id]["score"]
            assert score is not None
            assert 0.0 < score < 1.0
        assert any(
            item["methods"][method_id]["score"] >= thresholds[method_id]
            for method_id in METHODS
        )


def test_required_independent_solutions_are_lower_than_derived(experiment_result):
    for evaluation in experiment_result.method_evaluations:
        records = experiment_result.evaluation_records[evaluation.method_id]
        independent = [
            record.score
            for record in records
            if record.derivation_relation == "independent" and record.score is not None
        ]
        derived = [
            record.score
            for record in records
            if record.derivation_relation != "independent" and record.score is not None
        ]
        assert independent and derived
        assert statistics.mean(independent) < statistics.mean(derived)
        # На evaluation порог может отправлять на проверку все независимые
        # решения; проверяем, что хотя бы на всём корпусе правило не является
        # тривиальным «все подозрительны».
        corpus_independent = [
            record.score
            for record in (
                experiment_result.tuning_records[evaluation.method_id]
                + records
            )
            if record.derivation_relation == "independent"
            and record.score is not None
        ]
        assert any(score < evaluation.threshold for score in corpus_independent)


def test_required_allowed_template_is_explained(experiment_result):
    template_pairs = [
        item for item in experiment_result.comparisons if item["template_only"]
    ]
    assert len(template_pairs) >= 5
    feature_by_pair = {
        row["pair_id"]: row for row in experiment_result.feature_rows
    }

    for item in template_pairs:
        for method_id in METHODS:
            payload = item["methods"][method_id]
            assert payload["template_evidence"] is not None
            assert payload["template_evidence"]["matched_lines"]
            assert "не является основанием" in payload["template_note"]
        row = feature_by_pair[item["pair_id"]]
        assert row["features"]["template_matched_lines"] > 0
        assert row["reference_label"] == "derived"


def test_required_empty_input_stays_undefined():
    empty = CorpusSubmission("u1", "empty", "hw", code="")
    comments_only = CorpusSubmission(
        "u2", "comments", "hw", code="# только комментарий\n"
    )
    result = TokenNGramBaseline().compare(empty, comments_only)

    assert result.score is None
    assert result.warnings
    assert review_required(result.score, 0.1) is None
    assert review_required(None, 0.1) is None


def test_required_syntax_error_does_not_crash_pipeline():
    broken_code = "def broken(:\n    return 1\n"
    with pytest.raises(SyntaxError):
        ast.parse(broken_code)

    broken = CorpusSubmission("u1", "broken", "hw", code=broken_code)
    valid = CorpusSubmission(
        "u2", "valid", "hw", code="def f(x):\n    return x + 1\n"
    )

    baseline_result = TokenNGramBaseline().compare(broken, valid)
    assert baseline_result.score is None or 0.0 <= baseline_result.score <= 1.0

    vectorizer = TfidfVectorizer(max_df=1.0).fit(
        [valid.code, "def g(y):\n    return y * 2\n"]
    )
    tfidf_result = TfidfSimilarityMethod(vectorizer).compare(broken, valid)
    assert tfidf_result.score is None or 0.0 <= tfidf_result.score <= 1.0

    features = tokenize_code(broken_code)
    assert features.tokens


def test_required_self_match_is_excluded():
    query = CorpusSubmission(
        "u1", "query", "hw", code="def f(x):\n    return x + 1\n"
    )
    other = CorpusSubmission(
        "u2", "other", "hw", code="def f(x):\n    return x + 2\n"
    )
    vectorizer = TfidfVectorizer(max_df=1.0).fit([query.code, other.code])
    index = TfidfIndex(vectorizer)
    index.add(query)
    index.add(other)

    ids = {item.submission_id for item in index.retrieve(query, top_k=10)}
    assert "query" not in ids
    assert "other" in ids

    pairs = list(iter_candidate_pairs([query, other]))
    assert all(left.submission_id != right.submission_id for left, right in pairs)
    assert all(left.student_id != right.student_id for left, right in pairs)


def test_required_repeated_student_versions_are_skipped():
    submissions = [
        CorpusSubmission("u1", "s1", "hw", code="x = 1\n"),
        CorpusSubmission("u1", "s2", "hw", code="x = 2\n"),
        CorpusSubmission("u2", "s3", "hw", code="x = 1\n"),
    ]
    stats = PairGenerationStats()
    pairs = list(iter_candidate_pairs(submissions, stats))

    assert len(pairs) == 2
    assert stats.same_student_skipped == 1
    assert stats.repeated_student_submissions == 1
    assert all(left.student_id != right.student_id for left, right in pairs)
    assert {right.submission_id for _, right in pairs} == {"s3"}


def test_required_no_family_leakage_between_tuning_and_evaluation(
    experiment_result, ml_result
):
    pairs = list(iter_corpus_pairs(CORPUS / "pairs.jsonl"))
    splits_by_family: dict[str, set[str]] = {}
    for pair in pairs:
        splits_by_family.setdefault(pair.family_id, set()).add(pair.split)
    assert all(len(values) == 1 for values in splits_by_family.values())

    tuning_families = {
        pair.family_id for pair in pairs if pair.split == "tuning"
    }
    evaluation_families = {
        pair.family_id for pair in pairs if pair.split == "evaluation"
    }
    assert tuning_families.isdisjoint(evaluation_families)

    feature_tuning = {
        row["family_id"]
        for row in experiment_result.feature_rows
        if row["split"] == "tuning"
    }
    feature_evaluation = {
        row["family_id"]
        for row in experiment_result.feature_rows
        if row["split"] == "evaluation"
    }
    assert feature_tuning.isdisjoint(feature_evaluation)

    assert experiment_result.summary["tfidf"]["fit_split"] == "tuning"
    assert ml_result["split"]["split_unit"] == "family_id"
    assert ml_result["split"]["families_disjoint"] is True
    assert set(ml_result["split"]["tuning"]["families"]).isdisjoint(
        ml_result["split"]["evaluation"]["families"]
    )


def test_required_invalid_model_scores_are_rejected(monkeypatch, experiment_result):
    assert validate_model_scores([0, 0.5, 1]) == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="not finite"):
        validate_model_scores([0.5, float("nan")])
    with pytest.raises(ValueError, match="not finite"):
        validate_model_scores([float("inf")])
    with pytest.raises(ValueError, match="outside"):
        validate_model_scores([1.5])
    with pytest.raises(ValueError, match="not numeric"):
        validate_model_scores(["score"])

    class InvalidBundle:
        def predict(self, rows):
            return [float("nan") for _ in rows]

        def to_dict(self):
            return {}

    import submission_system.similarity.ml as ml_module

    monkeypatch.setattr(
        ml_module,
        "train_model",
        lambda *args, **kwargs: InvalidBundle(),
    )
    with pytest.raises(ValueError, match="not finite"):
        ml_module.run_ml_experiment(
            experiment_result.feature_rows,
            experiment_result.method_evaluations,
            seeds=1,
            epochs=1,
        )


def test_required_analysis_service_marks_invalid_model_result_failed():
    valid = AnalysisReport(
        analyzer_id="valid",
        analyzer_version="1",
        score=0.5,
    )
    validate_analysis_report(valid)

    class AnalyzerWithReport:
        analyzer_id = "invalid_model"
        analyzer_version = "1"

        def __init__(self, report):
            self._report = report

        def analyze(self, submission, context):
            return self._report

    submission = Submission(
        student_id="u1",
        submission_id="s1",
        timestamp="2026-01-01T00:00:00+00:00",
        code="x = 1\n",
    )
    assignment = Assignment(assignment_id="hw")

    cases = [
        (AnalysisReport("invalid_model", "1", score=float("nan")), "not finite"),
        (AnalysisReport("invalid_model", "1", score=1.5), "outside"),
        (AnalysisReport("invalid_model", "1", similarity=-0.1), "outside"),
        (None, "expected AnalysisReport"),
    ]
    for report, expected_error in cases:
        service = AnalysisService([AnalyzerWithReport(report)])
        results = service.run_analysis(InMemorySource([submission]), assignment)
        payload = results[0]["analyses"]["invalid_model"]
        assert payload["status"] == "failed"
        assert expected_error in payload["error"]


def test_required_mlflow_payload_contains_params_metrics_and_split(ml_result):
    params = build_params(ml_result)
    metrics = build_metrics(ml_result)
    tags = build_tags(ml_result)

    assert params["split.split_unit"] == "family_id"
    assert params["split.families_disjoint"] is True
    assert params["split.tuning.families"]
    assert params["split.evaluation.families"]
    assert params["features.count"] == len(ml_result["features"])
    assert "evaluation.f1" in metrics
    assert "instability.bootstrap.f1.mean" in metrics
    assert tags["task"] == "review_required"


def test_required_report_has_version_and_config_fingerprint(experiment_result):
    report = experiment_result.summary["report"]
    assert report["report_version"] == REPORT_VERSION
    assert len(report["config_hash"]) == 16
    assert len(report["corpus_fingerprint"]) == 16
    assert len(report["template_fingerprint"]) == 16


def test_required_template_change_updates_config_without_changing_corpus():
    demo = run_template_change_demo(CORPUS)
    change = demo["template_change"]

    assert demo["report_version"] == REPORT_VERSION
    assert change["before_config_hash"] != change["after_config_hash"]
    assert (
        change["before_template_fingerprint"]
        != change["after_template_fingerprint"]
    )
    assert change["corpus_unchanged"] is True
    assert demo["aggregate"]["template_pairs"] == 10
    assert demo["changed_review_decisions"]
    assert any(
        row["methods"][method_id]["delta"] not in (None, 0.0)
        for row in demo["pairs"]
        for method_id in row["methods"]
    )


def test_required_saved_experiment_is_reproducible(tmp_path):
    first = run_experiment(CORPUS)
    second = run_experiment(CORPUS)

    assert first.to_summary_dict() == second.to_summary_dict()
    assert first.comparisons == second.comparisons
    assert first.feature_rows == second.feature_rows
    assert first.threshold_predictions == second.threshold_predictions

    first_paths = save_experiment(first, tmp_path / "first")
    second_paths = save_experiment(second, tmp_path / "second")
    for key in ("summary", "comparisons", "features", "predictions"):
        assert first_paths[key].read_bytes() == second_paths[key].read_bytes()


def test_required_ml_experiment_is_reproducible(experiment_result, ml_result, tmp_path):
    repeated = run_ml_experiment(
        experiment_result.feature_rows,
        experiment_result.method_evaluations,
        seeds=5,
        epochs=300,
    )

    assert repeated["threshold_selected_on_tuning"] == pytest.approx(
        ml_result["threshold_selected_on_tuning"]
    )
    assert repeated["evaluation"] == ml_result["evaluation"]
    assert repeated["instability"] == ml_result["instability"]
    assert repeated["predictions"] == ml_result["predictions"]

    first_paths = save_ml_results(ml_result, tmp_path / "first")
    second_paths = save_ml_results(repeated, tmp_path / "second")
    for key in ("results", "predictions", "split"):
        assert first_paths[key].read_bytes() == second_paths[key].read_bytes()

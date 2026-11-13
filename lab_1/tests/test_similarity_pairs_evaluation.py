import json

import pytest

from submission_system.similarity.corpus import CorpusSubmission
from submission_system.similarity.evaluation import (
    classification_metrics,
    evaluate_method,
    roc_auc,
    select_threshold,
)
from submission_system.similarity.pairs import (
    PairGenerationStats,
    iter_candidate_pairs,
    iter_candidate_pairs_streaming,
)


def sub(submission_id: str, student_id: str, assignment_id: str = "hw", code: str = "x = 1\n"):
    return CorpusSubmission(
        student_id=student_id,
        submission_id=submission_id,
        assignment_id=assignment_id,
        code=code,
    )


def test_pair_generation_skips_self_match_and_counts_repeats():
    submissions = [
        sub("s1", "u1"),
        sub("s2", "u1"),
        sub("s3", "u2"),
        sub("s4", "u3"),
    ]
    stats = PairGenerationStats()
    pairs = list(iter_candidate_pairs(submissions, stats))

    assert len(pairs) == 5  # каждая из двух версий u1 сравнивается с u2 и u3
    assert all(left.student_id != right.student_id for left, right in pairs)
    assert stats.same_student_skipped == 1
    assert stats.repeated_student_submissions == 1
    assert stats.students_with_repeats == 1
    assert stats.cross_student_pairs == 5


def test_pair_generation_is_per_assignment():
    submissions = [sub("s1", "u1", "a"), sub("s2", "u2", "b")]
    assert list(iter_candidate_pairs(submissions)) == []


def test_streaming_pairs_requires_sorted_assignment_file(tmp_path):
    path = tmp_path / "submissions.jsonl"
    records = [
        {"student_id": "u1", "submission_id": "s1", "assignment_id": "hw", "code": "x=1"},
        {"student_id": "u1", "submission_id": "s2", "assignment_id": "hw", "code": "x=2"},
        {"student_id": "u2", "submission_id": "s3", "assignment_id": "hw", "code": "x=3"},
    ]
    path.write_text(
        "\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8"
    )
    stats = PairGenerationStats()
    pairs = list(iter_candidate_pairs_streaming(path, stats))

    assert len(pairs) == 2
    assert stats.same_student_skipped == 1
    assert stats.submissions_seen == 3


def test_classification_metrics_known_values():
    metrics = classification_metrics([0.9, 0.8, 0.2, 0.1], [1, 0, 1, 0], 0.5)
    assert (metrics.tp, metrics.fp, metrics.tn, metrics.fn) == (1, 1, 1, 1)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)


def test_threshold_is_selected_on_tuning_only():
    tuning_scores = [0.9, 0.8, 0.2, 0.1]
    tuning_labels = [1, 1, 0, 0]
    evaluation_scores = [0.95, 0.05]
    evaluation_labels = [1, 0]

    tuning_threshold = select_threshold(tuning_scores, tuning_labels)
    evaluation = evaluate_method(
        "m",
        [
            type("R", (), {"score": score, "label": label})()
            for score, label in zip(tuning_scores, tuning_labels)
        ],
        [
            type("R", (), {"score": score, "label": label})()
            for score, label in zip(evaluation_scores, evaluation_labels)
        ],
    )

    assert evaluation.threshold == tuning_threshold
    assert evaluation.evaluation.f1 == pytest.approx(1.0)


def test_roc_auc_detects_ranking_and_handles_ties():
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert roc_auc([0.5, 0.5], [1, 0]) == pytest.approx(0.5)
    assert roc_auc([0.1, 0.2], [0, 0]) is None


def test_none_scores_are_skipped_not_treated_as_zero():
    records = [
        type("R", (), {"score": None, "label": 1})(),
        type("R", (), {"score": 0.5, "label": 1})(),
    ]
    evaluation = evaluate_method("m", records, records)
    assert evaluation.skipped_tuning == 1
    assert evaluation.evaluation.support == 1

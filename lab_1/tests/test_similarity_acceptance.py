"""Приёмка этапа А ЛР2.

Проверки сформулированы как пользовательские сценарии, а не как тесты
внутренних деталей:
* точная копия даёт максимум;
* шаблон объясняется, а не превращается в «чёрный ящик»;
* пустой и битый код не роняют анализ;
* top-k не содержит self-match и чужие задания;
* evidence ссылается на реальные строки.
"""

from pathlib import Path

import pytest

from submission_system.similarity.baseline import TokenNGramBaseline
from submission_system.similarity.corpus import (
    CorpusSubmission,
    iter_corpus_pairs,
    iter_corpus_submissions,
    load_assignments,
)
from submission_system.similarity.retrieval import TfidfIndex
from submission_system.similarity.tfidf import TfidfSimilarityMethod, TfidfVectorizer

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


@pytest.fixture(scope="module")
def corpus():
    assignments = load_assignments(CORPUS / "assignments.jsonl")
    submissions = list(iter_corpus_submissions(CORPUS / "submissions.jsonl"))
    pairs = list(iter_corpus_pairs(CORPUS / "pairs.jsonl"))
    return assignments, submissions, pairs


@pytest.fixture(scope="module")
def fitted_tfidf(corpus):
    assignments, submissions, _ = corpus
    tuning = [item for item in submissions if item.split == "tuning"]
    vectorizer = TfidfVectorizer(max_df=0.9)
    vectorizer.fit(
        [(item.code, assignments[item.assignment_id].allowed_template) for item in tuning]
    )
    method = TfidfSimilarityMethod(vectorizer)
    method.fitted_on = tuple(item.submission_id for item in tuning)
    return method


def _line_count(code: str) -> int:
    return max(1, len(code.splitlines()))


def _assert_evidence_lines(comparison, left_code: str, right_code: str) -> None:
    for item in comparison.evidence:
        for line in item.left_lines:
            assert 1 <= line <= _line_count(left_code), (comparison.method_id, item)
        for line in item.right_lines:
            assert 1 <= line <= _line_count(right_code), (comparison.method_id, item)
    if comparison.template_evidence:
        for line in comparison.template_evidence.left_line_numbers:
            assert 1 <= line <= _line_count(left_code)
        for line in comparison.template_evidence.right_line_numbers:
            assert 1 <= line <= _line_count(right_code)
        left_lines = {line.strip() for line in left_code.splitlines()}
        right_lines = {line.strip() for line in right_code.splitlines()}
        for matched in comparison.template_evidence.matched_lines:
            assert matched in left_lines
            assert matched in right_lines


def test_exact_copy_has_maximum_non_empty_similarity(corpus, fitted_tfidf):
    assignments, _, pairs = corpus
    baseline = TokenNGramBaseline()

    exact_pairs = [pair for pair in pairs if pair.derivation_relation == "exact_copy"]
    assert exact_pairs

    for pair in exact_pairs:
        template = assignments[pair.assignment_id].allowed_template
        baseline_result = baseline.compare(pair.left, pair.right, template)
        tfidf_result = fitted_tfidf.compare(pair.left, pair.right, template)

        assert baseline_result.score == pytest.approx(1.0)
        assert tfidf_result.score == pytest.approx(1.0)
        assert baseline_result.evidence
        assert tfidf_result.evidence
        _assert_evidence_lines(baseline_result, pair.left.code, pair.right.code)
        _assert_evidence_lines(tfidf_result, pair.left.code, pair.right.code)


def test_common_template_is_explained_not_hidden(corpus, fitted_tfidf):
    assignments, _, pairs = corpus
    baseline = TokenNGramBaseline()
    template_pairs = [pair for pair in pairs if pair.template_only]
    assert len(template_pairs) >= 5

    for pair in template_pairs:
        template = assignments[pair.assignment_id].allowed_template
        for result in (
            baseline.compare(pair.left, pair.right, template),
            fitted_tfidf.compare(pair.left, pair.right, template),
        ):
            assert result.template_evidence is not None
            assert result.template_evidence.matched_lines
            assert result.template_note
            assert "не является основанием" in result.template_note
            assert result.to_dict()["template_note"] == result.template_note
            _assert_evidence_lines(result, pair.left.code, pair.right.code)

        tfidf_result = fitted_tfidf.compare(pair.left, pair.right, template)
        if tfidf_result.score is not None and tfidf_result.score > 0.6:
            assert any("шаблон" in warning for warning in tfidf_result.warnings)


def test_empty_and_broken_code_are_processed(fitted_tfidf):
    baseline = TokenNGramBaseline()
    empty = CorpusSubmission("u1", "empty", "hw", code="")
    comments_only = CorpusSubmission("u2", "comments", "hw", code="# только комментарий\n")
    broken = CorpusSubmission(
        "u3", "broken", "hw", code="def broken(\n    return 1\n"
    )
    valid = CorpusSubmission("u4", "valid", "hw", code="def f(x):\n    return x + 1\n")

    empty_result = baseline.compare(empty, comments_only)
    assert empty_result.score is None
    assert empty_result.warnings

    broken_result = baseline.compare(broken, valid)
    assert broken_result.score is None or 0.0 <= broken_result.score <= 1.0
    _assert_evidence_lines(broken_result, broken.code, valid.code)

    tfidf_result = fitted_tfidf.compare(empty, broken)
    assert tfidf_result.score is None
    assert tfidf_result.warnings


def test_top_k_excludes_self_other_assignment_and_same_student():
    query = CorpusSubmission("u1", "query", "hw_a", code="def f(x):\n    return x + 1\n")
    same_student = CorpusSubmission("u1", "same", "hw_a", code="def f(x):\n    return x\n")
    other_student = CorpusSubmission("u2", "other", "hw_a", code="def f(x):\n    return x + 1\n")
    other_assignment = CorpusSubmission(
        "u3", "other_hw", "hw_b", code="def f(x):\n    return x + 1\n"
    )

    vectorizer = TfidfVectorizer(max_df=1.0)
    documents = [query, same_student, other_student, other_assignment]
    vectorizer.fit([item.code for item in documents])
    index = TfidfIndex(vectorizer)
    for item in documents:
        index.add(item)

    candidates = index.retrieve(query, top_k=10)
    ids = {item.submission_id for item in candidates}

    assert "query" not in ids
    assert "same" not in ids
    assert "other_hw" not in ids
    assert "other" in ids


def test_all_corpus_evidence_points_to_existing_lines(corpus, fitted_tfidf):
    assignments, _, pairs = corpus
    baseline = TokenNGramBaseline()

    for pair in pairs:
        template = assignments[pair.assignment_id].allowed_template
        for result in (
            baseline.compare(pair.left, pair.right, template),
            fitted_tfidf.compare(pair.left, pair.right, template),
        ):
            _assert_evidence_lines(result, pair.left.code, pair.right.code)

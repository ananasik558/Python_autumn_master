import pytest

from submission_system.similarity.corpus import CorpusSubmission
from submission_system.similarity.tfidf import (
    TfidfSimilarityMethod,
    TfidfVectorizer,
    cosine_similarity,
)


def test_vectorizer_fits_only_on_given_documents():
    vectorizer = TfidfVectorizer(max_df=1.0)
    vectorizer.fit(["alpha beta", "alpha gamma"])

    assert "alpha" in vectorizer.vocabulary
    assert "beta" in vectorizer.vocabulary
    assert "delta" not in vectorizer.vocabulary

    transformed = vectorizer.transform("alpha delta")
    assert "alpha" in transformed
    assert "delta" not in transformed


def test_transform_before_fit_is_rejected():
    with pytest.raises(RuntimeError):
        TfidfVectorizer().transform("alpha")


def test_exact_copy_cosine_is_one():
    vectorizer = TfidfVectorizer(max_df=1.0).fit(["a b c", "d e f"])
    left = vectorizer.transform("a b c")
    right = vectorizer.transform("a b c")
    assert cosine_similarity(left, right) == pytest.approx(1.0)


def test_disjoint_documents_have_zero_cosine():
    vectorizer = TfidfVectorizer(max_df=1.0).fit(["a b c", "d e f"])
    assert cosine_similarity(vectorizer.transform("a b"), vectorizer.transform("d e")) == 0.0
    assert cosine_similarity({}, vectorizer.transform("a")) is None


def test_max_df_removes_ubiquitous_tokens():
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=1).fit(
        ["def a", "def b", "def c"]
    )
    assert "def" not in vectorizer.vocabulary
    assert {"a", "b", "c"} & set(vectorizer.vocabulary)


def test_template_tokens_are_not_in_vocabulary():
    template = "LIMIT = 3\n"
    vectorizer = TfidfVectorizer(max_df=1.0).fit([(template + "x = 1\n", template)])
    assert "LIMIT" not in vectorizer.vocabulary
    assert "x" in vectorizer.vocabulary


def test_method_requires_fit_and_compare_returns_evidence():
    left = CorpusSubmission("u1", "s1", "hw", code="def f(x):\n    return x\n")
    right = CorpusSubmission("u2", "s2", "hw", code="def f(x):\n    return x\n")
    method = TfidfSimilarityMethod(TfidfVectorizer())

    with pytest.raises(RuntimeError):
        method.compare(left, right)

    method.fit([(left.code, None), (right.code, None)])
    vocabulary_before = dict(method.vectorizer.vocabulary)
    result = method.compare(left, right)

    assert result.score == pytest.approx(1.0)
    assert result.evidence
    assert result.method_id == "tfidf_cosine"
    assert method.vectorizer.vocabulary == vocabulary_before

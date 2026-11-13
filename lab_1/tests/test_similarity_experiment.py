from pathlib import Path

from submission_system.similarity.baseline import TokenNGramBaseline
from submission_system.similarity.corpus import CorpusSubmission
from submission_system.similarity.experiment import run_experiment, save_experiment
from submission_system.similarity.features import build_pair_features
from submission_system.similarity.retrieval import (
    RetrievalFeedbackGenerator,
    TfidfIndex,
)
from submission_system.similarity.tfidf import TfidfVectorizer
from submission_system.models import Assignment

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


def sub(submission_id: str, student_id: str, code: str, assignment_id: str = "hw"):
    return CorpusSubmission(
        student_id=student_id,
        submission_id=submission_id,
        assignment_id=assignment_id,
        code=code,
    )


def test_retrieval_excludes_self_and_same_student():
    documents = [
        sub("s1", "u1", "def f(x): return x + 1\n"),
        sub("s2", "u1", "def f(x): return x + 1\n"),
        sub("s3", "u2", "def f(x): return x + 1\n"),
        sub("s4", "u3", "def g(y): return y * 2\n"),
    ]
    vectorizer = TfidfVectorizer(max_df=1.0).fit([item.code for item in documents])
    index = TfidfIndex(vectorizer)
    for item in documents:
        index.add(item)

    candidates = index.retrieve(documents[0], top_k=10, exclude_same_student=True)
    ids = {item.submission_id for item in candidates}

    assert "s1" not in ids
    assert "s2" not in ids
    assert "s3" in ids
    assert candidates[0].score >= candidates[-1].score


def test_feedback_uses_retrieved_candidates_and_has_caveat():
    left = sub("s1", "u1", "def f(x):\n    return x + 1\n")
    right = sub("s2", "u2", "def f(x):\n    return x + 1\n")
    comparison = TokenNGramBaseline().compare(left, right)
    vectorizer = TfidfVectorizer(max_df=1.0).fit([left.code, right.code])
    index = TfidfIndex(vectorizer)
    index.add(left)
    index.add(right)

    candidates = index.retrieve(left, top_k=1)
    draft = RetrievalFeedbackGenerator().generate(left, comparison, candidates)

    assert draft.candidates
    assert "s2" in draft.text
    assert "вердикт" in draft.caveat
    assert draft.to_dict()["candidates"][0]["submission_id"] == "s2"


def test_pair_features_are_numeric_and_label_is_separate():
    assignment = Assignment(assignment_id="hw", allowed_template="")
    left = sub("s1", "u1", "def f(x):\n    return x + 1\n")
    right = sub("s2", "u2", "def f(x):\n    return x + 1\n")
    from submission_system.similarity.corpus import CorpusPair

    pair = CorpusPair(
        pair_id="p1",
        assignment_id="hw",
        family_id="F1",
        split="tuning",
        derivation_relation="exact_copy",
        template_only=False,
        left=left,
        right=right,
    )
    baseline = TokenNGramBaseline().compare(left, right)
    from submission_system.similarity.tfidf import TfidfSimilarityMethod

    method = TfidfSimilarityMethod(TfidfVectorizer(max_df=1.0)).fit(
        [(left.code, None), (right.code, None)]
    )

    row = build_pair_features(pair, baseline, method.compare(left, right), {"hw": assignment})

    assert row["reference_label"] == "derived"
    assert row["features"]["token_jaccard"] == 1.0
    assert row["features"]["tfidf_cosine"] == 1.0
    assert "derivation_relation" not in row["features"]


def test_full_experiment_on_educational_corpus(tmp_path):
    result = run_experiment(CORPUS)

    assert len(result.comparisons) == 40
    assert len(result.feature_rows) == 40
    assert result.summary["corpus"]["pairs"] == 40
    assert result.summary["tfidf"]["fit_split"] == "tuning"
    assert result.summary["tfidf"]["fit_documents"] == 27
    assert result.summary["tfidf"]["evaluation_submissions_transformed_without_fit"] == 33
    assert {item.method_id for item in result.method_evaluations} == {
        "token_3gram_jaccard",
        "tfidf_cosine",
    }
    for evaluation in result.method_evaluations:
        assert evaluation.threshold >= 0.0
        assert evaluation.evaluation.support > 0

    paths = save_experiment(result, tmp_path)
    assert paths["summary"].exists()
    assert paths["comparisons"].exists()
    assert paths["features"].exists()

from submission_system.similarity.baseline import TokenNGramBaseline
from submission_system.similarity.corpus import CorpusSubmission


def make_submission(code: str, submission_id: str = "s1", student_id: str = "u1") -> CorpusSubmission:
    return CorpusSubmission(
        student_id=student_id,
        submission_id=submission_id,
        assignment_id="hw",
        code=code,
    )


def test_exact_copy_has_score_one_and_evidence():
    code = "def f(x):\n    return x + 1\n"
    result = TokenNGramBaseline().compare(make_submission(code), make_submission(code, "s2", "u2"))

    assert result.score == 1.0
    assert result.shared_count == result.union_count
    assert result.evidence
    assert result.evidence[0].left_lines
    assert result.evidence[0].right_lines


def test_formatting_change_keeps_score_one():
    left = "def f(x):\n    return x+1\n"
    right = "# комментарий\n\ndef f(x):\n    return x + 1\n"
    result = TokenNGramBaseline().compare(make_submission(left), make_submission(right, "s2", "u2"))

    assert result.score == 1.0


def test_renaming_lowers_similarity_but_is_not_zero():
    left = "def calculate(x):\n    result = x + 1\n    return result\n"
    right = "def calc(value):\n    answer = value + 1\n    return answer\n"
    result = TokenNGramBaseline().compare(make_submission(left), make_submission(right, "s2", "u2"))

    assert result.score is not None
    assert 0.0 < result.score < 1.0


def test_empty_union_returns_none():
    result = TokenNGramBaseline().compare(
        make_submission("# только комментарий\n"),
        make_submission("# другой комментарий\n", "s2", "u2"),
    )

    assert result.score is None
    assert result.warnings


def test_template_is_excluded_from_ngrams_and_reported():
    template = "LIMIT = 3\n"
    left = template + "\ndef count_long(words):\n    return len(words)\n"
    right = template + "\ndef count_long(items):\n    total = 0\n    return total\n"

    result = TokenNGramBaseline().compare(
        make_submission(left),
        make_submission(right, "s2", "u2"),
        allowed_template=template,
    )

    assert result.template_evidence is not None
    assert "LIMIT = 3" in result.template_evidence.matched_lines
    assert result.template_evidence.left_line_numbers == (1,)
    assert result.template_evidence.right_line_numbers == (1,)
    assert result.score is not None

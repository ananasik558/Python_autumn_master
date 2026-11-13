import pytest

from submission_system.tokenizer import (
    ngram_line_index,
    ngram_values,
    tokenize_code,
)


def test_tokenize_excludes_comments_and_service_tokens():
    code = "# комментарий\ndef f(x):\n    # внутри\n    return x + 1\n\n"
    result = tokenize_code(code)
    values = result.token_values

    assert "def" in values
    assert "return" in values
    assert "комментарий" not in values
    assert "внутри" not in values
    assert "#" not in values
    assert "\n" not in values
    assert all(token.line >= 1 for token in result.tokens)


def test_template_lines_are_separated_from_own_code():
    template = "from typing import Iterable\n\nLIMIT = 3\n"
    code = template + "\n\ndef count_long(words):\n    return len(words)\n"

    result = tokenize_code(code, template)

    assert "Iterable" not in result.token_values
    assert "LIMIT" not in result.token_values
    assert "Iterable" in result.template_values
    assert "LIMIT" in result.template_values
    assert result.template_line_numbers == (1, 3)
    assert "count_long" in result.token_values


def test_tokenize_survives_broken_syntax():
    result = tokenize_code("def broken(\n    return 1\n")
    assert result.meaningful_token_count > 0


def test_ngram_values_and_line_index():
    code = "a = 1\nb = 2\nc = 3\n"
    tokens = tokenize_code(code).tokens
    grams = ngram_values(tokens, 3)
    index = ngram_line_index(tokens, 3)

    assert grams
    assert set(grams) == set(index)
    assert 1 in index[("a", "=", "1")]
    with pytest.raises(ValueError):
        ngram_values(tokens, 0)

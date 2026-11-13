"""Токенизация student-кода для анализа сходства.

Модуль намеренно не зависит от сторонних библиотек: используется стандартный
``tokenize``. Комментарии и служебные токены (отступы, переводы строк,
ENCODING/ENDMARKER) отбрасываются, потому что они не являются признаками
решения. Строки разрешённого шаблона можно исключить отдельно: это позволяет
не путать «оба студента использовали выданный шаблон» с «решения совпадают».
"""

from __future__ import annotations

import io
import token as token_module
import tokenize as py_tokenize
from dataclasses import dataclass
from typing import Iterable, Sequence


_MEANINGFUL_TOKEN_TYPES = {
    token_module.NAME,
    token_module.NUMBER,
    token_module.STRING,
    token_module.OP,
}

_FSTRING_TYPES = {
    getattr(token_module, name)
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END")
    if hasattr(token_module, name)
}


@dataclass(frozen=True)
class CodeToken:
    """Один значимый токен с координатой в исходном файле."""

    value: str
    kind: str
    line: int
    column: int


@dataclass(frozen=True)
class TokenizedCode:
    """Результат токенизации одного фрагмента кода.

    ``tokens`` содержит только собственный код студента, ``template_tokens`` —
    токены строк, совпавших с разрешённым шаблоном. Разделение нужно для
    честного сравнения: общий шаблон не должен повышать сходство решений.
    """

    tokens: tuple[CodeToken, ...]
    template_tokens: tuple[CodeToken, ...]
    template_line_numbers: tuple[int, ...]

    @property
    def token_values(self) -> tuple[str, ...]:
        return tuple(token.value for token in self.tokens)

    @property
    def template_values(self) -> tuple[str, ...]:
        return tuple(token.value for token in self.template_tokens)

    @property
    def meaningful_token_count(self) -> int:
        return len(self.tokens) + len(self.template_tokens)


def normalize_template_lines(template: str | None) -> frozenset[str]:
    """Возвращает непустые строки шаблона без отступов.

    Сравнение идёт по точному тексту строки, поэтому шаблон должен быть
    скопирован в студенческий файл без изменений. Это осознанное ограничение:
    нечёткое сопоставление шаблона — отдельная задача.
    """

    if not template:
        return frozenset()
    return frozenset(
        line.strip() for line in template.splitlines() if line.strip()
    )


def tokenize_code(code: str, allowed_template: str | None = None) -> TokenizedCode:
    """Токенизирует код и отделяет строки разрешённого шаблона.

    Функция устойчива к синтаксически неполному коду: ``tokenize`` может
    выдать часть токенов до ошибки. В этом случае возвращается то, что удалось
    прочитать, без исключения наружу — анализатор не должен падать из-за одной
    повреждённой сдачи.
    """

    template_lines = normalize_template_lines(allowed_template)
    own_tokens: list[CodeToken] = []
    template_tokens: list[CodeToken] = []
    used_template_lines: set[int] = set()

    reader = io.StringIO(code).readline
    try:
        stream = py_tokenize.generate_tokens(reader)
        for token in stream:
            if token.type not in _MEANINGFUL_TOKEN_TYPES and token.type not in _FSTRING_TYPES:
                continue

            line_number = token.start[0]
            # token.line содержит исходную строку целиком; сравниваем её без
            # отступов с разрешённым шаблоном.
            is_template_line = bool(template_lines) and token.line.strip() in template_lines
            if is_template_line:
                used_template_lines.add(line_number)

            item = CodeToken(
                value=token.string,
                kind=token_module.tok_name.get(token.type, str(token.type)),
                line=line_number,
                column=token.start[1],
            )
            if is_template_line:
                template_tokens.append(item)
            else:
                own_tokens.append(item)
    except (py_tokenize.TokenError, IndentationError, SyntaxError):
        pass

    return TokenizedCode(
        tokens=tuple(own_tokens),
        template_tokens=tuple(template_tokens),
        template_line_numbers=tuple(sorted(used_template_lines)),
    )


def ngram_values(tokens: Sequence[CodeToken], n: int = 3) -> tuple[tuple[str, ...], ...]:
    """Возвращает все n-граммы значений токенов (с повторами)."""

    if n <= 0:
        raise ValueError("n должно быть положительным")
    values = [token.value for token in tokens]
    if len(values) < n:
        return ()
    return tuple(tuple(values[index:index + n]) for index in range(len(values) - n + 1))


def ngram_line_index(tokens: Sequence[CodeToken], n: int = 3) -> dict[tuple[str, ...], set[int]]:
    """Строит индекс n-грамм: n-грамма -> номера исходных строк.

    Индекс хранит множества строк, а не полные тексты, поэтому память растёт
    примерно как O(число уникальных n-грамм * число вхождений строк).
    """

    if n <= 0:
        raise ValueError("n должно быть положительным")
    index: dict[tuple[str, ...], set[int]] = {}
    for start in range(max(0, len(tokens) - n + 1)):
        window = tokens[start:start + n]
        key = tuple(token.value for token in window)
        lines = index.setdefault(key, set())
        lines.update(token.line for token in window)
    return index


def token_documents(
    code: str,
    allowed_template: str | None = None,
) -> tuple[str, ...]:
    """Удобная обёртка: значения значимых токенов без токенов шаблона."""

    return tokenize_code(code, allowed_template).token_values

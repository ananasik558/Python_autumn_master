"""Token baseline: n-граммы значимых токенов и Jaccard.

Метод намеренно простой и воспроизводимый. Он не «решает», списан ли код:
он возвращает числовое сходство, основания (совпавшие n-граммы и диапазоны
строк) и отдельно сведения об использовании разрешённого шаблона.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..tokenizer import (
    CodeToken,
    ngram_line_index,
    normalize_template_lines,
    tokenize_code,
)
from .corpus import CorpusSubmission


@dataclass(frozen=True)
class MatchEvidence:
    """Совпавшая n-грамма и строки, в которых она встретилась."""

    ngram: tuple[str, ...]
    left_lines: tuple[int, ...]
    right_lines: tuple[int, ...]

    @property
    def preview(self) -> str:
        return " ".join(self.ngram)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ngram": list(self.ngram),
            "preview": self.preview,
            "left_lines": list(self.left_lines),
            "right_lines": list(self.right_lines),
        }


@dataclass(frozen=True)
class TemplateEvidence:
    """Сведения о совпадении с разрешённым шаблоном, а не о плагиате."""

    matched_lines: tuple[str, ...]
    left_line_numbers: tuple[int, ...]
    right_line_numbers: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_lines": list(self.matched_lines),
            "left_line_numbers": list(self.left_line_numbers),
            "right_line_numbers": list(self.right_line_numbers),
        }


@dataclass(frozen=True)
class PairComparison:
    """Результат сравнения одной пары одним методом."""

    method_id: str
    left_submission_id: str
    right_submission_id: str
    score: Optional[float]
    shared_count: int = 0
    left_count: int = 0
    right_count: int = 0
    union_count: int = 0
    evidence: tuple[MatchEvidence, ...] = ()
    template_evidence: Optional[TemplateEvidence] = None
    warnings: tuple[str, ...] = ()

    @property
    def score_or_zero(self) -> float:
        """Только для агрегированных метрик; исходный score остаётся None."""

        return 0.0 if self.score is None else self.score

    @property
    def template_note(self) -> str | None:
        """Явное объяснение общего шаблона: сходство, а не обвинение."""

        if self.template_evidence and self.template_evidence.matched_lines:
            return (
                "Обнаружен разрешённый шаблон; его строки исключены из признаков "
                "собственного кода. Совпадение шаблона не является основанием "
                "для обвинения."
            )
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "left_submission_id": self.left_submission_id,
            "right_submission_id": self.right_submission_id,
            "score": self.score,
            "shared_count": self.shared_count,
            "left_count": self.left_count,
            "right_count": self.right_count,
            "union_count": self.union_count,
            "evidence": [item.to_dict() for item in self.evidence],
            "template_evidence": (
                self.template_evidence.to_dict() if self.template_evidence else None
            ),
            "template_note": self.template_note,
            "warnings": list(self.warnings),
        }


def _template_evidence(
    left: CorpusSubmission,
    right: CorpusSubmission,
    allowed_template: str | None,
) -> Optional[TemplateEvidence]:
    template_lines = normalize_template_lines(allowed_template)
    if not template_lines:
        return None

    left_features = tokenize_code(left.code, allowed_template)
    right_features = tokenize_code(right.code, allowed_template)
    left_lines = set(left_features.template_line_numbers)
    right_lines = set(right_features.template_line_numbers)
    if not left_lines and not right_lines:
        return None

    matched_text = tuple(
        line for line in sorted(template_lines)
        if line in {part.strip() for part in left.code.splitlines()}
        and line in {part.strip() for part in right.code.splitlines()}
    )
    return TemplateEvidence(
        matched_lines=matched_text,
        left_line_numbers=tuple(sorted(left_lines)),
        right_line_numbers=tuple(sorted(right_lines)),
    )


def _safe_ngram_index(tokens: Sequence[CodeToken], n: int) -> dict[tuple[str, ...], set[int]]:
    if not tokens:
        return {}
    return ngram_line_index(tokens, n)


class TokenNGramBaseline:
    """Jaccard по множествам 3-грамм значимых токенов.

    Формула: ``|A ∩ B| / |A ∪ B|``. Если объединение пусто (например, код
    состоит только из комментариев), возвращается ``None`` — это не ноль,
    а отсутствие данных для сравнения.
    """

    method_id = "token_3gram_jaccard"
    method_version = "1.0.0"

    def __init__(self, n: int = 3, max_evidence: int = 50) -> None:
        if n <= 0:
            raise ValueError("n должно быть положительным")
        if max_evidence < 0:
            raise ValueError("max_evidence не может быть отрицательным")
        self.n = n
        self.max_evidence = max_evidence

    def compare(
        self,
        left: CorpusSubmission,
        right: CorpusSubmission,
        allowed_template: str | None = None,
    ) -> PairComparison:
        left_features = tokenize_code(left.code, allowed_template)
        right_features = tokenize_code(right.code, allowed_template)

        left_index = _safe_ngram_index(left_features.tokens, self.n)
        right_index = _safe_ngram_index(right_features.tokens, self.n)
        left_keys = set(left_index)
        right_keys = set(right_index)
        union = left_keys | right_keys
        shared = left_keys & right_keys

        warnings: list[str] = []
        if not left_features.tokens:
            warnings.append("В левой работе нет значимых токенов вне шаблона")
        if not right_features.tokens:
            warnings.append("В правой работе нет значимых токенов вне шаблона")
        if left_features.template_line_numbers or right_features.template_line_numbers:
            warnings.append("Строки разрешённого шаблона исключены из n-грамм")

        score: Optional[float]
        if not union:
            score = None
        else:
            score = len(shared) / len(union)

        evidence: list[MatchEvidence] = []
        for ngram in sorted(shared)[: self.max_evidence]:
            evidence.append(
                MatchEvidence(
                    ngram=ngram,
                    left_lines=tuple(sorted(left_index[ngram])),
                    right_lines=tuple(sorted(right_index[ngram])),
                )
            )

        return PairComparison(
            method_id=self.method_id,
            left_submission_id=left.submission_id,
            right_submission_id=right.submission_id,
            score=score,
            shared_count=len(shared),
            left_count=len(left_keys),
            right_count=len(right_keys),
            union_count=len(union),
            evidence=tuple(evidence),
            template_evidence=_template_evidence(left, right, allowed_template),
            warnings=tuple(warnings),
        )

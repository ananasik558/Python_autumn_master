"""Второй метод: TF-IDF по токенам кода и косинусное сходство.

Ключевое требование эксперимента: векторизатор обучается только на tuning-
наборе. Для evaluation вызывается ``transform`` уже обученного векторизатора,
повторный ``fit`` запрещён. Это защищает оценку от утечки данных.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from ..tokenizer import CodeToken, tokenize_code
from .baseline import MatchEvidence, PairComparison, TemplateEvidence, _template_evidence
from .corpus import CorpusSubmission


DocumentInput = Union[str, tuple[str, Optional[str]]]


@dataclass
class TfidfVectorizer:
    """Минимальный TF-IDF без внешних зависимостей.

    ``fit`` принимает документы в виде ``str`` (код без шаблона) или
    ``(код, allowed_template)``. Внутри используется тот же tokenizer, что и в
    baseline, поэтому комментарии/служебные токены и строки шаблона не
    попадают в словарь.
    """

    max_features: int = 5000
    min_df: int = 1
    max_df: float = 1.0
    vocabulary: dict[str, int] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    document_count: int = 0
    _fitted: bool = field(default=False, init=False, repr=False)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _tokens(self, document: DocumentInput) -> tuple[str, ...]:
        if isinstance(document, tuple):
            code, template = document
            return tokenize_code(code, template).token_values
        return tokenize_code(document).token_values

    def fit(self, documents: Iterable[DocumentInput]) -> "TfidfVectorizer":
        """Обучает словарь и IDF только на переданных документах."""

        document_frequency: Counter[str] = Counter()
        document_count = 0
        for document in documents:
            tokens = self._tokens(document)
            if not tokens:
                continue
            document_count += 1
            document_frequency.update(set(tokens))

        if document_count == 0:
            self.vocabulary = {}
            self.idf = {}
            self.document_count = 0
            self._fitted = True
            return self

        # max_df отбрасывает почти служебную лексику (def/return/for), которая
        # встречается почти в каждой работе и не различает решения. Порог
        # фиксирован до эксперимента и не подбирается на evaluation.
        max_documents = max(1, int(self.max_df * document_count)) if self.max_df < 1.0 else document_count
        filtered = [
            (term, frequency)
            for term, frequency in document_frequency.items()
            if frequency >= self.min_df and frequency <= max_documents
        ]
        ranked = sorted(filtered, key=lambda item: (-item[1], item[0]))
        if self.max_features > 0:
            ranked = ranked[: self.max_features]

        selected = sorted(term for term, _ in ranked)
        self.vocabulary = {term: index for index, term in enumerate(selected)}
        self.idf = {
            term: math.log((1 + document_count) / (1 + document_frequency[term])) + 1.0
            for term in selected
        }
        self.document_count = document_count
        self._fitted = True
        return self

    def transform(self, document: DocumentInput) -> dict[str, float]:
        """Преобразует документ без изменения словаря и IDF."""

        if not self._fitted:
            raise RuntimeError("TfidfVectorizer.transform вызван до fit")

        tokens = self._tokens(document)
        if not tokens:
            return {}

        counts = Counter(token for token in tokens if token in self.vocabulary)
        if not counts:
            return {}

        vector = {
            term: (1.0 + math.log(count)) * self.idf[term]
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0.0:
            return {}
        return {term: value / norm for term, value in vector.items()}


def cosine_similarity(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> Optional[float]:
    """Косинус разреженных векторов; ``None``, если один вектор пуст."""

    if not left or not right:
        return None
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    if dot == 0.0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _term_line_index(tokens: Sequence[CodeToken]) -> dict[str, set[int]]:
    index: dict[str, set[int]] = {}
    for token in tokens:
        index.setdefault(token.value, set()).add(token.line)
    return index


@dataclass
class TfidfSimilarityMethod:
    """Cosine-сходство в пространстве TF-IDF tuning-корпуса."""

    method_id = "tfidf_cosine"
    method_version = "1.0.0"

    vectorizer: TfidfVectorizer
    max_evidence: int = 50
    fitted_on: tuple[str, ...] = ()

    def fit(
        self,
        documents: Iterable[tuple[str, Optional[str]]],
        document_ids: Iterable[str] = (),
    ) -> "TfidfSimilarityMethod":
        self.vectorizer.fit(documents)
        self.fitted_on = tuple(document_ids)
        return self

    def _transform_submission(
        self,
        submission: CorpusSubmission,
        allowed_template: Optional[str],
    ) -> tuple[dict[str, float], tuple[CodeToken, ...], int]:
        features = tokenize_code(submission.code, allowed_template)
        vector = self.vectorizer.transform((submission.code, allowed_template))
        unknown = sum(
            1 for token in features.tokens if token.value not in self.vectorizer.vocabulary
        )
        return vector, features.tokens, unknown

    def compare(
        self,
        left: CorpusSubmission,
        right: CorpusSubmission,
        allowed_template: Optional[str] = None,
    ) -> PairComparison:
        if not self.vectorizer.is_fitted:
            raise RuntimeError(
                "TF-IDF метод не обучен: вызовите fit на tuning-наборе"
            )

        left_vector, left_tokens, left_unknown = self._transform_submission(
            left, allowed_template
        )
        right_vector, right_tokens, right_unknown = self._transform_submission(
            right, allowed_template
        )
        score = cosine_similarity(left_vector, right_vector)

        shared_terms = sorted(
            set(left_vector) & set(right_vector),
            key=lambda term: (-(left_vector[term] * right_vector[term]), term),
        )
        left_lines = _term_line_index(left_tokens)
        right_lines = _term_line_index(right_tokens)
        evidence = tuple(
            MatchEvidence(
                ngram=(term,),
                left_lines=tuple(sorted(left_lines.get(term, ()))),
                right_lines=tuple(sorted(right_lines.get(term, ()))),
            )
            for term in shared_terms[: self.max_evidence]
        )

        warnings: list[str] = []
        if left_tokens or right_tokens:
            total_tokens = len(left_tokens) + len(right_tokens)
            unknown_ratio = (left_unknown + right_unknown) / max(total_tokens, 1)
            if unknown_ratio > 0.5:
                warnings.append(
                    f"Доля OOV-токенов {unknown_ratio:.0%}: словарь обучен только на tuning"
                )
        if not left_vector:
            warnings.append("Левая работа не содержит токенов из tuning-словаря")
        if not right_vector:
            warnings.append("Правая работа не содержит токенов из tuning-словаря")

        template = _template_evidence(left, right, allowed_template)
        if template is None and allowed_template:
            template = TemplateEvidence((), (), ())
        if template and template.matched_lines:
            warnings.append(
                "Обнаружен разрешённый шаблон; его строки исключены из TF-IDF, "
                "но общая лексика собственной логики может повышать cosine"
            )

        return PairComparison(
            method_id=self.method_id,
            left_submission_id=left.submission_id,
            right_submission_id=right.submission_id,
            score=score,
            shared_count=len(shared_terms),
            left_count=len(left_vector),
            right_count=len(right_vector),
            union_count=len(set(left_vector) | set(right_vector)),
            evidence=evidence,
            template_evidence=template,
            warnings=tuple(warnings),
        )


def document_inputs(
    submissions: Iterable[CorpusSubmission],
    assignments: Mapping[str, Any],
) -> Iterable[tuple[str, Optional[str]]]:
    """Пары (код, шаблон) для fit/transform."""

    for submission in submissions:
        assignment = assignments.get(submission.assignment_id)
        template = getattr(assignment, "allowed_template", "") if assignment else ""
        yield submission.code, template

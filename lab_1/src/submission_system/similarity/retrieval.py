"""Retrieval кандидатов и детерминированная LLM-заглушка.

По заданию LLM не должна отвечать на вопрос «плагиат или нет» одним запросом.
Сначала retrieval находит похожие работы по TF-IDF, затем генератор feedback
получает уже найденные кандидаты и числовые основания. Реальный вызов LLM в
этой лабораторной не выполняется: интерфейс и порядок шагов зафиксированы.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from ..tokenizer import tokenize_code
from .baseline import PairComparison
from .corpus import CorpusSubmission
from .tfidf import TfidfVectorizer, cosine_similarity


@dataclass(frozen=True)
class RetrievedCandidate:
    submission_id: str
    student_id: str
    assignment_id: str
    score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "submission_id": self.submission_id,
            "student_id": self.student_id,
            "assignment_id": self.assignment_id,
            "score": self.score,
        }


@dataclass
class TfidfIndex:
    """Индекс векторов, построенный обученным векторизатором.

    Индекс хранится в памяти. Это осознанный компромисс: для больших корпусов
    нужен внешний индекс/ANN, иначе память растёт вместе с числом сдач.
    """

    vectorizer: TfidfVectorizer
    _vectors: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)
    _submissions: dict[str, CorpusSubmission] = field(default_factory=dict, repr=False)

    def add(
        self,
        submission: CorpusSubmission,
        allowed_template: Optional[str] = None,
    ) -> None:
        if not self.vectorizer.is_fitted:
            raise RuntimeError("Нельзя строить индекс до fit векторизатора")
        vector = self.vectorizer.transform((submission.code, allowed_template))
        self._vectors[submission.submission_id] = vector
        self._submissions[submission.submission_id] = submission

    def add_many(
        self,
        submissions: Iterable[CorpusSubmission],
        assignments: Mapping[str, object],
    ) -> None:
        for submission in submissions:
            assignment = assignments.get(submission.assignment_id)
            template = getattr(assignment, "allowed_template", "") if assignment else ""
            self.add(submission, template)

    def retrieve(
        self,
        query: CorpusSubmission,
        allowed_template: Optional[str] = None,
        top_k: int = 5,
        exclude_same_student: bool = True,
    ) -> list[RetrievedCandidate]:
        if top_k <= 0:
            return []

        query_vector = self.vectorizer.transform((query.code, allowed_template))
        candidates: list[RetrievedCandidate] = []
        for submission_id, vector in self._vectors.items():
            candidate = self._submissions[submission_id]
            if submission_id == query.submission_id:
                continue
            if candidate.assignment_id != query.assignment_id:
                continue
            if exclude_same_student and candidate.student_id == query.student_id:
                continue
            score = cosine_similarity(query_vector, vector)
            if score is None:
                continue
            candidates.append(
                RetrievedCandidate(
                    submission_id=candidate.submission_id,
                    student_id=candidate.student_id,
                    assignment_id=candidate.assignment_id,
                    score=score,
                )
            )

        candidates.sort(key=lambda item: (-item.score, item.submission_id))
        return candidates[:top_k]


@dataclass(frozen=True)
class FeedbackDraft:
    query_submission_id: str
    method_id: str
    candidates: tuple[RetrievedCandidate, ...]
    text: str
    caveat: str = "Это обнаруженное сходство, а не вердикт о нарушении."

    def to_dict(self) -> dict[str, object]:
        return {
            "query_submission_id": self.query_submission_id,
            "method_id": self.method_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "text": self.text,
            "caveat": self.caveat,
        }


class RetrievalFeedbackGenerator:
    """Шаблон генерации feedback после retrieval (без реального LLM)."""

    method_id = "retrieval_llm_stub"
    prompt_template_version = "1.0.0"

    def generate(
        self,
        query: CorpusSubmission,
        comparison: PairComparison,
        candidates: Iterable[RetrievedCandidate],
    ) -> FeedbackDraft:
        candidate_list = tuple(candidates)
        if candidate_list:
            candidate_text = "; ".join(
                f"{item.submission_id} ({item.score:.2f})"
                for item in candidate_list
            )
        else:
            candidate_text = "кандидаты не найдены"

        evidence_lines = [
            f"строки {','.join(map(str, item.left_lines))} <-> "
            f"{','.join(map(str, item.right_lines))}: {item.preview}"
            for item in comparison.evidence[:3]
        ]
        evidence_text = "; ".join(evidence_lines) if evidence_lines else "нет совпавших n-грамм"

        template_text = f" {comparison.template_note}" if comparison.template_note else ""
        text = (
            f"Кандидаты для проверки: {candidate_text}. "
            f"Метод {comparison.method_id} дал score={comparison.score}. "
            f"Основания: {evidence_text}.{template_text} "
            f"Порог и решение о нарушении принимает преподаватель."
        )
        return FeedbackDraft(
            query_submission_id=query.submission_id,
            method_id=self.method_id,
            candidates=candidate_list,
            text=text,
        )


def retrieved_token_overlap(
    query: CorpusSubmission,
    candidate: CorpusSubmission,
    allowed_template: Optional[str] = None,
) -> float:
    """Простая доля общих токенов — дополнительный сигнал для retrieval."""

    left = set(tokenize_code(query.code, allowed_template).token_values)
    right = set(tokenize_code(candidate.code, allowed_template).token_values)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)

"""Признаки пары для ML-компонента.

Сходство — не единственный сигнал. Для будущего классификатора/решающего
правила сохраняются длина работ, доля шаблона, число общих n-грамм и
расхождение двух методов. Происхождение пары хранится отдельно как
справочная метка, а не как признак модели.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..models import Assignment
from ..tokenizer import tokenize_code
from .baseline import PairComparison
from .corpus import CorpusPair


def _submission_features(code: str, allowed_template: str | None) -> dict[str, Any]:
    features = tokenize_code(code, allowed_template)
    return {
        "token_count": len(features.tokens),
        "template_token_count": len(features.template_tokens),
        "template_line_count": len(features.template_line_numbers),
        "char_count": len(code),
        "line_count": len(code.splitlines()),
    }


def build_pair_features(
    pair: CorpusPair,
    baseline: PairComparison,
    tfidf: PairComparison,
    assignments: Mapping[str, Assignment],
) -> dict[str, Any]:
    assignment = assignments.get(pair.assignment_id)
    template = assignment.allowed_template if assignment else ""

    left = _submission_features(pair.left.code, template)
    right = _submission_features(pair.right.code, template)
    left_tokens = max(left["token_count"], 1)
    right_tokens = max(right["token_count"], 1)

    jaccard = baseline.score
    cosine = tfidf.score
    disagreement = (
        abs(jaccard - cosine)
        if jaccard is not None and cosine is not None
        else None
    )

    return {
        "pair_id": pair.pair_id,
        "assignment_id": pair.assignment_id,
        "family_id": pair.family_id,
        "split": pair.split,
        "derivation_relation": pair.derivation_relation,
        "template_only": pair.template_only,
        "reference_label": "derived" if pair.is_derived_reference else "independent",
        "features": {
            "token_jaccard": jaccard,
            "tfidf_cosine": cosine,
            "method_disagreement": disagreement,
            "shared_ngrams": baseline.shared_count,
            "union_ngrams": baseline.union_count,
            "ngram_overlap_fraction_of_min": (
                baseline.shared_count / max(min(baseline.left_count, baseline.right_count), 1)
            ),
            "left_token_count": left["token_count"],
            "right_token_count": right["token_count"],
            "token_count_ratio": min(left_tokens, right_tokens) / max(left_tokens, right_tokens),
            "left_char_count": left["char_count"],
            "right_char_count": right["char_count"],
            "char_count_ratio": min(left["char_count"], right["char_count"])
            / max(max(left["char_count"], right["char_count"]), 1),
            "left_template_lines": left["template_line_count"],
            "right_template_lines": right["template_line_count"],
            "template_matched_lines": (
                len(baseline.template_evidence.matched_lines)
                if baseline.template_evidence
                else 0
            ),
        },
    }

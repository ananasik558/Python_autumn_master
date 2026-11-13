"""Демонстрация добавления нового разрешённого шаблона.

На защите нужно показать, что шаблон — это элемент конфигурации: его изменение
меняет отпечаток конфигурации и может изменить score/review_required, но не
меняет код студентов и не «подгоняет» метрики под ответ.

Модуль не редактирует `data/corpus/assignments.jsonl`. Он строит второй прогон
с изменённой копией `Assignment` и сохраняет сравнение до/после.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..models import Assignment
from .corpus import iter_corpus_pairs, load_assignments
from .evaluation import review_required
from .experiment import REPORT_VERSION, run_experiment

DEFAULT_ADDED_BLOCK = """def count_long(words):
    total = 0
    for word in words:
        if len(word) > LIMIT:
            total += 1
    return total
"""


def _comparison_map(result) -> dict[str, dict[str, Any]]:
    return {item["pair_id"]: item for item in result.comparisons}


def _thresholds(result) -> dict[str, float]:
    return {item.method_id: item.threshold for item in result.method_evaluations}


def _evaluation_snapshot(result) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for item in result.method_evaluations:
        snapshot[item.method_id] = {
            "threshold_selected_on_tuning": item.threshold,
            "evaluation": item.evaluation.to_dict(),
            "evaluation_auc": item.evaluation_auc,
            "review_rate": item.evaluation.review_rate,
            "undefined_rate": item.evaluation_undefined_rate,
        }
    return snapshot


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def run_template_change_demo(
    corpus_dir: str | Path,
    *,
    assignment_id: str = "hw_template",
    added_block: str = DEFAULT_ADDED_BLOCK,
) -> dict[str, Any]:
    """Сравнивает эксперимент до и после добавления блока в шаблон."""

    corpus_dir = Path(corpus_dir)
    base_assignments = load_assignments(corpus_dir / "assignments.jsonl")
    if assignment_id not in base_assignments:
        raise ValueError(f"В корпусе нет задания {assignment_id}")

    original = base_assignments[assignment_id]
    before_template = original.allowed_template
    after_template = (
        before_template.rstrip()
        + "\n\n"
        + added_block.strip()
        + "\n"
    )
    updated_assignments: dict[str, Assignment] = dict(base_assignments)
    updated_assignments[assignment_id] = replace(
        original,
        allowed_template=after_template,
    )

    before = run_experiment(corpus_dir)
    after = run_experiment(corpus_dir, assignments_override=updated_assignments)
    before_comparisons = _comparison_map(before)
    after_comparisons = _comparison_map(after)
    before_thresholds = _thresholds(before)
    after_thresholds = _thresholds(after)

    pairs = [
        pair
        for pair in iter_corpus_pairs(corpus_dir / "pairs.jsonl")
        if pair.assignment_id == assignment_id
    ]
    pair_rows: list[dict[str, Any]] = []
    changed_decisions: list[dict[str, Any]] = []
    before_scores: dict[str, list[float]] = {
        method_id: [] for method_id in before_thresholds
    }
    after_scores: dict[str, list[float]] = {
        method_id: [] for method_id in after_thresholds
    }
    before_reviews: dict[str, int] = {
        method_id: 0 for method_id in before_thresholds
    }
    after_reviews: dict[str, int] = {
        method_id: 0 for method_id in after_thresholds
    }

    for pair in pairs:
        before_payload = before_comparisons[pair.pair_id]["methods"]
        after_payload = after_comparisons[pair.pair_id]["methods"]
        method_rows: dict[str, Any] = {}
        for method_id in before_thresholds:
            before_score = before_payload[method_id]["score"]
            after_score = after_payload[method_id]["score"]
            before_review = review_required(
                before_score, before_thresholds[method_id]
            )
            after_review = review_required(
                after_score, after_thresholds[method_id]
            )
            if before_score is not None:
                before_scores[method_id].append(before_score)
            if after_score is not None:
                after_scores[method_id].append(after_score)
            if before_review:
                before_reviews[method_id] += 1
            if after_review:
                after_reviews[method_id] += 1
            if before_review != after_review:
                changed_decisions.append(
                    {
                        "pair_id": pair.pair_id,
                        "method_id": method_id,
                        "before_review_required": before_review,
                        "after_review_required": after_review,
                        "before_score": before_score,
                        "after_score": after_score,
                    }
                )
            method_rows[method_id] = {
                "before_score": before_score,
                "after_score": after_score,
                "delta": (
                    after_score - before_score
                    if before_score is not None and after_score is not None
                    else None
                ),
                "before_review_required": before_review,
                "after_review_required": after_review,
                "after_template_evidence_lines": (
                    len(
                        (
                            after_payload[method_id].get("template_evidence")
                            or {}
                        ).get("matched_lines")
                        or []
                    )
                ),
            }
        pair_rows.append(
            {
                "pair_id": pair.pair_id,
                "split": pair.split,
                "derivation_relation": pair.derivation_relation,
                "methods": method_rows,
            }
        )

    payload = {
        "report_version": REPORT_VERSION,
        "assignment_id": assignment_id,
        "template_change": {
            "added_block_lines": added_block.strip().splitlines(),
            "before_template_fingerprint": before.summary["report"][
                "template_fingerprint"
            ],
            "after_template_fingerprint": after.summary["report"][
                "template_fingerprint"
            ],
            "before_config_hash": before.summary["report"]["config_hash"],
            "after_config_hash": after.summary["report"]["config_hash"],
            "corpus_fingerprint": before.summary["report"][
                "corpus_fingerprint"
            ],
            "corpus_unchanged": (
                before.summary["report"]["corpus_fingerprint"]
                == after.summary["report"]["corpus_fingerprint"]
            ),
        },
        "aggregate": {
            "template_pairs": len(pairs),
            "methods": {
                method_id: {
                    "mean_score_before": _mean(before_scores[method_id]),
                    "mean_score_after": _mean(after_scores[method_id]),
                    "review_required_before": before_reviews[method_id],
                    "review_required_after": after_reviews[method_id],
                }
                for method_id in before_thresholds
            },
        },
        "evaluation_before": _evaluation_snapshot(before),
        "evaluation_after": _evaluation_snapshot(after),
        "changed_review_decisions": changed_decisions,
        "pairs": pair_rows,
        "interpretation": (
            "Добавление разрешённого шаблона — изменение конфигурации. Оно "
            "исключает новые строки шаблона из признаков и может снизить score "
            "и число review_required. Это не переобучение на evaluation: "
            "шаблон задаётся до анализа, а corpus_fingerprint не меняется."
        ),
    }
    return payload


def save_template_change_demo(
    payload: Mapping[str, Any],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path

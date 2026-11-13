"""CLI для экспериментов ЛР2.

Примеры:
    python -m submission_system.similarity.cli evaluate --corpus data/corpus
    python -m submission_system.similarity.cli retrieve --corpus data/corpus \
        --submission sub-F18-a --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .baseline import PairComparison, TokenNGramBaseline
from .corpus import iter_corpus_submissions, load_assignments
from .error_analysis import analyze_errors, save_error_analysis
from .experiment import run_experiment, save_experiment
from .ml import run_ml_experiment, save_ml_results
from .mlflow_logging import log_stage_b
from .retrieval import RetrievalFeedbackGenerator, TfidfIndex
from .template_demo import (
    DEFAULT_ADDED_BLOCK,
    run_template_change_demo,
    save_template_change_demo,
)
from .tfidf import TfidfVectorizer


def _evaluate(args: argparse.Namespace) -> int:
    result = run_experiment(
        args.corpus,
        ngram_size=args.ngram_size,
        max_features=args.max_features,
    )
    paths = save_experiment(result, args.output)
    summary = result.to_summary_dict()

    print("method                         threshold  precision  recall  f1     auc")
    for item in result.method_evaluations:
        metrics = item.evaluation
        auc = "n/a" if item.evaluation_auc is None else f"{item.evaluation_auc:.3f}"
        print(
            f"{item.method_id:<30} {item.threshold:>9.3f} "
            f"{metrics.precision:>10.3f} {metrics.recall:>7.3f} "
            f"{metrics.f1:>6.3f} {auc:>7}"
        )
    selected = result.selected_method()
    report = summary["report"]
    print(
        f"\nreport version: {report['report_version']}  "
        f"config: {report['config_hash']}  "
        f"templates: {report['template_fingerprint']}"
    )
    print(f"selected by tuning F1: {selected.method_id}")
    print(f"summary: {paths['summary']}")
    print(f"comparisons: {paths['comparisons']}")
    print(f"pair features: {paths['features']}")
    print(f"threshold predictions: {paths['predictions']}")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _retrieve(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    assignments = load_assignments(corpus / "assignments.jsonl")
    submissions = list(iter_corpus_submissions(corpus / "submissions.jsonl"))
    by_id = {item.submission_id: item for item in submissions}
    if args.submission not in by_id:
        print(f"Сдача не найдена: {args.submission}")
        return 1

    tuning = [item for item in submissions if item.split == "tuning"]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(
        [
            (item.code, assignments[item.assignment_id].allowed_template)
            for item in tuning
        ]
    )
    index = TfidfIndex(vectorizer)
    index.add_many(submissions, assignments)

    query = by_id[args.submission]
    template = assignments[query.assignment_id].allowed_template
    candidates = index.retrieve(query, template, top_k=args.top_k)
    if candidates:
        top = by_id[candidates[0].submission_id]
        comparison = TokenNGramBaseline().compare(query, top, template)
    else:
        comparison = PairComparison(
            method_id="token_3gram_jaccard",
            left_submission_id=query.submission_id,
            right_submission_id="",
            score=None,
        )
    draft = RetrievalFeedbackGenerator().generate(query, comparison, candidates)

    print(json.dumps(draft.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _errors(args: argparse.Namespace) -> int:
    result = run_experiment(
        args.corpus,
        ngram_size=args.ngram_size,
        max_features=args.max_features,
    )
    analysis = analyze_errors(
        result.method_evaluations,
        result.tuning_records,
        result.evaluation_records,
        result.comparisons,
        max_cases=args.max_cases,
    )
    path = save_error_analysis(
        analysis,
        Path(args.output) / "error_analysis.json",
    )

    print(f"errors: {analysis['total_errors']}  by_type: {analysis['by_type']}")
    print(
        "template errors: "
        f"{analysis['template_error_count']}  "
        f"template case included: {analysis['template_case_included']}"
    )
    for case in analysis["cases"]:
        score = "None" if case["score"] is None else f"{case['score']:.3f}"
        print(
            f"- {case['method_id']} {case['pair_id']} ({case['split']}, "
            f"{case['derivation_relation']}, {case['error_type']}, score={score})"
        )
    print(f"saved: {path}")
    return 0


def _ml(args: argparse.Namespace) -> int:
    result = run_experiment(
        args.corpus,
        ngram_size=args.ngram_size,
        max_features=args.max_features,
    )
    save_experiment(result, args.output)

    analysis = analyze_errors(
        result.method_evaluations,
        result.tuning_records,
        result.evaluation_records,
        result.comparisons,
    )
    error_path = save_error_analysis(
        analysis,
        Path(args.output) / "error_analysis.json",
    )

    ml_results = run_ml_experiment(
        result.feature_rows,
        result.method_evaluations,
        seeds=args.seeds,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
    )
    paths = save_ml_results(ml_results, args.output)

    print("method                              threshold   precision  recall   f1      auc")
    for row in ml_results["comparison"]:
        auc = "n/a" if row["evaluation_auc"] is None else f"{row['evaluation_auc']:.3f}"
        print(
            f"{row['method_id']:<35} {row['threshold_selected_on_tuning']:>9.3f} "
            f"{row['evaluation_precision']:>11.3f} "
            f"{row['evaluation_recall']:>7.3f} "
            f"{row['evaluation_f1']:>6.3f} {auc:>7}"
        )

    instability = ml_results["instability"]
    bootstrap = instability["bootstrap"].get("f1") or {}
    grouped = instability["grouped_cv_by_family"].get("f1") or {}
    if bootstrap.get("mean") is None:
        print("\nbootstrap F1: n/a")
    else:
        print(
            "\nbootstrap F1: "
            f"mean={bootstrap['mean']:.3f} std={bootstrap['std']:.3f}"
        )
    if grouped.get("mean") is None:
        print("grouped CV F1: n/a")
    else:
        print(
            "grouped CV F1: "
            f"mean={grouped['mean']:.3f} std={grouped['std']:.3f}"
        )

    artifacts = [
        error_path,
        paths["results"],
        paths["predictions"],
        paths["split"],
    ]
    logging_payload = dict(ml_results)
    logging_payload["report"] = result.summary["report"]
    logging_info = log_stage_b(
        logging_payload,
        artifacts,
        output_dir=args.output,
    )
    print(f"\nml results: {paths['results']}")
    print(f"ml predictions: {paths['predictions']}")
    print(f"ml split: {paths['split']}")
    print(f"error analysis: {error_path}")
    print(f"tracking: {logging_info['backend']}", end="")
    if logging_info.get("path"):
        print(f" ({logging_info['path']})")
    elif logging_info.get("run_id"):
        print(f" (run_id={logging_info['run_id']})")
    else:
        print()
    if logging_info.get("summary_path"):
        print(f"mlflow summary: {logging_info['summary_path']}")
    return 0


def _template_demo(args: argparse.Namespace) -> int:
    payload = run_template_change_demo(
        args.corpus,
        assignment_id=args.assignment,
        added_block=args.added_block or DEFAULT_ADDED_BLOCK,
    )
    path = save_template_change_demo(
        payload,
        Path(args.output) / "template_change_demo.json",
    )

    change = payload["template_change"]
    print(f"report version: {payload['report_version']}")
    print(f"assignment: {payload['assignment_id']}")
    print(
        "config hash: "
        f"{change['before_config_hash']} -> {change['after_config_hash']}"
    )
    print(
        "template fingerprint: "
        f"{change['before_template_fingerprint']} -> "
        f"{change['after_template_fingerprint']}"
    )
    print(f"corpus unchanged: {change['corpus_unchanged']}")
    for method_id, metrics in payload["aggregate"]["methods"].items():
        print(
            f"- {method_id}: mean score "
            f"{metrics['mean_score_before']:.3f} -> "
            f"{metrics['mean_score_after']:.3f}, "
            f"review_required {metrics['review_required_before']} -> "
            f"{metrics['review_required_after']}"
        )
    print(f"changed decisions: {len(payload['changed_review_decisions'])}")
    print(f"saved: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Similarity-анализ ЛР2: baseline, TF-IDF и retrieval"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="сравнить методы на корпусе")
    evaluate.add_argument("--corpus", default="data/corpus")
    evaluate.add_argument("--output", default="artifacts/lab2")
    evaluate.add_argument("--ngram-size", type=int, default=3)
    evaluate.add_argument("--max-features", type=int, default=5000)
    evaluate.add_argument("--json", action="store_true", help="напечатать полный summary")
    evaluate.set_defaults(handler=_evaluate)

    retrieve = subparsers.add_parser("retrieve", help="retrieval кандидатов для LLM-feedback")
    retrieve.add_argument("--corpus", default="data/corpus")
    retrieve.add_argument("--submission", required=True)
    retrieve.add_argument("--top-k", type=int, default=5)
    retrieve.set_defaults(handler=_retrieve)

    errors = subparsers.add_parser("errors", help="разбор FP/FN, включая общий шаблон")
    errors.add_argument("--corpus", default="data/corpus")
    errors.add_argument("--output", default="artifacts/lab2")
    errors.add_argument("--ngram-size", type=int, default=3)
    errors.add_argument("--max-features", type=int, default=5000)
    errors.add_argument("--max-cases", type=int, default=6)
    errors.set_defaults(handler=_errors)

    ml = subparsers.add_parser("ml", help="ML-классификатор против threshold baseline")
    ml.add_argument("--corpus", default="data/corpus")
    ml.add_argument("--output", default="artifacts/lab2")
    ml.add_argument("--ngram-size", type=int, default=3)
    ml.add_argument("--max-features", type=int, default=5000)
    ml.add_argument("--seeds", type=int, default=20)
    ml.add_argument("--learning-rate", type=float, default=0.2)
    ml.add_argument("--epochs", type=int, default=600)
    ml.add_argument("--l2", type=float, default=0.05)
    ml.set_defaults(handler=_ml)

    template_demo = subparsers.add_parser(
        "template-demo",
        help="показать влияние нового разрешённого шаблона",
    )
    template_demo.add_argument("--corpus", default="data/corpus")
    template_demo.add_argument("--output", default="artifacts/lab2")
    template_demo.add_argument("--assignment", default="hw_template")
    template_demo.add_argument(
        "--added-block",
        default="",
        help="новый блок шаблона; по умолчанию демонстрационный count_long",
    )
    template_demo.set_defaults(handler=_template_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows-консоль часто использует cp1251; JSON с русским текстом иначе
    # падает с UnicodeEncodeError.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

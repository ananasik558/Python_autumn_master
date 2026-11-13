"""Similarity-анализ ЛР2: baseline, TF-IDF, пары, оценка и retrieval."""

from .baseline import MatchEvidence, PairComparison, TemplateEvidence, TokenNGramBaseline
from .corpus import (
    CorpusFormatError,
    CorpusPair,
    CorpusSubmission,
    CorpusSummary,
    iter_corpus_pairs,
    iter_corpus_submissions,
    iter_jsonl,
    load_assignments,
    summarize_corpus,
    write_jsonl,
)
from .evaluation import (
    ClassificationMetrics,
    MethodEvaluation,
    ScoredPair,
    classification_metrics,
    evaluate_method,
    relation_to_label,
    review_required,
    roc_auc,
    select_threshold,
)
from .error_analysis import (
    ErrorCase,
    analyze_errors,
    collect_errors,
    save_error_analysis,
    summarize_errors,
)
from .features import build_pair_features
from .ml import (
    FEATURE_NAMES,
    LogisticRegression,
    ModelBundle,
    run_ml_experiment,
    save_ml_results,
    train_model,
    validate_model_scores,
)
from .mlflow_logging import log_stage_b
from .pairs import (
    PairGenerationStats,
    iter_candidate_pairs,
    iter_candidate_pairs_streaming,
    pair_key,
)
from .retrieval import (
    FeedbackDraft,
    RetrievalFeedbackGenerator,
    RetrievedCandidate,
    TfidfIndex,
)
from .template_demo import (
    DEFAULT_ADDED_BLOCK,
    run_template_change_demo,
    save_template_change_demo,
)
from .tfidf import TfidfSimilarityMethod, TfidfVectorizer, cosine_similarity

__all__ = [
    "ClassificationMetrics",
    "CorpusFormatError",
    "CorpusPair",
    "CorpusSubmission",
    "CorpusSummary",
    "DEFAULT_ADDED_BLOCK",
    "ErrorCase",
    "FEATURE_NAMES",
    "FeedbackDraft",
    "LogisticRegression",
    "MatchEvidence",
    "MethodEvaluation",
    "ModelBundle",
    "PairComparison",
    "PairGenerationStats",
    "RetrievalFeedbackGenerator",
    "RetrievedCandidate",
    "ScoredPair",
    "TemplateEvidence",
    "TfidfIndex",
    "TfidfSimilarityMethod",
    "TfidfVectorizer",
    "TokenNGramBaseline",
    "analyze_errors",
    "build_pair_features",
    "classification_metrics",
    "collect_errors",
    "cosine_similarity",
    "evaluate_method",
    "iter_candidate_pairs",
    "iter_candidate_pairs_streaming",
    "iter_corpus_pairs",
    "iter_corpus_submissions",
    "iter_jsonl",
    "load_assignments",
    "log_stage_b",
    "pair_key",
    "relation_to_label",
    "review_required",
    "roc_auc",
    "run_ml_experiment",
    "run_template_change_demo",
    "save_error_analysis",
    "save_ml_results",
    "save_template_change_demo",
    "select_threshold",
    "summarize_corpus",
    "summarize_errors",
    "train_model",
    "validate_model_scores",
    "write_jsonl",
]

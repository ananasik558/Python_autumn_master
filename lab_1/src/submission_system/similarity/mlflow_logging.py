"""Логирование этапа Б в MLflow.

Параметры, метрики и артефакты (включая разбиение tuning/evaluation) пишутся в
локальный ``mlruns``. Если пакета нет, создаётся JSON-fallback: эксперимент
остаётся воспроизводимым, но честно помечается, что реальный MLflow-run не
выполнялся.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _flatten(prefix: str, value: Any, target: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, target)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        target[prefix] = value
    elif isinstance(value, bool):
        target[prefix] = float(value)


def build_params(ml_results: Mapping[str, Any]) -> dict[str, Any]:
    dataset = ml_results.get("dataset") or {}
    hyper = ml_results.get("hyperparameters") or {}
    split = ml_results.get("split") or {}
    report = ml_results.get("report") or {}
    tuning_split = split.get("tuning") or {}
    evaluation_split = split.get("evaluation") or {}
    params = {
        "method_id": ml_results.get("method_id"),
        "features": ",".join(ml_results.get("features") or []),
        "features.count": len(ml_results.get("features") or []),
        "report.version": report.get("report_version"),
        "report.config_hash": report.get("config_hash"),
        "report.corpus_fingerprint": report.get("corpus_fingerprint"),
        "report.template_fingerprint": report.get("template_fingerprint"),
        "dataset.tuning_pairs": dataset.get("tuning_pairs"),
        "dataset.evaluation_pairs": dataset.get("evaluation_pairs"),
        "dataset.tuning_families": dataset.get("tuning_families"),
        "dataset.evaluation_families": dataset.get("evaluation_families"),
        "split.split_unit": split.get("split_unit"),
        "split.families_disjoint": split.get("families_disjoint"),
        "split.tuning.pairs": len(tuning_split.get("pairs") or []),
        "split.evaluation.pairs": len(evaluation_split.get("pairs") or []),
        "split.tuning.families": ",".join(tuning_split.get("families") or []),
        "split.evaluation.families": ",".join(
            evaluation_split.get("families") or []
        ),
        "hyperparameters.learning_rate": hyper.get("learning_rate"),
        "hyperparameters.epochs": hyper.get("epochs"),
        "hyperparameters.l2": hyper.get("l2"),
        "hyperparameters.bootstrap_seeds": hyper.get("bootstrap_seeds"),
        "threshold_selected_on_tuning": ml_results.get(
            "threshold_selected_on_tuning"
        ),
    }
    # MLflow не принимает None в params; отсутствующие значения просто не
    # логируются, чтобы не подменять их пустой строкой.
    return {key: value for key, value in params.items() if value is not None}


def build_metrics(ml_results: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, Any] = {}
    for split_name in ("tuning", "evaluation"):
        split_metrics = ml_results.get(split_name) or {}
        for name in (
            "precision",
            "recall",
            "f1",
            "accuracy",
            "review_rate",
            "threshold",
        ):
            if name in split_metrics and split_metrics[name] is not None:
                metrics[f"{split_name}.{name}"] = split_metrics[name]
        confusion = split_metrics.get("confusion_matrix") or {}
        for name, value in confusion.items():
            metrics[f"{split_name}.{name}"] = value
    if ml_results.get("evaluation_auc") is not None:
        metrics["evaluation.auc"] = ml_results["evaluation_auc"]

    instability = ml_results.get("instability") or {}
    for name, block in instability.items():
        for metric_name in ("precision", "recall", "f1", "review_rate"):
            summary = (block or {}).get(metric_name) or {}
            for stat in ("mean", "std", "min", "max"):
                value = summary.get(stat)
                if value is not None:
                    metrics[f"instability.{name}.{metric_name}.{stat}"] = value
    return metrics


def build_tags(ml_results: Mapping[str, Any]) -> dict[str, str]:
    split = ml_results.get("split") or {}
    report = ml_results.get("report") or {}
    return {
        "stage": "B",
        "task": "review_required",
        "method": str(ml_results.get("method_id") or ""),
        "report_version": str(report.get("report_version") or ""),
        "split_unit": str(split.get("split_unit") or ""),
        "families_disjoint": str(bool(split.get("families_disjoint"))).lower(),
        "data": "educational_synthetic_corpus",
    }


def _artifact_subdir(path: Path) -> str:
    """Раскладывает артефакты по смысловым папкам внутри run."""

    name = path.name
    if name.startswith("ml_split"):
        return "split"
    if name.startswith("ml_results") or name.startswith("ml_predictions"):
        return "model"
    if name.startswith("error_analysis"):
        return "errors"
    return "artifacts"


def log_stage_b(
    ml_results: Mapping[str, Any],
    artifact_paths: Sequence[str | Path],
    *,
    output_dir: str | Path = "artifacts/lab2",
    experiment_name: str = "lab2_similarity",
    run_name: str = "stage-b-ml-vs-threshold",
) -> dict[str, Any]:
    """Пишет run в MLflow или сохраняет локальный fallback."""

    params = build_params(ml_results)
    metrics = build_metrics(ml_results)
    artifact_list = [str(path) for path in artifact_paths]

    try:
        import mlflow  # type: ignore[import-not-found]
    except ImportError:
        fallback = {
            "backend": "json-fallback",
            "reason": "MLflow не установлен; сохранены params/metrics/artifacts локально",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_name": experiment_name,
            "run_name": run_name,
            "params": params,
            "metrics": metrics,
            "artifacts": artifact_list,
        }
        path = Path(output_dir) / "mlflow_fallback.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(fallback, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "backend": "json-fallback",
            "path": str(path),
            "reason": fallback["reason"],
        }

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(build_tags(ml_results))
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        logged_artifacts: list[dict[str, str]] = []
        for artifact in artifact_list:
            artifact_path = Path(artifact)
            if not artifact_path.exists():
                continue
            subdir = _artifact_subdir(artifact_path)
            mlflow.log_artifact(str(artifact_path), artifact_path=subdir)
            logged_artifacts.append(
                {
                    "local_path": str(artifact_path),
                    "artifact_path": f"{subdir}/{artifact_path.name}",
                }
            )

        run_summary = {
            "backend": "mlflow",
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment_name": experiment_name,
            "experiment_id": run.info.experiment_id,
            "run_id": run.info.run_id,
            "artifact_uri": run.info.artifact_uri,
            "params": params,
            "metrics": metrics,
            "artifacts": logged_artifacts,
        }
        summary_path = Path(output_dir) / "mlflow_run.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(summary_path), artifact_path="tracking")

        return {
            "backend": "mlflow",
            "run_id": run.info.run_id,
            "experiment_name": experiment_name,
            "experiment_id": run.info.experiment_id,
            "tracking_uri": mlflow.get_tracking_uri(),
            "artifact_uri": run.info.artifact_uri,
            "summary_path": str(summary_path),
        }

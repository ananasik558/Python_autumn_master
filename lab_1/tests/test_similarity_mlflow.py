"""Интеграционная проверка MLflow: params, metrics и split-артефакт.

Тест пропускается, если optional-зависимость `mlflow` не установлена.
Tracking URI направляется во временную папку, чтобы не засорять рабочий
каталог и не зависеть от предыдущих run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from submission_system.similarity.experiment import run_experiment
from submission_system.similarity.ml import run_ml_experiment
from submission_system.similarity.mlflow_logging import log_stage_b

mlflow = pytest.importorskip("mlflow")

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


def test_mlflow_stores_params_metrics_and_split_artifact(tmp_path):
    experiment = run_experiment(CORPUS)
    ml_result = run_ml_experiment(
        experiment.feature_rows,
        experiment.method_evaluations,
        seeds=2,
        epochs=100,
    )
    ml_result["report"] = {
        "report_version": "test-version",
        "config_hash": "test-config",
        "corpus_fingerprint": "test-corpus",
        "template_fingerprint": "test-template",
    }

    split_path = tmp_path / "ml_split.json"
    split_path.write_text(
        json.dumps(ml_result["split"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    tracking_db = tmp_path / "mlflow.db"
    artifact_dir = tmp_path / "artifacts"
    tracking_uri = f"sqlite:///{tracking_db.as_posix()}"
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "lab2_similarity_test"
    try:
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_dir.as_uri(),
            )
        info = log_stage_b(
            ml_result,
            [split_path],
            output_dir=tmp_path,
            experiment_name=experiment_name,
            run_name="test-stage-b",
        )
        assert info["backend"] == "mlflow"
        assert info["summary_path"]

        client = mlflow.MlflowClient()
        run = client.get_run(info["run_id"])
        assert run.data.params["split.split_unit"] == "family_id"
        assert run.data.params["split.families_disjoint"].lower() == "true"
        assert run.data.params["features.count"] == str(len(ml_result["features"]))
        assert run.data.params["report.version"] == "test-version"
        assert run.data.params["report.config_hash"] == "test-config"
        assert "evaluation.f1" in run.data.metrics
        assert "instability.bootstrap.f1.mean" in run.data.metrics

        split_artifacts = client.list_artifacts(info["run_id"], "split")
        assert any(
            artifact.path.endswith("ml_split.json")
            for artifact in split_artifacts
        )
    finally:
        mlflow.set_tracking_uri(previous_uri)

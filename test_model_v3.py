import json
from pathlib import Path

import joblib

from model_v3 import FEATURES, MODEL_NAME, MODEL_VERSION, train_and_save


def test_model_v3_artifact_generation(tmp_path):
    model_path = tmp_path / "model_v3.joblib"
    metadata_path = tmp_path / "model_v3_metadata.json"

    model, metadata = train_and_save(
        model_path=str(model_path),
        metadata_path=str(metadata_path),
    )

    assert model_path.exists()
    assert metadata_path.exists()
    assert metadata["model_version"] == MODEL_VERSION
    assert metadata["model_name"] == MODEL_NAME
    assert metadata["features"] == FEATURES
    assert metadata["holdout_metrics"]["r2"] > 0

    loaded = joblib.load(model_path)
    assert list(loaded.feature_names_in_) == FEATURES

    stored = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert stored["source_commit"]
    assert stored["usable_rows"] > 0

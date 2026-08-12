"""Reproducible training and artifact generation for Model v3."""

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_VERSION = "v3"
MODEL_NAME = "Quality_Size_Basics"
FEATURES = [
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "OverallQual",
    "TotalBsmtSF",
    "GarageCars",
]
TARGET = "SalePrice"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(data_path="train.csv"):
    data = pd.read_csv(data_path)
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return data[required].dropna().copy()


def evaluate_holdout(data):
    train, test = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model = LinearRegression()
    model.fit(train[FEATURES], train[TARGET])
    predictions = model.predict(test[FEATURES])
    return {
        "rmse": float(mean_squared_error(test[TARGET], predictions) ** 0.5),
        "mae": float(mean_absolute_error(test[TARGET], predictions)),
        "r2": float(r2_score(test[TARGET], predictions)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }


def train_and_save(data_path="train.csv", model_path="model_v3.joblib", metadata_path="model_v3_metadata.json"):
    data = load_data(data_path)
    holdout_metrics = evaluate_holdout(data)

    final_model = LinearRegression()
    final_model.fit(data[FEATURES], data[TARGET])
    joblib.dump(final_model, model_path)

    metadata = {
        "model_version": MODEL_VERSION,
        "model_name": MODEL_NAME,
        "features": FEATURES,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "dataset_path": data_path,
        "usable_rows": int(len(data)),
        "holdout_metrics": holdout_metrics,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.getenv("GITHUB_SHA", "local"),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "artifact": Path(model_path).name,
    }

    Path(metadata_path).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return final_model, metadata


if __name__ == "__main__":
    _, metadata = train_and_save()
    print(f"Model v3 artifact created: {metadata['artifact']}")
    print(f"Features: {', '.join(FEATURES)}")
    print(
        "Locked holdout: "
        f"RMSE={metadata['holdout_metrics']['rmse']:.2f}, "
        f"MAE={metadata['holdout_metrics']['mae']:.2f}, "
        f"R2={metadata['holdout_metrics']['r2']:.4f}"
    )

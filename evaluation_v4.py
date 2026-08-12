"""Model v4: nested feature-selection evaluation with a locked outer test set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = Path("train.csv")
TARGET = "SalePrice"
RANDOM_STATE = 42
OUTER_TEST_SIZE = 0.20
INNER_FOLDS = 5

BASELINE_FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath"]
CANDIDATE_FEATURES = {
    "Baseline_3": BASELINE_FEATURES,
    "Add_OverallQual": BASELINE_FEATURES + ["OverallQual"],
    "Quality_Size_Basics": ["OverallQual", "GrLivArea", "TotalBsmtSF"],
    "Quality_Size_Age": ["OverallQual", "GrLivArea", "TotalBsmtSF", "YearBuilt"],
    "Quality_Size_Garage": ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars"],
    "Quality_Size_Floor": ["OverallQual", "GrLivArea", "TotalBsmtSF", "1stFlrSF"],
}


def _pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression()),
    ])


def _dataset_hash(path: Path = DATA_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    required = sorted({TARGET, *[f for fs in CANDIDATE_FEATURES.values() for f in fs]})
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[[*sorted({f for fs in CANDIDATE_FEATURES.values() for f in fs}), TARGET]].copy(), df[TARGET].copy()


def run_v4() -> dict:
    df = pd.read_csv(DATA_PATH)
    required = sorted({TARGET, *[f for fs in CANDIDATE_FEATURES.values() for f in fs]})
    df = df[required].dropna(subset=[TARGET]).reset_index(drop=True)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_dev, X_outer, y_dev, y_outer = train_test_split(
        X, y, test_size=OUTER_TEST_SIZE, random_state=RANDOM_STATE
    )

    inner_cv = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner_results = []
    for name, features in CANDIDATE_FEATURES.items():
        scores = cross_validate(
            _pipeline(), X_dev[features], y_dev, cv=inner_cv,
            scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=None,
        )
        inner_results.append({
            "name": name,
            "features": features,
            "cv_rmse": float(-scores["test_rmse"].mean()),
            "cv_mae": float(-scores["test_mae"].mean()),
            "cv_r2": float(scores["test_r2"].mean()),
        })

    inner_results.sort(key=lambda r: r["cv_rmse"])
    winner = inner_results[0]

    final_model = _pipeline()
    final_model.fit(X_dev[winner["features"]], y_dev)
    predictions = final_model.predict(X_outer[winner["features"]])

    outer = {
        "rmse": float(np.sqrt(mean_squared_error(y_outer, predictions))),
        "mae": float(mean_absolute_error(y_outer, predictions)),
        "r2": float(r2_score(y_outer, predictions)),
    }

    result = {
        "model_version": "v4-experiment",
        "outer_test_random_state": RANDOM_STATE,
        "outer_test_size": OUTER_TEST_SIZE,
        "inner_cv_folds": INNER_FOLDS,
        "selected_model": winner["name"],
        "selected_features": winner["features"],
        "inner_cv": inner_results,
        "outer_test": outer,
        "dataset_sha256": _dataset_hash(),
    }
    return result


def main() -> None:
    result = run_v4()
    print(json.dumps(result, indent=2))
    Path("model_v4_evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

# Evaluation v3: controlled numeric feature engineering against the v2 baseline

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

TARGET = "SalePrice"
BASELINE_FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath"]
CANDIDATE_FEATURE_SETS = {
    "Baseline_3": BASELINE_FEATURES,
    "Add_OverallQual": BASELINE_FEATURES + ["OverallQual"],
    "Add_TotalBsmtSF": BASELINE_FEATURES + ["TotalBsmtSF"],
    "Add_GarageCars": BASELINE_FEATURES + ["GarageCars"],
    "Add_1stFlrSF": BASELINE_FEATURES + ["1stFlrSF"],
    "Add_YearBuilt": BASELINE_FEATURES + ["YearBuilt"],
    "Quality_Size_Basics": BASELINE_FEATURES + ["OverallQual", "TotalBsmtSF", "GarageCars"],
    "Quality_Size_Age": BASELINE_FEATURES + ["OverallQual", "TotalBsmtSF", "YearBuilt"],
}


def load_data(data_path="train.csv"):
    all_features = sorted({feature for features in CANDIDATE_FEATURE_SETS.values() for feature in features})
    required = set(all_features) | {TARGET}
    data = pd.read_csv(data_path)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return data[list(required)].dropna().copy()


def cross_validate_feature_set(data, features, n_splits=5):
    X = data[features]
    y = data[TARGET]
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        LinearRegression(),
        X,
        y,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
        },
    )
    return {
        "cv_rmse": float(np.sqrt(-scores["test_mse"].mean())),
        "cv_mae": float(-scores["test_mae"].mean()),
        "cv_r2": float(scores["test_r2"].mean()),
    }


def evaluate_candidates(data_path="train.csv"):
    data = load_data(data_path)
    results = {}
    for name, features in CANDIDATE_FEATURE_SETS.items():
        metrics = cross_validate_feature_set(data, features)
        results[name] = {"features": features, **metrics}

    baseline = results["Baseline_3"]
    for name, metrics in results.items():
        metrics["rmse_improvement_pct"] = (
            (baseline["cv_rmse"] - metrics["cv_rmse"]) / baseline["cv_rmse"] * 100
        )
        metrics["r2_delta"] = metrics["cv_r2"] - baseline["cv_r2"]

    return results


def select_best(results):
    return min(results.items(), key=lambda item: item[1]["cv_rmse"])


def main():
    results = evaluate_candidates()
    baseline = results["Baseline_3"]
    best_name, best = select_best(results)

    print("Evaluation v3 - controlled feature engineering")
    print(f"Baseline CV RMSE: {baseline['cv_rmse']:.2f}")
    print(f"Baseline CV R2: {baseline['cv_r2']:.4f}")
    print("\nCandidate results:")
    for name, metrics in sorted(results.items(), key=lambda item: item[1]["cv_rmse"]):
        print(
            f"{name}: CV RMSE={metrics['cv_rmse']:.2f}, "
            f"CV MAE={metrics['cv_mae']:.2f}, CV R2={metrics['cv_r2']:.4f}, "
            f"RMSE improvement={metrics['rmse_improvement_pct']:.2f}%"
        )

    if best_name == "Baseline_3":
        print("\nNo candidate improves the v2 baseline.")
    else:
        print(f"\nBest candidate: {best_name}")
        print(f"CV RMSE improvement: {best['rmse_improvement_pct']:.2f}%")


if __name__ == "__main__":
    main()

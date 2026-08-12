# Evaluation v2 for the house-price regression baseline

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

TARGET = "SalePrice"
FEATURE_SETS = {
    "GrLivArea": ["GrLivArea"],
    "GrLivArea_Bedroom_FullBath": ["GrLivArea", "BedroomAbvGr", "FullBath"],
}


def load_data(data_path="train.csv"):
    data = pd.read_csv(data_path)
    required = set([TARGET]) | set(sum(FEATURE_SETS.values(), []))
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return data.dropna(subset=sorted(required)).copy()


def baseline_predictions(y_train, y_test):
    return np.full(len(y_test), y_train.mean())


def evaluate_feature_set(data, features, random_state=42):
    X = data[features]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    baseline = baseline_predictions(y_train, y_test)

    return {
        "features": features,
        "mse": mean_squared_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "baseline_rmse": np.sqrt(mean_squared_error(y_test, baseline)),
        "baseline_mae": mean_absolute_error(y_test, baseline),
        "baseline_r2": r2_score(y_test, baseline),
    }


def cross_validate_feature_set(data, features, n_splits=5):
    X = data[features]
    y = data[TARGET]
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        LinearRegression(),
        X,
        y,
        cv=cv,
        scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error"},
    )
    return {
        "cv_r2_mean": scores["test_r2"].mean(),
        "cv_mae_mean": -scores["test_mae"].mean(),
        "cv_rmse_mean": np.sqrt(-scores["test_mse"].mean()),
    }


def run_evaluation(data_path="train.csv"):
    data = load_data(data_path)
    results = {}
    for name, features in FEATURE_SETS.items():
        holdout = evaluate_feature_set(data, features)
        cv = cross_validate_feature_set(data, features)
        results[name] = {**holdout, **cv}
    return results


def main():
    results = run_evaluation()
    print("Evaluation v2")
    for name, metrics in results.items():
        print(f"\n{name}")
        print(f"Holdout RMSE: {metrics['rmse']:.2f}")
        print(f"Holdout MAE: {metrics['mae']:.2f}")
        print(f"Holdout R2: {metrics['r2']:.4f}")
        print(f"Baseline RMSE: {metrics['baseline_rmse']:.2f}")
        print(f"CV RMSE: {metrics['cv_rmse_mean']:.2f}")
        print(f"CV MAE: {metrics['cv_mae_mean']:.2f}")
        print(f"CV R2: {metrics['cv_r2_mean']:.4f}")


if __name__ == "__main__":
    main()

# Final locked comparison of the v2 baseline and v3 champion on a fixed holdout.

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TARGET = "SalePrice"
BASELINE_FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath"]
CHAMPION_FEATURES = BASELINE_FEATURES + ["OverallQual", "TotalBsmtSF", "GarageCars"]


def evaluate_feature_set(data, features, test_indices):
    train = data.drop(index=test_indices)
    test = data.loc[test_indices]

    model = LinearRegression()
    model.fit(train[features], train[TARGET])
    predictions = model.predict(test[features])

    return {
        "rmse": float(mean_squared_error(test[TARGET], predictions) ** 0.5),
        "mae": float(mean_absolute_error(test[TARGET], predictions)),
        "r2": float(r2_score(test[TARGET], predictions)),
    }


def run_final_validation(data_path="train.csv"):
    data = pd.read_csv(data_path)
    required = set(BASELINE_FEATURES + CHAMPION_FEATURES + [TARGET])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data[list(required)].dropna().copy()
    train_indices, test_indices = train_test_split(
        data.index, test_size=0.2, random_state=42
    )

    baseline = evaluate_feature_set(data, BASELINE_FEATURES, test_indices)
    champion = evaluate_feature_set(data, CHAMPION_FEATURES, test_indices)

    champion["rmse_improvement_pct"] = (
        (baseline["rmse"] - champion["rmse"]) / baseline["rmse"] * 100
    )
    champion["r2_delta"] = champion["r2"] - baseline["r2"]

    return baseline, champion


def main():
    baseline, champion = run_final_validation()
    print("Final locked holdout comparison")
    print(
        f"Baseline: RMSE={baseline['rmse']:.2f}, "
        f"MAE={baseline['mae']:.2f}, R2={baseline['r2']:.4f}"
    )
    print(
        f"Quality_Size_Basics: RMSE={champion['rmse']:.2f}, "
        f"MAE={champion['mae']:.2f}, R2={champion['r2']:.4f}"
    )
    print(f"RMSE improvement: {champion['rmse_improvement_pct']:.2f}%")
    print(f"R2 delta: {champion['r2_delta']:.4f}")


if __name__ == "__main__":
    main()

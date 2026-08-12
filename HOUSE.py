# House Price Prediction using Linear Regression

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath"]
TARGET = "SalePrice"


def train_model(data_path="train.csv"):
    """Load the dataset, train the model, and return evaluation metrics."""
    data = pd.read_csv(data_path)
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data[required_columns].dropna()
    if data.empty:
        raise ValueError("Dataset contains no usable rows after removing missing values")

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, y_test, y_pred, mse, r2


def main():
    model, y_test, y_pred, mse, r2 = train_model()

    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)

    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual House Price")
    plt.ylabel("Predicted House Price")
    plt.title("Actual vs Predicted House Prices")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png")
    plt.close()


if __name__ == "__main__":
    main()

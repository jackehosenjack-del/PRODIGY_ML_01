from HOUSE import train_model


def test_model_training():
    model, y_test, y_pred, mse, r2 = train_model("train.csv")

    assert len(y_test) > 0
    assert len(y_pred) == len(y_test)
    assert mse >= 0
    assert r2 <= 1
    assert model.coef_.shape[0] == 3

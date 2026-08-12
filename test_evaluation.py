import math

from evaluation import run_evaluation


def test_evaluation_v2_produces_valid_metrics():
    results = run_evaluation()
    assert results

    for metrics in results.values():
        for key in ["mse", "rmse", "mae", "r2", "baseline_rmse", "baseline_mae", "baseline_r2", "cv_rmse_mean", "cv_mae_mean", "cv_r2_mean"]:
            assert math.isfinite(metrics[key])

        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["baseline_rmse"] >= 0
        assert metrics["baseline_mae"] >= 0
        assert metrics["r2"] <= 1
        assert metrics["cv_r2_mean"] <= 1


def test_three_feature_model_beats_mean_baseline_on_holdout():
    results = run_evaluation()
    metrics = results["GrLivArea_Bedroom_FullBath"]
    assert metrics["rmse"] < metrics["baseline_rmse"]
    assert metrics["mae"] < metrics["baseline_mae"]

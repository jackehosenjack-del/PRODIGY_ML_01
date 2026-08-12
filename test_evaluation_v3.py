import math

from evaluation_v3 import BASELINE_FEATURES, evaluate_candidates, select_best


def test_evaluation_v3_produces_valid_candidate_metrics():
    results = evaluate_candidates()
    assert len(results) >= 2
    assert "Baseline_3" in results

    for metrics in results.values():
        for key in ["cv_rmse", "cv_mae", "cv_r2", "rmse_improvement_pct", "r2_delta"]:
            assert math.isfinite(metrics[key])
        assert metrics["cv_rmse"] >= 0
        assert metrics["cv_mae"] >= 0
        assert metrics["cv_r2"] <= 1
        assert metrics["features"][: len(BASELINE_FEATURES)] == BASELINE_FEATURES


def test_best_candidate_is_at_least_as_good_as_baseline():
    results = evaluate_candidates()
    best_name, best = select_best(results)
    baseline = results["Baseline_3"]

    assert best["cv_rmse"] <= baseline["cv_rmse"]
    assert best_name in results

import math

from final_validation import run_final_validation


def test_locked_champion_beats_baseline_on_holdout():
    baseline, champion = run_final_validation()

    for metrics in (baseline, champion):
        assert all(math.isfinite(metrics[key]) for key in ["rmse", "mae", "r2"])
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["r2"] <= 1

    assert champion["rmse"] < baseline["rmse"]
    assert champion["rmse_improvement_pct"] > 0

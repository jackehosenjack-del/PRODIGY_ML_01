import json
from pathlib import Path

from evaluation_v4 import (
    BASELINE_FEATURES,
    INNER_FOLDS,
    OUTER_TEST_SIZE,
    RANDOM_STATE,
    run_v4,
)


def test_v4_nested_evaluation_contract():
    result = run_v4()

    assert result["model_version"] == "v4-experiment"
    assert result["outer_test_random_state"] == RANDOM_STATE
    assert result["outer_test_size"] == OUTER_TEST_SIZE
    assert result["inner_cv_folds"] == INNER_FOLDS
    assert result["selected_features"]
    assert len(result["inner_cv"]) == 6
    assert result["outer_test"]["rmse"] > 0
    assert result["outer_test"]["mae"] > 0
    assert -1 <= result["outer_test"]["r2"] <= 1


def test_v3_baseline_is_available_for_comparison():
    result = run_v4()
    names = {row["name"] for row in result["inner_cv"]}
    assert "Baseline_3" in names
    assert BASELINE_FEATURES == ["GrLivArea", "BedroomAbvGr", "FullBath"]

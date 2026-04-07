import numpy as np
import pandas as pd

from causalopt import optimum_threshold, tradeoff_threshold

# ------------------------------------------------------------
# Shared data generator
# ------------------------------------------------------------


def _make_df(n=800, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "p_hat": rng.uniform(0, 1, size=n),
            "y": 1 + rng.normal(scale=0.3, size=n),
            "y1": 1 + rng.normal(scale=0.3, size=n),
            "y2": -0.5 + rng.normal(scale=0.3, size=n),
        }
    )


# ============================================================
# optimum_threshold
# ============================================================


def test_optimum_threshold_full_pipeline():
    rng = np.random.default_rng(8)
    n = 1000

    df = pd.DataFrame(
        {
            "y": 1 + rng.normal(size=n),
            "p_hat": rng.uniform(0, 1, size=n),
        }
    )

    threshold = 0.5

    out = optimum_threshold(
        df=df,
        outcome_col="y",
        prob_col="p_hat",
        threshold=threshold,
    )

    expected_keys = {
        "data_descriptives",
        "rd_results",
        "rdplot",
        "predictions",
        "welfare",
        "optimum_thresholds",
        "current_threshold",
    }
    assert expected_keys.issubset(out.keys())

    assert out["current_threshold"] == threshold
    assert isinstance(out["predictions"], pd.DataFrame)
    assert isinstance(out["welfare"], pd.DataFrame)
    assert isinstance(out["optimum_thresholds"], dict)


def test_optimum_threshold_threshold_shift_invariance():
    rng = np.random.default_rng(9)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=800),
            "p_hat": rng.uniform(0, 1, size=800),
        }
    )

    out1 = optimum_threshold(df, "y", "p_hat", threshold=0.3)
    out2 = optimum_threshold(df, "y", "p_hat", threshold=0.5)

    assert isinstance(out1["optimum_thresholds"]["Recommendation"], str)
    assert isinstance(out2["optimum_thresholds"]["Recommendation"], str)


# ============================================================
# tradeoff_threshold
# ============================================================


def test_tradeoff_threshold_single_outcome():
    df = _make_df(seed=7)

    out = tradeoff_threshold(
        df=df,
        outcomes=["y1"],
        prob_col="p_hat",
        threshold=0.5,
    )

    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {"x", "n", "p", "gain_y1"}
    assert len(out) > 0


def test_tradeoff_threshold_multiple_outcomes_merge():
    df = _make_df(seed=8)

    out = tradeoff_threshold(
        df=df,
        outcomes=["y1", "y2"],
        prob_col="p_hat",
        threshold=0.5,
    )

    assert isinstance(out, pd.DataFrame)
    assert "gain_y1" in out.columns
    assert "gain_y2" in out.columns

    for col in ("x", "n", "p"):
        assert col in out.columns


def test_tradeoff_threshold_consistent_support_across_outcomes():
    df = _make_df(seed=9)

    out = tradeoff_threshold(
        df=df,
        outcomes=["y1", "y2"],
        prob_col="p_hat",
        threshold=0.4,
    )

    assert out[["x", "n", "p"]].duplicated().sum() == 0


def test_tradeoff_threshold_handles_threshold_shift():
    df = _make_df(seed=10)

    out1 = tradeoff_threshold(
        df=df,
        outcomes=["y1"],
        prob_col="p_hat",
        threshold=0.3,
    )

    out2 = tradeoff_threshold(
        df=df,
        outcomes=["y1"],
        prob_col="p_hat",
        threshold=0.6,
    )

    assert np.all(np.isfinite(out1["gain_y1"]))
    assert np.all(np.isfinite(out2["gain_y1"]))

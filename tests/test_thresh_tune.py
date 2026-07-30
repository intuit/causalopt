import numpy as np
import pandas as pd

from causalopt.thresh_tune import (
    get_rd_objects,
    optim_thresh,
    predictions,
    welfare,
)

# ------------------------------------------------------------
# Test data generator
# ------------------------------------------------------------


def _make_data(n=800, tau=1.5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=n)
    eps = rng.normal(scale=0.2, size=n)

    y0 = 1 + 0.5 * x + 0.25 * x**2
    y = y0 + tau * (x >= 0) + eps

    return y, x


# ============================================================
# get_rd_objects
# ============================================================


def test_get_rd_objects_structure_and_keys():
    y, x = _make_data(seed=1)

    out = get_rd_objects(y=y, x=x, c=0)

    assert set(out.keys()) == {
        "rd_estimates",
        "rd_plot_objects",
        "b_l",
        "b_r",
        "v_l",
        "v_r",
        "bins",
        "estimates",
    }

    # estimates: [point, lb, ub]
    assert len(out["estimates"]) == 3
    assert out["estimates"][0] >= out["estimates"][1]
    assert out["estimates"][0] <= out["estimates"][2]

    # polynomial objects (1D coefficient vectors after s_Y contraction)
    assert out["b_l"].ndim == 1
    assert out["b_r"].ndim == 1
    assert out["b_l"].shape[0] == out["b_r"].shape[0]
    assert out["v_l"].shape[0] == out["v_l"].shape[1]
    assert out["v_r"].shape[0] == out["v_r"].shape[1]

    # bins dataframe
    bins = out["bins"]
    assert isinstance(bins, pd.DataFrame)
    for col in ("mean_x", "mean_y", "N", "ci_lower", "ci_upper"):
        assert col in bins.columns


# ============================================================
# predictions
# ============================================================


def test_predictions_output_dataframe_and_columns():
    y, x = _make_data(seed=2)
    rdres = get_rd_objects(y, x, c=0)

    df = predictions(rdres)

    assert isinstance(df, pd.DataFrame)

    expected_cols = {
        "x",
        "n",
        "p",
        "y_mean",
        "y_ci_l",
        "y_ci_r",
        "y_hat_l",
        "y_hat_lower_l",
        "y_hat_upper_l",
        "y_hat_r",
        "y_hat_lower_r",
        "y_hat_upper_r",
    }
    assert expected_cols.issubset(df.columns)

    # probabilities sum to 1
    assert np.isclose(df["p"].sum(), 1.0)

    # prediction bands ordered
    assert np.all(df["y_hat_lower_l"] <= df["y_hat_upper_l"])
    assert np.all(df["y_hat_lower_r"] <= df["y_hat_upper_r"])


def test_predictions_shapes_consistent():
    y, x = _make_data(seed=3)
    rdres = get_rd_objects(y, x, c=0)

    df = predictions(rdres)

    assert len(df["x"]) == len(rdres["bins"])
    assert len(df["y_hat_l"]) == len(df["x"])
    assert len(df["y_hat_r"]) == len(df["x"])


# ============================================================
# welfare
# ============================================================


def test_welfare_output_structure_and_monotonicity():
    y, x = _make_data(seed=4)
    rdres = get_rd_objects(y, x, c=0)

    pred = predictions(rdres)
    welfare_df = welfare(pred, rdres)

    expected_cols = {
        "x",
        "n",
        "p",
        "w_optimum",
        "w_conservative",
        "w_aggressive",
        "welfare_optimum",
        "welfare_conservative",
        "welfare_aggressive",
    }
    assert expected_cols.issubset(welfare_df.columns)

    # welfare is cumulative
    assert welfare_df["welfare_optimum"].is_monotonic_increasing
    assert welfare_df["welfare_conservative"].is_monotonic_increasing
    assert welfare_df["welfare_aggressive"].is_monotonic_increasing


def test_welfare_sign_logic_runs_both_branches():
    # negative treatment effect
    y, x = _make_data(tau=-1.0, seed=5)
    rdres = get_rd_objects(y, x, c=0)

    pred = predictions(rdres)
    welfare_df = welfare(pred, rdres)

    assert "welfare_optimum" in welfare_df.columns


# ============================================================
# optim_thresh
# ============================================================


def test_optim_thresh_output_structure():
    y, x = _make_data(seed=6)
    rdres = get_rd_objects(y, x, c=0)

    pred = predictions(rdres)
    welfare_df = welfare(pred, rdres)

    res = optim_thresh(welfare_df, rdres, current_threshold=0.3)

    assert set(res.keys()) == {
        "Recommendation",
        "Thresholds",
        "Additional Gain",
    }

    assert set(res["Thresholds"].keys()) == {"Optimum", "Conservative", "Aggressive"}

    assert set(res["Additional Gain"].keys()) == {
        "Optimum",
        "Conservative",
        "Aggressive",
    }

    for v in res["Additional Gain"].values():
        assert np.isfinite(v)


def test_optim_thresh_recommendation_strings():
    y, x = _make_data(tau=2.0, seed=7)
    rdres = get_rd_objects(y, x, c=0)

    pred = predictions(rdres)
    welfare_df = welfare(pred, rdres)

    res = optim_thresh(welfare_df, rdres, current_threshold=0.0)

    assert isinstance(res["Recommendation"], str)
    assert len(res["Recommendation"]) > 0

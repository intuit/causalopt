import numpy as np
import pandas as pd
import pytest

from causalopt.thresh_tradeoff import (
    gains_eval,
    pred_tradeoff,
)
from causalopt.thresh_tune import get_rd_objects

# ------------------------------------------------------------
# Data generator
# ------------------------------------------------------------


def _make_data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "p_hat": rng.uniform(0, 1, size=n),
            "y1": 1 + rng.normal(scale=0.3, size=n),
            "y2": -0.5 + rng.normal(scale=0.3, size=n),
        }
    )
    return df


# ============================================================
# pred_tradeoff
# ============================================================


def test_pred_tradeoff_without_inputs_uses_bins():
    df = _make_data(seed=1)
    y = df["y1"]
    x = df["p_hat"] - 0.5

    res = get_rd_objects(y, x, c=0)
    pred = pred_tradeoff(res)

    assert isinstance(pred, pd.DataFrame)

    for col in ("x", "n", "p", "y_hat_l", "y_hat_r"):
        assert col in pred.columns

    # probabilities sum to 1
    assert np.isclose(pred["p"].sum(), 1.0)

    # shapes
    assert len(pred["x"]) == len(res["bins"])


def test_pred_tradeoff_with_inputs_overrides_bins():
    df = _make_data(seed=2)
    y = df["y1"]
    x = df["p_hat"] - 0.4

    res = get_rd_objects(y, x, c=0)

    # fake external support
    inputs = pd.DataFrame(
        {
            "x": np.linspace(-0.2, 0.2, 10),
            "n": np.repeat(100, 10),
            "p": np.repeat(0.1, 10),
        }
    )

    pred = pred_tradeoff(res, inputs=inputs)

    assert len(pred) == len(inputs)
    assert np.allclose(pred["x"], inputs["x"])
    assert np.allclose(pred["p"], inputs["p"])


def test_pred_tradeoff_left_right_predictions_finite():
    df = _make_data(seed=3)
    y = df["y1"]
    x = df["p_hat"] - 0.5

    res = get_rd_objects(y, x, c=0)
    pred = pred_tradeoff(res)

    assert np.all(np.isfinite(pred["y_hat_l"]))
    assert np.all(np.isfinite(pred["y_hat_r"]))


# ============================================================
# gains_eval
# ============================================================


def test_gains_eval_structure_and_columns():
    df = _make_data(seed=4)
    y = df["y1"]
    x = df["p_hat"] - 0.5

    res = get_rd_objects(y, x, c=0)
    pred = pred_tradeoff(res)

    w = gains_eval(pred, outcome="y1")

    assert isinstance(w, pd.DataFrame)
    assert set(w.columns) == {"x", "n", "p", "gain_y1"}


def test_gains_eval_current_threshold_zero_gain():
    # At x closest to zero, gain must be zero by construction
    df = _make_data(seed=5)
    y = df["y1"]
    x = df["p_hat"] - 0.5

    res = get_rd_objects(y, x, c=0)
    pred = pred_tradeoff(res)

    w = gains_eval(pred, "y1")

    idx0 = w["x"].abs().idxmin()
    assert w.loc[idx0, "gain_y1"] == pytest.approx(0.0)


def test_gains_eval_welfare_is_cumsum_and_zero_at_current_threshold():
    df = _make_data(seed=6)
    y = df["y1"]
    x = df["p_hat"] - 0.5

    res = get_rd_objects(y, x, c=0)
    pred = pred_tradeoff(res)

    w = gains_eval(pred, "y1")

    # reconstruct internal logic
    pred_sorted = pred.sort_values("x", ascending=False)
    w_outcome = (pred_sorted["y_hat_r"] - pred_sorted["y_hat_l"]) * pred_sorted["p"]
    welfare_manual = w_outcome.cumsum().to_numpy()

    # welfare column equals cumsum by construction
    np.testing.assert_allclose(
        w.sort_values("x", ascending=False)["gain_y1"].to_numpy(),
        pred_sorted["n"].to_numpy()
        * (welfare_manual - welfare_manual[np.argmin(np.abs(pred_sorted["x"]))]),
        rtol=1e-12,
        atol=1e-12,
    )

    # gain at current threshold is zero
    idx0 = w["x"].abs().idxmin()
    assert w.loc[idx0, "gain_y1"] == pytest.approx(0.0)

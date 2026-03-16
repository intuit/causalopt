import numpy as np
import pandas as pd
from plotnine.ggplot import ggplot

from causalopt.plots import (
    plot_thresh,
    rdd_impact,
)
from causalopt.thresh_tune import exc_optim_thresh

# ------------------------------------------------------------
# Test data generator
# ------------------------------------------------------------


def _make_df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "y": 1 + rng.normal(scale=0.3, size=n),
            "p_hat": rng.uniform(0, 1, size=n),
        }
    )


# ============================================================
# rdd_impact
# ============================================================


def test_rdd_impact_returns_ggplot():
    df = _make_df(seed=1)

    results = exc_optim_thresh(
        df=df,
        outcome_col="y",
        prob_col="p_hat",
        threshold=0.5,
    )

    g = rdd_impact(results, outcome_col="y")

    assert isinstance(g, ggplot)


def test_rdd_impact_contains_expected_layers():
    df = _make_df(seed=2)

    results = exc_optim_thresh(
        df=df,
        outcome_col="y",
        prob_col="p_hat",
        threshold=0.5,
    )

    g = rdd_impact(results, "y")

    # ggplot stores layers in a list
    layer_geoms = [type(layer.geom).__name__ for layer in g.layers]

    # required geoms
    assert "geom_point" in layer_geoms
    assert "geom_line" in layer_geoms
    assert "geom_ribbon" in layer_geoms
    assert "geom_vline" in layer_geoms


def test_rdd_impact_uses_correct_axis_labels():
    df = _make_df(seed=3)

    results = exc_optim_thresh(
        df=df,
        outcome_col="y",
        prob_col="p_hat",
        threshold=0.5,
    )

    g = rdd_impact(results, "y")

    labels = g.labels

    assert hasattr(labels, "x")
    assert hasattr(labels, "y")
    assert hasattr(labels, "title")
    assert hasattr(labels, "caption")

    assert "Centered Probability" in labels.x
    assert "Average y" in labels.y


def test_rdd_impact_respects_y_limits():
    df = _make_df(seed=4)

    results = exc_optim_thresh(
        df=df,
        outcome_col="y",
        prob_col="p_hat",
        threshold=0.5,
    )

    g = rdd_impact(results, "y")

    coord = g.coordinates
    assert coord is not None

    # coord_cartesian stores limits in coord.limits
    assert hasattr(coord, "limits")
    assert coord.limits.y is not None

    ymin, ymax = coord.limits.y
    assert ymin < ymax


# ============================================================
# plot_thresh
# ============================================================


def test_plot_thresh_returns_ggplot():
    df = _make_df(seed=5)

    results = exc_optim_thresh(
        df=df,
        outcome_col="y",
        prob_col="p_hat",
        threshold=0.5,
    )

    g = plot_thresh(results, "y")

    assert isinstance(g, ggplot)


def test_plot_thresh_contains_threshold_lines():
    df = _make_df(seed=6)

    results = exc_optim_thresh(
        df=df,
        outcome_col="y",
        prob_col="p_hat",
        threshold=0.5,
    )

    g = plot_thresh(results, "y")

    layer_geoms = [type(layer.geom).__name__ for layer in g.layers]

    # threshold visualization relies heavily on vertical lines
    assert "geom_vline" in layer_geoms
    assert "geom_line" in layer_geoms
    assert "geom_ribbon" in layer_geoms


def test_plot_thresh_positive_and_negative_branches():
    # Positive effect
    df_pos = _make_df(seed=7)
    res_pos = exc_optim_thresh(
        df=df_pos,
        outcome_col="y",
        prob_col="p_hat",
        threshold=0.5,
    )
    g_pos = plot_thresh(res_pos, "y")
    assert isinstance(g_pos, ggplot)

    # Negative effect (flip outcome sign)
    df_neg = df_pos.copy()
    df_neg["y"] = -df_neg["y"]

    res_neg = exc_optim_thresh(
        df=df_neg,
        outcome_col="y",
        prob_col="p_hat",
        threshold=0.5,
    )
    g_neg = plot_thresh(res_neg, "y")
    assert isinstance(g_neg, ggplot)


def test_plot_thresh_labels_present():
    df = _make_df(seed=8)

    results = exc_optim_thresh(
        df=df,
        outcome_col="y",
        prob_col="p_hat",
        threshold=0.5,
    )

    g = plot_thresh(results, "y")

    labels = g.labels

    assert hasattr(labels, "title")
    assert hasattr(labels, "x")
    assert hasattr(labels, "y")

    assert "Probability" in labels.x

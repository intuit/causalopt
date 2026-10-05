import numpy as np
import pandas as pd
import pytest

from causalopt import causalopt, optimum_threshold, tradeoff_threshold
from causalopt.utils import bin_data

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


# ============================================================
# out_of_bandwidth (Point 2)
# ============================================================


def test_out_of_bandwidth_widens_1d():
    df = _make_df(seed=5, n=2000)

    narrow = tradeoff_threshold(
        df, ["y1"], "p_hat", threshold=0.5, out_of_bandwidth=False
    )
    wide = tradeoff_threshold(
        df, ["y1"], "p_hat", threshold=0.5, out_of_bandwidth=True
    )

    # The eval grid widens to the full support while the fit is unchanged.
    assert wide["x"].max() >= narrow["x"].max()
    assert wide["x"].min() <= narrow["x"].min()
    assert (wide["x"].max() - wide["x"].min()) > (
        narrow["x"].max() - narrow["x"].min()
    )


# ============================================================
# Binning (Point 3)
# ============================================================


def test_binned_fit_recovers_linear_jump_and_matches_raw():
    rng = np.random.default_rng(3)
    n = 6000
    x = rng.uniform(0.0, 1.0, n)
    jump = 2.0
    y = 1.0 + 0.5 * x + jump * (x >= 0.5) + rng.normal(0, 0.05, n)
    df = pd.DataFrame({"score": x, "Y": y})

    raw = causalopt(df, ["Y"], "score", mode="binary", threshold=0.5)
    binned = causalopt(
        df, ["Y"], "score", mode="binary", threshold=0.5, bin=True, bin_spec=40
    )

    est_raw = raw["details"]["estimates"][0]
    est_bin = binned["details"]["estimates"][0]

    # binned WLS recovers the true jump and tracks the raw fit
    assert abs(est_bin - jump) < 0.3
    assert abs(est_bin - est_raw) < 0.3
    # bandwidth selection is skipped on binned data
    assert binned["details"]["rd_results"] is None


def test_binned_data_ingest_matches_internal_binning():
    rng = np.random.default_rng(4)
    n = 4000
    x = rng.uniform(0.0, 1.0, n)
    y = 1.0 + 2.0 * (x >= 0.5) + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"score": x, "Y": y})

    pre = bin_data(df, ["score"], ["Y"], cutoff=0.5, bin_spec=40)
    out = optimum_threshold(
        df, "Y", "score", threshold=0.5, binned_data=False, bin=True, bin_spec=40
    )
    out_pre = optimum_threshold(
        pre, "Y", "score", threshold=0.5, binned_data=True, weight_col="n"
    )

    # both binned routes produce a usable recommendation, rd_results skipped
    assert isinstance(out["optimum_thresholds"]["Recommendation"], str)
    assert out["rd_results"] is None
    assert out_pre["rd_results"] is None


def test_optimum_threshold_binning_guardrails():
    df = _make_df(seed=6)
    with pytest.raises(ValueError):
        optimum_threshold(df, "y1", "p_hat", threshold=0.5, bin=True, binned_data=True)
    with pytest.raises(ValueError):
        optimum_threshold(
            df, "y1", "p_hat", threshold=0.5, covariates=["y2"], bin=True
        )


# ============================================================
# Unified causalopt entry point (Point 4)
# ============================================================


def test_causalopt_binary_shared_grid_and_both_analyses():
    rng = np.random.default_rng(7)
    n = 2000
    s = rng.uniform(0.0, 1.0, n)
    df = pd.DataFrame(
        {
            "score": s,
            "Y1": 1 + 2 * (s > 0.5) + rng.normal(0, 0.1, n),
            "Y2": 3 - 1 * (s > 0.5) + rng.normal(0, 0.1, n),
        }
    )

    r = causalopt(df, ["Y1", "Y2"], "score", mode="binary", threshold=0.5)

    assert r["mode"] == "binary"
    assert {"current", "frontier", "optimum", "details"}.issubset(r.keys())
    # tradeoff frontier always present, both outcomes on the SAME grid
    assert {"x", "n", "p", "gain_Y1", "gain_Y2"}.issubset(r["frontier"].columns)
    assert r["frontier"][["x", "n", "p"]].duplicated().sum() == 0
    # optimum is the optim_thresh result on the primary outcome
    assert "Recommendation" in r["optimum"]
    assert r["current"]["threshold"] == 0.5

    # secondary RD fits are kept, not thrown away
    d = r["details"]
    assert set(d["by_outcome"]) == {"Y1", "Y2"}
    assert d["by_outcome"]["Y1"]["estimates"] is d["estimates"]
    assert d["by_outcome"]["Y1"]["rd_results"] is d["rd_results"]
    assert d["by_outcome"]["Y2"]["rd_results"] is not None

    ate = d["ate"]
    assert list(ate.columns) == [
        "outcome", "coef", "se", "ci_lower", "ci_upper",
        "h", "b", "n_left", "n_right",
    ]
    assert list(ate["outcome"]) == ["Y1", "Y2"]
    ate = ate.set_index("outcome")
    assert abs(ate.loc["Y1", "coef"] - 2.0) < 0.3
    assert abs(ate.loc["Y2", "coef"] + 1.0) < 0.3
    assert (ate["ci_lower"] <= ate["coef"]).all()
    assert (ate["coef"] <= ate["ci_upper"]).all()
    assert (ate["se"] > 0).all()
    assert (ate["h"] > 0).all()
    assert (ate["h"] <= ate["b"]).all()
    assert ((ate["n_left"] + ate["n_right"]) <= n).all()
    # the table agrees with the per-outcome estimates list
    assert np.isclose(ate.loc["Y2", "coef"], d["by_outcome"]["Y2"]["estimates"][0])


def test_causalopt_binary_ate_table_binned():
    rng = np.random.default_rng(11)
    n = 4000
    s = rng.uniform(0.0, 1.0, n)
    df = pd.DataFrame(
        {
            "score": s,
            "Y1": 1 + 2 * (s > 0.5) + rng.normal(0, 0.1, n),
            "Y2": 3 - 1 * (s > 0.5) + rng.normal(0, 0.1, n),
        }
    )

    r = causalopt(
        df, ["Y1", "Y2"], "score", mode="binary", threshold=0.5,
        bin=True, bin_spec=40,
    )
    d = r["details"]

    for o in ("Y1", "Y2"):
        assert d["by_outcome"][o]["rd_results"] is None

    ate = d["ate"].set_index("outcome")
    assert abs(ate.loc["Y1", "coef"] - 2.0) < 0.3
    assert abs(ate.loc["Y2", "coef"] + 1.0) < 0.3
    assert ate["h"].isna().all()
    assert ate["b"].isna().all()
    assert (ate["se"] > 0).all()
    # binned WLS uses the full support: all weight is accounted for
    assert np.allclose(ate["n_left"] + ate["n_right"], n)


def test_causalopt_multiclass_passthrough():
    rng = np.random.default_rng(8)
    K = 3
    P = rng.dirichlet(np.ones(K), size=1500)
    df = pd.DataFrame(P, columns=[f"p{k}" for k in range(K)])
    df["Y1"] = P @ np.array([1.0, 2.0, 3.0]) + rng.normal(0, 0.05, 1500)

    r = causalopt(df, ["Y1"], [f"p{k}" for k in range(K)], mode="multiclass", B=150)

    assert r["mode"] == "multiclass"
    assert {"current", "frontier", "optimum", "details"}.issubset(r.keys())
    assert {"tau_1", "tau_2", "tau_3"}.issubset(r["frontier"].columns)
    assert r["details"]["B"] == 150
    assert r["details"]["probabilities"] is True


def test_causalopt_dispatch_guards():
    df = _make_df(seed=9)

    with pytest.raises(ValueError):
        causalopt(df, ["y1"], "p_hat", mode="bogus", threshold=0.5)
    with pytest.raises(ValueError):
        causalopt(df, ["y1"], "p_hat", mode="binary")  # missing threshold
    with pytest.raises(ValueError):
        causalopt(df, ["y1"], "p_hat", mode="binary", threshold=0.5, tau=[0.5, 0.5])
    with pytest.raises(ValueError):
        causalopt(df, ["y1"], ["p_hat", "y2"], mode="binary", threshold=0.5)  # 2 scores
    with pytest.raises(ValueError):
        causalopt(df, ["y1"], ["p_hat"], mode="multiclass")  # <2 score cols
    with pytest.raises(ValueError):
        causalopt(
            df, ["y1"], ["p_hat", "y2"], mode="multiclass", threshold=0.5
        )  # threshold in MC

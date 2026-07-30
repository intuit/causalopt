import matplotlib

matplotlib.use("Agg")  # headless backend for plot_simplex

import numpy as np
import pandas as pd
import pytest

from causalopt.multiclass import (
    generate_probabilities,
    generate_outcome,
    generate_R,
    generate_simulated_data,
    rename_to_prob_k,
    dist_to_bound,
    pairwise_selection,
    decision_rule,
    validate_d,
    d_to_tau,
    build_d_grid,
    frontier_grid_to_tau_K,
    union_bw_mask,
    triangular_kernel_multi,
    poly_terms,
    rd_fit_multi,
    sim_poly_multi,
    assign_predicted_outcomes,
    bdselect,
    est_multi,
    pred_multi,
    find_thresh,
    select_optimum,
    get_thresholds,
    plot_simplex,
)
from causalopt.utils import bin_data


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def _make_prob_df(K=3, n=800, seed=0):
    """Probability DataFrame with Prob_1..K, T, and two outcomes Y1, Y2."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(K), size=n)
    df = pd.DataFrame(P, columns=[f"Prob_{k}" for k in range(1, K + 1)])
    tau = np.full(K, 1.0 / K)
    df["T"] = decision_rule(P, tau)
    df["Y1"] = P @ np.linspace(1.0, 2.0, K) + rng.normal(0, 0.05, n)
    df["Y2"] = P @ np.linspace(2.0, 1.0, K) + rng.normal(0, 0.05, n)
    return df, tau


def _make_prepared_df(K=3, n=800, seed=0):
    """As _make_prob_df but with the dist_j_k columns added."""
    df, tau = _make_prob_df(K=K, n=n, seed=seed)
    df = dist_to_bound(df, tau)
    return df, tau


def _make_score_df(K=3, n=800, seed=0, signed=False):
    """Raw-score DataFrame (non-simplex) with Prob_1..K, T, outcomes Y1, Y2.

    signed=False -> case 1 (positive, unbounded); signed=True -> case 2
    (signed net values). T is the plain argmax(value) baseline (tau=0).
    """
    rng = np.random.default_rng(seed)
    if signed:
        V = rng.normal(0.0, 3.0, size=(n, K))
    else:
        V = rng.gamma(shape=2.0, scale=2.0, size=(n, K))
    df = pd.DataFrame(V, columns=[f"Prob_{k}" for k in range(1, K + 1)])
    tau = np.zeros(K)
    df["T"] = decision_rule(V, tau)
    df["Y1"] = V @ np.linspace(1.0, 2.0, K) + rng.normal(0, 0.05, n)
    df["Y2"] = V @ np.linspace(2.0, 1.0, K) + rng.normal(0, 0.05, n)
    return df, tau


# ============================================================
# Data generation
# ============================================================


def test_generate_probabilities_shape_and_simplex():
    P = generate_probabilities(n=500, K=4, seed=1)
    assert P.shape == (500, 4)
    assert np.all(P >= 0)
    assert np.allclose(P.sum(axis=1), 1.0)
    # reproducible
    P2 = generate_probabilities(n=500, K=4, seed=1)
    assert np.array_equal(P, P2)


def test_generate_outcome_binary_and_reproducible():
    P = generate_probabilities(n=300, K=3, seed=2)
    T = decision_rule(P, np.full(3, 1 / 3))
    alpha_T = np.array([0.0, -0.2, 0.1, 0.3])  # index 0 unused (T is 1-based)
    beta = np.array([0.5, 0.5, 0.5])
    Y = generate_outcome(P, T, alpha_T, beta, sigma=1.0, seed=3)
    assert Y.shape == (300,)
    assert set(np.unique(Y)).issubset({0, 1})
    Y2 = generate_outcome(P, T, alpha_T, beta, sigma=1.0, seed=3)
    assert np.array_equal(Y, Y2)


def test_generate_R_zero_when_Y_zero():
    rng = np.random.default_rng(4)
    Y = rng.integers(0, 2, size=200)
    T = rng.integers(1, 4, size=200)
    R = generate_R(Y, T, mu_R=1.0, delta_R=0.1, sigma_R=0.5, seed=5)
    assert R.shape == (200,)
    assert np.all(R >= 0)
    assert np.all(R[Y == 0] == 0)
    assert np.all(R[Y == 1] > 0)
    R2 = generate_R(Y, T, mu_R=1.0, delta_R=0.1, sigma_R=0.5, seed=5)
    assert np.array_equal(R, R2)


def test_generate_simulated_data_columns_and_R_flag():
    alpha_T = np.array([0.0, 0.1, 0.2, 0.3])
    beta = np.array([0.5, 0.5, 0.5])
    data = generate_simulated_data(
        n=400, K=3, tau=np.full(3, 1 / 3),
        alpha_dirichlet=None, alpha_T=alpha_T, beta=beta,
        add_R=True, mu_R=1.0, delta_R=0.0, seed=6,
    )
    for col in ["Prob_1", "Prob_2", "Prob_3", "T", "Y", "R"]:
        assert col in data.columns
    assert set(np.unique(data["T"])).issubset({1, 2, 3})

    # add_R without mu_R must raise
    with pytest.raises(ValueError):
        generate_simulated_data(
            n=10, K=3, tau=np.full(3, 1 / 3),
            alpha_dirichlet=None, alpha_T=alpha_T, beta=beta,
            add_R=True, mu_R=None, seed=6,
        )


# ============================================================
# Data preparation
# ============================================================


def test_rename_to_prob_k():
    df = pd.DataFrame({"a": [0.2, 0.5], "b": [0.3, 0.1], "c": [0.5, 0.4]})
    out, mapping = rename_to_prob_k(df, ["a", "b", "c"])
    assert mapping == {"a": "Prob_1", "b": "Prob_2", "c": "Prob_3"}
    assert list(out.columns) == ["Prob_1", "Prob_2", "Prob_3"]


def test_dist_to_bound_values():
    df = pd.DataFrame({"Prob_1": [0.5], "Prob_2": [0.3], "Prob_3": [0.2]})
    tau = np.array([0.2, 0.3, 0.5])
    out = dist_to_bound(df, tau)
    for c in ["dist_1_2", "dist_1_3", "dist_2_3"]:
        assert c in out.columns
    # dist_1_2 = (P1 - tau1) - (P2 - tau2)
    exp_12 = (0.5 - 0.2) - (0.3 - 0.3)
    assert np.isclose(out["dist_1_2"].iloc[0], exp_12)


def test_pairwise_selection_filters_and_renames():
    df, tau = _make_prepared_df(seed=7)
    out = pairwise_selection(df, (1, 2), d=None, runing_var="x")
    assert set(np.unique(out["T"])).issubset({1, 2})
    assert "x" in out.columns
    assert "dist_1_2" not in out.columns  # renamed to x

    with pytest.raises(ValueError):
        pairwise_selection(df.drop(columns=["dist_1_2"]), (1, 2))


def test_decision_rule_argmax():
    Prob = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    tau = np.zeros(3)
    T = decision_rule(Prob, tau)
    assert list(T) == [1, 3]


# ============================================================
# Grid / tau conversion
# ============================================================


def test_validate_d_transitivity():
    good = {(1, 2): 0.1, (1, 3): 0.3, (2, 3): 0.2}  # 0.1 + 0.2 == 0.3
    assert validate_d(good, tol=1e-9)
    bad = {(1, 2): 0.1, (1, 3): 0.3, (2, 3): 0.5}
    assert not validate_d(bad, tol=1e-9)


def test_d_to_tau_roundtrip_and_simplex():
    tau = d_to_tau({(1, 2): 0.0, (1, 3): 0.0, (2, 3): 0.0}, K=3)
    assert np.allclose(tau, (1 / 3, 1 / 3, 1 / 3))
    assert np.isclose(sum(tau), 1.0)
    # transitivity violation -> ValueError
    with pytest.raises(ValueError):
        d_to_tau({(1, 2): 0.1, (1, 3): 0.3, (2, 3): 0.5}, K=3)


def test_d_to_tau_scores_offset_convention():
    # probabilities=False: tau_1 == 0 and tau_j == -d_{1,j}; no box check.
    d = {(1, 2): 0.4, (1, 3): -0.9, (2, 3): -1.3}
    tau = d_to_tau(d, K=3, probabilities=False)
    assert np.isclose(tau[0], 0.0)
    assert np.isclose(tau[1], -0.4)  # -d_1_2
    assert np.isclose(tau[2], 0.9)   # -d_1_3
    # differences are preserved regardless of level convention
    assert np.isclose(tau[0] - tau[1], d[(1, 2)])
    assert np.isclose(tau[0] - tau[2], d[(1, 3)])

    # Large differences that would fall outside the simplex must NOT raise
    # when probabilities=False (the box is skipped).
    big = {(1, 2): 5.0, (1, 3): -7.0, (2, 3): -12.0}
    tau_big = d_to_tau(big, K=3, probabilities=False)
    assert np.isclose(tau_big[0], 0.0)
    # ...but the same differences violate the simplex box when probabilities=True
    with pytest.raises(ValueError):
        d_to_tau(big, K=3, probabilities=True)


def test_build_d_grid_valid_and_transitive():
    bounds = {(1, 2): 0.1, (1, 3): 0.1, (2, 3): 0.0}
    grid = build_d_grid(bounds, B=4, grid="full")
    assert len(grid) == 4 ** 2  # K-1 = 2 independent differences
    pairs = [(1, 2), (1, 3), (2, 3)]
    for tup in grid:
        d = dict(zip(pairs, tup))
        assert validate_d(d, tol=1e-9)


def test_frontier_grid_to_tau_K_simplex():
    bounds = {(1, 2): 0.15, (1, 3): 0.15, (2, 3): 0.0}
    taus = frontier_grid_to_tau_K(bounds, B=8, grid="full")
    assert len(taus) > 0
    for tau in taus:
        assert len(tau) == 3
        assert np.isclose(sum(tau), 1.0)


# ============================================================
# Kernel weighting
# ============================================================


def test_union_bw_mask_manual():
    df = pd.DataFrame({
        "T": [1, 2, 3],
        "dist_1_2": [0.05, 5.0, 5.0],
        "dist_1_3": [5.0, 5.0, 0.05],
        "dist_2_3": [5.0, 5.0, 5.0],
    })
    thresh = {(1, 2): 0.1, (1, 3): 0.1, (2, 3): 0.1}
    mask = union_bw_mask(df, thresh)
    # row0: T=1 near (1,2); row2: T=3 near (1,3); row1: far everywhere
    assert list(mask) == [True, False, True]


def test_triangular_kernel_multi_all_kernels():
    df = pd.DataFrame({
        "dist_1_2": [0.00, 0.00, 5.00],
        "dist_1_3": [4.00, 0.00, 5.00],
        "dist_2_3": [4.00, 4.00, 5.00],
    })
    h = {(1, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0}
    # kt = [[1,0,0],[1,1,0],[0,0,0]]
    w_sum, info = triangular_kernel_multi(df, h=h, kernel="sum")
    assert info["kernel"] == "sum"
    assert np.allclose(w_sum, [1.0, 2.0, 0.0])
    assert w_sum[1] > w_sum[0]  # near two borders beats near one

    assert np.allclose(
        triangular_kernel_multi(df, h=h, kernel="soft_or")[0], [1.0, 1.0, 0.0]
    )
    assert np.allclose(
        triangular_kernel_multi(df, h=h, kernel="nearest")[0], [1.0, 1.0, 0.0]
    )
    assert np.allclose(
        triangular_kernel_multi(df, h=h, kernel="prod")[0], [0.0, 0.0, 0.0]
    )
    w_far = triangular_kernel_multi(df, h=h, kernel="farthest")[0]
    assert np.allclose(w_far, [0.0, 0.0, 0.0])
    w_alias, ia = triangular_kernel_multi(df, h=h, kernel="min_dist")
    assert np.allclose(w_alias, w_far)
    assert ia["kernel"] == "min_dist"

    with pytest.raises(ValueError):
        triangular_kernel_multi(df, h=h, kernel="nope")


def test_triangular_kernel_multi_formula_nonzero():
    df = pd.DataFrame({"dist_1_2": [0.2], "dist_1_3": [0.5], "dist_2_3": [0.4]})
    h = {(1, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0}
    kt = np.array([0.8, 0.5, 0.6])
    for name, expected in [
        ("sum", kt.sum()),
        ("soft_or", 1 - np.prod(1 - kt)),
        ("nearest", kt.max()),
        ("prod", kt.prod()),
        ("farthest", kt.min()),
    ]:
        w = triangular_kernel_multi(df, h=h, kernel=name)[0]
        assert np.allclose(w, [expected]), name


# ============================================================
# Local polynomial regression
# ============================================================


def test_poly_terms_intercept_and_rank():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(200, 2))
    R, degs = poly_terms(X, p=1)
    assert R.shape == (200, 3)  # intercept + 2 coords
    assert np.allclose(R[:, 0], 1.0)  # intercept column
    assert degs[0] == 0
    assert np.linalg.matrix_rank(R) == 3


def test_rd_fit_multi_structure_and_recovery():
    rng = np.random.default_rng(9)
    n = 600
    X = rng.uniform(-0.5, 0.5, size=(n, 2))
    y = 1.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(0, 0.02, n)
    W = np.ones(n)
    mm = rd_fit_multi(y=y, X=X, W=W, p=1)
    assert set(mm.keys()) == {"coef_poly", "V_poly"}
    for k in ("conventional", "bias_corrected"):
        assert k in mm["coef_poly"]
        assert k in mm["V_poly"]
    b = mm["coef_poly"]["conventional"].ravel()
    assert b.shape[0] == 3
    assert np.allclose(b, [1.0, 0.5, -0.3], atol=0.05)
    V = mm["V_poly"]["conventional"]
    assert V.shape == (3, 3)


def test_sim_poly_multi_bands_and_reproducible():
    rng = np.random.default_rng(10)
    n = 500
    X = rng.uniform(-0.5, 0.5, size=(n, 2))
    y = 1.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(0, 0.02, n)
    mm = rd_fit_multi(y=y, X=X, W=np.ones(n), p=1)
    b = mm["coef_poly"]["bias_corrected"]
    V = mm["V_poly"]["bias_corrected"]

    Xnew = rng.uniform(-0.3, 0.3, size=(20, 2))
    y_hat, mean_fit, lower, upper = sim_poly_multi(
        Xnew, b, V, p=1, nsim=2000, seed=0
    )
    assert y_hat.shape == (20,)
    assert np.all(lower <= mean_fit) and np.all(mean_fit <= upper)

    out2 = sim_poly_multi(Xnew, b, V, p=1, nsim=2000, seed=0)
    assert np.allclose(y_hat, out2[0])
    assert np.allclose(lower, out2[2])


# ============================================================
# Prediction / assignment
# ============================================================


def test_assign_predicted_outcomes_gather():
    df = pd.DataFrame({
        "T": [1, 2, 3],
        "T_alt": [2, 3, 1],
        "Y1_hat_1": [10.0, 11.0, 12.0],
        "Y1_hat_2": [20.0, 21.0, 22.0],
        "Y1_hat_3": [30.0, 31.0, 32.0],
    })
    out = assign_predicted_outcomes(df, ["Y1"])
    assert list(out["Y1_hat_current"]) == [10.0, 21.0, 32.0]
    assert list(out["Y1_hat_alt"]) == [20.0, 31.0, 12.0]

    bad = df.copy()
    bad["T"] = [1, 2, 9]
    with pytest.raises(ValueError):
        assign_predicted_outcomes(bad, ["Y1"])


def test_bdselect_structure():
    df, tau = _make_prepared_df(seed=11)
    tt = bdselect(df, ["Y1"], tau)
    assert set(tt.keys()) == {"Y1"}
    assert set(tt["Y1"].keys()) == {(1, 2), (1, 3), (2, 3)}
    for h in tt["Y1"].values():
        assert h > 0


def test_est_multi_and_pred_multi():
    df, tau = _make_prepared_df(seed=12)
    tt = bdselect(df, ["Y1"], tau)
    est = est_multi(df, ["Y1"], tt, K=3, kernel="sum")
    assert set(est["Y1"].keys()) == {1, 2, 3}
    for j in (1, 2, 3):
        assert "coef_poly" in est["Y1"][j]

    pred = pred_multi(df, ["Y1"], est, K=3)
    for j in (1, 2, 3):
        assert f"Y1_hat_{j}" in pred.columns
    assert len(pred) == len(df)


def test_find_thresh_structure():
    rng = np.random.default_rng(13)
    n = 60
    P = rng.dirichlet(np.ones(3), size=n)
    df = pd.DataFrame(P, columns=["Prob_1", "Prob_2", "Prob_3"])
    df["T"] = decision_rule(P, np.full(3, 1 / 3))
    for j in (1, 2, 3):
        df[f"Y1_hat_{j}"] = rng.normal(size=n)
    pair_thresh = {"Y1": {(1, 2): 0.1, (1, 3): 0.1, (2, 3): 0.1}}
    B = 100
    res = find_thresh(df, ["Y1"], pair_thresh, B=B)
    for c in ["tau_1", "tau_2", "tau_3", "mean_Y1", "total_Y1"]:
        assert c in res.columns
    assert len(res) > 0
    grids = max(2, int(B ** (1.0 / 2)))
    assert len(res) <= grids ** 2  # K-1 = 2 independent axes


# ============================================================
# End-to-end
# ============================================================


@pytest.mark.parametrize(
    "kernel", ["sum", "soft_or", "nearest", "prod", "farthest"]
)
def test_get_thresholds_kernels_K3(kernel):
    df, tau = _make_prob_df(K=3, n=800, seed=14)
    out = get_thresholds(
        data=df,
        outcome_cols=["Y1"],
        probability_cols=["Prob_1", "Prob_2", "Prob_3"],
        tau=tau,
        B=200,
        kernel=kernel,
    )
    frontier = out["frontier"]
    assert isinstance(frontier, pd.DataFrame)
    assert len(frontier) > 0
    assert {"tau_1", "tau_2", "tau_3"}.issubset(frontier.columns)
    assert "optimum" in out and "current" in out


def test_get_thresholds_K4_smoke():
    df, tau = _make_prob_df(K=4, n=800, seed=15)
    out = get_thresholds(
        data=df,
        outcome_cols=["Y1"],
        probability_cols=["Prob_1", "Prob_2", "Prob_3", "Prob_4"],
        tau=tau,
        B=150,
    )
    frontier = out["frontier"]
    assert isinstance(frontier, pd.DataFrame)
    assert len(frontier) > 0
    assert {"tau_1", "tau_2", "tau_3", "tau_4"}.issubset(frontier.columns)


# ============================================================
# Unbounded scores (probabilities=False, cases 1 and 2)
# ============================================================


@pytest.mark.parametrize("signed", [False, True])
def test_get_thresholds_scores_K3(signed):
    # case 1 (positive) and case 2 (signed) end-to-end with probabilities=False
    df, tau = _make_score_df(K=3, n=800, seed=20, signed=signed)
    out = get_thresholds(
        data=df,
        outcome_cols=["Y1"],
        probability_cols=["Prob_1", "Prob_2", "Prob_3"],
        tau=tau,
        B=200,
        probabilities=False,
    )
    frontier = out["frontier"]
    assert isinstance(frontier, pd.DataFrame)
    assert len(frontier) > 0
    assert {"tau_1", "tau_2", "tau_3"}.issubset(frontier.columns)
    # offset convention: tau_1 anchored at 0
    assert np.allclose(frontier["tau_1"].to_numpy(), 0.0)


def test_get_thresholds_scores_K4_smoke():
    df, tau = _make_score_df(K=4, n=800, seed=21, signed=True)
    out = get_thresholds(
        data=df,
        outcome_cols=["Y1"],
        probability_cols=["Prob_1", "Prob_2", "Prob_3", "Prob_4"],
        tau=tau,
        B=150,
        probabilities=False,
    )
    frontier = out["frontier"]
    assert isinstance(frontier, pd.DataFrame)
    assert len(frontier) > 0
    assert {"tau_1", "tau_2", "tau_3", "tau_4"}.issubset(frontier.columns)
    assert np.allclose(frontier["tau_1"].to_numpy(), 0.0)


def test_get_thresholds_tau_none_defaults():
    # probabilities=True defaults tau to uniform 1/K
    dfp, _ = _make_prob_df(K=3, n=600, seed=22)
    out_p = get_thresholds(
        data=dfp,
        outcome_cols=["Y1"],
        probability_cols=["Prob_1", "Prob_2", "Prob_3"],
        tau=None,
        B=150,
    )
    assert len(out_p["frontier"]) > 0

    # probabilities=False defaults tau to zeros
    dfs, _ = _make_score_df(K=3, n=600, seed=23)
    out_s = get_thresholds(
        data=dfs,
        outcome_cols=["Y1"],
        probability_cols=["Prob_1", "Prob_2", "Prob_3"],
        tau=None,
        B=150,
        probabilities=False,
    )
    assert len(out_s["frontier"]) > 0
    assert np.allclose(out_s["frontier"]["tau_1"].to_numpy(), 0.0)


def test_get_thresholds_fail_fast_on_non_simplex():
    # Non-simplex data with probabilities=True must raise...
    df, _ = _make_score_df(K=3, n=200, seed=24)
    with pytest.raises(ValueError):
        get_thresholds(
            data=df,
            outcome_cols=["Y1"],
            probability_cols=["Prob_1", "Prob_2", "Prob_3"],
            tau=None,
            B=100,
            probabilities=True,
        )
    # ...but the same data runs fine as scores.
    out = get_thresholds(
        data=df,
        outcome_cols=["Y1"],
        probability_cols=["Prob_1", "Prob_2", "Prob_3"],
        tau=None,
        B=100,
        probabilities=False,
    )
    assert len(out["frontier"]) > 0


def test_select_optimum_argmax():
    # Hand-built frontier: per-outcome argmax over mean_, tie-break on total_.
    frontier = pd.DataFrame({
        "tau_1": [0.0, 0.0, 0.0],
        "tau_2": [0.1, 0.2, 0.3],
        "tau_3": [0.9, 0.8, 0.7],
        "mean_Y1": [1.0, 3.0, 2.0],
        "total_Y1": [10.0, 30.0, 20.0],
        "mean_Y2": [5.0, 1.0, 4.0],
        "total_Y2": [50.0, 10.0, 40.0],
    })

    opt = select_optimum(frontier, ["Y1", "Y2"], by="mean")

    # Y1 maximised at row 1 (mean 3.0)
    assert opt["Y1"]["tau"] == (0.0, 0.2, 0.8)
    assert opt["Y1"]["mean"] == 3.0
    assert opt["Y1"]["total"] == 30.0
    assert opt["Y1"]["outcomes_at_tau"]["Y2"]["mean"] == 1.0

    # Y2 maximised at row 0 (mean 5.0)
    assert opt["Y2"]["tau"] == (0.0, 0.1, 0.9)
    assert opt["Y2"]["mean"] == 5.0

    # by="total" selects the same argmax row here
    opt_t = select_optimum(frontier, ["Y1"], by="total")
    assert opt_t["Y1"]["total"] == 30.0

    with pytest.raises(ValueError):
        select_optimum(frontier, ["Y1"], by="bad")


# ============================================================
# out_of_bandwidth (Point 2)
# ============================================================


def test_get_thresholds_out_of_bandwidth_widens():
    df, tau = _make_prob_df(K=3, n=800, seed=30)
    cols = ["Prob_1", "Prob_2", "Prob_3"]
    base = get_thresholds(
        df, ["Y1"], cols, tau=tau, B=200, out_of_bandwidth=False
    )["frontier"]
    wide = get_thresholds(
        df, ["Y1"], cols, tau=tau, B=200, out_of_bandwidth=True
    )["frontier"]

    def span(fr, c):
        return fr[c].max() - fr[c].min()

    # widening the candidate range grows the observed tau span
    assert span(wide, "tau_2") > span(base, "tau_2")


# ============================================================
# Binning (Point 3)
# ============================================================


def test_get_thresholds_binned_end_to_end():
    df, tau = _make_prob_df(K=3, n=2000, seed=31)
    cols = ["Prob_1", "Prob_2", "Prob_3"]
    res = get_thresholds(df, ["Y1"], cols, tau=tau, B=200, bin=True, bin_spec=5)

    fr = res["frontier"]
    assert len(fr) > 0
    assert {"tau_1", "tau_2", "tau_3"}.issubset(fr.columns)
    assert "Y1" in res["optimum"]


def test_get_thresholds_binned_data_ingest():
    df, tau = _make_prob_df(K=3, n=2000, seed=32)
    cols = ["Prob_1", "Prob_2", "Prob_3"]
    T = decision_rule(df[cols].values, tau)

    # pre-binned centroids stay on the simplex (mean of simplex points)
    pre = bin_data(df, cols, ["Y1"], group=T, bin_spec=5)
    res = get_thresholds(
        pre, ["Y1"], cols, tau=tau, B=200, binned_data=True, weight_col="n"
    )
    assert len(res["frontier"]) > 0


def test_get_thresholds_binning_guardrail():
    df, tau = _make_prob_df(K=3, n=400, seed=33)
    cols = ["Prob_1", "Prob_2", "Prob_3"]
    with pytest.raises(ValueError):
        get_thresholds(df, ["Y1"], cols, tau=tau, B=100, bin=True, binned_data=True)


# ============================================================
# Visualization
# ============================================================


def test_plot_simplex_runs_and_guards():
    df, tau = _make_prob_df(K=3, n=100, seed=16)
    # K=3 executes without error under Agg
    assert plot_simplex(df, tau, step=0.05) is None

    # non-3 class data raises
    df4, tau4 = _make_prob_df(K=4, n=20, seed=17)
    with pytest.raises(NotImplementedError):
        plot_simplex(df4, tau4, step=0.1)

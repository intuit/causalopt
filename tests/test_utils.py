import numpy as np
import pandas as pd
import pytest

from causalopt.utils import (
    _bw_mse,
    _prepare_inputs,
    _split_lr,
    bin_data,
    complete_cases,
    covs_drop_fun,
    crossprod,
    inv_chol,
    make_dups,
    nanmat,
    ncol,
    nn_residuals,
    poly_eval,
    qrXXinv,
    resolve_binning,
    sandwich_se,
    sim_poly_ic,
    tomat,
    triangular_kernel,
)

# ============================================================
# poly_eval
# ============================================================


def test_poly_eval_matches_manual():
    x = np.array([-2.0, 0.0, 3.0])
    b = np.array([1.5, -2.0, 0.25])
    got = poly_eval(x, b)
    exp = b[0] + b[1] * x + b[2] * x**2
    np.testing.assert_allclose(got, exp)


def test_poly_eval_constant():
    x = np.linspace(-1, 1, 5)
    b = [3.0]
    np.testing.assert_allclose(poly_eval(x, b), 3.0)


# ============================================================
# sim_poly_ic
# ============================================================


def test_sim_poly_ic_shapes_and_reproducibility():
    x = np.array([-1.0, 0.0, 1.0])
    b = np.array([1.0, 2.0])
    V = np.eye(2) * 0.1

    fit1, lo1, hi1 = sim_poly_ic(x, b, V, nsim=4000, seed=123)
    fit2, lo2, hi2 = sim_poly_ic(x, b, V, nsim=4000, seed=123)

    assert fit1.shape == x.shape
    np.testing.assert_allclose(fit1, fit2)
    np.testing.assert_allclose(lo1, lo2)
    np.testing.assert_allclose(hi1, hi2)
    assert np.all(lo1 <= hi1)


def test_sim_poly_ic_mean_matches_poly_eval():
    x = np.linspace(-2, 2, 9)
    b = np.array([0.5, -1.0, 0.25])
    V = np.diag([0.02, 0.02, 0.02])

    fit, _, _ = sim_poly_ic(x, b, V, nsim=20000, seed=0)
    np.testing.assert_allclose(fit, poly_eval(x, b), atol=0.02)


# ============================================================
# Matrix utilities
# ============================================================


def test_tomat():
    x = np.array([1, 2, 3])
    M = tomat(x)
    assert M.shape == (3, 1)


def test_ncol():
    assert ncol(np.array([1, 2, 3])) == 1
    assert ncol(np.ones((4, 2))) == 2


def test_crossprod():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Y = np.array([[5.0], [6.0]])
    np.testing.assert_allclose(crossprod(X), X.T @ X)
    np.testing.assert_allclose(crossprod(X, Y), X.T @ Y)


def test_nanmat():
    M = nanmat(3, 2)
    assert np.all(np.isnan(M))


def test_inv_chol_equals_inverse():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    np.testing.assert_allclose(inv_chol(A), np.linalg.inv(A))


def test_qrXXinv():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    np.testing.assert_allclose(qrXXinv(X), np.linalg.inv(X.T @ X), atol=1e-10)


def test_complete_cases():
    X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])
    np.testing.assert_array_equal(complete_cases(X), [False, True, False])


def test_covs_drop_fun_rank():
    z = np.column_stack([np.arange(5), np.arange(5), np.ones(5)])
    kept = covs_drop_fun(z)
    assert np.linalg.matrix_rank(kept) == kept.shape[1]


# ============================================================
# Kernel + residual utilities
# ============================================================


def test_triangular_kernel():
    X = np.array([-1.5, -1.0, 0.0, 1.0, 1.5])
    w = triangular_kernel(X, c=0.0, h=1.0)

    assert w[2] == pytest.approx(1.0)
    assert w[0] == 0.0
    assert w[-1] == 0.0


def test_nn_residuals_all_neighbors():
    X = np.array([-1.0, 0.0, 1.0])
    y = np.array([1.0, 2.0, 3.0])

    dups, dupsid = make_dups(X)
    res = nn_residuals(X, y, None, matches=2, dups=dups, dupsid=dupsid)

    Ji = 2
    scale = np.sqrt(Ji / (Ji + 1))
    for i in range(3):
        mean_others = (y.sum() - y[i]) / Ji
        assert res[i, 0] == pytest.approx(scale * (y[i] - mean_others))


def test_sandwich_se():
    rng = np.random.default_rng(1)
    RX = rng.normal(size=(10, 2))
    res = rng.normal(size=(10, 1))

    np.testing.assert_allclose(sandwich_se(RX, res), (RX * res).T @ (RX * res))


# ============================================================
# _bw_mse (smoke test)
# ============================================================


def test_bw_mse_smoke():
    rng = np.random.default_rng(0)
    X = np.linspace(-1, 1, 50)
    Y = 2 + 3 * X + rng.normal(scale=0.1, size=50)

    dups, dupsid = make_dups(X)

    V, B, R, rate = _bw_mse(
        Y=Y,
        X=X,
        Z=None,
        c=0.0,
        p=1,
        deriv=0,
        p_bias=2,
        h_var=0.8,
        h_bias=1.2,
        n_matches=49,
        dups=dups,
        dupsid=dupsid,
    )

    assert np.isfinite(V)
    assert np.isfinite(B)
    assert np.isfinite(R)
    assert rate == pytest.approx(1 / 5)


# ============================================================
# Preprocessing helpers
# ============================================================


def test_prepare_inputs():
    x = np.array([3.0, 1.0, np.nan, 2.0])
    y = np.array([30.0, np.nan, 20.0, 10.0])
    Z = np.array([1.0, 2.0, 3.0, 4.0])

    x2, y2, Z2, w2, c2 = _prepare_inputs(y, x, c=1.0, Z=Z)

    assert c2 == 0
    assert w2 is None
    assert np.all(np.diff(x2[:, 0]) >= 0)
    assert x2.shape[0] == 2

    # weights are carried through the same subset/NaN/sort alignment
    wts = np.array([10.0, 20.0, 30.0, 40.0])
    x3, y3, Z3, w3, _c3 = _prepare_inputs(y, x, c=1.0, Z=Z, w=wts)
    assert w3.shape[0] == x3.shape[0] == 2
    # rows with NaN (indices 1 and 2) dropped -> weights 10 (x=3) and 40 (x=2)
    # after sorting by x: x=2 (w=40) then x=3 (w=10)
    assert np.allclose(w3[:, 0], np.array([40.0, 10.0]))


def test_split_lr():
    x = np.array([[-1.0], [0.0], [1.0]])
    y = np.array([[1.0], [2.0], [3.0]])

    out = _split_lr(x, y, None)

    assert out["N_l"] == 1
    assert out["N_r"] == 2


def test_make_dups():
    x = np.array([1.0, 1.0, 2.0, 1.0])
    dups, dupsid = make_dups(x)

    np.testing.assert_array_equal(dups, [3, 3, 1, 3])
    np.testing.assert_array_equal(dupsid, [1, 2, 1, 3])


# ============================================================
# bin_data
# ============================================================


def test_bin_data_1d_shapes_counts_and_no_straddle():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 500)
    y = 2.0 + 3.0 * x + rng.normal(0, 0.01, 500)
    df = pd.DataFrame({"score": x, "Y": y})

    out = bin_data(df, ["score"], ["Y"], cutoff=0.0, bin_spec=10)

    # required columns
    assert {"score", "Y", "sd_Y", "n", "side"}.issubset(out.columns)
    # counts partition the raw rows
    assert out["n"].sum() == len(df)
    # no cell straddles the cutoff: side sign matches centroid sign
    assert ((out["score"] >= 0) == (out["side"] == 1)).all()
    # centroids preserve the linear relationship (resid ~ noise scale)
    resid = np.abs(out["Y"] - (2.0 + 3.0 * out["score"]))
    assert resid.max() < 0.1


def test_bin_data_multidim_group_and_sparsity():
    rng = np.random.default_rng(1)
    K = 3
    P = rng.dirichlet(np.ones(K), size=400)
    df = pd.DataFrame(P, columns=[f"p{k}" for k in range(K)])
    df["Y"] = P @ np.array([1.0, 2.0, 3.0]) + rng.normal(0, 0.01, 400)
    T = P.argmax(axis=1)

    out = bin_data(df, ["p0", "p1", "p2"], ["Y"], group=T, bin_spec=4)

    assert {"p0", "p1", "p2", "Y", "sd_Y", "n", "group"}.issubset(out.columns)
    assert out["n"].sum() == len(df)
    # sparse: only non-empty cells, far fewer than 4**3 * K
    assert len(out) <= 4 ** 3 * K


def test_bin_data_cutoff_requires_single_coord():
    df = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0], "Y": [1.0, 2.0]})
    with pytest.raises(ValueError):
        bin_data(df, ["a", "b"], ["Y"], cutoff=0.5)


# ============================================================
# resolve_binning
# ============================================================


def test_resolve_binning_three_cases():
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 300)
    df = pd.DataFrame({"score": x, "Y": 1.0 + x})

    # case 1: raw -> no weights
    work1, w1 = resolve_binning(df, ["score"], ["Y"])
    assert w1 is None
    assert len(work1) == len(df)

    # case 2: bin now -> weights are the per-bin counts summing to N
    work2, w2 = resolve_binning(df, ["score"], ["Y"], cutoff=0.0, bin=True, bin_spec=8)
    assert w2 is not None
    assert w2.sum() == len(df)
    assert len(work2) == len(w2)

    # case 3: already binned -> weights read from weight_col
    work3, w3 = resolve_binning(work2, ["score"], ["Y"], binned_data=True, weight_col="n")
    assert np.allclose(w3, work2["n"].to_numpy())


def test_resolve_binning_guardrails():
    df = pd.DataFrame({"score": [0.0, 1.0], "Y": [1.0, 2.0]})
    with pytest.raises(ValueError):
        resolve_binning(df, ["score"], ["Y"], bin=True, binned_data=True)
    with pytest.raises(ValueError):
        # binned_data but no weight column
        resolve_binning(df, ["score"], ["Y"], binned_data=True, weight_col="n")

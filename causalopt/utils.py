import numpy as np
from numpy.random import default_rng
from scipy.linalg import qr


def poly_eval(x_vec: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised polynomial evaluation.

    Parameters
    ----------
    x_vec : array-like, shape (m,)
        1‑D sample points.
    v : array-like, shape (n,)
        Coefficients ``v[0] + v[1]·x + … + v[n-1]·xⁿ`` (ascending order).

    Returns
    -------
    np.ndarray, shape (m,)
        Evaluated polynomial values.
    """
    x = np.vander(x_vec, N=len(b), increasing=True)
    b = np.asarray(b).ravel()
    return x @ b


def sim_poly_ic(
    x: np.ndarray,
    b: np.ndarray,
    V: np.ndarray,
    nsim: int = 10000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulation-based confidence / prediction band.

    Parameters
    ----------
    x       : 1-D array of points to predict
    b, V    : coefficient array and robust VC matrix
    nsim    : number of Monte-Carlo draws
    alpha   : nominal coverage, e.g. 0.95
    seed    : int or None for reproducibility
    """
    rng = default_rng(seed)
    p = len(b)
    X = np.vander(x, N=p, increasing=True)
    b = np.asarray(b).ravel()
    V = np.asarray(V)

    # 1. Draw beta
    beta_draws = rng.multivariate_normal(mean=b, cov=V, size=nsim)  # (nsim, p+1)

    # 2. Predict
    yhat_sim = X @ beta_draws.T  # (n_pts, nsim)

    # 3. Summaries
    fit = yhat_sim.mean(axis=1)
    lower = np.quantile(yhat_sim, alpha / 2, axis=1)
    upper = np.quantile(yhat_sim, 1 - alpha / 2, axis=1)
    return fit, lower, upper


# ============================================================
# Basic matrix utilities
# ============================================================


def tomat(x: np.ndarray) -> np.ndarray:
    """Reshape a 1-D array into a 2-D column matrix of shape (len(x), 1)."""
    return x.reshape(len(x), -1)


def ncol(x: np.ndarray) -> int:
    """Return the number of columns in x, or 1 if x is 1-D."""
    try:
        return x.shape[1]
    except Exception:
        return 1


def crossprod(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Compute x.T @ x, or x.T @ y if y is provided."""
    if y is None:
        return x.T @ x
    return x.T @ y


def nanmat(n: int, m: int | None = None) -> np.ndarray:
    """Create a NaN-filled array of shape (n,) or (n, m) if m is given."""
    if m is None:
        M = np.empty((n,))
    else:
        M = np.empty((n, m))
    M.fill(np.nan)
    return M


def inv_chol(x: np.ndarray) -> np.ndarray:
    """Compute the inverse of a symmetric positive-definite matrix via Cholesky decomposition."""
    Linv = np.linalg.inv(np.linalg.cholesky(x))
    return crossprod(Linv, Linv)


def qrXXinv(x: np.ndarray) -> np.ndarray:
    """Compute (x.T @ x)^{-1} using Cholesky decomposition."""
    return inv_chol(crossprod(x, x))


def complete_cases(x: np.ndarray) -> np.ndarray:
    """Return a boolean mask of rows in x that contain no NaN values."""
    return np.all(~np.isnan(x), axis=1)


def covs_drop_fun(z: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    """Drop linearly dependent columns from z using QR decomposition with column pivoting."""
    q, r, pivot = qr(a=z, pivoting=True)
    keep = pivot[np.abs(np.diagonal(r)) > tol]
    return z[:, keep]


# ============================================================
# Kernel and RD-specific utilities
# ============================================================


def triangular_kernel(X: np.ndarray, c: float, h: float) -> np.ndarray:
    """Compute triangular kernel weights centered at c with bandwidth h."""
    u = (X - c) / h
    return ((1 - np.abs(u)) * (np.abs(u) <= 1)) / h


def nn_residuals(
    X: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray | None,
    matches: int,
    dups: np.ndarray,
    dupsid: np.ndarray,
) -> np.ndarray:
    """
    Compute nearest-neighbor residuals for variance estimation in RD designs.

    For each observation, the residual is formed using its nearest neighbors
    (determined by matches, dups, and dupsid), producing heteroskedasticity-
    robust inputs for the variance estimator.
    """
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = len(y)

    if Z is not None:
        Z = np.asarray(Z)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        dZ = ncol(Z)
    else:
        dZ = 0

    res = nanmat(n, 1 + dZ)

    for pos in range(n):
        rpos = dups[pos] - dupsid[pos]
        lpos = dupsid[pos] - 1

        while lpos + rpos < min(matches, n - 1):
            if pos - lpos - 1 < 0:
                rpos += dups[pos + rpos + 1]
            elif pos + rpos + 1 >= n:
                lpos += dups[pos - lpos - 1]
            elif (X[pos] - X[pos - lpos - 1]) > (X[pos + rpos + 1] - X[pos]):
                rpos += dups[pos + rpos + 1]
            else:
                lpos += dups[pos - lpos - 1]

        ind_J = np.arange(max(0, pos - lpos), min(n, pos + rpos) + 1)

        y_J = np.sum(y[ind_J]) - y[pos]
        Ji = len(ind_J) - 1
        res[pos, 0] = np.sqrt(Ji / (Ji + 1)) * (y[pos] - y_J / Ji)

        if Z is not None:
            for i in range(dZ):
                Z_J = np.sum(Z[ind_J, i]) - Z[pos, i]
                res[pos, 1 + i] = np.sqrt(Ji / (Ji + 1)) * (Z[pos, i] - Z_J / Ji)

    return res


def sandwich_se(RX: np.ndarray, res: np.ndarray, s: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the heteroskedasticity-robust sandwich variance matrix.

    Returns (RX * u).T @ (RX * u), where u is the effective residual vector
    (optionally projected onto direction s).
    """
    RX = np.asarray(RX)
    res = np.asarray(res)

    if s is None:
        u = res[:, [0]]
    else:
        s = np.asarray(s).reshape(-1, 1)
        u = res @ s

    return crossprod(RX * u, RX * u)


# ============================================================
# RD bandwidth core
# ============================================================


def _bw_mse(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray | None,
    c: float,
    p: int,
    deriv: int,
    p_bias: int,
    h_var: float,
    h_bias: float,
    n_matches: int,
    dups: np.ndarray,
    dupsid: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Core variance and bias components for MSE-optimal RD bandwidth selection.

    Implements the plug-in bandwidth formula from Calonico, Cattaneo, and
    Titiunik (2014) under a fixed configuration:
      - triangular kernel
      - nearest-neighbor variance estimator
      - sharp RD design
      - scale regularization always ON

    Returns the four scalars (V, B, R, rate) needed by the three-step
    bandwidth selector in bw_select.
    """

    # ----------------------------
    # Variance estimation (at h_var)
    # ----------------------------
    w_h = triangular_kernel(X, c, h_var).reshape(-1, 1)
    mask_h = (w_h > 0).reshape(-1)

    eY = Y[mask_h].reshape(-1, 1)
    eX = X[mask_h].reshape(-1, 1)
    eW = w_h[mask_h].reshape(-1, 1)

    n_h = int(np.sum(mask_h))

    Rp = nanmat(n_h, p + 1)
    for j in range(p + 1):
        Rp[:, j] = (eX[:, 0] - c) ** j

    Gp_inv = qrXXinv(Rp * np.sqrt(eW))

    s = np.array([1.0])
    dZ = 0
    eZ = None

    if Z is not None:
        Z = np.asarray(Z)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        eZ = Z[mask_h, :]
        dZ = ncol(eZ)

        D_h = np.column_stack((eY, eZ))
        U = crossprod(Rp * eW, D_h)
        ZWD = crossprod(eZ * eW, D_h)

        colsZ = np.arange(1, 1 + dZ)
        UiGU = crossprod(U[:, colsZ], Gp_inv @ U)

        ZWZ = ZWD[:, colsZ] - UiGU[:, colsZ]
        ZWY = ZWD[:, :1] - UiGU[:, :1]

        gamma = np.linalg.pinv(ZWZ) @ ZWY
        s = np.concatenate(([1.0], -gamma[:, 0]))

    res_h = nn_residuals(eX.flatten(), eY.flatten(), eZ, n_matches, dups[mask_h], dupsid[mask_h])

    WRp = Rp * eW
    res_eff = res_h @ s.reshape(-1, 1)

    Vce_h = sandwich_se(WRp, res_eff)
    var_h = (Gp_inv @ Vce_h @ Gp_inv)[deriv, deriv]

    bias_vec = crossprod(WRp, ((eX[:, 0] - c) / h_var) ** (p + 1))
    H_pow = np.array([h_var**j for j in range(p + 1)]).reshape(-1, 1)
    bias_const = (H_pow * (Gp_inv @ bias_vec.reshape(-1, 1)))[deriv, 0]

    # ----------------------------
    # Bias estimation (at h_bias)
    # ----------------------------
    w_b = triangular_kernel(X, c, h_bias).reshape(-1, 1)
    mask_b = (w_b > 0).reshape(-1)

    y_b = Y[mask_b].reshape(-1, 1)
    x_b = X[mask_b].reshape(-1, 1)
    W_b = w_b[mask_b].reshape(-1, 1)

    Rq = nanmat(int(np.sum(mask_b)), p_bias + 1)
    for j in range(p_bias + 1):
        Rq[:, j] = (x_b[:, 0] - c) ** j

    Gq_inv = qrXXinv(Rq * np.sqrt(W_b))

    if Z is not None:
        eZB = Z[mask_b, :]
        D_b = np.column_stack((y_b, eZB))
    else:
        D_b = y_b

    beta_q = Gq_inv @ crossprod(Rq * W_b, D_b)
    beta_p1 = float(np.dot(s, beta_q[-1, :])) if beta_q.ndim == 2 else float(beta_q[-1])

    res_b = nn_residuals(
        x_b.flatten(),
        y_b.flatten(),
        eZB if Z is not None else None,
        n_matches,
        dups[mask_b],
        dupsid[mask_b],
    )

    WRq = Rq * W_b
    res_b_eff = res_b @ s.reshape(-1, 1)

    Vce_b = sandwich_se(WRq, res_b_eff)
    var_b = (Gq_inv @ Vce_b @ Gq_inv)[-1, -1]

    reg = 3 * (bias_const**2) * var_b

    B = np.sqrt(2 * (p + 1 - deriv)) * bias_const * beta_p1
    V = (2 * deriv + 1) * (h_var ** (2 * deriv + 1)) * var_h
    R = (2 * (p + 1 - deriv)) * reg
    rate = 1 / (2 * p + 3)

    return V, B, R, rate


# ============================================================
# Shared preprocessing helpers
# ============================================================


def _prepare_inputs(
    y: np.ndarray,
    x: np.ndarray,
    c: float,
    Z: np.ndarray | None,
    subset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    """
    Validate and preprocess RD inputs.

    Converts y, x, and Z to arrays, applies an optional row subset, recenters
    x at the cutoff c, removes rows with missing values, and returns the data
    sorted by x. The returned cutoff is always 0.
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)

    if Z is not None:
        Z = np.asarray(Z)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

    if subset is not None:
        x = x[subset]
        y = y[subset]
        if Z is not None:
            Z = Z[subset]

    if c != 0:
        x = x - c
        c = 0

    ok = complete_cases(x) & complete_cases(y)
    if Z is not None:
        ok &= complete_cases(Z)

    x, y = x[ok], y[ok]
    if Z is not None:
        Z = Z[ok]

    order = np.argsort(x[:, 0])
    x, y = x[order], y[order]
    if Z is not None:
        Z = Z[order]

    return x, y, Z, c


def _split_lr(x: np.ndarray, y: np.ndarray, Z: np.ndarray | None) -> dict:
    """Split x, y, and Z into left (x < 0) and right (x >= 0) subsamples."""
    left = x[:, 0] < 0
    right = ~left

    out = {
        "X_l": x[left],
        "X_r": x[right],
        "Y_l": y[left],
        "Y_r": y[right],
        "N_l": np.sum(left),
        "N_r": np.sum(right),
    }

    if Z is not None:
        out["Z_l"] = Z[left]
        out["Z_r"] = Z[right]
    else:
        out["Z_l"] = out["Z_r"] = None

    return out


def make_dups(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute duplicate counts and within-duplicate IDs for each element of x.

    Returns dups (how many times each value appears in x) and dupsid (the
    sequential index of each occurrence within its group of duplicates).
    """
    x = np.asarray(x).flatten()
    uniq, cnt = np.unique(x, return_counts=True)
    cnt_map = dict(zip(uniq, cnt))
    dups = np.array([cnt_map[v] for v in x], dtype=int)

    seen = {}
    dupsid = np.zeros_like(dups)
    for i, v in enumerate(x):
        seen[v] = seen.get(v, 0) + 1
        dupsid[i] = seen[v]

    return dups, dupsid

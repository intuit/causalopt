import numpy as np
from numpy.random import default_rng
from scipy.linalg import qr


def poly_eval(x_vec, b):
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


def sim_poly_ic(x, b, V, nsim=10000, alpha=0.05, seed=None):
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


def tomat(x):
    return x.reshape(len(x), -1)


def ncol(x):
    try:
        return x.shape[1]
    except Exception:
        return 1


def crossprod(x, y=None):
    if y is None:
        return x.T @ x
    return x.T @ y


def nanmat(n, m=None):
    if m is None:
        M = np.empty((n,))
    else:
        M = np.empty((n, m))
    M.fill(np.nan)
    return M


def inv_chol(x):
    Linv = np.linalg.inv(np.linalg.cholesky(x))
    return crossprod(Linv, Linv)


def qrXXinv(x):
    return inv_chol(crossprod(x, x))


def complete_cases(x):
    return np.all(~np.isnan(x), axis=1)


def covs_drop_fun(z, tol=1e-5):
    q, r, pivot = qr(a=z, pivoting=True)
    keep = pivot[np.abs(np.diagonal(r)) > tol]
    return z[:, keep]


# ============================================================
# Kernel and RD-specific utilities
# ============================================================


def rdrobust_kweight(X, c, h):
    u = (X - c) / h
    return ((1 - np.abs(u)) * (np.abs(u) <= 1)) / h


def rdrobust_res(X, y, Z, matches, dups, dupsid):
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


def rdrobust_vce(RX, res, s=None):
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


def rdrobust_bw(Y, X, Z, c, o, nu, o_B, h_V, h_B, nnmatch, dups, dupsid):
    """
    Core building block for RD bandwidth selection.
    Fixed configuration:
      - triangular kernel
      - nearest-neighbor variance
      - sharp RD
      - scale regularization always ON
    """

    # ----------------------------
    # Variance part
    # ----------------------------
    w = rdrobust_kweight(X, c, h_V).reshape(-1, 1)
    ind_V = (w > 0).reshape(-1)

    eY = Y[ind_V].reshape(-1, 1)
    eX = X[ind_V].reshape(-1, 1)
    eW = w[ind_V].reshape(-1, 1)

    n_V = int(np.sum(ind_V))

    R_V = nanmat(n_V, o + 1)
    for j in range(o + 1):
        R_V[:, j] = (eX[:, 0] - c) ** j

    invG_V = qrXXinv(R_V * np.sqrt(eW))

    s = np.array([1.0])
    dZ = 0
    eZ = None

    if Z is not None:
        Z = np.asarray(Z)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        eZ = Z[ind_V, :]
        dZ = ncol(eZ)

        D_V = np.column_stack((eY, eZ))

        U = crossprod(R_V * eW, D_V)
        ZWD = crossprod(eZ * eW, D_V)

        colsZ = np.arange(1, 1 + dZ)
        UiGU = crossprod(U[:, colsZ], invG_V @ U)

        ZWZ = ZWD[:, colsZ] - UiGU[:, colsZ]
        ZWY = ZWD[:, :1] - UiGU[:, :1]

        gamma = np.linalg.pinv(ZWZ) @ ZWY
        s = np.concatenate(([1.0], -gamma[:, 0]))

    res_V = rdrobust_res(eX.flatten(), eY.flatten(), eZ, nnmatch, dups[ind_V], dupsid[ind_V])

    RX_V = R_V * eW
    res_eff = res_V @ s.reshape(-1, 1)

    aux_V = rdrobust_vce(RX_V, res_eff)
    V_V = (invG_V @ aux_V @ invG_V)[nu, nu]

    v = crossprod(RX_V, ((eX[:, 0] - c) / h_V) ** (o + 1))

    Hp = np.array([h_V**j for j in range(o + 1)]).reshape(-1, 1)
    BConst = (Hp * (invG_V @ v.reshape(-1, 1)))[nu, 0]

    # ----------------------------
    # Bias part
    # ----------------------------
    wB = rdrobust_kweight(X, c, h_B).reshape(-1, 1)
    ind_B = (wB > 0).reshape(-1)

    eYB = Y[ind_B].reshape(-1, 1)
    eXB = X[ind_B].reshape(-1, 1)
    eWB = wB[ind_B].reshape(-1, 1)

    R_B = nanmat(int(np.sum(ind_B)), o_B + 1)
    for j in range(o_B + 1):
        R_B[:, j] = (eXB[:, 0] - c) ** j

    invG_B = qrXXinv(R_B * np.sqrt(eWB))

    if Z is not None:
        eZB = Z[ind_B, :]
        D_B = np.column_stack((eYB, eZB))
    else:
        D_B = eYB

    beta_B = invG_B @ crossprod(R_B * eWB, D_B)

    beta_last = float(np.dot(s, beta_B[-1, :])) if beta_B.ndim == 2 else float(beta_B[-1])

    res_B = rdrobust_res(
        eXB.flatten(),
        eYB.flatten(),
        eZB if Z is not None else None,
        nnmatch,
        dups[ind_B],
        dupsid[ind_B],
    )

    RX_B = R_B * eWB
    resB_eff = res_B @ s.reshape(-1, 1)

    aux_B = rdrobust_vce(RX_B, resB_eff)
    V_B = (invG_B @ aux_B @ invG_B)[-1, -1]

    BWreg = 3 * (BConst**2) * V_B

    B = np.sqrt(2 * (o + 1 - nu)) * BConst * beta_last
    V = (2 * nu + 1) * (h_V ** (2 * nu + 1)) * V_V
    R = (2 * (o + 1 - nu)) * BWreg
    rate = 1 / (2 * o + 3)

    return V, B, R, rate


# ============================================================
# Shared preprocessing helpers
# ============================================================


def _prepare_inputs(y, x, c, Z, subset=None):
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


def _split_lr(x, y, Z):
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


def make_dups(x):
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

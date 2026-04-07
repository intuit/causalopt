import numpy as np
import pandas as pd
import scipy.stats as sct
from scipy.stats import t as student_t

from causalopt.utils import (
    _bw_mse,
    _prepare_inputs,
    _split_lr,
    crossprod,
    inv_chol,
    make_dups,
    ncol,
    nn_residuals,
    qrXXinv,
    sandwich_se,
    triangular_kernel,
)


def bw_select(
    y: np.ndarray,
    x: np.ndarray,
    c: float = 0,
    p: int = 1,
    covariates: np.ndarray | None = None,
    subset: np.ndarray | None = None,
) -> dict:
    """
    Select MSE-optimal bandwidths for sharp Regression Discontinuity (RD)
    estimation under a fully fixed configuration.

    This function implements the MSE-optimal bandwidth selection procedure
    for sharp RD designs using local polynomial regression with a triangular
    kernel and nearest-neighbor variance estimation.

    The implementation follows the MSE-optimal bandwidth procedure from
    Calonico, Cattaneo, and Titiunik (2014) under a fixed configuration:

        • Sharp RD design
        • Triangular kernel
        • Nearest-neighbor variance estimator ("nn")
        • MSE-optimal bandwidth selection ("mserd")
        • Regularization always enabled
        • Local polynomial order p for point estimation
        • Bias polynomial order q = p + 1

    The procedure proceeds in three steps:
        1. Selection of a preliminary bandwidth
        2. Selection of a bias bandwidth
        3. Selection of the final estimation bandwidth

    When covariates are supplied, they are incorporated linearly in all local
    polynomial regressions using a Frisch–Waugh–Lovell partialling-out strategy
    following the Frisch–Waugh–Lovell partialling-out approach described in
    Calonico, Cattaneo, and Titiunik (2014).

    Parameters
    ----------
    y : array-like
        Outcome variable.

    x : array-like
        Running (forcing) variable.

    c : float, default = 0
        Cutoff value. If nonzero, the running variable is internally recentered
        so that the cutoff is at zero.

    p : int, default = 1
        Order of the local polynomial used for point estimation.

    covariates : array-like or None, default = None
        Optional covariates to be included linearly in the local polynomial
        regressions.

    Returns
    -------
    results : dict
        Dictionary containing selected bandwidths and metadata. The structure is:

        {
            "bwselect": "mserd",
            "kernel": "tri",
            "vce": "nn",
            "p": p,
            "q": p + 1,
            "c": 0,
            "N": {"left": N_l, "right": N_r},
            "M": {"left": M_l, "right": M_r},
            "bandwidths": {
                "h_left": h,
                "h_right": h,
                "b_left": b,
                "b_right": b
            },
            "internals": {
                "c_bw": preliminary bandwidth,
                "d_bw": intermediate bandwidth
            }
        }

    Notes
    -----
    • The same bandwidth is returned on both sides of the cutoff.
    • No inference or point estimation is performed in this function.
    • All calculations are deterministic and reproducible.
    """

    x, y, Z, c = _prepare_inputs(y, x, c, covariates, subset=subset)
    parts = _split_lr(x, y, Z)

    q = p + 1
    n_matches = 3

    x_iq = np.quantile(x, 0.75) - np.quantile(x, 0.25)
    BWp = min(np.std(x, ddof=1), x_iq / 1.349)

    M_l = len(np.unique(parts["X_l"]))
    M_r = len(np.unique(parts["X_r"]))
    M = M_l + M_r

    C_c = 2.576
    c_bw = C_c * BWp * (M ** (-1 / 5))
    bw_max = max(abs(x.min()), abs(x.max()))
    c_bw = min(c_bw, bw_max)

    dups_l, dupsid_l = make_dups(parts["X_l"])
    dups_r, dupsid_r = make_dups(parts["X_r"])

    range_l = abs(parts["X_l"].min())
    range_r = abs(parts["X_r"].max())

    def side_bw(Y, X, Z, dups, dupsid, p, deriv, p_bias, h_var, h_bias):
        return _bw_mse(Y, X, Z, c, p, deriv, p_bias, h_var, h_bias, n_matches, dups, dupsid)

    C_d_l = side_bw(
        parts["Y_l"],
        parts["X_l"],
        parts["Z_l"],
        dups_l,
        dupsid_l,
        q + 1,
        q + 1,
        q + 2,
        c_bw,
        range_l,
    )
    C_d_r = side_bw(
        parts["Y_r"],
        parts["X_r"],
        parts["Z_r"],
        dups_r,
        dupsid_r,
        q + 1,
        q + 1,
        q + 2,
        c_bw,
        range_r,
    )

    d_bw = ((C_d_l[0] + C_d_r[0]) / (C_d_r[1] - C_d_l[1]) ** 2) ** C_d_l[3]
    d_bw = min(d_bw, bw_max)

    C_b_l = side_bw(
        parts["Y_l"],
        parts["X_l"],
        parts["Z_l"],
        dups_l,
        dupsid_l,
        q,
        p + 1,
        q + 1,
        c_bw,
        d_bw,
    )
    C_b_r = side_bw(
        parts["Y_r"],
        parts["X_r"],
        parts["Z_r"],
        dups_r,
        dupsid_r,
        q,
        p + 1,
        q + 1,
        c_bw,
        d_bw,
    )

    b_bw = ((C_b_l[0] + C_b_r[0]) / ((C_b_r[1] - C_b_l[1]) ** 2 + (C_b_r[2] + C_b_l[2]))) ** C_b_l[
        3
    ]
    b_bw = min(b_bw, bw_max)

    C_h_l = side_bw(parts["Y_l"], parts["X_l"], parts["Z_l"], dups_l, dupsid_l, p, 0, q, c_bw, b_bw)
    C_h_r = side_bw(parts["Y_r"], parts["X_r"], parts["Z_r"], dups_r, dupsid_r, p, 0, q, c_bw, b_bw)

    h_bw = ((C_h_l[0] + C_h_r[0]) / ((C_h_r[1] - C_h_l[1]) ** 2 + (C_h_r[2] + C_h_l[2]))) ** C_h_l[
        3
    ]
    h_bw = min(h_bw, bw_max)

    return {
        "bwselect": "mserd",
        "kernel": "tri",
        "vce": "nn",
        "p": p,
        "q": q,
        "c": 0,
        "N": {"left": parts["N_l"], "right": parts["N_r"]},
        "M": {"left": M_l, "right": M_r},
        "bandwidths": {
            "h_left": float(h_bw),
            "h_right": float(h_bw),
            "b_left": float(b_bw),
            "b_right": float(b_bw),
        },
        "internals": {"c_bw": float(c_bw), "d_bw": float(d_bw)},
    }


def rd_estimate(
    y: np.ndarray,
    x: np.ndarray,
    c: float = 0,
    p: int = 1,
    covariates: np.ndarray | None = None,
    subset: np.ndarray | None = None,
) -> dict:
    """
    Estimate a sharp Regression Discontinuity (RD) treatment effect at the
    cutoff using local polynomial regression.

    This function computes conventional and bias-corrected RD point estimates,
    standard errors, and confidence intervals under a fully fixed estimation
    configuration following Calonico, Cattaneo, and Titiunik (2014).

    The estimator uses:
        • A sharp RD design
        • Local polynomial regression of order p
        • A triangular kernel
        • Nearest-neighbor variance estimation
        • MSE-optimal bandwidths selected internally via `bw_select`
        • Robust bias correction following Calonico, Cattaneo, and Titiunik (2014)

    Covariates, if supplied, are incorporated linearly using a
    Frisch–Waugh–Lovell partialling-out approach.

    Parameters
    ----------
    y : array-like
        Outcome variable.

    x : array-like
        Running (forcing) variable.

    c : float, default = 0
        Cutoff value. If nonzero, the running variable is internally recentered
        so that the cutoff is at zero.

    p : int, default = 1
        Order of the local polynomial used for point estimation.

    covariates : array-like or None, default = None
        Optional covariates to be included linearly in the local polynomial
        regressions.

    Returns
    -------
    results : dict
        Dictionary containing RD point estimates, inference quantities, and
        bandwidths. The structure is:

        {
            "tau": {
                "conventional": conventional estimate,
                "bias_corrected": bias-corrected estimate
            },
            "se": {
                "conventional": conventional standard error,
                "robust": robust standard error
            },
            "ci": {
                "conventional": (lower, upper),
                "bias_corrected": (lower, upper),
                "robust": (lower, upper)
            },
            "bandwidths": {
                "h_l": left estimation bandwidth,
                "h_r": right estimation bandwidth,
                "b_l": left bias bandwidth,
                "b_r": right bias bandwidth
            },
            "p": p,
            "q": p + 1,
            "c": 0,
            "N": {"left": N_l, "right": N_r},
            "N_h": {"left": N_h_l, "right": N_h_r},
            "N_b": {"left": N_b_l, "right": N_b_r}
        }

    Notes
    -----
    • Confidence intervals are computed using a normal approximation.
    • The cutoff is always normalized to zero internally.
    • All reported quantities are invariant to monotone transformations of x.
    • This function performs estimation and inference only; no plotting
      functionality is included.
    """
    x, y, Z, c = _prepare_inputs(y, x, c, covariates)

    q = p + 1
    deriv = 0
    n_matches = 3
    level = 95
    scalepar = 1.0

    # --------------------------------------------------
    # Bandwidth selection
    # --------------------------------------------------
    bw = bw_select(y, x, c=0, p=p, covariates=Z)
    h_l = bw["bandwidths"]["h_left"]
    h_r = bw["bandwidths"]["h_right"]
    b_l = bw["bandwidths"]["b_left"]
    b_r = bw["bandwidths"]["b_right"]

    # --------------------------------------------------
    # Split sample
    # --------------------------------------------------
    left = x[:, 0] < 0
    right = ~left

    X_l, X_r = x[left, 0], x[right, 0]
    Y_l, Y_r = y[left, 0], y[right, 0]

    Z_l = Z[left, :] if Z is not None else None
    Z_r = Z[right, :] if Z is not None else None

    N_l, N_r = len(X_l), len(X_r)

    # --------------------------------------------------
    # Kernel weights
    # --------------------------------------------------
    w_h_l = triangular_kernel(X_l, 0, h_l)
    w_h_r = triangular_kernel(X_r, 0, h_r)
    w_b_l = triangular_kernel(X_l, 0, b_l)
    w_b_r = triangular_kernel(X_r, 0, b_r)

    ind_h_l, ind_h_r = w_h_l > 0, w_h_r > 0
    ind_b_l, ind_b_r = w_b_l > 0, w_b_r > 0

    ind_l = ind_b_l if h_l <= b_l else ind_h_l
    ind_r = ind_b_r if h_r <= b_r else ind_h_r

    N_h_l, N_h_r = ind_h_l.sum(), ind_h_r.sum()
    N_b_l, N_b_r = ind_b_l.sum(), ind_b_r.sum()

    # --------------------------------------------------
    # Effective samples
    # --------------------------------------------------
    eX_l, eX_r = X_l[ind_l], X_r[ind_r]
    eY_l, eY_r = Y_l[ind_l], Y_r[ind_r]

    eZ_l = Z_l[ind_l, :] if Z is not None else None
    eZ_r = Z_r[ind_r, :] if Z is not None else None

    W_h_l = w_h_l[ind_l].reshape(-1, 1)
    W_h_r = w_h_r[ind_r].reshape(-1, 1)
    W_b_l = w_b_l[ind_l].reshape(-1, 1)
    W_b_r = w_b_r[ind_r].reshape(-1, 1)

    # --------------------------------------------------
    # NN duplicates
    # --------------------------------------------------
    dups_l, dupsid_l = make_dups(eX_l)
    dups_r, dupsid_r = make_dups(eX_r)

    # --------------------------------------------------
    # Design matrices
    # --------------------------------------------------
    u_l = ((eX_l - c) / h_l).reshape(-1, 1)
    u_r = ((eX_r - c) / h_r).reshape(-1, 1)

    R_q_l = np.column_stack([(eX_l - c) ** j for j in range(q + 1)])
    R_q_r = np.column_stack([(eX_r - c) ** j for j in range(q + 1)])

    R_p_l = R_q_l[:, : p + 1]
    R_p_r = R_q_r[:, : p + 1]

    invG_p_l = qrXXinv(np.sqrt(W_h_l) * R_p_l)
    invG_p_r = qrXXinv(np.sqrt(W_h_r) * R_p_r)
    invG_q_l = qrXXinv(np.sqrt(W_b_l) * R_q_l)
    invG_q_r = qrXXinv(np.sqrt(W_b_r) * R_q_r)

    # --------------------------------------------------
    # Outcomes (with covariates)
    # --------------------------------------------------
    D_l = eY_l.reshape(-1, 1)
    D_r = eY_r.reshape(-1, 1)

    if Z is not None:
        D_l = np.column_stack((D_l, eZ_l))
        D_r = np.column_stack((D_r, eZ_r))
        U_p_l = crossprod(R_p_l * W_h_l, D_l)
        U_p_r = crossprod(R_p_r * W_h_r, D_r)

    # --------------------------------------------------
    # Bias correction matrices
    # --------------------------------------------------
    L_l = crossprod(R_p_l * W_h_l, u_l ** (p + 1))
    L_r = crossprod(R_p_r * W_h_r, u_r ** (p + 1))

    e_p1 = np.zeros((q + 1, 1))
    e_p1[p + 1] = 1

    Q_q_l = (
        (R_p_l * W_h_l).T - h_l ** (p + 1) * L_l @ e_p1.T @ ((invG_q_l @ R_q_l.T).T * W_b_l).T
    ).T

    Q_q_r = (
        (R_p_r * W_h_r).T - h_r ** (p + 1) * L_r @ e_p1.T @ ((invG_q_r @ R_q_r.T).T * W_b_r).T
    ).T

    # --------------------------------------------------
    # Estimation
    # --------------------------------------------------
    beta_p_l = invG_p_l @ crossprod(R_p_l * W_h_l, D_l)
    beta_p_r = invG_p_r @ crossprod(R_p_r * W_h_r, D_r)

    beta_bc_l = invG_p_l @ crossprod(Q_q_l, D_l)
    beta_bc_r = invG_p_r @ crossprod(Q_q_r, D_r)

    beta_p = beta_p_r - beta_p_l
    beta_bc = beta_bc_r - beta_bc_l

    # --------------------------------------------------
    # Covariate adjustment
    # --------------------------------------------------
    if Z is None:
        s_Y = np.array([1.0])
        tau_cl = scalepar * beta_p[deriv, 0]
        tau_bc = scalepar * beta_bc[deriv, 0]
    else:
        dZ = ncol(eZ_l)
        colsZ = np.arange(1, 1 + dZ)

        ZWD_l = crossprod(eZ_l * W_h_l, D_l)
        ZWD_r = crossprod(eZ_r * W_h_r, D_r)

        UiGU_l = crossprod(U_p_l[:, colsZ], invG_p_l @ U_p_l)
        UiGU_r = crossprod(U_p_r[:, colsZ], invG_p_r @ U_p_r)

        ZWZ = (ZWD_l[:, colsZ] - UiGU_l[:, colsZ]) + (ZWD_r[:, colsZ] - UiGU_r[:, colsZ])
        ZWY = (ZWD_l[:, :1] - UiGU_l[:, :1]) + (ZWD_r[:, :1] - UiGU_r[:, :1])

        gamma = np.linalg.pinv(ZWZ) @ ZWY
        s_Y = np.concatenate(([1.0], -gamma[:, 0]))

        tau_cl = float(s_Y @ beta_p[deriv, :])
        tau_bc = float(s_Y @ beta_bc[deriv, :])

    # --------------------------------------------------
    # Variance estimation
    # --------------------------------------------------

    res_h_l = nn_residuals(eX_l, eY_l, eZ_l, n_matches, dups_l, dupsid_l)
    res_h_r = nn_residuals(eX_r, eY_r, eZ_r, n_matches, dups_r, dupsid_r)

    res_h_l = res_h_l @ s_Y.reshape(-1, 1)
    res_h_r = res_h_r @ s_Y.reshape(-1, 1)

    V_Y_cl_l = invG_p_l @ sandwich_se(R_p_l * W_h_l, res_h_l) @ invG_p_l
    V_Y_cl_r = invG_p_r @ sandwich_se(R_p_r * W_h_r, res_h_r) @ invG_p_r

    V_Y_rb_l = invG_p_l @ sandwich_se(Q_q_l, res_h_l) @ invG_p_l
    V_Y_rb_r = invG_p_r @ sandwich_se(Q_q_r, res_h_r) @ invG_p_r

    V_tau_cl = (V_Y_cl_l + V_Y_cl_r)[deriv, deriv]
    V_tau_rb = (V_Y_rb_l + V_Y_rb_r)[deriv, deriv]

    se_cl = np.sqrt(V_tau_cl)
    se_rb = np.sqrt(V_tau_rb)

    z = -sct.norm.ppf((1 - level / 100) / 2)

    return {
        "tau": {"conventional": tau_cl, "bias_corrected": tau_bc},
        "se": {"conventional": se_cl, "robust": se_rb},
        "ci": {
            "conventional": (tau_cl - z * se_cl, tau_cl + z * se_cl),
            "bias_corrected": (tau_bc - z * se_cl, tau_bc + z * se_cl),
            "robust": (tau_bc - z * se_rb, tau_bc + z * se_rb),
        },
        "bandwidths": {"h_l": h_l, "h_r": h_r, "b_l": b_l, "b_r": b_r},
        "p": p,
        "q": q,
        "c": 0,
        "N": {"left": N_l, "right": N_r},
        "N_h": {"left": N_h_l, "right": N_h_r},
        "N_b": {"left": N_b_l, "right": N_b_r},
        "coef_poly": {
            "conventional": {"left": beta_p_l, "right": beta_p_r},
            "bias_corrected": {"left": beta_bc_l, "right": beta_bc_r},
        },
        "V_poly": {
            "conventional": {"left": V_Y_cl_l, "right": V_Y_cl_r},
            "bias_corrected": {"left": V_Y_rb_l, "right": V_Y_rb_r},
        },
    }


def rd_objects(
    y: np.ndarray,
    x: np.ndarray,
    c: float = 0,
    p: int = 1,
    covariates: np.ndarray | None = None,
    bw: list | None = None,
    subset: np.ndarray | None = None,
) -> dict:
    """
    Construct numerical objects required to produce a Regression Discontinuity
    (RD) plot under a fixed graphical configuration.

    Generates all intermediate numerical quantities needed to construct an RD
    plot following the binning strategy of Calonico, Cattaneo, Farrell, and
    Titiunik (2015), but does not perform any plotting itself. Returns
    bin-level summaries, fitted global polynomials, and associated variance
    estimates that can be used to construct custom RD visualizations.

    Fixed configuration:

        • Sharp RD design
        • Triangular kernel
        • Global polynomial of order p on each side of the cutoff
        • Full-support bandwidth for polynomial fitting
        • Mimicking-variance, evenly spaced bins ("esmv")
        • No clustering, no external weights
        • Optional linear covariate adjustment via FWL partialling-out

    When covariates are supplied, the outcome is residualized using the same
    Frisch–Waugh–Lovell logic as `rdplot`.

    Parameters
    ----------
    y : array-like
        Outcome variable.

    x : array-like
        Running (forcing) variable.

    c : float, default = 0
        Cutoff value.

    p : int, default = 1
        Order of the global polynomial fitted separately on each side of the
        cutoff.

    covariates : array-like or None, default = None
        Optional covariates to be included linearly in the global polynomial
        regressions.

    Returns
    -------
    results : dict
        Dictionary containing numerical objects used to construct an RD plot:

        {
            "binselect": "esmv",
            "kernel": "tri",
            "c": cutoff,
            "p": polynomial order,

            "N": {
                "left": number of observations left of cutoff,
                "right": number of observations right of cutoff
            },

            "h": {
                "left": polynomial bandwidth (left),
                "right": polynomial bandwidth (right)
            },

            "J": {
                "left": number of bins (left),
                "right": number of bins (right)
            },

            "J_IMSE": {
                "left": IMSE-optimal number of bins (left),
                "right": IMSE-optimal number of bins (right)
            },

            "J_MV": {
                "left": mimicking-variance number of bins (left),
                "right": mimicking-variance number of bins (right)
            },

            "bins": pandas.DataFrame
                Bin-level summaries including means, standard errors,
                confidence intervals, and bin boundaries.

            "poly": {
                "x_left": grid of x values (left),
                "y_left": fitted polynomial values (left),
                "x_right": grid of x values (right),
                "y_right": fitted polynomial values (right)
            },

            "coefficients": {
                "left": polynomial coefficients (left),
                "right": polynomial coefficients (right)
            },

            "vcov": {
                "left": variance–covariance matrix (left),
                "right": variance–covariance matrix (right)
            },

            "se_coef": {
                "left": standard errors of coefficients (left),
                "right": standard errors of coefficients (right)
            }
        }

    Notes
    -----
    • This function is descriptive and graphical in nature.
    • No RD treatment effect estimation or inference is performed.
    • The results are intended for visualization, diagnostics, and debugging.
    • The returned objects are sufficient to reproduce `rdplot` numerically.
    """

    x, y, Z, c = _prepare_inputs(y, x, c, covariates, subset=subset)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if not (x_min < c < x_max):
        raise ValueError("c must be strictly inside the support of x.")

    left = x[:, 0] < c
    right = ~left

    x_l, x_r = x[left], x[right]
    y_l, y_r = y[left], y[right]

    z_l = Z[left] if Z is not None else None
    z_r = Z[right] if Z is not None else None

    n_l, n_r = x_l.shape[0], x_r.shape[0]
    n = n_l + n_r
    if n < 20:
        raise ValueError("Not enough observations to perform bin calculations.")

    range_l = c - x_min
    range_r = x_max - c

    # --------------------------------------------------
    # Polynomial fit bandwidths
    # --------------------------------------------------
    if bw is None:
        h_l = range_l
        h_r = range_r
    else:
        h_l = bw[0]
        h_r = bw[1]

    # --------------------------------------------------
    # Polynomial design matrices
    # --------------------------------------------------
    R_p_l = np.column_stack([(x_l[:, 0] - c) ** j for j in range(p + 1)])
    R_p_r = np.column_stack([(x_r[:, 0] - c) ** j for j in range(p + 1)])

    W_l = triangular_kernel(x_l[:, 0], c, h_l).reshape(-1, 1)
    W_r = triangular_kernel(x_r[:, 0], c, h_r).reshape(-1, 1)

    invG_l = qrXXinv(np.sqrt(W_l) * R_p_l)
    invG_r = qrXXinv(np.sqrt(W_r) * R_p_r)

    # --------------------------------------------------
    # Polynomial coefficients (with covariates)
    # --------------------------------------------------
    if Z is None:
        gamma_l = invG_l @ crossprod(R_p_l * W_l, y_l)
        gamma_r = invG_r @ crossprod(R_p_r * W_r, y_r)
    else:
        D_l = np.column_stack((y_l, z_l))
        D_r = np.column_stack((y_r, z_r))

        U_l = crossprod(R_p_l * W_l, D_l)
        U_r = crossprod(R_p_r * W_r, D_r)

        beta_l = invG_l @ crossprod(R_p_l * W_l, D_l)
        beta_r = invG_r @ crossprod(R_p_r * W_r, D_r)

        ZWD_l = crossprod(z_l * W_l, D_l)
        ZWD_r = crossprod(z_r * W_r, D_r)

        dZ = ncol(z_l)
        colsZ = np.arange(1, 1 + dZ)

        UiGU_l = crossprod(U_l[:, colsZ], invG_l @ U_l)
        UiGU_r = crossprod(U_r[:, colsZ], invG_r @ U_r)

        ZWZ = (ZWD_l[:, colsZ] - UiGU_l[:, colsZ]) + (ZWD_r[:, colsZ] - UiGU_r[:, colsZ])
        ZWY = (ZWD_l[:, :1] - UiGU_l[:, :1]) + (ZWD_r[:, :1] - UiGU_r[:, :1])

        gamma = inv_chol(ZWZ) @ ZWY
        s_Y = np.concatenate(([1.0], -gamma[:, 0])).reshape(-1, 1)

        gamma_l = (s_Y.T @ beta_l.T).T
        gamma_r = (s_Y.T @ beta_r.T).T

    # --------------------------------------------------
    # Polynomial curves
    # --------------------------------------------------
    nplot = 500
    x_plot_l = np.linspace(c - h_l, c, nplot)
    x_plot_r = np.linspace(c, c + h_r, nplot)

    rplot_l = np.column_stack([(x_plot_l - c) ** j for j in range(p + 1)])
    rplot_r = np.column_stack([(x_plot_r - c) ** j for j in range(p + 1)])

    y_hat_l = rplot_l @ gamma_l
    y_hat_r = rplot_r @ gamma_r

    # --------------------------------------------------
    # Variance of coefficients
    # --------------------------------------------------
    res_l = y_l - R_p_l @ gamma_l
    res_r = y_r - R_p_r @ gamma_r

    V_l = invG_l @ sandwich_se(R_p_l * W_l, res_l) @ invG_l
    V_r = invG_r @ sandwich_se(R_p_r * W_r, res_r) @ invG_r

    se_beta_l = np.sqrt(np.diag(V_l))
    se_beta_r = np.sqrt(np.diag(V_r))

    # --------------------------------------------------
    # ESMV bin selection
    # --------------------------------------------------
    ord_l = np.argsort(x_l[:, 0])
    ord_r = np.argsort(x_r[:, 0])

    x_i_l, y_i_l = x_l[ord_l, 0], y_l[ord_l, 0]
    x_i_r, y_i_r = x_r[ord_r, 0], y_r[ord_r, 0]

    dx_l, dy_l = np.diff(x_i_l), np.diff(y_i_l)
    dx_r, dy_r = np.diff(x_i_r), np.diff(y_i_r)

    var_y_l = np.var(y_l)
    var_y_r = np.var(y_r)

    V_es_l = (0.5 / (c - x_min)) * np.sum(dx_l * dy_l**2)
    V_es_r = (0.5 / (x_max - c)) * np.sum(dx_r * dy_r**2)

    J_MV_l = int(np.ceil((var_y_l / V_es_l) * (n / (np.log(n) ** 2))))
    J_MV_r = int(np.ceil((var_y_r / V_es_r) * (n / (np.log(n) ** 2))))

    J_l = max(1, J_MV_l)
    J_r = max(1, J_MV_r)

    # --------------------------------------------------
    # Construct bins (WITH bin_min / bin_max restored)
    # --------------------------------------------------
    edges_l = np.linspace(x_min, c, J_l + 1)
    edges_r = np.linspace(c, x_max, J_r + 1)

    bin_l = np.searchsorted(edges_l, x_l[:, 0], side="right") - (J_l + 1)
    bin_l[bin_l == 0] = -1

    bin_r = np.searchsorted(edges_r, x_r[:, 0], side="left")
    bin_r[bin_r == J_r] = J_r - 1

    df_l = pd.DataFrame({"bin": bin_l, "x": x_l[:, 0], "y": y_l[:, 0]})
    df_r = pd.DataFrame({"bin": bin_r, "x": x_r[:, 0], "y": y_r[:, 0]})

    def summarize(df, edges, side):
        out = (
            df.groupby("bin")
            .agg(
                mean_x=("x", "mean"),
                mean_y=("y", "mean"),
                N=("y", "size"),
                sd_y=("y", "std"),
            )
            .reset_index()
        )

        out["sd_y"] = out["sd_y"].fillna(0.0)
        out["se"] = out["sd_y"] / np.sqrt(out["N"])

        if side == "left":
            out["bin_min"] = edges[out["bin"]]
            out["bin_max"] = edges[out["bin"] + 1]
        else:
            out["bin_min"] = edges[out["bin"]]
            out["bin_max"] = edges[out["bin"] + 1]

        return out

    gb_l = summarize(df_l, edges_l, "left")
    gb_r = summarize(df_r, edges_r, "right")

    def tcrit(n):
        return -student_t.ppf(0.025, max(n - 1, 1))

    for gb in (gb_l, gb_r):
        gb["ci_lower"] = gb["mean_y"] - gb["se"] * gb["N"].map(tcrit)
        gb["ci_upper"] = gb["mean_y"] + gb["se"] * gb["N"].map(tcrit)

    bins = pd.concat([gb_l.sort_values("bin"), gb_r.sort_values("bin")], ignore_index=True)

    return {
        "binselect": "esmv",
        "kernel": "tri",
        "c": c,
        "p": p,
        "N": {"left": n_l, "right": n_r},
        "h": {"left": h_l, "right": h_r},
        "J": {"left": J_l, "right": J_r},
        "J_IMSE": {"left": J_l, "right": J_r},
        "J_MV": {"left": J_MV_l, "right": J_MV_r},
        "bins": bins,
        "poly": {
            "x_left": x_plot_l,
            "y_left": y_hat_l.flatten(),
            "x_right": x_plot_r,
            "y_right": y_hat_r.flatten(),
        },
        "coefficients": {
            "left": gamma_l.flatten(),
            "right": gamma_r.flatten(),
        },
        "vcov": {"left": V_l, "right": V_r},
        "se_coef": {"left": se_beta_l, "right": se_beta_r},
    }

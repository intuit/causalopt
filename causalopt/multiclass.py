import re
import itertools
from itertools import combinations, product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.linalg import qr
from scipy.special import expit
from scipy.stats import norm

from causalopt.utils import *
from causalopt.estimation import bw_select




# ── Data Generation ──────────────────────────────────────────────────────────

def generate_probabilities(n, K, alpha=None, seed=None):
    """
    Draw class probability vectors from a Dirichlet distribution.

    Parameters
    ----------
    n : int
        Number of observations.
    K : int
        Number of classes.
    alpha : array-like or None
        Dirichlet concentration (length K). Defaults to ones (uniform).
    seed : int or None
        Optional RNG seed.

    Returns
    -------
    ndarray, shape (n, K)
        Rows are probability vectors summing to 1.
    """
    if seed is not None:
        np.random.seed(seed)

    if alpha is None:
        alpha = np.ones(K)

    Prob = np.random.dirichlet(alpha, size=n)
    return Prob


def generate_outcome(Prob, T, alpha_T, beta, sigma=1.0, seed=None):
    """
    Simulate a binary outcome from a latent-index model.

    The latent index is ``alpha_T[T] + Prob @ beta + N(0, sigma)`` and the
    outcome is 1 when the latent index is positive.

    Parameters
    ----------
    Prob : ndarray, shape (n, K)
        Class probabilities.
    T : array-like, shape (n,)
        Assigned class per observation, used to index ``alpha_T``.
    alpha_T : array-like
        Class-specific intercepts (indexed by the values in ``T``).
    beta : array-like, shape (K,)
        Coefficients on the probability vector.
    sigma : float
        Standard deviation of the Gaussian error.
    seed : int or None
        Optional RNG seed.

    Returns
    -------
    ndarray, shape (n,)
        Binary outcomes in {0, 1}.
    """
    if seed is not None:
        np.random.seed(seed)

    eta = np.array([alpha_T[t] for t in T]) + Prob @ beta
    e = np.random.normal(0, sigma, size=len(T))

    latent = eta + e
    Y = (latent > 0).astype(int)

    return Y


def generate_R(Y, T, mu_R, delta_R, sigma_R=1.0, seed=None):
    """
    R | (Y=1, T=t) ~ LogNormal( mu_R / (1+delta_R)^(t-1) , sigma_R )
    R = 0 if Y = 0
    """
    if seed is not None:
        np.random.seed(seed)

    R = np.zeros(len(Y))
    idx = (Y == 1)

    mu_t = np.array([
        mu_R / ((1.0 + delta_R) ** (t - 1))
        for t in T[idx]
    ])

    R[idx] = np.random.lognormal(mean=mu_t, sigma=sigma_R)
    return R


def generate_simulated_data(
    n,
    K,
    tau,
    alpha_dirichlet,
    alpha_T,
    beta,
    sigma=1.0,
    add_R=False,
    mu_R=None,
    delta_R=0.0,
    sigma_R=1.0,
    seed=None
):
    """
    Assemble a simulated multiclass dataset for the RD pipeline.

    Draws class probabilities, applies the ``decision_rule`` to get the
    assigned class ``T``, simulates a binary outcome ``Y``, and optionally a
    revenue-style outcome ``R``.

    Parameters
    ----------
    n, K : int
        Number of observations and classes.
    tau : array-like, shape (K,)
        Decision thresholds passed to ``decision_rule``.
    alpha_dirichlet : array-like or None
        Dirichlet concentration for the probabilities.
    alpha_T, beta, sigma : see ``generate_outcome``.
    add_R : bool
        If True, also simulate ``R`` (requires ``mu_R``).
    mu_R, delta_R, sigma_R : see ``generate_R``.
    seed : int or None
        Optional RNG seed.

    Returns
    -------
    pandas.DataFrame
        Columns ``Prob_1..Prob_K``, ``T``, ``Y`` (and ``R`` if ``add_R``).

    Raises
    ------
    ValueError
        If ``add_R`` is True but ``mu_R`` is None.
    """
    Prob = generate_probabilities(
        n=n,
        K=K,
        alpha=alpha_dirichlet,
        seed=seed
    )

    T = decision_rule(Prob, tau)

    Y = generate_outcome(
        Prob=Prob,
        T=T,
        alpha_T=alpha_T,
        beta=beta,
        sigma=sigma,
        seed=seed
    )

    data = pd.DataFrame(
        Prob,
        columns=[f"Prob_{k+1}" for k in range(K)]
    )
    data["T"] = T
    data["Y"] = Y

    if add_R:
        if mu_R is None:
            raise ValueError("When add_R=True, mu must be provided.")
        data["R"] = generate_R(
            Y=Y,
            T=T,
            mu_R=mu_R,
            delta_R=delta_R,
            sigma_R=sigma_R,
            seed=seed
        )

    return data



# ── Data Preparation ─────────────────────────────────────────────────────────

def rename_to_prob_k(data, cols):
    """
    Rename selected columns to Prob_k in order.

    Parameters
    ----------
    data : DataFrame
    cols : list of str
        Columns to map to Prob_1, Prob_2, ...

    Returns
    -------
    DataFrame, mapping dict
    """
    mapping = {col: f"Prob_{i+1}" for i, col in enumerate(cols)}
    return data.rename(columns=mapping), mapping


def dist_to_bound(data, tau, prob_cols=None):
    """
    Wide-format distances to all (j,k) boundaries.

    Returns
    -------
    DataFrame with columns:
        dist_1_2, dist_1_3, ..., dist_(K-1)_K
    """
    if prob_cols is None:
        prob_cols = [c for c in data.columns if c.startswith("Prob_")]

    Prob = data[prob_cols].values
    scores = Prob - tau

    n, K = scores.shape
    out = data.copy()

    for j, k in combinations(range(K), 2):
        out[f"dist_{j+1}_{k+1}"] = scores[:, j] - scores[:, k]

    return out


def pairwise_selection(data, pair, d=None, runing_var=None):
    """
    Select observations for pairwise (i,j) boundary analysis.

    Parameters
    ----------
    data : DataFrame
        Must contain:
        - T : predicted class (1-based)
        - dist_i_j : signed distance to (i,j) boundary
    pair : list or tuple
        [i, j] class indices (1-based)
    d : float or None
        Optional bandwidth around the boundary
    runing_var : str, default "x"
        Name of running variable if renamed

    Returns
    -------
    DataFrame
    """

    i, j = pair
    col_ij = f"dist_{min(i,j)}_{max(i,j)}"

    if col_ij not in data.columns:
        raise ValueError(f"Column {col_ij} not found. Run dist_to_bound first.")

    out = data.copy()

    # Selection: only classes i or j
    idx = out["T"].isin([i, j])

    # Optional bandwidth restriction
    if d is not None:
        idx &= (out[col_ij].abs() <= d)

    out = out.loc[idx].copy()

    # Optional renaming
    if runing_var is not None:
        out = out.rename(columns={col_ij: runing_var})

    return out


# ── Decision Rule ─────────────────────────────────────────────────────────────

def decision_rule(Prob, tau):
    """
    Decision rule:
    T = argmax_j (Prob_j - tau_j)

    Parameters
    ----------
    Prob : ndarray, shape (n, K)
    tau : ndarray, shape (K,), sum(tau) = 1

    Returns
    -------
    T : ndarray, shape (n,)
        Recommended class in {1,...,K}
    """
    scores = Prob - tau
    T = np.argmax(scores, axis=1) + 1
    return T


# ── Threshold Grid and Tau Conversion ────────────────────────────────────────

def validate_d(d_dict, tol):
    """
    Check cycle (transitivity) constraints:
      d_{j,k} + d_{k,l} = d_{j,l} for all j<k<l

    Parameters
    ----------
    d_dict : dict
        {(j,k): value} with j<k

    tol : float
        Absolute tolerance used to accept near-equality on a discrete grid.
        (Should be tied to grid step size.)
    """
    classes = sorted({i for jk in d_dict.keys() for i in jk})

    for j, k, l in combinations(classes, 3):
        if (j, k) not in d_dict or (k, l) not in d_dict or (j, l) not in d_dict:
            return False

        lhs = d_dict[(j, k)] + d_dict[(k, l)]
        rhs = d_dict[(j, l)]

        if abs(lhs - rhs) > tol:
            return False

    return True


def d_to_tau(d, K, pairs=None, tol=1e-10, probabilities=True):
    """
    Convert a pairwise-difference vector d into tau.

    The tau level is unidentified from the differences alone (only tau
    differences drive the decision), so a convention pins it:
      - probabilities=True: simplex convention tau_1 = (1 + sum_j d_1j) / K,
        which forces sum(tau) = 1, and tau is required to be interior to the
        simplex (tau_k in [0, 1]).
      - probabilities=False: raw-score convention tau_1 = 0, so
        tau_j = -d_1j; no box constraint (support is handled upstream by the
        bandwidth-derived grid bounds).

    Parameters
    ----------
    d : dict or tuple
        - dict: {(j,k): d_jk}
        - tuple: (d_1_2, d_1_3, d_2_3, ...) ordered according to `pairs`

    K : int
        Number of classes

    pairs : list of tuples, optional
        Required if d is a tuple.
        Ordering of pairwise differences, e.g. [(1,2),(1,3),(2,3),...]

    tol : float
        Numerical tolerance

    probabilities : bool
        If True, use the simplex convention/constraint; if False, use the
        raw-score offset convention (tau_1 = 0) with no simplex box.

    Returns
    -------
    tuple
        (tau_1, ..., tau_K)
    """
    # Normalize input to dict
    if isinstance(d, tuple):
        if pairs is None:
            raise ValueError("pairs must be provided when d is a tuple")
        d_dict = dict(zip(pairs, d))
    else:
        d_dict = d

    # Validate transitivity
    if not validate_d(d_dict, tol=tol):
        raise ValueError("Invalid d: transitivity constraints violated")

    # Recover independent differences d_{1,j}
    d1 = np.zeros(K + 1)
    for j in range(2, K + 1):
        if (1, j) not in d_dict:
            raise ValueError(f"Missing d_(1,{j})")
        d1[j] = d_dict[(1, j)]

    # Recover tau. The overall level is unidentified; pin it by convention.
    tau = np.zeros(K + 1)
    if probabilities:
        # Simplex convention: forces sum(tau) = 1
        tau[1] = (1 + d1[2:].sum()) / K
    else:
        # Raw-score offset convention: anchor tau_1 = 0
        tau[1] = 0.0

    for j in range(2, K + 1):
        tau[j] = tau[1] - d1[j]

    if probabilities:
        # Recommended threshold must be interior to the simplex
        if np.any(tau[1:] < -tol) or np.any(tau[1:] > 1 + tol):
            raise ValueError("Tau outside simplex")

    return tuple(tau[1:])


def build_d_grid(bounds, B, grid="rdd"):
    """
    Build a grid of VALID pairwise differences d by gridding only
    the independent differences (1,j), j=2,...,K, and deriving the rest
    by transitivity.

    Parameters
    ----------
    bounds : dict
        {(j,k): d_jk0} baseline pairwise differences (j < k).
        Must contain all (1,j) for j=2,...,K.

    B : int
        Number of grid points per independent difference.

    grid : {"rdd", "full", "observed"}
        - "rdd":  d_{1,j} ∈ [min(0,d_1j0), max(0,d_1j0)]
        - "full": d_{1,j} ∈ [-|d_1j0|, |d_1j0|]
        - "observed": d_{1,j} ∈ [lo, hi] taken directly from bounds[(1,j)],
          which must be an (lo, hi) tuple. Used by the out_of_bandwidth path,
          where the candidate range is the observed span of dist_1_j rather
          than a symmetric bandwidth interval.

    Returns
    -------
    out : list of tuples
        Each tuple is a VALID d-vector ordered by:
        pairs = [(1,2), (1,3), (2,3), ..., (K-1,K)]
    """
    # Infer K
    classes = sorted({i for jk in bounds.keys() for i in jk})
    K = max(classes)

    # Independent pairs (1,j)
    base_pairs = [(1, j) for j in range(2, K + 1)]
    for pair in base_pairs:
        if pair not in bounds:
            raise ValueError(f"Missing bound for pair {pair}")

    # Build grids for independent differences
    grids = []
    for (_, j) in base_pairs:
        d0 = bounds[(1, j)]

        if grid == "rdd":
            lo = min(0.0, d0)
            hi = max(0.0, d0)
        elif grid == "full":
            lo = -abs(d0)
            hi = abs(d0)
        elif grid == "observed":
            lo, hi = d0  # d0 is an (lo, hi) tuple of the observed dist range
        else:
            raise ValueError("grid must be 'rdd', 'full', or 'observed'")

        grids.append(np.linspace(lo, hi, B))

    # Full list of pairs for output ordering
    pairs = [(j, k) for j in range(1, K + 1) for k in range(j + 1, K + 1)]

    out = []

    # Enumerate independent grid and derive full d
    for values in product(*grids):
        d = {}

        # Set independent differences
        for idx, (_, j) in enumerate(base_pairs):
            d[(1, j)] = values[idx]

        # Derive remaining differences by transitivity
        for j in range(2, K + 1):
            for k in range(j + 1, K + 1):
                d[(j, k)] = d[(1, k)] - d[(1, j)]

        # Return as tuple in fixed order
        out.append(tuple(d[p] for p in pairs))

    return out


def frontier_grid_to_tau_K(
    bounds,
    B,
    grid="rdd",
    tol=1e-10,
    probabilities=True,
    decimals=12
):
    """
    Generate all DISTINCT tau-vectors implied by a grid of
    pairwise differences.

    probabilities is passed through to d_to_tau: when True, candidates are
    kept on the simplex (level sum(tau)=1 and box tau_k in [0,1]); when False,
    the raw-score offset convention (tau_1=0) is used and no simplex filtering
    is applied.
    """

    # Collapse bounds if outcomes are provided
    if bounds and isinstance(next(iter(bounds.values())), dict):
        collapsed_bounds = {}

        all_pairs = set(
            jk for outcome_bounds in bounds.values() for jk in outcome_bounds
        )

        for jk in all_pairs:
            vals = [b[jk] for b in bounds.values() if jk in b]

            if len(bounds) == 1:
                # original behavior: [0, v]
                collapsed_bounds[jk] = vals[0]
            else:
                # symmetric interval covering all outcomes
                lo = min(vals + [0.0])
                hi = max(vals + [0.0])
                collapsed_bounds[jk] = max(abs(lo), abs(hi))

        bounds = collapsed_bounds

    # Infer K
    classes = sorted({i for jk in bounds.keys() for i in jk})
    K = max(classes)

    # Build d-grid (unchanged, scalar bounds only)
    d_grid = build_d_grid(bounds, B, grid=grid)

    pairs = [(j, k) for j in range(1, K + 1) for k in range(j + 1, K + 1)]

    tau_set = set()

    for d_tuple in d_grid:
        try:
            tau = d_to_tau(
                d_tuple,
                K=K,
                pairs=pairs,
                tol=tol,
                probabilities=probabilities
            )
        except ValueError:
            continue

        tau_set.add(tuple(np.round(tau, decimals=decimals)))

    return list(tau_set)


# ── Binning and Bandwidth Selection ──────────────────────────────────────────

def union_bw_mask(df, thresh):
    """
    Construct a boolean mask selecting observations for which
    AT LEAST ONE pairwise distance dist_{j}_{k} is close to the
    decision boundary, considering only observations with T=j or T=k,
    and using |dist_{j}_{k}| <= |thresh[j,k]|.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns named dist_j_k and a column 'T'
        indicating the assigned class (0-based).

    thresh : dict
        Dictionary with keys (j, k) and scalar thresholds tau_{j,k}.
        Indices j,k are assumed to be 0-based.

    Returns
    -------
    mask : numpy.ndarray (bool)
        True if observation belongs to the union-of-slabs region.
    """
    n = len(df)
    mask = np.zeros(n, dtype=bool)

    for (j, k) in combinations(
        sorted({i for jk in thresh.keys() for i in jk}), 2
    ):
        if (j, k) not in thresh:
            continue

        col = f"dist_{j}_{k}"
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in DataFrame")

        tau = abs(thresh[(j, k)])

        mask |= (
            (df["T"].values == j) | (df["T"].values == k)
        ) & (
            np.abs(df[col].values) <= tau
        )

    return mask


# ── Kernel Weighting ──────────────────────────────────────────────────────────

def triangular_kernel_multi(df, h=None, kernel="sum"):
    """
    Multidimensional triangular kernel for RD-style weighting,
    computed directly from a DataFrame containing dist_j_k columns.

    Each pairwise margin m contributes a truncated triangular weight
    kt_m = max(0, 1 - |dist_m| / h_m); the ``kernel`` argument controls how
    the per-margin weights are combined into a single observation weight.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns named dist_j_k for j < k.

    h : None, dict, or array-like (optional)
        Bandwidths for each dist_j_k.
        - If None: h_jk = max(dist_j_k) - min(dist_j_k) (full support; no
          truncation binds, so union-style kernels degenerate toward the
          central-point behavior -- pass a real per-pair bandwidth to get
          genuine truncation).
        - If dict: keys (j,k) -> scalar bandwidth
        - If array-like: must align with sorted dist_j_k columns

    kernel : {"sum", "soft_or", "nearest", "prod", "farthest"}
        How the per-margin triangular weights kt_m are combined:
        - "sum" (default): sum_m kt_m. Rewards proximity to any border and
          strictly more for being near more borders. Range [0, M].
        - "soft_or": 1 - prod_m (1 - kt_m). Union-style but saturates at 1.
        - "nearest": max_m kt_m. Weight from the single closest border only.
        - "prod": prod_m kt_m. Product (AND) kernel; high only when near all
          borders. Zeroed if any margin is outside its bandwidth.
        - "farthest": min_m kt_m. Driven by the farthest border (AND-style).
          Accepts the legacy alias "min_dist".

    Returns
    -------
    w : ndarray, shape (n,)
        Kernel weights.
    info : dict
        Metadata with keys:
        - "dist_cols"
        - "pairs"
        - "h"
        - "kernel"
    """
    # Identify dist_j_k columns and sort them
    pattern = re.compile(r"^dist_(\d+)_(\d+)$")
    dist_cols = []
    pairs = []

    for c in df.columns:
        m = pattern.match(c)
        if m:
            j, k = int(m.group(1)), int(m.group(2))
            if j < k:
                dist_cols.append(c)
                pairs.append((j, k))

    if len(dist_cols) == 0:
        raise ValueError("No dist_j_k columns found in df.")

    # Sort consistently by (j,k)
    pairs, dist_cols = zip(*sorted(zip(pairs, dist_cols)))
    dist_cols = list(dist_cols)
    pairs = list(pairs)

    # Build D matrix
    D = df[dist_cols].values.astype(float)
    n, M = D.shape

    # Bandwidth selection
    if h is None:
        # Full-support bandwidths
        h_vec = np.array([
            np.nanmax(D[:, m]) - np.nanmin(D[:, m])
            for m in range(M)
        ])
    elif isinstance(h, dict):
        h_vec = np.array([h[pair] for pair in pairs], dtype=float)
    else:
        h_vec = np.asarray(h, dtype=float)

    if h_vec.shape != (M,):
        raise ValueError("Bandwidth h has incorrect shape.")

    if np.any(h_vec <= 0):
        raise ValueError("All bandwidths must be strictly positive.")

    # Standardized absolute distances
    u = np.abs(D) / h_vec

    # Per-margin truncated triangular weights: kt_m = max(0, 1 - u_m).
    # A margin outside its bandwidth (u_m > 1) contributes exactly 0.
    kt = np.clip(1.0 - u, 0.0, None)

    # Combine per-margin weights into a single observation weight
    if kernel == "sum":
        w = kt.sum(axis=1)

    elif kernel == "soft_or":
        w = 1.0 - np.prod(1.0 - kt, axis=1)

    elif kernel == "nearest":
        w = kt.max(axis=1)

    elif kernel == "prod":
        w = kt.prod(axis=1)

    elif kernel in ("farthest", "min_dist"):
        w = kt.min(axis=1)

    else:
        raise ValueError(
            "kernel must be one of "
            "{'sum', 'soft_or', 'nearest', 'prod', 'farthest'}."
        )

    info = {
        "dist_cols": dist_cols,
        "pairs": pairs,
        "h": h_vec,
        "kernel": kernel
    }

    return w, info


# ── Local Polynomial Regression ───────────────────────────────────────────────

def poly_terms(X, p, tol=1e-5):
    """
    Polynomial design matrix for a local polynomial fit over the supplied
    running coordinates.

    The function:
      • Treats every column of X as a free running coordinate.
      • Includes an explicit intercept (degree 0).
      • Includes all monomials of total degree from 1 to p.
      • Drops any remaining multicollinear columns via QR pivoting.

    For p = 1 with d running coordinates this keeps:
      1) 1             (intercept)
      2) x_k           for k = 1,...,d

    Parameters
    ----------
    X : array-like, shape (n, d)
        Running coordinates (e.g. the K-1 independent signed distances to the
        decision boundaries). No sum-to-1 or baseline assumption is made.
    p : int
        Total polynomial degree (p >= 1).
    tol : float
        Tolerance for detecting multicollinearity (QR pivoting).

    Returns
    -------
    R : ndarray, shape (n, k)
        Polynomial design matrix (intercept included) with full column rank.
    degs : ndarray, shape (k,)
        Total degree of each column in R (0 for the intercept).
    """

    X = np.asarray(X)
    n, d = X.shape

    if p < 1:
        raise ValueError("p must be at least 1.")

    # Intercept (degree 0) plus monomials of degree 1..p over all coordinates
    cols = [np.ones(n)]
    degs = [0]

    for deg in range(1, p + 1):
        for idx in itertools.combinations_with_replacement(range(d), deg):
            cols.append(np.prod(X[:, idx], axis=1))
            degs.append(deg)

    R_full = np.column_stack(cols)
    degs = np.asarray(degs, dtype=int)

    # QR WITH PIVOTING (SciPy)
    Q, Rmat, piv = qr(R_full, mode="economic", pivoting=True)

    keep = piv[np.abs(np.diag(Rmat)) > tol]
    keep = np.sort(keep)

    R = R_full[:, keep]
    degs = degs[keep]

    return R, degs


def rd_fit_multi(
    y,
    X,
    W,
    c=None,
    p=1,
    Z=None,
    nnmatch=3,
    tol=1e-5,
):
    """
    One-sided multivariate local polynomial RD fit with externally supplied
    kernel weights.

    X holds the running coordinates (e.g. the K-1 independent signed distances
    to the decision boundaries); the intercept and polynomial construction are
    handled by poly_terms.
    """

    # Basic setup
    y = np.asarray(y).reshape(-1, 1)
    X = np.asarray(X)
    W = np.asarray(W).reshape(-1, 1)

    if Z is not None:
        Z = np.asarray(Z)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

    n, K = X.shape
    if c is None:
        c = np.zeros(K)
    c = np.asarray(c)

    Xc = X - c

    # Polynomial bases (simplex, reduced)
    q = p + 1

    R_p, deg_p = poly_terms(Xc, p, tol=tol)
    R_q, deg_q = poly_terms(Xc, q, tol=tol)

    # Outcomes
    D = y if Z is None else np.column_stack((y, Z))

    # Weighted Gram inverses
    invG_p = qrXXinv(np.sqrt(W) * R_p)
    invG_q = qrXXinv(np.sqrt(W) * R_q)

    # Conventional estimate
    beta_p = invG_p @ crossprod(R_p * W, D)

    # Bias correction (multivariate CCT, block form)
    idx_p1 = np.where(deg_q == (p + 1))[0]

    if idx_p1.size == 0:
        # No higher-order terms survived → no bias correction
        Q_q = R_p * W
    else:
        R_p1 = R_q[:, idx_p1]                     # (n, k_{p+1})
        L = crossprod(R_p * W, R_p1)              # (k_p, k_{p+1})
        P_q = invG_q @ (R_q * W).T                # (k_q, n)
        P_q1 = P_q[idx_p1, :]                     # (k_{p+1}, n)
        Q_q = (R_p * W) - (L @ P_q1).T             # (n, k_p)

    beta_bc = invG_p @ crossprod(Q_q, D)

    # Covariate adjustment (FWL)
    if Z is None:
        s = np.array([1.0])
        beta_p_eff = beta_p
        beta_bc_eff = beta_bc
        Z_for_res = None
    else:
        dZ = Z.shape[1]
        colsZ = np.arange(1, 1 + dZ)

        U = crossprod(R_p * W, D)
        UiGU = crossprod(U[:, colsZ], invG_p @ U)

        ZWZ = UiGU[:, colsZ]
        ZWY = UiGU[:, :1]

        gamma = np.linalg.pinv(ZWZ) @ ZWY
        s = np.concatenate(([1.0], -gamma[:, 0]))

        beta_p_eff = beta_p @ s.reshape(-1, 1)
        beta_bc_eff = beta_bc @ s.reshape(-1, 1)
        Z_for_res = Z

    # Robust variance (NN residuals)
    u = np.linalg.norm(Xc, axis=1)

    dups, dupsid = make_dups(u)

    res = nn_residuals(
        X = u,
        y = y.flatten(),
        Z = Z_for_res,
        matches=3,
        dups=dups,
        dupsid=dupsid,
    )

    res_eff = res @ s.reshape(-1, 1)

    V_p  = invG_p @ sandwich_se(R_p * W, res_eff) @ invG_p
    V_bc = invG_p @ sandwich_se(Q_q,   res_eff) @ invG_p

    return {
        "coef_poly": {
            "conventional": beta_p_eff,
            "bias_corrected": beta_bc_eff,
        },
        "V_poly": {
            "conventional": V_p,
            "bias_corrected": V_bc,
        },
    }


def sim_poly_multi(X, b, V, p, nsim=10000, alpha=0.05, seed=None, tol=1e-5):
    """
    Simulation-based confidence / prediction band for the multivariate
    polynomial fit produced by rd_fit_multi.

    This is the multivariate analogue of sim_poly_ic, but instead of building a
    Vandermonde matrix from a 1-D x, it:
      1) builds the multivariate polynomial design matrix R_new using poly_terms,
      2) computes point predictions y_hat = R_new @ b,
      3) draws beta ~ N(b, V) and computes simulated predictions R_new @ beta,
         then returns the mean and pointwise quantiles.

    Parameters
    ----------
    X : array-like, shape (m, d)
        New points at which to predict, given as the same running coordinates
        used to fit b and V (the K-1 independent signed distances).

    b : array-like, shape (k,) or (k,1)
        Coefficient vector from rd_fit_multi (e.g., mm['coef_poly']['bias_corrected']).

    V : array-like, shape (k, k)
        Covariance matrix for b from rd_fit_multi (e.g., mm['V_poly']['bias_corrected']).

    p : int
        Polynomial degree used in rd_fit_multi / poly_terms. Must match the degree
        used to estimate b and V.

    nsim : int, default = 10000
        Number of Monte Carlo draws.

    alpha : float, default = 0.05
        Tail probability for pointwise intervals (e.g., 0.05 gives 95% intervals).

    seed : int or None, default = None
        Random seed for reproducibility.

    tol : float, default = 1e-5
        Tolerance passed to poly_terms for its internal multicollinearity dropping.
        Must match what was used when fitting, otherwise the basis dimension may differ.

    Returns
    -------
    y_hat : ndarray, shape (m,)
        Plug-in point predictions R_new @ b.

    mean_fit : ndarray, shape (m,)
        Monte Carlo mean of simulated predictions.

    lower : ndarray, shape (m,)
        Pointwise lower bound (alpha/2 quantile) of simulated predictions.

    upper : ndarray, shape (m,)
        Pointwise upper bound (1 - alpha/2 quantile) of simulated predictions.
    """
    X = np.asarray(X)
    b = np.asarray(b).ravel()
    V = np.asarray(V)

    # Polynomial transformation
    R_new, _ = poly_terms(X, p, tol=tol)  # (m, k)

    if R_new.shape[1] != b.shape[0]:
        raise ValueError(
            f"Basis dimension mismatch: R_new has {R_new.shape[1]} columns "
            f"but b has length {b.shape[0]}. Ensure p and tol match the fit."
        )
    if V.shape != (b.shape[0], b.shape[0]):
        raise ValueError(
            f"V must be shape ({b.shape[0]}, {b.shape[0]}), got {V.shape}."
        )

    # Point prediction
    y_hat = R_new @ b  # (m,)

    # Simulation-based bands
    rng = default_rng(seed)
    beta_draws = rng.multivariate_normal(mean=b, cov=V, size=nsim)  # (nsim, k)
    yhat_sim = R_new @ beta_draws.T  # (m, nsim)

    mean_fit = yhat_sim.mean(axis=1)
    lower = np.quantile(yhat_sim, alpha / 2, axis=1)
    upper = np.quantile(yhat_sim, 1 - alpha / 2, axis=1)

    return y_hat, mean_fit, lower, upper


# ── Estimation Pipeline ───────────────────────────────────────────────────────

def assign_predicted_outcomes(
    df,
    outcome_cols,
    T_col="T",
    T_alt_col="T_alt",
):
    df = df.copy()

    if isinstance(outcome_cols, str):
        outcome_cols = [outcome_cols]

    for outcome in outcome_cols:

        yhat_prefix = f"{outcome}_hat_"

        # Identify K from available Y_hat columns
        yhat_cols = sorted(
            [c for c in df.columns if c.startswith(yhat_prefix)],
            key=lambda x: int(x.replace(yhat_prefix, ""))
        )

        K = len(yhat_cols)
        if K == 0:
            raise ValueError(f"No predicted outcome columns found for {outcome}.")

        # Build prediction matrix (n x K)
        Y_hat = df[yhat_cols].values

        # Convert assignments to 0-based indices
        T_idx = df[T_col].values.astype(int) - 1
        T_alt_idx = df[T_alt_col].values.astype(int) - 1

        # Safety checks
        if np.any(T_idx < 0) or np.any(T_idx >= K):
            raise ValueError("T contains invalid class indices.")
        if np.any(T_alt_idx < 0) or np.any(T_alt_idx >= K):
            raise ValueError("T_alt contains invalid class indices.")

        rows = np.arange(len(df))

        df[f"{outcome}_hat_current"] = Y_hat[rows, T_idx]
        df[f"{outcome}_hat_alt"] = Y_hat[rows, T_alt_idx]

    return df


def bdselect(df, outcome_cols, tau):
    """
    Select MSE-optimal bandwidths for every boundary, per outcome.

    For each outcome and each class pair ``(j, k)``, restrict to observations
    assigned to ``j`` or ``k`` and run ``bw_select`` on the signed distance to
    that boundary, returning the left MSE-optimal bandwidth ``h_jk``.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``Prob_*`` columns, ``T``, and the ``dist_j_k`` columns.
    outcome_cols : list of str
        Outcome column names.
    tau : array-like
        Decision thresholds (unused directly here; kept for API symmetry).

    Returns
    -------
    dict
        ``{outcome: {(j, k): h_jk}}`` with 1-based pair keys.
    """
    # Determine number of alternatives
    prob_cols = [c for c in df.columns if c.startswith("Prob_")]
    K = len(prob_cols)

    # Prepare data to apply RDD
    pair_thresh = {}

    for outcome in outcome_cols:
        pair_thresh[outcome] = {}

        # Select MSE-optimal bandwidth pairwise
        for j, k in combinations(range(K), 2):
            jk = (j + 1, k + 1)

            dfreg = pairwise_selection(df, jk, d=None, runing_var="x")
            cov_cols = [c for c in dfreg.columns if c.startswith("dist_")]
            Z = dfreg[cov_cols].to_numpy() if cov_cols else None

            bws = bw_select(
                y=dfreg[outcome].to_numpy(),
                x=dfreg["x"].to_numpy(),
                c=0,
                covariates=Z,
            )

            pair_thresh[outcome][jk] = bws["bandwidths"]["h_left"]

    return pair_thresh


def est_multi(df, outcome_cols, pair_thresh, K, kernel="sum", weight_col=None):
    """
    Fit per-class local polynomial RD models near the decision boundaries.

    For each outcome, select observations within the union of pairwise
    bandwidth slabs, weight them with ``triangular_kernel_multi`` (using the
    per-pair bandwidths so distant borders truncate to 0), and fit a weighted
    local polynomial for each class ``j``.

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared data with ``dist_j_k`` columns, ``T`` and the outcomes.
    outcome_cols : list of str
        Outcome column names.
    pair_thresh : dict or None
        ``{outcome: {(j, k): h_jk}}`` bandwidths from ``bdselect``. If None
        (binned / no-bandwidth mode), the union-of-slabs selection is skipped
        (all rows kept) and ``triangular_kernel_multi`` uses full-support
        weights (``h=None``, no truncation).
    K : int
        Number of classes.
    kernel : str
        Kernel combine rule passed to ``triangular_kernel_multi``.
    weight_col : str or None, default None
        Optional per-row frequency-weight column (bin counts). When present the
        kernel weight is multiplied by it, so the fit is WLS on binned data.

    Returns
    -------
    dict
        ``{outcome: {j: rd_fit_multi(...) result}}``.
    """
    estplan = {}

    for outcome in outcome_cols:
        estplan[outcome] = {}

        df_aux = df.copy()

        if pair_thresh is None:
            # Binned / no-bandwidth: keep all rows, full-support kernel.
            h = None
        else:
            # Select observations near the threshold (union of slabs).
            mask = union_bw_mask(df_aux, pair_thresh[outcome])
            df_aux['select'] = mask
            df_aux = df_aux[(df_aux['select']==True)]
            h = pair_thresh[outcome]

        # Weight with the per-pair MSE-optimal bandwidths so that distant
        # borders truncate to 0 (real union semantics), then combine margins
        # according to the requested kernel. With h=None the kernel is
        # full-support (no truncation).
        w, info = triangular_kernel_multi(df_aux, h=h, kernel=kernel)
        df_aux["w"] = w

        # Run local polynomial regressions for each category
        for j in range(1, K + 1):

            # Restrict to observations assigned to class j
            df_reg = df_aux[df_aux["T"] == j].copy()

            # Outcome
            y = df_reg[outcome].values

            # Running variables: the K-1 independent signed distances
            # dist_1_2..dist_1_K (already centered at the boundary). This is a
            # full-rank basis; the intercept is added inside poly_terms.
            indep_cols = [f"dist_1_{k}" for k in range(2, K + 1)]
            X = df_reg[indep_cols].values

            # Kernel weights (triangular, product or min_dist), times per-bin
            # counts when fitting on binned data.
            W = df_reg["w"].values
            if weight_col is not None and weight_col in df_reg.columns:
                W = W * df_reg[weight_col].values

            # Local linear RD via WLS
            mm = rd_fit_multi(y=y, X=X, W=W)

            estplan[outcome][j] = mm

    return estplan


def pred_multi(df, outcome_cols, estimates, K):
    """
    Predict each class's fitted surface for every observation.

    Using the fits from ``est_multi``, evaluate the (bias-corrected) local
    polynomial for every class ``j`` at all observations, producing the
    counterfactual columns ``{outcome}_hat_j``.

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared data with the ``dist_1_k`` running coordinates.
    outcome_cols : list of str
        Outcome column names.
    estimates : dict
        Output of ``est_multi``.
    K : int
        Number of classes.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with added ``{outcome}_hat_j`` columns for j = 1..K.
    """
    dfpred = df.copy()

    # Same full-rank running-variable basis used in est_multi: the K-1
    # independent signed distances dist_1_2..dist_1_K (intercept handled inside
    # poly_terms). Using a fixed set keeps the fit and prediction bases aligned.
    indep_cols = [f"dist_1_{k}" for k in range(2, K + 1)]
    X = dfpred[indep_cols].values

    for outcome in outcome_cols:
        est = estimates[outcome]

        for j in range(1, K + 1):
            b = est[j]['coef_poly']['bias_corrected']
            V = est[j]['V_poly']['bias_corrected']
            y_hat, mean_fit, lower, upper = sim_poly_multi(X=X, b=b, V=V, p=1, nsim=10000, alpha=0.05, seed=None, tol=1e-5)
            dfpred[f"{outcome}_hat_{j}"] = y_hat

    return dfpred


def fitted_curves(df, outcome_cols, estimates, pair_thresh, K,
                  n_grid=100, out_of_bandwidth=False, nsim=10000,
                  alpha=0.05, seed=None):
    """
    Grid-evaluated fitted RD curves per outcome, class, and boundary axis.

    This is the multiclass analogue of the binary ``rd_plot_objects``: it turns
    the per-class local-polynomial fits from ``est_multi`` into plottable
    curves. For each outcome and each independent boundary axis ``dist_1_j``
    (j = 2..K), the axis is swept over a grid while the other distance axes are
    held at 0 (i.e. on the boundary), and every class ``c`` fitted surface is
    evaluated along that slice together with a simulation-based confidence band
    (reusing ``sim_poly_multi`` exactly as ``pred_multi`` does).

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared data with the ``dist_1_k`` running coordinates (``dfprep``).
        Only used to derive the observed grid range when no bandwidth is
        available (binned / no-bandwidth mode) or when ``out_of_bandwidth``.
    outcome_cols : list of str
        Outcome column names.
    estimates : dict
        Output of ``est_multi`` (``{outcome: {class: rd_fit_multi(...)}}``).
    pair_thresh : dict or None
        Per-pair bandwidths from ``bdselect`` (``{outcome: {(j, k): h_jk}}``).
        When None (binned / no-bandwidth mode) the grid range falls back to the
        observed support of each ``dist_1_j``.
    K : int
        Number of classes.
    n_grid : int, default 100
        Number of grid points per curve.
    out_of_bandwidth : bool, default False
        If True, use the observed support of each ``dist_1_j`` as the grid range
        instead of the per-pair bandwidth (matching ``find_thresh``).
    nsim, alpha, seed
        Forwarded to ``sim_poly_multi`` for the confidence band.

    Returns
    -------
    dict
        ``{outcome: {(1, j): DataFrame}}`` where each DataFrame has columns
        ``x`` (grid over ``dist_1_j``) and, per class ``c``, ``y_hat_{c}``,
        ``lower_{c}`` and ``upper_{c}``.
    """
    curves = {}

    for outcome in outcome_cols:
        curves[outcome] = {}
        est = estimates[outcome]

        for j in range(2, K + 1):
            axis_idx = j - 2  # position of dist_1_j within the K-1 basis
            dist_col = f"dist_1_{j}"

            # Grid range: symmetric bandwidth window when available, else the
            # observed support of the axis (binned / no-bandwidth / OOB).
            h = None
            if pair_thresh is not None and not out_of_bandwidth:
                h = pair_thresh[outcome].get((1, j))
            if h is None:
                lo, hi = float(df[dist_col].min()), float(df[dist_col].max())
            else:
                lo, hi = -float(h), float(h)

            x_grid = np.linspace(lo, hi, n_grid)

            # Hold every other axis at 0 (on the boundary) and sweep dist_1_j.
            X = np.zeros((n_grid, K - 1))
            X[:, axis_idx] = x_grid

            out = pd.DataFrame({"x": x_grid})
            for c in range(1, K + 1):
                b = est[c]["coef_poly"]["bias_corrected"]
                V = est[c]["V_poly"]["bias_corrected"]
                # tol=-1.0 keeps every basis column: the held-at-0 axes are
                # zero-norm columns that the default QR-pivoting tol would drop,
                # which would break the dimension match against the fitted b.
                y_hat, _mean_fit, lower, upper = sim_poly_multi(
                    X=X, b=b, V=V, p=1, nsim=nsim, alpha=alpha, seed=seed,
                    tol=-1.0,
                )
                out[f"y_hat_{c}"] = y_hat
                out[f"lower_{c}"] = lower
                out[f"upper_{c}"] = upper

            curves[outcome][(1, j)] = out

    return curves


def multiclass_ate(df, outcome_cols, estimates, pair_thresh, K,
                   kind="both", alpha=0.05, out_of_bandwidth=False,
                   weight_col=None):
    """
    Average treatment effects at the class boundaries (the discontinuities).

    In the binary case the ATE is a single scalar: the outcome jump at the one
    cutoff. In the multiclass case each pairwise class boundary ``(j, k)`` is
    its own discontinuity, so this reports one effect per ordered pair
    ``j < k``, per outcome, using two evaluation rules:

    - **junction** - the jump at the K-way meeting point (all independent
      distances 0, which lies on every boundary at once). This is the intercept
      difference ``b_k[0] - b_j[0]`` of the two per-class fits, the direct
      analogue of the binary intercept jump.
    - **facet** - the density-weighted average of ``m_k(d) - m_j(d)`` over the
      observations straddling boundary ``(j, k)`` (within the pairwise bandwidth
      slab, or all j/k rows in binned / no-bandwidth / out-of-bandwidth mode).

    Both use the bias-corrected coefficients and covariances. The per-class fits
    are run on disjoint decision regions (``T == j``) so they are independent
    and the variance of a pairwise difference is the sum of the two variances.
    Because both estimators are linear in the coefficients, the confidence
    intervals are closed-form (no simulation).

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared data (``dfprep``) with the ``dist_j_k`` boundary distances,
        ``dist_1_*`` running coordinates, and the current assignment ``T``.
    outcome_cols : list of str
        Outcome column names.
    estimates : dict
        Output of ``est_multi`` (``{outcome: {class: rd_fit_multi(...)}}``).
    pair_thresh : dict or None
        Per-pair bandwidths from ``bdselect``. When None (binned /
        no-bandwidth), the facet slab falls back to all rows in ``{j, k}``.
    K : int
        Number of classes.
    kind : {"both", "junction", "facet"}, default "both"
        Which estimator(s) to return.
    alpha : float, default 0.05
        Tail probability for the (two-sided) confidence intervals.
    out_of_bandwidth : bool, default False
        If True, the facet slab uses all ``{j, k}`` rows instead of the
        bandwidth window (matching ``find_thresh``).
    weight_col : str or None, default None
        Optional per-row frequency-weight column (bin counts) used for the
        facet averaging.

    Returns
    -------
    dict
        ``{outcome: {...}}`` where each value has the requested tables under
        keys "junction" and/or "facet". Each table is a DataFrame with columns
        ``class_j, class_k, estimate, se, ci_l, ci_r`` (facet also ``n``); the
        reported effect is class ``k`` relative to class ``j``.
    """
    if kind not in ("both", "junction", "facet"):
        raise ValueError("kind must be 'both', 'junction', or 'facet'.")

    z = norm.ppf(1 - alpha / 2)
    want_junction = kind in ("both", "junction")
    want_facet = kind in ("both", "facet")

    # Same p=1 running-variable basis used in est_multi / pred_multi.
    indep_cols = [f"dist_1_{k}" for k in range(2, K + 1)]

    ate = {}
    for outcome in outcome_cols:
        est = estimates[outcome]
        junction_rows = []
        facet_rows = []

        for j in range(1, K + 1):
            for k in range(j + 1, K + 1):
                b_j = np.asarray(est[j]["coef_poly"]["bias_corrected"]).ravel()
                b_k = np.asarray(est[k]["coef_poly"]["bias_corrected"]).ravel()
                V_j = np.asarray(est[j]["V_poly"]["bias_corrected"])
                V_k = np.asarray(est[k]["V_poly"]["bias_corrected"])

                if want_junction:
                    est_jk = float(b_k[0] - b_j[0])
                    se_jk = float(np.sqrt(V_k[0, 0] + V_j[0, 0]))
                    junction_rows.append({
                        "class_j": j,
                        "class_k": k,
                        "estimate": est_jk,
                        "se": se_jk,
                        "ci_l": est_jk - z * se_jk,
                        "ci_r": est_jk + z * se_jk,
                    })

                if want_facet:
                    # Observations straddling the (j, k) boundary.
                    in_pair = df["T"].isin([j, k])
                    h = None
                    if pair_thresh is not None and not out_of_bandwidth:
                        h = pair_thresh[outcome].get((j, k))
                    if h is not None:
                        mask = in_pair & (df[f"dist_{j}_{k}"].abs() <= h)
                    else:
                        mask = in_pair

                    sub = df[mask]
                    n_slab = int(len(sub))

                    if n_slab == 0:
                        est_f = np.nan
                        se_f = np.nan
                    else:
                        X = sub[indep_cols].to_numpy()
                        R, _ = poly_terms(X, 1)
                        if R.shape[1] != b_j.shape[0]:
                            raise ValueError(
                                f"ATE basis dimension mismatch for outcome "
                                f"'{outcome}', pair ({j}, {k}): design has "
                                f"{R.shape[1]} columns but the fit has "
                                f"{b_j.shape[0]} coefficients (expected a "
                                f"full-rank p=1 basis)."
                            )
                        if weight_col is not None and weight_col in sub.columns:
                            wts = sub[weight_col].to_numpy(dtype=float)
                            rbar = np.average(R, axis=0, weights=wts)
                        else:
                            rbar = R.mean(axis=0)

                        est_f = float(rbar @ (b_k - b_j))
                        var_f = float(rbar @ V_k @ rbar + rbar @ V_j @ rbar)
                        se_f = float(np.sqrt(var_f))

                    facet_rows.append({
                        "class_j": j,
                        "class_k": k,
                        "estimate": est_f,
                        "se": se_f,
                        "ci_l": est_f - z * se_f,
                        "ci_r": est_f + z * se_f,
                        "n": n_slab,
                    })

        out = {}
        if want_junction:
            out["junction"] = pd.DataFrame(
                junction_rows,
                columns=["class_j", "class_k", "estimate", "se", "ci_l", "ci_r"],
            )
        if want_facet:
            out["facet"] = pd.DataFrame(
                facet_rows,
                columns=["class_j", "class_k", "estimate", "se", "ci_l", "ci_r", "n"],
            )
        ate[outcome] = out

    return ate


def find_thresh(df, outcome_cols, pair_thresh, B=10_000, probabilities=True,
                out_of_bandwidth=False, weight_col=None):
    """B is the total search budget; per-axis grid resolution is
    floor(B**(1/(K-1))), so the candidate-tau grid stays ~B regardless of K.

    probabilities is forwarded to frontier_grid_to_tau_K: when False the
    candidate taus use the raw-score offset convention and are not filtered to
    the simplex (so the grid keeps the full grids**(K-1) candidates).

    out_of_bandwidth controls the candidate-tau range. When False (default) the
    range is the per-pair bandwidth pair_thresh (h_jk), so candidates stay
    within the estimation window. When True the range is the observed span of
    each dist_1_j column in df, so the frontier is reported over the whole
    provided support (the fit in est_multi/pred_multi is unchanged).

    weight_col, when set and present in df, makes the per-candidate gain
    aggregates (mean_/total_{outcome}) frequency-weighted by that column (bin
    counts), so the frontier is correct when fitting on binned data."""

    rows = []

    prob_cols = [c for c in df.columns if c.startswith("Prob_")]
    Probs = df[prob_cols].values
    K = len(prob_cols)

    if weight_col is not None and weight_col in df.columns:
        wts = df[weight_col].to_numpy(dtype=float)
    else:
        wts = None

    grids = max(2, int(B ** (1.0 / (K - 1)))) if K > 1 else 1

    if out_of_bandwidth:
        # Candidate range = observed span of each independent dist_1_j.
        obs_bounds = {}
        for j in range(2, K + 1):
            vals = df[f"dist_1_{j}"].to_numpy()
            obs_bounds[(1, j)] = (float(np.nanmin(vals)), float(np.nanmax(vals)))
        grid_tau = frontier_grid_to_tau_K(
            obs_bounds, grids, grid="observed", probabilities=probabilities
        )
    else:
        grid_tau = frontier_grid_to_tau_K(
            pair_thresh, grids, grid="full", probabilities=probabilities
        )

    for cc, tau in enumerate(grid_tau):

        df_aux = df.copy()

        TT = decision_rule(Probs, tau)
        df_aux["T_alt"] = TT

        df_aux = assign_predicted_outcomes(df_aux, outcome_cols)

        # Build one row
        row = {}

        # thresholds: tau_1, ..., tau_K
        for k, val in enumerate(tau, start=1):
            row[f"tau_{k}"] = val

        # outcomes (count-weighted when fitting on binned data)
        for outcome in outcome_cols:
            diff = (
                df_aux[f"{outcome}_hat_alt"]
                - df_aux[f"{outcome}_hat_current"]
            ).to_numpy()
            if wts is not None:
                row[f"mean_{outcome}"] = float(np.average(diff, weights=wts))
                row[f"total_{outcome}"] = float(np.sum(diff * wts))
            else:
                row[f"mean_{outcome}"] = float(diff.mean())
                row[f"total_{outcome}"] = float(diff.sum())

        rows.append(row)

    return pd.DataFrame(rows)


def select_optimum(frontier, outcome_cols, by="mean"):
    """
    Pick the welfare-optimal tau per outcome from a frontier table.

    Packages the "best row" step that previously lived in the example notebook:
    for each outcome, sort the frontier by that outcome's gain in descending
    order and take the top row. Regime-agnostic - it only reads ``tau_*`` and
    ``mean_/total_`` columns, so it works for both ``probabilities=True``
    (simplex taus) and ``probabilities=False`` (offset taus, ``tau_1=0``).

    Parameters
    ----------
    frontier : pandas.DataFrame
        Output of ``find_thresh``: columns ``tau_1..K`` plus ``mean_{o}`` and
        ``total_{o}`` for each outcome.
    outcome_cols : list of str
        Outcome column names to select an optimum for.
    by : {"mean", "total"}, default "mean"
        Which gain column to maximise (``mean_{o}`` per-observation average gain
        or ``total_{o}`` summed gain). The other is the tie-break.

    Returns
    -------
    dict
        ``{outcome: {"tau": (...), "mean": float, "total": float,
        "outcomes_at_tau": {o2: {"mean": ..., "total": ...}}}}``.
    """
    if by not in ("mean", "total"):
        raise ValueError("by must be 'mean' or 'total'.")

    tau_cols = sorted(
        [c for c in frontier.columns if c.startswith("tau_")],
        key=lambda c: int(c.split("_")[1]),
    )
    tie = "total" if by == "mean" else "mean"

    optimum = {}
    for o in outcome_cols:
        best = frontier.sort_values(
            [f"{by}_{o}", f"{tie}_{o}"], ascending=False
        ).iloc[0]
        optimum[o] = {
            "tau": tuple(float(best[c]) for c in tau_cols),
            "mean": float(best[f"mean_{o}"]),
            "total": float(best[f"total_{o}"]),
            "outcomes_at_tau": {
                o2: {
                    "mean": float(best[f"mean_{o2}"]),
                    "total": float(best[f"total_{o2}"]),
                }
                for o2 in outcome_cols
            },
        }

    return optimum


def get_thresholds(data, outcome_cols, probability_cols, tau=None, B=10_000,
                   kernel="sum", probabilities=True, out_of_bandwidth=False,
                   bin=False, binned_data=False, bin_spec=None, weight_col="n"):
    """B is the total search budget for the threshold grid (see find_thresh).

    kernel controls how the per-boundary triangular weights are combined in
    the local polynomial fits (passed through to est_multi ->
    triangular_kernel_multi). One of:
      - "sum" (default): reward proximity to any border, more for more borders.
      - "soft_or": union-style, saturating at 1.
      - "nearest": weight from the single closest border only.
      - "prod": product (AND) kernel; high only when near all borders.
      - "farthest": AND-style, driven by the farthest border ("min_dist" alias).

    probabilities selects the input regime:
      - True (default): the input columns are class probabilities on the
        simplex (each row sums to 1, values in [0, 1]). This is validated
        fail-fast; reported tau lie on the simplex (sum(tau)=1, tau_k in [0,1]).
      - False: the input columns are arbitrary real-valued scores (case 1:
        positive/unbounded values; case 2: signed net values). No simplex
        constraint is imposed and reported tau are offsets anchored at tau_1=0.

    tau is the current threshold vector. If None it defaults to the plain
    argmax baseline: uniform 1/K when probabilities=True, zeros when
    probabilities=False. Internally the columns are held as Prob_* regardless
    of regime; when probabilities=False they carry raw scores.

    out_of_bandwidth widens the candidate-tau frontier to the full observed
    support of each dist_1_j instead of the per-pair bandwidth. The pairwise
    fits (est_multi/pred_multi) are unchanged; only the reported frontier range
    grows. Beyond the bandwidth the counterfactual predictions extrapolate.

    bin / binned_data / bin_spec / weight_col control the binning path for
    scalability. bin=True aggregates the raw scores into centroids (grouped by
    the current decision region so no cell straddles a boundary) before fitting;
    binned_data=True treats the ingested frame as already binned with per-bin
    counts in weight_col. In either binned mode, bandwidth selection (bdselect)
    is SKIPPED, the fit uses full-support kernels times bin counts over all
    bins, and the frontier is reported over the observed support (out_of_bandwidth
    is effectively forced True). At most one of bin / binned_data may be set.

    Returns
    -------
    dict with keys:
        "frontier" : pandas.DataFrame
            The candidate-tau frontier (columns tau_1..K, mean_/total_{outcome}),
            i.e. the object this function returned historically.
        "optimum" : dict
            Per-outcome welfare-optimal tau from select_optimum.
        "current" : dict
            Baseline at the supplied tau (gain 0 reference), with keys "tau" and
            "outcomes_at_tau".
        "details" : dict
            The intermediates computed along the way, for inspection/plotting:
                "data_descriptives" - describe() of the working (post-binning) frame.
                "prepared"          - the prepared frame (Prob_* columns, the
                                       dist_j_k boundary distances, and the current
                                       assignment column T).
                "bandwidths"        - the bdselect pair_thresh
                                       ({outcome: {(j, k): h_jk}}); None in
                                       binned / no-bandwidth mode.
                "estimates"         - the est_multi per-class RD fits.
                "predictions"       - the pred_multi frame with the counterfactual
                                       surfaces {outcome}_hat_j.
                "curves"            - fitted_curves output: grid-evaluated fitted
                                       curves with confidence bands per outcome,
                                       class, and boundary axis.
                "ate"               - multiclass_ate output: the treatment effect
                                       at each pairwise class boundary, per
                                       outcome, as bias-corrected estimates with
                                       confidence intervals. Two tables per
                                       outcome: "junction" (jump at the K-way
                                       meeting point) and "facet" (density-
                                       weighted average over units straddling the
                                       boundary).
    """

    df = data.copy()
    K = len(probability_cols)  # Number of classes

    # Default current thresholds: plain argmax baseline
    if tau is None:
        tau = np.full(K, 1.0 / K) if probabilities else np.zeros(K)
    tau = np.asarray(tau, dtype=float)

    # ------------------------------------------------------------------
    # Binning (optional). In any binned mode there is no bandwidth: bdselect is
    # skipped and the frontier is reported over the observed support.
    # ------------------------------------------------------------------
    if bin and binned_data:
        raise ValueError("Set at most one of bin / binned_data.")

    binned = bin or binned_data
    if bin:
        # Group by the current decision region so cells respect the boundaries
        # (regions are convex, so a cell centroid stays in its region).
        T_raw = decision_rule(df[probability_cols].values, tau)
        df, _w = resolve_binning(
            df, list(probability_cols), list(outcome_cols),
            cutoff=None, group=T_raw, bin=True, binned_data=False,
            bin_spec=bin_spec, weight_col=weight_col,
        )
        wname = "n"
    elif binned_data:
        if weight_col not in df.columns:
            raise ValueError(
                f"binned_data=True but weight column '{weight_col}' is not in "
                "the ingested frame."
            )
        wname = weight_col
    else:
        wname = None

    # Rename the columns
    dfprep, mapping = rename_to_prob_k(df, probability_cols)
    prob_cols = [f"Prob_{i+1}" for i in range(K)]

    # Fail-fast: if the caller claims probabilities, the inputs must actually
    # lie on the simplex; otherwise the simplex box in d_to_tau would silently
    # empty the candidate grid.
    if probabilities:
        vals = dfprep[prob_cols].values
        row_sums = vals.sum(axis=1)
        if (not np.allclose(row_sums, 1.0, atol=1e-6)) or \
           np.any(vals < -1e-6) or np.any(vals > 1 + 1e-6):
            raise ValueError(
                "probabilities=True but the input columns are not on the "
                "simplex (each row must sum to 1 and lie in [0, 1]). Check the "
                "columns or set probabilities=False to treat them as scores."
            )

    # Transform probabilities and thresholds to distance-to-the-boundary, so we can apply RDD
    dfprep = dist_to_bound(dfprep, tau)

    # Current assignment under the supplied thresholds (baseline for gains).
    dfprep["T"] = decision_rule(dfprep[prob_cols].values, tau)

    if binned:
        # No bandwidth on binned data: skip bdselect, fit full-support kernels
        # times bin counts over all bins, and report over the observed support.
        tt = None
        oob = True
    else:
        # Select bandwidths using pairwise comparisons
        tt = bdselect(df=dfprep, outcome_cols=outcome_cols, tau=tau)
        oob = out_of_bandwidth

    esthh = est_multi(
        df=dfprep, outcome_cols=outcome_cols, pair_thresh=tt, K=K, kernel=kernel,
        weight_col=wname,
    )

    pred_df = pred_multi(df=dfprep, outcome_cols=outcome_cols, estimates=esthh, K=K)

    frontier = find_thresh(
        df=pred_df, outcome_cols=outcome_cols, pair_thresh=tt, B=B,
        probabilities=probabilities, out_of_bandwidth=oob, weight_col=wname,
    )

    optimum = select_optimum(frontier, outcome_cols)
    current = {
        "tau": tuple(float(t) for t in tau),
        "outcomes_at_tau": {o: {"mean": 0.0, "total": 0.0} for o in outcome_cols},
    }

    curves = fitted_curves(
        dfprep, outcome_cols, esthh, tt, K, out_of_bandwidth=oob,
    )

    ate = multiclass_ate(
        dfprep, outcome_cols, esthh, tt, K,
        out_of_bandwidth=oob, weight_col=wname,
    )

    details = {
        "data_descriptives": pd.DataFrame(df.describe()),
        "prepared": dfprep,
        "bandwidths": tt,
        "estimates": esthh,
        "predictions": pred_df,
        "curves": curves,
        "ate": ate,
    }

    return {
        "frontier": frontier,
        "optimum": optimum,
        "current": current,
        "details": details,
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_simplex(data, tau, T="T", step=0.005, alpha_region=0.25):
    """
    Plot simplex decision regions and observations (K=3 only).
    """

    n_prob = data.filter(like="Prob_").shape[1]
    if n_prob != 3:
        raise NotImplementedError(
            f"plot_simplex only supports K=3 (found {n_prob} Prob_ columns)."
        )

    Prob = data[["Prob_1", "Prob_2", "Prob_3"]].values

    def simplex_to_xy(p):
        x = p[:, 1] + 0.5 * p[:, 2]
        y = (np.sqrt(3) / 2) * p[:, 2]
        return x, y

    # Grid for regions
    grid = []
    for p1 in np.arange(0, 1 + step, step):
        for p2 in np.arange(0, 1 - p1 + step, step):
            p3 = 1 - p1 - p2
            if p3 >= 0:
                grid.append([p1, p2, p3])
    grid = np.array(grid)

    T_grid = decision_rule(grid, tau)
    xg, yg = simplex_to_xy(grid)

    xo, yo = simplex_to_xy(Prob)

    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}

    fig, ax = plt.subplots(figsize=(8, 7))

    # Regions
    for k in [1, 2, 3]:
        ax.scatter(
            xg[T_grid == k],
            yg[T_grid == k],
            s=6,
            color=colors[k],
            alpha=alpha_region,
            edgecolors="none"
        )

    # Points
    for k in [1, 2, 3]:
        ax.scatter(
            xo[data[T] == k],
            yo[data[T] == k],
            s=15,
            color=colors[k],
            label=f"T={k}"
        )

    # Simplex boundary
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [0, 0]
    ])
    ax.plot(triangle[:, 0], triangle[:, 1], color="black")

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.legend()
    plt.tight_layout()
    plt.show()

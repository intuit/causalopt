import pandas as pd

from causalopt.thresh_tradeoff import gains_eval, pred_tradeoff
from causalopt.thresh_tune import get_rd_objects, optim_thresh, predictions, welfare
from causalopt.utils import resolve_binning
from causalopt.multiclass import get_thresholds


def optimum_threshold(
    df: pd.DataFrame,
    outcome_col: str,
    prob_col: str,
    threshold: float,
    covariates: list | None = None,
    out_of_bandwidth: bool = False,
    bin: bool = False,
    binned_data: bool = False,
    bin_spec=None,
    weight_col: str = "n",
) -> dict:
    """
    Run the full single-outcome RD threshold optimization pipeline.

    This is the main entry point for threshold tuning on a single outcome variable.
    It centers the running variable at the current threshold, estimates a sharp RD
    model, builds polynomial predictions with simulation-based confidence bands, and
    identifies three candidate thresholds (optimum, conservative, aggressive) by
    maximising a welfare surface.

    The welfare surface measures the cumulative expected gain from shifting the
    threshold. At each candidate threshold value x*, the gain is the difference
    between the cumulative welfare at x* and the cumulative welfare at the current
    threshold, scaled by total population size.

    Three scenarios are produced:
        - **Optimum**: uses the point estimates of the left/right polynomials.
        - **Conservative**: uses the lower confidence band of the active-side
          polynomial (the side toward which the threshold would move).
        - **Aggressive**: uses the upper confidence band of the active-side
          polynomial.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing both the outcome and the running variable.
    outcome_col : str
        Name of the outcome column in df.
    prob_col : str
        Name of the running variable column in df. Typically a model score or
        probability. The variable is internally centered at threshold so the
        cutoff is at zero.
    threshold : float
        Current operational threshold in the original (uncentered) scale of
        prob_col. This is used both to center the running variable and as the
        reference point for computing welfare gains.

    covariates : list of str or None, default None
        Column names in df to include as covariates in the RD estimation via
        Frisch-Waugh-Lovell partialling-out. Covariates improve efficiency of
        the treatment effect estimate but do not change its interpretation.

    out_of_bandwidth : bool, default False
        If True, report predictions/welfare over the full observed support of
        the running variable rather than only within the estimation bandwidth.
        The fit (bandwidths, coefficients, point estimate) is unchanged; only
        the evaluation grid widens. Beyond the bandwidth this is extrapolation.

    Returns
    -------
    dict with the following keys:

        "data_descriptives" : pd.DataFrame
            Summary statistics (describe()) of the full input DataFrame.

        "rd_results" : dict
            RD estimation output from rd_estimate, including conventional and
            bias-corrected point estimates, standard errors, confidence intervals,
            and selected bandwidths.

        "rdplot" : dict
            Numerical objects for the RD plot from rd_objects, including bin
            summaries and fitted polynomial curves.

        "predictions" : pd.DataFrame
            Bin-level prediction DataFrame with columns:
                x               - bin midpoints (centered at threshold)
                n               - total observations (scalar, same for all rows)
                p               - proportion of observations in each bin
                y_mean          - observed bin mean
                y_ci_l/y_ci_r   - t-based confidence interval for the bin mean
                y_hat_l/y_hat_r - left/right polynomial point predictions
                y_hat_lower_*/y_hat_upper_* - simulation-based CI for each side

        "welfare" : pd.DataFrame
            Welfare surface with columns x, n, p, and cumulative welfare values
            for optimum, conservative, and aggressive scenarios.

        "optimum_thresholds" : dict
            Threshold recommendation with keys:
                "Recommendation" - text recommendation based on sign and
                                   statistical significance of the RD estimate
                "Thresholds"     - dict with Optimum, Conservative, Aggressive
                                   threshold values in the original prob_col scale
                "Additional Gain"- dict with expected welfare gain (in outcome
                                   units × observations) for each scenario

        "current_threshold" : float
            The threshold value passed as input, for reference.

    Notes
    -----
    - The running variable is recentered internally; all x values in predictions
      and welfare are relative to threshold (i.e., threshold maps to x=0).
    - Threshold values in "optimum_thresholds" are converted back to the original
      scale by adding the current threshold.
    - Use rdd_impact and plot_thresh from causalopt.plots to visualize the output.
    """
    if covariates is not None and (bin or binned_data):
        raise ValueError(
            "covariates are not supported with bin/binned_data "
            "(aggregation covers outcomes only)."
        )

    work, w = resolve_binning(
        df, [prob_col], [outcome_col], cutoff=threshold,
        bin=bin, binned_data=binned_data, bin_spec=bin_spec, weight_col=weight_col,
    )

    data_descriptives = pd.DataFrame(work.describe())

    y = work[outcome_col]
    x = work[prob_col] - threshold
    c = 0
    Z = work[covariates].to_numpy() if covariates is not None else None

    rd_obj = get_rd_objects(
        y, x, c, covariates=Z, out_of_bandwidth=out_of_bandwidth, weights=w
    )
    rd_predictions = predictions(rd_obj)
    welfare_results = welfare(rd_predictions, rd_obj)
    optimum_thresholds = optim_thresh(welfare_results, rd_obj, threshold)

    return {
        "data_descriptives": data_descriptives,
        "rd_results": rd_obj["rd_estimates"],
        "rdplot": rd_obj["rd_plot_objects"],
        "predictions": rd_predictions,
        "welfare": welfare_results,
        "optimum_thresholds": optimum_thresholds,
        "current_threshold": threshold,
    }


def tradeoff_threshold(
    df: pd.DataFrame,
    outcomes: list,
    prob_col: str,
    threshold: float,
    out_of_bandwidth: bool = False,
    bin: bool = False,
    binned_data: bool = False,
    bin_spec=None,
    weight_col: str = "n",
) -> pd.DataFrame:
    """
    Compute welfare gains across multiple outcomes on a shared threshold support.

    This function is designed for settings where adjusting the threshold affects
    several outcomes simultaneously and you want to understand the tradeoffs.
    For example, lowering a credit score threshold may increase approvals (positive
    outcome) but also increase defaults (negative outcome).

    The first element of outcomes is treated as the **primary outcome**: a full RD
    model is estimated for it, and its bin support (x grid, n, p) is used as the
    reference. Each subsequent outcome is evaluated on that same support rather
    than its own grid, so all welfare gain columns are directly comparable and
    can be merged without interpolation.

    For each outcome, the welfare gain at threshold x* is:
        gain = n × Σ_{x ≥ x*} [(y_hat_r(x) - y_hat_l(x)) × p(x)] - current_welfare

    where current_welfare is the value at x=0 (the current threshold).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing all outcome and running variable columns.
    outcomes : list of str
        List of outcome column names. The first element is the primary outcome
        used to define the RD model and bin support. Subsequent elements are
        secondary outcomes evaluated on the same support.
    prob_col : str
        Name of the running variable column in df. Centered at threshold
        internally so the cutoff is at zero.
    threshold : float
        Current operational threshold in the original scale of prob_col.
    out_of_bandwidth : bool, default False
        If True, evaluate the tradeoff gains over the full observed support of
        the running variable instead of only within the estimation bandwidth.
        The per-outcome fits are unchanged; only the shared eval grid widens.

    Returns
    -------
    pd.DataFrame
        One row per bin midpoint (candidate threshold), with columns:
            x               - bin midpoint in the centered scale (x=0 is current threshold)
            n               - total population size
            p               - proportion of observations in the bin
            gain_{outcome}  - welfare gain (outcome units × observations) for each
                              outcome if the threshold were set to x

        All gain columns are expressed relative to the welfare at the current
        threshold (x=0), so positive values indicate improvement over status quo.
        The DataFrame is sorted by x (ascending) and can be plotted directly to
        visualize the tradeoff curve.

    Notes
    -----
    - A full RD estimation (bandwidth selection + local polynomial) is run for
      each outcome, but only the primary outcome's bin grid is used as the x
      support. This ensures the gain columns are aligned and mergeable.
    - If outcomes contains only one element, the result is equivalent to the
      welfare gain column from optimum_threshold but without the full pipeline
      output.
    """
    work, w = resolve_binning(
        df, [prob_col], list(outcomes), cutoff=threshold,
        bin=bin, binned_data=binned_data, bin_spec=bin_spec, weight_col=weight_col,
    )

    y = work[outcomes[0]]
    x = work[prob_col] - threshold
    c = 0

    rdobj_main = get_rd_objects(y, x, c, out_of_bandwidth=out_of_bandwidth, weights=w)
    pred_main = pred_tradeoff(rdobj_main, inputs=None)
    w_main = gains_eval(pred_main, outcomes[0])

    final_result = w_main

    for out in outcomes[1:]:
        y = work[out]
        rdobj_alt = get_rd_objects(y, x, c, out_of_bandwidth=out_of_bandwidth, weights=w)
        pred_alt = pred_tradeoff(rdobj_alt, pred_main[["x", "n", "p"]])
        w_alt = gains_eval(pred_alt, out)
        final_result = pd.merge(final_result, w_alt, on=["x", "n", "p"], how="inner")

    return final_result


def causalopt(
    df: pd.DataFrame,
    outcomes,
    score_cols,
    mode: str,
    threshold: float | None = None,
    tau=None,
    covariates: list | None = None,
    B: int = 10_000,
    kernel: str = "sum",
    probabilities: bool | None = None,
    out_of_bandwidth: bool = False,
    bin: bool = False,
    binned_data: bool = False,
    bin_spec=None,
    weight_col: str = "n",
) -> dict:
    """
    Unified entry point that dispatches to the binary or multiclass tuner.

    Dispatch is explicit via ``mode`` (no fragile inference), with fail-fast
    validation of the mode-specific arguments, and a standardized return shape
    across both modes.

    mode = "binary"
        The classical single-threshold case: a single running variable
        (``score_cols`` is one column) crossing a
        scalar ``threshold``. Both analyses run on ONE shared bin grid derived
        from the primary outcome (``outcomes[0]``): the optimum (via
        ``optim_thresh`` on the primary) and the tradeoff frontier (gains for
        every outcome, evaluated on the primary's grid). The primary RD objects
        are built once and reused, so the optimum recommendation and the
        tradeoff curve live on identical bins. The tradeoff frontier is always
        returned (a single-outcome frontier is just one gain column).

    mode = "multiclass"
        ``score_cols`` is the list of K>=2 probability/score columns. This is a
        thin pass-through of :func:`~causalopt.multiclass.get_thresholds`: its
        ``{frontier, optimum, current}`` dict is returned verbatim, wrapped with
        ``mode`` and ``details``. The optimum is the per-outcome argmax already
        computed by ``select_optimum`` inside ``get_thresholds`` (no second
        call).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    outcomes : str or list of str
        Outcome column(s). In binary mode, ``outcomes[0]`` is the primary outcome.
    score_cols : str or list of str
        The running variable (binary mode, one column) or probability/score
        columns (multiclass, list of >=2).
    mode : {"binary", "multiclass"}
        Which tuner to run.
    threshold : float or None
        Current cutoff (required for binary; must be None for multiclass).
    tau : array-like or None
        Current thresholds (multiclass-only; must be None for binary).
    covariates : list of str or None
        Covariates for FWL adjustment (binary-only; not supported with binning).
    B, kernel, probabilities
        Multiclass-only knobs (search budget, kernel combine rule, input regime).
        For binary mode they keep harmless defaults and are unused;
        ``probabilities`` resolves None -> True for the multiclass call.
    out_of_bandwidth, bin, binned_data, bin_spec, weight_col
        Shared flags forwarded to both paths (see the sub-functions).

    Returns
    -------
    dict
        ``{"mode", "current", "frontier", "optimum", "details"}``. In binary
        mode, ``details`` carries data_descriptives / rdplot / predictions / welfare
        (and rd_results / estimates, which are None in binned mode); in
        multiclass, ``details`` carries the call params (kernel, probabilities,
        B, tau) plus the intermediates from ``get_thresholds``
        (data_descriptives, prepared, bandwidths, estimates, predictions, and the
        grid-evaluated fitted curves).
    """
    if mode not in ("binary", "multiclass"):
        raise ValueError("mode must be 'binary' or 'multiclass'.")

    if isinstance(outcomes, str):
        outcomes = [outcomes]
    outcomes = list(outcomes)
    if len(outcomes) < 1:
        raise ValueError("outcomes must have at least one column.")

    if mode == "binary":
        if isinstance(score_cols, (list, tuple)):
            if len(score_cols) != 1:
                raise ValueError(
                    f"mode='binary' expects a single score column; got a list of "
                    f"length {len(score_cols)}."
                )
            prob_col = score_cols[0]
        else:
            prob_col = score_cols

        if threshold is None:
            raise ValueError("mode='binary' requires threshold (the current cutoff).")
        if tau is not None or probabilities is not None:
            raise ValueError(
                "tau/probabilities are multiclass-only; leave them unset for "
                "mode='binary'."
            )
        if covariates is not None and (bin or binned_data):
            raise ValueError(
                "covariates are not supported with bin/binned_data "
                "(aggregation covers outcomes only)."
            )

        work, w = resolve_binning(
            df, [prob_col], outcomes, cutoff=threshold,
            bin=bin, binned_data=binned_data, bin_spec=bin_spec,
            weight_col=weight_col,
        )

        x = (work[prob_col] - threshold).to_numpy()
        Z = work[covariates].to_numpy() if covariates is not None else None

        # Primary outcome: built ONCE, reused for the optimum and the shared grid.
        rdobj = get_rd_objects(
            work[outcomes[0]].to_numpy(), x, 0,
            covariates=Z, out_of_bandwidth=out_of_bandwidth, weights=w,
        )

        preds = predictions(rdobj)
        wf = welfare(preds, rdobj)
        opt = optim_thresh(wf, rdobj, threshold)

        pred0 = pred_tradeoff(rdobj, inputs=None)
        front = gains_eval(pred0, outcomes[0])
        for o in outcomes[1:]:
            ro = get_rd_objects(
                work[o].to_numpy(), x, 0,
                out_of_bandwidth=out_of_bandwidth, weights=w,
            )
            front = front.merge(
                gains_eval(pred_tradeoff(ro, pred0[["x", "n", "p"]]), o),
                on=["x", "n", "p"],
            )

        details = {
            "data_descriptives": pd.DataFrame(work.describe()),
            "rdplot": rdobj["rd_plot_objects"],
            "predictions": preds,
            "welfare": wf,
            "rd_results": rdobj["rd_estimates"],
            "estimates": rdobj["estimates"],
        }

        return {
            "mode": "binary",
            "current": {"threshold": threshold},
            "frontier": front,
            "optimum": opt,
            "details": details,
        }

    # mode == "multiclass"
    if not isinstance(score_cols, (list, tuple)) or len(score_cols) < 2:
        raise ValueError(
            "mode='multiclass' expects score_cols to be a list of >=2 columns."
        )
    if threshold is not None or covariates is not None:
        raise ValueError(
            "threshold/covariates are binary-only; leave them unset for "
            "mode='multiclass'."
        )

    prob = True if probabilities is None else probabilities
    res = get_thresholds(
        df, outcomes, list(score_cols), tau=tau, B=B, kernel=kernel,
        probabilities=prob, out_of_bandwidth=out_of_bandwidth,
        bin=bin, binned_data=binned_data, bin_spec=bin_spec, weight_col=weight_col,
    )

    details = res.pop("details", {})
    details.update({"kernel": kernel, "probabilities": prob, "B": B, "tau": tau})
    return {"mode": "multiclass", **res, "details": details}

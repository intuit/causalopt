import pandas as pd

from causalopt.thresh_tradeoff import gains_eval, pred_tradeoff
from causalopt.thresh_tune import get_rd_objects, optim_thresh, predictions, welfare


def optimum_threshold(
    df: pd.DataFrame,
    outcome_col: str,
    prob_col: str,
    threshold: float,
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
    data_descriptives = pd.DataFrame(df.describe())

    y = df[outcome_col]
    x = df[prob_col] - threshold
    c = 0

    rd_obj = get_rd_objects(y, x, c)
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
    y = df[outcomes[0]]
    x = df[prob_col] - threshold
    c = 0

    rdobj_main = get_rd_objects(y, x, c)
    pred_main = pred_tradeoff(rdobj_main, inputs=None)
    w_main = gains_eval(pred_main, outcomes[0])

    final_result = w_main

    for out in outcomes[1:]:
        y = df[out]
        rdobj_alt = get_rd_objects(y, x, c)
        pred_alt = pred_tradeoff(rdobj_alt, pred_main[["x", "n", "p"]])
        w_alt = gains_eval(pred_alt, out)
        final_result = pd.merge(final_result, w_alt, on=["x", "n", "p"], how="inner")

    return final_result
